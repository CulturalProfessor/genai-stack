import json
import logging
import os
import time
from collections import defaultdict, deque
from collections.abc import Generator
from queue import Empty, Queue
from threading import Lock, Thread

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

from chains import (
    configure_llm_only_chain,
    configure_qa_rag_chain,
    generate_ticket,
    load_embedding_model,
    load_llm,
)
from utils import BaseLogger, create_fulltext_index, create_vector_index

load_dotenv(".env")

logger = logging.getLogger(__name__)

url = os.getenv("NEO4J_URI")
username = os.getenv("NEO4J_USERNAME")
password = os.getenv("NEO4J_PASSWORD")
ollama_base_url = os.getenv("OLLAMA_BASE_URL")
embedding_model_name = os.getenv("EMBEDDING_MODEL")
llm_name = os.getenv("LLM")

if not url:
    raise RuntimeError("NEO4J_URI is not set - copy env.example to .env first")
# Remapping for Langchain Neo4j integration
os.environ["NEO4J_URL"] = url

# Empty means "no auth", which is the local-development default.
API_KEY = os.getenv("API_KEY", "")
ALLOWED_ORIGINS = [
    origin.strip()
    for origin in os.getenv(
        "ALLOWED_ORIGINS", "http://localhost:8505,http://localhost:5173"
    ).split(",")
    if origin.strip()
]
RATE_LIMIT_REQUESTS = int(os.getenv("RATE_LIMIT_REQUESTS", "30"))
RATE_LIMIT_WINDOW_SECONDS = int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60"))

embeddings, dimension = load_embedding_model(
    embedding_model_name,
    config={"ollama_base_url": ollama_base_url},
    logger=BaseLogger(),
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(url=url, username=username, password=password)
create_vector_index(neo4j_graph, dimension)
create_fulltext_index(neo4j_graph)

llm = load_llm(
    llm_name, logger=BaseLogger(), config={"ollama_base_url": ollama_base_url}
)

llm_chain = configure_llm_only_chain(llm)
rag_chain = configure_qa_rag_chain(
    llm, embeddings, embeddings_store_url=url, username=username, password=password
)


class QueueCallback(BaseCallbackHandler):
    """Callback handler for streaming LLM responses to a queue."""

    def __init__(self, q):
        self.q = q

    def on_llm_new_token(self, token: str, **kwargs) -> None:
        self.q.put(token)

    def on_llm_end(self, *args, **kwargs) -> None:
        return self.q.empty()


def stream(cb, q) -> Generator:
    job_done = object()
    error: list[BaseException] = []

    def task():
        try:
            cb()
        except BaseException as exc:  # noqa: BLE001 - re-raised on the main thread
            # Without this the sentinel is never queued and the generator below
            # blocks forever on a request that already failed.
            error.append(exc)
        finally:
            q.put(job_done)

    t = Thread(target=task)
    t.start()

    content = ""

    # Get each new token from the queue and yield for our generator
    while True:
        try:
            next_token = q.get(True, timeout=1)
            if next_token is job_done:
                break
            content += next_token
            yield next_token, content
        except Empty:
            continue

    if error:
        raise error[0]


app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=ALLOWED_ORIGINS,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()


def rate_limit(request: Request) -> None:
    """Cap requests per client IP so an exposed instance can't be drained."""
    if RATE_LIMIT_REQUESTS <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_limit_lock:
        hits = _rate_limit_hits[client]
        while hits and now - hits[0] > RATE_LIMIT_WINDOW_SECONDS:
            hits.popleft()
        if len(hits) >= RATE_LIMIT_REQUESTS:
            raise HTTPException(status_code=429, detail="Too many requests")
        hits.append(now)


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)):
    """Enforce API_KEY when one is configured; no-op otherwise.

    EventSource cannot set request headers, so the streaming endpoint also
    accepts the key as a query parameter.
    """
    if not API_KEY:
        return
    provided = x_api_key or request.query_params.get("api_key")
    if provided != API_KEY:
        raise HTTPException(status_code=401, detail="Invalid or missing API key")


guarded = [Depends(require_api_key), Depends(rate_limit)]


@app.get("/")
async def root():
    return {"message": "Hello World"}


class Question(BaseModel):
    text: str
    rag: bool = False


class BaseTicket(BaseModel):
    text: str


def _chain_for(rag: bool):
    return rag_chain if rag else llm_chain


# EventSource only issues GET requests, so this endpoint takes query parameters.
@app.get("/query-stream", dependencies=guarded)
def qstream(question: Question = Depends()):
    output_function = _chain_for(question.rag)

    q = Queue()

    def cb():
        output_function(
            {"question": question.text, "chat_history": []},
            callbacks=[QueueCallback(q)],
        )

    def generate():
        yield json.dumps({"init": True, "model": llm_name})
        try:
            for token, _ in stream(cb, q):
                yield json.dumps({"token": token})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Streaming query failed")
            yield json.dumps({"error": str(exc)})

    return EventSourceResponse(generate(), media_type="text/event-stream")


@app.post("/query", dependencies=guarded)
async def ask(question: Question):
    output_function = _chain_for(question.rag)
    result = output_function(
        {"question": question.text, "chat_history": []}, callbacks=[]
    )

    return {"result": result["answer"], "model": llm_name}


@app.post("/generate-ticket", dependencies=guarded)
async def generate_ticket_api(question: BaseTicket):
    new_title, new_question = generate_ticket(
        neo4j_graph=neo4j_graph,
        llm_chain=llm_chain,
        input_question=question.text,
    )
    return {"result": {"title": new_title, "text": new_question}, "model": llm_name}

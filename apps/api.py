import json
import logging
import sys
import time
from collections import defaultdict, deque
from collections.abc import Generator
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread

from fastapi import Depends, FastAPI, Header, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_neo4j import Neo4jGraph
from pydantic import BaseModel
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.chains import (
    answer_question,
    configure_llm_only_chain,
    configure_qa_rag_chain,
    generate_ticket,
)
from genai_stack.llm import load_embedding_model, load_llm
from genai_stack.schema import BaseLogger, apply_schema
from genai_stack.settings import load_settings

logger = logging.getLogger(__name__)
settings = load_settings()
llm_name = settings.llm

embeddings, dimension = load_embedding_model(
    settings.embedding_model, config=settings.embedding_config, logger=BaseLogger()
)

# if Neo4j is local, you can go to http://localhost:7474/ to browse the database
neo4j_graph = Neo4jGraph(
    url=settings.neo4j_uri,
    username=settings.neo4j_username,
    password=settings.neo4j_password,
)
apply_schema(neo4j_graph, dimension)

llm = load_llm(llm_name, logger=BaseLogger(), config=settings.embedding_config)

llm_chain = configure_llm_only_chain(llm)
rag_chain = configure_qa_rag_chain(
    llm,
    embeddings,
    embeddings_store_url=settings.neo4j_uri,
    username=settings.neo4j_username,
    password=settings.neo4j_password,
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
    allow_origins=settings.allowed_origins,
    allow_credentials=True,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

_rate_limit_hits: dict[str, deque[float]] = defaultdict(deque)
_rate_limit_lock = Lock()


def rate_limit(request: Request) -> None:
    """Cap requests per client IP so an exposed instance can't be drained."""
    if settings.rate_limit_requests <= 0:
        return
    client = request.client.host if request.client else "unknown"
    now = time.monotonic()
    with _rate_limit_lock:
        hits = _rate_limit_hits[client]
        while hits and now - hits[0] > settings.rate_limit_window_seconds:
            hits.popleft()
        if len(hits) >= settings.rate_limit_requests:
            raise HTTPException(status_code=429, detail="Too many requests")
        hits.append(now)


def require_api_key(request: Request, x_api_key: str | None = Header(default=None)):
    """Enforce API_KEY when one is configured; no-op otherwise.

    EventSource cannot set request headers, so the streaming endpoint also
    accepts the key as a query parameter.
    """
    if not settings.api_key:
        return
    provided = x_api_key or request.query_params.get("api_key")
    if provided != settings.api_key:
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
        answer_question(
            output_function, question.text, callbacks=[QueueCallback(q)]
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
    answer = answer_question(_chain_for(question.rag), question.text)

    return {"result": answer, "model": llm_name}


@app.post("/generate-ticket", dependencies=guarded)
async def generate_ticket_api(question: BaseTicket):
    new_title, new_question = generate_ticket(
        neo4j_graph=neo4j_graph,
        llm_chain=llm_chain,
        input_question=question.text,
    )
    return {"result": {"title": new_title, "text": new_question}, "model": llm_name}

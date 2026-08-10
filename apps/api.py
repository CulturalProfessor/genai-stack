import json
import logging
import sys
import time
from collections import defaultdict, deque
from collections.abc import Generator
from dataclasses import asdict
from io import BytesIO
from pathlib import Path
from queue import Empty, Queue
from threading import Lock, Thread

from fastapi import Depends, FastAPI, File, Header, HTTPException, Request, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from langchain.chains import RetrievalQA
from langchain_core.callbacks.base import BaseCallbackHandler
from langchain_neo4j import Neo4jGraph, Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pydantic import BaseModel
from pypdf import PdfReader
from sse_starlette.sse import EventSourceResponse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.chains import (
    answer_question,
    configure_llm_only_chain,
    configure_qa_rag_chain,
    generate_ticket,
)
from genai_stack.code_index import count_symbols
from genai_stack.conversations import (
    KIND_CODE,
    KIND_PDF,
    append_message,
    chat_history,
    create_conversation,
    delete_conversation,
    delete_conversations_for,
    get_conversation,
    list_conversations,
    load_messages,
    rename_conversation,
    set_title_if_untitled,
)
from genai_stack.extract import build_payload
from genai_stack.ingest import (
    import_index,
    list_repositories,
    record_import,
    reset_graph,
)
from genai_stack.llm import load_embedding_model, load_llm
from genai_stack.pdf_store import (
    DOCUMENT_ID_PROPERTY,
    PDF_CHUNK_LABEL,
    PDF_INDEX_NAME,
    chunk_metadata,
    delete_document,
    get_document,
    list_documents,
    register_document,
)
from genai_stack.pdf_store import document_id as pdf_document_id
from genai_stack.schema import BaseLogger, apply_schema
from genai_stack.settings import load_settings
from genai_stack.sources import allowed_hosts, cloned_repository, parse_repo_url

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
    # Every method the API serves. Renaming a conversation is a PATCH and
    # deleting one is a DELETE; leaving those out makes the browser refuse
    # them at the preflight, while curl - which does not preflight - works.
    allow_methods=["GET", "POST", "PATCH", "DELETE", "OPTIONS"],
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


# --------------------------------------------------------------------------
# Conversations
#
# Everything here is a thin wrapper over genai_stack.conversations, which owns
# the Cypher; this layer only turns it into JSON.
# --------------------------------------------------------------------------


class NewConversation(BaseModel):
    kind: str = KIND_CODE
    subject: str | None = None


class RenameConversation(BaseModel):
    title: str


class Ask(BaseModel):
    text: str
    rag: bool = True


def _conversation_or_404(conversation_id: str):
    conversation = get_conversation(neo4j_graph, conversation_id)
    if conversation is None:
        raise HTTPException(status_code=404, detail="No such conversation")
    return conversation


@app.get("/conversations", dependencies=guarded)
async def conversations(kind: str = KIND_CODE, subject: str | None = None):
    return {
        "conversations": [
            asdict(c) for c in list_conversations(neo4j_graph, kind=kind, subject=subject)
        ]
    }


@app.post("/conversations", dependencies=guarded)
async def new_conversation(body: NewConversation):
    conversation_id = create_conversation(
        neo4j_graph, kind=body.kind, subject=body.subject
    )
    return asdict(_conversation_or_404(conversation_id))


@app.get("/conversations/{conversation_id}", dependencies=guarded)
async def one_conversation(conversation_id: str):
    conversation = _conversation_or_404(conversation_id)
    return {
        "conversation": asdict(conversation),
        "messages": [asdict(m) for m in load_messages(neo4j_graph, conversation_id)],
    }


@app.patch("/conversations/{conversation_id}", dependencies=guarded)
async def rename(conversation_id: str, body: RenameConversation):
    _conversation_or_404(conversation_id)
    rename_conversation(neo4j_graph, conversation_id, body.title)
    return asdict(_conversation_or_404(conversation_id))


@app.delete("/conversations/{conversation_id}", dependencies=guarded)
async def remove_conversation(conversation_id: str):
    _conversation_or_404(conversation_id)
    delete_conversation(neo4j_graph, conversation_id)
    return {"deleted": conversation_id}


def _pdf_answer(document_id: str, text: str, callbacks):
    """Retrieval limited to one uploaded PDF."""
    store = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        index_name=PDF_INDEX_NAME,
        node_label=PDF_CHUNK_LABEL,
    )
    chain = RetrievalQA.from_chain_type(
        llm=llm,
        chain_type="stuff",
        retriever=store.as_retriever(
            search_kwargs={"filter": {DOCUMENT_ID_PROPERTY: document_id}}
        ),
    )
    return chain.invoke({"query": text}, config={"callbacks": callbacks})["result"]


# EventSource only issues GET requests, so the question travels as a query
# parameter here just as it does on /query-stream.
@app.get("/conversations/{conversation_id}/ask", dependencies=guarded)
def ask_in_conversation(conversation_id: str, question: Ask = Depends()):
    conversation = _conversation_or_404(conversation_id)
    history = chat_history(load_messages(neo4j_graph, conversation_id))
    # A PDF conversation is answered from its document, not the code graph.
    # The conversation itself says which, so the client does not have to.
    pdf = conversation.kind == KIND_PDF
    mode = "PDF" if pdf else ("Graph RAG" if question.rag else "LLM only")

    # Saved before the answer, so a question that fails is not lost.
    append_message(neo4j_graph, conversation_id, "user", question.text)
    set_title_if_untitled(neo4j_graph, conversation_id, question.text)

    q = Queue()

    def cb():
        if pdf:
            return _pdf_answer(conversation.subject, question.text, [QueueCallback(q)])
        return answer_question(
            _chain_for(question.rag),
            question.text,
            history=history,
            callbacks=[QueueCallback(q)],
        )

    def generate():
        yield json.dumps({"init": True, "model": llm_name, "mode": mode})
        answer = ""
        try:
            for token, content in stream(cb, q):
                answer = content
                yield json.dumps({"token": token})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Streaming query failed")
            yield json.dumps({"error": str(exc)})
            return
        append_message(neo4j_graph, conversation_id, "assistant", answer, mode=mode)
        conversation = get_conversation(neo4j_graph, conversation_id)
        yield json.dumps({"done": True, "title": conversation.title if conversation else ""})

    return EventSourceResponse(generate(), media_type="text/event-stream")


# --------------------------------------------------------------------------
# The code graph: what is in it, and putting a repository into it
# --------------------------------------------------------------------------

GRAPH_COUNTS = """
RETURN COUNT { MATCH (f:File) } AS files,
       COUNT { MATCH (c:Code) } AS symbols,
       COUNT { MATCH (:Code)-[:CALLS]->(:Code) } AS calls,
       COUNT {
           MATCH (a:Code)-[:CALLS]->(b:Code)
           WHERE a.file_path <> b.file_path
       } AS cross_file,
       COUNT { MATCH (:File)-[:IMPORTS]->(:File) } AS imports
"""


@app.get("/graph", dependencies=guarded)
async def graph_summary():
    return {
        "counts": neo4j_graph.query(GRAPH_COUNTS)[0],
        "repositories": list_repositories(neo4j_graph),
        "allowed_hosts": allowed_hosts(),
        # Here rather than in its own endpoint: this is the one call the client
        # makes to find out whether the stack is answering, and what with.
        "model": llm_name,
    }


@app.delete("/graph", dependencies=guarded)
async def clear_graph():
    """Drops the code graph. Conversations and PDFs are left alone."""
    reset_graph(neo4j_graph)
    return {"counts": neo4j_graph.query(GRAPH_COUNTS)[0]}


class IndexRequest(BaseModel):
    url: str
    branch: str | None = None


@app.get("/graph/index", dependencies=guarded)
def index_repository(request: IndexRequest = Depends()):
    """Clone, parse and import a repository, reporting progress as it goes."""

    def generate():
        try:
            ref = parse_repo_url(request.url, request.branch or None)
        except ValueError as exc:
            yield json.dumps({"error": str(exc)})
            return

        try:
            yield json.dumps({"stage": "cloning", "repo": ref.slug})
            with cloned_repository(ref) as checkout:
                yield json.dumps({"stage": "parsing", "repo": ref.slug})
                index = build_payload(
                    checkout, repo_name=ref.name, source_url=ref.clone_url
                )

            total = count_symbols(index)
            if not total:
                yield json.dumps(
                    {"error": f"No TypeScript or JavaScript symbols found in {ref.slug}"}
                )
                return
            yield json.dumps(
                {"stage": "parsed", "symbols": total, "files": len(index["files"])}
            )

            # import_index reports progress through a callback, but this is a
            # generator - so collect into a queue and drain it as it fills.
            progress: Queue = Queue()

            def on_progress(done: int, of: int) -> None:
                progress.put((done, of))

            done_marker = object()
            result: dict = {}

            def work():
                try:
                    result["imported"] = import_index(
                        index,
                        graph=neo4j_graph,
                        embeddings=embeddings,
                        batch_size=settings.import_batch_size,
                        on_progress=on_progress,
                    )
                    record_import(neo4j_graph, index, result["imported"])
                except BaseException as exc:  # noqa: BLE001 - reported below
                    result["error"] = str(exc)
                finally:
                    progress.put(done_marker)

            worker = Thread(target=work)
            worker.start()
            while True:
                try:
                    item = progress.get(True, timeout=1)
                except Empty:
                    continue
                if item is done_marker:
                    break
                imported, of = item
                yield json.dumps(
                    {"stage": "importing", "imported": imported, "total": of}
                )
            worker.join()

            if "error" in result:
                yield json.dumps({"error": result["error"]})
                return
            yield json.dumps(
                {
                    "done": True,
                    "imported": result.get("imported", 0),
                    "repo": ref.slug,
                    "counts": neo4j_graph.query(GRAPH_COUNTS)[0],
                }
            )
        except RuntimeError as exc:  # clone failures carry a readable message
            yield json.dumps({"error": str(exc)})
        except Exception as exc:  # noqa: BLE001 - surfaced to the client
            logger.exception("Indexing failed")
            yield json.dumps({"error": str(exc)})

    return EventSourceResponse(generate(), media_type="text/event-stream")


@app.post("/graph/import", dependencies=guarded)
async def import_index_file(file: UploadFile = File(...)):
    """Import an index built by `python3 -m genai_stack.extract`.

    The CLI only writes JSON, and /graph/index only clones from an allowed
    host - so this is what puts a local checkout into the graph.
    """
    try:
        index = json.loads(await file.read())
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"Not valid JSON: {exc}")

    if not isinstance(index, dict) or "files" not in index:
        raise HTTPException(
            status_code=422,
            detail="Not a code index - expected an object with a 'files' key",
        )
    if not count_symbols(index):
        raise HTTPException(status_code=422, detail="The index contains no symbols")

    imported = import_index(
        index,
        graph=neo4j_graph,
        embeddings=embeddings,
        batch_size=settings.import_batch_size,
    )
    record_import(neo4j_graph, index, imported)
    return {
        "imported": imported,
        "repo": index.get("repo") or "unknown",
        "counts": neo4j_graph.query(GRAPH_COUNTS)[0],
    }


# --------------------------------------------------------------------------
# Uploaded PDFs
# --------------------------------------------------------------------------


@app.get("/pdf/documents", dependencies=guarded)
async def pdf_documents():
    return {"documents": [asdict(d) for d in list_documents(neo4j_graph)]}


@app.post("/pdf/documents", dependencies=guarded)
async def upload_pdf(file: UploadFile = File(...)):
    content = await file.read()
    name = file.filename or "document.pdf"
    doc_id = pdf_document_id(name, content)

    existing = get_document(neo4j_graph, doc_id)
    if existing:
        return {"document": asdict(existing), "reused": True}

    reader = PdfReader(BytesIO(content))
    text = "".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        raise HTTPException(
            status_code=422,
            detail="No text could be extracted - is this a scanned PDF?",
        )

    chunks = RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_text(text)
    Neo4jVector.from_texts(
        chunks,
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        embedding=embeddings,
        metadatas=chunk_metadata(doc_id, name, len(chunks)),
        index_name=PDF_INDEX_NAME,
        node_label=PDF_CHUNK_LABEL,
    )
    register_document(
        neo4j_graph, doc_id, name, chunks=len(chunks), pages=len(reader.pages)
    )
    return {"document": asdict(get_document(neo4j_graph, doc_id)), "reused": False}


@app.delete("/pdf/documents/{document_id}", dependencies=guarded)
async def remove_pdf(document_id: str):
    if get_document(neo4j_graph, document_id) is None:
        raise HTTPException(status_code=404, detail="No such document")
    delete_document(neo4j_graph, document_id)
    # Its chats are about a document that no longer exists.
    delete_conversations_for(neo4j_graph, document_id)
    return {"deleted": document_id}


@app.post("/generate-ticket", dependencies=guarded)
async def generate_ticket_api(question: BaseTicket):
    new_title, new_question = generate_ticket(
        neo4j_graph=neo4j_graph,
        llm_chain=llm_chain,
        input_question=question.text,
    )
    return {"result": {"title": new_title, "text": new_question}, "model": llm_name}

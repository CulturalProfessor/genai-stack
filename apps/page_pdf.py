"""Ask questions about uploaded PDFs, one document at a time."""
from __future__ import annotations

import streamlit as st
from langchain.chains import RetrievalQA
from langchain_neo4j import Neo4jVector
from langchain_text_splitters import RecursiveCharacterTextSplitter
from pypdf import PdfReader
from streamlit.logger import get_logger

from genai_stack.conversations import (
    KIND_PDF,
    append_message,
    delete_conversations_for,
    load_messages,
    set_title_if_untitled,
)
from genai_stack.pdf_store import (
    PDF_CHUNK_LABEL,
    PDF_INDEX_NAME,
    DOCUMENT_ID_PROPERTY,
    chunk_metadata,
    delete_document,
    document_id,
    get_document,
    list_documents,
    register_document,
)
from conversation_ui import conversation_sidebar, current_conversation_id
from page_chat import StreamHandler
from resources import (
    get_embeddings,
    get_graph,
    get_llm,
    get_settings,
    llm_unavailable,
)

logger = get_logger(__name__)

SELECTED_DOCUMENT = "pdf:selected"


def split(text: str) -> list[str]:
    return RecursiveCharacterTextSplitter(
        chunk_size=1000, chunk_overlap=200, length_function=len
    ).split_text(text)


def vector_store():
    """The shared pdf_bot index. Never pre-deleted - documents coexist now."""
    settings = get_settings()
    return Neo4jVector.from_existing_index(
        embedding=get_embeddings()[0],
        url=settings.neo4j_uri,
        username=settings.neo4j_username,
        password=settings.neo4j_password,
        index_name=PDF_INDEX_NAME,
        node_label=PDF_CHUNK_LABEL,
    )


def ingest(graph, uploaded) -> str | None:
    """Embed a newly uploaded PDF, or recognise one already stored."""
    content = uploaded.getvalue()
    doc_id = document_id(uploaded.name, content)

    existing = get_document(graph, doc_id)
    if existing:
        st.info(f"{uploaded.name} is already indexed.", icon="📄")
        return doc_id

    reader = PdfReader(uploaded)
    text = "".join(page.extract_text() or "" for page in reader.pages)
    if not text.strip():
        st.warning("No text could be extracted - is this a scanned PDF?", icon="⚠️")
        return None

    chunks = split(text)
    settings = get_settings()
    with st.status(f"Indexing {uploaded.name}", expanded=True) as status:
        st.write(f"{len(reader.pages)} pages, {len(chunks)} chunks")
        st.write("Embedding...")
        Neo4jVector.from_texts(
            chunks,
            url=settings.neo4j_uri,
            username=settings.neo4j_username,
            password=settings.neo4j_password,
            embedding=get_embeddings()[0],
            metadatas=chunk_metadata(doc_id, uploaded.name, len(chunks)),
            index_name=PDF_INDEX_NAME,
            node_label=PDF_CHUNK_LABEL,
        )
        register_document(
            graph, doc_id, uploaded.name, chunks=len(chunks), pages=len(reader.pages)
        )
        status.update(label=f"Indexed {uploaded.name}", state="complete")
    return doc_id


def document_sidebar(graph, documents, selected: str | None) -> None:
    with st.sidebar:
        st.markdown("#### PDFs")
        if not documents:
            st.caption("Upload one to get started.")
            return

        for document in documents:
            open_col, menu_col = st.columns([5, 1])
            open_col.button(
                document.name,
                key=f"pdf-{document.id}",
                use_container_width=True,
                type="primary" if document.id == selected else "secondary",
                on_click=lambda did=document.id: st.session_state.update(
                    {SELECTED_DOCUMENT: did}
                ),
            )
            with menu_col.popover("⋯", use_container_width=True):
                st.caption(f"{document.pages} pages, {document.chunks} chunks")
                if st.button(
                    "Delete", key=f"pdf-delete-{document.id}", use_container_width=True
                ):
                    delete_document(graph, document.id)
                    # Its chats are about a document that no longer exists.
                    delete_conversations_for(graph, document.id)
                    if st.session_state.get(SELECTED_DOCUMENT) == document.id:
                        st.session_state[SELECTED_DOCUMENT] = None
                    st.rerun()


def selected_document(documents):
    """The open PDF, defaulting to the most recent one."""
    chosen = st.session_state.get(SELECTED_DOCUMENT)
    if chosen and any(document.id == chosen for document in documents):
        return next(document for document in documents if document.id == chosen)
    if not documents:
        return None
    st.session_state[SELECTED_DOCUMENT] = documents[0].id
    return documents[0]


def answer(document, question: str, callbacks) -> str:
    """Retrieval limited to one document, so other PDFs cannot bleed in."""
    retriever = vector_store().as_retriever(
        search_kwargs={"filter": {DOCUMENT_ID_PROPERTY: document.id}}
    )
    chain = RetrievalQA.from_chain_type(
        llm=get_llm(), chain_type="stuff", retriever=retriever
    )
    return chain.invoke({"query": question}, config={"callbacks": callbacks})["result"]


def render() -> None:
    st.header("Chat with a PDF")

    try:
        graph = get_graph()
    except Exception as exc:  # noqa: BLE001 - the page explains and stays up
        logger.exception("Could not reach Neo4j")
        st.error(f"Neo4j is not reachable: {exc}", icon="🚨")
        return

    uploaded = st.file_uploader("Upload a PDF", type="pdf")
    if uploaded is not None:
        try:
            doc_id = ingest(graph, uploaded)
        except Exception as exc:  # noqa: BLE001 - surfaced to the user
            logger.exception("Could not embed the PDF")
            st.error(f"Error: {exc}", icon="🚨")
            doc_id = None
        if doc_id:
            st.session_state[SELECTED_DOCUMENT] = doc_id

    documents = list_documents(graph)
    document = selected_document(documents)
    document_sidebar(graph, documents, document.id if document else None)

    if document is None:
        st.caption(
            "Stored separately from the code graph, under its own `PdfBotChunk` "
            "label. Uploaded PDFs stay until you delete them."
        )
        return

    st.caption(
        f"Asking **{document.name}** - {document.pages} pages, "
        f"{document.chunks} chunks. Answers come from this document only."
    )

    conversation_id = current_conversation_id(graph, kind=KIND_PDF, subject=document.id)
    conversation_sidebar(graph, kind=KIND_PDF, subject=document.id)
    messages = load_messages(graph, conversation_id)

    for message in messages:
        with st.chat_message(message.role):
            st.write(message.content)

    question = st.chat_input(f"Ask a question about {document.name}")
    if not question:
        return

    append_message(graph, conversation_id, "user", question)
    set_title_if_untitled(graph, conversation_id, question)

    with st.chat_message("user"):
        st.write(question)
    with st.chat_message("assistant"):
        try:
            reply = answer(document, question, [StreamHandler(st.empty())])
        except Exception as exc:  # noqa: BLE001 - the page explains and stays up
            logger.exception("PDF query failed")
            llm_unavailable(exc)
            return

    append_message(graph, conversation_id, "assistant", reply)
    st.rerun()

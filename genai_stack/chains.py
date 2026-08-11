"""The LLM-only and GraphRAG chains, and the ticket-drafting helper."""
from __future__ import annotations

from typing import Any, List

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.base import Chain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.documents import Document
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    MessagesPlaceholder,
    SystemMessagePromptTemplate,
)
from langchain_core.retrievers import BaseRetriever

from genai_stack.pdf_store import DOCUMENT_ID_PROPERTY
from genai_stack.retrieval import build_document_store, build_vector_store

# How many symbols to retrieve. Context budget: extract.py truncates a symbol at
# 4000 characters, so k=4 is at most ~16k characters (~4k tokens) and fits the
# 10k token window set in load_llm alongside the prompt and the answer. Counting
# document tokens instead (reduce_k_below_max_tokens) would pull in a GPT-2
# tokenizer for the Ollama models, so the limit is enforced by chunk size and k.
RETRIEVER_K = 4

# When PDFs are attached they share that budget rather than adding to it. A PDF
# chunk is 1000 characters against a symbol's 4000, so the two together come to
# about the same ~12k characters as code alone.
CODE_K_WITH_DOCUMENTS = 2
DOCUMENT_K = 4


def extract_title_and_question(input_string):
    lines = input_string.strip().split("\n")

    title = ""
    question = ""
    is_question = False  # flag to know if we are inside a "Question" block

    for line in lines:
        if line.startswith("Title:"):
            title = line.split("Title: ", 1)[1].strip()
        elif line.startswith("Question:"):
            question = line.split("Question: ", 1)[1].strip()
            is_question = (
                True  # set the flag to True once we encounter a "Question:" line
            )
        elif is_question:
            # if the line does not start with "Question:" but we are inside a "Question" block,
            # then it is a continuation of the question
            question += "\n" + line.strip()

    return title, question


def as_messages(history: List[Any]) -> list:
    """(question, answer) pairs as the alternating messages a chat model wants."""
    messages = []
    for question, answer in history or []:
        messages.append(HumanMessage(content=question))
        messages.append(AIMessage(content=answer))
    return messages


def format_history(history: List[Any]) -> str:
    """The same pairs as plain text, for prompts that take a string."""
    if not history:
        return "(this is the first question of the conversation)"
    return "\n".join(
        f"Q: {question}\nA: {answer}" for question, answer in history
    )


def widen_question(question: str, history: List[Any], turns: int = 1) -> str:
    """A retrieval query for a follow-up, carrying the previous question along.

    Cheaper than asking the LLM to rewrite the question into a standalone one,
    and enough for the pronoun case ("what calls it?") that otherwise retrieves
    on a handful of stop words.
    """
    previous = [question for question, _ in (history or [])][-turns:]
    return "\n".join([*previous, question])


def configure_llm_only_chain(llm):
    # LLM only response
    template = """
    You are a helpful assistant that helps a support agent with answering programming questions.
    If you don't know the answer, just say that you don't know, you must not make up an answer.
    """
    system_message_prompt = SystemMessagePromptTemplate.from_template(template)
    human_template = "{question}"
    human_message_prompt = HumanMessagePromptTemplate.from_template(human_template)
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_message_prompt,
            # Optional, so a caller passing its own prompt (generate_ticket) is
            # unaffected and a first question needs no placeholder value.
            MessagesPlaceholder("chat_history", optional=True),
            human_message_prompt,
        ]
    )

    def generate_llm_output(
        user_input: str,
        callbacks: List[Any] = None,
        prompt=chat_prompt,
        chat_history: List[Any] = None,
    ) -> str:
        chain = prompt | llm
        inputs = {"question": user_input}
        if "chat_history" in set(prompt.input_variables) | set(prompt.optional_variables):
            inputs["chat_history"] = as_messages(chat_history)
        answer = chain.invoke(
            inputs, config={"callbacks": callbacks or []}
        ).content
        return {"answer": answer}

    return generate_llm_output


class ChatRetrievalQAWithSourcesChain(RetrievalQAWithSourcesChain):
    """Retrieval QA that can search on something other than the raw question.

    A follow-up like "and what calls it?" retrieves nothing on its own. The
    caller passes `retrieval_question` - the question widened with what came
    before it - while the LLM still answers the question actually asked.
    """

    def _get_docs(self, inputs: dict, *, run_manager) -> list:
        question = inputs.get("retrieval_question") or inputs[self.question_key]
        docs = self.retriever.invoke(
            question, config={"callbacks": run_manager.get_child()}
        )
        return self._reduce_tokens_below_limit(docs)


class CodeAndDocumentsRetriever(BaseRetriever):
    """One retriever over both things a chat can be answered from.

    A conversation may have PDFs attached to it, and a question can want either
    side or both - "does this match the spec?" needs the code and the document
    in the same context. Each half is optional: with neither there is nothing
    to retrieve, and the caller uses the LLM-only chain instead.
    """

    code: BaseRetriever | None = None
    documents: BaseRetriever | None = None

    def _get_relevant_documents(self, query: str, *, run_manager) -> list[Document]:
        found: list[Document] = []
        if self.code is not None:
            found.extend(
                self.code.invoke(query, config={"callbacks": run_manager.get_child()})
            )
        if self.documents is not None:
            for chunk in self.documents.invoke(
                query, config={"callbacks": run_manager.get_child()}
            ):
                # The chain cites metadata["source"], which for a symbol is the
                # file it lives in. A chunk knows the PDF it was cut from and,
                # since chunks are cut per page, which page that was - so a
                # citation can point at somewhere in the document.
                source = chunk.metadata.get("document_name") or "an attached PDF"
                page = chunk.metadata.get("page")
                chunk.metadata["source"] = f"{source} (p. {page})" if page else source
                found.append(chunk)
        return found


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    """The GraphRAG chain: hybrid search, then graph expansion, then the LLM.

    Returns a factory rather than a chain. What a question is answered from is
    a property of the conversation asking it - the code graph, the PDFs
    attached to that chat, or both - and only the retriever differs between
    them, so the prompt and the LLM call are built once here.
    """
    general_system_template = """
    You are an assistant answering questions about a specific codebase, and
    about any documents attached to this conversation.
    Use the following pieces of context to answer the question at the end.
    A piece of context is either a symbol (a function, method, class or type)
    taken from the codebase, together with the file it was defined in, or an
    excerpt from an attached PDF, whose Source is the name of that document.

    A symbol may be followed by comment lines describing its place in the call
    graph:
      "// Calls:"          symbols this code invokes, with their locations
      "// Called by:"      symbols that invoke this code, with their locations
      "// Types used:"     types named in this symbol's signature
      "// Used as a type by:"  code that takes or returns this type
      "// This file imports:"  other files this one depends on
    Use those to explain how the pieces fit together and to point at where
    related logic lives, but remember you were given only their signatures -
    never describe the body of a symbol whose code is not in the context.

    Base your answer on the code and the excerpts in the context, not on
    general knowledge about the libraries involved, and quote the relevant
    lines when it helps. Say which of the two an answer came from when both are
    present, and say so plainly when they disagree.
    If the context does not contain the answer, say that you could not find it
    in the indexed code or the attached documents. Do not invent files,
    functions, behaviour or passages.

    Earlier turns of this conversation, so you can resolve what a follow-up
    question refers to. Treat them as background, not as context to quote:
    ----
    {chat_history}
    ----
    {summaries}
    ----
    End every answer with a "Sources" section listing the file paths and
    document names you used, given under the Source value of the context. Only
    list sources that appear in the context. Keep the answer concise.
    """
    general_user_template = "Question:```{question}```"
    messages = [
        SystemMessagePromptTemplate.from_template(general_system_template),
        HumanMessagePromptTemplate.from_template(general_user_template),
    ]
    qa_prompt = ChatPromptTemplate.from_messages(messages)

    qa_chain = load_qa_with_sources_chain(
        llm,
        chain_type="stuff",
        prompt=qa_prompt,
    )

    kg = build_vector_store(embeddings, embeddings_store_url, username, password)
    # Opened on first use, not here: the pdf_bot index is created by the first
    # upload, so on a stack nobody has put a PDF into there is nothing to open.
    document_store = None

    def chain_for(document_ids=None, *, code: bool = True):
        """The chain that answers one question, given what it may look at."""
        nonlocal document_store
        documents = None
        if document_ids:
            if document_store is None:
                document_store = build_document_store(
                    embeddings, embeddings_store_url, username, password
                )
            documents = document_store.as_retriever(
                search_kwargs={
                    "k": DOCUMENT_K,
                    "filter": {DOCUMENT_ID_PROPERTY: {"$in": list(document_ids)}},
                }
            )

        return ChatRetrievalQAWithSourcesChain(
            combine_documents_chain=qa_chain,
            retriever=CodeAndDocumentsRetriever(
                code=kg.as_retriever(
                    search_kwargs={
                        "k": CODE_K_WITH_DOCUMENTS if documents else RETRIEVER_K
                    }
                )
                if code
                else None,
                documents=documents,
            ),
            reduce_k_below_max_tokens=False,
            max_tokens_limit=10075,
        )

    return chain_for


def answer_question(chain, question: str, history: List[Any] = None, callbacks=None) -> str:
    """Ask either chain the same way, and give both the conversation so far.

    The two have different shapes - the LLM-only chain is a plain function, the
    RAG chain is a langchain Chain - which every caller used to paper over by
    passing an inputs dict to both. The function took that dict as the question
    itself and stringified it into the prompt.
    """
    history = history or []
    callbacks = callbacks or []
    if isinstance(chain, Chain):
        return chain(
            {
                "question": question,
                "chat_history": format_history(history),
                "retrieval_question": widen_question(question, history),
            },
            callbacks=callbacks,
        )["answer"]
    return chain(question, callbacks=callbacks, chat_history=history)["answer"]


def generate_ticket(neo4j_graph, llm_chain, input_question):
    # Get high ranked questions
    records = neo4j_graph.query(
        "MATCH (q:Question) RETURN q.title AS title, q.body AS body ORDER BY q.score DESC LIMIT 3"
    )
    questions = []
    for i, question in enumerate(records, start=1):
        questions.append((question["title"], question["body"]))
    # Ask LLM to generate new question in the same style
    questions_prompt = ""
    for i, question in enumerate(questions, start=1):
        questions_prompt += f"{i}. {question[0]}\n"
        questions_prompt += f"{question[1]}\n\n"
        questions_prompt += "----\n\n"

    # The code graph has no Question nodes, so there may be no examples to
    # imitate - fall back to a plain instruction instead of an empty example list.
    style_instruction = (
        f"""Can you formulate a question in the same style, detail and tone as the following example questions?
    {questions_prompt}"""
        if questions_prompt
        else "Can you formulate a clear, detailed and well structured question?"
    )

    gen_system_template = f"""
    You're an expert in formulating high quality questions.
    {style_instruction}
    ---

    Don't make anything up, only use information in the following question.
    Return a title for the question, and the question post itself.

    Return example:
    ---
    Title: How do I use the Neo4j Python driver?
    Question: I'm trying to connect to Neo4j using the Python driver, but I'm getting an error.
    ---
    """
    # we need jinja2 since the questions themselves contain curly braces
    system_prompt = SystemMessagePromptTemplate.from_template(
        gen_system_template, template_format="jinja2"
    )
    chat_prompt = ChatPromptTemplate.from_messages(
        [
            system_prompt,
            SystemMessagePromptTemplate.from_template(
                """
                Respond in the following format or you will be unplugged.
                ---
                Title: New title
                Question: New question
                ---
                """
            ),
            HumanMessagePromptTemplate.from_template("{question}"),
        ]
    )
    llm_response = llm_chain(
        f"Here's the question to rewrite in the expected format: ```{input_question}```",
        [],
        chat_prompt,
    )
    new_title, new_question = extract_title_and_question(llm_response["answer"])
    return (new_title, new_question)

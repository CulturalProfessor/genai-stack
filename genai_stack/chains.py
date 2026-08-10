"""The LLM-only and GraphRAG chains, and the ticket-drafting helper."""
from __future__ import annotations

from typing import Any, List

from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    HumanMessagePromptTemplate,
    SystemMessagePromptTemplate,
)

from genai_stack.retrieval import build_vector_store

# How many symbols to retrieve. Context budget: extract.py truncates a symbol at
# 4000 characters, so k=4 is at most ~16k characters (~4k tokens) and fits the
# 10k token window set in load_llm alongside the prompt and the answer. Counting
# document tokens instead (reduce_k_below_max_tokens) would pull in a GPT-2
# tokenizer for the Ollama models, so the limit is enforced by chunk size and k.
RETRIEVER_K = 4


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
        [system_message_prompt, human_message_prompt]
    )

    def generate_llm_output(
        user_input: str, callbacks: List[Any], prompt=chat_prompt
    ) -> str:
        chain = prompt | llm
        answer = chain.invoke(
            {"question": user_input}, config={"callbacks": callbacks}
        ).content
        return {"answer": answer}

    return generate_llm_output


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    """The GraphRAG chain: hybrid search, then graph expansion, then the LLM."""
    general_system_template = """
    You are a code assistant answering questions about a specific codebase.
    Use the following pieces of context to answer the question at the end.
    Each piece of context is a symbol (a function, method, class or type) taken
    from the codebase, together with the file it was defined in.

    A symbol may be followed by comment lines describing its place in the call
    graph:
      "// Calls:"          symbols this code invokes, with their locations
      "// Called by:"      symbols that invoke this code, with their locations
      "// This file imports:"  other files this one depends on
    Use those to explain how the pieces fit together and to point at where
    related logic lives, but remember you were given only their signatures -
    never describe the body of a symbol whose code is not in the context.

    Base your answer on the code in the context, not on general knowledge about
    the libraries involved, and quote the relevant lines when it helps.
    If the context does not contain the answer, say that you could not find it
    in the indexed code. Do not invent files, functions or behaviour.
    ----
    {summaries}
    ----
    End every answer with a "Sources" section listing the file paths you used,
    given under the Source value of the context. Only list paths that appear in
    the context. Keep the answer concise.
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

    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": RETRIEVER_K}),
        reduce_k_below_max_tokens=False,
        max_tokens_limit=10075,
    )
    return kg_qa


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

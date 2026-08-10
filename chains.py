from langchain_community.chat_models import BedrockChat, ChatOllama
from langchain_community.embeddings import (
    BedrockEmbeddings,
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from langchain_neo4j import Neo4jVector
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
from langchain_core.prompts.chat import (
    ChatPromptTemplate,
    SystemMessagePromptTemplate,
    HumanMessagePromptTemplate,
)
from typing import List, Any
from utils import (
    CODE_EMBEDDING_PROPERTY,
    CODE_LABEL,
    CODE_TEXT_PROPERTY,
    FILE_LABEL,
    FULLTEXT_INDEX_NAME,
    VECTOR_INDEX_NAME,
    BaseLogger,
    extract_title_and_question,
)


def load_embedding_model(embedding_model_name: str, logger=BaseLogger(), config={}):
    if embedding_model_name == "ollama":
        embeddings = OllamaEmbeddings(
            base_url=config["ollama_base_url"], model="llama2"
        )
        dimension = 4096
        logger.info("Embedding: Using Ollama")
    elif embedding_model_name == "openai":
        embeddings = OpenAIEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using OpenAI")
    elif embedding_model_name == "aws":
        embeddings = BedrockEmbeddings()
        dimension = 1536
        logger.info("Embedding: Using AWS")
    else:
        embeddings = SentenceTransformerEmbeddings(
            model_name="all-MiniLM-L6-v2", cache_folder="/embedding_model"
        )
        dimension = 384
        logger.info("Embedding: Using SentenceTransformer")
    return embeddings, dimension


def load_llm(llm_name: str, logger=BaseLogger(), config={}):
    if llm_name == "gpt-4":
        logger.info("LLM: Using GPT-4")
        return ChatOpenAI(temperature=0, model_name="gpt-4", streaming=True)
    elif llm_name == "gpt-3.5":
        logger.info("LLM: Using GPT-3.5")
        return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)
    elif llm_name == "claudev2":
        logger.info("LLM: ClaudeV2")
        return BedrockChat(
            model_id="anthropic.claude-v2",
            model_kwargs={"temperature": 0.0, "max_tokens_to_sample": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            # seed=2,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=10072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)


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


# This is the graph half of the retrieval. `node` and `score` are bound by
# Neo4jVector from the vector/keyword search; from there the query walks CALLS
# in both directions and IMPORTS to pull in code the question never matched
# directly, and folds that into the `text` the LLM sees. Without this the graph
# would only be storage - the relationships have to be traversed at query time
# for the retrieval to be graph-based at all.
#
# The query must return `text`, `score` and `metadata`; `metadata.source` is
# what the chain cites.
CODE_RETRIEVAL_QUERY = f"""
WITH node, score

// The file the matched symbol lives in.
OPTIONAL MATCH (f:{FILE_LABEL})-[:CONTAINS]->(node)

// What this symbol calls.
OPTIONAL MATCH (node)-[:CALLS]->(callee:{CODE_LABEL})
WITH node, score, f, collect(DISTINCT callee)[..5] AS callees

// What calls this symbol - the callers are usually where the behaviour a
// question asks about actually lives.
OPTIONAL MATCH (caller:{CODE_LABEL})-[:CALLS]->(node)
WITH node, score, f, callees, collect(DISTINCT caller)[..5] AS callers

// Files this file depends on.
OPTIONAL MATCH (f)-[:IMPORTS]->(imported:{FILE_LABEL})
WITH node, score, f, callees, callers,
     collect(DISTINCT imported.path)[..5] AS imported_files

WITH node, score, f, callees, callers, imported_files,
     [c IN callees |
        '  ' + c.qualified_name + '  (' + c.file_path + ':' +
        toString(c.start_line) + ')  ' + coalesce(c.signature, '')] AS callee_lines,
     [c IN callers |
        '  ' + c.qualified_name + '  (' + c.file_path + ':' +
        toString(c.start_line) + ')  ' + coalesce(c.signature, '')] AS caller_lines

RETURN
    node.{CODE_TEXT_PROPERTY}
    + CASE WHEN size(callee_lines) > 0
        THEN '\\n\\n// Calls:\\n' + apoc.text.join(callee_lines, '\\n')
        ELSE '' END
    + CASE WHEN size(caller_lines) > 0
        THEN '\\n\\n// Called by:\\n' + apoc.text.join(caller_lines, '\\n')
        ELSE '' END
    + CASE WHEN size(imported_files) > 0
        THEN '\\n\\n// This file imports: ' + apoc.text.join(imported_files, ', ')
        ELSE '' END
    AS text,
    score,
    {{
        source: coalesce(f.path, node.file_path),
        symbol: node.qualified_name,
        kind: node.kind,
        signature: node.signature,
        lines: toString(node.start_line) + '-' + toString(node.end_line),
        language: f.language,
        calls: [c IN callees | c.qualified_name],
        called_by: [c IN callers | c.qualified_name],
        imports: imported_files,
        siblings: [(f)-[:CONTAINS]->(other) WHERE other <> node | other.signature][..10]
    }} AS metadata
"""


def configure_qa_rag_chain(llm, embeddings, embeddings_store_url, username, password):
    # RAG response
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

    # Vector + Knowledge Graph response. Hybrid search because code questions
    # often name a symbol exactly, which lexical matching handles better.
    kg = Neo4jVector.from_existing_index(
        embedding=embeddings,
        url=embeddings_store_url,
        username=username,
        password=password,
        database="neo4j",  # neo4j by default
        index_name=VECTOR_INDEX_NAME,
        keyword_index_name=FULLTEXT_INDEX_NAME,
        search_type="hybrid",
        node_label=CODE_LABEL,
        text_node_property=CODE_TEXT_PROPERTY,
        embedding_node_property=CODE_EMBEDDING_PROPERTY,
        retrieval_query=CODE_RETRIEVAL_QUERY,
    )

    # Context budget: fs.py truncates a symbol at 4000 characters, so k=4 is at
    # most ~16k characters (~4k tokens) and fits the 10k token context set in
    # load_llm alongside the prompt and the answer. Token-counting the documents
    # instead (reduce_k_below_max_tokens) would pull in a GPT-2 tokenizer for
    # the Ollama models, so the limit is enforced by chunk size and k.
    kg_qa = RetrievalQAWithSourcesChain(
        combine_documents_chain=qa_chain,
        retriever=kg.as_retriever(search_kwargs={"k": 4}),
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

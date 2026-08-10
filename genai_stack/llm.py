"""Loading the chat and embedding models selected by the environment."""
from __future__ import annotations

import os

from langchain_community.chat_models import BedrockChat, ChatOllama
from langchain_community.embeddings import (
    BedrockEmbeddings,
    OllamaEmbeddings,
    SentenceTransformerEmbeddings,
)
from langchain_openai import ChatOpenAI, OpenAIEmbeddings

from genai_stack.schema import BaseLogger

# anthropic.claude-v2 is retired. Bedrock model ids carry an `anthropic.` prefix;
# override with BEDROCK_MODEL_ID to pin a different one.
DEFAULT_BEDROCK_MODEL_ID = "anthropic.claude-sonnet-5"


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
    elif llm_name in ("claude", "claudev2"):  # claudev2 kept as an alias
        model_id = os.getenv("BEDROCK_MODEL_ID", DEFAULT_BEDROCK_MODEL_ID)
        logger.info(f"LLM: Using Bedrock: {model_id}")
        return BedrockChat(
            model_id=model_id,
            # Current Claude models take `max_tokens`, not the Claude 2-era
            # `max_tokens_to_sample`, and reject non-default sampling params.
            model_kwargs={"max_tokens": 1024},
            streaming=True,
        )
    elif len(llm_name):
        logger.info(f"LLM: Using Ollama: {llm_name}")
        return ChatOllama(
            temperature=0,
            base_url=config["ollama_base_url"],
            model=llm_name,
            streaming=True,
            top_k=10,  # A higher value (100) will give more diverse answers, while a lower value (10) will be more conservative.
            top_p=0.3,  # Higher value (0.95) will lead to more diverse text, while a lower value (0.5) will generate more focused text.
            num_ctx=10072,  # Sets the size of the context window used to generate the next token.
        )
    logger.info("LLM: Using GPT-3.5")
    return ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo", streaming=True)

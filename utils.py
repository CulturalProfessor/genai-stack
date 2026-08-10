"""Neo4j schema helpers shared by the loader and the retrieval chain."""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

# Graph schema. Defined once here so the loader and the retriever agree.
FILE_LABEL = "File"
CODE_LABEL = "Code"
CODE_TEXT_PROPERTY = "text"
CODE_EMBEDDING_PROPERTY = "embedding"
VECTOR_INDEX_NAME = "code_embeddings"
FULLTEXT_INDEX_NAME = "code_fulltext"


class BaseLogger:
    def __init__(self) -> None:
        self.info = print


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


def create_constraints(driver) -> None:
    """Uniqueness constraints that MERGE relies on to avoid duplicate nodes."""
    driver.query(
        f"CREATE CONSTRAINT file_path IF NOT EXISTS "
        f"FOR (f:{FILE_LABEL}) REQUIRE f.path IS UNIQUE"
    )
    driver.query(
        f"CREATE CONSTRAINT code_id IF NOT EXISTS "
        f"FOR (c:{CODE_LABEL}) REQUIRE c.id IS UNIQUE"
    )


def _existing_vector_dimension(driver) -> int | None:
    rows = driver.query(
        """
        SHOW VECTOR INDEXES YIELD name, options
        WHERE name = $name
        RETURN options.indexConfig['vector.dimensions'] AS dimension
        """,
        {"name": VECTOR_INDEX_NAME},
    )
    return rows[0]["dimension"] if rows else None


def create_vector_index(driver, dimension: int) -> None:
    """Create the Code embedding index, recreating it if the dimension changed.

    Switching EMBEDDING_MODEL changes the vector width, and an index built for
    the old width silently rejects the new vectors — so rebuild it instead.
    """
    current = _existing_vector_dimension(driver)
    if current is not None and current != dimension:
        logger.warning(
            "Vector index %s has dimension %s but the embedding model produces %s. "
            "Dropping the index; re-run the import to repopulate embeddings.",
            VECTOR_INDEX_NAME,
            current,
            dimension,
        )
        driver.query(f"DROP INDEX {VECTOR_INDEX_NAME} IF EXISTS")
        current = None

    if current is None:
        driver.query(
            f"""
            CREATE VECTOR INDEX {VECTOR_INDEX_NAME} IF NOT EXISTS
            FOR (c:{CODE_LABEL}) ON (c.{CODE_EMBEDDING_PROPERTY})
            OPTIONS {{indexConfig: {{
                `vector.dimensions`: $dimension,
                `vector.similarity_function`: 'cosine'
            }}}}
            """,
            {"dimension": dimension},
        )


def create_fulltext_index(driver) -> None:
    """Keyword index backing hybrid search.

    Code questions often name a symbol exactly ("what does syncDataToUser do"),
    which lexical matching handles better than embeddings.
    """
    driver.query(
        f"""
        CREATE FULLTEXT INDEX {FULLTEXT_INDEX_NAME} IF NOT EXISTS
        FOR (c:{CODE_LABEL})
        ON EACH [c.name, c.signature, c.{CODE_TEXT_PROPERTY}]
        """
    )

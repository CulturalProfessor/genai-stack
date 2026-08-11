"""Neo4j schema for the code graph: labels, constraints and indexes.

    (:Repository {name, source_url, files, symbols, indexed_at})
    (:File {path, name, language, depth})
    (:Code {id, name, qualified_name, kind, parent, start_line, end_line,
            signature, file_path, file_name, text, embedding})

    (:File)-[:CONTAINS]->(:Code)     a file defines a symbol
    (:Code)-[:CALLS]->(:Code)        a symbol invokes another
    (:File)-[:IMPORTS]->(:File)      a file depends on another

Repository is a record of what was imported, not part of retrieval: the
importer resolves edges by name across the whole graph, so a File is not
owned by one repository.
"""
from __future__ import annotations

import logging

logger = logging.getLogger(__name__)

FILE_LABEL = "File"
CODE_LABEL = "Code"
REPOSITORY_LABEL = "Repository"
CODE_TEXT_PROPERTY = "text"
CODE_EMBEDDING_PROPERTY = "embedding"
VECTOR_INDEX_NAME = "code_embeddings"
FULLTEXT_INDEX_NAME = "code_fulltext"


class BaseLogger:
    def __init__(self) -> None:
        self.info = print


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
    driver.query(
        f"CREATE CONSTRAINT repository_name IF NOT EXISTS "
        f"FOR (r:{REPOSITORY_LABEL}) REQUIRE r.name IS UNIQUE"
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
    the old width silently rejects the new vectors - so rebuild it instead.
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


def apply_schema(driver, dimension: int) -> None:
    """Everything the loader and the retriever need to exist up front."""
    create_constraints(driver)
    create_vector_index(driver, dimension)
    create_fulltext_index(driver)
    # Imported here rather than at module scope: neither module imports from
    # this one, and keeping the direction one-way avoids a cycle.
    from genai_stack.conversations import create_constraints as conversation_constraints
    from genai_stack.pdf_store import adopt_legacy_conversations
    from genai_stack.pdf_store import create_constraints as pdf_constraints

    conversation_constraints(driver)
    pdf_constraints(driver)
    # A database written before PDFs became attachments still has the old
    # shape; every entry point comes through here, so this is where it is fixed.
    adopt_legacy_conversations(driver)

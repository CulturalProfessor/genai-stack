"""Environment configuration, parsed once instead of in every app."""
from __future__ import annotations

import os
from dataclasses import dataclass, field

from dotenv import load_dotenv


@dataclass(frozen=True)
class Settings:
    neo4j_uri: str
    neo4j_username: str
    neo4j_password: str
    ollama_base_url: str | None
    embedding_model: str | None
    llm: str | None
    import_batch_size: int
    api_key: str
    allowed_origins: list[str] = field(default_factory=list)
    rate_limit_requests: int = 30
    rate_limit_window_seconds: int = 60
    embedding_cache: bool = True
    embedding_cache_path: str | None = None

    @property
    def embedding_config(self) -> dict:
        """Config passed to the embedding/chat model loaders."""
        return {"ollama_base_url": self.ollama_base_url}


def _split(value: str) -> list[str]:
    return [item.strip() for item in value.split(",") if item.strip()]


def load_settings() -> Settings:
    load_dotenv(".env")

    uri = os.getenv("NEO4J_URI")
    if not uri:
        raise RuntimeError("NEO4J_URI is not set - copy env.example to .env first")
    # Remapping for the LangChain Neo4j integration.
    os.environ["NEO4J_URL"] = uri

    return Settings(
        neo4j_uri=uri,
        neo4j_username=os.getenv("NEO4J_USERNAME", "neo4j"),
        neo4j_password=os.getenv("NEO4J_PASSWORD", ""),
        ollama_base_url=os.getenv("OLLAMA_BASE_URL"),
        embedding_model=os.getenv("EMBEDDING_MODEL"),
        llm=os.getenv("LLM"),
        import_batch_size=int(os.getenv("IMPORT_BATCH_SIZE", "100")),
        # Empty means no auth, which is the local-development default.
        api_key=os.getenv("API_KEY", ""),
        allowed_origins=_split(
            os.getenv("ALLOWED_ORIGINS", "http://localhost:5173")
        ),
        rate_limit_requests=int(os.getenv("RATE_LIMIT_REQUESTS", "30")),
        rate_limit_window_seconds=int(os.getenv("RATE_LIMIT_WINDOW_SECONDS", "60")),
        # On by default: re-importing the same repository is the normal case
        # while the extractor is being worked on, and a stale entry is not
        # reachable - the key covers the text and the model.
        embedding_cache=os.getenv("EMBEDDING_CACHE", "1") not in ("0", "false", "False"),
        embedding_cache_path=os.getenv("EMBEDDING_CACHE_PATH"),
    )

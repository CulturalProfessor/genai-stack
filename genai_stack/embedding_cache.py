"""A disk cache for embedding vectors, keyed by the text that produced them.

Importing an index embeds every symbol from scratch, and the work ahead - doc
comments in the embedded text, new relationship kinds, a different extractor
backend - means re-importing the same repository many times over. Most symbols
are unchanged between two runs, so most of that embedding work is repeated.

The key includes the model, because switching EMBEDDING_MODEL changes both the
vector's meaning and its width; a cache hit from the previous model would be
silently wrong in the first case and rejected by the vector index in the second.

Deliberately free of LangChain imports: the cache is duck-typed against the
`embed_documents`/`embed_query` pair that Neo4jVector calls, which keeps it
testable without the runtime stack installed - the same reason code_index.py
stays import-free.
"""
from __future__ import annotations

import hashlib
import logging
import sqlite3
import struct
from pathlib import Path
from typing import Sequence

logger = logging.getLogger(__name__)

DEFAULT_CACHE_PATH = Path(".cache/embeddings.sqlite")

_SCHEMA = """
CREATE TABLE IF NOT EXISTS embeddings (
    key       TEXT PRIMARY KEY,
    dimension INTEGER NOT NULL,
    vector    BLOB NOT NULL
)
"""


def _pack(vector: Sequence[float]) -> bytes:
    # Stored as float64 rather than float32. The extra 4 bytes a dimension is
    # nothing at this scale (~10MB for a 3k-symbol repo) and it removes any
    # question of a cached vector differing from a freshly computed one.
    return struct.pack(f"<{len(vector)}d", *vector)


def _unpack(blob: bytes) -> list[float]:
    return list(struct.unpack(f"<{len(blob) // 8}d", blob))


class CachedEmbeddings:
    """Wraps an embeddings object, serving repeats from disk.

    Both halves of the interface are cached. Documents are the bulk of the win
    on an import; queries matter for the eval harness, which asks the same
    questions on every run.
    """

    def __init__(self, inner, model_key: str, path: Path | str | None = None):
        self.inner = inner
        self.model_key = model_key
        self.path = Path(path or DEFAULT_CACHE_PATH)
        self.hits = 0
        self.misses = 0
        self.path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        # A connection per operation rather than one held open: the API serves
        # requests from several threads, and sqlite connections are not shared
        # across them. These are batch calls, so the overhead is not on any
        # hot path.
        connection = sqlite3.connect(self.path)
        connection.execute("PRAGMA journal_mode=WAL")
        return connection

    def _key(self, text: str) -> str:
        digest = hashlib.sha256()
        digest.update(self.model_key.encode("utf-8"))
        digest.update(b"\0")
        digest.update(text.encode("utf-8"))
        return digest.hexdigest()

    def _read(self, keys: set[str]) -> dict[str, list[float]]:
        if not keys:
            return {}
        found: dict[str, list[float]] = {}
        with self._connect() as connection:
            # Chunked because sqlite caps the number of bound parameters.
            ordered = list(keys)
            for start in range(0, len(ordered), 500):
                window = ordered[start : start + 500]
                placeholders = ",".join("?" * len(window))
                rows = connection.execute(
                    f"SELECT key, vector FROM embeddings WHERE key IN ({placeholders})",
                    window,
                )
                for key, blob in rows:
                    found[key] = _unpack(blob)
        return found

    def _write(self, entries: dict[str, list[float]]) -> None:
        if not entries:
            return
        with self._connect() as connection:
            connection.executemany(
                "INSERT OR REPLACE INTO embeddings (key, dimension, vector) "
                "VALUES (?, ?, ?)",
                [(key, len(v), _pack(v)) for key, v in entries.items()],
            )

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        keys = [self._key(text) for text in texts]
        vectors = self._read(set(keys))

        # Deduplicated by key, so the same text appearing twice in one batch
        # costs one call rather than two.
        pending: dict[str, str] = {}
        for key, text in zip(keys, texts):
            if key not in vectors:
                pending.setdefault(key, text)

        if pending:
            fresh = self.inner.embed_documents(list(pending.values()))
            computed = dict(zip(pending.keys(), fresh))
            self._write(computed)
            vectors.update(computed)

        self.misses += len(pending)
        self.hits += len(texts) - len(pending)
        return [vectors[key] for key in keys]

    def embed_query(self, text: str) -> list[float]:
        return self.embed_documents([text])[0]

    def stats(self) -> dict:
        total = self.hits + self.misses
        return {
            "hits": self.hits,
            "misses": self.misses,
            "hit_rate": self.hits / total if total else 0.0,
            "path": str(self.path),
        }


def wrap(inner, model_name: str | None, dimension: int, path=None, enabled: bool = True):
    """Wrap an embeddings object, or hand it back untouched when disabled."""
    if not enabled:
        return inner
    return CachedEmbeddings(
        inner, model_key=f"{model_name or 'sentence_transformer'}:{dimension}", path=path
    )

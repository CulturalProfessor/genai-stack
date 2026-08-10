"""Reading and reshaping the code index produced by fs.py.

Kept free of Neo4j and LangChain imports so the shape of the data
can be tested without the runtime stack installed.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

# Extensions tried when a relative import omits one, in resolution order.
IMPORT_SUFFIXES = ["", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs"]

# TypeScript's ESM resolution requires import specifiers to name the *emitted*
# file, so `./delay.js` is how you import `delay.ts`. Without this mapping every
# import in a modern TS codebase fails to resolve.
JS_TO_TS_EXTENSIONS = {
    ".js": [".ts", ".tsx", ".d.ts"],
    ".jsx": [".tsx"],
    ".mjs": [".mts"],
    ".cjs": [".cts"],
}


def load_index(path: Path) -> dict:
    index = json.loads(Path(path).read_text(encoding="utf-8"))
    if "files" not in index:
        raise ValueError(
            f"{path} is not a code index - regenerate it with `python3 fs.py`"
        )
    return index


def count_symbols(index: dict) -> int:
    return sum(len(f.get("symbols", [])) for f in index.get("files", []))


def build_text(file_record: dict, symbol: dict) -> str:
    """The text that gets embedded and returned to the LLM as context.

    The path and qualified name are included because a bare function body is
    often ambiguous on its own - plenty of files define a `handle` or `parse`.
    """
    header = f"File: {file_record['path']}\n{symbol['kind']} {symbol['qualified_name']}"
    return f"{header}\n\n{symbol['code']}"


def iter_rows(index: dict) -> Iterator[dict]:
    """Flatten the nested fs.py output into one row per Code node."""
    for file_record in index.get("files", []):
        for symbol in file_record.get("symbols", []):
            yield {
                "id": symbol["id"],
                "name": symbol["name"],
                "qualified_name": symbol["qualified_name"],
                "kind": symbol["kind"],
                "parent": symbol.get("parent"),
                "start_line": symbol["start_line"],
                "end_line": symbol["end_line"],
                "signature": symbol["signature"],
                "file_path": file_record["path"],
                "file_name": file_record["name"],
                "language": file_record["language"],
                "depth": file_record["depth"],
                "text": build_text(file_record, symbol),
                "calls": symbol.get("calls", []),
            }


def resolve_import(from_path: str, specifier: str, known_paths: set[str]) -> str | None:
    """Resolve a relative import to a file in the index, or None.

    Bare specifiers (`meteor/accounts-base`, `react`) are external packages and
    are deliberately not resolved - only edges inside the indexed repo are useful.
    """
    if not specifier.startswith("."):
        return None
    try:
        base = (PurePosixPath(from_path).parent / specifier).as_posix()
    except ValueError:
        return None
    # PurePosixPath keeps '..' segments, so collapse them by hand.
    parts: list[str] = []
    for part in base.split("/"):
        if part in ("", "."):
            continue
        if part == "..":
            if parts:
                parts.pop()
            continue
        parts.append(part)
    resolved = "/".join(parts)

    candidates = [resolved + suffix for suffix in IMPORT_SUFFIXES]

    # `./delay.js` -> `delay.ts`, the TypeScript ESM convention.
    for js_ext, ts_exts in JS_TO_TS_EXTENSIONS.items():
        if resolved.endswith(js_ext):
            stem = resolved[: -len(js_ext)]
            candidates += [stem + ts_ext for ts_ext in ts_exts]
            break

    candidates += [f"{resolved}/index{suffix}" for suffix in IMPORT_SUFFIXES if suffix]
    for candidate in candidates:
        if candidate in known_paths:
            return candidate
    return None


def iter_import_edges(index: dict) -> Iterator[dict]:
    """File-to-file edges for imports that point inside the indexed repo."""
    known_paths = {f["path"] for f in index.get("files", [])}
    for file_record in index.get("files", []):
        from_path = file_record["path"]
        seen: set[str] = set()
        for specifier in file_record.get("imports", []):
            to_path = resolve_import(from_path, specifier, known_paths)
            if to_path and to_path != from_path and to_path not in seen:
                seen.add(to_path)
                yield {"from_path": from_path, "to_path": to_path}


def batched(rows: Iterable[dict], size: int) -> Iterator[list[dict]]:
    if size < 1:
        raise ValueError("size must be at least 1")
    batch: list[dict] = []
    for row in rows:
        batch.append(row)
        if len(batch) == size:
            yield batch
            batch = []
    if batch:
        yield batch

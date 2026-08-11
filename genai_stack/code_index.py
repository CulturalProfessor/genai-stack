"""Reading and reshaping the code index produced by fs.py.

Kept free of Neo4j and LangChain imports so the shape of the data
can be tested without the runtime stack installed.
"""
from __future__ import annotations

import json
from pathlib import Path, PurePosixPath
from typing import Iterable, Iterator

# Extensions tried when a relative import omits one, in resolution order.
# `.d.ts` comes last so an implementation always wins over a declaration, but it
# has to be there: a package that ships its types as `foo.d.ts` is imported as
# `./foo`, and without this every one of those resolves to nothing.
IMPORT_SUFFIXES = [
    "", ".ts", ".tsx", ".mts", ".cts", ".js", ".jsx", ".mjs", ".cjs", ".d.ts",
]

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
    The doc comment leads because it describes the symbol in the vocabulary a
    question about it would use, which the code itself rarely does.
    """
    header = f"File: {file_record['path']}\n{symbol['kind']} {symbol['qualified_name']}"
    body = f"{symbol['doc']}\n{symbol['code']}" if symbol.get("doc") else symbol["code"]
    return f"{header}\n\n{body}"


def iter_file_rows(index: dict) -> Iterator[dict]:
    """One row per file, including the ones that define no symbols.

    `iter_rows` only reaches a file through its symbols, so a barrel or a
    constants module would never become a node - and an import naming one
    would find nothing to point at.
    """
    for file_record in index.get("files", []):
        yield {
            "path": file_record["path"],
            "name": file_record["name"],
            "language": file_record["language"],
            "depth": file_record["depth"],
        }


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
                "doc": symbol.get("doc"),
                "file_path": file_record["path"],
                "file_name": file_record["name"],
                "language": file_record["language"],
                "depth": file_record["depth"],
                "text": build_text(file_record, symbol),
                "calls": symbol.get("calls", []),
            }


def _candidates(base: str, known_paths: set[str]) -> str | None:
    """First of the extension and index-file spellings of `base` that exists."""
    candidates = [base + suffix for suffix in IMPORT_SUFFIXES]

    # `./delay.js` -> `delay.ts`, the TypeScript ESM convention.
    for js_ext, ts_exts in JS_TO_TS_EXTENSIONS.items():
        if base.endswith(js_ext):
            stem = base[: -len(js_ext)]
            candidates += [stem + ts_ext for ts_ext in ts_exts]
            break

    candidates += [f"{base}/index{suffix}" for suffix in IMPORT_SUFFIXES if suffix]
    for candidate in candidates:
        if candidate in known_paths:
            return candidate
    return None


def resolve_package_import(
    specifier: str, packages: dict[str, str], known_paths: set[str]
) -> str | None:
    """Resolve `@scope/pkg/sub` against the workspace packages of a monorepo.

    A bare specifier usually names an external dependency, but in a monorepo it
    just as often names a sibling package - and those are the edges that
    describe the architecture. Docusaurus imports `@docusaurus/utils` 97 times
    and every one of them resolved to nothing.
    """
    # Longest match, so `@x/utils-common` is not swallowed by `@x/utils`.
    name = max(
        (n for n in packages if specifier == n or specifier.startswith(n + "/")),
        key=len,
        default=None,
    )
    if name is None:
        return None

    directory = packages[name].rstrip("/")
    subpath = specifier[len(name) :].lstrip("/")
    if subpath:
        # `src/` first: a package's published subpath names the built layout,
        # and the source that mirrors it is what got indexed.
        for base in (f"{directory}/src/{subpath}", f"{directory}/{subpath}"):
            found = _candidates(base, known_paths)
            if found:
                return found
        return None

    for base in (f"{directory}/src/index", f"{directory}/index", f"{directory}/src"):
        found = _candidates(base, known_paths)
        if found:
            return found
    return None


def resolve_import(
    from_path: str,
    specifier: str,
    known_paths: set[str],
    packages: dict[str, str] | None = None,
) -> str | None:
    """Resolve an import to a file in the index, or None.

    Bare specifiers naming an external package (`react`) stay unresolved - only
    edges inside the indexed repo are useful - but one naming a workspace
    package is internal and does resolve.
    """
    if not specifier.startswith("."):
        if packages:
            return resolve_package_import(specifier, packages, known_paths)
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
    return _candidates("/".join(parts), known_paths)


def iter_import_edges(index: dict) -> Iterator[dict]:
    """File-to-file edges for imports that point inside the indexed repo."""
    known_paths = {f["path"] for f in index.get("files", [])}
    packages = index.get("packages") or {}
    for file_record in index.get("files", []):
        from_path = file_record["path"]
        seen: set[str] = set()
        for specifier in file_record.get("imports", []):
            to_path = resolve_import(from_path, specifier, known_paths, packages)
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

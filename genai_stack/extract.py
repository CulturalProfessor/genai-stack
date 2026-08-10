"""Extract code symbols from a TS/JS repository into JSON for the loader.

Walks a repository with tree-sitter and emits one record per file listing the
functions, methods, classes and type declarations it contains. The output feeds
``loader.py``, which embeds each symbol and writes it to Neo4j.

Usage:
    python3 fs.py --repo ../Rocket.Chat --output all_code_blocks.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import sys
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Iterator

import tree_sitter_javascript
import tree_sitter_typescript
from tree_sitter import Language, Node, Parser

logger = logging.getLogger("fs")

# Directory names skipped anywhere in the tree, and glob patterns matched
# against the repo-relative path.
DEFAULT_IGNORE_DIRS = [
    ".cache",
    ".changeset",
    ".git",
    ".github",
    ".husky",
    ".meteor",
    ".next",
    ".storybook",
    ".vscode",
    ".yarn",
    "__snapshots__",
    "assets",
    "build",
    "coverage",
    "dist",
    "i18n",
    "node_modules",
    "out",
    "public",
]
DEFAULT_IGNORE_GLOBS = ["*.min.js", "*.bundle.js", "*.test.*", "*.spec.*"]

# tree-sitter grammar to use per file extension.
LANGUAGE_BY_SUFFIX = {
    ".ts": "typescript",
    ".mts": "typescript",
    ".cts": "typescript",
    ".tsx": "tsx",
    ".js": "javascript",
    ".mjs": "javascript",
    ".cjs": "javascript",
    ".jsx": "javascript",
}

# Node types that define a symbol worth embedding on its own.
FUNCTION_NODES = {
    "function_declaration",
    "function_signature",
    "generator_function_declaration",
}
CLASS_NODES = {"class_declaration", "abstract_class_declaration"}
TYPE_NODES = {"interface_declaration", "type_alias_declaration", "enum_declaration"}
DECLARATION_NODES = {"lexical_declaration", "variable_declaration"}
MEMBER_NODES = {
    "abstract_method_signature",
    "method_definition",
    "method_signature",
    "public_field_definition",
}
FUNCTION_VALUE_NODES = {"arrow_function", "function", "function_expression"}
CALL_NODES = {"call_expression", "new_expression"}
# Bodies that can be reduced to a member list.
TYPE_BODY_NODES = {"interface_body", "object_type", "enum_body"}

_PARSERS: dict[str, Parser] = {}


def get_parser(language: str) -> Parser:
    """Return a cached parser for one of the grammars in LANGUAGE_BY_SUFFIX."""
    if language not in _PARSERS:
        if language == "typescript":
            grammar = tree_sitter_typescript.language_typescript()
        elif language == "tsx":
            grammar = tree_sitter_typescript.language_tsx()
        elif language == "javascript":
            grammar = tree_sitter_javascript.language()
        else:
            raise ValueError(f"Unsupported language: {language}")
        _PARSERS[language] = Parser(Language(grammar))
    return _PARSERS[language]


@dataclass
class Symbol:
    """One embeddable unit of code."""

    id: str
    name: str
    qualified_name: str
    kind: str
    parent: str | None
    start_line: int
    end_line: int
    signature: str
    code: str
    truncated: bool = False
    # Names invoked inside this symbol. The loader turns these into CALLS
    # relationships so retrieval can walk from a symbol to what it uses.
    calls: list[str] = field(default_factory=list)


@dataclass
class FileRecord:
    path: str
    name: str
    language: str
    depth: int
    symbols: list[Symbol] = field(default_factory=list)
    # Raw module specifiers; the loader resolves the relative ones to files.
    imports: list[str] = field(default_factory=list)


def is_ignored(rel_path: Path, ignore_dirs: set[str], ignore_globs: list[str]) -> bool:
    if any(part in ignore_dirs for part in rel_path.parts):
        return True
    return any(rel_path.match(pattern) for pattern in ignore_globs)


def iter_source_files(
    repo: Path, ignore_dirs: set[str], ignore_globs: list[str]
) -> Iterator[tuple[Path, str]]:
    """Yield (absolute path, language) for every parseable file under repo."""
    for root, dirnames, filenames in os.walk(repo):
        # Pruning in place stops os.walk from descending into ignored trees.
        dirnames[:] = [d for d in dirnames if d not in ignore_dirs]
        for filename in sorted(filenames):
            language = LANGUAGE_BY_SUFFIX.get(Path(filename).suffix)
            if language is None:
                continue
            path = Path(root) / filename
            if is_ignored(path.relative_to(repo), ignore_dirs, ignore_globs):
                continue
            yield path, language


def node_text(node: Node, source: bytes) -> str:
    return source[node.start_byte : node.end_byte].decode("utf-8", errors="replace")


def node_name(node: Node, source: bytes) -> str | None:
    name = node.child_by_field_name("name")
    return node_text(name, source) if name is not None else None


def signature_of(node: Node, source: bytes) -> str:
    """The declaration line(s) up to the body, e.g. ``function foo(a: number)``."""
    body = node.child_by_field_name("body")
    if body is None:
        # `const foo = () => {...}`: the body hangs off the assigned function.
        value = node.child_by_field_name("value")
        if value is not None:
            body = value.child_by_field_name("body")
    end = body.start_byte if body is not None else node.end_byte
    text = source[node.start_byte : end].decode("utf-8", errors="replace")
    return " ".join(text.split()).rstrip("{").strip()


def type_skeleton(node: Node, source: bytes, max_members: int = 40) -> str:
    """An interface/type/enum reduced to its members, without the doc comments.

    Type declarations are often mostly JSDoc explaining behaviour. Embedded
    whole, they out-score the code that implements that behaviour on exactly
    the questions users ask - and a type node has no CALLS edges, so a hit on
    one contributes nothing for the graph to traverse.
    """
    body = node.child_by_field_name("body") or node.child_by_field_name("value")
    if body is None or body.type not in TYPE_BODY_NODES:
        # Unions, function types and other short forms: keep them whole,
        # collapsed onto one line.
        return " ".join(node_text(node, source).split())

    header = source[node.start_byte : body.start_byte].decode("utf-8", errors="replace")
    lines = [" ".join(header.split()) + " {"]
    members = [m for m in body.named_children if m.type != "comment"]
    for member in members[:max_members]:
        lines.append("    " + " ".join(node_text(member, source).split()))
    if len(members) > max_members:
        lines.append(f"    // ... {len(members) - max_members} more members")
    lines.append("}")
    return "\n".join(lines)


def class_skeleton(node: Node, source: bytes) -> str:
    """A class header plus its member signatures, without the member bodies.

    Each method is embedded as its own symbol, so embedding the full class text
    too would just duplicate that content in the vector store.
    """
    lines = [signature_of(node, source) + " {"]
    body = node.child_by_field_name("body")
    if body is not None:
        for member in body.named_children:
            if member.type in MEMBER_NODES:
                lines.append("    " + signature_of(member, source))
    lines.append("}")
    return "\n".join(lines)


def callee_name(node: Node | None, source: bytes) -> str | None:
    """The name being invoked: `foo()` -> foo, `this.foo()`/`a.b.foo()` -> foo."""
    if node is None:
        return None
    if node.type == "identifier":
        return node_text(node, source)
    if node.type == "member_expression":
        prop = node.child_by_field_name("property")
        return node_text(prop, source) if prop is not None else None
    return None


def collect_calls(node: Node, source: bytes, exclude: str | None) -> list[str]:
    """Every function/constructor name invoked anywhere inside `node`."""
    names: set[str] = set()
    stack = [node]
    while stack:
        current = stack.pop()
        if current.type in CALL_NODES:
            target = current.child_by_field_name(
                "function"
            ) or current.child_by_field_name("constructor")
            name = callee_name(target, source)
            # Self-recursion would only add a self-loop to the graph.
            if name and name != exclude:
                names.add(name)
        stack.extend(current.named_children)
    return sorted(names)


def extract_imports(root: Node, source: bytes) -> list[str]:
    """Module specifiers pulled in by this file, from `import` and `require`."""
    specifiers: set[str] = set()
    stack = [root]
    while stack:
        current = stack.pop()
        if current.type in {"import_statement", "export_statement"}:
            source_node = current.child_by_field_name("source")
            if source_node is not None:
                specifiers.add(node_text(source_node, source).strip("'\"`"))
        elif current.type == "call_expression":
            target = current.child_by_field_name("function")
            if target is not None and node_text(target, source) == "require":
                args = current.child_by_field_name("arguments")
                if args is not None and args.named_children:
                    first = args.named_children[0]
                    if first.type == "string":
                        specifiers.add(node_text(first, source).strip("'\"`"))
        stack.extend(current.named_children)
    return sorted(specifiers)


class SymbolExtractor:
    """Collects symbols from one parsed file."""

    def __init__(self, source: bytes, rel_path: str, max_chars: int, min_chars: int):
        self.source = source
        self.rel_path = rel_path
        self.max_chars = max_chars
        self.min_chars = min_chars
        self.symbols: list[Symbol] = []

    def add(
        self,
        node: Node,
        name: str | None,
        kind: str,
        parent: str | None,
        code: str | None = None,
        calls: list[str] | None = None,
    ) -> None:
        if not name:
            return
        code = code if code is not None else node_text(node, self.source)
        if len(code.strip()) < self.min_chars:
            return
        truncated = len(code) > self.max_chars
        if truncated:
            code = code[: self.max_chars] + "\n// ... truncated"
        qualified_name = f"{parent}.{name}" if parent else name
        start_line = node.start_point[0] + 1
        self.symbols.append(
            Symbol(
                # Line number keeps the id unique when a name repeats in a file,
                # and keeps it stable across runs when the file is unchanged.
                id=f"{self.rel_path}:{start_line}:{qualified_name}",
                name=name,
                qualified_name=qualified_name,
                kind=kind,
                parent=parent,
                start_line=start_line,
                end_line=node.end_point[0] + 1,
                signature=signature_of(node, self.source),
                code=code,
                truncated=truncated,
                calls=(
                    calls
                    if calls is not None
                    else collect_calls(node, self.source, name)
                ),
            )
        )

    def visit(self, node: Node, parent: str | None = None) -> None:
        for child in node.named_children:
            kind = child.type
            if kind == "export_statement" or kind == "ambient_declaration":
                # Unwrap so `export function foo()` is treated as `function foo()`.
                self.visit(child, parent)
            elif kind in CLASS_NODES:
                name = node_name(child, self.source)
                # calls=[] because the class symbol is only a skeleton - the
                # calls inside each body belong to that method's symbol.
                self.add(
                    child,
                    name,
                    "class",
                    parent,
                    code=class_skeleton(child, self.source),
                    calls=[],
                )
                body = child.child_by_field_name("body")
                if body is not None:
                    self.visit_members(body, name or parent)
            elif kind in FUNCTION_NODES:
                self.add(child, node_name(child, self.source), "function", parent)
            elif kind in TYPE_NODES:
                self.add(
                    child,
                    node_name(child, self.source),
                    kind.replace("_declaration", ""),
                    parent,
                    code=type_skeleton(child, self.source),
                    calls=[],
                )
            elif kind in DECLARATION_NODES:
                self.visit_declaration(child, parent)
            else:
                # Recurse into namespaces, blocks and anything else that may
                # wrap a declaration. Recorded nodes are never re-entered, so
                # nested helpers are captured with their enclosing symbol.
                self.visit(child, parent)

    def visit_declaration(self, node: Node, parent: str | None) -> None:
        """Record `const foo = () => {}` style function bindings."""
        for declarator in node.named_children:
            if declarator.type != "variable_declarator":
                continue
            value = declarator.child_by_field_name("value")
            if value is not None and value.type in FUNCTION_VALUE_NODES:
                self.add(declarator, node_name(declarator, self.source), "function", parent)

    def visit_members(self, body: Node, class_name: str | None) -> None:
        for member in body.named_children:
            if member.type not in MEMBER_NODES:
                continue
            name = node_name(member, self.source)
            if member.type == "public_field_definition":
                # Only fields holding a function are worth their own symbol.
                value = member.child_by_field_name("value")
                if value is None or value.type not in FUNCTION_VALUE_NODES:
                    continue
            self.add(member, name, "method", class_name)


def parse_file(
    source: bytes,
    language: str,
    rel_path: str,
    max_chars: int = 4000,
    min_chars: int = 0,
) -> tuple[list[Symbol], list[str]]:
    """Parse once, returning the file's symbols and its import specifiers."""
    tree = get_parser(language).parse(source)
    extractor = SymbolExtractor(source, rel_path, max_chars, min_chars)
    extractor.visit(tree.root_node)
    return extractor.symbols, extract_imports(tree.root_node, source)


def extract_symbols(
    source: bytes,
    language: str,
    rel_path: str,
    max_chars: int = 4000,
    min_chars: int = 0,
) -> list[Symbol]:
    return parse_file(source, language, rel_path, max_chars, min_chars)[0]


def build_index(
    repo: Path,
    ignore_dirs: set[str],
    ignore_globs: list[str],
    max_chars: int,
    min_chars: int,
) -> list[FileRecord]:
    records: list[FileRecord] = []
    failures = 0
    for path, language in iter_source_files(repo, ignore_dirs, ignore_globs):
        rel_path = path.relative_to(repo).as_posix()
        try:
            source = path.read_bytes()
            symbols, imports = parse_file(
                source, language, rel_path, max_chars, min_chars
            )
        except Exception as exc:  # noqa: BLE001 - one bad file must not stop the run
            failures += 1
            logger.warning("Skipping %s: %s", rel_path, exc)
            continue
        if symbols:
            records.append(
                FileRecord(
                    path=rel_path,
                    name=path.name,
                    language=language,
                    depth=len(path.relative_to(repo).parts) - 1,
                    symbols=symbols,
                    imports=imports,
                )
            )
    if failures:
        logger.warning("%d file(s) could not be parsed", failures)
    return records


def build_payload(
    repo: Path,
    repo_name: str | None = None,
    ignore_dirs: set[str] | None = None,
    ignore_globs: list[str] | None = None,
    max_chars: int = 4000,
    min_chars: int = 0,
    source_url: str | None = None,
) -> dict:
    """Index a checkout and return the JSON structure the loader consumes.

    Shared by the CLI and the loader UI so both produce an identical index.
    """
    repo = Path(repo)
    records = build_index(
        repo,
        set(DEFAULT_IGNORE_DIRS) | set(ignore_dirs or ()),
        DEFAULT_IGNORE_GLOBS + list(ignore_globs or ()),
        max_chars,
        min_chars,
    )
    payload: dict = {
        "repo": repo_name or repo.name,
        "files": [asdict(record) for record in records],
    }
    if source_url:
        payload["source_url"] = source_url
    return payload


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    source = parser.add_mutually_exclusive_group()
    source.add_argument("--repo", type=Path, help="Local repository to index")
    source.add_argument(
        "--github-url",
        help="Public repository URL to clone and index, e.g. owner/repo",
    )
    parser.add_argument("--branch", help="Branch to clone (with --github-url)")
    # Written into index/, which docker-compose mounts into the loader container.
    parser.add_argument("--output", type=Path, default=Path("index/all_code_blocks.json"))
    # 4000 chars x the retriever's k=4 keeps the RAG prompt inside the LLM context.
    parser.add_argument("--max-chars", type=int, default=4000, help="Truncate symbols longer than this")
    parser.add_argument("--min-chars", type=int, default=0, help="Skip symbols shorter than this")
    parser.add_argument("--ignore-dir", action="append", default=[], help="Extra directory name to skip")
    parser.add_argument("--ignore-glob", action="append", default=[], help="Extra path glob to skip")
    parser.add_argument("--verbose", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s - %(levelname)s - %(message)s",
    )

    build_kwargs = dict(
        ignore_dirs=set(args.ignore_dir),
        ignore_globs=args.ignore_glob,
        max_chars=args.max_chars,
        min_chars=args.min_chars,
    )

    if args.github_url:
        from genai_stack.sources import cloned_repository, parse_repo_url

        try:
            ref = parse_repo_url(args.github_url, args.branch)
        except ValueError as exc:
            logger.error("%s", exc)
            return 1
        logger.info("Cloning %s...", ref.slug)
        try:
            with cloned_repository(ref) as checkout:
                payload = build_payload(
                    checkout,
                    repo_name=ref.name,
                    source_url=ref.clone_url,
                    **build_kwargs,
                )
        except RuntimeError as exc:
            logger.error("%s", exc)
            return 1
    else:
        repo = (args.repo or Path("../Rocket.Chat")).resolve()
        if not repo.is_dir():
            logger.error("Repository not found: %s", repo)
            return 1
        payload = build_payload(repo, **build_kwargs)

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    total_symbols = sum(len(f["symbols"]) for f in payload["files"])
    logger.info(
        "Wrote %d symbols from %d files to %s",
        total_symbols,
        len(payload["files"]),
        args.output,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Static health metrics for the code graph.

This is not the retrieval eval. It answers a narrower and much cheaper question
- is the graph well formed? - with no questions, no gold data and no LLM. That
is enough to score the relationship-modelling work: resolving calls through the
import graph and giving type nodes edges both show up here directly.

Two modes, one report:

    --index index/all_code_blocks.json    from the extractor's output
    --neo4j                               from the live graph

The index mode is fast enough for a tight loop, and runs with nothing installed
beyond the standard library. It calls the same CallResolver the loader does, so
it reports what would be written rather than a second opinion about it. The
Neo4j mode is the check that the writing actually happened.

Usage:
    python3 -m eval.graph_health --index index/all_code_blocks.json
    python3 -m eval.graph_health --neo4j --json
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from dataclasses import dataclass, field
from pathlib import Path, PurePosixPath

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.code_index import (
    REFERENCE_EDGES,
    CallResolver,
    iter_reference_edges,
    resolve_import,
)

# The relationships a symbol can be reachable through, taken from the loader's
# own mapping rather than restated here. Neo4j mode used to ask only about
# CALLS, which is how it kept reporting the graph as it stood before type edges
# existed: a type node has no calls by construction, so 891 of them read as
# unreachable while index mode put the same graph at 80% reachable.
REFERENCE_PATTERN = "|".join(sorted(set(REFERENCE_EDGES.values())))

# Kinds that carry no call edges by construction, so their reachability depends
# entirely on the type edges that do not exist yet.
TYPE_KINDS = {"interface", "type_alias", "enum"}

# An in-degree at or above this floods the "Called by" list of every symbol that
# touches the node. Reported rather than enforced; the cap belongs in retrieval.
SUPER_NODE_DEGREE = 20

# Suffixes a bundler resolves but the extractor never will. `import './x.css'`
# is not a dangling edge, it is not an edge at all - counting these as failures
# put the unresolved rate at 48% when the real figure is a seventh of that.
ASSET_SUFFIXES = {
    ".css", ".scss", ".sass", ".less",
    ".png", ".jpg", ".jpeg", ".gif", ".svg", ".webp", ".ico",
    ".json", ".html", ".md", ".mdx", ".yml", ".yaml", ".txt",
    ".woff", ".woff2", ".ttf", ".eot",
}


@dataclass
class HealthReport:
    source: str
    files: int = 0
    symbols: int = 0
    kinds: dict = field(default_factory=dict)

    # Call resolution. Only the index knows about calls that resolved to
    # nothing - once imported, a dropped call leaves no trace in the graph.
    call_sites: int | None = None
    resolved_local: int | None = None
    resolved_import: int | None = None
    resolved_global: int | None = None
    dropped_ambiguous: int | None = None
    dropped_external: int | None = None

    call_edges: int = 0
    import_edges: int = 0
    edges_by_type: dict = field(default_factory=dict)
    # Relative specifiers that name code, and how many of them land on a file
    # the index knows about. Asset imports are counted apart, since they are
    # unresolvable by design rather than by failure.
    code_imports: int | None = None
    dangling_code_imports: int | None = None
    asset_imports: int | None = None
    # Bare specifiers that turned out to name a package of this monorepo.
    package_imports: int | None = None

    isolated_symbols: int = 0
    type_symbols: int = 0
    type_symbols_with_edges: int = 0
    symbols_with_doc: int = 0

    in_degree_p50: int = 0
    in_degree_p95: int = 0
    in_degree_max: int = 0
    super_nodes: int = 0
    super_node_edges: int = 0

    @property
    def resolvable_call_sites(self) -> int | None:
        """Call sites naming something defined inside the repo."""
        if self.call_sites is None:
            return None
        return (
            self.resolved_local
            + self.resolved_import
            + self.resolved_global
            + self.dropped_ambiguous
        )

    @property
    def resolution_rate(self) -> float | None:
        resolvable = self.resolvable_call_sites
        if not resolvable:
            return None
        return (
            self.resolved_local + self.resolved_import + self.resolved_global
        ) / resolvable


def _percentile(values: list[int], fraction: float) -> int:
    if not values:
        return 0
    ordered = sorted(values)
    index = min(int(fraction * len(ordered)), len(ordered) - 1)
    return ordered[index]


def _degree_stats(report: HealthReport, in_degrees: list[int]) -> None:
    report.in_degree_p50 = _percentile(in_degrees, 0.50)
    report.in_degree_p95 = _percentile(in_degrees, 0.95)
    report.in_degree_max = max(in_degrees, default=0)
    heavy = [d for d in in_degrees if d >= SUPER_NODE_DEGREE]
    report.super_nodes = len(heavy)
    report.super_node_edges = sum(heavy)


def from_index(index: dict) -> HealthReport:
    """Metrics computed from the extractor's JSON, without touching Neo4j."""
    report = HealthReport(source="index")
    files = index.get("files", [])
    report.files = len(files)

    files_by_path = {f["path"]: f for f in files}
    by_name: Counter = Counter()
    in_file: set[tuple[str, str]] = set()
    for record in files:
        for symbol in record.get("symbols", []):
            by_name[symbol["name"]] += 1
            in_file.add((record["path"], symbol["name"]))

    report.symbols = sum(len(f.get("symbols", [])) for f in files)
    report.kinds = dict(
        Counter(s["kind"] for f in files for s in f.get("symbols", []))
    )
    report.symbols_with_doc = sum(
        1 for f in files for s in f.get("symbols", []) if s.get("doc")
    )

    # The real resolver, not a copy of it. An earlier version of this reported
    # a classification that mirrored the Cypher in ingest.py, which meant the
    # instrument could quietly disagree with the thing it was measuring.
    resolver = CallResolver(index)
    local = via_import = global_unique = ambiguous = external = 0
    in_degree: Counter = Counter()
    has_out: set[str] = set()

    for record in files:
        path = record["path"]
        for symbol in record.get("symbols", []):
            for callee in symbol.get("calls", []):
                target = resolver.resolve(path, callee)
                if target is None:
                    if by_name.get(callee, 0) > 1:
                        ambiguous += 1
                    else:
                        external += 1
                    continue

                if target == path:
                    local += 1
                elif target == "*":
                    global_unique += 1
                else:
                    via_import += 1

    report.call_sites = local + via_import + global_unique + ambiguous + external
    report.resolved_local = local
    report.resolved_import = via_import
    report.resolved_global = global_unique
    report.dropped_ambiguous = ambiguous
    report.dropped_external = external

    # Every edge kind, not just calls: a type node has no calls by
    # construction, so counting only those would show it as unreachable even
    # once something references it.
    connected: set[str] = set()
    for edge in iter_reference_edges(index):
        report.edges_by_type[edge["type"]] = (
            report.edges_by_type.get(edge["type"], 0) + 1
        )
        connected.add(edge["caller_id"])
        connected.add(edge["callee_id"])
        in_degree[edge["callee_id"]] += 1
    report.call_edges = report.edges_by_type.get("CALLS", 0)

    known_paths = {f["path"] for f in files}
    packages = index.get("packages") or {}
    aliases = index.get("aliases") or []
    code_imports = dangling = assets = package_imports = 0
    import_pairs: set[tuple[str, str]] = set()
    for record in files:
        for specifier in record.get("imports", []):
            if not specifier.startswith("."):
                # Bare, so external unless it names a package of this monorepo.
                target = resolve_import(
                    record["path"], specifier, known_paths, packages, aliases
                )
                if target is None:
                    continue
                package_imports += 1
                if target != record["path"]:
                    import_pairs.add((record["path"], target))
                continue
            if PurePosixPath(specifier).suffix in ASSET_SUFFIXES:
                assets += 1
                continue
            code_imports += 1
            target = resolve_import(
                record["path"], specifier, known_paths, packages, aliases
            )
            if target is None:
                # Almost always a module the extractor produced no symbols for:
                # build_index drops a file with an empty symbol list, and
                # visit_declaration only records a const whose value is a
                # function. Barrels and constant modules therefore never become
                # nodes, and every import naming one dangles.
                dangling += 1
            elif target != record["path"]:
                import_pairs.add((record["path"], target))
    report.code_imports = code_imports
    report.dangling_code_imports = dangling
    report.asset_imports = assets
    report.package_imports = package_imports
    report.import_edges = len(import_pairs)

    isolated = 0
    types = types_linked = 0
    for record in files:
        for symbol in record.get("symbols", []):
            linked = symbol["id"] in connected
            if not linked:
                isolated += 1
            if symbol["kind"] in TYPE_KINDS:
                types += 1
                if linked:
                    types_linked += 1
    report.isolated_symbols = isolated
    report.type_symbols = types
    report.type_symbols_with_edges = types_linked

    _degree_stats(report, list(in_degree.values()))
    return report


def from_neo4j(graph) -> HealthReport:
    """The same metrics read back out of the database.

    Anything the import discards - an ambiguous call, an import that resolved to
    nothing - is not recoverable here, so the resolution breakdown stays unset
    and only the realised edge counts are comparable with index mode.
    """
    from genai_stack.schema import CODE_LABEL, FILE_LABEL

    report = HealthReport(source="neo4j")
    counts = graph.query(
        f"""
        CALL () {{ MATCH (f:{FILE_LABEL}) RETURN count(f) AS files }}
        CALL () {{ MATCH (c:{CODE_LABEL}) RETURN count(c) AS symbols }}
        CALL () {{ MATCH (:{FILE_LABEL})-[r:IMPORTS]->() RETURN count(r) AS import_edges }}
        RETURN files, symbols, import_edges
        """
    )[0]
    report.files = counts["files"]
    report.symbols = counts["symbols"]
    report.import_edges = counts["import_edges"]

    report.edges_by_type = {
        row["type"]: row["total"]
        for row in graph.query(
            f"""
            MATCH (:{CODE_LABEL})-[r:{REFERENCE_PATTERN}]->(:{CODE_LABEL})
            RETURN type(r) AS type, count(*) AS total
            """
        )
    }
    report.call_edges = report.edges_by_type.get("CALLS", 0)

    # Only ask about relationships the graph actually has. Naming an absent one
    # is not an error, but Neo4j warns about it on every query, and a repository
    # with no class heritage would print three warnings per run.
    present = "|".join(sorted(report.edges_by_type)) or REFERENCE_PATTERN

    report.kinds = {
        row["kind"]: row["total"]
        for row in graph.query(
            f"MATCH (c:{CODE_LABEL}) RETURN c.kind AS kind, count(*) AS total"
        )
    }
    report.symbols_with_doc = graph.query(
        f"MATCH (c:{CODE_LABEL}) WHERE c.doc IS NOT NULL RETURN count(*) AS total"
    )[0]["total"]

    report.isolated_symbols = graph.query(
        f"MATCH (c:{CODE_LABEL}) WHERE NOT (c)-[:{present}]-() "
        "RETURN count(*) AS total"
    )[0]["total"]

    types = graph.query(
        f"""
        MATCH (c:{CODE_LABEL}) WHERE c.kind IN $kinds
        RETURN count(*) AS total,
               count(CASE WHEN (c)-[:{present}]-() THEN 1 END) AS linked
        """,
        {"kinds": sorted(TYPE_KINDS)},
    )[0]
    report.type_symbols = types["total"]
    report.type_symbols_with_edges = types["linked"]

    in_degrees = [
        row["degree"]
        for row in graph.query(
            f"""
            MATCH (c:{CODE_LABEL})
            WITH c, count {{ (c)<-[:{present}]-() }} AS degree
            WHERE degree > 0
            RETURN degree
            """
        )
    ]
    _degree_stats(report, in_degrees)
    return report


def _line(label: str, value, total=None) -> str:
    if value is None:
        return f"  {label:<38} -"
    if total:
        return f"  {label:<38} {value:>7,}  {value / total:>6.1%}"
    return f"  {label:<38} {value:>7,}"


def format_report(report: HealthReport) -> str:
    out = [f"graph health ({report.source})", ""]
    out.append(_line("files", report.files))
    out.append(_line("symbols", report.symbols))
    for kind, total in sorted(report.kinds.items(), key=lambda kv: -kv[1]):
        out.append(_line(f"  kind: {kind}", total, report.symbols))
    out.append("")

    if report.call_sites is not None:
        out.append(_line("call sites", report.call_sites))
        out.append(_line("  resolved in-file", report.resolved_local, report.call_sites))
        out.append(_line("  resolved via an import", report.resolved_import, report.call_sites))
        out.append(_line("  guessed by unique name", report.resolved_global, report.call_sites))
        out.append(_line("  DROPPED ambiguous", report.dropped_ambiguous, report.call_sites))
        out.append(_line("  external / unknown", report.dropped_external, report.call_sites))
        rate = report.resolution_rate
        out.append(
            f"  {'intra-repo resolution rate':<38} {rate:>7.1%}"
            if rate is not None
            else _line("intra-repo resolution rate", None)
        )
        out.append("")

    for name, total in sorted(report.edges_by_type.items(), key=lambda kv: -kv[1]):
        out.append(_line(f"{name} edges", total))
    out.append(_line("IMPORTS edges", report.import_edges))
    if report.code_imports:
        out.append(_line("relative code imports", report.code_imports))
        out.append(
            _line(
                "  DANGLING (target not indexed)",
                report.dangling_code_imports,
                report.code_imports,
            )
        )
        out.append(_line("workspace package imports", report.package_imports))
        out.append(_line("relative asset imports (not edges)", report.asset_imports))
    out.append("")

    out.append(_line("isolated symbols (no edge either way)", report.isolated_symbols, report.symbols))
    out.append(_line("type/interface symbols", report.type_symbols, report.symbols))
    out.append(
        _line("  reachable in the graph", report.type_symbols_with_edges, report.type_symbols or None)
    )
    out.append(_line("symbols carrying a doc comment", report.symbols_with_doc, report.symbols))
    out.append("")

    out.append(_line("in-degree p50", report.in_degree_p50))
    out.append(_line("in-degree p95", report.in_degree_p95))
    out.append(_line("in-degree max", report.in_degree_max))
    out.append(
        _line(f"nodes with in-degree >= {SUPER_NODE_DEGREE}", report.super_nodes)
    )
    out.append(
        _line(
            "  edges they absorb",
            report.super_node_edges,
            sum(report.edges_by_type.values()) or None,
        )
    )
    return "\n".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--index", type=Path, help="Index JSON to measure")
    parser.add_argument("--neo4j", action="store_true", help="Measure the live graph")
    parser.add_argument("--json", action="store_true", help="Emit JSON, for snapshots")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if not args.index and not args.neo4j:
        args.index = Path("index/all_code_blocks.json")

    reports = []
    if args.index:
        if not args.index.is_file():
            print(f"No such index: {args.index}", file=sys.stderr)
            return 1
        reports.append(from_index(json.loads(args.index.read_text(encoding="utf-8"))))
    if args.neo4j:
        from langchain_neo4j import Neo4jGraph

        from genai_stack.settings import load_settings

        settings = load_settings()
        reports.append(
            from_neo4j(
                Neo4jGraph(
                    url=settings.neo4j_uri,
                    username=settings.neo4j_username,
                    password=settings.neo4j_password,
                )
            )
        )

    if args.json:
        payload = [
            {
                k: v
                for k, v in vars(report).items()
            }
            | {"resolution_rate": report.resolution_rate}
            for report in reports
        ]
        print(json.dumps(payload, indent=2))
    else:
        print("\n\n".join(format_report(report) for report in reports))

    return 0


if __name__ == "__main__":
    sys.exit(main())

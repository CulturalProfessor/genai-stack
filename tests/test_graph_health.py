import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.graph_health import from_index


def symbol(name, calls=(), kind="function", line=1, doc=None):
    record = {
        "id": f"{name}:{line}",
        "name": name,
        "qualified_name": name,
        "kind": kind,
        "parent": None,
        "start_line": line,
        "end_line": line + 1,
        "signature": f"function {name}()",
        "code": f"function {name}() {{}}",
        "calls": list(calls),
    }
    if doc is not None:
        record["doc"] = doc
    return record


def file_record(path, symbols, imports=()):
    return {
        "path": path,
        "name": path.rsplit("/", 1)[-1],
        "language": "typescript",
        "depth": path.count("/"),
        "symbols": symbols,
        "imports": list(imports),
    }


def test_call_sites_are_classified_the_way_the_import_does():
    """Same file wins, then a name unique across the index; with no import
    binding to go on, anything else is dropped."""
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["helper", "unique", "shared", "fetch"]),
                                 symbol("helper", line=10)]),
            file_record("b.ts", [symbol("unique"), symbol("shared", line=20)]),
            file_record("c.ts", [symbol("shared", line=30)]),
        ]
    }
    report = from_index(index)

    assert report.call_sites == 4
    assert report.resolved_local == 1       # helper, defined in a.ts
    assert report.resolved_global == 1      # unique, one definition anywhere
    assert report.dropped_ambiguous == 1    # shared, defined twice
    assert report.dropped_external == 1     # fetch, defined nowhere
    assert report.resolution_rate == 2 / 3


def test_a_local_name_defined_twice_in_one_file_makes_two_edges():
    index = {
        "files": [
            file_record(
                "a.ts",
                [symbol("caller", ["dup"]), symbol("dup", line=5), symbol("dup", line=9)],
            )
        ]
    }
    assert from_index(index).call_edges == 2


def test_isolated_symbols_are_those_with_no_call_either_way():
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["callee"]), symbol("lonely", line=7)]),
            file_record("b.ts", [symbol("callee")]),
        ]
    }
    report = from_index(index)

    assert report.symbols == 3
    assert report.isolated_symbols == 1


def test_type_nodes_are_counted_and_start_unreachable():
    index = {
        "files": [
            file_record(
                "a.ts",
                [symbol("Props", kind="interface"), symbol("Opts", kind="type_alias", line=5)],
            )
        ]
    }
    report = from_index(index)

    assert report.type_symbols == 2
    assert report.type_symbols_with_edges == 0


def test_asset_imports_are_not_counted_as_dangling():
    """`import './x.css'` is not an edge that failed, it is not an edge."""
    index = {
        "files": [
            file_record(
                "a.ts",
                [symbol("f")],
                imports=["./styles.module.css", "./logo.png", "react", "./b"],
            ),
            file_record("b.ts", [symbol("g")]),
        ]
    }
    report = from_index(index)

    assert report.asset_imports == 2
    assert report.code_imports == 1
    assert report.dangling_code_imports == 0
    assert report.import_edges == 1


def test_an_import_of_an_unindexed_module_dangles():
    index = {"files": [file_record("a.ts", [symbol("f")], imports=["./constants"])]}
    report = from_index(index)

    assert report.code_imports == 1
    assert report.dangling_code_imports == 1
    assert report.import_edges == 0


def test_doc_coverage_counts_symbols_carrying_a_comment():
    index = {
        "files": [
            file_record(
                "a.ts",
                [symbol("documented", doc="/** why */"), symbol("bare", line=9)],
            )
        ]
    }
    assert from_index(index).symbols_with_doc == 1

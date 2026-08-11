import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.code_index import CallResolver, iter_call_edges


def symbol(name, calls=(), exported=False, line=1):
    return {
        "id": f"{name}:{line}", "name": name, "qualified_name": name,
        "kind": "function", "parent": None, "start_line": line, "end_line": line + 1,
        "signature": f"function {name}()", "code": "", "calls": list(calls),
        "exported": exported,
    }


def file_record(path, symbols, bindings=(), re_exports=(), imports=()):
    return {
        "path": path, "name": path.rsplit("/", 1)[-1], "language": "typescript",
        "depth": path.count("/"), "symbols": symbols, "imports": list(imports),
        "bindings": list(bindings), "re_exports": list(re_exports),
    }


def test_a_definition_in_the_same_file_wins():
    index = {"files": [file_record("a.ts", [symbol("caller", ["helper"]), symbol("helper", line=9)])]}
    assert CallResolver(index).resolve("a.ts", "helper") == "a.ts"


def test_an_import_picks_the_right_one_of_two_same_named_symbols():
    """The whole point: `handle` is ambiguous by name, but this file imports
    the one from b.ts."""
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["handle"])],
                        bindings=[{"local": "handle", "imported": "handle", "specifier": "./b"}]),
            file_record("b.ts", [symbol("handle", exported=True)]),
            file_record("c.ts", [symbol("handle", exported=True)]),
        ]
    }
    assert CallResolver(index).resolve("a.ts", "handle") == "b.ts"


def test_an_alias_is_looked_up_under_its_original_name():
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["run"])],
                        bindings=[{"local": "run", "imported": "execute", "specifier": "./b"}]),
            file_record("b.ts", [symbol("execute", exported=True)]),
        ]
    }
    assert CallResolver(index).resolve("a.ts", "run") == "b.ts"


def test_a_barrel_is_followed_to_the_file_behind_it():
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["deep"])],
                        bindings=[{"local": "deep", "imported": "deep", "specifier": "./pkg"}]),
            file_record("pkg/index.ts", [], re_exports=[{"specifier": "./impl", "imported": "*", "local": "*"}]),
            file_record("pkg/impl.ts", [symbol("deep", exported=True)]),
        ]
    }
    assert CallResolver(index).resolve("a.ts", "deep") == "pkg/impl.ts"


def test_a_cycle_between_two_barrels_terminates():
    index = {
        "files": [
            file_record("a/index.ts", [], re_exports=[{"specifier": "../b/index", "imported": "*", "local": "*"}]),
            file_record("b/index.ts", [], re_exports=[{"specifier": "../a/index", "imported": "*", "local": "*"}]),
        ]
    }
    assert CallResolver(index).find_export("a/index.ts", "missing") is None


def test_a_unique_name_still_resolves_without_an_import():
    index = {"files": [
        file_record("a.ts", [symbol("caller", ["only"])]),
        file_record("b.ts", [symbol("only")]),
    ]}
    assert CallResolver(index).resolve("a.ts", "only") == "*"


def test_an_ambiguous_name_with_no_import_is_dropped():
    index = {"files": [
        file_record("a.ts", [symbol("caller", ["shared"])]),
        file_record("b.ts", [symbol("shared")]),
        file_record("c.ts", [symbol("shared")]),
    ]}
    assert CallResolver(index).resolve("a.ts", "shared") is None


def test_a_language_global_is_not_linked_to_a_symbol_of_the_same_name():
    """A repo defining its own Error should not collect an edge from every
    `new Error()` in the codebase."""
    index = {"files": [
        file_record("a.ts", [symbol("caller", ["Error"])]),
        file_record("b.ts", [symbol("Error")]),
    ]}
    assert CallResolver(index).resolve("a.ts", "Error") is None


def test_an_unexported_symbol_is_not_reachable_through_an_import():
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["hidden"])],
                        bindings=[{"local": "hidden", "imported": "hidden", "specifier": "./b"}]),
            file_record("b.ts", [symbol("hidden", exported=False)]),
            file_record("c.ts", [symbol("hidden", exported=False)]),
        ]
    }
    assert CallResolver(index).resolve("a.ts", "hidden") is None


def test_a_namespace_import_gives_nothing_to_look_up_by():
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["helper"])],
                        bindings=[{"local": "ns", "imported": "*", "specifier": "./b"}]),
            file_record("b.ts", [symbol("helper", exported=True)]),
        ]
    }
    # Falls through to the unique-name rule rather than the namespace binding.
    assert CallResolver(index).resolve("a.ts", "helper") == "*"


def test_call_edges_name_the_file_that_disambiguates_them():
    index = {
        "files": [
            file_record("a.ts", [symbol("caller", ["handle"])],
                        bindings=[{"local": "handle", "imported": "handle", "specifier": "./b"}]),
            file_record("b.ts", [symbol("handle", exported=True)]),
            file_record("c.ts", [symbol("handle", exported=True)]),
        ]
    }
    assert list(iter_call_edges(index)) == [
        {"caller_id": "caller:1", "callee_file": "b.ts", "callee_name": "handle"}
    ]


def test_a_unique_name_edge_carries_no_file():
    index = {"files": [
        file_record("a.ts", [symbol("caller", ["only"])]),
        file_record("b.ts", [symbol("only")]),
    ]}
    assert list(iter_call_edges(index))[0]["callee_file"] is None

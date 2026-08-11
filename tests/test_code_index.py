import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.code_index import (
    batched,
    build_text,
    count_symbols,
    iter_file_rows,
    iter_rows,
    load_index,
    resolve_import,
)

INDEX = {
    "repo": "Rocket.Chat",
    "files": [
        {
            "path": "apps/meteor/app/crowd/server/crowd.ts",
            "name": "crowd.ts",
            "language": "typescript",
            "depth": 5,
            "symbols": [
                {
                    "id": "apps/meteor/app/crowd/server/crowd.ts:24:CrowdSync.syncDataToUser",
                    "name": "syncDataToUser",
                    "qualified_name": "CrowdSync.syncDataToUser",
                    "kind": "method",
                    "parent": "CrowdSync",
                    "start_line": 24,
                    "end_line": 33,
                    "signature": "async syncDataToUser(id: string)",
                    "code": "async syncDataToUser(id: string) { return id; }",
                    "truncated": False,
                }
            ],
        },
        {
            "path": "apps/meteor/app/reactions/server/setReaction.ts",
            "name": "setReaction.ts",
            "language": "typescript",
            "depth": 5,
            "symbols": [
                {
                    "id": "apps/meteor/app/reactions/server/setReaction.ts:3:setReaction",
                    "name": "setReaction",
                    "qualified_name": "setReaction",
                    "kind": "function",
                    "parent": None,
                    "start_line": 3,
                    "end_line": 9,
                    "signature": "setReaction = async (room: string) =>",
                    "code": "setReaction = async (room: string) => { return room; }",
                    "truncated": False,
                }
            ],
        },
    ],
}

# Every parameter the loader's Cypher statement reads off a row.
EXPECTED_ROW_KEYS = {
    "id",
    "name",
    "qualified_name",
    "kind",
    "parent",
    "start_line",
    "end_line",
    "signature",
    "file_path",
    "file_name",
    "language",
    "depth",
    "text",
}


def test_count_symbols():
    assert count_symbols(INDEX) == 2
    assert count_symbols({"files": []}) == 0


def test_iter_rows_flattens_every_symbol():
    rows = list(iter_rows(INDEX))
    assert len(rows) == 2
    assert {row["name"] for row in rows} == {"syncDataToUser", "setReaction"}


def test_rows_supply_every_field_the_import_query_needs():
    for row in iter_rows(INDEX):
        assert EXPECTED_ROW_KEYS <= set(row)


def test_rows_carry_their_own_file_path():
    """Code nodes are keyed per file, so CONTAINS cannot link the wrong file."""
    rows = {row["name"]: row for row in iter_rows(INDEX)}
    assert rows["syncDataToUser"]["file_path"].endswith("crowd.ts")
    assert rows["setReaction"]["file_path"].endswith("setReaction.ts")


def test_embedded_text_keeps_the_code_and_adds_context():
    row = next(iter_rows(INDEX))
    assert "async syncDataToUser(id: string) { return id; }" in row["text"]
    assert "apps/meteor/app/crowd/server/crowd.ts" in row["text"]
    assert "CrowdSync.syncDataToUser" in row["text"]


def test_build_text_is_stable_for_the_same_symbol():
    file_record = INDEX["files"][0]
    symbol = file_record["symbols"][0]
    assert build_text(file_record, symbol) == build_text(file_record, symbol)


def test_batched_splits_and_keeps_the_remainder():
    rows = [{"n": i} for i in range(5)]
    assert [len(b) for b in batched(rows, 2)] == [2, 2, 1]
    assert [len(b) for b in batched(rows, 10)] == [5]
    assert list(batched([], 3)) == []


def test_batched_rejects_a_zero_size():
    with pytest.raises(ValueError):
        list(batched([{"n": 1}], 0))


def test_load_index_roundtrip(tmp_path):
    path = tmp_path / "index.json"
    path.write_text(json.dumps(INDEX))
    assert count_symbols(load_index(path)) == 2


def test_load_index_rejects_unrelated_json(tmp_path):
    """Guards against pointing the loader at some other JSON file."""
    path = tmp_path / "index.json"
    path.write_text(json.dumps([{"file": "crowd.ts"}]))
    with pytest.raises(ValueError):
        load_index(path)


from genai_stack.code_index import iter_import_edges, resolve_import

KNOWN = {
    "src/crowd.ts",
    "src/util/text.ts",
    "src/util/index.ts",
    "src/legacy/sync.js",
}


def test_resolve_import_adds_the_extension():
    assert resolve_import("src/crowd.ts", "./util/text", KNOWN) == "src/util/text.ts"


def test_resolve_import_walks_up():
    assert resolve_import("src/util/text.ts", "../legacy/sync", KNOWN) == "src/legacy/sync.js"


def test_resolve_import_falls_back_to_index():
    assert resolve_import("src/crowd.ts", "./util", KNOWN) == "src/util/index.ts"


def test_resolve_import_ignores_external_packages():
    """A node_modules package is not part of the indexed repo."""
    assert resolve_import("src/crowd.ts", "meteor/accounts-base", KNOWN) is None
    assert resolve_import("src/crowd.ts", "react", KNOWN) is None


def test_resolve_import_returns_none_for_unindexed_files():
    assert resolve_import("src/crowd.ts", "./missing", KNOWN) is None


def test_iter_import_edges_deduplicates():
    index = {
        "files": [
            {
                "path": "src/crowd.ts",
                "name": "crowd.ts",
                "language": "typescript",
                "depth": 1,
                "symbols": [],
                "imports": ["./util/text", "./util/text.ts", "react", "./self"],
            },
            {
                "path": "src/util/text.ts",
                "name": "text.ts",
                "language": "typescript",
                "depth": 2,
                "symbols": [],
                "imports": [],
            },
        ]
    }
    assert list(iter_import_edges(index)) == [
        {"from_path": "src/crowd.ts", "to_path": "src/util/text.ts"}
    ]


def test_rows_carry_call_targets():
    index = {
        "files": [
            {
                "path": "a.ts",
                "name": "a.ts",
                "language": "typescript",
                "depth": 0,
                "imports": [],
                "symbols": [
                    {
                        "id": "a.ts:1:a",
                        "name": "a",
                        "qualified_name": "a",
                        "kind": "function",
                        "parent": None,
                        "start_line": 1,
                        "end_line": 2,
                        "signature": "function a()",
                        "code": "function a() { return b(); }",
                        "calls": ["b"],
                    }
                ],
            }
        ]
    }
    assert next(iter_rows(index))["calls"] == ["b"]


def test_rows_tolerate_an_index_without_calls():
    """Indexes generated before CALLS extraction must still import."""
    assert next(iter_rows(INDEX))["calls"] == []


def test_resolve_import_maps_js_specifier_to_ts_file():
    """TypeScript ESM imports name the emitted .js, not the .ts on disk."""
    known = {"source/utils/delay.ts", "source/core/Ky.ts"}
    assert resolve_import("source/core/Ky.ts", "../utils/delay.js", known) == "source/utils/delay.ts"


def test_resolve_import_maps_jsx_and_mjs_specifiers():
    known = {"src/View.tsx", "src/mod.mts", "src/a.ts"}
    assert resolve_import("src/a.ts", "./View.jsx", known) == "src/View.tsx"
    assert resolve_import("src/a.ts", "./mod.mjs", known) == "src/mod.mts"


def test_resolve_import_prefers_a_real_js_file_over_the_ts_mapping():
    """If the .js actually exists, it wins - the mapping is only a fallback."""
    known = {"src/util.js", "src/util.ts", "src/a.ts"}
    assert resolve_import("src/a.ts", "./util.js", known) == "src/util.js"


def test_a_declaration_file_is_resolved_when_the_import_omits_the_suffix():
    """A package shipping its types as `foo.d.ts` is imported as `./foo`."""
    known = {"src/common.d.ts"}
    assert resolve_import("src/index.ts", "./common", known) == "src/common.d.ts"


def test_an_implementation_wins_over_a_declaration():
    known = {"src/common.d.ts", "src/common.ts"}
    assert resolve_import("src/index.ts", "./common", known) == "src/common.ts"


def test_files_without_symbols_still_get_a_row():
    """Barrels and constant modules define nothing worth embedding, but they
    are the target of imports."""
    index = {
        "files": [
            {"path": "a.ts", "name": "a.ts", "language": "typescript", "depth": 0,
             "symbols": [], "imports": []},
        ]
    }
    assert [row["path"] for row in iter_file_rows(index)] == ["a.ts"]
    assert list(iter_rows(index)) == []


PACKAGES = {
    "@scope/utils": "packages/scope-utils",
    "@scope/utils-common": "packages/scope-utils-common",
}


def test_a_workspace_package_resolves_to_its_source_entry_point():
    known = {"packages/scope-utils/src/index.ts"}
    assert (
        resolve_import("app/main.ts", "@scope/utils", known, PACKAGES)
        == "packages/scope-utils/src/index.ts"
    )


def test_a_subpath_of_a_workspace_package_resolves():
    known = {"packages/scope-utils/src/paths.ts"}
    assert (
        resolve_import("app/main.ts", "@scope/utils/paths", known, PACKAGES)
        == "packages/scope-utils/src/paths.ts"
    )


def test_the_longest_package_name_wins():
    """`@scope/utils-common` must not be swallowed by `@scope/utils`."""
    known = {"packages/scope-utils-common/src/index.ts"}
    assert (
        resolve_import("app/main.ts", "@scope/utils-common", known, PACKAGES)
        == "packages/scope-utils-common/src/index.ts"
    )


def test_an_external_package_stays_unresolved():
    assert resolve_import("app/main.ts", "react", {"react.ts"}, PACKAGES) is None


def test_without_a_package_map_bare_specifiers_are_left_alone():
    known = {"packages/scope-utils/src/index.ts"}
    assert resolve_import("app/main.ts", "@scope/utils", known) is None


ALIASES = [{"prefix": "@/lib/", "targets": ["src/lib"]}, {"prefix": "@/", "targets": ["src"]}]


def test_a_tsconfig_alias_resolves():
    known = {"src/views/ChatView.tsx"}
    assert (
        resolve_import("src/App.tsx", "@/views/ChatView", known, None, ALIASES)
        == "src/views/ChatView.tsx"
    )


def test_the_longest_alias_prefix_wins():
    known = {"src/lib/api.ts"}
    assert (
        resolve_import("src/App.tsx", "@/lib/api", known, None, ALIASES) == "src/lib/api.ts"
    )


def test_an_unmatched_bare_specifier_is_still_external():
    assert resolve_import("src/App.tsx", "react", {"react.ts"}, None, ALIASES) is None

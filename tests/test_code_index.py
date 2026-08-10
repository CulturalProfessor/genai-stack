import json
import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from genai_stack.code_index import batched, build_text, count_symbols, iter_rows, load_index

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

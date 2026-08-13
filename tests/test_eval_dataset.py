"""The dataset's own rules, and a guard on the data actually shipped."""
import json
from pathlib import Path

import pytest

from eval.dataset import (
    DatasetError,
    Gold,
    check_against_index,
    load_dataset,
    normalise_path,
    parse_question,
)

ROOT = Path(__file__).resolve().parents[1]


def record(**overrides):
    base = {
        "id": "q1",
        "question": "where are sidebars generated?",
        "gold": [{"file": "src/generator.ts", "qualified_name": "generate"}],
        "category": "locate",
    }
    return {**base, **overrides}


def test_gold_keeps_the_file_and_the_name():
    question = parse_question(record(), "pr", "pr.jsonl:1")
    assert question.gold == (Gold("src/generator.ts", "generate"),)


def test_a_missing_qualified_name_means_the_whole_file():
    question = parse_question(
        record(gold=[{"file": "src/generator.ts"}]), "pr", "where"
    )
    assert question.gold[0].qualified_name is None


def test_an_abstain_question_may_not_carry_gold():
    with pytest.raises(DatasetError, match="no gold"):
        parse_question(record(category="abstain"), "pr", "where")


def test_a_question_with_no_gold_must_be_declared_abstain():
    """Otherwise an unanswerable question silently scores as a retrieval
    failure, and every change that retrieves more looks like an improvement."""
    with pytest.raises(DatasetError, match="category is 'abstain'"):
        parse_question(record(gold=[]), "pr", "where")


def test_unknown_categories_are_rejected():
    with pytest.raises(DatasetError, match="unknown category"):
        parse_question(record(category="vibes"), "pr", "where")


def test_a_missing_field_names_the_line_it_is_missing_from():
    with pytest.raises(DatasetError, match="pr.jsonl:7: missing 'question'"):
        parse_question({"id": "q", "gold": [], "category": "abstain"}, "pr", "pr.jsonl:7")


def test_paths_are_normalised():
    assert normalise_path("./src/a.ts") == "src/a.ts"
    assert normalise_path("src\\a.ts") == "src/a.ts"


def test_duplicate_ids_across_splits_are_refused(tmp_path):
    (tmp_path / "a.jsonl").write_text(json.dumps(record()) + "\n")
    (tmp_path / "b.jsonl").write_text(json.dumps(record()) + "\n")
    with pytest.raises(DatasetError, match="duplicate id"):
        load_dataset(tmp_path)


def test_splits_come_from_the_file_name(tmp_path):
    (tmp_path / "pr.jsonl").write_text(json.dumps(record()) + "\n")
    (tmp_path / "generated.jsonl").write_text(json.dumps(record(id="q2")) + "\n")
    loaded = load_dataset(tmp_path)
    assert {q.split for q in loaded} == {"pr", "generated"}
    # A model writing questions about code writes them in that code's own
    # words, so that split never reaches the headline number.
    assert [q.id for q in loaded if q.headline] == ["q1"]


INDEX = {
    "files": [
        {"path": "src/generator.ts", "symbols": [{"qualified_name": "generate"}]},
    ]
}


def test_gold_that_still_resolves_reports_nothing():
    questions = [parse_question(record(), "pr", "w")]
    assert check_against_index(questions, INDEX) == []


def test_a_moved_file_is_reported_rather_than_scored_as_a_miss():
    questions = [
        parse_question(record(gold=[{"file": "src/moved.ts"}]), "pr", "w")
    ]
    assert "no such file" in check_against_index(questions, INDEX)[0]


def test_a_renamed_symbol_is_reported():
    questions = [
        parse_question(
            record(gold=[{"file": "src/generator.ts", "qualified_name": "gone"}]),
            "pr",
            "w",
        )
    ]
    assert "has no symbol" in check_against_index(questions, INDEX)[0]


def test_the_shipped_dataset_loads():
    questions = load_dataset(ROOT / "eval" / "dataset")
    assert len(questions) > 20


def test_the_shipped_gold_resolves_against_an_index_if_one_is_present():
    """The dataset is data, and data rots. The index is not in the repository,
    so this can only run where someone has built one - `python -m eval.run
    --check` is the guard that always runs."""
    index_path = ROOT / "index" / "all_code_blocks.json"
    if not index_path.is_file():
        pytest.skip("no index built; run extract.py first")
    index = json.loads(index_path.read_text(encoding="utf-8"))
    questions = load_dataset(ROOT / "eval" / "dataset")
    assert check_against_index(questions, index) == []


def test_the_shipped_dataset_has_unanswerable_questions():
    questions = load_dataset(ROOT / "eval" / "dataset")
    assert sum(q.abstains for q in questions) >= 5

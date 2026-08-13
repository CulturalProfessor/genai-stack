"""Scoring rules. No Neo4j, no model, no snapshot on disk."""
from dataclasses import dataclass

from eval.dataset import Gold, Question
from eval.metrics import (
    Seed,
    aggregate,
    hits,
    mcnemar,
    peak_sharpness,
    score_question,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict


def question(gold, category="locate", ident="q1"):
    return Question(
        id=ident,
        question="where?",
        gold=tuple(gold),
        category=category,
        source="test",
        split="pr",
    )


def seed(file, symbol, mentions=(), named=(), score=0.5, chars=100):
    return Seed(
        file=file, symbol=symbol, score=score, chars=chars,
        mentions=tuple(mentions), named_files=tuple(named),
    )


def test_file_level_gold_is_satisfied_by_any_symbol_in_that_file():
    assert hits(Gold("a.ts", None), "a.ts", "anything")


def test_named_gold_needs_that_symbol():
    """Landing in the right file and the wrong function is not an answer."""
    assert hits(Gold("a.ts", "parse"), "a.ts", "parse")
    assert not hits(Gold("a.ts", "parse"), "a.ts", "format")


def test_the_same_name_in_another_file_is_not_a_match():
    assert not hits(Gold("a.ts", "parse"), "b.ts", "parse")


def test_a_seed_puts_gold_in_the_retrieved_tier():
    score = score_question(question([Gold("a.ts", "parse")]), [seed("a.ts", "parse")], 4)
    assert (score.retrieved, score.mentioned, score.covered) == (1.0, 1.0, True)


def test_a_neighbour_mentions_gold_without_retrieving_it():
    """The neighbour arrives as a signature line, not as code, so it counts in
    the weaker tier only."""
    score = score_question(
        question([Gold("a.ts", "parse")]),
        [seed("b.ts", "caller", mentions=[("a.ts", "parse")])],
        4,
    )
    assert score.retrieved == 0.0
    assert score.mentioned == 1.0


def test_the_imports_line_only_satisfies_file_level_gold():
    """It names a path and nothing inside it, so it cannot answer a question
    whose gold is a particular symbol."""
    file_level = score_question(
        question([Gold("a.ts", None)]), [seed("b.ts", "x", named=["a.ts"])], 4
    )
    assert (file_level.mentioned, file_level.named) == (0.0, 1.0)

    symbol_level = score_question(
        question([Gold("a.ts", "parse")]), [seed("b.ts", "x", named=["a.ts"])], 4
    )
    assert symbol_level.named == 0.0


def test_coverage_only_counts_documents_that_fit_the_context():
    """Retrieval goes deeper than the prompt does; a hit at rank 5 is not in
    the context when the app sends four."""
    seeds = [seed(f"pad{i}.ts", "x") for i in range(4)] + [seed("a.ts", "parse")]
    score = score_question(question([Gold("a.ts", "parse")]), seeds, 4)
    assert score.retrieved == 0.0
    assert score.recall[10] == 1.0
    assert score.recall[4] == 0.0


def test_partial_coverage_of_multi_file_gold():
    gold = [Gold("a.ts", None), Gold("b.ts", None), Gold("c.ts", None)]
    score = score_question(question(gold), [seed("a.ts", "x"), seed("b.ts", "y")], 4)
    assert round(score.retrieved, 3) == 0.667
    assert score.covered is False
    assert score.missed == ["c.ts::*"]


def test_reciprocal_rank_uses_the_first_gold_seed():
    seeds = [seed("x.ts", "x"), seed("a.ts", "parse")]
    assert score_question(question([Gold("a.ts", "parse")]), seeds, 4).rr == 0.5


def test_an_abstain_question_scores_no_coverage_but_keeps_its_top_score():
    score = score_question(
        question([], category="abstain"), [seed("a.ts", "x", score=0.9)], 4
    )
    assert score.abstains and score.covered and score.retrieved == 0.0
    assert score.top_score == 0.9


def test_coverage_averages_over_questions_not_over_gold_entries():
    """One sprawling impact question should not outvote five small ones."""
    wide = score_question(
        question([Gold(f"{i}.ts", None) for i in range(10)], ident="wide"),
        [seed("0.ts", "x")],
        4,
    )
    narrow = score_question(
        question([Gold("a.ts", None)], ident="narrow"), [seed("a.ts", "x")], 4
    )
    assert aggregate([wide, narrow])["coverage_retrieved"] == 0.55


def test_the_top_score_carries_no_information():
    """Hybrid search divides by the best score in the same query, so the top
    hit is 1.0 on every question. Only the shape behind it can separate them."""
    assert peak_sharpness([1.0, 1.0, 1.0]) == 0.0
    assert peak_sharpness([1.0, 0.5, 0.5]) == 0.5


def test_one_result_alone_has_no_shape_to_measure():
    assert peak_sharpness([1.0]) == 0.0


def test_the_abstain_gap_compares_how_peaked_the_search_was():
    """A question the repo can answer has one standout hit and then a drop.
    One it cannot returns a flat run of equally mediocre matches."""
    answerable = score_question(
        question([Gold("a.ts", None)]),
        [seed("a.ts", "x", score=1.0), seed("b.ts", "y", score=0.5)],
        4,
    )
    unanswerable = score_question(
        question([], category="abstain", ident="q2"),
        [seed("y.ts", "x", score=1.0), seed("z.ts", "y", score=1.0)],
        4,
    )
    assert aggregate([answerable, unanswerable])["abstain_gap"] == 0.5


def test_context_tokens_are_estimated_from_characters():
    score = score_question(
        question([Gold("a.ts", None)]), [seed("a.ts", "x", chars=400)], 4
    )
    assert aggregate([score])["context_tokens_est"] == 100


def test_a_document_flattens_to_a_seed():
    document = FakeDocument(
        page_content="code",
        metadata={
            "source": "./a.ts",
            "symbol": "parse",
            "score": 0.7,
            "calls": [{"file": "b.ts", "symbol": "helper"}],
            "called_by": [{"file": "c.ts", "symbol": "main"}],
            "imports": ["d.ts"],
        },
    )
    flattened = Seed.from_document(document)
    assert flattened.file == "a.ts"
    assert flattened.mentions == (("b.ts", "helper"), ("c.ts", "main"))
    assert flattened.named_files == ("d.ts",)


def test_a_snapshot_taken_before_neighbours_carried_files_still_loads():
    """Those mentions can only match file-agnostic gold, but an old run is
    still worth diffing against."""
    document = FakeDocument(
        page_content="code",
        metadata={"source": "a.ts", "symbol": "parse", "calls": ["helper"]},
    )
    assert Seed.from_document(document).mentions == (("", "helper"),)


def test_a_seed_survives_a_round_trip_through_a_snapshot():
    original = seed("a.ts", "parse", mentions=[("b.ts", "helper")], named=["c.ts"])
    assert Seed.from_row(original.to_row()) == original


def test_mcnemar_ignores_the_questions_that_did_not_move():
    before = {"a": True, "b": False, "c": True}
    after = {"a": True, "b": True, "c": True}
    result = mcnemar(before, after)
    assert (result["c"], result["b"], result["gained"]) == (1, 0, ["b"])


def test_an_even_split_of_gains_and_losses_is_not_evidence():
    result = mcnemar({"a": True, "b": False}, {"a": False, "b": True})
    assert result["p"] == 1.0


def test_five_gains_and_no_losses_clears_the_usual_threshold():
    before = {f"q{i}": False for i in range(5)}
    after = {f"q{i}": True for i in range(5)}
    assert round(mcnemar(before, after)["p"], 4) == 0.0625


def test_two_runs_that_agree_everywhere_report_no_difference():
    same = {"a": True, "b": False}
    assert mcnemar(same, same)["p"] == 1.0

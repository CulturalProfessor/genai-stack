"""Scoring, with no Neo4j and no model anywhere near it.

Everything here runs over the rows a run records, so a snapshot can be
re-scored months later without re-retrieving, and a scoring change can be
applied to old runs instead of invalidating them.

Three tiers, because what reaches the LLM is not one thing:

  retrieved  the gold symbol's full text is in the context
  mentioned  it appears as a neighbour line: name, location and signature
  named      its *file* is listed on the `This file imports:` line, nothing more

The headline is `retrieved` and `mentioned`. `named` is reported apart and
never folded in: a comma-separated list of paths is not context, and letting it
count would score a context made entirely of file names as a win. It is broken
out rather than dropped because it is the only measurable thing the IMPORTS
traversal does, and an eval should be able to say whether that earns its place.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

from eval.dataset import Gold, Question, normalise_path

# Seed depths to report. 4 is what the app uses (RETRIEVER_K in chains.py); the
# deeper cuts say whether a reranker would have anything to work with.
RECALL_DEPTHS = (4, 10, 20)

# Characters per token. The app deliberately has no tokenizer - counting real
# tokens would pull a GPT-2 vocabulary in for the Ollama models - so the
# context guardrail is an estimate, and is only ever compared against itself.
CHARS_PER_TOKEN = 4


@dataclass
class Seed:
    """One retrieved document, flattened to what scoring needs."""

    file: str
    symbol: str | None
    score: float = 0.0
    chars: int = 0
    mentions: tuple[tuple[str, str | None], ...] = ()
    named_files: tuple[str, ...] = ()

    @classmethod
    def from_document(cls, document) -> "Seed":
        """Flatten a LangChain Document returned by `retrieve`."""
        meta = document.metadata
        mentions = []
        for key in ("calls", "called_by", "uses_types", "used_as_type_by"):
            for entry in meta.get(key) or ():
                # Older runs recorded bare names. Those carry no file, so they
                # can only ever match file-agnostic gold; they are not dropped
                # because a snapshot from before the metadata change is still
                # worth diffing against.
                if isinstance(entry, str):
                    mentions.append(("", entry))
                else:
                    mentions.append(
                        (normalise_path(entry.get("file") or ""), entry.get("symbol"))
                    )
        return cls(
            file=normalise_path(meta.get("source") or ""),
            symbol=meta.get("symbol"),
            score=float(meta.get("score") or 0.0),
            chars=len(document.page_content or ""),
            mentions=tuple(mentions),
            named_files=tuple(
                normalise_path(path) for path in (meta.get("imports") or ())
            ),
        )

    def to_row(self) -> dict:
        return {
            "file": self.file,
            "symbol": self.symbol,
            "score": self.score,
            "chars": self.chars,
            "mentions": [list(pair) for pair in self.mentions],
            "named_files": list(self.named_files),
        }

    @classmethod
    def from_row(cls, row: dict) -> "Seed":
        return cls(
            file=row["file"],
            symbol=row.get("symbol"),
            score=float(row.get("score") or 0.0),
            chars=int(row.get("chars") or 0),
            mentions=tuple((m[0], m[1]) for m in row.get("mentions") or ()),
            named_files=tuple(row.get("named_files") or ()),
        )


def hits(gold: Gold, file: str, symbol: str | None) -> bool:
    """Does `(file, symbol)` satisfy this gold entry?

    A null `qualified_name` means the whole file is the answer, which is what
    a mined pull request knows. A named symbol must match exactly: a question
    is not answered by landing in the right file and the wrong function.
    """
    if gold.file != file:
        return False
    if gold.qualified_name is None:
        return True
    return symbol == gold.qualified_name


@dataclass
class QuestionScore:
    id: str
    category: str
    split: str
    abstains: bool
    gold: int = 0
    retrieved: float = 0.0
    mentioned: float = 0.0
    named: float = 0.0
    covered: bool = False
    recall: dict[int, float] = field(default_factory=dict)
    rr: float = 0.0
    context_chars: int = 0
    top_score: float = 0.0
    peak: float = 0.0
    found: list[str] = field(default_factory=list)
    missed: list[str] = field(default_factory=list)


def score_question(
    question: Question, seeds: list[Seed], context_k: int
) -> QuestionScore:
    """Score one question against the documents its run retrieved.

    `context_k` is how many of those documents the app would actually put in a
    prompt. The run retrieves deeper so recall at greater depths can be
    reported from the same pass: the traversal is evaluated per node, so the
    first four rows of a twenty-row retrieval are exactly what a four-row
    retrieval returns.
    """
    score = QuestionScore(
        id=question.id,
        category=question.category,
        split=question.split,
        abstains=question.abstains,
        gold=len(question.gold),
        top_score=seeds[0].score if seeds else 0.0,
    )
    context = seeds[:context_k]
    score.context_chars = sum(seed.chars for seed in context)
    score.peak = peak_sharpness([seed.score for seed in context])

    if question.abstains:
        # Nothing to cover. The only signal before the LLM is how confident
        # the search was, which `abstain_gap` compares across questions.
        score.covered = True
        return score

    retrieved = mentioned = named = 0
    for gold in question.gold:
        in_text = any(hits(gold, seed.file, seed.symbol) for seed in context)
        in_mention = in_text or any(
            hits(gold, file, symbol)
            for seed in context
            for file, symbol in seed.mentions
        )
        # A path on the imports line names a file and nothing inside it, so it
        # can only satisfy gold that is file-level to begin with.
        in_named = in_mention or (
            gold.qualified_name is None
            and any(gold.file in seed.named_files for seed in context)
        )
        retrieved += in_text
        mentioned += in_mention
        named += in_named
        (score.found if in_mention else score.missed).append(str(gold))

    total = len(question.gold)
    score.retrieved = retrieved / total
    score.mentioned = mentioned / total
    score.named = named / total
    score.covered = mentioned == total

    for depth in RECALL_DEPTHS:
        found = sum(
            any(hits(gold, seed.file, seed.symbol) for seed in seeds[:depth])
            for gold in question.gold
        )
        score.recall[depth] = found / total

    for rank, seed in enumerate(seeds, 1):
        if any(hits(gold, seed.file, seed.symbol) for gold in question.gold):
            score.rr = 1 / rank
            break
    return score


def peak_sharpness(scores: list[float]) -> float:
    """How far the best hit stands above the ones behind it, from 0 to 1.

    The absolute score cannot be used for this. Hybrid search divides every
    score by the best score in that same query, so the top hit is 1.0 on every
    question ever asked, answerable or not, and a threshold on it separates
    nothing. What survives normalisation is the shape: a question the corpus
    can answer tends to have one standout hit and then a drop, while a question
    it cannot answer returns a flat run of equally mediocre matches.
    """
    if len(scores) < 2 or scores[0] <= 0:
        return 0.0
    tail = scores[1:]
    return 1 - (sum(tail) / len(tail)) / scores[0]


def _mean(values) -> float:
    values = list(values)
    return sum(values) / len(values) if values else 0.0


def aggregate(scores: list[QuestionScore]) -> dict:
    """Macro averages. Every question weighs the same regardless of how many
    gold entries it carries, so one sprawling impact question cannot swing the
    headline on its own."""
    answerable = [s for s in scores if not s.abstains]
    abstaining = [s for s in scores if s.abstains]

    summary = {
        "questions": len(scores),
        "answerable": len(answerable),
        "abstain": len(abstaining),
        "coverage_retrieved": _mean(s.retrieved for s in answerable),
        "coverage_mentioned": _mean(s.mentioned for s in answerable),
        "coverage_named": _mean(s.named for s in answerable),
        "fully_covered": _mean(s.covered for s in answerable),
        "mrr": _mean(s.rr for s in answerable),
        "context_chars": _mean(s.context_chars for s in scores),
        "context_tokens_est": _mean(s.context_chars for s in scores) / CHARS_PER_TOKEN,
    }
    for depth in RECALL_DEPTHS:
        summary[f"recall@{depth}"] = _mean(s.recall.get(depth, 0.0) for s in answerable)

    # If the search is no more peaked on an answerable question than on one the
    # repo cannot answer, no threshold downstream can make the system decline.
    if abstaining and answerable:
        summary["abstain_peak_answerable"] = _mean(s.peak for s in answerable)
        summary["abstain_peak_unanswerable"] = _mean(s.peak for s in abstaining)
        summary["abstain_gap"] = (
            summary["abstain_peak_answerable"] - summary["abstain_peak_unanswerable"]
        )
    return summary


def by_category(scores: list[QuestionScore]) -> dict[str, dict]:
    categories: dict[str, list[QuestionScore]] = {}
    for score in scores:
        categories.setdefault(score.category, []).append(score)
    summary = {}
    for category, group in sorted(categories.items()):
        answerable = [s for s in group if not s.abstains]
        summary[category] = {
            "n": len(group),
            # None, not zero: an abstain question has no gold to cover, and a
            # column of 0.0% there reads as a failure rather than as N/A.
            "coverage_mentioned": _mean(s.mentioned for s in answerable)
            if answerable
            else None,
            "fully_covered": _mean(s.covered for s in group),
        }
    return summary


def mcnemar(before: dict[str, bool], after: dict[str, bool]) -> dict:
    """Exact paired test on the questions both runs asked.

    Two runs over the same questions are not independent samples, so comparing
    their proportions with a two-sample test overstates the evidence. What
    carries information is the disagreements: questions that flipped.
    """
    shared = sorted(set(before) & set(after))
    gained = [q for q in shared if after[q] and not before[q]]
    lost = [q for q in shared if before[q] and not after[q]]
    b, c = len(lost), len(gained)
    n = b + c
    if n == 0:
        p = 1.0
    else:
        tail = sum(math.comb(n, i) for i in range(min(b, c) + 1))
        p = min(1.0, 2 * tail / (2**n))
    return {"gained": gained, "lost": lost, "b": b, "c": c, "p": p, "compared": len(shared)}

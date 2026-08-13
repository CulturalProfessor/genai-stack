"""The question set, and the rules a question has to obey to be in it.

Gold is `(file, qualified_name)` and never a symbol id: an id embeds the line
the symbol starts on, so any edit to the file above it silently invalidates
gold that is still perfectly correct.

`qualified_name` may be null, meaning "any symbol in this file counts". That is
the shape mined pull requests produce - a PR names the files it changed, not
the functions - and refusing it would throw away the only source of questions
written by someone who had not read the code.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

# Categories exist so a result is attributable to a mechanism rather than to
# "retrieval". `explain` should sit near 100% and is a canary: if it drops,
# something is broken upstream of any ranking change. `abstain` carries no gold
# on purpose - without it, every change that retrieves more scores better and
# the system is optimised toward never saying "I don't know".
CATEGORIES = {"locate", "explain", "trace", "impact", "config", "abstain"}

# Splits kept out of the headline number. `generated` is circular, since a model
# writing questions about code writes them in that code's own vocabulary;
# `heldout` is deliberately consulted rarely, and averaging it in would make it
# just another part of the training signal.
NON_HEADLINE_SPLITS = {"generated", "heldout"}


class DatasetError(ValueError):
    """A malformed question. Loading is all-or-nothing: a typo in one record
    would otherwise quietly shrink the set and move every aggregate."""


@dataclass(frozen=True)
class Gold:
    file: str
    qualified_name: str | None = None

    def __str__(self) -> str:
        return f"{self.file}::{self.qualified_name or '*'}"


@dataclass(frozen=True)
class Question:
    id: str
    question: str
    gold: tuple[Gold, ...]
    category: str
    source: str
    split: str
    sha: str | None = None
    note: str | None = None

    @property
    def abstains(self) -> bool:
        return not self.gold

    @property
    def headline(self) -> bool:
        return self.split not in NON_HEADLINE_SPLITS


def normalise_path(path: str) -> str:
    """Index paths are repo-relative and posix. Gold written by hand is not
    always either."""
    cleaned = path.strip().replace("\\", "/").lstrip("./")
    return cleaned.rstrip("/")


def _gold_from(raw, where: str) -> tuple[Gold, ...]:
    if not isinstance(raw, list):
        raise DatasetError(f"{where}: 'gold' must be a list")
    items = []
    for entry in raw:
        if not isinstance(entry, dict) or "file" not in entry:
            raise DatasetError(f"{where}: every gold entry needs a 'file'")
        name = entry.get("qualified_name")
        items.append(Gold(normalise_path(entry["file"]), name or None))
    return tuple(items)


def parse_question(raw: dict, split: str, where: str) -> Question:
    for field in ("id", "question", "gold", "category"):
        if field not in raw:
            raise DatasetError(f"{where}: missing '{field}'")
    if raw["category"] not in CATEGORIES:
        raise DatasetError(
            f"{where}: unknown category {raw['category']!r}, "
            f"expected one of {sorted(CATEGORIES)}"
        )
    gold = _gold_from(raw["gold"], where)
    if raw["category"] == "abstain" and gold:
        raise DatasetError(f"{where}: an abstain question must carry no gold")
    if raw["category"] != "abstain" and not gold:
        raise DatasetError(
            f"{where}: no gold. If the repo cannot answer it, category is 'abstain'"
        )
    return Question(
        id=str(raw["id"]),
        question=str(raw["question"]),
        gold=gold,
        category=raw["category"],
        source=str(raw.get("source", split)),
        split=split,
        sha=raw.get("sha"),
        note=raw.get("note"),
    )


def load_split(path: Path) -> list[Question]:
    """One JSONL file. The stem names the split."""
    split = path.stem
    questions = []
    for number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        line = line.strip()
        if not line or line.startswith("//"):
            continue
        where = f"{path.name}:{number}"
        try:
            raw = json.loads(line)
        except json.JSONDecodeError as error:
            raise DatasetError(f"{where}: {error}") from error
        questions.append(parse_question(raw, split, where))
    return questions


def load_dataset(directory: Path, splits: list[str] | None = None) -> list[Question]:
    """Every `*.jsonl` under `directory`, or just the named splits."""
    directory = Path(directory)
    if not directory.is_dir():
        raise DatasetError(f"No such dataset directory: {directory}")

    questions: list[Question] = []
    for path in sorted(directory.glob("*.jsonl")):
        if splits and path.stem not in splits:
            continue
        questions.extend(load_split(path))

    seen: dict[str, str] = {}
    for question in questions:
        if question.id in seen:
            raise DatasetError(
                f"duplicate id {question.id!r} in {seen[question.id]} and {question.split}"
            )
        seen[question.id] = question.split
    if splits:
        missing = set(splits) - {q.split for q in questions}
        if missing:
            raise DatasetError(f"no such split(s): {sorted(missing)}")
    return questions


def index_symbols(index: dict) -> dict[str, set[str]]:
    """`{file path: {qualified names}}`, for checking gold against a corpus."""
    return {
        record["path"]: {s["qualified_name"] for s in record.get("symbols", [])}
        for record in index.get("files", [])
    }


def check_against_index(questions: list[Question], index: dict) -> list[str]:
    """Gold that no longer names anything in the indexed corpus.

    Gold rots: files move, symbols get renamed, and a question that scores zero
    because its gold moved looks exactly like a retrieval regression. Run this
    after every re-index.
    """
    known = index_symbols(index)
    problems = []
    for question in questions:
        for gold in question.gold:
            if gold.file not in known:
                problems.append(f"{question.id}: no such file in the index: {gold.file}")
            elif gold.qualified_name and gold.qualified_name not in known[gold.file]:
                problems.append(
                    f"{question.id}: {gold.file} has no symbol "
                    f"{gold.qualified_name!r} in the index"
                )
    return problems

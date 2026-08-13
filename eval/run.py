"""The retrieval eval: ask the question set, score what came back, snapshot it.

    python -m eval.run --config baseline          # run, print, snapshot
    python -m eval.run --config no-expand         # the control arm
    python -m eval.run --diff eval/snapshots/no-expand.json eval/snapshots/baseline.json
    python -m eval.run --rescore eval/snapshots/baseline.json
    python -m eval.run --check --index index/all_code_blocks.json

Only the first two need Neo4j and an embedding model. Scoring, diffing and gold
checking run on a snapshot file with nothing installed, which is what keeps the
`tests/` suite free of the runtime stack.

There is no LLM judge here on purpose. Every change on the roadmap lands in
seeding or in context assembly, and both are fully measurable without one; a
judge would price every run, add variance on top of the variance already in a
fifty-question set, and hand us a second prompt to tune.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import logging
import statistics
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dataset import Question, check_against_index, load_dataset
from eval.metrics import (
    RECALL_DEPTHS,
    Seed,
    aggregate,
    by_category,
    mcnemar,
    score_question,
)

DATASET_DIR = Path("eval/dataset")
SNAPSHOT_DIR = Path("eval/snapshots")


@dataclass(frozen=True)
class Config:
    name: str
    expand: bool
    # Retrieve deeper than the app does so recall at depth comes out of the
    # same pass; only the first `context_k` rows are scored as context.
    k: int = max(RECALL_DEPTHS)
    context_k: int = 4
    note: str = ""


CONFIGS = {
    config.name: config
    for config in (
        Config("baseline", expand=True, note="hybrid search + graph expansion"),
        Config(
            "no-expand",
            expand=False,
            note="the same search with the traversal removed",
        ),
    )
}


def fingerprint(questions: list[Question]) -> str:
    """Identifies the question set, so a diff can refuse to compare a run
    against one asked a different set of questions."""
    digest = hashlib.sha256()
    for question in sorted(questions, key=lambda q: q.id):
        digest.update(question.id.encode())
        digest.update(question.question.encode())
        for gold in question.gold:
            digest.update(str(gold).encode())
    return digest.hexdigest()[:12]


def run_config(config: Config, questions: list[Question]) -> dict:
    """Retrieve for every question. The only part that needs the stack."""
    # The retrieval query names EXTENDS and IMPLEMENTS, and a repository with no
    # class heritage has neither, so Neo4j returns a notification per query and
    # the driver logs the entire statement with it. Three of those per question
    # buries the report under its own query text.
    logging.getLogger("neo4j.notifications").setLevel(logging.ERROR)

    try:
        from genai_stack.retrieval import retrieve
    except ImportError as error:
        raise SystemExit(
            f"This needs the runtime stack, which is not installed: {error}\n"
            "requirements-dev.txt covers the tests only, by design. Install "
            "requirements.txt and\nstart Neo4j, or work from a snapshot with "
            "--rescore / --diff / --check."
        ) from error

    rows: dict[str, dict] = {}
    for number, question in enumerate(questions, 1):
        started = time.perf_counter()
        documents = retrieve(question.question, k=config.k, expand=config.expand)
        elapsed = (time.perf_counter() - started) * 1000
        rows[question.id] = {
            "seeds": [Seed.from_document(d).to_row() for d in documents],
            "latency_ms": round(elapsed, 1),
        }
        print(
            f"  [{number}/{len(questions)}] {question.id} "
            f"{len(documents)} docs in {elapsed:.0f}ms",
            file=sys.stderr,
        )
    return rows


def score_snapshot(snapshot: dict, questions: list[Question]) -> list:
    """Score recorded rows. Questions with no row are skipped rather than
    scored as zero: a run that crashed halfway should not read as a
    catastrophic regression."""
    by_id = {q.id: q for q in questions}
    scores = []
    for question_id, row in snapshot["questions"].items():
        question = by_id.get(question_id)
        if question is None:
            continue
        seeds = [Seed.from_row(s) for s in row["seeds"]]
        scores.append(score_question(question, seeds, snapshot["context_k"]))
    return scores


def _line(label: str, value, width: int = 34) -> str:
    if isinstance(value, float):
        return f"  {label:<{width}} {value:>8.3f}"
    return f"  {label:<{width}} {value:>8}"


def _percent(label: str, value: float, width: int = 34) -> str:
    return f"  {label:<{width}} {value:>7.1%}"


def out_of_order(snapshot: dict) -> list[str]:
    """Questions whose documents did not come back in descending score order.

    Retrieval is supposed to hand back its best hit first, and the app puts the
    first four in the prompt. A retrieval query that re-sorts the stream - a
    bare ORDER BY will - silently replaces the ranking with something else,
    and every coverage number measured at a context cut becomes meaningless.
    """
    broken = []
    for question_id, row in snapshot["questions"].items():
        scores = [seed["score"] for seed in row["seeds"]]
        if any(b > a + 1e-9 for a, b in zip(scores, scores[1:])):
            broken.append(question_id)
    return broken


def format_report(snapshot: dict, scores: list) -> str:
    summary = aggregate(scores)
    unordered = out_of_order(snapshot)
    latencies = [row["latency_ms"] for row in snapshot["questions"].values()]
    out = [
        f"retrieval eval ({snapshot['config']}) - {snapshot.get('note', '')}".rstrip(" -"),
        f"  {snapshot['questions_asked']} questions, "
        f"splits {', '.join(snapshot['splits']) or 'all'}, "
        f"k={snapshot['k']}, context_k={snapshot['context_k']}",
        "",
        "context coverage",
        _percent("retrieved (full text in context)", summary["coverage_retrieved"]),
        _percent("+ mentioned (signature only)", summary["coverage_mentioned"]),
        _percent("+ named on the imports line", summary["coverage_named"]),
        _percent("questions fully covered", summary["fully_covered"]),
        "",
        "seeding",
    ]
    for depth in RECALL_DEPTHS:
        out.append(_percent(f"recall@{depth}", summary[f"recall@{depth}"]))
    out.append(_line("MRR", summary["mrr"]))
    if "abstain_gap" in summary:
        out += ["", "abstention"]
        out.append(_line("peak sharpness, answerable", summary["abstain_peak_answerable"]))
        out.append(
            _line("peak sharpness, unanswerable", summary["abstain_peak_unanswerable"])
        )
        out.append(_line("  gap", summary["abstain_gap"]))

    out += ["", "guardrails"]
    out.append(_line("context chars (mean)", round(summary["context_chars"])))
    out.append(_line("context tokens (est, mean)", round(summary["context_tokens_est"])))
    if latencies:
        out.append(_line("retrieval ms (median)", round(statistics.median(latencies))))
        out.append(_line("retrieval ms (max)", round(max(latencies))))

    out += ["", "by category"]
    for category, stats in by_category(scores).items():
        covered = stats["coverage_mentioned"]
        out.append(
            f"  {category:<12} n={stats['n']:<4} "
            f"covered {'     -' if covered is None else format(covered, '>6.1%')}  "
            f"fully {stats['fully_covered']:>6.1%}"
        )

    if unordered:
        out += [
            "",
            f"  WARNING: {len(unordered)}/{len(snapshot['questions'])} questions came "
            "back out of score order.",
            "  The retrieval query is re-sorting the result stream, so the context "
            "cut above is",
            "  not the top-scoring documents and every coverage number here is "
            "measuring the",
            "  wrong four rows.",
        ]

    worst = sorted(
        (s for s in scores if not s.abstains and s.missed), key=lambda s: s.mentioned
    )
    if worst:
        out += ["", "missing gold (worst first)"]
        for score in worst[:10]:
            out.append(f"  {score.id:<16} {', '.join(score.missed)[:90]}")
    return "\n".join(out)


def format_diff(before: dict, after: dict, questions: list[Question]) -> str:
    before_scores = score_snapshot(before, questions)
    after_scores = score_snapshot(after, questions)
    left, right = aggregate(before_scores), aggregate(after_scores)

    out = [f"{before['config']}  ->  {after['config']}", ""]
    if before["fingerprint"] != after["fingerprint"]:
        out.append(
            "  WARNING: different question sets. The numbers below are not comparable."
        )
        out.append("")
    header = f"  {'metric':<34} {'before':>9} {'after':>9} {'delta':>9}"
    out += [header, "  " + "-" * (len(header) - 2)]
    keys = [
        "coverage_retrieved",
        "coverage_mentioned",
        "coverage_named",
        "fully_covered",
        *[f"recall@{d}" for d in RECALL_DEPTHS],
        "mrr",
        "context_tokens_est",
    ]
    for key in keys:
        a, b = left.get(key, 0.0), right.get(key, 0.0)
        if key == "context_tokens_est":
            out.append(f"  {key:<34} {a:>9.0f} {b:>9.0f} {b - a:>+9.0f}")
        else:
            out.append(f"  {key:<34} {a:>8.1%} {b:>8.1%} {(b - a) * 100:>+8.1f}pp")

    # `coverage_mentioned` on the expand run minus the no-expand run is the
    # graph lift. Named separately here only when the pair is that comparison.
    if not before.get("expand", True) and after.get("expand", True):
        lift = right["coverage_mentioned"] - left["coverage_mentioned"]
        out += ["", f"  graph lift (coverage with expansion - without): {lift:+.1%}"]
        # The two arms run the same hybrid search and differ only in what the
        # retrieval query does with each hit, so seeding cannot move between
        # them. If it has, something other than the traversal changed and the
        # lift above is measuring that instead.
        if abs(right["recall@4"] - left["recall@4"]) > 1e-9:
            out.append(
                "  WARNING: recall@4 differs between the arms. They are not the "
                "same search,\n  so this is not a clean lift measurement."
            )

    test = mcnemar(
        {s.id: s.covered for s in before_scores},
        {s.id: s.covered for s in after_scores},
    )
    out += [
        "",
        f"  paired over {test['compared']} questions: "
        f"{test['c']} gained, {test['b']} lost, exact McNemar p = {test['p']:.3f}",
    ]
    if test["p"] > 0.05:
        out.append(
            "  Not distinguishable from noise at this sample size. The per-question"
        )
        out.append("  list below is the part worth reading.")
    if test["gained"]:
        out.append(f"  gained: {', '.join(test['gained'])}")
    if test["lost"]:
        out.append(f"  lost:   {', '.join(test['lost'])}")
    return "\n".join(out)


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", choices=sorted(CONFIGS), help="Which arm to run")
    parser.add_argument("--dataset", type=Path, default=DATASET_DIR)
    parser.add_argument(
        "--split",
        action="append",
        default=[],
        help="Only these splits (repeatable). Default: every split in the directory",
    )
    parser.add_argument("--headline", action="store_true", help="Drop non-headline splits")
    parser.add_argument("--limit", type=int, help="Ask only the first N questions")
    parser.add_argument("--out", type=Path, help="Where to write the snapshot")
    parser.add_argument("--rescore", type=Path, help="Re-score a snapshot, no retrieval")
    parser.add_argument("--diff", nargs=2, type=Path, metavar=("BEFORE", "AFTER"))
    parser.add_argument("--check", action="store_true", help="Validate gold and exit")
    parser.add_argument("--index", type=Path, default=Path("index/all_code_blocks.json"))
    parser.add_argument("--json", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    questions = load_dataset(args.dataset, args.split or None)
    if args.headline:
        questions = [q for q in questions if q.headline]
    if not questions:
        print("The dataset is empty. See eval/dataset/README.md.", file=sys.stderr)
        return 1

    if args.check:
        if not args.index.is_file():
            print(f"No such index: {args.index}", file=sys.stderr)
            return 1
        index = json.loads(args.index.read_text(encoding="utf-8"))
        problems = check_against_index(questions, index)
        splits: dict[str, int] = {}
        for question in questions:
            splits[question.split] = splits.get(question.split, 0) + 1
        print(f"{len(questions)} questions, fingerprint {fingerprint(questions)}")
        commit = index.get("commit")
        print(f"  index built from   {commit or 'an unrecorded commit'}")
        if not commit:
            print(
                "  Re-index to record one. Without it, gold that rotted and gold "
                "that regressed\n  look the same."
            )
        for split, count in sorted(splits.items()):
            print(f"  {split:<12} {count}")
        if problems:
            print(f"\n{len(problems)} gold problem(s) against {args.index}:")
            for problem in problems:
                print(f"  {problem}")
            return 1
        print(f"\nAll gold resolves against {args.index}.")
        return 0

    if args.diff:
        before, after = (
            json.loads(path.read_text(encoding="utf-8")) for path in args.diff
        )
        print(format_diff(before, after, questions))
        return 0

    if args.rescore:
        snapshot = json.loads(args.rescore.read_text(encoding="utf-8"))
        scores = score_snapshot(snapshot, questions)
        if args.json:
            print(json.dumps(aggregate(scores), indent=2))
        else:
            print(format_report(snapshot, scores))
        return 0

    if not args.config:
        print("Nothing to do: pass --config, --diff, --rescore or --check.", file=sys.stderr)
        return 1

    config = CONFIGS[args.config]
    if args.limit:
        questions = questions[: args.limit]

    print(f"{config.name}: {len(questions)} questions", file=sys.stderr)
    snapshot = {
        "config": config.name,
        "note": config.note,
        "expand": config.expand,
        "k": config.k,
        "context_k": config.context_k,
        "dataset": str(args.dataset),
        "splits": sorted({q.split for q in questions}),
        "questions_asked": len(questions),
        "fingerprint": fingerprint(questions),
        "created_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "questions": run_config(config, questions),
    }
    scores = score_snapshot(snapshot, questions)
    snapshot["summary"] = aggregate(scores)

    # A truncated run gets its own file. Writing it to the config's usual name
    # would let a three-question smoke test quietly replace the full run that
    # everything else diffs against.
    suffix = f"-first{args.limit}" if args.limit else ""
    out = args.out or SNAPSHOT_DIR / f"{config.name}{suffix}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(snapshot, indent=1), encoding="utf-8")

    if args.json:
        print(json.dumps(snapshot["summary"], indent=2))
    else:
        print(format_report(snapshot, scores))
    print(f"\nsnapshot: {out}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())

"""Turn merged pull requests into candidate questions.

Pull requests are the best gold this project can get. The title was written by
someone describing a problem, usually before they had read the code that fixes
it, so the question does not share the vocabulary of its answer the way a
docstring or a test name does. The files the PR changed are the answer.

Output is candidates, not a dataset. Every line still needs a human to read it,
assign a category, reject the ones whose title says nothing ("fix typo",
"v3.1.0"), and tighten file-level gold to a symbol where the diff makes that
obvious. Write the survivors to `eval/dataset/pr.jsonl`.

    python -m eval.mine_prs --limit 30 --out /tmp/candidates.jsonl

Unauthenticated GitHub allows 60 requests an hour and this spends one per pull
request, so set GITHUB_TOKEN for anything larger.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from eval.dataset import normalise_path

API = "https://api.github.com"
SOURCE_SUFFIXES = (".ts", ".tsx", ".js", ".jsx", ".mjs", ".cjs")

# A title that is only a version bump, a dependency update or a typo fix is not
# a question about the code, whatever it touched.
NOISE = (
    "bump", "chore(deps", "dependabot", "typo", "prettier", "lint",
    "release", "changelog", "translat", "revert", "v3.", "v2.",
)


def _get(path: str, token: str | None) -> object:
    request = urllib.request.Request(
        path if path.startswith("http") else f"{API}{path}",
        headers={
            "Accept": "application/vnd.github+json",
            "User-Agent": "genai-stack-eval",
            **({"Authorization": f"Bearer {token}"} if token else {}),
        },
    )
    with urllib.request.urlopen(request, timeout=30) as response:
        return json.loads(response.read().decode("utf-8"))


def indexed_paths(index_path: Path) -> set[str]:
    index = json.loads(index_path.read_text(encoding="utf-8"))
    return {record["path"] for record in index.get("files", [])}


def is_interesting(title: str) -> bool:
    lowered = title.lower()
    return len(title) > 25 and not any(word in lowered for word in NOISE)


def candidates(
    repo: str,
    known: set[str],
    limit: int,
    max_files: int,
    token: str | None,
) -> list[dict]:
    listing = _get(
        f"/repos/{repo}/pulls?state=closed&per_page=100&sort=updated&direction=desc",
        token,
    )
    found = []
    for pull in listing:
        if len(found) >= limit:
            break
        if not pull.get("merged_at") or not is_interesting(pull["title"]):
            continue
        try:
            files = _get(f"/repos/{repo}/pulls/{pull['number']}/files?per_page=100", token)
        except urllib.error.HTTPError as error:
            print(f"  pr {pull['number']}: {error}", file=sys.stderr)
            if error.code in (403, 429):
                break
            continue

        touched = [
            normalise_path(entry["filename"])
            for entry in files
            if entry["filename"].endswith(SOURCE_SUFFIXES)
            and "__tests__" not in entry["filename"]
            and not entry["filename"].endswith((".test.ts", ".spec.ts"))
        ]
        if not touched or len(touched) > max_files:
            continue
        # Gold has to name something the index knows about, or the question
        # scores zero for a reason that has nothing to do with retrieval.
        if not all(path in known for path in touched):
            continue

        body = (pull.get("body") or "").strip().splitlines()
        first = next((line for line in body if line.strip() and not line.startswith(("#", "<!--", "-", "*"))), "")
        found.append(
            {
                "id": f"pr-{pull['number']}",
                "question": pull["title"],
                "gold": [{"file": path, "qualified_name": None} for path in touched],
                "category": "",
                "source": "pr",
                "sha": pull.get("merge_commit_sha"),
                "note": first[:200],
                "_url": pull["html_url"],
            }
        )
        print(f"  {pull['number']:<6} {pull['title'][:70]}", file=sys.stderr)
    return found


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", default="facebook/docusaurus")
    parser.add_argument("--index", type=Path, default=Path("index/all_code_blocks.json"))
    parser.add_argument("--limit", type=int, default=25)
    parser.add_argument("--max-files", type=int, default=3)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args(argv)

    known = indexed_paths(args.index)
    print(f"{len(known)} indexed files; mining {args.repo}", file=sys.stderr)
    found = candidates(
        args.repo, known, args.limit, args.max_files, os.getenv("GITHUB_TOKEN")
    )
    lines = "\n".join(json.dumps(candidate) for candidate in found)
    if args.out:
        args.out.write_text(lines + "\n", encoding="utf-8")
        print(f"{len(found)} candidates -> {args.out}", file=sys.stderr)
    else:
        print(lines)
    return 0


if __name__ == "__main__":
    sys.exit(main())

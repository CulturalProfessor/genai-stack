# The question set

Gold is `(file, qualified_name)`. Never a symbol id: an id embeds the line the
symbol starts on, so an edit anywhere above it invalidates gold that is still
correct. `qualified_name` may be null, meaning any symbol in that file counts.

One JSONL file per split. The file name is the split name.

| split | how it was made | in the headline |
|---|---|---|
| `pr.jsonl` | merged pull requests, mined then curated by hand | yes |
| `abstain.jsonl` | hand-written questions this repo cannot answer | yes |
| `generated.jsonl` | written by a model | no |
| `heldout.jsonl` | reserved, consulted rarely | no |

## Why pull requests

A pull request title is written by someone describing a problem, usually before
they have read the code that fixes it. That is the only source here where the
question does not share the vocabulary of its answer. A docstring sits next to
the symbol it describes and is embedded with it; a test name sits in the same
directory; a model asked to write questions writes them in the code's own
words. All three flatter retrieval. Pull requests do not.

The files a pull request changed are the answer to the question its title asks,
which also gives multi-file gold for free, and that is where a graph should
earn its keep over plain chunk search.

## Known bias in this split

Titles keep their conventional-commit prefix, so `fix(utils): ...` hands the
retriever the package name. It is left in because the dataset has to be
reproducible from `eval.mine_prs`, and removing it by hand would make the set
unreproducible in exchange for a bias that is at least uniform across every
question and every configuration.

## Adding to it

```bash
python -m eval.mine_prs --limit 30 --out /tmp/candidates.jsonl
```

Then read every line and decide:

1. **Does the title name a behaviour?** "fix website Showcase image bug" does
   not. Reject it. Roughly one candidate in ten is a version bump or a typo fix
   wearing a descriptive title.
2. **Assign a category.** `locate`, `explain`, `trace`, `impact`, `config`.
   Categories exist so a movement is attributable to a mechanism; `explain`
   should sit near 100% and is the canary for something broken upstream.
3. **Tighten the gold to a symbol.** The miner lists every source file the diff
   touched. Drop fixtures and dogfooding files, then read the diff and name the
   symbol that actually changed. This is the step it is most tempting to skip
   and the most expensive to skip: with file-level gold the measured graph lift
   was 0.0 points on every question, because a neighbour pointing into a file
   that already had a seed in it covers nothing new. The same runs, rescored
   against symbol-level gold, put the lift at 11.1 points. Fetch the diff from
   `https://github.com/<repo>/pull/<n>.diff`, which costs no API quota.
4. **Keep the merge SHA.** It pins the question to a point in history.

Then check the result against the corpus you index:

```bash
python -m eval.run --check --index index/all_code_blocks.json
```

That fails on gold naming a file or symbol the index does not have. Run it
after every re-index. Gold rots, and rotted gold scores exactly like a
retrieval regression.

## Unanswerable questions

`abstain.jsonl` holds questions with no answer in this repo, phrased so they
sound like they might have one. Without them, every change that retrieves more
scores better, and the set quietly rewards a system that never declines to
answer, which the chain's prompt promises it will do.

Nothing here scores whether the model abstained, because no model runs. What is
measured is how *peaked* the search was: whether one hit stands clearly above
the rest, or the results are a flat run of equally mediocre matches.

It cannot be the top score itself. Hybrid search divides every score by the best
score in that same query, so the top hit is 1.0 on every question ever asked and
a threshold on it separates nothing. On the first run the shape did separate the
two groups in the expected direction and nowhere near enough to threshold on.

## Size

Twenty-four questions is enough to catch a change that breaks retrieval and not
enough to resolve a small one. At fifty, the standard error on a 70% score is
about 6.5 points, so aggregate movements under roughly 10 points are noise.
Read the per-question diff instead, which is why every run snapshots one.

# GraphRAG improvement plan

Working plan for improving relationship modelling, retrieval accuracy, multi-language
support and document chunking. Written to be implemented from directly — each phase
lists the files it touches, the concrete change, and how it is verified.

## Baseline (measured, `index/all_code_blocks.json`, facebook/docusaurus)

```
774 files · 3163 symbols
kinds: function 2210 · type_alias 654 · interface 236 · method 42 · class 21

6170 call sites →  local-resolved       820  13.3%
                   global-unique       1063  17.2%
                   ambiguous DROPPED    542   8.8%   (22% of intra-repo calls)
                   external/unknown    3745  60.7%

fully isolated nodes (no in-edge, no out-edge):  1721 / 3163  = 54%
symbols carrying their JSDoc in embedded text:      0 / 3163  =  0%
in-degree ≥ 20: 4 nodes carrying 140 / 1883 edges (super-node skew)
```

Two facts drive the whole plan: **54% of nodes have no edges**, so for a majority of
hits the graph expansion is a no-op and this degrades to plain vector search; and
**no symbol carries its docblock**, so the one piece of natural language next to each
symbol never reaches the embedding.

## Sequencing rationale

The work splits into three kinds, and they need different treatment:

- **Bugs** — no measurement can tell us anything we don't know (JSDoc dropped, PDF
  pages glued together). Fix directly.
- **Structural gaps** — measurable *statically*, for free, with no gold data
  (edge resolution rate, orphan fraction, degree distribution).
- **Bets** — unfalsifiable without a retrieval eval (neighbour ranking, reranking,
  community layer).

So: cheap instrument → bugs → eval → bets. This supersedes the earlier "eval before
everything" ordering; phases 3 and 4 are scored by the Phase 0 instrument and were
never blocked on the eval.

## Groundwork that makes every later phase faster

**G1. Embedding cache.** `import_index` re-embeds all 3163 symbols on every import
([ingest.py:100](../genai_stack/ingest.py)). Phases 1, 3, 4 and 7 all require re-indexing,
and Phase 1 changes the embedded text. Add a disk cache keyed `sha256(text)` →
vector, checked before `embeddings.embed_documents`. Turns a ~minutes re-index into
seconds for unchanged symbols, and we will re-index many times.

**G2. Index format version.** `load_index` only checks for a `files` key
([code_index.py:26](../genai_stack/code_index.py)). Phases 3/4/7 change the record shape.
Add `"version": N` to `build_payload` output and have `load_index` reject or migrate
older versions, so a stale `index/all_code_blocks.json` fails loudly instead of
importing a half-populated graph.

**G3. Fixture drift.** `tests/test_code_index.py` and `tests/test_extract.py` carry
inline index fixtures. Every extractor change touches them. Move the fixture to one
shared `tests/fixtures.py` builder so a shape change is edited once.

---

## Phase 0 — Graph health instrument

**Goal:** a static scoreboard for phases 3–5 that needs no gold data, no LLM, no
questions. This is not the eval; it answers "is the graph well-formed", not "is
retrieval good".

**New:** `eval/graph_health.py`

Two modes, same report:
- `--index path.json` — computes from the index JSON, no infrastructure needed.
- `--neo4j` — same metrics via Cypher against the live graph.

Both matter: the JSON mode is fast enough for a tight loop, the Neo4j mode confirms
that what `ingest.py` *stores* matches what the extractor *intended*. The baseline
numbers above are JSON-mode only and have not yet been checked against the database.

**Metrics:**
| Metric | Baseline | Target after Phase 4 |
|---|---|---|
| intra-repo call sites resolved | 78% (1883/2425) | > 95% |
| fully isolated nodes | 54% | < 20% |
| type/interface nodes with any edge | 0% | > 60% |
| in-degree p50 / p95 / max | — | track, cap in Phase 5 |
| files with unresolved relative imports | — | < 5% |

**Exit:** report runs in both modes, numbers agree within a rounding margin.

### Phase 0 findings (done)

Running it turned up three things the earlier read had missed.

**F1. Barrels and constant modules are invisible, so 12.4% of code imports dangle.**
`build_index` only emits a `FileRecord` when the file yielded symbols
([extract.py:427](../genai_stack/extract.py)), and `visit_declaration` only records a
`const` whose value is a function ([extract.py:367](../genai_stack/extract.py)). A module
of plain constants, or a barrel that only re-exports, therefore never becomes a node —
and every import naming one dangles. Confirmed absent from the index:
`client/exports/ExecutionEnvironment`, `client/common`, `ssg/ssgEnv`,
`commands/build/build`, `docusaurus-theme-common/src/index`.

This is a **prerequisite for Phase 3**, not a separate nicety: import-scoped call
resolution walks *to the imported module's exports*, and if the module is not a node
the walk falls straight through to the global-unique rule it was meant to replace.
Barrels are how a monorepo exposes its packages, so this bites hardest exactly where
cross-package structure lives. Phase 3 also needs re-export following
(`export * from './x'`, `export { a } from './x'`).

**F2. The graph cannot tell repositories apart.** `Repository` has **zero
relationships** — verified on a live database holding four repos. Two consequences:

- File paths are repo-relative and `file_path` is globally unique, so the same path in
  two repos merges into one node. Measured: 70 indexed file records → 67 `File` nodes.
- `GLOBAL_CALLS_QUERY`'s "unique across the whole graph" test ranges over *every repo
  ever imported*. A name unique within one repo but present in another is classified
  ambiguous and dropped, so **resolution quality degrades as more repos are added** —
  the graph gets worse with more data.

`schema.py` documents the absent ownership edge as deliberate ("a File is not owned by
one repository") to let edges resolve across the whole graph. The cross-repo merging
looks like an unintended consequence of that choice rather than the intent. Needs a
decision: scope resolution per repository (add `(:Repository)-[:CONTAINS]->(:File)` and
qualify paths), or keep the global namespace and accept the degradation. Phase 3 fixes
the call half incidentally by scoping through imports; the node-merging half is separate.

**F3. The 54% isolation figure is not a Docusaurus artefact.** The live database holds
four unrelated small repos (ky, p-limit, p-map, inventory-system) and reports 54.9%
isolated against the index's 54.4% — independent corpora, same number.

**Not yet done:** index mode and Neo4j mode have not been run against the *same*
corpus, so the reimplementation of `ingest.py`'s resolution rules is unverified. That
needs a Docusaurus import into a database, which mutates whatever is loaded — worth
doing against a throwaway instance rather than the working one.

---

## Phase 1 — Data-loss fixes — **done**

Both are text corruption. Neither needs measurement.

Verified against a fresh `facebook/docusaurus` checkout, which reproduces the committed
baseline exactly (774 files, 3163 symbols, 479 IMPORTS, 68 dangling, 54.4% isolated) —
so the extraction pipeline matches whatever produced the committed index.

| | before | after |
|---|---|---|
| symbols carrying a doc comment | 0 | **615 (19.4%)** |
| files | 774 | **961** |
| IMPORTS edges | 479 | **687 (+43%)** |
| imports pointing at nothing | 68 (12.4%) | **6 (0.9%)** |

F1 was fixed here rather than in Phase 3: files are now merged in a pass of their own
before the symbols, and `IMPORT_SUFFIXES` tries `.d.ts` last. The six remaining
danglers are legitimate — two name build output absent from a checkout, four sit under
a source directory called `build` that `DEFAULT_IGNORE_DIRS` prunes.

### 1a. Attach JSDoc to symbols
**Files:** `genai_stack/extract.py`

The `comment` node is a **sibling of the outermost declaration wrapper**, not a child
of the declaration:

```
program
  comment                    ← the JSDoc
  export_statement
    function_declaration     ← what SymbolExtractor.add() receives
```

`visit()` unwraps `export_statement` / `ambient_declaration` before calling `add()`
([extract.py:324](../genai_stack/extract.py)), so looking at `node.prev_named_sibling`
from inside `add()` finds nothing. Walk up while the node is its parent's first named
child *and* the parent is a wrapper, then take `prev_named_sibling` and keep it only
if it is a `comment` that is adjacent (no blank-line gap — otherwise a file-header
banner gets attached to the first function in the file).

Inside `class_body` the comment is a direct sibling of `method_definition`, so the
same helper covers methods without special-casing.

- New `Symbol.doc: str | None`.
- `build_text` ([code_index.py:39](../genai_stack/code_index.py)) prepends the doc to
  the embedded text.
- Leave `type_skeleton` alone — it strips comments deliberately and for a stated
  reason; that reasoning does not extend to functions.

**Verify:** unit test on a source with `/** */` before an exported function, before a
method, and a file-header banner that must *not* attach. Then re-run Phase 0 and
confirm docblock coverage moves off 0%. Requires a full re-embed — hence G1 first.

### 1b. Preserve PDF page boundaries
**Files:** `apps/api.py:540`

`"".join(page.extract_text() or "" for page in reader.pages)` glues page N's last word
to page N+1's first word. Join on `"\n\n"`, and track per-page character offsets so
each chunk can carry the page number it started on. Feeds Phase 6's citations.

**Verify:** unit test on a two-page PDF asserting no token spans the boundary and that
chunk page numbers are monotonic.

---

## Phase 2 — Retrieval eval harness

**Goal:** measure stages A (seeding) and B (context assembly). No LLM judge in v1 —
every planned change lands in A or B, and the judge adds cost, noise and its own
prompt-engineering problem. Add stage C later as a rare sanity check.

### Code seams needed
1. `genai_stack/retrieval.py` — a plain `retrieve(question, k, expand=True) -> list[Document]`
   that does not go through `ChatRetrievalQAWithSourcesChain`.
2. A no-expansion variant of `CODE_RETRIEVAL_QUERY` to compute graph lift.
3. Neighbour metadata to carry `(file_path, qualified_name)`, not just qualified name
   ([retrieval.py:78](../genai_stack/retrieval.py)), so coverage scores against gold.

### Dataset
`eval/dataset/*.jsonl`, pinned to a **specific Docusaurus commit SHA** — gold rots
otherwise. Gold identity is `(file_path, qualified_name)`, never symbol ids, because
ids embed line numbers and break on any edit.

```json
{"id": "pr-9821", "question": "sidebar items are duplicated when a category has an index page",
 "gold": [{"file": "packages/docusaurus-plugin-content-docs/src/sidebars/generator.ts",
           "qualified_name": "DefaultSidebarItemsGenerator"}],
 "category": "impact", "source": "pr", "sha": "..."}
```

**Sources, in descending trustworthiness:**
1. **Merged PRs** — index at commit `C`, mine PRs merged after `C` touching 1–3 source
   files. Title/body is the question, changed files are gold. The only source where
   the question was written by someone who had not read the code, so no lexical
   leakage. Also yields multi-file gold, where the graph should earn its keep.
2. **Docusaurus's own docs** (`website/docs/**`) — it documents its own plugin API.
   Exercises the doc↔code path too.
3. **Test descriptions** — `describe`/`it` strings; mild leakage since they sit next
   to the code.
4. **LLM-generated** — circular (the model writes questions in the code's own
   vocabulary). Separate split, reported separately, never in the headline.

Target ~50 hand-curated from (1) and (2).

**Categories**, so results are attributable to a mechanism: `locate`, `explain`
(should be ~100%; if not something is badly broken), `trace`, `impact`, `config`,
and `abstain` — questions with no answer in the repo. Without `abstain`, every change
that retrieves *more* scores better and we optimise toward a system that never says
"I don't know", which the prompt at [chains.py:201](../genai_stack/chains.py) promises it will.

### Metrics
- **Primary — context coverage**, in two tiers: `retrieved` (full text in context) vs
  `mentioned` (signature only). Neighbours are injected as signatures only
  ([retrieval.py:51](../genai_stack/retrieval.py)), so collapsing these would let a
  context full of one-line signatures score as a win.
- **Graph lift** = coverage(expand) − coverage(seeds only). The number that says
  whether this is GraphRAG or vector search with extra steps.
- Seed recall@{4,10,20}, MRR. If recall@20 ≫ recall@4, a reranker is the cheapest win;
  if @20 is also bad, seeding is broken and reranking will not save it.
- **Guardrails:** context tokens, latency, index build time. A change that lifts
  coverage 3% while doubling context tokens is a bad trade we would otherwise ship happily.

### Statistics
At n=50 the standard error on a 70% score is ~6.5 points, so **aggregate movements
under ~10 points are noise**. Mitigations: snapshot per-question results and diff runs
(actionable: "these 4 started passing"), and use paired McNemar rather than comparing
two independent proportions.

### Placement
`eval/` — outside pytest's default run. `requirements-dev.txt` deliberately keeps
`tests/` runnable without the runtime stack; the eval needs Neo4j and an embedding
model. Do not break that property.

**Exit:** `python -m eval.run --config baseline` prints the table and writes a snapshot
that a second run can diff.

---

## Phase 3 — Import-scoped call resolution + `EXPORTS`

Biggest single graph-quality gain. Recovers most of the 542 dropped edges and makes
the 1063 "global-unique" ones trustworthy — today those are correct only by accident
of being the sole candidate.

**Files:** `extract.py`, `code_index.py`, `ingest.py`, `schema.py`

0. **Prerequisite (F1)** — *done in Phase 1*. Files are recorded whether or not they
   define symbols, and `.d.ts` resolves. Still outstanding from F1: recording
   non-function `const` exports, and following re-exports through barrels
   (`export * from './x'`), which step 2 needs to know what a barrel actually exposes.
1. `extract_imports` ([extract.py:248](../genai_stack/extract.py)) currently keeps only the
   module string. Record bindings instead:
   `{specifier, names: [{imported, local}], default_local, namespace_local}`.
2. Mark exported symbols during `visit()` — it already unwraps `export_statement`, so
   set `Symbol.exported` there. Add `(:File)-[:EXPORTS]->(:Code)`.
3. New resolution order in `ingest.py`, replacing the uniqueness guard at
   [ingest.py:57](../genai_stack/ingest.py):
   `same file → imported module's exports (via resolve_import) → globally unique → drop`.
4. Keep the unresolved receiver text (`userService` in `userService.create()`) as a
   property — strong disambiguation hint and useful metadata even unresolved.

**Verify:** Phase 0 resolution rate 78% → >95% of intra-repo call sites; isolated nodes
drop. Then Phase 2 coverage on `trace` and `impact` categories.

---

## Phase 4 — Type edges, heritage, `INSTANTIATES`

Makes the 890 type/interface nodes (28% of the graph) traversable instead of dead ends.

**Files:** `extract.py`, `ingest.py`, `schema.py`

- **`REFERENCES_TYPE`** — walk `formal_parameters → required_parameter →
  type_annotation → type_identifier` and the declaration's own return
  `type_annotation`. Confirmed present in the AST. A hit on `interface Props` then
  walks to every function taking or returning one.
- **`EXTENDS` / `IMPLEMENTS`** — node is `class_heritage`, containing `extends_clause`
  and `implements_clause`. (Not `heritage_clause`, as an earlier note said.)
- **`INSTANTIATES`** — split `new_expression` out of `CALLS`
  ([extract.py:80](../genai_stack/extract.py)). Different question, different answer.
- **`OVERRIDES`** — method → same-named method on an ancestor class, once `EXTENDS` exists.

Retrieval query gains these as separate optional traversals with their own labels in
the context header.

**Verify:** Phase 0 — type nodes with ≥1 edge from 0% to >60%, isolated nodes <20%.

---

## Phase 5 — Neighbour ranking, super-node caps, dedup

The first phase that is a genuine bet — needs Phase 2 to score.

- `collect(DISTINCT callee)[..5]` ([retrieval.py:37](../genai_stack/retrieval.py)) takes an
  arbitrary five in planner order. Store a call-site `count` on `CALLS` and rank by
  `edge weight × inverse target degree` — IDF for graph neighbours.
- Exclude nodes above a degree threshold. Measured skew is mild here (`stringify` at
  51, 4 nodes ≥20) but Docusaurus is small; on Rocket.Chat this is the difference
  between a useful `Called by` list and 800 lines of noise.
- Dedup: two hits in the same file each re-emit the identical `// This file imports:`
  line, and a hit plus its own callee can both land in the k, sending the same code twice.

**Verify:** Phase 2 coverage flat-or-up while context tokens go *down*.

---

## Phase 6 — Document graph

The doc half has no traversal at all — `PdfDocument` has no edge to its chunks, by
design, because Neo4jVector writes them ([pdf_store.py:12](../genai_stack/pdf_store.py)).

- Write chunks ourselves → `HAS_CHUNK`, a `NEXT` chain (expand a hit to its neighbours,
  small-to-big retrieval), `Section` hierarchy from headings.
- Carry the page number from Phase 1b so citations can say "p. 14" instead of only the
  document name ([chains.py:164](../genai_stack/chains.py)).
- Fulltext index on chunks — code gets hybrid, docs get vector-only today
  ([retrieval.py:108](../genai_stack/retrieval.py)).
- **Contextual retrieval** — prepend an LLM-generated one-sentence "where this sits in
  the document" to each chunk before embedding. One-time index cost.
- **`(:PdfChunk)-[:MENTIONS]->(:Code)`** — link identifiers in doc chunks to symbols.
  This is what unifies the two stores rather than running them side by side, and it
  directly serves the "does this match the spec?" case the prompt already advertises
  at [chains.py:180](../genai_stack/chains.py).

---

## Phase 7 — Multi-language backend

Language-specific surface is small: `LANGUAGE_BY_SUFFIX`, the five node-type sets,
`signature_of`, `callee_name`, `extract_imports`, `resolve_import`.

Introduce one `SymbolBackend` interface so `code_index.py` and `ingest.py` do not care
which produced the JSON, then:

1. **`tags.scm` queries** as the default. Upstream grammars ship these for ~20
   languages with `@definition.function` / `@reference.call` already tagged — breadth
   nearly free. They do *not* cover imports or heritage, so keep a small per-language
   supplement for those.
2. **SCIP indexers** (`scip-typescript`, `scip-python`, …) as an opt-in precise mode.
   Real resolved cross-file references with stable symbol identity — the entire
   name-ambiguity problem disappears. Cost: each language's toolchain and a working build.

Also index non-code files: Markdown, `package.json` workspaces → `Package` nodes
(monorepos), OpenAPI, SQL migrations.

---

## Phase 8 — Rerank, then community summaries

- **Exact-identifier seeding** — if the question contains a camelCase/PascalCase or
  backticked token, do a direct `name` lookup and seed traversal from it rather than
  hoping hybrid search stumbles onto it. Cheap, and it is the most common code-question shape.
- **Rerank** — retrieve ~30 with RRF fusion, rerank to 6. `k=4` fixed today
  ([chains.py:27](../genai_stack/chains.py)). Only worth it if Phase 2 shows recall@20 ≫ recall@4.
- **Two-hop with decay** for `trace` questions, budgeted by tokens not a fixed neighbour cap.
- **Community layer** — Leiden/label propagation over `IMPORTS`+`CALLS` (Neo4j GDS),
  LLM-summarise each community at index time, store `(:Module {summary, embedding})`
  and search alongside `Code`. Answers "how does X work overall", which no single
  symbol matches today. Directory structure is a free approximation to start — `depth`
  is already computed and never used.

---

## Risks

- **50 questions is small.** Per-question diffing mitigates; it does not fix.
- **Overfitting to the eval set.** Keep a held-out split, consult it rarely.
- **One repo.** Docusaurus is a plugin-architecture monorepo; findings may not transfer
  to Rocket.Chat's shape and certainly not to Python/Go. Want 2–3 corpora eventually.
- **Embedding drift.** Pin eval runs to the local sentence-transformer; hosted
  embeddings change between versions and we would misread that as a code regression.
- **Re-embed cost.** Phase 1 changes embedded text for every symbol. G1 (cache) before
  Phase 1, or expect a full re-embed on each iteration.

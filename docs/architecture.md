# Architecture — Agentic RAG Platform

## Overview

```
                        ┌─────────────────────┐
                        │   QueryEngine.ask   │
                        └──────┬──────────────┘
                               │ 1. ledger.check (fail-closed budget)
                               ▼
                 ┌─────────────────────────┐
                 │ planner.plan(question)  │  decompose + classify
                 └──────┬──────────────────┘
                        │ 2. retrieval rounds (≤ max_rounds)
                        ▼
        ┌──────────────────────────────────────────┐
        │ DocumentStore.retrieve per sub-query     │ hybrid:
        │  vector cosine (hashed embeddings)       │  w·vector + (1-w)·keyword
        │  + BM25-style keyword score              │
        └──────┬───────────────────────────────────┘
               │ 3. sufficiency.sufficiency (weakest link)
               ▼
        ┌──────────────────────────────────────────┐
        │ sufficient? ──yes──▶ generate            │
        │      │ no: follow_ups → retrieve again   │
        │      │ no (final round): tools           │
        └──────┬───────────────────────────────────┘
               │ 4. detect_tool_need → ToolRegistry.call
               ▼
        ┌──────────────────────────────────────────┐
        │ generator.generate (extractive, cited)  │
        │  or ABSTAIN with a reason                │
        └──────┬───────────────────────────────────┘
               │ 5. ledger.record + audit.append
               ▼
                        Answer
```

## Components

### Embedder (`embedder.py`)
SHA-256 hashed embeddings: each token hashes into one of 128 buckets,
counts accumulate, the vector is L2-normalised. Tokenisation lowercases,
keeps alphanumerics, applies light suffix stemming (`costs`→`cost`,
`points`→`point`), and drops a stopword list (including low-information
quantifiers `many`/`much`). Deterministic by construction; identical
text always yields the identical vector.

### Store (`store.py`)
Documents are chunked by packing sentences up to 60 words with a
one-sentence overlap. Each chunk is indexed twice: its hashed embedding
and its content-token list with document frequencies. Retrieval scores
every chunk with cosine similarity and a BM25-style keyword score
(IDF-weighted TF saturation, normalised by the query's total IDF —
unseen terms count in the denominator with IDF `log(N+1)`, so a query
full of absent terms scores low). The hybrid score is
`w·vector + (1-w)·keyword` with `w` from the active policy.

### Planner (`planner.py`)
Rule-based decomposition: a question joined by "and" splits only when
every clause after the first starts with a question word (so "buy X and
Y, what is the total?" stays whole); a keyword-only sub-query is added
for high-recall retrieval. Sub-queries are classified `arithmetic`
(numbers + total/sum), `date` (what day / days until), `definition`
(what is X), else `factual`.

### Sufficiency (`sufficiency.py`)
Coverage of a sub-query is its top-1 hybrid retrieval score. Overall
sufficiency is the minimum across sub-queries (weakest link). Uncovered
sub-queries generate follow-ups: keyword-only form, synonym expansion
from a small built-in map, and a definition-seeking form.

### Tools (`tools.py`)
`ToolRegistry` holds `ToolDefinition`s (JSON-schema subset) plus a
handler. `calculator` evaluates arithmetic via a restricted AST
(numbers and `+ - * / % **` only — no names, calls, or attribute
access, so injection strings fail). `date_lookup` answers `today`,
`day of week YYYY-MM-DD`, `days until`, and `days between` against the
configured reference date. `definition_lookup` reads the glossary.
`detect_tool_need` maps a sub-query to a tool call; arithmetic/date/
definition kinds always try their tool (retrieved text can state the
inputs but never supply a computed sum or a calendar fact), factual
kinds only when still uncovered. Definition lookup only fires when the
term is actually in the glossary.

### Generator (`generator.py`)
Extractive: sentences are selected from retrieved chunks and tool
results, emitted verbatim with `[doc#chunk]` or `[tool:name]`
citations. Selection is sub-query aware — each sub-query contributes
its best-overlapping sentence first — then fills remaining slots (one
per chunk first, so one document cannot dominate). A sentence must
share ≥2 content tokens with the question (≥1 if the question has only
one). Abstention triggers when sufficiency is below the policy's
abstain threshold (and no tool result exists) or when no evidence
clears the minimum evidence score.

### Ledger (`ledger.py`)
`Ledger.check` runs before execution and raises `BudgetExceeded` if
the tenant's cost, request, or token cap would be exceeded.
`Ledger.record` accumulates per-tenant usage after each request.
Token estimates are deterministic (~1.3 tokens/word); cost is
`(tokens_in + tokens_out)/1000 · cost_per_1k + rounds · cost_per_round
+ Σ tool costs`.

### Policies (`policies.py`)
`PolicyRegistry` holds immutable `RetrievalPolicy` versions; exactly
one is ACTIVE. `gated_promote` activates a candidate only if its
golden-set report clears absolute floors (recall@k 0.70, citation
precision 0.70, abstention correctness 0.80) and regresses no metric
vs the baseline by more than 0.05. A failed promotion leaves the
active policy untouched and records the reasons.

### Eval (`eval.py`)
Runs the golden set through the engine: answer accuracy (fraction of
expected keywords present; 1.0 for a correct abstention), citation
precision (fraction of cited doc sources that are expected; 1.0 when
correctly abstaining), abstention correctness (predicted abstention
matches expectation), retrieval recall@k (fraction of expected sources
in the top-k).

### Audit (`audit.py`)
Append-only list; each entry's SHA-256 hash commits to the previous
entry's hash (`GENESIS` for the first). `verify()` recomputes the chain
and reports the first mismatch — a tampered detail or a broken link.

## Request flow (detailed)

1. `ask` assigns a request id and runs the fail-closed budget check.
2. The planner decomposes the question (anchor + keyword sub-queries).
3. Up to `max_rounds` retrieval rounds: retrieve per open sub-query,
   merge hits by chunk (best score wins), score sufficiency; if below
   threshold and rounds remain, generate follow-ups and repeat.
4. If still insufficient, call tools for arithmetic/date/definition
   sub-queries and any uncovered factual ones. A successful tool call
   counts as filling its gap.
5. Generate: extractive cited answer, or abstain with a reason.
6. Record ledger usage and append the `answer` audit entry.

## Calibration

Measured on the 10-case demo golden set (hybrid scores, vector
weight 0.5, top-k 5):

| Case kind | Sufficiency |
| --- | --- |
| Grounded factual (returns, shipping, loyalty, store, discount) | 0.35 – 0.62 |
| Multi-part (weakest part) | 0.40 |
| Arithmetic (prices retrieved, sum via tool) | 0.33 |
| Out-of-corpus (quarterly revenue) | 0.18 |
| Date (calendar fact via tool) | 0.16 |

Thresholds: sufficiency 0.30, abstention 0.25 — between the lowest
grounded score (0.33) and the highest ungrounded score (0.18).
Golden-set results with these thresholds: answer accuracy 1.000,
citation precision 0.950, abstention correctness 1.000, recall@5
1.000. The single sub-perfect metric is the definition case's citation
precision (0.50): the answer also cites a product sentence that
truthfully states the same warranty fact but is not in the case's
expected-source list — grounded, but counted as imprecise by the
strict expected-source check.

## Failure modes (and what happens)

| Failure | Behaviour |
| --- | --- |
| Tenant over budget | `BLOCKED` answer before any execution; audit entry `request_blocked` |
| Unknown tool / invalid args | Tool call returns `ok=False`; engine continues with other evidence |
| Retrieval insufficient after all rounds | Follow-ups, then tools, then abstention — never a guess |
| Tool returns an error | Recorded in the audit log; generation proceeds on remaining evidence |
| Policy candidate fails a gate | Promotion refused with reasons; active policy untouched |
| Audit log tampered | `verify()` reports the first mismatched entry |

## Extension points

- **Real LLM generator**: replace `generator.generate` with a hosted
  model call behind the same signature; keep the abstention and
  citation rules — they are policy, not prompt.
- **Real embeddings**: swap `embedder.embed` for a model-backed
  embedder; the store, planner, and sufficiency layers are unchanged.
- **Persistent backends**: `DocumentStore`, `Ledger`, `AuditLog`, and
  `PolicyRegistry` are narrow in-memory classes; persist each behind
  its current interface without touching `QueryEngine`.
- **More tools**: register a `ToolDefinition` + handler in
  `ToolRegistry` and extend `detect_tool_need` — the engine needs no
  changes.

## CI

`docs/github-actions-ci.yml` ships under `docs/` because the GitHub
App integration used for this portfolio cannot write workflow files
directly. Activate it by moving it to `.github/workflows/ci.yml`; it
runs the test suite, the demo, and a compile check on Python 3.11/3.12.

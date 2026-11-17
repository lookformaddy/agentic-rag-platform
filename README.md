# Agentic RAG Platform

**Don't just retrieve — reason about whether the retrieval was enough.**
An agentic retrieval-augmented generation platform that decomposes a
question into sub-queries, runs iterative retrieval rounds with a
self-critique sufficiency check, calls schema-validated tools when
retrieval is insufficient, and generates citation-attached answers —
abstaining, explicitly, when the evidence is thin.

## The enterprise problem

Naive RAG fails in production in three predictable ways. It retrieves
once and trusts the result, so a multi-part question gets a partial
answer. It hallucinates when the corpus has no answer, because nothing
checks whether the retrieved passages actually cover the question. And
it has no cost or quality governance: no per-tenant budgets, no
versioned retrieval policy, no golden-set regression gate before a
retrieval change ships. This platform closes all three gaps in one
deterministic, offline, testable service:

- **Query decomposition planner** — splits multi-part questions into
  sub-queries ("What is the return window and how much does express
  shipping cost?" becomes two retrievable parts plus a keyword form).
- **Iterative retrieval with self-critique** — each round scores every
  sub-query's coverage (weakest link wins); uncovered sub-queries drive
  deterministic follow-up reformulations (keyword-only, synonym
  expansion, definition-seeking) and retrieval repeats, up to a
  per-policy round cap.
- **Schema-validated tools** — when retrieval cannot supply an answer,
  the agent calls scripted tools: `calculator` (safe AST arithmetic),
  `date_lookup` (deterministic calendar facts), `definition_lookup`
  (glossary). Arguments are validated against a JSON-schema subset
  before execution; unknown tools and bad args fail closed.
- **Citation-attached generation with abstention** — extractive,
  verbatim answers where every sentence carries its source
  (`doc#chunk` or `tool:<name>`). Selection is sub-query aware, so each
  decomposed part contributes its best sentence. If sufficiency is
  below the abstention threshold, or no evidence clears the minimum
  score, the answer is an explicit abstention — never a guess.
- **Cost / latency / token accounting with fail-closed tenant budgets**
  — caps are checked *before* execution; a tenant over its cost,
  request, or token cap gets a `BLOCKED` answer, not a bill.
- **Versioned retrieval policies with A/B comparison and gated
  promotion** — policies (weights, top-k, thresholds, round caps) are
  immutable versions; a candidate is promoted only if it clears
  recall@k, citation-precision, and abstention floors on the golden
  set and does not regress the active policy.
- **Golden-set eval harness** — answer accuracy, citation precision,
  abstention correctness, and retrieval recall@k per case.
- **Hash-chained audit log** — every request, retrieval round, tool
  call, and answer is appended to a SHA-256 hash chain with tamper
  detection.

## What it demonstrates

- An agentic loop with a real self-critique step: plan → retrieve →
  score sufficiency → reformulate → retrieve → tools → generate/abstain.
- Hybrid retrieval (deterministic hashed embeddings + BM25-style
  keyword scoring) with a calibration story: thresholds sit between
  grounded sub-queries (~0.33+) and out-of-corpus questions (~0.18).
- Fail-closed defaults everywhere: unknown tool, invalid args,
  insufficient evidence, exceeded budget all refuse *before* any side
  effect or invented answer.
- Deterministic, seeded, offline behaviour: no network, no API keys,
  no numpy/sklearn/torch — the whole core is pure stdlib, so the same
  question always yields the same answer, cost, and audit trail.

## Quickstart

```bash
make install   # create .venv and install dependencies
make test      # run the test suite
make demo      # six-scenario demo (see below)
make serve     # FastAPI service on :8000
```

Or without make:

```bash
python -m venv .venv && .venv/bin/pip install -r requirements.txt
PYTHONPATH=src python -m pytest -q
PYTHONPATH=src python scripts/demo.py
PYTHONPATH=src python -m uvicorn ragagent.service:app --port 8000
```

CLI:

```bash
PYTHONPATH=src python -m ragagent.cli ask "What is the return window?"
PYTHONPATH=src python -m ragagent.cli ask "If I buy the trail jacket at \$129.99 and the hiking boots at \$159.99, what is the total before tax?"
PYTHONPATH=src python -m ragagent.cli eval
PYTHONPATH=src python -m ragagent.cli policies
```

## The demo

`scripts/demo.py` runs six scenarios end to end:

1. **Simple factual answer** — "What is the return window?" answered
   with a citation to `returns-policy`.
2. **Multi-part decomposition** — the planner splits the question,
   both parts are grounded, and the answer carries "30" and "12.99".
3. **Tool-backed arithmetic** — retrieval states the two prices; the
   `calculator` tool supplies the total the text cannot: 289.98.
4. **Abstention on thin evidence** — a question about quarterly revenue
   (absent from the corpus) abstains with a reason instead of guessing.
5. **Fail-closed tenant budget** — a near-zero cost cap blocks the
   request before any execution.
6. **Golden-set eval + gated promotion** — the 10-case golden set scores
   answer accuracy 1.000, citation precision 0.950, abstention
   correctness 1.000, recall@5 1.000; a recall-heavy candidate policy
   (top-k 8) is evaluated A/B and promoted through the gate. The
   hash-chained audit log then verifies end to end.

## API

| Endpoint | Description |
| --- | --- |
| `GET /health` | Status, chunk count, active policy, audit size |
| `POST /ask` | Ask a question (`question`, `tenant`) — 429 when budget-blocked |
| `POST /ingest` | Add a document to the store |
| `GET /policies` | List retrieval policy versions |
| `POST /policies` | Register a draft policy version |
| `POST /policies/{version}/activate` | Activate a policy version |
| `POST /eval` | Run the golden-set eval harness |
| `GET /ledger/{tenant}` | Per-tenant usage vs caps |
| `GET /audit/verify` | Verify the hash-chained audit log |

## Project layout

```
src/ragagent/
  config.py       environment-driven settings (ARP_ prefix)
  models.py       plain dataclasses (no pydantic in the core)
  embedder.py     deterministic SHA-256 hashed embeddings + stemming
  store.py        hybrid retrieval store (vector + BM25-style keyword)
  planner.py      query decomposition + sub-query classification
  sufficiency.py  weakest-link coverage scoring + follow-up generation
  tools.py        schema-validated tool registry (calculator/date/glossary)
  generator.py    sub-query-aware extractive generation + abstention
  ledger.py       cost/latency/token accounting, fail-closed budgets
  policies.py     versioned policies, A/B compare, gated promotion
  eval.py         golden-set eval harness (4 metrics)
  audit.py        SHA-256 hash-chained audit log
  engine.py       the agentic loop (plan → retrieve → critique → tools → answer)
  service.py      FastAPI service (optional; auto-skips in tests without it)
  cli.py          ask / eval / policies / demo
  demo.py         engine factory + six-scenario demo
  demo_data.py    synthetic corpus, glossary, golden set (fictional retailer)
tests/            pytest suite (FastAPI tests auto-skip without FastAPI)
docs/architecture.md
docs/github-actions-ci.yml   # activate by moving to .github/workflows/ci.yml
```

## Key design decisions and trade-offs

- **Extractive, not generative, answers.** The generator emits retrieved
  sentences verbatim. That bounds hallucination to zero by construction
  and makes every claim checkable — the right default for enterprise
  QA. The trade-off is stilted prose; a hosted LLM can replace the
  generator behind the same interface when fluency matters, with the
  abstention and citation rules unchanged.
- **Deterministic hashed embeddings instead of a model.** No downloads,
  no GPU, no drift: identical text always yields the identical vector.
  The trade-off is no semantic generalisation beyond token overlap and
  stemming — acceptable here because the keyword channel carries the
  precision and the corpus is a controlled enterprise corpus.
- **Weakest-link sufficiency.** Overall sufficiency is the *minimum*
  sub-query coverage, not the mean: an answer is only as grounded as
  its least-covered part. This is stricter than most RAG systems and
  deliberately so — partial answers are how bad decisions get made.
- **Tools for gaps retrieval cannot fill.** Arithmetic, calendar, and
  glossary facts are computed/looked up, not retrieved. The agent calls
  a tool for arithmetic/date/definition sub-queries by kind, and for
  factual sub-queries only when still uncovered after the final round.
- **Budgets checked before execution.** The ledger check runs before
  planning, so a blocked request costs nothing and leaves no partial
  state — the same fail-closed posture as the rest of the platform.
- **In-memory stores.** The vector store, ledger, audit log, and policy
  registry are in-memory by design: this is a reference platform, not a
  deployment. Each has a narrow interface (`DocumentStore`, `Ledger`,
  `AuditLog`, `PolicyRegistry`) so a persistent backend can be swapped
  in without touching the engine.
- **Thresholds are calibrated, not guessed.** Sufficiency 0.30 and
  abstention 0.25 sit between measured grounded sub-queries (~0.33+)
  and out-of-corpus questions (~0.18) on the demo golden set; the
  calibration is documented in `docs/architecture.md` and re-checked
  by the eval harness on every run.

## Configuration

All settings are environment variables with the `ARP_` prefix; see
`.env.example`. The most load-bearing: `ARP_SUFFICIENCY_THRESHOLD`
(0.30), `ARP_ABSTAIN_THRESHOLD` (0.25), `ARP_DEFAULT_VECTOR_WEIGHT`
(0.5), `ARP_DEFAULT_TOP_K` (5), `ARP_TENANT_COST_CAP_USD` (1.00),
`ARP_REFERENCE_DATE` (2026-11-17, makes date tools deterministic).

## Clean-room note

All code is original, all data is synthetic, and all domains are
generic. The demo corpus describes "Northwind Outfitters", a fictional
retailer invented for this project. No employer's code, data, internal
architecture, project names, or proprietary designs appear anywhere.

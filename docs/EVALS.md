# Evaluation Guide (Phase 1)

This document explains the evaluation methodology and how to interpret reports.

## Golden Dataset
Location: `evals/golden_queries.jsonl`

Each line is a JSON object:
```json
{"id": "q1", "question": "...", "reference_answer": "...", "source_file": "hybrid-retrieval.md", "difficulty": "easy"}
```

- `source_file`: enables retrieval hit-rate (expected doc must appear in top-K)
- `reference_answer`: enables generation eval (faithfulness + relevance vs. gold)
- `difficulty`: slice metrics (easy/medium/hard)

## Metrics

### Retrieval
| Metric | Formula | Target |
|---|---|---|
| hit_rate@1 | % queries where expected source is rank 1 | > 0.4 |
| hit_rate@3 | % queries where expected source in top 3 | > 0.7 |
| hit_rate@5 | % queries where expected source in top 5 | > 0.85 |
| MRR | Mean Reciprocal Rank of first hit | > 0.5 |

### Generation
| Metric | Scale | Target |
|---|---|---|
| mean_faithfulness | 1–5 (LLM judge) | > 4.0 |
| mean_relevance | 1–5 (LLM judge) | > 4.0 |

Faithfulness: Is every claim supported by the retrieved context?
Relevance: Does the answer address the question?

## Running Evaluations

```bash
# Retrieval-only (no LLM key needed)
make eval

# With generation evals (requires TC_LLM_API_KEY in .env)
# Set key first, then:
make eval
```

Output example:
```
=== RETRIEVAL ===
n=18 MRR=0.62 hit@1=0.39 hit@3=0.72 hit@5=0.83
=== GENERATION ===
n=18 mean_faithfulness=4.2 mean_relevance=4.3
=== DIFF VS PREVIOUS ===
hit_rate_at_3: 0.67 -> 0.72 (+0.05)
mean_faithfulness: 3.9 -> 4.2 (+0.3)
```

Reports saved to `evals/reports/run_YYYYmmdd_HHMMSS.json` and `evals/reports/latest.json`.

## Eval-Driven Development Workflow

1. **Baseline**: Run evals on current code → record numbers in `docs/EVALS.md`
2. **Change ONE thing**: e.g., chunk size 500→400, or enable reranker
3. **Re-run**: `make eval`
4. **Diff**: Check `diff_vs_previous` in report
5. **Decision**:
   - If key metrics improve (hit@3 +0.02, MRR +0.01, faithfulness +0.2): keep change, commit with eval diff in message
   - If no improvement or regression: revert, try different change
6. **Log**: Append to `docs/EVALS.md`:
   ```markdown
   ## 2025-01-15: Chunk size 500→400, rerank enabled
   - Retrieval: hit@3 0.67→0.72 (+0.05), MRR 0.58→0.62 (+0.04)
   - Generation: faithfulness 3.9→4.2 (+0.3), relevance 4.0→4.3 (+0.3)
   ```

## Adding New Golden Queries

1. Use the app: `curl -X POST /ask -d '{"question": "your question"}'`
2. Verify answer quality manually
3. Note which source file(s) contain the answer
4. Add line to `evals/golden_queries.jsonl` with `reference_answer` (your verified good answer)
5. Re-run evals — new query appears in report

## Interpreting Diffs

| Situation | Action |
|---|---|
| Retrieval ↑, Generation ↔ | Retrieval change helped; generation bottleneck is prompt/model |
| Retrieval ↔, Generation ↑ | Prompt/judge/model change helped retrieval quality |
| Both ↓ | Revert; change introduced bug or overfit |
| Retrieval ↑, Generation ↓ | Reranker may be over-filtering; check top_k vs final_top_k |

## Continuous Integration

The workflow `.github/workflows/eval.yml` runs, on every push/PR:

1. `ruff check` + `pytest` (hermetic)
2. `python scripts/index_docs.py` (builds the index; downloads MiniLM once)
3. `python scripts/run_evals.py --gate --no-rerank --baseline evals/gate_baseline.json`
   — regressions vs the **committed pinned baseline** (a fresh checkout has no
   `evals/reports/` history) plus the absolute floors (see Phase 3 below)
4. `python scripts/drift_check.py` — trend bands vs the committed
   `evals/drift_baseline.json` (see *Drift monitoring*)

`TC_LLM_API_KEY` is read from the `LLM_API_KEY` repo secret and is optional:
generation/conversation evals skip without it and the gate treats skips as
passes. `TC_USE_RERANK=false` is set in the workflow so CI never downloads the
cross-encoder. After accepting a new performance level, refresh the pinned
baselines: copy `evals/reports/gate_latest.json` → `evals/gate_baseline.json`
and run `python scripts/drift_check.py --update-baseline`.

---

# Phase 2 — Conversation / Agent Evals

Phase 2 evals measure the **agent loop**, not just retrieval:

| Script | What it checks |
|---|---|
| `scripts/run_evals.py --conversation` | Multi-turn chat flows, expected tool calls. |
| `scripts/e2e_agent_check.py` | End-to-end agent loop with a **fake LLM** (no key needed). |
| `scripts/e2e_sse_check.py` | SSE streaming round-trip. |

## Conversation Golden Set (`evals/conversation_tests.jsonl`)

One JSON object per line:
```json
{"id": "conv1", "name": "RAG basics flow", "difficulty": "easy",
 "turns": [{"user": "...", "expected_tools": ["search_knowledge"],
            "expected_topics": ["hybrid-retrieval"], "min_faithfulness": 4}]}
```

## Running

```bash
make eval-conv        # or: python scripts/run_evals.py --conversation
make agent            # fake-LLM end-to-end agent loop
make sse              # SSE streaming check
```

## Metrics
- **tools_match**: did the agent call exactly the expected tools for that turn?
- **faithfulness**: LLM-judge rating of whether the answer stays grounded in context (needs `TC_LLM_API_KEY`).
- **expected_topics present**: did the answer address the topics the question requires?

These binary pass/fail checks gate Phase 3: a fine-tune only ships if cycle-detection and tools_match don't regress.

---

# Phase 3 — Eval-Gated Checkpoint Gate

`make gate` (or `python scripts/run_evals.py --gate --no-rerank`) chains every
eval and **exits non-zero on any regression**, so it can block a merge:

1. retrieval + generation (Phase 1, `--no-rerank` by default in the Makefile)
2. `scripts/e2e_agent_check.py` subprocess (must exit 0)
3. conversation evals (Phase 2; skipped cleanly without `TC_LLM_API_KEY`)
4. finetune export metrics + fake-LLM `tool_faithfulness` replay (Phase 3)

Then it diffs the collected metrics against the previous report in
`evals/reports/gate_latest.json` (a timestamped copy is kept alongside it) and
prints `FAIL:` lines for:

- any drop in MRR / hit@1 / hit@3 / hit@5, conversation pass rate, or
  `tool_faithfulness`;
- agent check exit code ≠ 0;
- `tool_faithfulness < 1.0` — the fake LLM replays each conversation turn with
  schema-valid tool args, so the best case is 1.0; anything lower means the
  loop/registry no longer produces the tool behaviour the training data assumes;
- fewer than 1 training example while raw turns exist;
- generation faithfulness below `settings.faithfulness_threshold` (only when
  the judge actually ran).

Skipped metrics (no API key, no judge) are **not** failures. An existing but
unreadable baseline **is** a failure — a corrupt `gate_latest.json` must not
silently disable regression checking. The first run establishes the baseline.
CI pins the comparison with `--baseline evals/gate_baseline.json` (the pinned
file is never overwritten; the current run always lands in
`evals/reports/gate_latest.json`).

The replay lives in `app/eval_finetune.py`; the pure diff/floor logic in
`app/eval_gate.py` (unit-tested hermetically by `tests/test_eval_gate.py`).

---

## Drift monitoring (P3.5)

The gate is a run-to-run regression check; drift asks whether metrics or
training data have moved materially away from an **accepted, pinned**
baseline:

```bash
make drift-check                            # compare vs evals/drift_baseline.json
python scripts/drift_check.py --update-baseline   # accept the current state
python scripts/drift_check.py --strict      # exit 1 on WARN-level drift too
```

- Compares the export health metrics (`n_examples`, `kept_ratio`,
  `tool_calls_success_rate`, `failed_generation_ratio`, split sizes, …) plus
  every gate metric against `evals/drift_baseline.json`.
- Only **unfavourable** movements are drift: declines for higher-is-better
  metrics, increases for `dedup_ratio` / `failed_generation_ratio`. Improvements
  are ignored.
- Relative bands: **WARN** at ≥ 10%, **FAIL** (exit 1) at ≥ 25%
  (`--warn`/`--fail` to override). Default exit is 0 on warns — a standing
  WARN about junk generations (`> 50%` of exports starting with
  `LLM call failed`) prints but does not fail unless `--strict`.
- A missing baseline is established on first run (exit 0); an existing but
  unreadable baseline **fails** — same contract as the gate.
- Pure logic in `app/drift.py`, CLI in `scripts/drift_check.py`, tests in
  `tests/test_drift.py`.

## Synthetic data never trains

`data/episodic/` is the LoRA input, so every synthetic writer is isolated:

- conversation-eval dumps (`eval_*` / `metadata.test_id`) are skipped by
  `load_episodic_turns` and eval runs call `end_conversation(persist=False)`;
- tests, the conversation harness, and the e2e scripts construct
  `MemoryManager` against temp/`traces/` paths, never the real
  `data/memory` + `data/episodic`;
- `make lora-data` on an empty/synthetic-only directory truncates the exports
  to empty and exits 1, so `--check-only` reflects reality instead of
  validating stale files.

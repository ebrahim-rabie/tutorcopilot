# Phase 3 Roadmap — LoRA Fine-Tuning, vLLM Serving, Eval-Gated CI/CD

Phase 3 turns the Phase 2 conversation traces into a **fine-tuned model** and
ships it behind an **eval-gated CI/CD pipeline**. It follows the same rule as
Phases 1 and 2: **every step is hermetic, eval-first, and gated** — you can
build and test every piece with no API key, no large model downloads, and no
network (exactly as Phase 2 used fake LLM clients + HashEmbedder).

## Goal & Guardrails

- **DPO/SFT data from real logs**: convert saved `traces/*.json` + episodic
  memory dumps into `(user, assistant[, rationale])` pairs usable by
  `trl`/`peft` (LoRA).
- **Judge-as-scorer**: reuse `app/eval_conversation.py` + `GuardrailManager`
  faithfulness to score generations; a candidate LoRA only lands if it does
  **not regress** the Phase 2 conversation evals (cycle detection, tools, weak
  topics).
- **vLLM serving**: expose the fine-tune through a vLLM-compatible endpoint so
  `/agent/ask` keeps working by swapping `TC_LLM_BASE_URL`.
- **CI/CD**: a `make` gate (`make gate`) that runs retrieval + agent +
  conversation evals from a pinned checkpoint and blocks merge on regression.

> Blocked today (needs your install): the cross-encoder reranker (~430 MB)
> and a `TC_LLM_API_KEY`. Everything below is implementable **now** with the
> fake-LLM + local-embedding hermetic path; real fine-tuning runs the moment
> you set the env var.

## Phases → Deliverables

### P3.1 — Trace → Training-Example Export (`app/finetune/`) ✅ DONE

Lifted from `traces/*.json` + `data/episodic/*.json(l)` dumps. Implemented as
one schema owner instead of the original three-module sketch:

- `app/finetune/schema.py`: `ToolCallRecord` / `TrainingExample` dataclasses
  plus the pure pipeline `load_episodic_turns` → `to_examples` (purity gate:
  non-empty question+generation) → `dedupe` (exact pairs) → `split`
  (seeded 80/20). Hermetic: no ML deps, importable without torch/transformers.
- `scripts/export_lora_data.py`: thin CLI over the schema module; writes
  `config/finetune/train.jsonl` + `val.jsonl`.

**Hermetic test** `tests/test_finetune_export.py` (passing): purity, dedupe,
deterministic split, hard-negative survival, package importable without ML deps.

```
make lora-data          # python scripts/export_lora_data.py
```

### P3.2 — Eval-Gated Checkpoint Gate (`scripts/run_evals.py` extension) ✅ DONE

- `app/eval_finetune.py`: `FinetuneEvalRunner` that measures:
  - `train_examples` (size + `dedup_ratio` + `kept_ratio`, `n_train`/`n_val`),
  - `tool_faithfulness` via the existing conversation harness on the exported
    answers — a fake LLM replays each turn (schema-valid tool args, answers
    cycled from the exports) as the hermetic best-case bound; must read 1.0.
- `app/eval_gate.py`: pure gate logic — diff vs the previous `gate_latest.json`
  report (MRR, hit@*, conversation pass rate, tool_faithfulness) plus absolute
  floors (agent exit code, train_examples ≥ 1, faithfulness ≥
  `settings.faithfulness_threshold`). Skipped metrics (no API key / no judge)
  are ignored; an unreadable baseline **fails** rather than silently passing.
- `--finetune` and `--gate` flags on `scripts/run_evals.py`; the gate chains
  retrieval + generation, the `e2e_agent_check` subprocess, conversation evals,
  and the finetune export/replay, then prints FAIL lines and exits non-zero.

```
make gate              # python scripts/run_evals.py --gate --no-rerank
```

**Hermetic test** `tests/test_eval_gate.py` (passing): export metrics on a temp
dump, replay-script alignment (incl. `run_python` needing `code`), regression
diff, floors, skipped-metric and corrupt-baseline handling.

### P3.3 — LoRA fine-tune + vLLM serve (needs your install) ✅ SCRIPTS DONE

- `scripts/finetune_lora.py`:
  - `--check-only` (hermetic, runs today): validates every export line against
    `TrainingExample.from_dict` + purity, requires ≥1 train/val example,
    flags `LLM call failed…` generations as curation warnings. Exit 0/1.
  - Full path (blocked until `pip install -e ".[train]"`): guarded import
    (one-line install hint instead of a traceback), re-runs `--check-only`
    and refuses bad data, then `trl.SFTTrainer` + `peft.LoraConfig` over
    `config/finetune/*.jsonl` → adapter in `config/lora_adapter/`.
    Training API pinned (`trl>=0.10,<0.12`) via the new `[train]` extra.
- `scripts/vllm_serve.py`: prints the `vllm serve --served-model-name`
  invocation + the `TC_LLM_BASE_URL` swap; `--probe` polls
  `GET /v1/models` (`app/serving.py`) until ready → exit 0/1.

```
make finetune-check       # hermetic schema validation (runs today)
make finetune             # real training; needs .[train]
make vllm-serve           # print serve cmd; --probe needs a running server
```

**Hermetic test** `tests/test_finetune_check.py` (passing): from_dict
round-trip/strictness, check-only exit codes, missing-dependency guard,
readiness probe against a local stub server + closed port.

### P3.4 — TRAINING tracker + docs ✅ DONE

- Makefile: `finetune`, `finetune-check`, `vllm-serve` (plus P3.1/P3.2's
  `lora-data`, `gate`).
- `docs/FINETUNE.md`: the whole flow with the hermetic path first, status
  table (what runs today vs. blocked on install), curation-hook note.
- README: structure tree, `make gate` usage, Phase 3 status line.

### P3.5 — CI/CD + drift monitoring + synthetic-data isolation ✅ DONE

- `.github/workflows/eval.yml`: lint → hermetic tests → index build →
  `run_evals.py --gate --no-rerank --baseline evals/gate_baseline.json`
  (fresh checkouts have no `evals/reports/` history, so CI diffs against a
  **committed pinned baseline**; the pinned file is never overwritten) →
  `drift_check.py`. Optional `LLM_API_KEY` secret; reranker disabled in CI.
- `scripts/drift_check.py` + `app/drift.py`: trend check vs the committed
  `evals/drift_baseline.json` — unfavourable moves only (direction-aware),
  WARN ≥ 10% / FAIL ≥ 25% relative bands, `--update-baseline`,
  `--strict`, junk-generation WARN floor; corrupt baseline = failure.
- **Synthetic data never trains** (found while wiring drift): every current
  `data/episodic` dump was fake — pytest's `MemoryManager()` defaulted to the
  real dirs (one hex dump per test run), conversation evals persisted
  `eval_conv*` with fake answers each gate, and the exports were stale
  `LLM call failed` rows from an earlier state. Fixes:
  `MemoryManager.end_conversation(persist=False)` for evals, isolated
  memory paths for tests/evals/e2e, `load_episodic_turns` skips
  `eval_*`/`test_id` dumps, `make lora-data` truncates exports when empty
  (exit 1), stale dumps + polluted `topics.json` purged.
  New reality: no real conversations exist yet (`TC_LLM_API_KEY` unset), so
  `--check-only` honestly fails until you run some.

```
make drift-check               # vs pinned evals/drift_baseline.json
python scripts/drift_check.py --update-baseline
```

**Hermetic test** `tests/test_drift.py` (13 tests, passing): direction-aware
findings, warn/fail bands, baseline round-trip, CLI exit codes (establish /
drift / corrupt / strict / update), synthetic-dump filtering, `persist=False`.


## Files touched

- New (done): `app/finetune/{__init__,schema,check}.py`, `app/fake_llm.py`,
  `app/eval_finetune.py`, `app/eval_gate.py`, `app/serving.py`, `app/drift.py`,
  `scripts/export_lora_data.py`, `scripts/finetune_lora.py`,
  `scripts/vllm_serve.py`, `scripts/drift_check.py`,
  `.github/workflows/eval.yml`, `docs/FINETUNE.md`,
  `tests/test_finetune_export.py`, `tests/test_eval_gate.py`,
  `tests/test_finetune_check.py`, `tests/test_drift.py`.
- Edit (done): `scripts/run_evals.py` (`--finetune`, `--gate`, `--baseline`),
  `scripts/e2e_{agent,sse}_check.py` (shared fake client + exit codes +
  isolated memory), `app/eval_conversation.py` (`persist=False`),
  `app/memory/manager.py` (`end_conversation(persist=...)`),
  `app/settings.py` (`episodic_dir`, `finetune_dir`,
  `faithfulness_threshold`), `Makefile` (`lora-data`, `gate`,
  `finetune{,-check}`, `vllm-serve`, `drift-check`),
  `pyproject.toml` (`[train]` extra), `README`, `docs/{PHASE3,EVALS,FINETUNE}.md`,
  `tests/{test_agent,test_agent_loop,test_finetune_check}.py` (isolation).

## Sequence

1. ✅ P3.1 export + dataset (hermetic tests) → `make lora-data`.
2. ✅ P3.2 gate runner + `--finetune` evals (hermetic, fake-LLM) → `make gate`.
3. ✅ P3.3 `finetune_lora.py --check-only` + `vllm_serve.py` smoke (no deps);
   real training/serving activates after `pip install -e ".[train]"` / `vllm`.
4. ✅ P3.4 docs + README — then run live once you set `TC_LLM_API_KEY`
   and install torch/vLLM/reranker.
5. ✅ P3.5 CI workflow + drift monitoring + synthetic-data isolation —
   then generate real conversations (key) so `make lora-data` produces
   trainable data, re-pin baselines, and train.

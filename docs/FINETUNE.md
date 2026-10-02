# Fine-Tuning (Phase 3)

How TutorCopilot turns Phase 2 agent logs into a LoRA adapter — and how the
whole pipeline is demo-able **today** with no model downloads, no API key, and
no `torch`-adjacent training stack.

The rule from Phases 1–2 holds: every step is hermetic and eval-gated. Anything
that needs a large install is behind a guarded command that fails with a one-line
hint instead of a traceback.

## Status

| Step | Command | Needs install? | Runs today? |
|---|---|---|---|
| 1. Generate **real** traces → episodic dumps | conversations with `TC_LLM_API_KEY` | no | ⛔ needs key |
| 2. Export → `config/finetune/{train,val}.jsonl` | `make lora-data` | no | ⛔ empty (no real data yet) |
| 3. Eval gate (retrieval + agent + conversation + export) | `make gate` | no | ✅ |
| 4. Validate export schema (`--check-only`) | `make finetune-check` | no | ⛔ fails until step 1–2 produce data |
| 5. Drift vs pinned baseline | `make drift-check` | no | ✅ |
| 6. LoRA fine-tune (`trl` + `peft`) | `make finetune` | `pip install -e ".[train]"` | ⛔ blocked |
| 7. Serve via vLLM + re-gate | `make vllm-serve` | `pip install vllm` | ⛔ blocked |

**Why steps 1/2/4 are empty:** `data/episodic/` used to be polluted by
synthetic runs (fake-LLM e2e/eval output and test state). The pipeline now
excludes every synthetic source — eval dumps are skipped, tests and eval
memory are isolated, and `make lora-data` truncates the exports when only
synthetic data exists instead of shipping stale files. The purge left zero
real conversations (no key has ever been set), so `--check-only` honestly
reports `needs >= 1 training example` until you run some real conversations.

## The flow

### 1–2. Traces → training data

Real agent runs (the API with `TC_LLM_API_KEY` set) persist episodic dumps
under `data/episodic/`; `make lora-data` loads them through the pure pipeline
in `app/finetune/schema.py` (purity gate → dedupe → seeded 80/20 split) and
writes:

```
config/finetune/train.jsonl
config/finetune/val.jsonl
```

Each line is a `TrainingExample`: `{question, generation, tool_calls[],
success, trace_id, conv_id}`.

Synthetic runs never land here: conversation-eval dumps (`eval_*` /
`metadata.test_id`) are skipped by the loader, evals/e2e/tests run against
isolated memory paths, and generations starting with `LLM call failed…` are
kept but flagged (`warn:` from `--check-only`) for curation.

### 3. The gate

```bash
make gate    # python scripts/run_evals.py --gate --no-rerank
```

Chains retrieval + generation, the fake-LLM agent check, conversation evals,
and the export/replay metrics, then diffs against
`evals/reports/gate_latest.json`. Non-zero exit on any regression or floor
violation (see `docs/EVALS.md` → *Phase 3 — Eval-Gated Checkpoint Gate*).

The export-specific signal is **`tool_faithfulness`**: a fake LLM replays every
conversation test through the *real* agent loop and tool registry with
schema-valid tool args, answering with exported generations. The best case is
`1.0` — a lower value means tool behaviour the training data assumes has
drifted.

### 4. `--check-only` (the smokeable half)

```bash
make finetune-check    # python scripts/finetune_lora.py --check-only
```

Validates every exported line against `TrainingExample.from_dict` (types,
required fields, purity) and requires ≥1 train and ≥1 val example. Hermetic —
no `torch`/`trl`/`peft` import happens. Exit 0 = trainable.

### 5. Drift monitoring

```bash
make drift-check                        # vs pinned evals/drift_baseline.json
python scripts/drift_check.py --update-baseline
```

Compares export + gate metrics against a committed baseline (WARN ≥ 10%,
FAIL ≥ 25% unfavourable relative move; `--strict` promotes warns). See
`docs/EVALS.md` → *Drift monitoring*. After a deliberate data change (like
the synthetic-purge), re-pin with `--update-baseline`.

### 6–7. Training + serving (blocked until installed)

```bash
pip install -e ".[train]"   # trl/peft/transformers/datasets — large
make finetune               # LoRA SFT -> config/lora_adapter/
pip install vllm            # large
make vllm-serve             # prints the serve command + TC_LLM_BASE_URL swap
```

`scripts/finetune_lora.py` re-runs `--check-only` and refuses bad data before
importing anything heavy, then trains `Qwen/Qwen2.5-0.5B-Instruct` (override
with `--model`) with `peft.LoraConfig(r=8)` via `trl.SFTTrainer`. The training
API is pinned (`trl>=0.10,<0.12`) so the `SFTConfig` kwargs stay valid.

Point the app at the fine-tuned model and re-gate:

```bash
# .env
TC_LLM_BASE_URL=http://localhost:9000/v1
make gate                  # the candidate must not regress the baseline
```

## Configuration

| Setting | Default | Purpose |
|---|---|---|
| `TC_EPISODIC_DIR` | `data/episodic` | Where agent runs persist conversation dumps |
| `TC_FINETUNE_DIR` | `config/finetune` | Export target (`train.jsonl` / `val.jsonl`) |
| `TC_FAITHFULNESS_THRESHOLD` | `3` | Gate floor for judge-rated faithfulness (1–5) |

## Tests

`tests/test_finetune_export.py`, `tests/test_finetune_check.py`,
`tests/test_eval_gate.py`, and `tests/test_drift.py` cover the pipeline
hermetically: schema purity, synthetic-dump exclusion, `--check-only` exit
codes, the missing-dependency guard, gate diff/floor logic, drift bands and
CLI exit codes, and the readiness probe against a local stub server.

.PHONY: setup index build eval eval-conv agent sse serve-e2e lora-data gate finetune finetune-check vllm-serve drift-check test lint clean

setup:
	python -m venv .venv
	.venv/Scripts/python -m pip install -e ".[dev]"

index:
	python scripts/index_docs.py

build: index

eval:
	python scripts/run_evals.py

# Phase 2: end-to-end agent check (uses fake LLM, no API key needed)
agent:
	python scripts/e2e_agent_check.py

# Phase 2: Conversation / agent-loop evals
eval-conv:
	python scripts/run_evals.py --conversation

# Phase 2: SSE streaming check
sse:
	python scripts/e2e_sse_check.py

serve-e2e: agent sse

# Phase 3: export episodic traces -> LoRA train/val JSONL (hermetic)
lora-data:
	python scripts/export_lora_data.py

# Phase 3: eval-gated CI gate (retrieval + agent + conversation + finetune).
# --no-rerank: the cross-encoder reranker download hangs in offline environments.
gate:
	python scripts/run_evals.py --gate --no-rerank

# Phase 3: validate LoRA exports (hermetic, no ML deps)
finetune-check:
	python scripts/finetune_lora.py --check-only

# Phase 3: full LoRA fine-tune (requires: pip install -e ".[train]")
finetune:
	python scripts/finetune_lora.py

# Phase 3: print the vLLM serve command + TC_LLM_BASE_URL swap
vllm-serve:
	python scripts/vllm_serve.py

# Phase 3: drift check vs the pinned evals/drift_baseline.json
# (WARN-level by default; --strict exits 1 on warnings too)
drift-check:
	python scripts/drift_check.py

serve:
	uvicorn app.main:app --reload --port 8000

test:
	pytest

lint:
	ruff check app scripts tests

clean:
	rm -rf .venv config/saved_index traces evals/reports
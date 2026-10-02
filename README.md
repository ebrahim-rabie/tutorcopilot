# TutorCopilot — Eval-Driven RAG → Production Agent

An **eval-first** RAG + **agent** system for AI-engineer portfolio. Phase 1 (retrieval/generation) and Phase 2 (agent loop, tools, memory, SSE streaming, guardrails) are implemented and tested.

> **Phase 1**: Eval-driven RAG with hybrid retrieval (BM25 + dense + RRF), reranking, citations, judge evals, and full traceability.
> **Phase 2**: ReAct **agent loop** (Plan → Act → Observe) with tool registry (search_knowledge, run_python, fetch_url, save_note, quiz_me), short/long-term **memory**, SSE **streaming**, OpenTelemetry traces, and guardrails + rate limiting.
> **Phase 3** (next): Fine-tuning (LoRA) on real conversation logs, served with vLLM, plus eval-gated CI/CD.

---

## Why This Project?

Every AI engineering interview asks:
- "How do you measure RAG quality?" → **Golden dataset + hit-rate/MRR + LLM-as-judge**
- "How do you debug a bad answer?" → **Full traces: query → retrieval scores → rerank → prompt → LLM output**
- "How do you improve without regressing?" → **Eval diff on every change, gated in CI**
- "Have you fine-tuned a model?" → **Phase 3: LoRA on your own conversation logs, served with vLLM**

This project answers all of them with working code.

---

## Quickstart

### 1. Prerequisites
- Python 3.11+
- Git
- (Optional) OpenAI-compatible API key for generation/evals

### 2. Setup
```bash
git clone <this-repo>
cd tutorcopilot
python -m venv .venv
.venv/Scripts/activate   # Windows
# source .venv/bin/activate  # Linux/macOS
pip install -e ".[dev]"
cp .env.example .env
# Edit .env and add TC_LLM_API_KEY if you want generation/evals
```

### 3. Build the Index
```bash
make index
# or: python scripts/index_docs.py
```
Downloads `all-MiniLM-L6-v2` (~80 MB) on first run, embeds 6 knowledge docs (~50 chunks).

### 4. Run Evaluations
```bash
make eval
# or: python scripts/run_evals.py
```
**Without LLM key**: runs retrieval evals only (hit-rate, MRR).
**With LLM key**: also runs generation evals (faithfulness, relevance via LLM judge).

Phase 3 eval gate (retrieval + agent + conversation + finetune export, fails on
regression vs the previous run — exits non-zero):
```bash
make gate
# or: python scripts/run_evals.py --gate --no-rerank
```

Expected retrieval baseline (markdown chunking + hybrid + rerank):
```
n=18 MRR≈0.62 hit@3≈0.72 hit@5≈0.83
```

### 5. Start the API
```bash
make serve
# or: uvicorn app.main:app --reload --port 8000
```
Open http://localhost:8000 — the chat UI streams agent steps live (thoughts,
tool calls, nudges) with a sidebar for health, weak topics, and conversation
state. The composer has a **mic** for dictation (browser speech recognition,
falling back to Groq Whisper via `POST /agent/transcribe`) and a voice toggle
that **speaks answers** (browser TTS or server edge-tts via `POST /agent/speak`).
`scripts/try_tutor.ps1` walks every REST/SSE endpoint from the CLI.

### 6. Try It
```bash
curl -X POST http://localhost:8000/ask \
  -H "Content-Type: application/json" \
  -d '{"question": "What is RRF and what k value is typical?"}'
```

Response:
```json
{
  "question": "What is RRF and what k value is typical?",
  "answer": "Reciprocal Rank Fusion (RRF) merges ranked lists using score = Σ 1/(k+rank). The typical k value is 60. [1]",
  "citations": [{"index": 1, "source": "hybrid-retrieval.md", "section": "RRF Formula", ...}],
  "confidence": 0.95,
  "latency_ms": 420,
  "usage": {"prompt_tokens": 180, "completion_tokens": 45},
  "model": "gpt-4o-mini",
  "trace_id": "a1b2c3d4e5f6",
  "retrieval_path": [...],
  "top_chunks": [...]
}
```

### 7. Inspect a Trace
```bash
curl http://localhost:8000/traces/a1b2c3d4e5f6
```
Shows every step with timings and intermediate results.

---

## Project Structure

```
tutorcopilot/
├── app/                    # Core package
│   ├── agent/              # Phase 2: agent loop (plan -> act -> observe)
│   ├── chunking.py         # Recursive / Markdown / Semantic chunkers
│   ├── embedding.py        # Local (ST) / API / Hash embedders
│   ├── eval_conversation.py # Phase 2: multi-turn conversation harness
│   ├── eval_finetune.py    # Phase 3: export metrics + tool_faithfulness replay
│   ├── eval_gate.py        # Phase 3: gate diff/floor logic (pure)
│   ├── eval_harness.py     # Retrieval + generation evals with diffs
│   ├── drift.py            # Phase 3.5: drift bands vs pinned baseline (pure)
│   ├── fake_llm.py         # Shared scripted fake LLM client (hermetic)
│   ├── finetune/           # Phase 3: schema, export pipeline, --check-only
│   ├── generate.py         # Generator (LLM) + Judge (LLM-as-judge)
│   ├── index.py            # Hybrid search (BM25+dense+RRF), rerank, persistence
│   ├── main.py             # FastAPI: /health, /ask, /agent/*, SSE, SPA at /
│   ├── memory/             # Phase 2: conversation memory + episodic dumps
│   ├── pipeline.py         # answer_question() orchestration
│   ├── rerank.py           # CrossEncoderReranker
│   ├── serving.py          # Phase 3: OpenAI-compatible readiness probe
│   ├── settings.py         # Pydantic Settings (TC_* env vars)
│   ├── static/             # Chat UI (HTML/CSS/JS) served at /
│   ├── streaming/          # Phase 2: SSE streaming
│   ├── tools/              # Phase 2: tool registry (search, python, ...)
│   └── tracing.py          # TraceRecorder + JSONL trace files
├── config/                 # Generated (gitignored)
│   ├── saved_index/        # chunks.json, matrix.npy, corpus_tokens.json, meta.json
│   └── finetune/           # Phase 3: train.jsonl / val.jsonl exports
├── data/
│   ├── episodic/           # Phase 2/3: conversation dumps (export input)
│   └── sources/            # 6 knowledge markdown files (the "corpus")
├── docs/
│   ├── DESIGN.md           # Architecture + data flow
│   ├── EVALS.md            # Evaluation methodology + gate contract
│   ├── FINETUNE.md         # Phase 3: fine-tune + serve flow
│   └── PHASE3.md           # Phase 3 roadmap (status tracked here)
├── evals/
│   ├── golden_queries.jsonl    # 18 curated Q/A with source mapping
│   ├── conversation_tests.jsonl # Phase 2: multi-turn golden set
│   ├── gate_baseline.json      # Pinned CI gate baseline (committed)
│   ├── drift_baseline.json     # Pinned drift baseline (committed)
│   └── reports/                # Timestamped eval JSON + gate_latest.json
├── scripts/
│   ├── index_docs.py       # Build + save index
│   ├── run_evals.py        # Eval CLI (--conversation --finetune --gate)
│   ├── e2e_agent_check.py  # Phase 2: fake-LLM agent loop (exit-coded)
│   ├── e2e_sse_check.py    # Phase 2: SSE round-trip (exit-coded)
│   ├── export_lora_data.py # Phase 3: traces -> train/val JSONL
│   ├── finetune_lora.py    # Phase 3: LoRA train + --check-only
│   ├── vllm_serve.py       # Phase 3: print serve cmd / readiness probe
│   ├── drift_check.py      # Phase 3.5: drift vs pinned baseline
│   └── try_tutor.ps1       # Endpoint walkthrough (health/ask/agent/SSE/traces)
├── .github/workflows/eval.yml # CI: lint + tests + gate + drift
├── tests/                  # Pytest suite (hermetic, no downloads)
│   ├── conftest.py
│   ├── helpers.py          # HashEmbedder, FakeGenerator
│   └── test_*.py           # chunking, search, pipeline, agent, finetune, gate...
├── traces/                 # Generated (gitignored) — per-request JSON traces
├── .env.example            # Template for TC_* settings
├── .gitignore
├── Dockerfile
├── docker-compose.yml
├── Makefile
├── pyproject.toml
├── README.md
└── requirements.txt
```

---

## Configuration (`.env`)

| Variable | Default | Description |
|---|---|---|
| `TC_LLM_API_KEY` | *empty* | OpenAI-compatible key. Empty = retrieval-only mode. |
| `TC_LLM_BASE_URL` | `https://api.openai.com/v1` | Override for Ollama, vLLM, etc. |
| `TC_LLM_MODEL` | `gpt-4o-mini` | Generation model. |
| `TC_LLM_JUDGE_MODEL` | *same as model* | Separate judge model (optional). |
| `TC_EMBEDDING_MODE` | `local` | `local` (sentence-transformers) or `api` (OpenAI-compatible). |
| `TC_EMBEDDING_MODEL` | `all-MiniLM-L6-v2` | Local ST model name. |
| `TC_USE_RERANK` | `true` | Enable cross-encoder reranking. |
| `TC_RERANKER_MODEL` | `cross-encoder/ms-marco-MiniLM-L-6-v2` | Reranker model. |
| `TC_CHUNK_STRATEGY` | `markdown` | `markdown` \| `recursive` \| `semantic`. |
| `TC_CHUNK_SIZE` | `500` | Target chars per chunk. |
| `TC_CHUNK_OVERLAP` | `80` | Overlap chars between adjacent chunks. |
| `TC_HYBRID_TOP_K` | `12` | Candidates from each retriever before RRF. |
| `TC_FINAL_TOP_K` | `4` | Final contexts sent to LLM. |
| `TC_RRF_K` | `60` | RRF constant. |

---

## Running Tests

```bash
pytest -q
```
Tests are fully hermetic — use `HashEmbedder` and `FakeGenerator`, no network calls.

---

## Evaluation-Driven Development

Every change to retrieval/generation should be validated:

```bash
# 1. Baseline
make eval   # saves evals/reports/latest.json

# 2. Make ONE change (e.g., chunk_size=400 in .env)

# 3. Re-index + re-eval
make index && make eval

# 3. Check diff in output or evals/reports/latest.json["diff_vs_previous"]
```

Before merging an agent/tooling change, run the Phase 3 gate — it chains the
retrieval, agent, conversation, and finetune-export evals and blocks on any
regression vs `evals/reports/gate_latest.json`:

```bash
make gate   # or: python scripts/run_evals.py --gate --no-rerank
make drift-check   # trend bands vs pinned evals/drift_baseline.json
```

Log improvements in `docs/EVALS.md` with before/after numbers.

---

## Docker

```bash
docker compose up --build
# API on http://localhost:8000
```
Mounts `./traces` for persistence.

---

## Roadmap (Phases 2 & 3)

| Phase | Focus | Key Deliverables |
|---|---|---|
| **2** | Agent | Tool loop, memory, streaming UI, OpenTelemetry traces, guardrails |
| **3** | MLOps | LoRA fine-tune on Phase 2 logs, vLLM serving, eval-gated CI/CD, drift monitoring |

Phase 3 status: P3.1 export ✅ · P3.2 eval gate (`make gate`) ✅ · P3.3
`finetune_lora.py`/`vllm_serve.py` ✅ (actual training/serving blocked on
`pip install -e ".[train]"` / `pip install vllm`) · P3.4 docs ✅ ·
P3.5 CI + drift monitoring (`make drift-check`) + synthetic-data isolation ✅ —
see `docs/PHASE3.md` and `docs/FINETUNE.md`. Training data is intentionally
empty until real conversations exist (set `TC_LLM_API_KEY` and run some).

The corpus (`data/sources/`) is AI engineering knowledge — you learn the material while building the system that teaches it.

---

## License

MIT — use freely for learning, portfolio, or production.
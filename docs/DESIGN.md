# TutorCopilot — Architecture Overview (Phase 1)

## High-Level Flow
```
User Question
     │
     ▼
/ask endpoint  ──▶  TraceRecorder (new trace_id)
     │
     ▼
HybridRetriever  (BM25 + dense + RRF)
     │
     ├──▶ Optional: CrossEncoderReranker (top 12 → top 4)
     │
     ▼
Generator (OpenAI-compatible LLM, structured JSON output)
     │
     ▼
Response: {answer, citations[], confidence, trace_id}
     │
     ▼
TraceRecorder.save()  (writes traces/{trace_id}.json)
```

## Components

| Module | Responsibility |
|---|---|
| `settings.py` | Pydantic settings from `.env` + env vars (`TC_*` prefix) |
| `tracing.py` | `Trace` event log + `load_trace()` for `/traces/{id}` |
| `embedding.py` | `LocalEmbedder` (sentence-transformers), `ApiEmbedder`, `HashEmbedder` (tests) |
| `chunking.py` | `RecursiveChunker`, `MarkdownChunker`, `SemanticChunker` |
| `index.py` | `Index` — build, hybrid search (RRF), rerank, save/load |
| `rerank.py` | `CrossEncoderReranker` thin wrapper |
| `generate.py` | `Generator` (LLM call + JSON parse + citation validation), `Judge` (LLM-as-judge) |
| `pipeline.py` | `answer_question()` orchestrates retrieval + generation + tracing |
| `eval_harness.py` | `EvalHarness` runs retrieval + generation evals, saves JSON reports with diffs |
| `main.py` | FastAPI app, deps, `/health`, `/ask`, `/traces/{id}` |

## Data Flow Details

### Index Build (`scripts/index_docs.py`)
1. Load all `*.md`, `*.txt` from `data/sources/`
2. Chunk via configured strategy (default: markdown-aware)
3. Embed chunks with `LocalEmbedder` (all-MiniLM-L6-v2)
4. Build BM25 index (`rank_bm25.BM25Okapi`)
5. Persist to `config/saved_index/`:
   - `chunks.json` — list of Chunk dicts
   - `matrix.npy` — (N_chunks, dim) float32 dense vectors
   - `corpus_tokens.json` — tokenized texts for BM25 rebuild
   - `meta.json` — config snapshot for mismatch detection

### `/ask` Request
1. `Trace` created, logs `hybrid_search` event
2. `Index.hybrid_search(query, top_k=12)` → RRF fused candidates
3. If `use_rerank`: `CrossEncoderReranker.predict([(q, chunk)...])` → top 4
4. Build prompt with numbered context blocks
5. `Generator.generate()` → LLM call, `response_format=json_object`
6. Parse JSON, validate citation indices exist in context
7. Log `generate` event, save trace
8. Return `AskResponse` with answer, citations, retrieval_path, trace_id

### Eval Harness (`scripts/run_evals.py`)
1. Load `evals/golden_queries.jsonl`
2. **Retrieval eval**: hit-rate@1/3/5, MRR against `source_file`
3. **Generation eval** (if LLM configured):
   - Run `answer_question` per query
   - `Judge.rate_grounded()` on answer vs. retrieved context → faithfulness 1-5, relevance 1-5
4. Save timestamped report + `latest.json` with diff vs previous run

## Configuration (via `.env`)
```
TC_LLM_API_KEY=               # required for generation/evals
TC_LLM_MODEL=gpt-4o-mini
TC_EMBEDDING_MODE=local
TC_EMBEDDING_MODEL=all-MiniLM-L6-v2
TC_USE_RERANK=true
TC_CHUNK_STRATEGY=markdown
TC_HYBRID_TOP_K=12
TC_FINAL_TOP_K=4
```

## Extensibility Points
- **Embedder**: swap `LocalEmbedder` → `ApiEmbedder` via `TC_EMBEDDING_MODE=api`
- **Chunker**: `TC_CHUNK_STRATEGY=recursive|semantic`
- **Reranker**: `TC_USE_RERANK=false` disables; `TC_RERANKER_MODEL` for alt model
- **LLM**: `TC_LLM_BASE_URL` for Ollama/vLLM/OpenAI-compatible endpoints
- **Eval Judge**: `TC_LLM_JUDGE_MODEL` to use a different model for scoring
---

# Phase 2 � Tool-Using Agent (Production)

Phase 2 turns the Phase 1 RAG into a **ReAct agent**: it can *plan ? act ? observe*, call tools, remember what you've already discussed, and stream results back. Everything is traced end-to-end so you can debug and build Phase 3 fine-tuning data.

## Agent Loop

```
User Question
   �
   ?
build_system_prompt (tool schemas)
   �
   ?
+---------- PLAN --------------+
� LLM returns JSON: thought / tool / args / final_answer     �
+------------------------------+
             � (tool, args)
             ?
+---------- ACT ---------------+
� registry.get(tool).execute() -> ToolResult               �
+------------------------------+
             � (result)
             ?
+---------- OBSERVE -----------+
� append result to context ? loop back to PLAN             �
+------------------------------+
        � final_answer reached / cycle detected / max iterations
        ?
AgentResult { answer, steps[], tools_called, success, tokens }
```

Guards:
- **Max iterations** (`TC_AGENT_MAX_ITERATIONS`, default 6) + cycle detection (same tool called 3+ times ? stop).
- **Tool arg validation** via JSON schema before execution.
- **Faithfulness guardrail** (Phase 2.4): `GuardrailManager` checks the final answer is grounded in retrieved context.
- **Rate limiting** (per-IP token bucket).

## Tools (`app/tools/registry.py`)

| Tool | Purpose |
|---|---|
| `search_knowledge` | Hybrid BM25+dense+RRF search over the Phase 1 index. |
| `run_python` | Execute code (`python -c` subprocess, timeout+no-network sandbox). |
| `fetch_url` | Fetch a URL ? text extraction via BeautifulSoup (read docs). |
| `save_note` | Write a `.md` note into the user's knowledge base. |
| `quiz_me` | Generate a quiz on a weak topic tracked in long-term memory. |

## Memory (`app/memory/manager.py`)

Three levels:
1. **Short-term** per-turn history (system prompt injection).
2. **Long-term** topic weakness tracking � update when a turn fails; `get_weakest_topic()`.
3. **Episodic** conversation dumps persisted to disk for Phase 3 fine-tuning + trace replays.

## Streaming (`app/streaming/sse.py`)

`POST /agent/stream` returns **Server-Sent Events** (`text/event-stream`):
`step_start ? tool_call ? tool_result ? ... ? final_answer`. The frontend can render a live "thought ? tool ? answer" transcript.

## Observability (`app/observability/otel.py`)

- Optional OpenTelemetry (no-op if not installed) � `traced()` decorator + `llm_span / tool_span / db_span` helpers + RRF/DB markers.
- Every loop logs a JSONL `Trace` (same `trace_dir` as Phase 1) ? `scripts/view_trace.py`.

## Route Changes

- `PHASE2_AVAILABLE` gate set when `app/agent` imports succeed.
- New dependency provider `get_tool_registry_dep` wired into `app.state`.
- Health now reports `phase2`, `llm_configured`, `index_chunks`, and `phase` (2 vs 1).

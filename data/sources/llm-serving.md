# LLM Serving: vLLM, Batching & Cost

## Why vLLM?
- **PagedAttention**: manages KV cache in non-contiguous blocks → near-zero fragmentation, supports continuous batching.
- **Continuous batching**: new requests join running batches without waiting for full batch completion → higher throughput, lower tail latency.
- **OpenAI-compatible API**: drop-in replacement for `openai.ChatCompletion`.

## Key Config Knobs
| Parameter | Effect | Typical |
|---|---|---|
| `max_num_batched_tokens` | Total tokens (prompt+gen) per forward pass | 4096–8192 |
| `max_num_seqs` | Max concurrent sequences | 256–1024 |
| `gpu_memory_utilization` | Fraction of VRAM for KV cache | 0.85–0.95 |
| `enforce_eager` | Disable CUDA graph (debug) | false |

## Deployment Patterns
### Single Model, Multi-LoRA
Load base + multiple LoRA adapters; serve different personas from one GPU.
```python
# vLLM OpenAI server with LoRA
vllm serve meta-llama/Llama-3.1-8B \
    --lora-modules support_copilot=./adapters/support_copilot \
    --max-lora-rank 16
```

### Quantization
- **AWQ / GPTQ**: 4-bit weights, FP16 activations; requires calibration data.
- **BitsAndBytes 8-bit / 4-bit**: Dynamic quantization, no calibration; small quality hit.
- **GGUF**: llama.cpp format; CPU + GPU offload; best for edge.

## Cost Modeling
| Metric | Formula |
|---|---|
| **Per-request cost** | `(prompt_tokens + completion_tokens) * $/1M` |
| **GPU-hour cost** | Cloud GPU $/hr / requests_per_hour |
| **Break-even** | If self-hosted GPU < API cost at your volume |

Example: A100 80GB @ $1.50/hr serves ~500 req/min (7B, 512 out tokens) → $0.0003/req vs. OpenAI $0.0015/req.

## Observability
- **Request logs**: prompt_hash, latency_p50/p99, tokens, finish_reason.
- **Queue depth**: time waiting for batch slot (should be < 50ms at healthy load).
- **KV cache usage**: % of `gpu_memory_utilization`; if > 95%, increase VRAM or reduce `max_num_seqs`.

## Scaling
- **Horizontal**: multiple replicas behind load balancer; vLLM has built-in prefix caching for shared prompts.
- **Model parallel**: tensor parallelism across 2+ GPUs for models > single GPU VRAM.

## Common Issues
- **OOM**: Reduce `max_num_seqs` or `max_num_batched_tokens`; enable `gpu_memory_utilization=0.85`.
- **Slow first token**: Warm up with dummy requests on startup; CUDA graph capture on first batch.
- **Prefix cache misses**: Ensure system prompt is identical across requests; use `enable_prefix_caching=True`.
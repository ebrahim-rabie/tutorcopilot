# Fine-Tuning Small Models with LoRA

## When to Fine-Tune
- You have **domain-specific data** (your product's conversation logs, proprietary docs) that base models handle poorly.
- You need **consistent style/format** (e.g., always cite sources, always output JSON).
- You want **lower latency/cost** by distilling a 70B behavior into a 7B model.

## LoRA (Low-Rank Adaptation)
Freezes base weights; trains small adapter matrices `A (r × d)` and `B (d × r)` where `r ≪ d` (typically 8–64).
- **Params trained**: ~0.1–1% of base model.
- **Storage**: MBs not GBs; multiple adapters can share base.
- **Mergeable**: `W + BA` can be fused into base for inference (no adapter overhead).

## Data Preparation
1. **Source**: Real user conversations + human-annotated corrections.
2. **Format**: ChatML / ShareGPT → `{"messages": [{"role": "user", "content": "..."}, {"role": "assistant", "content": "..."}]}`.
3. **Quality filters**:
   - Remove PII, toxic, hallucinated turns.
   - Deduplicate (exact + near-duplicate via MinHash).
   - Length filter: 50–4000 tokens per conversation.
4. **Splits**: 90/5/5 train/val/test; ensure no user overlap across splits.

## Training Config (7B, 1×A100/A6000)
```yaml
model: meta-llama/Llama-3.1-8B
lora_r: 16
lora_alpha: 32
lora_dropout: 0.05
target_modules: ["q_proj", "v_proj", "k_proj", "o_proj"]
learning_rate: 2e-4
num_train_epochs: 3
per_device_train_batch_size: 4
gradient_accumulation_steps: 4
lr_scheduler: cosine
warmup_ratio: 0.03
bf16: true
max_seq_length: 4096
```

## Evaluation During Training
- **Perplexity on val**: Early-stop if no improvement for 2 evals.
- **Task-specific eval**: Run your golden queries through the fine-tuned model (via generation evals).
- **Regression check**: Compare base vs. fine-tuned on *general* benchmarks (MMLU, GSM8K) to ensure no catastrophic forgetting.

## Quantization & Serving
| Method | Size (7B) | Quality Loss | Speedup |
|---|---|---|---|
| FP16 | 14 GB | baseline | 1× |
| AWQ 4-bit | 4.5 GB | ~1–2% | 2–3× |
| GGUF Q4_K_M | 4.2 GB | ~1% | 2–3× (llama.cpp) |

**Serving**: vLLM (PagedAttention) for throughput; llama.cpp for CPU/edge.

## Eval-Gated Deployment
CI pipeline:
```
train → eval on golden set → if faithfulness ≥ baseline AND latency ≤ budget → promote to staging
```
No human approval needed if metrics pass. Rollback is `docker pull previous-tag`.
# Prompt Engineering for Grounded RAG

## System Prompt Anatomy
```
Role: You are X.
Rules:
1. Only use provided context.
2. Cite with [n] tags.
3. Say "I don't know" if context lacks answer.
Format: JSON with keys {answer, citations, confidence}
```

### Why This Works
- **Role** sets behavior frame.
- **Rules** are negative constraints (what NOT to do) — LLMs follow "don't" better than "do".
- **Citation requirement** forces grounding; the model must map claims to context indices.
- **Structured output** enables programmatic validation.

## Context Construction
```
Context:
[1] chunk text from source A
[2] chunk text from source B
[3] chunk text from source A (different section)
```
- Numbered blocks, 1-based.
- Include metadata (source, section) only in the citation map returned to the app, not in the prompt (saves tokens).

## Citation Validation
After generation, verify every cited index exists in the provided context. Reject or flag answers citing `[5]` when only 3 blocks were provided.

## Prompt Variants to Test
| Variant | Change | Expected Effect |
|---|---|---|
| Baseline | Standard rules | Reference |
| + "Be concise" | Add length constraint | Shorter answers, possibly less detail |
| + Chain-of-thought | "Think step by step" | Better reasoning, more tokens |
| Few-shot | 2–3 examples in context | Better format adherence |
| Negative examples | Show bad citations + corrections | Fewer hallucinated citations |

## Structured Output
Use `response_format={"type": "json_object"}` (OpenAI) or `{"type": "json_schema", ...}` for strict schema. Always parse defensively:
```python
def parse_generation(text):
    try: return json.loads(text)
    except: extract between first { and last }
```

## Temperature
- **RAG answer**: 0.0–0.3 (deterministic, grounded).
- **Judge**: 0.0 (consistent scoring).
- **Creative/rewrite**: 0.7+ (not used in this pipeline).

## Prompt Versioning
Store prompts as files (`prompts/v1_answer.txt`, `prompts/v2_answer.txt`). Git-track. Every eval run records prompt version. When a prompt change improves evals, commit with diff.

## Common Failures
| Symptom | Likely Cause | Fix |
|---|---|---|
| Hallucinated citations | Model invents numbers > context size | Validate + retry with "only cite 1–N" |
| No citations at all | Prompt didn't emphasize format | Add "MUST include citations array" |
| Low confidence on correct answers | Model unsure | Lower temperature; add "confidence: 1.0 if certain" |
| JSON parse errors | Model adds prose | `response_format: json_object` + strip fences |
# Evaluation for RAG & Agent Systems

## Golden Datasets
A golden dataset is a curated set of (question, reference_answer, expected_source) triples.
- **Size**: 50–200 queries for initial development; 500+ for production gates.
- **Difficulty tags**: easy / medium / hard to track per-slice performance.
- **Source traceability**: every query maps to at least one source document so retrieval can be measured.

## Retrieval Metrics
| Metric | Definition | When to Use |
|---|---|---|
| Hit-rate@K | Fraction of queries where expected doc is in top-K | Quick sanity check |
| Recall@K | Fraction of *all relevant* docs found in top-K | Multi-relevant scenarios |
| MRR | Mean Reciprocal Rank of first relevant doc | Ranking quality |

## Generation Metrics
Two orthogonal axes:

### Faithfulness (Groundedness)
Does every claim in the answer have support in the retrieved context?
- **LLM-as-judge**: Prompt a capable model to score 1–5 with reasons.
- **Automated entailment**: NLI models (e.g., DeBERTa-v3) classify (context → answer) as entail/neutral/contradiction.
- **Rule of thumb**: If answer says "context doesn't contain X", that's *not* a faithfulness failure.

### Relevance (Answer Quality)
Does the answer actually address the user's question?
- **Reference-based**: Compare answer to gold answer via LLM judge or BERTScore.
- **Reference-free**: Judge rates "does this answer the question?" without gold.

## LLM-as-Judge Best Practices
1. **Binary vs. Likert**: 1–5 Likert gives more signal than binary pass/fail.
2. **Calibration**: Run 20 human labels first; check judge correlation (Spearman > 0.7).
3. **Context inclusion**: Always provide the retrieved chunks to the judge for faithfulness.
4. **Chain-of-thought prompts**: "Explain your reasoning, then output JSON with scores" reduces noise.

## Eval-Driven Development Loop
1. Build baseline → run evals → record numbers.
2. Change ONE thing (chunking, embedder, reranker, prompt).
3. Run evals → diff vs. baseline.
4. If delta > threshold (e.g., MRR +0.02), keep; else revert.
5. Repeat. Every change is a commit with eval diff.

## Common Anti-Patterns
- **No golden set** → vibe-based tuning, no regression detection.
- **Only end-to-end evals** → can't isolate retrieval vs. generation failures.
- **Static eval set** → overfit to the test queries; refresh quarterly with real traffic samples.
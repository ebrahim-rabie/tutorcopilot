# Hybrid Retrieval: BM25 + Dense + RRF

## Overview
Pure vector search misses exact keywords. Pure BM25 misses semantic matches. Hybrid retrieval fuses both for better recall and precision.

## Why Hybrid?
- **Dense embeddings** capture semantic similarity (e.g., "car" ≈ "automobile") but struggle with rare terms, codes, acronyms.
- **BM25** excels at exact lexical overlap (product IDs, error codes, specific names) but ignores synonymy.
- **Reciprocal Rank Fusion (RRF)** merges ranked lists without needing score calibration.

## RRF Formula
For each document appearing in multiple rankers:
```
score = Σ 1 / (k + rank_i)
```
Typical `k = 60`. Higher `k` dampens influence of top ranks; lower `k` makes fusion more aggressive.

## Implementation Notes
1. Run dense and sparse searches independently, collect top-N from each (e.g., 50 each).
2. Apply RRF on combined candidate set.
3. Take top-K fused results for reranking.
4. Cross-encoder reranker on final candidates corrects false positives from either retriever.

## When to Use Which
| Scenario | Recommended |
|---|---|
| Code/API docs with exact symbols | BM25-heavy (or pure BM25) |
| General QA over prose | Hybrid RRF |
| High-recall legal/medical search | Hybrid + rerank |
| Low-resource languages | BM25 or hybrid with multilingual embeddings |

## Common Pitfalls
- **Unnormalized dense scores**: dot product on non-unit vectors distorts RRF. Always normalize embeddings.
- **Mismatched candidate pools**: if dense returns 100 and BM25 returns 5, RRF is skewed. Use same `candidate_count` per source.
- **k too small**: k < 20 makes fusion brittle; k > 100 dampens useful signal. 60 is the empirically sweet spot.
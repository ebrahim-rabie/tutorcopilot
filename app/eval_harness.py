from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

from .generate import Generator, Judge, retry_transient
from .index import Index
from .pipeline import answer_question
from .settings import Settings

METRIC_KEYS = [
    "hit_rate_at_1",
    "hit_rate_at_3",
    "hit_rate_at_5",
    "mrr",
    "mean_faithfulness",
    "mean_relevance",
]


@dataclass
class GoldenQuery:
    id: str
    question: str
    reference_answer: str
    source_file: str
    difficulty: str = "medium"

    @classmethod
    def from_dict(cls, data: dict) -> GoldenQuery:
        return cls(
            id=str(data["id"]),
            question=str(data["question"]),
            reference_answer=str(data.get("reference_answer", "")),
            source_file=str(data.get("source_file", "")),
            difficulty=str(data.get("difficulty", "medium")),
        )


def load_golden_queries(path: str | Path) -> list[GoldenQuery]:
    queries: list[GoldenQuery] = []
    for raw_line in Path(path).read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if line and not line.startswith("#"):
            queries.append(GoldenQuery.from_dict(json.loads(line)))
    return queries


def run_retrieval_evals(index: Index, queries: list[GoldenQuery], top_k: int = 5) -> dict:
    """Hit-rate / MRR against the expected source file per golden query."""
    hits_at = {k: 0 for k in (1, 2, 3, 5)}
    n = 0
    rr_sum = 0.0
    per_query: list[dict] = []
    for query in queries:
        if not query.source_file:
            continue
        n += 1
        results = index.hybrid_search(query.question, top_k=max(top_k, 5))
        sources = [r.chunk.source for r in results]
        hit = False
        rr = 0.0
        for rank, src in enumerate(sources, start=1):
            if src == query.source_file and not hit:
                hit = True
                rr = 1.0 / rank
        rr_sum += rr
        for k in hits_at:
            if any(src == query.source_file for src in sources[:k]):
                hits_at[k] += 1
        per_query.append(
            {
                "id": query.id,
                "question": query.question,
                "expected_source": query.source_file,
                "retrieved_sources": sources[:top_k],
                "hit": hit,
                "rr": round(rr, 4),
            }
        )
    return {
        "n_queries": n,
        "hit_rate_at_1": round(hits_at[1] / n, 4) if n else 0.0,
        "hit_rate_at_2": round(hits_at[2] / n, 4) if n else 0.0,
        "hit_rate_at_3": round(hits_at[3] / n, 4) if n else 0.0,
        "hit_rate_at_5": round(hits_at[5] / n, 4) if n else 0.0,
        "mrr": round(rr_sum / n, 4) if n else 0.0,
        "per_query": per_query,
    }


def run_generation_evals(
    index: Index,
    generator: Generator,
    judge: Judge,
    queries: list[GoldenQuery],
    settings: Settings,
) -> dict:
    if not generator.available():
        return {
            "n_queries": 0,
            "skipped": True,
            "reason": "LLM not configured (set TC_LLM_API_KEY)",
        }

    rows: list[dict] = []
    n_failed = 0
    for query in queries:
        if not query.reference_answer:
            continue
        try:
            result = retry_transient(
                lambda q=query: answer_question(q.question, index, generator, settings)
            )
            verdict: dict = {}
            if judge.available():
                context_text = "\n".join(
                    [f"[{i + 1}] {c['text']}" for i, c in enumerate(result.top_chunks)]
                )
                verdict = retry_transient(
                    lambda q=query, r=result, c=context_text: judge.rate_grounded(
                        q.question, r.answer, c
                    )
                )
        except Exception:
            n_failed += 1  # transient upstream failure after retries — record, continue
            continue
        rows.append(
            {
                "id": query.id,
                "question": query.question,
                "reference_answer": query.reference_answer,
                "answer": result.answer,
                "citations": result.citations,
                "confidence": result.confidence,
                "faithfulness": verdict.get("faithfulness"),
                "faithfulness_reason": verdict.get("faithfulness_reason", ""),
                "relevance": verdict.get("relevance"),
                "relevance_reason": verdict.get("relevance_reason", ""),
            }
        )

    faithful = [r for r in rows if r["faithfulness"] is not None]
    relevant = [r for r in rows if r["relevance"] is not None]
    return {
        "n_queries": len(rows),
        "n_failed": n_failed,
        "mean_faithfulness": round(sum(r["faithfulness"] for r in faithful) / len(faithful), 4)
        if faithful
        else None,
        "mean_relevance": round(sum(r["relevance"] for r in relevant) / len(relevant), 4)
        if relevant
        else None,
        "rows": rows,
    }


def _diff_with_previous(latest_path: Path, report: dict) -> dict | None:
    if not latest_path.exists():
        return None
    prev = json.loads(latest_path.read_text(encoding="utf-8"))
    diff: dict = {}
    for key in METRIC_KEYS:
        a = prev.get("retrieval", {}).get(key, prev.get("generation", {}).get(key))
        b = report.get("retrieval", {}).get(key, report.get("generation", {}).get(key))
        if a is not None and b is not None:
            diff[key] = {"previous": a, "current": b, "delta": round(b - a, 4)}
    return diff or None


class EvalHarness:
    def __init__(
        self,
        settings: Settings,
        index: Index,
        generator: Generator,
        judge: Judge,
    ):
        self.settings = settings
        self.index = index
        self.generator = generator
        self.judge = judge
        self.last_report_path: Path | None = None

    def run(self, queries: list[GoldenQuery] | None = None) -> dict:
        queries = queries or load_golden_queries(self.settings.eval_data_path)
        start = time.time()
        retrieval = run_retrieval_evals(self.index, queries)
        generation = run_generation_evals(
            self.index, self.generator, self.judge, queries, self.settings
        )
        report = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "config": {
                "embedding_model": getattr(self.index.embedder, "model_name", None)
                or getattr(self.index.embedder, "model", None),
                "chunk_strategy": self.settings.chunk_strategy,
                "chunk_size": self.settings.chunk_size,
                "chunk_overlap": self.settings.chunk_overlap,
                "hybrid_top_k": self.settings.hybrid_top_k,
                "final_top_k": self.settings.final_top_k,
                "use_rerank": self.settings.use_rerank,
            },
            "retrieval": retrieval,
            "generation": generation,
            "duration_sec": round(time.time() - start, 2),
        }
        self._save_report(report)
        return report

    def _save_report(self, report: dict) -> None:
        out_dir = self.settings.eval_report_dir
        out_dir.mkdir(parents=True, exist_ok=True)
        name = datetime.now().strftime("run_%Y%m%d_%H%M%S.json")
        path = out_dir / name
        latest = out_dir / "latest.json"
        diff = _diff_with_previous(latest, report)
        if diff is not None:
            report["diff_vs_previous"] = diff
        path.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        latest.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
        self.last_report_path = path

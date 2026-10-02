import asyncio
import subprocess
import sys
from datetime import datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.embedding import get_embedder
from app.eval_conversation import ConversationEvalHarness
from app.eval_finetune import FinetuneEvalRunner
from app.eval_gate import evaluate_gate, load_gate_report, write_gate_report
from app.eval_harness import EvalHarness
from app.generate import Generator, Judge
from app.index import Index
from app.memory.manager import MemoryManager
from app.rerank import CrossEncoderReranker
from app.settings import Settings
from app.tools.registry import build_tool_registry


def run_retrieval_generation(use_rerank: bool | None = None):
    """Run Phase 1 retrieval + generation evals."""
    settings = Settings()
    if use_rerank is not None:
        settings.use_rerank = use_rerank
    embedder = get_embedder(settings)
    reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_rerank else None
    index = Index(settings, embedder, reranker)
    if not index.load():
        index.build_from_directory()
        index.save()
    generator = Generator(settings)
    judge = Judge(settings)
    harness = EvalHarness(settings, index, generator, judge)
    report = harness.run()
    r = report["retrieval"]
    g = report["generation"]
    print("=== RETRIEVAL ===")
    print(
        f"n={r['n_queries']} MRR={r['mrr']:.3f} hit@1={r['hit_rate_at_1']:.3f} hit@3={r['hit_rate_at_3']:.3f} hit@5={r['hit_rate_at_5']:.3f}"
    )
    print("=== GENERATION ===")
    if g.get("skipped"):
        print(f"skipped: {g['reason']}")
    else:
        print(
            f"n={g['n_queries']} mean_faithfulness={g['mean_faithfulness']} mean_relevance={g['mean_relevance']}"
        )
    if "diff_vs_previous" in report:
        print("=== DIFF VS PREVIOUS ===")
        for k, v in report["diff_vs_previous"].items():
            print(f"{k}: {v['previous']} -> {v['current']} ({v['delta']:+.4f})")
    if harness.last_report_path:
        print(f"Report: {harness.last_report_path}")
    return index, generator, judge, report


async def run_conversation_evals(use_rerank: bool | None = None, index=None):
    """Run Phase 2 conversation evals (builds its own index unless given one)."""
    settings = Settings()
    if use_rerank is not None:
        settings.use_rerank = use_rerank
    if index is None:
        embedder = get_embedder(settings)
        reranker = CrossEncoderReranker(settings.reranker_model) if settings.use_rerank else None
        index = Index(settings, embedder, reranker)
        if not index.load():
            index.build_from_directory()
            index.save()
    generator = Generator(settings)
    # Isolated memory: eval runs are synthetic and must not pollute the real
    # data/memory + data/episodic state (training data + weak-topic tracking).
    memory = MemoryManager(
        long_term_path=settings.trace_dir / "eval_memory",
        episodic_path=settings.trace_dir / "eval_memory",
    )
    registry = build_tool_registry(index, memory, settings.use_rerank)
    harness = ConversationEvalHarness(settings, generator, registry, memory)
    report = await harness.run("evals/conversation_tests.jsonl")
    print("=== CONVERSATION EVALS ===")
    if report.get("skipped"):
        print(f"skipped: {report['reason']}")
    else:
        for r in report["results"]:
            status = "PASS" if r["all_tools_match"] else "FAIL"
            faith = f" faithfulness={r['avg_faithfulness']:.1f}" if r["avg_faithfulness"] else ""
            print(f"  {r['test_id']} ({r['name']}): {status}{faith} n_turns={r['n_turns']}")
            if status == "FAIL":
                for t in r["turn_results"]:
                    if not t["tools_match"]:
                        print(
                            f"    turn {t['turn']}: expected {t['expected_tools']} "
                            f"got {t['tools_called']} faith={t['faithfulness']}"
                        )
        print(f"Duration: {report['duration_sec']}s")
    return report


async def run_finetune_evals(index=None, use_rerank: bool = False) -> dict:
    """Run Phase 3 export-health metrics + fake-LLM tool_faithfulness replay."""
    runner = FinetuneEvalRunner(Settings())
    metrics = runner.data_metrics()
    replay = await runner.run_replay(index, use_rerank)
    print("=== FINETUNE EVALS ===")
    print(
        f"raw_turns={metrics['n_raw_turns']} pure={metrics['n_pure_turns']} "
        f"kept={metrics['n_examples']} kept_ratio={metrics['kept_ratio']} "
        f"dedup_ratio={metrics['dedup_ratio']}"
    )
    print(
        f"train={metrics['n_train']} val={metrics['n_val']} "
        f"tool_calls={metrics['tool_calls_total']} "
        f"tool_success={metrics['tool_calls_success_rate']}"
    )
    if replay.get("skipped"):
        print(f"replay skipped: {replay.get('reason')}")
    elif replay.get("error"):
        print(f"replay error: {replay['error']}")
    else:
        print(
            f"tool_faithfulness={replay.get('tool_faithfulness')} "
            f"pass_rate={replay.get('pass_rate')} n_turns={replay.get('n_turns')}"
        )
    return {**metrics, "replay": replay}


def run_gate(use_rerank: bool, baseline_path: str | Path | None = None) -> int:
    """Phase 3 gate: retrieval + agent + conversation + finetune, then diff
    against a baseline and exit non-zero on any regression.

    ``baseline_path`` pins the comparison target (CI passes the committed
    ``evals/gate_baseline.json``). The current run is always written to
    ``evals/reports/gate_latest.json``; a pinned baseline is never overwritten.
    """
    settings = Settings()
    failures: list[str] = []
    gate: dict = {"created_at": datetime.now().isoformat(timespec="seconds")}

    print("=== GATE: retrieval + generation ===")
    index, _gen, _judge, report = run_retrieval_generation(use_rerank)
    gate["retrieval"] = {
        k: report["retrieval"].get(k)
        for k in ("n_queries", "mrr", "hit_rate_at_1", "hit_rate_at_3", "hit_rate_at_5")
    }
    gen = report.get("generation") or {}
    gate["generation"] = {
        "skipped": bool(gen.get("skipped")),
        "reason": gen.get("reason"),
        "mean_faithfulness": gen.get("mean_faithfulness"),
        "mean_relevance": gen.get("mean_relevance"),
    }

    print("=== GATE: agent e2e check ===")
    proc = subprocess.run(
        [sys.executable, "scripts/e2e_agent_check.py"],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )
    gate["agent"] = {"returncode": proc.returncode}
    if proc.returncode != 0:
        tail = ((proc.stdout or "") + (proc.stderr or ""))[-2000:]
        print(tail)

    print("=== GATE: conversation evals ===")
    conv = asyncio.run(run_conversation_evals(use_rerank, index=index))
    if conv.get("skipped"):
        gate["conversation"] = {
            "skipped": True,
            "reason": conv.get("reason"),
            "n_tests": 0,
            "pass_rate": None,
        }
    else:
        results = conv.get("results") or []
        n_pass = sum(1 for r in results if r["all_tools_match"])
        gate["conversation"] = {
            "skipped": False,
            "n_tests": len(results),
            "pass_rate": round(n_pass / len(results), 4) if results else None,
        }

    print("=== GATE: finetune export + tool replay ===")
    gate["finetune"] = asyncio.run(
        run_finetune_evals(index=index, use_rerank=use_rerank)
    )

    latest_path = settings.eval_report_dir / "gate_latest.json"
    prev_path = Path(baseline_path) if baseline_path else latest_path
    prev = load_gate_report(prev_path)
    if prev_path.exists() and prev is None:
        failures.append(f"gate baseline unreadable: {prev_path}")
    _passed, gate_failures = evaluate_gate(prev, gate, settings)
    failures.extend(gate_failures)

    gate["failures"] = failures
    gate["passed"] = not failures
    write_gate_report(latest_path, gate)
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    write_gate_report(settings.eval_report_dir / f"gate_{stamp}.json", gate)

    print("=== GATE RESULT ===")
    if prev is None and not prev_path.exists():
        if baseline_path:
            print(f"warning: pinned baseline not found: {prev_path} (no regression diff)")
        else:
            print("baseline established (first gate run)")
    if failures:
        for f in failures:
            print(f"FAIL: {f}")
        print("GATE: FAIL")
        return 1
    print("GATE: PASS")
    return 0


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--no-rerank", action="store_true", help="Disable reranker")
    parser.add_argument(
        "--conversation", action="store_true", help="Run Phase 2 conversation evals"
    )
    parser.add_argument(
        "--finetune", action="store_true", help="Run Phase 3 finetune data evals"
    )
    parser.add_argument(
        "--gate",
        action="store_true",
        help="Run every eval gate (retrieval + agent + conversation + finetune) "
        "and exit non-zero on any regression",
    )
    parser.add_argument("--phase1-only", action="store_true", help="Run only Phase 1 evals")
    parser.add_argument(
        "--baseline",
        default=None,
        help="Pinned gate baseline JSON to diff against (default: "
        "evals/reports/gate_latest.json, the previous local run)",
    )
    args = parser.parse_args()

    use_rerank = not args.no_rerank

    if args.gate:
        raise SystemExit(run_gate(use_rerank, baseline_path=args.baseline))

    if (args.conversation or args.finetune) and not args.phase1_only:
        index, _gen, _judge, _report = run_retrieval_generation(use_rerank)
        if args.conversation:
            print()
            asyncio.run(run_conversation_evals(use_rerank, index=index))
        if args.finetune:
            print()
            asyncio.run(run_finetune_evals(index=index, use_rerank=use_rerank))
    else:
        run_retrieval_generation(use_rerank)


if __name__ == "__main__":
    main()

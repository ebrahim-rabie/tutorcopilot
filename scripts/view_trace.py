import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.settings import Settings
from app.tracing import load_trace


def view_trace(trace_id: str):
    settings = Settings()
    trace = load_trace(settings.trace_dir, trace_id)
    if not trace:
        print(f"Trace {trace_id} not found in {settings.trace_dir}")
        return

    print(f"\n=== Trace {trace_id} ===")
    print(f"Started: {trace['started_at']}")
    print(f"Events: {len(trace['events'])}")
    print("-" * 60)

    for event in trace["events"]:
        name = event["name"]
        data = event.get("data") or {}
        print(f"\n>> {name}")
        if name.startswith("iteration"):
            print(f"  iteration: {name}")
            continue
        if name == "tool_call":
            print(
                f"  tool={data.get('tool')} args={json.dumps(data.get('args'), ensure_ascii=False)[:120]}"
            )
        elif name == "tool_result":
            print(
                f"  tool={data.get('tool')} success={data.get('success')} latency_ms={data.get('latency_ms')}"
            )
        elif name == "final_answer":
            print(f"  answer: {data.get('answer', '')[:200]}")
        elif name == "tool_error":
            print(f"  tool={data.get('tool')} error={data.get('error')}")
        elif name == "cycle_detected":
            print(f"  tool={data.get('tool')}")
        elif name == "max_iterations_reached":
            print(f"  iterations={data.get('iterations')}")
        else:
            print(f"  {json.dumps(data, ensure_ascii=False)[:200]}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="View an agent trace")
    parser.add_argument("trace_id", help="Trace ID to view")
    args = parser.parse_args()
    view_trace(args.trace_id)

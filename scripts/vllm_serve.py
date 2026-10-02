"""Phase 3 (P3.3): print the vLLM serve command — or probe a running endpoint.

Print mode (default) is pure output, no network:

    python scripts/vllm_serve.py

emits the ``vllm serve`` invocation for a LoRA adapter plus the
``TC_LLM_BASE_URL`` one-liner that points ``/agent/ask`` at it.

Probe mode waits for an already-running OpenAI-compatible endpoint:

    python scripts/vllm_serve.py --probe --timeout 60

Exit codes: 0 = printed / endpoint ready, 1 = probe timed out.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.serving import wait_ready


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--served-name", default="tutorcopilot-lora")
    parser.add_argument(
        "--port",
        type=int,
        default=9000,
        help="vLLM port (default 9000 so it does not clash with the FastAPI dev server on 8000)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument(
        "--base-url",
        default=None,
        help="Probe target (default: http://localhost:{port}/v1)",
    )
    parser.add_argument("--probe", action="store_true", help="Wait for readiness instead of printing")
    parser.add_argument("--timeout", type=float, default=60.0, help="Probe timeout in seconds")
    parser.add_argument("--interval", type=float, default=1.0, help="Probe poll interval in seconds")
    args = parser.parse_args(argv)

    base_url = args.base_url or f"http://localhost:{args.port}/v1"

    if args.probe:
        print(f"probing {base_url} (timeout {args.timeout:.0f}s)...")
        if wait_ready(base_url, timeout=args.timeout, interval=args.interval):
            print("READY: endpoint serves /v1/models")
            return 0
        print(f"NOT READY: no OpenAI-compatible models at {base_url} within {args.timeout:.0f}s")
        return 1

    print("# requires: pip install vllm  (large download; blocked until installed)")
    print(
        f"vllm serve {args.model} "
        f"--served-model-name {args.served_name} "
        f"--host {args.host} --port {args.port}"
    )
    print("# adapter + base model in one process; for a LoRA adapter add:")
    print(f"#   --enable-lora --lora-modules {args.served_name}=config/lora_adapter")
    print("# point the app at the fine-tuned model (.env):")
    print(f"TC_LLM_BASE_URL={base_url}")
    print("# then restart the app and confirm with:")
    print(f"#   python scripts/vllm_serve.py --probe --base-url {base_url}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

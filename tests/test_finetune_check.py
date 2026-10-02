"""Hermetic gate for Phase 3 P3.3: export --check-only + serving probe.

No torch/trl/peft/datasets installs, no real vLLM:

* ``TrainingExample.from_dict`` round-trip + strict rejection of schema drift;
* ``check_exports`` on valid/invalid temp JSONL (line-numbered errors, empty
  train/val refused) and on the live ``config/finetune`` exports when present;
* ``finetune_lora`` CLI contracts: ``--check-only`` exit codes and the
  training-path import guard failing with an install hint (not a traceback);
* ``app.serving`` readiness probe against a local stub HTTP server and a
  closed port (refused fast, bounded by timeout).
"""

from __future__ import annotations

import http.server
import json
import socket
import threading
from pathlib import Path

import pytest

from app.finetune.check import check_export_file, check_exports
from app.finetune.schema import TrainingExample
from app.serving import is_ready, wait_ready
from scripts import finetune_lora, vllm_serve

GOOD_ROWS = [
    {
        "question": "What is RRF?",
        "generation": "RRF merges ranked lists [1].",
        "tool_calls": [{"name": "search_knowledge", "args": {"query": "rrf"}, "success": True}],
        "success": True,
        "trace_id": "t1",
    },
    {
        "question": "What is MRR?",
        "generation": "Mean reciprocal rank of the first hit.",
        "tool_calls": [],
        "success": True,
    },
]


def _write_jsonl(path: Path, rows: list) -> Path:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")
    return path


def test_from_dict_round_trip_and_strictness():
    ex = TrainingExample.from_dict(GOOD_ROWS[0])
    again = TrainingExample.from_dict(ex.to_dict())
    assert (again.question, again.generation, again.success) == (
        ex.question,
        ex.generation,
        ex.success,
    )
    assert again.tool_calls[0].name == "search_knowledge"

    with pytest.raises(ValueError):
        TrainingExample.from_dict({"question": "x"})  # generation missing
    with pytest.raises(ValueError):
        TrainingExample.from_dict({"question": ["x"], "generation": "y"})
    with pytest.raises(ValueError):
        TrainingExample.from_dict(
            {"question": "x", "generation": "y", "tool_calls": [{"args": {}}]}
        )  # tool name missing
    with pytest.raises(ValueError):
        TrainingExample.from_dict(
            {"question": "x", "generation": "y", "tool_calls": "nope"}
        )
    with pytest.raises(ValueError):
        TrainingExample.from_dict({"question": "x", "generation": "y", "success": "yes"})


def test_check_exports_valid_and_invalid(tmp_path: Path):
    train = _write_jsonl(tmp_path / "train.jsonl", GOOD_ROWS)
    val = _write_jsonl(tmp_path / "val.jsonl", GOOD_ROWS[:1])
    result = check_exports(train, val)
    assert result["ok"] is True
    assert result["train"]["n_valid"] == 2
    assert result["val"]["n_valid"] == 1
    assert result["problems"] == []

    # corrupt: bad JSON + schema drift + purity violation, with line numbers
    bad = tmp_path / "bad.jsonl"
    bad.write_text(
        "\n".join(
            [
                "{not json",
                json.dumps({"question": "missing generation"}),
                json.dumps({"question": "   ", "generation": "x"}),
            ]
        ),
        encoding="utf-8",
    )
    report = check_export_file(bad)
    assert report["n_valid"] == 0
    assert len(report["errors"]) == 3
    assert report["errors"][0].startswith("bad.jsonl:1:")
    assert report["errors"][1].startswith("bad.jsonl:2:")
    assert report["errors"][2].startswith("bad.jsonl:3:")

    empty_train = _write_jsonl(tmp_path / "empty_train.jsonl", [])
    result = check_exports(empty_train, val)
    assert result["ok"] is False
    assert any(">= 1 training example" in p for p in result["problems"])

    missing = check_exports(tmp_path / "nope.jsonl", val)
    assert missing["ok"] is False
    assert any("file not found" in e for e in missing["train"]["errors"])


def test_check_real_exports_if_present():
    """Live config/finetune exports, when non-empty, must satisfy --check-only.

    Empty is a legitimate state (no real conversations yet — the export
    pipeline refuses synthetic dumps), so empty files skip instead of fail.
    """
    train = Path("config/finetune/train.jsonl")
    val = Path("config/finetune/val.jsonl")
    if not train.exists() or not train.stat().st_size:
        pytest.skip("no training data exported yet (needs real conversations)")
    result = check_exports(train, val)
    assert result["ok"] is True, result
    assert result["train"]["n_valid"] >= 1
    assert result["val"]["n_valid"] >= 1


def test_finetune_lora_check_only_exit_codes(tmp_path: Path, capsys):
    good_train = _write_jsonl(tmp_path / "train.jsonl", GOOD_ROWS)
    good_val = _write_jsonl(tmp_path / "val.jsonl", GOOD_ROWS[:1])
    assert (
        finetune_lora.main(
            ["--check-only", "--train", str(good_train), "--val", str(good_val)]
        )
        == 0
    )
    assert "CHECK-ONLY: OK" in capsys.readouterr().out

    bad = tmp_path / "bad.jsonl"
    bad.write_text("{broken", encoding="utf-8")
    val = _write_jsonl(tmp_path / "val.jsonl", GOOD_ROWS[:1])
    assert finetune_lora.main(["--check-only", "--train", str(bad), "--val", str(val)]) == 1
    out = capsys.readouterr().out
    assert "invalid JSON" in out
    assert "CHECK-ONLY: FAIL" in out


def test_finetune_lora_train_guard_without_ml_stack(tmp_path: Path, capsys):
    """Full training without [train] installed must fail with a hint, not a traceback."""
    train = _write_jsonl(tmp_path / "train.jsonl", GOOD_ROWS)
    val = _write_jsonl(tmp_path / "val.jsonl", GOOD_ROWS[:1])
    rc = finetune_lora.main(["--train", str(train), "--val", str(val)])
    out = capsys.readouterr().out
    if rc == 1 and "missing training dependency" in out:
        # expected on a box without trl/peft/datasets
        assert 'pip install -e ".[train]"' in out
        assert "--check-only" in out
    else:
        # training stack IS installed -> it must have passed the guard and run
        assert rc == 0, out


def test_vllm_serve_print_mode_is_pure(capsys):
    assert vllm_serve.main([]) == 0
    out = capsys.readouterr().out
    assert "vllm serve" in out
    assert "--served-model-name" in out
    assert "TC_LLM_BASE_URL=http://localhost:9000/v1" in out


class _ModelsHandler(http.server.BaseHTTPRequestHandler):
    payload: bytes = b""

    def do_GET(self):
        if self.path == "/v1/models":
            self.send_response(200)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(self.payload)))
            self.end_headers()
            self.wfile.write(self.payload)
        else:
            self.send_response(404)
            self.end_headers()

    def log_message(self, *args):
        pass  # keep test output clean


def _serve(payload: dict) -> tuple[http.server.ThreadingHTTPServer, str]:
    handler = type("H", (_ModelsHandler,), {"payload": json.dumps(payload).encode()})
    server = http.server.ThreadingHTTPServer(("127.0.0.1", 0), handler)
    threading.Thread(target=server.serve_forever, daemon=True).start()
    return server, f"http://127.0.0.1:{server.server_address[1]}"


def _closed_port_url() -> str:
    sock = socket.socket()
    sock.bind(("127.0.0.1", 0))
    port = sock.getsockname()[1]
    sock.close()
    return f"http://127.0.0.1:{port}"


def test_serving_probe_ready_empty_and_down():
    server, url = _serve({"object": "list", "data": [{"id": "tutorcopilot-lora"}]})
    try:
        assert is_ready(url, timeout=2.0) is True
        assert wait_ready(url, timeout=2.0, interval=0.1) is True
    finally:
        server.shutdown()

    server, url = _serve({"object": "list", "data": []})
    try:
        assert is_ready(url, timeout=2.0) is False  # no models listed
    finally:
        server.shutdown()

    down = _closed_port_url()
    assert is_ready(down, timeout=1.0) is False
    assert wait_ready(down, timeout=0.4, interval=0.1) is False


def test_vllm_serve_probe_against_stub():
    server, url = _serve({"object": "list", "data": [{"id": "m"}]})
    try:
        assert vllm_serve.main(["--probe", "--base-url", url, "--timeout", "3", "--interval", "0.1"]) == 0
    finally:
        server.shutdown()
    assert vllm_serve.main(
        ["--probe", "--base-url", _closed_port_url(), "--timeout", "0.4", "--interval", "0.1"]
    ) == 1

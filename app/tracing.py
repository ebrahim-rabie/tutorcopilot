from __future__ import annotations

import json
import math
import time
import uuid
from pathlib import Path

import numpy as np


def _jsonable(value):
    """Coerce numpy / Path / nested objects into plain JSON-safe types."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {k: _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and (math.isnan(value) or math.isinf(value)):
        return None
    if isinstance(value, (int, float, str, bool)) or value is None:
        return value
    return str(value)


class Trace:
    """Structured event recorder; each /ask writes one trace file for debugging."""

    def __init__(self, trace_dir: str | Path, trace_id: str | None = None):
        self.trace_dir = Path(trace_dir)
        self.trace_dir.mkdir(parents=True, exist_ok=True)
        self.trace_id = trace_id or uuid.uuid4().hex[:12]
        self.started_at = time.time()
        self.events: list[dict] = []

    def log(self, name: str, data: dict | None = None) -> None:
        self.events.append(
            {"ts": round(time.time(), 4), "name": name, "data": _jsonable(data) if data else None}
        )

    def save(self) -> Path:
        payload = {"trace_id": self.trace_id, "started_at": self.started_at, "events": self.events}
        path = self.trace_dir / f"{self.trace_id}.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return path

    def to_dict(self) -> dict:
        return {"trace_id": self.trace_id, "started_at": self.started_at, "events": self.events}


def load_trace(trace_dir: str | Path, trace_id: str) -> dict | None:
    path = Path(trace_dir) / f"{trace_id}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))

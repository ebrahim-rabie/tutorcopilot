"""Phase 3 P3.1: export episodic traces into LoRA training JSONL (hermetic).

Reads episodic conversation dumps (``data/episodic/*.json``) via
``app.finetune.schema.load_episodic_turns``, maps+dedupes+splits them with the
schema's pure helpers, and writes a seeded 80/20 ``train.jsonl``/``val.jsonl``
under the finetune data dir.

Hermetic: no LLM, no network, no index, no torch — safe to run any time:

    python scripts/export_lora_data.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.schema import (
    dedupe,
    load_episodic_turns,
    split,
    to_examples,
)
from app.settings import Settings


def main() -> None:
    settings = Settings()
    episodic_dir = Path(settings.episodic_dir or "data/episodic")
    finetune_dir = Path(getattr(settings, "finetune_dir", "config/finetune"))
    finetune_dir.mkdir(parents=True, exist_ok=True)

    turns = load_episodic_turns(episodic_dir)
    examples = dedupe(to_examples(turns))
    if not examples:
        # Truncate stale exports so --check-only reflects reality instead of
        # silently validating old data, then fail: there is nothing to train.
        for name in ("train.jsonl", "val.jsonl"):
            (finetune_dir / name).write_text("", encoding="utf-8")
        print(f"no examples found under {episodic_dir} - run an agent turn first")
        print("exports cleared (empty train/val); set TC_LLM_API_KEY and run conversations")
        raise SystemExit(1)

    train, val = split(examples)
    train_path = finetune_dir / "train.jsonl"
    val_path = finetune_dir / "val.jsonl"
    for path, subset in ((train_path, train), (val_path, val)):
        with path.open("w", encoding="utf-8") as fh:
            for e in subset:
                fh.write(json.dumps(e.to_dict(), ensure_ascii=False) + "\n")

    print("=== LoRA TRAINING DATA EXPORT (hermetic) ===")
    print(f"episodic dumps: {len([p for p in episodic_dir.glob('*.json*')])}")
    print(f"raw turns: {len(turns)}  after purity+dedupe: {len(examples)}")
    print(f"tool_calls kept: {sum(len(e.tool_calls) for e in examples)}")
    print(f"train: {len(train)}  val: {len(val)}")
    print(f"train.jsonl: {train_path}")
    print(f"val.jsonl:   {val_path}")
    print("seed: 42  val_ratio: 0.20")
    ok = len(train) >= 1 and len(val) >= 1
    print(f"ok: {ok}")
    if not ok:
        raise SystemExit(1)


if __name__ == "__main__":
    main()

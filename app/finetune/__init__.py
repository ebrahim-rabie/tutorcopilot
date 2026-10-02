"""Phase 3 fine-tuning package: hermetic trace->LoRA-data export (P3.1).

Public API
----------
- ``TrainingExample`` / ``ToolCallRecord``: validated schema for one
  (question -> assistant generation) pair ready for LoRA fine-tuning
  (re-exported from :mod:`app.finetune.schema`).
- ``load_episodic_turns`` / ``to_examples`` / ``dedupe`` / ``split``:
  hermetic helpers used by ``scripts/export_lora_data.py`` that turn episodic
  conversation dumps into seeded train/val JSONL.

Everything here is hermetic: no torch, no vLLM, no network, no API key.
The LoRA job itself (``scripts/finetune_lora.py``) is network-hermetic too and
only activates when ``TC_LLM_API_KEY`` is present in the environment (same
pattern as ``app.generate.Generator.available()`` in Phase 1).
"""

from __future__ import annotations

from app.finetune.schema import ToolCallRecord, TrainingExample, TrainingExampleSplit

__all__ = [
    "ToolCallRecord",
    "TrainingExample",
    "TrainingExampleSplit",
]

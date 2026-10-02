"""Phase 3 (P3.3): LoRA fine-tune over the config/finetune exports.

Hermetic today (no ML installs):

    python scripts/finetune_lora.py --check-only

Full training — blocked until `pip install -e ".[train]"` — runs
`trl.SFTTrainer` + `peft.LoraConfig` against `config/finetune/{train,val}.jsonl`
and saves a LoRA adapter that `scripts/vllm_serve.py` shows how to serve.
The training path re-runs `--check-only` first and refuses bad data; the
import guard fails with a one-line install hint instead of a traceback.

Training API note: written against `trl>=0.10,<0.12`
(`SFTConfig(dataset_text_field=..., max_seq_length=...)`) — pinned in the
`[train]` extra so the kwargs stay valid.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.finetune.check import check_exports


def run_check(args: argparse.Namespace) -> int:
    result = check_exports(args.train, args.val)
    for side in ("train", "val"):
        f = result[side]
        print(
            f"{side}: {f['path']}\n"
            f"  lines={f['n_lines']} valid={f['n_valid']} "
            f"hard_negatives={f['n_failed']}"
        )
        for err in f["errors"]:
            print(f"  ERROR {err}")
        if f["n_suspicious"]:
            print(
                f"  warn: {f['n_suspicious']} generation(s) start with "
                f"'LLM call failed' (curation hook, not fatal)"
            )
    for problem in result["problems"]:
        print(f"  FAIL {problem}")
    if result["ok"]:
        print("CHECK-ONLY: OK (schema valid, ready for training)")
        return 0
    print("CHECK-ONLY: FAIL")
    return 1


def run_train(args: argparse.Namespace) -> int:
    # Refuse to touch the training stack on top of bad data.
    result = check_exports(args.train, args.val)
    if not result["ok"]:
        print("refusing to train: exports failed --check-only")
        for side in ("train", "val"):
            for err in result[side]["errors"]:
                print(f"  ERROR {err}")
        for problem in result["problems"]:
            print(f"  FAIL {problem}")
        return 1

    try:
        import torch  # noqa: F401  (presence check; installs with sentence-transformers)
        from datasets import Dataset
        from peft import LoraConfig
        from transformers import AutoModelForCausalLM, AutoTokenizer
        from trl import SFTConfig, SFTTrainer
    except ImportError as exc:
        print(f"missing training dependency: {exc}")
        print("install the training stack with: pip install -e \".[train]\"")
        print("no install needed to validate exports: python scripts/finetune_lora.py --check-only")
        return 1

    print(f"base model: {args.model}")
    print(f"exports: {args.train} ({result['train']['n_valid']} rows), "
          f"{args.val} ({result['val']['n_valid']} rows)")

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    def load_rows(path: str) -> list[dict]:
        return [
            json.loads(line)
            for line in Path(path).read_text(encoding="utf-8").splitlines()
            if line.strip()
        ]

    def to_text(row: dict) -> str:
        return tokenizer.apply_chat_template(
            [
                {"role": "user", "content": row["question"]},
                {"role": "assistant", "content": row["generation"]},
            ],
            tokenize=False,
        )

    train_ds = Dataset.from_list([{"text": to_text(r)} for r in load_rows(args.train)])
    eval_ds = Dataset.from_list([{"text": to_text(r)} for r in load_rows(args.val)])

    model = AutoModelForCausalLM.from_pretrained(args.model)
    lora = LoraConfig(
        r=args.lora_r,
        lora_alpha=args.lora_alpha,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
    )
    sft_config = SFTConfig(
        output_dir=str(args.out),
        dataset_text_field="text",
        max_seq_length=args.max_seq_length,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=2,
        per_device_eval_batch_size=2,
        gradient_accumulation_steps=4,
        learning_rate=args.lr,
        warmup_ratio=0.05,
        logging_steps=5,
        eval_strategy="epoch",
        save_strategy="epoch",
        report_to=[],
        seed=42,
    )
    trainer = SFTTrainer(
        model=model,
        args=sft_config,
        train_dataset=train_ds,
        eval_dataset=eval_ds,
        tokenizer=tokenizer,
        peft_config=lora,
    )
    trainer.train()
    trainer.save_model(str(args.out))
    tokenizer.save_pretrained(str(args.out))

    served_name = Path(args.out).name
    print(f"adapter saved to {args.out}")
    print("next steps:")
    print(f"  python scripts/vllm_serve.py --model {args.model} --served-name {served_name}")
    print("  .env: TC_LLM_BASE_URL=http://localhost:9000/v1")
    print("  re-run the eval gate against the candidate: make gate")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate export schema and exit (no torch/trl/peft, hermetic)",
    )
    parser.add_argument("--train", default="config/finetune/train.jsonl")
    parser.add_argument("--val", default="config/finetune/val.jsonl")
    parser.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    parser.add_argument("--out", default="config/lora_adapter")
    parser.add_argument("--epochs", type=float, default=3.0)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--lora-r", type=int, default=8)
    parser.add_argument("--lora-alpha", type=int, default=16)
    parser.add_argument("--max-seq-length", type=int, default=512)
    args = parser.parse_args(argv)

    if args.check_only:
        return run_check(args)
    return run_train(args)


if __name__ == "__main__":
    raise SystemExit(main())

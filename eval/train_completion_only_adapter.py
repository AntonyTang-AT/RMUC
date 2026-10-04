"""Train a completion-only QLoRA candidate on the frozen forecast dataset."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--records", default="out_m2/m2_dataset.jsonl")
    parser.add_argument("--out-dir", default="out_m2/finetuned_temporal_completion_only")
    parser.add_argument("--device", default="cuda")
    parser.add_argument("--max-len", type=int, default=1024)
    parser.add_argument("--batch-size", type=int, default=1)
    parser.add_argument("--grad-accum", type=int, default=8)
    parser.add_argument("--epochs", type=int, default=1)
    parser.add_argument("--lr", type=float, default=2e-4)
    args = parser.parse_args()

    records = [json.loads(line) for line in Path(args.records).read_text(encoding="utf-8").splitlines()
               if line.strip()]
    from finetune.lora_train import train_lora
    adapter, stats = train_lora(
        records, out_dir=args.out_dir, device=args.device,
        max_len=args.max_len, batch_size=args.batch_size,
        grad_accum=args.grad_accum, epochs=args.epochs, lr=args.lr,
        completion_only=True)
    print(json.dumps({"adapter": adapter, "records": len(records), "stats": stats},
                     ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()

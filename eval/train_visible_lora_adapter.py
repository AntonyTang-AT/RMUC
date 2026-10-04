"""Train a named visible-context LoRA candidate without replacing prior adapters."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--records', required=True)
    parser.add_argument('--out-dir', required=True)
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--max-len', type=int, default=1024)
    parser.add_argument('--batch-size', type=int, default=1)
    parser.add_argument('--grad-accum', type=int, default=8)
    parser.add_argument('--epochs', type=int, default=1)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--target-modules', nargs='+', default=('q_proj', 'v_proj'))
    parser.add_argument('--completion-only', action='store_true')
    parser.add_argument('--no-class-balance', action='store_true')
    parser.add_argument('--save-steps', type=int, default=100)
    parser.add_argument('--resume-from-checkpoint', default=None)
    parser.add_argument('--init-adapter', default=None,
                        help='start from a compatible LoRA adapter without resuming optimizer state')
    args = parser.parse_args(argv)

    records = [json.loads(line) for line in Path(args.records).read_text(encoding='utf-8').splitlines()
               if line.strip()]
    from finetune.lora_train import train_lora
    adapter, stats = train_lora(
        records, out_dir=args.out_dir, device=args.device,
        max_len=args.max_len, batch_size=args.batch_size,
        grad_accum=args.grad_accum, epochs=args.epochs, lr=args.lr,
        completion_only=bool(args.completion_only),
        class_balance=not bool(args.no_class_balance),
        target_modules=list(args.target_modules), save_steps=args.save_steps,
        resume_from_checkpoint=args.resume_from_checkpoint,
        init_adapter=args.init_adapter)
    print(json.dumps({
        'adapter': adapter, 'records': len(records), 'stats': stats,
        'completion_only': bool(args.completion_only),
        'class_balance': not bool(args.no_class_balance),
        'target_modules': list(args.target_modules),
        'save_steps': int(args.save_steps),
        'resume_from_checkpoint': args.resume_from_checkpoint,
        'init_adapter': args.init_adapter,
    }, ensure_ascii=False), flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

"""Candidate-probability probe for the project's Qwen2.5 LoRA baseline."""

from __future__ import annotations

import argparse
import gzip
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[1]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from forecast.temporal import HORIZONS, MODES, feature_vector, target_labels  # noqa: E402
from eval.probe_open_jev import _choose, _groups, _state  # noqa: E402


def _candidate_probs(model, tokenizer, prefix, candidates, device):
    prefix_ids = tokenizer(prefix, add_special_tokens=False)["input_ids"]
    rows = []
    for candidate in candidates:
        rows.append(prefix_ids + tokenizer(candidate, add_special_tokens=False)["input_ids"])
    max_len = max(map(len, rows))
    input_ids = torch.full((len(rows), max_len), tokenizer.pad_token_id, dtype=torch.long, device=device)
    attention = torch.zeros_like(input_ids)
    for i, row in enumerate(rows):
        input_ids[i, :len(row)] = torch.tensor(row, dtype=torch.long, device=device)
        attention[i, :len(row)] = 1
    with torch.inference_mode():
        logits = model(input_ids=input_ids, attention_mask=attention, use_cache=False).logits.float()
    scores = []
    prefix_len = len(prefix_ids)
    for i, candidate in enumerate(candidates):
        candidate_ids = tokenizer(candidate, add_special_tokens=False)["input_ids"]
        token_positions = torch.arange(prefix_len, prefix_len + len(candidate_ids), device=device)
        token_ids = torch.tensor(candidate_ids, dtype=torch.long, device=device)
        token_logp = torch.log_softmax(logits[i, token_positions - 1], dim=-1)
        scores.append(float(token_logp[torch.arange(len(token_ids), device=device), token_ids].sum().item()))
    scores = np.asarray(scores, dtype=float)
    scores -= scores.max()
    probs = np.exp(scores)
    return (probs / probs.sum()).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", default="out_m2/forecast_dataset/test.jsonl.gz")
    ap.add_argument("--adapter", default="out_m2/finetuned_lookahead/adapter")
    ap.add_argument("--output", default="out_m2/temporal_forecast/qwen_probe.json")
    ap.add_argument("--max-examples", type=int, default=8)
    ap.add_argument("--context", type=int, default=2)
    args = ap.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    from peft import PeftModel

    meta = Path(args.adapter).parent / "meta.json"
    base = json.loads(meta.read_text(encoding="utf-8"))["base"] if meta.is_file() else str(REPO / "models_hf" / "Qwen2.5-3B-Instruct")
    tokenizer = AutoTokenizer.from_pretrained(base)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    quant = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                               bnb_4bit_quant_type="nf4", bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=quant, device_map={"": 0})
    model = PeftModel.from_pretrained(model, args.adapter).eval()
    device = next(model.parameters()).device

    groups = _groups(Path(args.input))
    selected = _choose(groups, args.max_examples)
    rows = []
    for count, (gid, group_rows, index) in enumerate(selected, 1):
        current = group_rows[index]
        state = _state(group_rows, index, args.context)
        context = json.dumps(state, ensure_ascii=False, separators=(",", ":"))
        probabilities = {}
        for horizon in HORIZONS:
            prefix = ("You are a robotic competition tactical analyst.\n"
                      f"Predict the team's global tactical mode {horizon} seconds after the current time.\n"
                      "Use the causal observable history below.\nContext:\n" + context +
                      "\nAnswer with one label: 进攻, 均衡, or 防守. Answer: ")
            probabilities[str(horizon)] = _candidate_probs(model, tokenizer, prefix, list(MODES), device)
        rows.append({
            "game_id": gid, "side": current["side"], "issued_at": current["issued_at"],
            "probabilities": probabilities,
            "label": {str(h): target_labels(current, h)["global_mode"] for h in HORIZONS},
        })
        if count % 4 == 0:
            print(f"probed {count}/{len(selected)}", flush=True)
    metrics = {}
    for horizon in HORIZONS:
        probs = np.asarray([r["probabilities"][str(horizon)] for r in rows], dtype=float)
        labels = np.asarray([MODES.index(r["label"][str(horizon)]) for r in rows], dtype=int)
        one_hot = np.eye(len(MODES))[labels]
        metrics[f"global_mode@{horizon}s"] = {
            "n": len(rows), "accuracy": float(np.mean(probs.argmax(1) == labels)),
            "brier": float(np.mean(np.sum((probs - one_hot) ** 2, axis=1))),
            "log_loss": float(-np.mean(np.log(np.clip(probs[np.arange(len(labels)), labels], 1e-12, 1.0)))),
        }
    report = {"model": "Qwen2.5-3B-Instruct + finetuned_lookahead LoRA", "base": base,
              "adapter": args.adapter, "input_contract": "observable-only causal feature history",
              "n_examples": len(rows), "metrics": metrics, "rows": rows}
    out = Path(args.output); out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"output": str(out), "n_examples": len(rows), "metrics": metrics}, ensure_ascii=False))


if __name__ == "__main__":
    main()

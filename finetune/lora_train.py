# -*- coding: utf-8 -*-
"""M2 QLoRA 微调: 4bit NF4 加载 Qwen2.5-3B-Instruct(OOM -> 1.5B 兜底), LoRA 注入, 蒸馏。
torch/transformers/peft/datasets 全部在 train_lora 内惰性 import; 模块级零 torch。
"""
import os, time, json


def _resolve(name):
    """优先解析到项目本地 models_hf/<name>(local_dir 下载, 不在 HF 默认缓存), 否则回退 HF 名。"""
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models_hf', name)
    return local if os.path.isdir(local) else f'Qwen/{name}'


LORA_BASE = _resolve('Qwen2.5-3B-Instruct')
FALLBACK_BASE = _resolve('Qwen2.5-1.5B-Instruct')
SYSTEM = ("你是一台比赛策略决策哨兵。根据双方当前世界状态, 输出一组分层宏观决策"
          "(global_mode + 六类单位的 target/move/fire/risk)。")


def _format(rec):
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{rec['instruction']}<|im_end|>\n"
            f"<|im_start|>assistant\n{rec['completion']}<|im_end|>")


def _train_once(records, out_dir, base, max_len, epochs, batch_size, grad_accum, lr, device):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import LoraConfig, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    import torch

    os.makedirs(out_dir, exist_ok=True)
    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _map(rec):
        enc = tok(_format(rec), truncation=True, max_length=max_len, padding='max_length')
        return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'],
                'labels': list(enc['input_ids'])}

    ds = Dataset.from_list([_map(r) for r in records])
    ds.set_format('torch')

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map=device)
    model = prepare_model_for_kbit_training(model)
    lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type='CAUSAL_LM',
                      target_modules=['q_proj', 'k_proj', 'v_proj', 'o_proj',
                                      'gate_proj', 'up_proj', 'down_proj'])
    model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    args = TrainingArguments(
        output_dir=os.path.join(out_dir, 'ckpt'), per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, learning_rate=lr, num_train_epochs=epochs,
        logging_steps=10, save_strategy='no', fp16=True, gradient_checkpointing=True,
        report_to=[], remove_unused_columns=False, dataloader_pin_memory=False, seed=42)
    trainer = Trainer(model=model, args=args, train_dataset=ds)
    t0 = time.time()
    trainer.train()
    model.save_pretrained(os.path.join(out_dir, 'adapter'))
    tok.save_pretrained(os.path.join(out_dir, 'adapter'))
    adapter_path = os.path.join(out_dir, 'adapter')
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'base': base, 'epochs': epochs, 'seconds': int(time.time() - t0),
                   'n_records': len(records), 'tokenizer': str(tok.name_or_path)}, f, ensure_ascii=False, indent=2)
    return adapter_path, {'base': base, 'epochs': epochs, 'seconds': int(time.time() - t0)}


def train_lora(records, out_dir='out_m2/finetuned', base=LORA_BASE, max_len=512,
               epochs=1, batch_size=1, grad_accum=8, lr=2e-4, device='cuda'):
    """instruction/completion -> QLoRA 微调; 返回 (adapter_path, stats)。
    加载失败或训练期 CUDA OOM -> 自动回退 1.5B(FALLBACK_BASE)。"""
    import torch
    try:
        return _train_once(records, out_dir, base, max_len, epochs, batch_size, grad_accum, lr, device)
    except Exception as e:
        low = str(e).lower()
        if 'out of memory' in low or 'cuda' in low or 'cuda out of memory' in low:
            print('训练 OOM -> fallback 1.5B:', e)
            return _train_once(records, out_dir, FALLBACK_BASE, max_len, epochs, batch_size, grad_accum, lr, device)
        if 'model' in low or 'load' in low or 'from_pretrained' in low:
            print('加载失败 -> fallback 1.5B:', e)
            return _train_once(records, out_dir, FALLBACK_BASE, max_len, epochs, batch_size, grad_accum, lr, device)
        raise

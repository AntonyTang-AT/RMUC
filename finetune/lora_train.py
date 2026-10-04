# -*- coding: utf-8 -*-
"""M2 QLoRA 微调: 4bit NF4 加载 Instruct 基座, LoRA 注入, 蒸馏。
默认优先 Qwen2.5-7B(5090 等大显存), 否则 3B; OOM/加载失败 -> 1.5B 兜底。
可用环境变量 RMUC_LORA_BASE 强制指定模型目录名(如 Qwen2.5-3B-Instruct)。
torch/transformers/peft/datasets 全部在 train_lora 内惰性 import; 模块级零 torch。
"""
import os, time, json

# 只用 PyTorch: 机器装了 tensorflow+Keras3, 若不显式禁用, transformers 导入 Trainer 时会尝试拉 TF 集成并报 Keras3 不兼容
os.environ.setdefault('USE_TF', '0')
os.environ.setdefault('USE_JAX', '0')
os.environ.setdefault('USE_TORCH', '1')


def _resolve(name):
    """优先解析到项目本地 models_hf/<name>(local_dir 下载, 不在 HF 默认缓存), 否则回退 HF 名。"""
    local = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), 'models_hf', name)
    return local if os.path.isdir(local) else f'Qwen/{name}'


def _pick_base():
    """显存够时用 7B; 本地无 7B 权重则回退 3B。RMUC_LORA_BASE 可覆盖。"""
    override = os.environ.get('RMUC_LORA_BASE', '').strip()
    if override:
        return _resolve(override)
    seven = _resolve('Qwen2.5-7B-Instruct')
    if os.path.isdir(seven) and (
        os.path.exists(os.path.join(seven, 'config.json'))
        or os.path.exists(os.path.join(seven, 'model.safetensors.index.json'))
        or any(n.endswith('.safetensors') for n in os.listdir(seven))
    ):
        return seven
    return _resolve('Qwen2.5-3B-Instruct')


LORA_BASE = _pick_base()
FALLBACK_BASE = _resolve('Qwen2.5-1.5B-Instruct')
DEFAULT_LORA_TARGET_MODULES = (
    'q_proj', 'k_proj', 'v_proj', 'o_proj',
    'gate_proj', 'up_proj', 'down_proj',
)
SYSTEM = ("你是一台比赛策略决策哨兵。根据双方当前可见世界状态和预测参考信息，输出一组分层宏观决策"
          "(global_mode + 六类单位的 target/move/fire/risk)。预测参考只用于判断未来风险，不能覆盖当前事实；"
          "输入中若提供全局分析，按现状描述（分敌我并突出特殊事件/位置/状态）、预测解析、整体建议三部分理解；"
          "逐单位区分敌方状态与未来预测、我方状态与建议动作及理由。预测只能参考，不能覆盖当前事实。"
          "global_mode仍使用合法三类，细分战术通过单位target/move/fire/risk表达；短期只选当前或可达路线目标，"
          "远期只参考区域和总体方向。未知字段保持未知，不要频繁切换模式。"
          "risk是封闭枚举，只能输出高、中、低；即使target和move都是无，也必须输出risk=低，禁止输出risk=无或其他值。"
          "预测参考中会同时给出模型原始判断、持久性基线和最终融合结果：先以当前可见事实为准，"
          "只有模型与基线有足够证据的分歧才改变动作；低置信度时保持保守并给出触发条件。"
          "决策理由必须能对应到可见单位、区域、速度、血量或预测中的明确字段，不要编造未提供的目标。")


def _digest_section(text, marker, limit):
    """Extract one pipe-delimited digest section without carrying the whole UI dump."""
    start = text.find(marker)
    if start < 0:
        return ''
    end = text.find(' | ', start)
    if end < 0:
        end = len(text)
    return text[start:min(end, start + int(limit))].strip()


def _compact_forecast_reference(text, max_chars=500):
    """Keep forecast class evidence and its persistence comparison in the prompt."""
    marker = '预测参考信息'
    start = str(text).find(marker)
    if start < 0:
        return ''
    object_start = str(text).find('{', start)
    if object_start < 0:
        return str(text)[start:start + int(max_chars)].strip()
    try:
        context, _end = json.JSONDecoder().raw_decode(str(text)[object_start:])
    except (json.JSONDecodeError, TypeError, ValueError):
        return str(text)[start:start + int(max_chars)].strip()

    tasks = (('global_mode', '模式'), ('opponent_trend', '趋势'),
             ('enemy_region', '区域'))
    rows = ['预测参考(非事实):']
    for record in context.get('predictions') or []:
        try:
            horizon = int(record.get('horizon_seconds'))
        except (TypeError, ValueError):
            continue
        fields = [f'{horizon}s']
        comparisons = record.get('model_comparison') or {}
        for task, label in tasks:
            head = record.get(task) or {}
            selected = record.get('selected') or {}
            selected_key = {'global_mode': 'mode', 'opponent_trend': 'trend',
                            'enemy_region': 'region'}[task]
            top = head.get('top', selected.get(selected_key))
            if top is None:
                continue
            confidence = head.get('confidence')
            try:
                confidence_text = f'{float(confidence):.2f}'
            except (TypeError, ValueError):
                confidence_text = '?'
            comparison = comparisons.get(task) or {}
            model = (comparison.get('model') or head.get('model_top') or
                     (head.get('model_vs_baseline') or {}).get('model_top'))
            baseline = (comparison.get('baseline') or head.get('baseline_top') or
                        (head.get('model_vs_baseline') or {}).get('baseline_top'))
            detail = f'{label}={top}({confidence_text}'
            probabilities = head.get('probabilities') or {}
            if probabilities:
                ranked = sorted(probabilities.items(), key=lambda item: item[1],
                                reverse=True)[:2]
                detail += ';P=' + ','.join(
                    f'{name}:{float(probability):.2f}'
                    for name, probability in ranked)
            if model is not None and baseline is not None:
                detail += f';M={model}/P={baseline}'
            detail += ')'
            fields.append(detail)
        row = ' '.join(fields)
        candidate = ' | '.join(rows + [row])
        if len(candidate) > int(max_chars):
            break
        rows.append(row)
    return ' | '.join(rows) if len(rows) > 1 else ''


def compact_training_instruction(instruction, max_chars=760):
    """Compress a long replay digest while retaining decision-critical evidence.

    The full replay digest is useful for visualization but is 1.8k--2.2k
    tokens after the chat template.  Truncating it blindly cuts off the
    assistant completion.  This compact form keeps the current observable
    facts, enemy/self situation, horizon summaries, and action trigger in a
    bounded order so the completion remains inside the token budget.
    """
    text = str(instruction or '')
    if len(text) <= int(max_chars):
        return text
    sections = []
    # Keep the timestamp and a bounded prefix of visible unit facts.
    if text.startswith('局势:'):
        head_end = text.find(' | 时间尺度:')
        head = text[:head_end if head_end >= 0 else len(text)]
        sections.append(head[:260])
    forecast = text.find('预测参考信息')
    forecast_summary = ''
    if forecast >= 0:
        forecast_summary = _compact_forecast_reference(
            text[forecast:], max_chars=min(560, max(120, int(max_chars) - 300)))
    if forecast_summary:
        sections.append(forecast_summary)
    remaining = int(max_chars) - sum(len(section) for section in sections) - len(sections)
    for marker, limit in (
        ('全局分析:', 100),
        ('现状（己方）:', 75),
        ('现状（敌方）:', 85),
        ('预测解析:', 85),
        ('整体建议（仅作建议）:', 80),
        ('自然语言态势:', 60),
    ):
        if remaining <= 0:
            break
        value = _digest_section(text, marker, min(limit, remaining))
        if value:
            sections.append(value)
            remaining -= len(value) + 1
    compact = '\n'.join(sections)
    if len(compact) > int(max_chars):
        compact = compact[:int(max_chars)]
    return compact


def _format(rec):
    instruction = compact_training_instruction(rec['instruction'])
    return prompt_from_instruction(instruction, completion=rec['completion'])


def prompt_from_instruction(instruction, completion=None):
    """Build the one canonical chat wrapper for training and inference."""
    output_contract = (
        '输出契约：只输出一个JSON对象，不要解释或Markdown。units必须且只能包含六个键：'
        '英雄、工程、步兵3、步兵4、空中、哨兵；即使输入中看不到某单位，也必须保留该键并使用保守动作。'
        '每个单位值必须包含target、move、fire、risk；global_mode只能是进攻、均衡、防守；'
        'risk只能是高、中、低，target=无且move=无时risk必须为低，禁止risk=无。'
    )
    suffix = '' if completion is None else f'{completion}<|im_end|>'
    return (f"<|im_start|>system\n{SYSTEM}<|im_end|>\n"
            f"<|im_start|>user\n{instruction}\n{output_contract}<|im_end|>\n"
            f"<|im_start|>assistant\n{suffix}")


def eval_prompt(w, forecast_context=None):
    """评估提示: 与训练 _format **同款** chat 模板(同 SYSTEM), 停在 assistant-start 让模型续写决策。
    不同于裸喂 situation_digest(w)(finetune/eval.py 旧版 bug: 不套模板 -> 模型按模板学的, 裸喂低估合法率)。
    惰性 import situation_digest 以避免循环依赖(make_dataset 不 import lora_train)。"""
    from finetune.make_dataset import situation_digest
    instruction = situation_digest(w, forecast_context=forecast_context)
    if forecast_context is not None:
        instruction = compact_training_instruction(instruction)
    return prompt_from_instruction(instruction)


def _mode_sample_weights(records, beta=0.999):
    """按 global_mode 有效样本量反比得到 per-example 权重(用于 WeightedRandomSampler / CE)。"""
    from collections import Counter
    from finetune.make_dataset import record_mode
    modes = [record_mode(r) for r in records]
    cnt = Counter(modes)
    # Cui et al. effective number of samples
    eff = {}
    for m, c in cnt.items():
        if c <= 0:
            eff[m] = 1.0
        else:
            eff[m] = (1.0 - (beta ** c)) / (1.0 - beta)
    inv = {m: 1.0 / max(eff[m], 1e-8) for m in eff}
    # 归一化使均值约 1
    mean_inv = sum(inv[m] * cnt[m] for m in cnt) / max(sum(cnt.values()), 1)
    inv = {m: inv[m] / mean_inv for m in inv}
    return [float(inv[m]) for m in modes], inv


def _encode_record(tok, rec, *, max_len, completion_only=False):
    """Tokenize one record and optionally mask all non-answer labels."""
    text = _format(rec)
    enc = tok(text, truncation=True, max_length=int(max_len), padding=False,
              add_special_tokens=False)
    labels = list(enc['input_ids'])
    if completion_only:
        # The base model already knows how to read the system/user prompt.
        # Spend the adapter's limited updates on the structured answer,
        # while retaining the assistant end marker as a stopping signal.
        prefix = text.rsplit(rec['completion'], 1)[0]
        prefix_ids = tok(prefix, add_special_tokens=False)['input_ids']
        prompt_tokens = min(len(prefix_ids), len(labels))
        labels[:prompt_tokens] = [-100] * prompt_tokens
    return {'input_ids': enc['input_ids'], 'attention_mask': enc['attention_mask'],
            'labels': labels}


def _train_once(records, out_dir, base, max_len, epochs, batch_size, grad_accum, lr, device,
                 class_balance=True, pad_to_longest=True, use_tf32=True, dataloader_workers=0,
                 completion_only=False, target_modules=None, gradient_checkpointing=True,
                 save_steps=100, resume_from_checkpoint=None, init_adapter=None):
    """pad_to_longest: 按 batch 最长序列 pad（显著减无效 pad 算力）；否则 pad 到 max_len。
    use_tf32: Ampere+ 矩阵乘加速（对 4bit 前向收益有限，几乎无害）。"""
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig, TrainingArguments, Trainer
    from peft import LoraConfig, PeftModel, get_peft_model, prepare_model_for_kbit_training
    from datasets import Dataset
    from torch.utils.data import DataLoader, WeightedRandomSampler
    import torch
    import torch.nn.functional as F

    os.makedirs(out_dir, exist_ok=True)
    if use_tf32 and str(device).startswith('cuda'):
        try:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True
            torch.set_float32_matmul_precision('high')
        except Exception:
            pass

    tok = AutoTokenizer.from_pretrained(base)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token

    def _map(rec):
        # 不在 map 阶段 pad；交给 collator 按 batch 最长 pad，减少空算
        return _encode_record(tok, rec, max_len=max_len,
                              completion_only=completion_only)

    mapped = [_map(r) for r in records]
    sample_weights, class_w = _mode_sample_weights(records) if class_balance else (
        [1.0] * len(records), {})
    for i, row in enumerate(mapped):
        row['sample_weight'] = sample_weights[i]
    ds = Dataset.from_list(mapped)

    bnb = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.float16,
                             bnb_4bit_quant_type='nf4', bnb_4bit_use_double_quant=True)
    if str(device).startswith('cuda'):
        _idx = int(device.split(':')[1]) if ':' in str(device) else torch.cuda.current_device()
        device_map = {'': _idx}
    else:
        device_map = {'': device}
    model = AutoModelForCausalLM.from_pretrained(base, quantization_config=bnb, device_map=device_map)
    model = prepare_model_for_kbit_training(model)
    selected_target_modules = list(target_modules or DEFAULT_LORA_TARGET_MODULES)
    if init_adapter:
        if not os.path.isdir(init_adapter):
            raise FileNotFoundError(f'initial adapter does not exist: {init_adapter}')
        model = PeftModel.from_pretrained(model, init_adapter, is_trainable=True)
    else:
        lora = LoraConfig(r=8, lora_alpha=16, lora_dropout=0.05, task_type='CAUSAL_LM',
                          target_modules=selected_target_modules)
        model = get_peft_model(model, lora)
    model.print_trainable_parameters()

    # 动态 pad：有效序列常远短于 576，batch 内 pad 可砍掉大量空 token
    pad_id = tok.pad_token_id

    def _collate(features):
        import torch
        max_L = max(len(f['input_ids']) for f in features)
        if not pad_to_longest:
            max_L = max_len
        batch = {'input_ids': [], 'attention_mask': [], 'labels': [], 'sample_weight': []}
        for f in features:
            L = len(f['input_ids'])
            pad_n = max_L - L
            batch['input_ids'].append(f['input_ids'] + [pad_id] * pad_n)
            batch['attention_mask'].append(f['attention_mask'] + [0] * pad_n)
            # pad 位置 labels=-100，不进 CE
            batch['labels'].append(f['labels'] + [-100] * pad_n)
            batch['sample_weight'].append(float(f.get('sample_weight', 1.0)))
        return {k: torch.tensor(v) for k, v in batch.items()}

    ta_kw = dict(
        output_dir=os.path.join(out_dir, 'ckpt'), per_device_train_batch_size=batch_size,
        gradient_accumulation_steps=grad_accum, learning_rate=lr, num_train_epochs=epochs,
        logging_steps=10, save_strategy='steps', save_steps=max(1, int(save_steps)),
        save_total_limit=2, fp16=True,
        gradient_checkpointing=bool(gradient_checkpointing),
        report_to=[], remove_unused_columns=False,
        dataloader_pin_memory=bool(str(device).startswith('cuda')),
        dataloader_num_workers=int(dataloader_workers or 0),
        seed=42,
    )
    # 可选加速；旧 transformers / 无 bnb 8bit optim 时降级
    try:
        args = TrainingArguments(**ta_kw, optim='paged_adamw_8bit',
                                 tf32=bool(use_tf32 and str(device).startswith('cuda')))
    except TypeError:
        try:
            args = TrainingArguments(**ta_kw, optim='paged_adamw_8bit')
        except Exception:
            args = TrainingArguments(**ta_kw)

    class BalancedTrainer(Trainer):
        def __init__(self, sample_weights, *a, **kw):
            super().__init__(*a, **kw)
            self._sample_weights = sample_weights

        def get_train_dataloader(self):
            if not class_balance:
                return super().get_train_dataloader()
            sampler = WeightedRandomSampler(
                weights=self._sample_weights,
                num_samples=len(self._sample_weights),
                replacement=True)
            return DataLoader(
                self.train_dataset,
                batch_size=self.args.per_device_train_batch_size,
                sampler=sampler,
                collate_fn=self.data_collator,
                drop_last=self.args.dataloader_drop_last,
                num_workers=self.args.dataloader_num_workers,
                pin_memory=self.args.dataloader_pin_memory,
            )

        def compute_loss(self, model, inputs, return_outputs=False, **kwargs):
            weights = inputs.pop('sample_weight', None)
            labels = inputs.get('labels')
            outputs = model(**inputs)
            logits = outputs.logits
            # 标准 shift CE, 再按样本权重加权; ignore_index=-100 跳过 pad
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            flat_logits = shift_logits.view(-1, shift_logits.size(-1))
            flat_labels = shift_labels.view(-1)
            token_loss = F.cross_entropy(flat_logits, flat_labels, reduction='none',
                                         ignore_index=-100)
            token_loss = token_loss.view(shift_labels.size(0), -1)
            valid = (shift_labels != -100).float()
            per_ex = (token_loss * valid).sum(dim=1) / valid.sum(dim=1).clamp_min(1.0)
            if weights is not None:
                w = weights.to(per_ex.dtype).to(per_ex.device)
                loss = (per_ex * w).sum() / w.sum().clamp_min(1e-8)
            else:
                loss = per_ex.mean()
            return (loss, outputs) if return_outputs else loss

    trainer = BalancedTrainer(
        sample_weights, model=model, args=args, train_dataset=ds, data_collator=_collate)
    t0 = time.time()
    trainer.train(resume_from_checkpoint=resume_from_checkpoint)
    model.save_pretrained(os.path.join(out_dir, 'adapter'))
    tok.save_pretrained(os.path.join(out_dir, 'adapter'))
    adapter_path = os.path.join(out_dir, 'adapter')
    with open(os.path.join(out_dir, 'meta.json'), 'w', encoding='utf-8') as f:
        json.dump({'base': base, 'epochs': epochs, 'seconds': int(time.time() - t0),
                   'n_records': len(records), 'tokenizer': str(tok.name_or_path),
                   'class_balance': class_balance,
                   'completion_only': bool(completion_only),
                   'max_len': max_len, 'batch_size': batch_size, 'grad_accum': grad_accum,
                   'pad_to_longest': pad_to_longest,
                   'target_modules': selected_target_modules,
                    'gradient_checkpointing': bool(gradient_checkpointing),
                    'save_steps': int(save_steps),
                    'resumed_from_checkpoint': resume_from_checkpoint,
                    'init_adapter': init_adapter,
                    'class_weights': {k: round(v, 6) for k, v in class_w.items()}},
                  f, ensure_ascii=False, indent=2)
    return adapter_path, {'base': base, 'epochs': epochs, 'seconds': int(time.time() - t0),
                          'max_len': max_len, 'batch_size': batch_size,
                          'pad_to_longest': pad_to_longest,
                          'target_modules': selected_target_modules,
                           'gradient_checkpointing': bool(gradient_checkpointing),
                           'save_steps': int(save_steps),
                           'resumed_from_checkpoint': resume_from_checkpoint,
                           'init_adapter': init_adapter}


def train_lora(records, out_dir='out_m2/finetuned', base=LORA_BASE, max_len=512,
               epochs=1, batch_size=1, grad_accum=8, lr=2e-4, device='cuda',
               class_balance=True, pad_to_longest=True, use_tf32=True,
                dataloader_workers=0, completion_only=False, target_modules=None,
                gradient_checkpointing=True, save_steps=100,
                resume_from_checkpoint=None, init_adapter=None):
    """instruction/completion -> QLoRA 微调; 返回 (adapter_path, stats)。
    class_balance: WeightedRandomSampler + 有效样本量加权 CE。
    pad_to_longest: batch 内动态 pad（相对固定 576 pad 可明显加速）。
    加载失败或训练期 CUDA OOM -> 自动回退 1.5B(FALLBACK_BASE)。
    target_modules 可传 q_proj/v_proj 等轻量子集用于吞吐实验；默认保持全量目标层。"""
    try:
        return _train_once(records, out_dir, base, max_len, epochs, batch_size, grad_accum, lr,
                           device, class_balance=class_balance, pad_to_longest=pad_to_longest,
                           use_tf32=use_tf32, dataloader_workers=dataloader_workers,
                            completion_only=completion_only, target_modules=target_modules,
                            gradient_checkpointing=gradient_checkpointing,
                            save_steps=save_steps,
                            resume_from_checkpoint=resume_from_checkpoint,
                            init_adapter=init_adapter)
    except Exception as e:
        low = str(e).lower()
        if 'out of memory' in low or 'cuda' in low or 'cuda out of memory' in low:
            print('训练 OOM -> fallback 1.5B:', e)
            return _train_once(records, out_dir, FALLBACK_BASE, max_len, epochs, batch_size,
                               grad_accum, lr, device, class_balance=class_balance,
                               pad_to_longest=pad_to_longest, use_tf32=use_tf32,
                               dataloader_workers=dataloader_workers,
                                completion_only=completion_only, target_modules=target_modules,
                                gradient_checkpointing=gradient_checkpointing,
                                save_steps=save_steps,
                                resume_from_checkpoint=resume_from_checkpoint,
                                init_adapter=init_adapter)
        if 'model' in low or 'load' in low or 'from_pretrained' in low:
            print('加载失败 -> fallback 1.5B:', e)
            return _train_once(records, out_dir, FALLBACK_BASE, max_len, epochs, batch_size,
                               grad_accum, lr, device, class_balance=class_balance,
                               pad_to_longest=pad_to_longest, use_tf32=use_tf32,
                               dataloader_workers=dataloader_workers,
                                completion_only=completion_only, target_modules=target_modules,
                                gradient_checkpointing=gradient_checkpointing,
                                save_steps=save_steps,
                                resume_from_checkpoint=resume_from_checkpoint,
                                init_adapter=init_adapter)
        raise

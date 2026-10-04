# -*- coding: utf-8 -*-
"""Tier2 后端：无卡默认 Null；有 CUDA+adapter 时 LazyLora。

模块级不 import torch。
"""
from __future__ import annotations

import os

# The formal runtime candidate is trained on the current V5 observable-slot
# contract.  Before that artifact exists, ``resolve_tier2_backend('auto')``
# safely returns ``NullTier2`` and the rule path remains active.
DEFAULT_ADAPTER = 'out_m2/finetuned_visible_v5_risk_anchor_fast/adapter'


def _attach_generation_metadata(decision, result):
    """Attach parser provenance as a sidecar, never as an action field."""
    if not isinstance(decision, dict):
        return decision
    output = dict(decision)
    output['_tier2_generation'] = {
        'strict_parse_valid': bool(result.get('strict_parse_valid')),
        'first_pass_strict_parse_valid': bool(
            result.get('first_pass_strict_parse_valid', result.get('strict_parse_valid'))),
        'retry_attempted': 'retry_raw_output' in result,
        'retry_strict_parse_valid': result.get('retry_strict_parse_valid'),
        'contract_repairs': list(result.get('contract_repairs') or []),
        'parse_error': result.get('parse_error'),
        'contract_error': result.get('contract_error'),
    }
    return output


class NullTier2:
    """恒返回 None，不占内存。"""

    def available(self):
        return False

    def generate_one(self, dp, bounds, max_new=200, forecast_context=None):
        return None

    def generate_visible_one(self, visible_input, max_new=200, forecast_context=None):
        return None


class LazyLoraTier2:
    """惰性加载 LoRA；available() 为假时 generate_one 等同 Null。"""

    def __init__(self, adapter_path=DEFAULT_ADAPTER, device='cuda', base=None, max_new=200):
        self.adapter_path = adapter_path
        self.device = device
        self.base = base
        self.max_new = int(max_new)
        self._tok = None
        self._model = None
        self._load_failed = False

    def available(self):
        if self._load_failed:
            return False
        if not self.adapter_path or self.adapter_path in ('none', 'null', '-'):
            return False
        if not os.path.isdir(self.adapter_path):
            return False
        try:
            import torch
            if not str(self.device).startswith('cuda'):
                return False
            if not torch.cuda.is_available():
                return False
        except Exception:
            return False
        return True

    def _ensure_loaded(self):
        if self._model is not None or self._load_failed:
            return self._model is not None
        if not self.available():
            self._load_failed = True
            return False
        try:
            from finetune import infer_one as io
            self._tok, self._model = io.load_lora_bundle(
                self.adapter_path, base=self.base, device=self.device)
            return True
        except Exception:
            self._load_failed = True
            self._tok = None
            self._model = None
            return False

    def generate_one(self, dp, bounds, max_new=None, forecast_context=None):
        if not self._ensure_loaded():
            return None
        from finetune import infer_one as io
        mn = self.max_new if max_new is None else int(max_new)
        try:
            return io.generate_with_bundle(self._tok, self._model, dp, bounds, max_new=mn,
                                           forecast_context=forecast_context)
        except Exception:
            return None

    def generate_visible_one(self, visible_input, max_new=None, forecast_context=None):
        """Generate from a validated sensor/radar projection only.

        This is intentionally separate from ``generate_one`` for historical
        adapters that were trained against complete replay world states.  A
        production-visible adapter must take this path; callers fall back to
        Tier0/Tier1 if the input cannot pass the visibility contract.
        """
        if not self._ensure_loaded():
            return None
        from eval.llm_visible_input import format_visible_training_instruction
        from finetune import infer_one as io
        mn = self.max_new if max_new is None else int(max_new)
        try:
            instruction = format_visible_training_instruction(
                visible_input, forecast_context=forecast_context, max_chars=640)
            result = io.generate_instruction_results_with_bundle(
                self._tok, self._model, [instruction], max_new=mn,
                prompt_max_length=896)[0]
            return _attach_generation_metadata(result.get('decision'), result)
        except Exception:
            return None


def resolve_tier2_backend(adapter='auto', device='auto'):
    """CLI 友好工厂。

    adapter: 'auto'|路径|'none'
    device: 'auto'|'cuda'|'cpu'|...
    """
    if adapter in (None, 'none', 'null', '-'):
        return NullTier2()
    path = DEFAULT_ADAPTER if adapter in (None, 'auto', '') else str(adapter)
    if device in (None, 'auto', ''):
        try:
            import torch
            dev = 'cuda' if torch.cuda.is_available() else 'cpu'
        except Exception:
            return NullTier2()
    else:
        dev = str(device)
    if not str(dev).startswith('cuda'):
        return NullTier2()
    backend = LazyLoraTier2(adapter_path=path, device=dev)
    if not backend.available():
        return NullTier2()
    return backend

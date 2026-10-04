"""Backward-compatible import for the canonical hybrid forecaster.

The implementation lives in :mod:`forecast.hybrid_model`.  This small shim
keeps older experiment scripts that import ``forecast.hybrid`` working while
avoiding two divergent TCN+Transformer implementations.
"""

from __future__ import annotations

from .hybrid_model import (
    HORIZON_POOL,
    HybridTemporalForecastNet as _HybridTemporalForecastNet,
    TCNTransformerForecastNet,
)


class HybridTemporalForecastNet(_HybridTemporalForecastNet):
    """Compatibility wrapper using the old ``heads``/``max_context`` names."""

    model_type = _HybridTemporalForecastNet.architecture_name

    def __init__(self, *args, heads=4, tcn_dilations=None, max_context=256,
                 nhead=None, max_seq_len=None, **kwargs):
        # ``tcn_dilations`` was configurable in the provisional module.  The
        # canonical implementation derives a stable dilation schedule from
        # ``layers`` so old callers can pass the argument without changing
        # the selected architecture.
        del tcn_dilations
        super().__init__(*args, nhead=(nhead or heads),
                         max_seq_len=(max_seq_len or max_context), **kwargs)


__all__ = ["HORIZON_POOL", "HybridTemporalForecastNet", "TCNTransformerForecastNet"]

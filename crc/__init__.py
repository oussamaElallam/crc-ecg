"""Exact-binomial per-class risk control for multi-label classification."""

from .multilabel_crc import LegacyHoeffdingCRC, MultiLabelCRC
from .baselines import (
    baseline_class_matched_cp_multilabel,
    baseline_per_class_f1_threshold,
    baseline_standard_cp_multilabel,
    baseline_threshold,
)

__all__ = [
    "MultiLabelCRC",
    "LegacyHoeffdingCRC",
    "baseline_threshold",
    "baseline_per_class_f1_threshold",
    "baseline_standard_cp_multilabel",
    "baseline_class_matched_cp_multilabel",
]

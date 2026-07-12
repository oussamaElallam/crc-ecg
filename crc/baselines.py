"""Multi-label baselines used in the manuscript comparisons."""

from __future__ import annotations

from typing import Mapping

import numpy as np


def baseline_threshold(probs: np.ndarray, thr: float) -> np.ndarray:
    """Apply one fixed probability threshold to every class."""
    return (np.asarray(probs) >= float(thr)).astype(np.int8)


def baseline_per_class_f1_threshold(
    probs_cal: np.ndarray, y_cal: np.ndarray
) -> np.ndarray:
    """Tune one threshold per class to maximize calibration-set F1."""
    probs = np.asarray(probs_cal)
    labels = np.asarray(y_cal)
    if probs.shape != labels.shape or probs.ndim != 2:
        raise ValueError("probs_cal and y_cal must be equal-shaped 2-D arrays")
    thresholds = np.zeros(probs.shape[1], dtype=np.float64)
    candidates = np.linspace(0.05, 0.95, 19)
    for k in range(probs.shape[1]):
        best_f1, best_threshold = -1.0, 0.5
        truth = labels[:, k]
        if truth.sum() == 0:
            thresholds[k] = best_threshold
            continue
        for threshold in candidates:
            pred = probs[:, k] >= threshold
            tp = int(np.sum(pred & (truth == 1)))
            fp = int(np.sum(pred & (truth == 0)))
            fn = int(np.sum((~pred) & (truth == 1)))
            precision = tp / (tp + fp) if tp + fp else 0.0
            recall = tp / (tp + fn) if tp + fn else 0.0
            f1 = (
                2.0 * precision * recall / (precision + recall)
                if precision + recall else 0.0
            )
            if f1 > best_f1:
                best_f1, best_threshold = f1, float(threshold)
        thresholds[k] = best_threshold
    return thresholds


def _positive_quantile_threshold(
    probs: np.ndarray, labels: np.ndarray, k: int, alpha: float
) -> float:
    positive = labels[:, k] == 1
    n = int(positive.sum())
    if n < 2:
        return 0.0
    scores = 1.0 - probs[positive, k]
    q = min(float(np.ceil((n + 1) * (1.0 - alpha)) / n), 1.0)
    return float(1.0 - np.quantile(scores, q))


def baseline_standard_cp_multilabel(
    probs_cal: np.ndarray, y_cal: np.ndarray, alpha: float = 0.10
) -> np.ndarray:
    """Uniform-alpha, uncorrected positive-class split-CP baseline."""
    probs = np.asarray(probs_cal, dtype=np.float64)
    labels = np.asarray(y_cal)
    if probs.shape != labels.shape or probs.ndim != 2:
        raise ValueError("probs_cal and y_cal must be equal-shaped 2-D arrays")
    return np.array([
        _positive_quantile_threshold(probs, labels, k, float(alpha))
        for k in range(probs.shape[1])
    ])


def baseline_class_matched_cp_multilabel(
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
    class_alphas: Mapping[int, float],
    default_alpha: float = 0.10,
) -> np.ndarray:
    """Uncorrected CP using the proposed method's class-specific targets."""
    probs = np.asarray(probs_cal, dtype=np.float64)
    labels = np.asarray(y_cal)
    if probs.shape != labels.shape or probs.ndim != 2:
        raise ValueError("probs_cal and y_cal must be equal-shaped 2-D arrays")
    return np.array([
        _positive_quantile_threshold(
            probs, labels, k, float(class_alphas.get(k, default_alpha))
        )
        for k in range(probs.shape[1])
    ])

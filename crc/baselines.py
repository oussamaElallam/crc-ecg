"""Multi-label baselines for comparison with MultiLabelCRC."""

import numpy as np


# Baselines for multi-label
def baseline_threshold(probs: np.ndarray, thr: float) -> np.ndarray:
    return (probs >= thr).astype(int)

def baseline_per_class_f1_threshold(probs_cal: np.ndarray, y_cal: np.ndarray) -> np.ndarray:
    """Tune per-class threshold to maximize F1 on calibration data."""
    K = probs_cal.shape[1]
    best_thr = np.zeros(K)
    candidates = np.linspace(0.05, 0.95, 19)
    for k in range(K):
        best_f1, best_t = -1, 0.5
        y_k = y_cal[:, k]
        if y_k.sum() == 0:
            best_thr[k] = 0.5
            continue
        for t in candidates:
            pred_k = (probs_cal[:, k] >= t).astype(int)
            tp = ((pred_k == 1) & (y_k == 1)).sum()
            fp = ((pred_k == 1) & (y_k == 0)).sum()
            fn = ((pred_k == 0) & (y_k == 1)).sum()
            if tp + fp == 0 or tp + fn == 0:
                continue
            prec = tp / (tp + fp)
            rec = tp / (tp + fn)
            f1 = 2*prec*rec / (prec + rec) if (prec + rec) > 0 else 0
            if f1 > best_f1:
                best_f1, best_t = f1, t
        best_thr[k] = best_t
    return best_thr

def baseline_standard_cp_multilabel(probs_cal: np.ndarray, y_cal: np.ndarray,
                                    alpha: float = 0.10) -> np.ndarray:
    """
    Naive split-CP applied per class: single global Hoeffding-free quantile
    of nonconformity scores 1 - p_k(x_i) over positives for class k.
    """
    K = probs_cal.shape[1]
    thr = np.zeros(K)
    for k in range(K):
        pos = y_cal[:, k] == 1
        if pos.sum() < 2:
            thr[k] = 0.5
            continue
        scores = 1 - probs_cal[pos, k]
        n = pos.sum()
        q = np.ceil((n + 1) * (1 - alpha)) / n
        thr[k] = float(1 - np.quantile(scores, min(q, 1.0)))
    return thr  # interpret as: include class k if p_k >= thr[k]

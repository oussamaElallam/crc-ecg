"""Exact-binomial per-class risk control for multi-label classification.

The primary class uses one-sided Clopper-Pearson upper confidence bounds for
binary false-negative loss. The submitted Hoeffding-plus-floor rule is retained
only as ``LegacyHoeffdingCRC`` for transparent audit/ablation.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import beta


class MultiLabelCRC:
    """Per-class exact-binomial RCPS with optional family-wise control."""

    def __init__(
        self,
        default_alpha: float = 0.10,
        confidence: float = 0.95,
        simultaneous: bool = True,
        family_size: Optional[int] = None,
    ) -> None:
        if not 0.0 < default_alpha < 1.0:
            raise ValueError("default_alpha must be between 0 and 1")
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        if family_size is not None and family_size < 1:
            raise ValueError("family_size must be positive")
        self.default_alpha = float(default_alpha)
        self.confidence = float(confidence)
        self.simultaneous = bool(simultaneous)
        self.family_size = family_size
        self.lambdas: Dict[int, float] = {}
        self.calibration_info: Dict[int, Dict] = {}

    @staticmethod
    def exact_upper_bound(misses: int, n: int, delta: float) -> float:
        """One-sided ``1-delta`` Clopper-Pearson upper bound."""
        if n <= 0:
            return 1.0
        if not 0 <= misses <= n:
            raise ValueError("misses must lie between 0 and n")
        if not 0.0 < delta < 1.0:
            raise ValueError("delta must be between 0 and 1")
        if misses == n:
            return 1.0
        return float(beta.ppf(1.0 - delta, misses + 1, n - misses))

    def calibrate(
        self,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
        class_alphas: Optional[Mapping[int, float]] = None,
        *,
        controlled_classes: Optional[Iterable[int]] = None,
        guard_lambdas: Optional[Mapping[int, float]] = None,
    ) -> Dict[int, float]:
        """Calibrate monotone per-class thresholds using calibration data only."""
        probs = np.asarray(probs_cal, dtype=np.float64)
        labels = np.asarray(y_cal)
        if probs.ndim != 2 or labels.ndim != 2 or probs.shape != labels.shape:
            raise ValueError("probs_cal and y_cal must be equal-shaped 2-D arrays")
        if not np.isfinite(probs).all():
            raise ValueError("probs_cal contains NaN or infinite values")

        k_count = probs.shape[1]
        alphas = {
            k: float((class_alphas or {}).get(k, self.default_alpha))
            for k in range(k_count)
        }
        controlled = (
            set(range(k_count))
            if controlled_classes is None
            else {int(k) for k in controlled_classes}
        )
        if not controlled.issubset(set(range(k_count))):
            raise ValueError("controlled_classes contains an invalid index")

        family_size = self.family_size or max(len(controlled), 1)
        family_delta = 1.0 - self.confidence
        controlled_delta = (
            family_delta / family_size if self.simultaneous else family_delta
        )
        self.lambdas = {}
        self.calibration_info = {}

        for k in range(k_count):
            alpha = alphas[k]
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha for class {k} must be between 0 and 1")
            positive = labels[:, k] == 1
            scores = np.clip(1.0 - probs[positive, k], 0.0, 1.0)
            n = int(scores.size)
            is_controlled = k in controlled
            delta = controlled_delta if is_controlled else family_delta

            if n == 0:
                lam, misses, ucb = 1.0, 0, 0.0
                status, vacuous = "include-always: no positives", True
            else:
                selected = None
                for lam_candidate in np.unique(
                    np.concatenate(([0.0], scores.astype(float), [1.0]))
                ):
                    misses_candidate = int(np.count_nonzero(scores > lam_candidate))
                    ucb_candidate = self.exact_upper_bound(
                        misses_candidate, n, delta
                    )
                    if ucb_candidate <= alpha:
                        selected = (
                            float(lam_candidate), misses_candidate, ucb_candidate
                        )
                        break
                if selected is None:
                    lam, misses = 1.0, 0
                    ucb = self.exact_upper_bound(0, n, delta)
                    status, vacuous = "include-always fallback", True
                else:
                    lam, misses, ucb = selected
                    status = (
                        "simultaneous exact-binomial RCPS"
                        if self.simultaneous and is_controlled
                        else "per-class exact-binomial RCPS"
                    )
                    vacuous = bool(lam >= 1.0 - 1e-15)

            minimal_lam = float(lam)
            guard = None
            if guard_lambdas is not None and k in guard_lambdas:
                guard = float(guard_lambdas[k])
                lam = max(float(lam), guard)
                if n:
                    misses = int(np.count_nonzero(scores > lam))
                    ucb = self.exact_upper_bound(misses, n, delta)
                if guard > minimal_lam:
                    status += " with conservative safeguard"

            self.lambdas[k] = float(lam)
            self.calibration_info[k] = {
                "n_cal": n,
                "alpha_target": alpha,
                "delta": float(delta),
                "confidence": float(1.0 - delta),
                "controlled": is_controlled,
                "family_size": family_size if is_controlled else None,
                "threshold": float(lam),
                "probability_threshold": float(1.0 - lam),
                "calibration_misses": int(misses),
                "calibration_fnr": float(misses / n) if n else 0.0,
                "exact_ucb": float(ucb),
                "vacuous": bool(vacuous),
                "guarantee_valid": bool(
                    is_controlled and (n == 0 or ucb <= alpha + 1e-12)
                ),
                "minimal_exact_threshold": minimal_lam,
                "guard_threshold": guard,
                "guard_applied": bool(guard is not None and guard > minimal_lam),
                "status": status,
            }
        return dict(self.lambdas)

    def guarded_copy(
        self,
        guard_lambdas: Mapping[int, float],
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
    ) -> "MultiLabelCRC":
        """Return a more-inclusive copy and recompute its calibration UCBs."""
        if not self.lambdas:
            raise RuntimeError("calibrate before calling guarded_copy")
        guarded = deepcopy(self)
        probs = np.asarray(probs_cal, dtype=float)
        labels = np.asarray(y_cal)
        for k, old_lam in self.lambdas.items():
            new_lam = max(float(old_lam), float(guard_lambdas.get(k, old_lam)))
            scores = np.clip(1.0 - probs[labels[:, k] == 1, k], 0.0, 1.0)
            n = int(scores.size)
            misses = int(np.count_nonzero(scores > new_lam)) if n else 0
            delta = guarded.calibration_info[k]["delta"]
            ucb = self.exact_upper_bound(misses, n, delta) if n else 0.0
            guarded.lambdas[k] = new_lam
            guarded.calibration_info[k].update(
                threshold=new_lam,
                probability_threshold=1.0 - new_lam,
                calibration_misses=misses,
                calibration_fnr=(misses / n if n else 0.0),
                exact_ucb=ucb,
                guard_threshold=float(guard_lambdas.get(k, old_lam)),
                guard_applied=new_lam > old_lam,
                status="exact-binomial RCPS with conservative safeguard",
            )
        return guarded

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probs, dtype=float)
        if probabilities.ndim != 2:
            raise ValueError("probs must be a 2-D array")
        if len(self.lambdas) != probabilities.shape[1]:
            raise RuntimeError("calibrate before prediction")
        thresholds = np.array(
            [1.0 - self.lambdas[k] for k in range(probabilities.shape[1])]
        )
        return (probabilities >= thresholds[None, :]).astype(np.int8)

    @staticmethod
    def compute_metrics(
        pred: np.ndarray, y_true: np.ndarray, class_names: Sequence[str]
    ) -> Dict:
        predictions = np.asarray(pred)
        labels = np.asarray(y_true)
        if predictions.shape != labels.shape or predictions.ndim != 2:
            raise ValueError("pred and y_true must be equal-shaped 2-D arrays")
        out = {"class_fnr": {}, "class_fpr": {}, "class_tpr": {},
               "class_set_pct": {}}
        for k, name in enumerate(class_names):
            pos = labels[:, k] == 1
            neg = labels[:, k] == 0
            if pos.any():
                tpr = float(predictions[pos, k].mean())
                out["class_tpr"][name] = tpr
                out["class_fnr"][name] = 1.0 - tpr
            if neg.any():
                out["class_fpr"][name] = float(predictions[neg, k].mean())
            out["class_set_pct"][name] = float(predictions[:, k].mean())
        sizes = predictions.sum(axis=1)
        out["mean_set_size"] = float(sizes.mean())
        out["median_set_size"] = float(np.median(sizes))
        out["joint_coverage"] = float(np.all(predictions >= labels, axis=1).mean())
        macro = float(np.mean(list(out["class_tpr"].values())))
        out["macro_class_coverage"] = macro
        out["marginal_coverage"] = macro
        return out


class LegacyHoeffdingCRC:
    """Submitted Hoeffding-plus-floor rule, retained for ablation only."""

    def __init__(self, default_alpha: float = 0.10, confidence: float = 0.95):
        self.default_alpha = default_alpha
        self.delta = 1.0 - confidence
        self.lambdas: Dict[int, float] = {}
        self.calibration_info: Dict[int, Dict] = {}

    def calibrate(
        self,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
        class_alphas: Optional[Mapping[int, float]] = None,
    ) -> Dict[int, float]:
        probs = np.asarray(probs_cal, dtype=float)
        labels = np.asarray(y_cal)
        self.lambdas = {}
        self.calibration_info = {}
        for k in range(probs.shape[1]):
            pos = labels[:, k] == 1
            n = int(pos.sum())
            alpha = float((class_alphas or {}).get(k, self.default_alpha))
            if n < 2:
                lam, floor_active, correction = 1.0, False, 0.0
            else:
                correction = float(np.sqrt(np.log(1.0 / self.delta) / (2.0 * n)))
                raw = alpha - correction
                floor_active = raw < 0.005
                alpha_eff = max(0.005, raw)
                q = min(float(np.ceil((n + 1) * (1 - alpha_eff)) / n), 1.0)
                lam = float(np.quantile(1.0 - probs[pos, k], q))
            self.lambdas[k] = lam
            self.calibration_info[k] = {
                "n_cal": n,
                "alpha_target": alpha,
                "correction": correction,
                "threshold": lam,
                "floor_active": floor_active,
                "guarantee_valid": not floor_active and n >= 2,
            }
        return dict(self.lambdas)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probs, dtype=float)
        thresholds = np.array(
            [1.0 - self.lambdas[k] for k in range(probabilities.shape[1])]
        )
        return (probabilities >= thresholds[None, :]).astype(np.int8)

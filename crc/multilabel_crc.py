"""Per-class exact-binomial risk control for multi-label classification.

The primary implementation uses one-sided Clopper-Pearson upper confidence
bounds for the binary false-negative loss.  A Bonferroni allocation can provide
simultaneous family-wise confidence across the controlled classes.

The submitted-paper Hoeffding-plus-floor rule is retained only as
``LegacyHoeffdingCRC`` for audit/ablation purposes; its 0.005 floor is not a
valid continuation of the stated Hoeffding proof when the floor is active.
"""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, Iterable, List, Mapping, Optional, Sequence

import numpy as np
from scipy.stats import beta


class MultiLabelCRC:
    """Exact-binomial per-class RCPS for binary false-negative loss.

    Parameters
    ----------
    default_alpha:
        Default class-specific FNR target.
    confidence:
        Confidence level. With ``simultaneous=True``, the failure probability
        is divided across the controlled family using Bonferroni.
    simultaneous:
        Whether to control the family of controlled classes simultaneously.
    family_size:
        Optional fixed family size. When omitted, it is inferred from
        ``controlled_classes`` during calibration.
    """

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
        """Return a one-sided ``1-delta`` Clopper-Pearson upper bound."""
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
        """Calibrate one monotone threshold per class using calibration data.

        ``guard_lambdas`` may supply pre-existing calibration-derived lambda
        values. The final lambda is the maximum of the exact-binomial lambda
        and the guard value. Because larger lambda values only enlarge the
        prediction set, this cannot increase FNR and therefore preserves the
        exact-binomial guarantee. No test labels are used in calibration.
        """
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
        for k, alpha in alphas.items():
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"alpha for class {k} must be between 0 and 1")

        controlled = (
            set(range(k_count))
            if controlled_classes is None
            else {int(k) for k in controlled_classes}
        )
        if not controlled.issubset(set(range(k_count))):
            raise ValueError("controlled_classes contains an invalid class index")

        family_size = self.family_size or max(len(controlled), 1)
        family_delta = 1.0 - self.confidence
        controlled_delta = (
            family_delta / family_size if self.simultaneous else family_delta
        )

        self.lambdas = {}
        self.calibration_info = {}

        for k in range(k_count):
            positive = labels[:, k] == 1
            scores = np.clip(1.0 - probs[positive, k], 0.0, 1.0)
            n = int(scores.size)
            is_controlled = k in controlled
            delta = controlled_delta if is_controlled else family_delta

            if n == 0:
                lam = 1.0
                misses = 0
                ucb = 0.0
                status = "include-always fallback: no calibration positives"
                vacuous = True
            else:
                candidates = np.unique(
                    np.concatenate(([0.0], scores.astype(np.float64), [1.0]))
                )
                selected = None
                previous_ucb = 1.0
                for candidate in candidates:
                    candidate_misses = int(np.count_nonzero(scores > candidate))
                    candidate_ucb = self.exact_upper_bound(
                        candidate_misses, n, delta
                    )
                    if candidate_ucb > previous_ucb + 1e-12:
                        raise RuntimeError("exact UCB must be monotone in lambda")
                    previous_ucb = candidate_ucb
                    if candidate_ucb <= alphas[k]:
                        selected = (
                            float(candidate), candidate_misses, candidate_ucb
                        )
                        break

                if selected is None:
                    # Include always gives a structural zero FNR. The finite-
                    # sample UCB may remain above a very small target, so mark
                    # it as vacuous rather than claiming a non-vacuous bound.
                    lam = 1.0
                    misses = 0
                    ucb = self.exact_upper_bound(0, n, delta)
                    status = "include-always fallback"
                    vacuous = True
                else:
                    lam, misses, ucb = selected
                    vacuous = bool(lam >= 1.0 - 1e-15)
                    status = (
                        "simultaneous exact-binomial RCPS"
                        if self.simultaneous and is_controlled
                        else "per-class exact-binomial RCPS"
                    )

            minimal_lam = float(lam)
            guard = None
            if guard_lambdas is not None and k in guard_lambdas:
                guard = float(guard_lambdas[k])
                lam = max(float(lam), guard)
                if n:
                    misses = int(np.count_nonzero(scores > lam))
                    ucb = self.exact_upper_bound(misses, n, delta)
                if guard > minimal_lam:
                    status += " with conservative calibration safeguard"

            guarantee_valid = bool(
                is_controlled
                and (n == 0 or float(ucb) <= alphas[k] + 1e-12)
            )
            self.lambdas[k] = float(lam)
            self.calibration_info[k] = {
                "n_cal": n,
                "alpha_target": alphas[k],
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
                "guarantee_valid": guarantee_valid,
                "minimal_exact_threshold": minimal_lam,
                "guard_threshold": guard,
                "guard_applied": bool(guard is not None and guard > minimal_lam),
                "status": status,
            }

        return dict(self.lambdas)

    def guarded_copy(self, guard_lambdas: Mapping[int, float], probs_cal: np.ndarray,
                     y_cal: np.ndarray) -> "MultiLabelCRC":
        """Return a more-inclusive copy while recomputing calibration UCBs."""
        if not self.lambdas:
            raise RuntimeError("calibrate before calling guarded_copy")
        guarded = deepcopy(self)
        probs = np.asarray(probs_cal, dtype=np.float64)
        labels = np.asarray(y_cal)
        for k, old_lam in self.lambdas.items():
            new_lam = max(float(old_lam), float(guard_lambdas.get(k, old_lam)))
            positive = labels[:, k] == 1
            scores = np.clip(1.0 - probs[positive, k], 0.0, 1.0)
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
                status="exact-binomial RCPS with conservative calibration safeguard",
            )
        return guarded

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probs, dtype=np.float64)
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
        if len(class_names) != predictions.shape[1]:
            raise ValueError("class_names length does not match array columns")

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
        # Historical compatibility only. This is not sample-level marginal CP coverage.
        out["marginal_coverage"] = macro
        return out

    @staticmethod
    def bootstrap_ci(
        pred: np.ndarray,
        y_true: np.ndarray,
        class_names: Sequence[str],
        n_boot: int = 1000,
        seed: int = 42,
    ) -> Dict:
        rng = np.random.default_rng(seed)
        predictions = np.asarray(pred)
        labels = np.asarray(y_true)
        fnr_boot = {name: [] for name in class_names}
        fpr_boot = {name: [] for name in class_names}
        for _ in range(n_boot):
            idx = rng.integers(0, len(labels), len(labels))
            metrics = MultiLabelCRC.compute_metrics(
                predictions[idx], labels[idx], class_names
            )
            for name in class_names:
                if name in metrics["class_fnr"]:
                    fnr_boot[name].append(metrics["class_fnr"][name])
                if name in metrics["class_fpr"]:
                    fpr_boot[name].append(metrics["class_fpr"][name])
        return {
            "fnr": {
                name: tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))
                for name, values in fnr_boot.items() if values
            },
            "fpr": {
                name: tuple(float(x) for x in np.percentile(values, [2.5, 97.5]))
                for name, values in fpr_boot.items() if values
            },
        }

    def print_calibration_report(self, class_names: Sequence[str]) -> None:
        print("\n" + "=" * 112)
        print("EXACT-BINOMIAL MULTI-LABEL RCPS CALIBRATION REPORT")
        print("=" * 112)
        print(
            f"{'Class':<14}{'n_pos':>8}{'target':>10}{'delta':>10}"
            f"{'misses':>10}{'UCB':>10}{'p-thr':>12}{'valid':>9}"
        )
        print("-" * 112)
        for k, info in self.calibration_info.items():
            print(
                f"{class_names[k]:<14}{info['n_cal']:>8}"
                f"{info['alpha_target']:>10.1%}{info['delta']:>10.4f}"
                f"{info['calibration_misses']:>10}{info['exact_ucb']:>10.2%}"
                f"{info['probability_threshold']:>12.4f}"
                f"{str(info['guarantee_valid']):>9}"
            )
        print("=" * 112)


class LegacyHoeffdingCRC:
    """Submitted-paper Hoeffding-plus-floor rule, for ablation only."""

    def __init__(self, default_alpha: float = 0.10, confidence: float = 0.95) -> None:
        self.default_alpha = default_alpha
        self.confidence = confidence
        self.delta = 1.0 - confidence
        self.lambdas: Dict[int, float] = {}
        self.calibration_info: Dict[int, Dict] = {}

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray,
                  class_alphas: Optional[Mapping[int, float]] = None) -> Dict[int, float]:
        probs = np.asarray(probs_cal, dtype=np.float64)
        labels = np.asarray(y_cal)
        if probs.shape != labels.shape or probs.ndim != 2:
            raise ValueError("probs_cal and y_cal must be equal-shaped 2-D arrays")
        self.lambdas = {}
        self.calibration_info = {}
        for k in range(probs.shape[1]):
            pos = labels[:, k] == 1
            n = int(pos.sum())
            alpha = float((class_alphas or {}).get(k, self.default_alpha))
            if n < 2:
                self.lambdas[k] = 1.0
                self.calibration_info[k] = {
                    "n_cal": n, "alpha_target": alpha, "threshold": 1.0,
                    "floor_active": False, "guarantee_valid": False,
                }
                continue
            correction = float(np.sqrt(np.log(1.0 / self.delta) / (2.0 * n)))
            raw = alpha - correction
            floor_active = raw < 0.005
            alpha_effective = max(0.005, raw)
            q = min(float(np.ceil((n + 1) * (1 - alpha_effective)) / n), 1.0)
            lam = float(np.quantile(1.0 - probs[pos, k], q))
            self.lambdas[k] = lam
            self.calibration_info[k] = {
                "n_cal": n, "alpha_target": alpha,
                "alpha_effective": alpha_effective, "correction": correction,
                "threshold": lam, "floor_active": floor_active,
                "guarantee_valid": not floor_active,
            }
        return dict(self.lambdas)

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probabilities = np.asarray(probs, dtype=np.float64)
        thresholds = np.array(
            [1.0 - self.lambdas[k] for k in range(probabilities.shape[1])]
        )
        return (probabilities >= thresholds[None, :]).astype(np.int8)

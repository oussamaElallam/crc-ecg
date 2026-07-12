
"""
Calibration-only exact-binomial RCPS experiment.

Run inside the existing Colab notebook/runtime AFTER the original verification
notebook has completed:

    %run -i /content/exact_binomial_rcps_patch.py

No neural-network training is performed. The script reuses the retained
calibration/test probabilities.
"""

import copy
import json
import os
import time
from typing import Dict, List, Optional

import numpy as np
from scipy.stats import beta


REQUIRED_VARIABLES = [
    "probs_cal_ptb", "ycal", "probs_te_ptb", "yte",
    "PTBXL_CLASSES", "class_alphas_ptb", "crc_ptb",
    "probs_cal_chap", "ycal_c", "probs_te_chap", "yte_c",
    "CHAPMAN_CLASSES", "CHAPMAN_ALPHAS", "crc_chap",
    "LegacyHoeffdingCRC", "OUT_DIR",
]

missing = [name for name in REQUIRED_VARIABLES if name not in globals()]
if missing:
    raise RuntimeError(
        "Run this patch in the SAME live runtime after the verification "
        "notebook has completed. Missing variables: " + ", ".join(missing)
    )


class ExactBinomialCRC:
    """
    Per-class RCPS calibration for binary FNR loss using a one-sided exact
    Clopper-Pearson/binomial upper confidence bound.

    The set parameter lambda is the maximum accepted nonconformity score:
        score = 1 - probability
        include class k when score <= lambda_k

    Larger lambda produces a larger prediction set and a lower FNR.
    """

    def __init__(
        self,
        confidence: float = 0.95,
        class_deltas: Optional[Dict[int, float]] = None,
    ):
        if not 0.0 < confidence < 1.0:
            raise ValueError("confidence must be between 0 and 1")
        self.confidence = float(confidence)
        self.default_delta = 1.0 - self.confidence
        self.class_deltas = class_deltas or {}
        self.lambdas: Dict[int, float] = {}
        self.calibration_info: Dict[int, Dict] = {}

    @staticmethod
    def exact_upper_bound(misses: int, n: int, delta: float) -> float:
        """One-sided (1-delta) Clopper-Pearson upper bound."""
        if n <= 0:
            return 1.0
        if misses < 0 or misses > n:
            raise ValueError("misses must lie between 0 and n")
        if misses == n:
            return 1.0
        return float(beta.ppf(1.0 - delta, misses + 1, n - misses))

    def calibrate(
        self,
        probs_cal: np.ndarray,
        y_cal: np.ndarray,
        class_alphas: Dict[int, float],
    ) -> Dict[int, float]:
        probs_cal = np.asarray(probs_cal, dtype=np.float64)
        y_cal = np.asarray(y_cal)

        if probs_cal.ndim != 2 or y_cal.ndim != 2:
            raise ValueError("probs_cal and y_cal must be 2-D")
        if probs_cal.shape != y_cal.shape:
            raise ValueError(
                f"Shape mismatch: probs_cal={probs_cal.shape}, "
                f"y_cal={y_cal.shape}"
            )
        if not np.isfinite(probs_cal).all():
            raise ValueError("probs_cal contains NaN or infinite values")

        self.lambdas = {}
        self.calibration_info = {}
        K = probs_cal.shape[1]

        for k in range(K):
            alpha = float(class_alphas[k])
            delta = float(self.class_deltas.get(k, self.default_delta))
            if not 0.0 < alpha < 1.0:
                raise ValueError(f"Invalid alpha for class {k}: {alpha}")
            if not 0.0 < delta < 1.0:
                raise ValueError(f"Invalid delta for class {k}: {delta}")

            positive = y_cal[:, k] == 1
            scores = np.clip(1.0 - probs_cal[positive, k], 0.0, 1.0)
            n = int(scores.size)

            if n == 0:
                self.lambdas[k] = 1.0
                self.calibration_info[k] = {
                    "n_cal": 0,
                    "alpha_target": alpha,
                    "delta": delta,
                    "confidence": 1.0 - delta,
                    "threshold": 1.0,
                    "probability_threshold": 0.0,
                    "calibration_misses": 0,
                    "calibration_fnr": None,
                    "exact_ucb": None,
                    "feasible": False,
                    "vacuous": True,
                    "guarantee_valid": True,
                    "status": "include-always fallback: structurally zero FNR",
                }
                continue

            # Candidate lambdas include all score breakpoints. Because the
            # exact binomial UCB is monotone in the number of misses, choosing
            # the first passing lambda implements monotone UCB calibration.
            candidates = np.unique(
                np.concatenate(([0.0], scores.astype(np.float64), [1.0]))
            )

            chosen = None
            previous_ucb = 1.0

            for lam in candidates:
                misses = int(np.count_nonzero(scores > lam))
                ucb = self.exact_upper_bound(misses, n, delta)

                # Numerical check of the expected monotonicity.
                if ucb > previous_ucb + 1e-12:
                    raise RuntimeError(
                        f"Non-monotone exact UCB encountered for class {k}"
                    )
                previous_ucb = ucb

                if ucb <= alpha:
                    chosen = (float(lam), misses, ucb)
                    break

            if chosen is None:
                # The include-always rule has FNR exactly zero by construction.
                lam, misses, ucb = 1.0, 0, self.exact_upper_bound(0, n, delta)
                status = (
                    "include-always fallback: structurally zero FNR; "
                    "sample UCB alone is above target"
                )
                vacuous = True
                guarantee_valid = True
            else:
                lam, misses, ucb = chosen
                vacuous = bool(lam >= 1.0 - 1e-15)
                guarantee_valid = True
                status = (
                    "non-vacuous exact-binomial RCPS"
                    if not vacuous
                    else "include-always exact-binomial RCPS"
                )

            self.lambdas[k] = lam
            self.calibration_info[k] = {
                "n_cal": n,
                "alpha_target": alpha,
                "delta": delta,
                "confidence": 1.0 - delta,
                "threshold": lam,
                "probability_threshold": float(1.0 - lam),
                "calibration_misses": int(misses),
                "calibration_fnr": float(misses / n),
                "exact_ucb": float(ucb),
                "feasible": True,
                "vacuous": vacuous,
                "guarantee_valid": guarantee_valid,
                "status": status,
            }

        return self.lambdas

    def predict(self, probs: np.ndarray) -> np.ndarray:
        probs = np.asarray(probs, dtype=np.float64)
        thresholds = np.array(
            [1.0 - self.lambdas[k] for k in range(probs.shape[1])],
            dtype=np.float64,
        )
        return (probs >= thresholds[None, :]).astype(np.int8)


def make_guarded_exact(
    exact_model: ExactBinomialCRC,
    legacy_model,
    probs_cal: np.ndarray,
    y_cal: np.ndarray,
) -> ExactBinomialCRC:
    """
    Return a valid exact-binomial rule that is never less inclusive than the
    submitted legacy rule. This is a pre-specified calibration-only safeguard,
    not test-set tuning.
    """
    guarded = copy.deepcopy(exact_model)

    for k in guarded.lambdas:
        exact_lam = float(guarded.lambdas[k])
        legacy_lam = float(legacy_model.lambdas[k])
        final_lam = max(exact_lam, legacy_lam)

        guarded.lambdas[k] = final_lam
        info = guarded.calibration_info[k]
        positive = np.asarray(y_cal)[:, k] == 1
        scores = np.clip(
            1.0 - np.asarray(probs_cal)[positive, k],
            0.0,
            1.0,
        )
        n = int(scores.size)
        misses = int(np.count_nonzero(scores > final_lam))
        ucb = guarded.exact_upper_bound(misses, n, info["delta"])

        info.update({
            "threshold": final_lam,
            "probability_threshold": float(1.0 - final_lam),
            "calibration_misses": misses,
            "calibration_fnr": float(misses / n) if n else None,
            "exact_ucb": float(ucb) if n else None,
            "legacy_safeguard_applied": bool(legacy_lam > exact_lam),
            "legacy_threshold": legacy_lam,
            "exact_minimal_threshold": exact_lam,
            "vacuous": bool(final_lam >= 1.0 - 1e-15),
            "guarantee_valid": True,
            "status": (
                "exact-binomial RCPS with pre-specified legacy safeguard"
            ),
        })

        if n and ucb > info["alpha_target"] + 1e-12:
            raise RuntimeError(
                f"Guarded threshold unexpectedly failed the exact UCB "
                f"condition for class {k}"
            )

    return guarded


def metrics_for(model, probs_test, y_test, class_names):
    pred = model.predict(probs_test)
    metrics = LegacyHoeffdingCRC.compute_metrics(pred, y_test, class_names)
    return pred, metrics


def print_exact_table(
    title: str,
    model: ExactBinomialCRC,
    metrics: Dict,
    class_names: List[str],
    alpha_map: Dict[int, float],
):
    print("\n" + "=" * 116)
    print(title)
    print("=" * 116)
    print(
        f"{'Class':<11}{'n+':>7}{'m_cal':>8}{'UCB':>10}"
        f"{'target':>10}{'p-thr':>11}{'test FNR':>11}"
        f"{'test FPR':>11}{'met?':>7}"
    )
    print("-" * 116)

    for k, name in enumerate(class_names):
        info = model.calibration_info[k]
        fnr = metrics["class_fnr"].get(name, np.nan)
        fpr = metrics["class_fpr"].get(name, np.nan)
        target = float(alpha_map[k])
        met = "YES" if np.isfinite(fnr) and fnr <= target else "NO"
        print(
            f"{name:<11}{info['n_cal']:>7}{info['calibration_misses']:>8}"
            f"{info['exact_ucb']:>10.2%}{target:>10.1%}"
            f"{info['probability_threshold']:>11.4f}"
            f"{fnr:>11.2%}{fpr:>11.2%}{met:>7}"
        )

    print("-" * 116)
    print(
        f"Mean set size: {metrics['mean_set_size']:.3f} | "
        f"Joint coverage: {metrics['joint_coverage']:.2%} | "
        f"Macro class coverage: {metrics['macro_class_coverage']:.2%}"
    )


def target_count(metrics, class_names, alpha_map, excluded_name):
    controlled = [name for name in class_names if name != excluded_name]
    met = sum(
        metrics["class_fnr"][name]
        <= float(alpha_map[class_names.index(name)])
        for name in controlled
    )
    return int(met), len(controlled)


def nonvacuous_count(model, class_names, excluded_name):
    controlled_indices = [
        k for k, name in enumerate(class_names)
        if name != excluded_name
    ]
    return sum(
        model.calibration_info[k]["guarantee_valid"]
        and not model.calibration_info[k]["vacuous"]
        for k in controlled_indices
    )


ptb_alpha_map = {
    k: float(class_alphas_ptb[k])
    for k in range(len(PTBXL_CLASSES))
}
chap_alpha_map = {
    k: float(CHAPMAN_ALPHAS[name])
    for k, name in enumerate(CHAPMAN_CLASSES)
}

# -------------------------------------------------------------------------
# A. Per-class 95% exact-binomial RCPS
# -------------------------------------------------------------------------
exact_pc_ptb = ExactBinomialCRC(confidence=0.95)
exact_pc_ptb.calibrate(probs_cal_ptb, ycal, ptb_alpha_map)
pred_exact_pc_ptb, metrics_exact_pc_ptb = metrics_for(
    exact_pc_ptb, probs_te_ptb, yte, PTBXL_CLASSES
)

exact_pc_chap = ExactBinomialCRC(confidence=0.95)
exact_pc_chap.calibrate(probs_cal_chap, ycal_c, chap_alpha_map)
pred_exact_pc_chap, metrics_exact_pc_chap = metrics_for(
    exact_pc_chap, probs_te_chap, yte_c, CHAPMAN_CLASSES
)

print_exact_table(
    "PTB-XL — exact-binomial RCPS, 95% per class",
    exact_pc_ptb,
    metrics_exact_pc_ptb,
    PTBXL_CLASSES,
    ptb_alpha_map,
)
print_exact_table(
    "Chapman — exact-binomial RCPS, 95% per class",
    exact_pc_chap,
    metrics_exact_pc_chap,
    CHAPMAN_CLASSES,
    chap_alpha_map,
)

# -------------------------------------------------------------------------
# B. Simultaneous 95% control over the nine FNR-controlled classes
#    via Bonferroni: delta_k = 0.05 / 9.
# -------------------------------------------------------------------------
FAMILY_SIZE = 9
family_delta = 0.05 / FAMILY_SIZE

ptb_fwer_deltas = {
    k: (family_delta if name != "NORM" else 0.05)
    for k, name in enumerate(PTBXL_CLASSES)
}
chap_fwer_deltas = {
    k: (family_delta if name != "Normal" else 0.05)
    for k, name in enumerate(CHAPMAN_CLASSES)
}

exact_fwer_ptb = ExactBinomialCRC(
    confidence=0.95,
    class_deltas=ptb_fwer_deltas,
)
exact_fwer_ptb.calibrate(probs_cal_ptb, ycal, ptb_alpha_map)
pred_exact_fwer_ptb, metrics_exact_fwer_ptb = metrics_for(
    exact_fwer_ptb, probs_te_ptb, yte, PTBXL_CLASSES
)

exact_fwer_chap = ExactBinomialCRC(
    confidence=0.95,
    class_deltas=chap_fwer_deltas,
)
exact_fwer_chap.calibrate(probs_cal_chap, ycal_c, chap_alpha_map)
pred_exact_fwer_chap, metrics_exact_fwer_chap = metrics_for(
    exact_fwer_chap, probs_te_chap, yte_c, CHAPMAN_CLASSES
)

print_exact_table(
    "PTB-XL — exact-binomial RCPS, simultaneous 95% family-wise",
    exact_fwer_ptb,
    metrics_exact_fwer_ptb,
    PTBXL_CLASSES,
    ptb_alpha_map,
)
print_exact_table(
    "Chapman — exact-binomial RCPS, simultaneous 95% family-wise",
    exact_fwer_chap,
    metrics_exact_fwer_chap,
    CHAPMAN_CLASSES,
    chap_alpha_map,
)

# -------------------------------------------------------------------------
# C. Conservative exact-binomial safeguard.
#    This is never less inclusive than the legacy submitted rule.
# -------------------------------------------------------------------------
guarded_fwer_ptb = make_guarded_exact(
    exact_fwer_ptb, crc_ptb, probs_cal_ptb, ycal
)
pred_guarded_fwer_ptb, metrics_guarded_fwer_ptb = metrics_for(
    guarded_fwer_ptb, probs_te_ptb, yte, PTBXL_CLASSES
)

guarded_fwer_chap = make_guarded_exact(
    exact_fwer_chap, crc_chap, probs_cal_chap, ycal_c
)
pred_guarded_fwer_chap, metrics_guarded_fwer_chap = metrics_for(
    guarded_fwer_chap, probs_te_chap, yte_c, CHAPMAN_CLASSES
)

print_exact_table(
    "PTB-XL — simultaneous exact-binomial RCPS + legacy safeguard",
    guarded_fwer_ptb,
    metrics_guarded_fwer_ptb,
    PTBXL_CLASSES,
    ptb_alpha_map,
)
print_exact_table(
    "Chapman — simultaneous exact-binomial RCPS + legacy safeguard",
    guarded_fwer_chap,
    metrics_guarded_fwer_chap,
    CHAPMAN_CLASSES,
    chap_alpha_map,
)


# Optional distribution-shift metrics with the exact thresholds.
shift_results = {}
if all(
    name in globals()
    for name in [
        "probs_shift", "y_shift", "harmonized_names",
        "harmonized_ptb_indices"
    ]
):
    for label, model in [
        ("per_class_exact", exact_pc_ptb),
        ("simultaneous_exact", exact_fwer_ptb),
        ("simultaneous_guarded", guarded_fwer_ptb),
    ]:
        shift_thresholds = np.array([
            1.0 - model.lambdas[index]
            for index in harmonized_ptb_indices
        ])
        shift_pred = (
            np.asarray(probs_shift)
            >= shift_thresholds[None, :]
        ).astype(np.int8)
        shift_results[label] = LegacyHoeffdingCRC.compute_metrics(
            shift_pred,
            y_shift,
            harmonized_names,
        )


def serializable(value):
    if isinstance(value, dict):
        return {str(k): serializable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [serializable(v) for v in value]
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, (np.integer,)):
        return int(value)
    if isinstance(value, (np.floating,)):
        return float(value)
    return value


pc_ptb_met, pc_ptb_total = target_count(
    metrics_exact_pc_ptb, PTBXL_CLASSES, ptb_alpha_map, "NORM"
)
pc_chap_met, pc_chap_total = target_count(
    metrics_exact_pc_chap, CHAPMAN_CLASSES, chap_alpha_map, "Normal"
)
fw_ptb_met, fw_ptb_total = target_count(
    metrics_exact_fwer_ptb, PTBXL_CLASSES, ptb_alpha_map, "NORM"
)
fw_chap_met, fw_chap_total = target_count(
    metrics_exact_fwer_chap, CHAPMAN_CLASSES, chap_alpha_map, "Normal"
)
guard_ptb_met, _ = target_count(
    metrics_guarded_fwer_ptb, PTBXL_CLASSES, ptb_alpha_map, "NORM"
)
guard_chap_met, _ = target_count(
    metrics_guarded_fwer_chap, CHAPMAN_CLASSES, chap_alpha_map, "Normal"
)

summary = {
    "meta": {
        "date": time.strftime("%Y-%m-%d"),
        "method": "exact-binomial monotone UCB RCPS",
        "training_reused": True,
        "test_set_not_used_for_threshold_selection": True,
        "family_size": FAMILY_SIZE,
        "family_delta": family_delta,
    },
    "per_class_95": {
        "ptbxl": {
            "calibration": exact_pc_ptb.calibration_info,
            "metrics": metrics_exact_pc_ptb,
            "empirical_targets_met": pc_ptb_met,
            "target_count": pc_ptb_total,
            "nonvacuous_guarantees": nonvacuous_count(
                exact_pc_ptb, PTBXL_CLASSES, "NORM"
            ),
        },
        "chapman": {
            "calibration": exact_pc_chap.calibration_info,
            "metrics": metrics_exact_pc_chap,
            "empirical_targets_met": pc_chap_met,
            "target_count": pc_chap_total,
            "nonvacuous_guarantees": nonvacuous_count(
                exact_pc_chap, CHAPMAN_CLASSES, "Normal"
            ),
        },
        "combined_empirical_targets_met": pc_ptb_met + pc_chap_met,
        "combined_nonvacuous_guarantees": (
            nonvacuous_count(exact_pc_ptb, PTBXL_CLASSES, "NORM")
            + nonvacuous_count(exact_pc_chap, CHAPMAN_CLASSES, "Normal")
        ),
    },
    "simultaneous_95_bonferroni": {
        "ptbxl": {
            "calibration": exact_fwer_ptb.calibration_info,
            "metrics": metrics_exact_fwer_ptb,
            "empirical_targets_met": fw_ptb_met,
            "target_count": fw_ptb_total,
            "nonvacuous_guarantees": nonvacuous_count(
                exact_fwer_ptb, PTBXL_CLASSES, "NORM"
            ),
        },
        "chapman": {
            "calibration": exact_fwer_chap.calibration_info,
            "metrics": metrics_exact_fwer_chap,
            "empirical_targets_met": fw_chap_met,
            "target_count": fw_chap_total,
            "nonvacuous_guarantees": nonvacuous_count(
                exact_fwer_chap, CHAPMAN_CLASSES, "Normal"
            ),
        },
        "combined_empirical_targets_met": fw_ptb_met + fw_chap_met,
        "combined_nonvacuous_guarantees": (
            nonvacuous_count(exact_fwer_ptb, PTBXL_CLASSES, "NORM")
            + nonvacuous_count(exact_fwer_chap, CHAPMAN_CLASSES, "Normal")
        ),
    },
    "simultaneous_95_guarded": {
        "ptbxl": {
            "calibration": guarded_fwer_ptb.calibration_info,
            "metrics": metrics_guarded_fwer_ptb,
            "empirical_targets_met": guard_ptb_met,
            "target_count": fw_ptb_total,
        },
        "chapman": {
            "calibration": guarded_fwer_chap.calibration_info,
            "metrics": metrics_guarded_fwer_chap,
            "empirical_targets_met": guard_chap_met,
            "target_count": fw_chap_total,
        },
        "combined_empirical_targets_met": guard_ptb_met + guard_chap_met,
        "combined_nonvacuous_guarantees": (
            nonvacuous_count(guarded_fwer_ptb, PTBXL_CLASSES, "NORM")
            + nonvacuous_count(guarded_fwer_chap, CHAPMAN_CLASSES, "Normal")
        ),
    },
    "distribution_shift": shift_results,
}

summary = serializable(summary)

output_path = os.path.join(
    OUT_DIR,
    "exact_binomial_rcps_results_"
    f"{time.strftime('%Y%m%d_%H%M')}.json",
)
with open(output_path, "w", encoding="utf-8") as handle:
    json.dump(summary, handle, indent=2)

print("\n" + "#" * 90)
print("EXACT-BINOMIAL RCPS HEADLINE")
print("#" * 90)
print(
    "Per-class 95%: empirical targets "
    f"{pc_ptb_met + pc_chap_met}/9; "
    "non-vacuous guarantees "
    f"{summary['per_class_95']['combined_nonvacuous_guarantees']}/9"
)
print(
    "Simultaneous 95% Bonferroni: empirical targets "
    f"{fw_ptb_met + fw_chap_met}/9; "
    "non-vacuous guarantees "
    f"{summary['simultaneous_95_bonferroni']['combined_nonvacuous_guarantees']}/9"
)
print(
    "Simultaneous 95% guarded: empirical targets "
    f"{guard_ptb_met + guard_chap_met}/9; "
    "non-vacuous guarantees "
    f"{summary['simultaneous_95_guarded']['combined_nonvacuous_guarantees']}/9"
)
print(f"Saved: {output_path}")

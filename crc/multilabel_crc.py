"""Multi-Label Conformal Risk Control with Hoeffding finite-sample correction.

Reference: El Allam & Hamlich (2026), "Per-Class Conformal Risk Control for
Multi-Label ECG Classification."

Citation acknowledgement: the Hoeffding-style finite-sample correction itself
appears throughout the conformal-prediction literature — Vovk et al. (2005),
Romano et al. (2020), Bates et al. (2021), Angelopoulos and Bates (2021).
Our contribution is its application to multi-label ECG classification with
clinically-justified per-class alpha targets.
"""

from typing import Dict, List, Optional
import numpy as np


class MultiLabelCRC:
    """
    Multi-label per-class Conformal Risk Control with finite-sample correction.

    Each class k is treated as an independent binary problem:
        score_k(x) = 1 - p_k(x)
    Calibrated on positives for class k: {x_i : y_{i,k} = 1}.

    Hoeffding correction (Vovk 2005, Bates 2021, Romano 2020, Angelopoulos 2021):
        alpha_effective = alpha_k - sqrt(log(1/delta) / (2 n_k))
    gives a high-probability (1 - delta) guarantee that FNR_k <= alpha_k.
    """

    def __init__(self, default_alpha: float = 0.10, confidence: float = 0.95,
                 finite_sample_correction: bool = True):
        self.default_alpha = default_alpha
        self.confidence = confidence
        self.delta = 1 - confidence
        self.finite_sample_correction = finite_sample_correction
        self.lambdas: Dict[int, float] = {}
        self.calibration_info: Dict[int, Dict] = {}

    # ---------------- Calibration ----------------

    def calibrate(self, probs_cal: np.ndarray, y_cal: np.ndarray,
                  class_alphas: Optional[Dict[int, float]] = None) -> Dict[int, float]:
        """
        Args:
            probs_cal: [n_cal, K] sigmoid probabilities
            y_cal: [n_cal, K] multi-hot true labels
            class_alphas: {k: alpha_k} per-class FNR targets
        """
        n_cal, K = probs_cal.shape
        if class_alphas is None:
            class_alphas = {k: self.default_alpha for k in range(K)}

        self.lambdas = {}
        self.calibration_info = {}

        for k in range(K):
            pos_mask = y_cal[:, k] == 1
            n_k = int(pos_mask.sum())
            alpha_k = class_alphas.get(k, self.default_alpha)

            if n_k < 2:
                self.lambdas[k] = 1.0  # no positives -> include always (vacuous)
                self.calibration_info[k] = dict(
                    n_cal=n_k, alpha_target=alpha_k, alpha_effective=alpha_k,
                    correction=0.0, threshold=1.0, vacuous=True,
                )
                continue

            # Nonconformity scores on positives
            scores_k = 1 - probs_cal[pos_mask, k]

            # Hoeffding correction
            if self.finite_sample_correction and n_k >= 10:
                correction = np.sqrt(np.log(1 / self.delta) / (2 * n_k))
                alpha_effective = max(0.005, alpha_k - correction)
            else:
                correction = 0.0
                alpha_effective = alpha_k

            # Conformal quantile (Romano et al. 2020)
            q_level = np.ceil((n_k + 1) * (1 - alpha_effective)) / n_k
            q_level = float(min(q_level, 1.0))
            lam_k = float(np.quantile(scores_k, q_level))

            self.lambdas[k] = lam_k
            self.calibration_info[k] = dict(
                n_cal=n_k, alpha_target=alpha_k, alpha_effective=alpha_effective,
                correction=correction, threshold=lam_k, vacuous=False,
            )

        return self.lambdas

    # ---------------- Prediction ----------------

    def predict(self, probs: np.ndarray) -> np.ndarray:
        """Return multi-hot prediction array [n, K]."""
        K = probs.shape[1]
        thr = np.array([1 - self.lambdas[k] for k in range(K)])  # include if p_k >= 1 - lambda_k
        return (probs >= thr[None, :]).astype(int)

    # ---------------- Metrics ----------------

    @staticmethod
    def compute_metrics(pred: np.ndarray, y_true: np.ndarray,
                        class_names: List[str]) -> Dict:
        """
        FNR_k, FPR_k, set size, joint coverage, marginal coverage.
        """
        n, K = pred.shape
        out = dict(class_fnr={}, class_fpr={}, class_tpr={}, class_set_pct={})

        for k in range(K):
            pos = y_true[:, k] == 1
            neg = y_true[:, k] == 0
            if pos.sum() > 0:
                out['class_fnr'][class_names[k]] = float(1 - pred[pos, k].mean())
                out['class_tpr'][class_names[k]] = float(pred[pos, k].mean())
            if neg.sum() > 0:
                out['class_fpr'][class_names[k]] = float(pred[neg, k].mean())
            out['class_set_pct'][class_names[k]] = float(pred[:, k].mean())

        set_sizes = pred.sum(axis=1)
        joint_covered = np.all((pred >= y_true), axis=1)  # all positives captured
        out['mean_set_size'] = float(set_sizes.mean())
        out['median_set_size'] = float(np.median(set_sizes))
        out['joint_coverage'] = float(joint_covered.mean())
        out['marginal_coverage'] = float(np.mean(list(out['class_tpr'].values())))
        return out

    @staticmethod
    def bootstrap_ci(pred: np.ndarray, y_true: np.ndarray, class_names: List[str],
                     n_boot: int = 1000, seed: int = 42) -> Dict:
        """95% bootstrap CIs on per-class FNR and FPR."""
        rng = np.random.default_rng(seed)
        n = len(y_true)
        fnr_boot = {c: [] for c in class_names}
        fpr_boot = {c: [] for c in class_names}
        for _ in range(n_boot):
            idx = rng.integers(0, n, n)
            m = MultiLabelCRC.compute_metrics(pred[idx], y_true[idx], class_names)
            for c in class_names:
                if c in m['class_fnr']:
                    fnr_boot[c].append(m['class_fnr'][c])
                if c in m['class_fpr']:
                    fpr_boot[c].append(m['class_fpr'][c])
        ci = dict(fnr={}, fpr={})
        for c in class_names:
            if fnr_boot[c]:
                ci['fnr'][c] = (float(np.percentile(fnr_boot[c], 2.5)),
                                float(np.percentile(fnr_boot[c], 97.5)))
            if fpr_boot[c]:
                ci['fpr'][c] = (float(np.percentile(fpr_boot[c], 2.5)),
                                float(np.percentile(fpr_boot[c], 97.5)))
        return ci

    def print_calibration_report(self, class_names: List[str]):
        print("\n" + "="*78)
        print("MULTI-LABEL CRC CALIBRATION REPORT (Hoeffding-corrected, 95% conf.)")
        print("="*78)
        print(f"{'Class':<10}{'n_pos':>8}{'alpha':>10}{'alpha_eff':>12}"
              f"{'correction':>12}{'lambda':>10}")
        print("-"*78)
        for k, info in self.calibration_info.items():
            print(f"{class_names[k]:<10}{info['n_cal']:>8}{info['alpha_target']:>10.1%}"
                  f"{info['alpha_effective']:>12.1%}{info['correction']:>12.4f}"
                  f"{info['threshold']:>10.4f}")
        print("="*78)

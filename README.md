# Per-Class Conformal Risk Control for Multi-Label ECG Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oussamaElallam/crc-ecg/blob/main/multilabel_crc_ecg.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)

Reproducibility repository for the paper *"Per-Class Conformal Risk Control for Multi-Label ECG Classification: Achieving Clinically-Justified False Negative Rate Guarantees"* by Oussama El Allam and Mohamed Hamlich (2026).

**Click the "Open In Colab" badge above to run the full reproduction notebook in your browser — no local setup required.**

## What's in this repository

```
crc-ecg/
├── crc/                                       # Standalone Python module
│   ├── __init__.py
│   ├── multilabel_crc.py                      # MultiLabelCRC class
│   └── baselines.py                           # multi-label baselines
├── multilabel_crc_ecg.ipynb     # End-to-end reproduction notebook
├── example_results.json                       # Sample output (all numbers from the paper)
├── figures/                                   # Generated figures (PNG, 200 dpi)
├── requirements.txt
├── LICENSE                                    # MIT
├── CITATION.cff
└── README.md
```

## Quick start — use the CRC class directly

```python
from crc import MultiLabelCRC

# probs_cal:  [n_cal, K] sigmoid probabilities from your model on calibration data
# y_cal:      [n_cal, K] multi-hot true labels
# class_alphas: {k: alpha_k}  per-class FNR targets (e.g. MI -> 0.05, HYP -> 0.15)

crc = MultiLabelCRC(confidence=0.95, finite_sample_correction=True)
crc.calibrate(probs_cal, y_cal, class_alphas)

# At inference:
predictions = crc.predict(probs_test)
metrics = MultiLabelCRC.compute_metrics(predictions, y_test, class_names)
ci = MultiLabelCRC.bootstrap_ci(predictions, y_test, class_names, n_boot=500)
```

## Full reproduction of the paper

### 1. Environment

Python 3.10+, GPU recommended (Colab T4 / A100 sufficient).

```bash
pip install -r requirements.txt
```

### 2. Data

Both datasets are public. Download them once and point the notebook at them:

| Dataset | Source | DOI |
|---|---|---|
| **PTB-XL v1.0.3** | https://physionet.org/content/ptb-xl/1.0.3/ | 10.13026/x4td-x982 |
| **Chapman-Shaoxing** | https://www.kaggle.com/datasets/erarayamorenzomuten/chapmanshaoxing-12lead-ecg-database | 10.13026/wgex-er52 |

### 3. Run

Click the **"Open In Colab"** badge at the top of this page, or launch locally:

```bash
jupyter notebook multilabel_crc_ecg.ipynb
```

Total runtime: ~30 min on T4, ~12 min on A100.

The notebook writes all results to a `CRC_Revision/` directory containing a JSON of every reported number and PNGs of every figure. An example output is included in `example_results.json` and `figures/` for reference.

## Method in one paragraph

For each diagnostic class `k`, treat the problem as an independent binary calibration: (1) compute nonconformity scores `s_i = 1 − p̂_k(x_i)` on calibration positives; (2) apply the Hoeffding finite-sample correction `ε_k = √(log(1/δ) / (2 n_k))`, effective target `α'_k = max(0.005, α_k − ε_k)`; (3) set `λ_k` as the `⌈(n_k + 1)(1 − α'_k)⌉ / n_k` quantile of the class-k scores; (4) at inference, include class `k` iff `p̂_k(x) ≥ 1 − λ_k`.

The Hoeffding correction is a standard device from the conformal-prediction literature (Vovk et al. 2005; Romano et al. 2020; Bates et al. 2021; Angelopoulos & Bates 2021). Our contribution is its application — together with clinically-justified per-class α targets derived from FDA-cleared device benchmarks and AHA guidelines — to multi-label ECG classification.

## Headline results

| Dataset | Targets met (CRC) | Targets met (Standard CP) |
|---|---|---|
| PTB-XL | **4 / 4** (CD, HYP, MI, STTC) | 2 / 4 |
| Chapman-Shaoxing | **5 / 5** (AF, CD, PAC_PVC, ST, Other) | 2 / 5 |
| **Combined** | **9 / 9** | **4 / 9** |

Critical-class FNRs (target ≤ 5%): **MI 1.1%, STTC 1.4%, AF 0.0%, ST 0.0%**.

23-PTB-XL-subclass analysis: **17 / 17** feasible subclasses (n_pos ≥ 30) meet the 10% target; 6 inadequately-powered subclasses are explicitly flagged by the method's feasibility check.

Distribution shift (train PTB-XL → test Chapman, harmonised labels): marginal coverage drops 94.3% → 78.3%, confirming exchangeability is binding.

## Citation

```bibtex
@article{ElAllam2026CRC,
  title   = {Per-Class Conformal Risk Control for Multi-Label ECG Classification:
             Achieving Clinically-Justified False Negative Rate Guarantees},
  author  = {El Allam, Oussama and Hamlich, Mohamed},
  journal = {Physiological Measurement},
  year    = {2026}
}
```

## License

Code: MIT (see `LICENSE`). Datasets retain their original licenses: PTB-XL is CC BY 4.0, Chapman-Shaoxing is ODC-By v1.0.

## Contact

Oussama El Allam — elallamoussama7@gmail.com
Complex Cyber-Physical Systems Laboratory, ENSAM Casablanca, University Hassan II.

# Per-Class Exact-Binomial Risk Control for Multi-Label ECG Classification

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/oussamaElallam/crc-ecg/blob/main/multilabel_crc_ecg.ipynb)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/)

Reproducibility repository for *“Per-Class Conformal Risk Control for Multi-Label ECG Classification: Achieving Clinically-Justified False Negative Rate Guarantees”* by Oussama El Allam and Mohamed Hamlich.

## Corrected primary method

For each diagnostic class, the false-negative loss is binary. We calibrate the largest admissible probability threshold using a one-sided exact binomial (Clopper–Pearson) upper confidence bound. For the nine FNR-controlled classes, the total failure probability `0.05` is divided by nine using Bonferroni, yielding simultaneous 95% family-wise control.

The reported operating point also applies a conservative calibration safeguard: the final nonconformity threshold is never smaller than the earlier calibration-derived threshold. Enlarging the prediction set cannot increase FNR, so the exact-binomial condition is preserved. The held-out test set is not used to select thresholds.

The earlier Hoeffding rule with `max(0.005, alpha - epsilon)` is retained only as a transparent audit/ablation. When that floor binds, it is not presented as a valid continuation of the Hoeffding proof.

## Headline corrected results

| Dataset | Exact-binomial RCPS | Uniform Standard CP |
|---|---:|---:|
| PTB-XL | **4 / 4** targets | 1 / 4 |
| Chapman-Shaoxing | **5 / 5** targets | 4 / 5 |
| **Combined** | **9 / 9** | **5 / 9** |

The guarded exact-binomial method has **9/9 non-vacuous simultaneous guarantees** and meets **9/9 held-out empirical targets**.

Critical-class held-out FNRs at the reported operating point are approximately:

- MI: **0.00%** (target 5%)
- STTC: **1.04%** (target 5%)
- AF: **0.30%** (target 5%)
- ST: **0.47%** (target 5%)

This sensitivity comes with a substantial specificity/workload cost, especially on Chapman. The repository reports every per-class FPR and prediction-set size; the method should be interpreted as high-sensitivity risk control, not autonomous diagnosis. Publication figures are regenerated from the committed result files and are not versioned, preventing stale images from diverging from the code.

## Corrected data protocol

- PTB-XL retained records: **21,388**
- PTB-XL split: 15,046 train / 3,144 calibration / 3,198 test
- PTB-XL partitions are patient-disjoint
- Signal normalization is fitted only on the model-fitting partition
- Chapman retained records: **10,199**
- Cross-dataset evaluation applies PTB-XL fitting statistics to Chapman signals
- Co-occurrence matrices use `int64` accumulation to avoid integer overflow

The cross-dataset PTB-XL→Chapman experiment drops from approximately **94.3% to 45.5% macro class coverage**, illustrating that the in-distribution guarantee does not transport automatically under substantial covariate and label-definition shift.

## Repository layout

```text
crc-ecg/
├── crc/
│   ├── __init__.py
│   ├── multilabel_crc.py       # exact-binomial RCPS + legacy audit class
│   └── baselines.py            # fixed, F1, uniform CP, class-matched CP
├── multilabel_crc_ecg.ipynb    # complete corrected workflow
├── example_results.json        # concise corrected result manifest
├── results/
│   ├── verification_results_20260712_1446.json
│   └── exact_binomial_rcps_results_20260712_1459.json
├── figures/                    # generated locally (not versioned)
├── scripts/
│   └── exact_binomial_rcps_patch.py
├── tests/
├── REPRODUCIBILITY_AUDIT.md
├── requirements.txt
└── LICENSE
```

## Quick start

```bash
pip install -r requirements.txt
python -m unittest discover -s tests -v
```

```python
from crc import MultiLabelCRC

# FNR-controlled class indices only; exclude descriptive Normal/NORM classes.
controlled = [0, 1, 2, 4]
class_alphas = {0: 0.10, 1: 0.15, 2: 0.05, 3: 0.15, 4: 0.05}

rcps = MultiLabelCRC(confidence=0.95, simultaneous=True, family_size=9)
rcps.calibrate(
    probs_cal,
    y_cal,
    class_alphas,
    controlled_classes=controlled,
)
prediction_sets = rcps.predict(probs_test)
metrics = rcps.compute_metrics(prediction_sets, y_test, class_names)
```

The optional `guard_lambdas=` argument accepts calibration-derived legacy thresholds and produces a more inclusive operating point without consulting test labels.

## Full reproduction

Open `multilabel_crc_ecg.ipynb` in Colab. The notebook trains the models, reproduces the submitted-method audit, runs the corrected exact-binomial analysis, regenerates results JSON files and plots, and writes outputs to `CRC_Revision/` on Google Drive.

The historical key `marginal_coverage` is retained in JSON for compatibility, but it is the unweighted mean of class TPRs and should be read as **macro class coverage**, not conventional sample-level marginal conformal coverage.

## Transparency

See [REPRODUCIBILITY_AUDIT.md](REPRODUCIBILITY_AUDIT.md) for the exact corrections, unchanged components, and limitations.

## License and data

Code is MIT licensed. Datasets are not redistributed. PTB-XL and Chapman-Shaoxing retain their original licenses and must be downloaded from their official sources.

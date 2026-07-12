# Reproducibility audit and corrected analysis

## Why the repository changed

A final post-acceptance audit identified that the submitted Hoeffding implementation used

```python
alpha_effective = max(0.005, alpha - epsilon)
```

When `alpha - epsilon < 0.005`, the floor creates a conservative empirical threshold, but the stated Hoeffding argument no longer proves `FNR <= alpha`. The earlier repository also required corrections to PTB-XL patient grouping, normalization scope, cross-site preprocessing, coverage terminology and co-occurrence integer accumulation.

## Corrected primary calibration

The corrected method uses one-sided exact binomial upper confidence bounds for the binary calibration false-negative count. A threshold is accepted only when its upper bound is below the class target. The nine FNR-controlled classes use Bonferroni failure probability `0.05/9` for simultaneous 95% family-wise control.

The primary operating point uses a conservative safeguard: `lambda_final = max(lambda_exact, lambda_legacy)`. Because larger lambda values include more labels, they cannot increase FNR. All guard values are obtained from calibration data; test labels are not used for threshold choice.

## Corrected headline

- Proposed guarded exact-binomial RCPS: 9/9 empirical targets, 9/9 non-vacuous simultaneous guarantees.
- Uniform Standard CP: 5/9 targets.
- PTB-XL: 4/4 proposed versus 1/4 uniform Standard CP.
- Chapman: 5/5 proposed versus 4/5 uniform Standard CP.

## Other corrections

- PTB-XL uses patient-disjoint partitions.
- Normalization parameters come from the fitting partition only.
- PTB-XL fitting statistics are used for PTB-XL→Chapman shift preprocessing.
- Co-occurrence matrix multiplication casts labels to `int64` first.
- `marginal_coverage` is labelled as macro class coverage.
- The shift experiment is interpreted as combined covariate and label-definition shift.
- The old Hoeffding method remains available only as `LegacyHoeffdingCRC` for audit and ablation.

## Limitations retained in the corrected paper

The guarded method can produce large sets and high FPR, especially on Chapman. The statistical guarantee is in-distribution and exchangeability-dependent; it does not establish cross-hospital robustness or clinical deployment readiness. The 23-subclass experiment is retained as a stress-test audit and should not be conflated with the nine primary exact-binomial superclass guarantees.

# E0 baseline, isolated re-run: 5-fold CV summary

| | this run | original E0_ep120 |
|---|---|---|
| Pooled AUC | 0.819 | 0.823 |
| Fold AUC mean ± SD | 0.841 ± 0.056 | 0.825 ± 0.036 |
| Malignant caught | **34/50** (68%, 95% CI 54%–79%) | 35/50 |
| False alarms | 156/1293 benign (specificity 0.879) | 286 (0.779) |
| PPV (alarm is really malignant) | 17.9% | 10.9% |

## Per fold

| fold | AUC | malignant caught | false alarms | threshold | kept checkpoint |
|---|---|---|---|---|---|
| 0 | 0.867 | 8/10 | 28/256 | 0.0066 | stage B epoch 23 (overall 53) |
| 1 | 0.765 | 5/9 | 24/256 | 0.0364 | stage B epoch 64 (overall 94) |
| 2 | 0.823 | 9/12 | 63/258 | 0.3959 | stage B epoch 4 (overall 34) |
| 3 | 0.836 | 5/9 | 19/258 | 0.0171 | stage B epoch 26 (overall 56) |
| 4 | 0.916 | 7/10 | 22/265 | 0.0842 | stage B epoch 14 (overall 44) |

GPU training is not bit-for-bit deterministic, so expect small differences from the original run.
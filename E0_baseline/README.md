# E0 baseline: isolated

A standalone copy of experiment **E0_baseline_ep120** from `ovarian-twin`. That's the single model behind the "35/50 malignant
caught" result. Nothing here imports from `ovarian-twin/src`, and nothing here writes into `ovarian-twin`.

## What E0 is
- **Model:** ImageNet MobileNetV3-Large, then global average pooling, dropout 0.3, and a single sigmoid output for "malignant".
- **Images:** the cleaned grayscale set (`data/processed/images/`, 224×224 letterboxed PNGs).
- **Augmentation:** applied on the fly to training batches only: horizontal flip, rotation ±15°, brightness and contrast ±10%.
- **Class imbalance:** inverse-frequency class weights (malignant ≈ 13×).
- **Schedule:** stage A trains the head only for 30 epochs (lr 1e-3) and keeps the best-val-loss weights. Stage B unfreezes the top 50%
  of the backbone (BatchNorm stays frozen) and trains 90 epochs at lr 1e-4. There's no early stopping. The kept model is the
  epoch with the best validation AUC, across both stages.
- **Threshold:** Youden J on the validation fold. On ties it takes the lowest threshold, which favours sensitivity.

## How the 35/50 was measured
This is 5-fold cross-validation on the grouped folds in `data/cv_folds.csv` (a copy of `ovarian-twin/data/splits/cv_folds.csv`).
For fold k, **test = fold k**, **validation = fold k+1** (picks the checkpoint and threshold), and **train = the other three folds**.
Across the 5 folds, every one of the 50 malignant images is tested exactly once, by a model that never saw it. That gives 35/50.

## Run it (local GPU, WSL)
From Windows PowerShell:
```
wsl bash /mnt/d/Sridatta/Embedded_Systems_project/E0_baseline/run_e0.sh
```
This runs folds 0 to 4, then the pooled summary, then the deployable final model. It refuses to run without a GPU.
A finished fold is skipped if you re-run, so add `--force` to retrain. Individual steps, inside WSL with `~/ovenv` active:
```
python -u train_e0.py --mode cv --fold 0      # one fold
python -u summarize_cv.py                     # after all 5 folds
python -u train_e0.py --mode final            # train = folds 0-3, val = fold 4
```
Images are read from `~/ovarian-twin/data/processed/images/`. If yours are somewhere else, pass `--data-root <folder containing data/>`.

## What you get
```
outputs/
  run.log                          everything printed to the terminal
  cv/fold<k>/
    epoch_log.csv                  one row per epoch: train / val / test loss, accuracy, AUC, PR-AUC, recall, precision, lr, seconds
    curves/<metric>.png            one image per metric (6), train + validation + test lines, redrawn every epoch
    best.keras  final.keras        kept checkpoint (best val AUC) / weights at the last epoch
    metrics.json  threshold.json   result on the test fold at the validation threshold
    val_predictions.csv  test_predictions.csv
  cv/summary.md  summary.json      pooled 5-fold result, side by side with the original run
  cv/roc.png
  cv/curves_all_folds/<split>_<metric>.png   all 5 folds overlaid, one image per split and metric (18)
  final/                           same layout for the deployable model (no test curves: fold 4 is its validation set)
```
The **test curves are for watching only**. Checkpoint, restore and threshold all use validation, so the test numbers stay honest.

Each epoch took about 2 s on the GTX 1650 last time, so a fold takes roughly 5–8 min including the per-epoch test evaluation and plots.
GPU training isn't bit-for-bit reproducible, so expect a result close to the original 35/50, not necessarily identical.

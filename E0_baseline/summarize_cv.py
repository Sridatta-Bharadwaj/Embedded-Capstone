"""Pool the 5 cross-validation folds of the isolated E0 run into one result (the original "35/50" number).

Each fold's test predictions use that fold's own threshold (Youden J on its validation fold), so pooling them counts every
one of the 50 malignant images exactly once. Writes outputs/cv/summary.md, summary.json, roc.png and all-fold curve overlays.

  python summarize_cv.py              (after all 5 folds have finished)
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
CV = HERE / "outputs" / "cv"
ORIGINAL = {"pooled_auc": 0.823, "tp": 35, "malignant": 50, "specificity": 0.779, "fp": 286}  # E0_baseline_ep120, run 3
METRICS = ["loss", "accuracy", "auc", "pr_auc", "recall", "precision"]


def wilson(k, n, z=1.96):
    p, d = k / n, 1 + z * z / n
    c, h = (p + z * z / (2 * n)) / d, z * np.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def main():
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from sklearn.metrics import average_precision_score, roc_auc_score, roc_curve

    missing = [k for k in range(5) if not (CV / f"fold{k}" / "test_predictions.csv").exists()]
    if missing:
        sys.exit(f"Folds not finished yet: {missing}. Run train_e0.py --mode cv --fold <k> for each.")
    t = pd.concat([pd.read_csv(CV / f"fold{k}" / "test_predictions.csv") for k in range(5)], ignore_index=True)
    y, p, pred = t["label"].to_numpy(), t["prob"].to_numpy(), t["pred"].to_numpy()
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    rows = []
    for k, g in t.groupby("fold"):
        gy, gp = g["label"].to_numpy(), g["pred"].to_numpy()
        m = json.loads((CV / f"fold{k}" / "metrics.json").read_text())
        rows.append({"fold": k, "AUC": round(roc_auc_score(gy, g["prob"]), 3),
                     "malignant caught": f"{int(((gp == 1) & (gy == 1)).sum())}/{int(gy.sum())}",
                     "false alarms": f"{int(((gp == 1) & (gy == 0)).sum())}/{int((gy == 0).sum())}",
                     "threshold": round(float(g["threshold"].iloc[0]), 4), "kept checkpoint": m["kept_checkpoint"]})
    folds = pd.DataFrame(rows)
    s_ci, c_ci = wilson(tp, tp + fn), wilson(tn, tn + fp)
    res = {"pooled_auc": float(roc_auc_score(y, p)), "fold_auc_mean": float(folds["AUC"].mean()),
           "fold_auc_sd": float(folds["AUC"].std()), "pooled_pr_auc": float(average_precision_score(y, p)),
           "tp": tp, "malignant": tp + fn, "sensitivity": tp / (tp + fn), "sensitivity_ci": s_ci,
           "fp": fp, "benign": tn + fp, "specificity": tn / (tn + fp), "specificity_ci": c_ci,
           "ppv": tp / (tp + fp) if tp + fp else None, "original_run3": ORIGINAL}
    (CV / "summary.json").write_text(json.dumps(res, indent=2))

    md = [f"# E0 baseline, isolated re-run: 5-fold CV summary\n",
          f"| | this run | original E0_ep120 |", "|---|---|---|",
          f"| Pooled AUC | {res['pooled_auc']:.3f} | {ORIGINAL['pooled_auc']:.3f} |",
          f"| Fold AUC mean ± SD | {res['fold_auc_mean']:.3f} ± {res['fold_auc_sd']:.3f} | 0.825 ± 0.036 |",
          f"| Malignant caught | **{tp}/{tp + fn}** ({res['sensitivity']:.0%}, 95% CI {s_ci[0]:.0%}–{s_ci[1]:.0%}) | 35/50 |",
          f"| False alarms | {fp}/{tn + fp} benign (specificity {res['specificity']:.3f}) | 286 (0.779) |",
          f"| PPV (alarm is really malignant) | {res['ppv']:.1%} | 10.9% |", "",
          "## Per fold", "", "| " + " | ".join(folds.columns) + " |", "|" + "---|" * len(folds.columns),
          *["| " + " | ".join(str(v) for v in r) + " |" for r in folds.itertuples(index=False)], "",
          "GPU training is not bit-for-bit deterministic, so expect small differences from the original run."]
    (CV / "summary.md").write_text("\n".join(md))
    print("\n".join(md))

    fpr, tpr, _ = roc_curve(y, p)
    fig, ax = plt.subplots(figsize=(5.5, 5))
    ax.plot(fpr, tpr, color="#2a6fdb", lw=2, label=f"pooled out-of-fold, AUC {res['pooled_auc']:.3f}")
    ax.plot([0, 1], [0, 1], color="gray", ls=":")
    ax.plot(1 - res["specificity"], res["sensitivity"], "o", color="#c0392b",
            label=f"per-fold Youden thresholds: {tp}/{tp + fn} caught")
    ax.set_xlabel("false positive rate"); ax.set_ylabel("sensitivity"); ax.set_title("E0 baseline, 5-fold CV ROC")
    ax.legend(fontsize=8, loc="lower right"); ax.grid(alpha=0.3); fig.tight_layout()
    fig.savefig(CV / "roc.png", dpi=120); plt.close(fig)

    logs = {k: pd.read_csv(CV / f"fold{k}" / "epoch_log.csv") for k in range(5)}
    out = CV / "curves_all_folds"
    out.mkdir(exist_ok=True)
    for split in ("train", "val", "test"):
        for m in METRICS:
            col = m if split == "train" else f"{split}_{m}"
            fig, ax = plt.subplots(figsize=(7, 4.2))
            for k, h in logs.items():
                if col in h:
                    ax.plot(h["global_epoch"], h[col], lw=1.3, label=f"fold {k}")
            b = logs[0].loc[logs[0]["stage"] == "B", "global_epoch"].min()
            ax.axvline(b - 0.5, color="gray", ls="--", lw=1, label="start of fine-tuning")
            ax.set_title(f"{split} {m}, all 5 folds", fontsize=10); ax.set_xlabel("epoch")
            ax.grid(alpha=0.3); ax.legend(fontsize=8); fig.tight_layout()
            fig.savefig(out / f"{split}_{m}.png", dpi=110); plt.close(fig)
    print(f"\nSaved {CV / 'summary.md'}, {CV / 'roc.png'} and {len(list(out.glob('*.png')))} curves in {out}")


if __name__ == "__main__":
    main()

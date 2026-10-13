"""Generate every figure used in the E0 baseline project report into E0_report/figures/."""
import json
import shutil
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch
from PIL import Image
from sklearn.metrics import roc_auc_score, roc_curve

ROOT = Path(__file__).resolve().parent.parent
E0 = ROOT / "E0_baseline"
CV = E0 / "outputs" / "cv"
FIG = Path(__file__).resolve().parent / "figures"
FIG.mkdir(exist_ok=True)
plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10, "axes.spines.top": False,
                     "axes.spines.right": False})
C = {"train": "#2a6fdb", "val": "#e07b00", "test": "#2e9e5b", "red": "#c0392b", "gray": "#6b7280"}
FOLD_COLORS = ["#2a6fdb", "#e07b00", "#2e9e5b", "#8e44ad", "#c0392b"]


def save(fig, name):
    fig.savefig(FIG / name, dpi=200, bbox_inches="tight", facecolor="white")
    plt.close(fig)


# ------------------------------------------------------------------ diagrams
def box(ax, x, y, w, h, text, fc="#eef3fb", ec="#2a6fdb", fs=9, bold=False):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc=fc, ec=ec, lw=1.4))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
            fontweight="bold" if bold else "normal")


def arrow(ax, x1, y1, x2, y2):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle="-|>", mutation_scale=14, lw=1.4, color="#374151", zorder=5))


def architecture():
    fig, ax = plt.subplots(figsize=(12, 6.2))
    ax.set_xlim(0, 12); ax.set_ylim(0, 6.2); ax.axis("off")
    layers = [
        ("DATA LAYER", "#f3f4f6", ["OTU_2D dataset\n1,469 B-mode images\n8 tumour categories", "Binary labelling\nhigh-grade serous = 1\nothers = 0"]),
        ("PREPROCESSING LAYER", "#eef3fb", ["Overlay removal\n(calipers, text,\ntables inpainted)", "Fan crop +\nletterbox 224x224", "pHash de-duplication\n+ similarity groups"]),
        ("SPLITTING LAYER", "#fff4e5", ["StratifiedGroupKFold\n5 folds", "Fold k = test\nfold k+1 = validation\n3 folds = train"]),
        ("LEARNING LAYER", "#eaf7ef", ["MobileNetV3-Large\n(ImageNet weights)", "Stage A: head only\n30 ep, lr 1e-3", "Stage B: top 50%\nfine-tuned, 90 ep,\nlr 1e-4"]),
        ("EVALUATION LAYER", "#fdecea", ["Best-val-AUC\ncheckpoint +\nYouden threshold", "Pooled out-of-fold\nAUC, sensitivity,\nspecificity, CIs", "Grad-CAM\nexplainability"]),
    ]
    xs = [0.15, 2.55, 4.95, 7.35, 9.75]
    for (title, fc, items), x in zip(layers, xs):
        ax.add_patch(FancyBboxPatch((x, 0.2), 2.1, 5.7, boxstyle="round,pad=0.02,rounding_size=0.12", fc=fc,
                                    ec="#9ca3af", lw=1.0))
        ax.text(x + 1.05, 5.6, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#111827")
        n = len(items)
        h = 1.2
        gap = (5.0 - n * h) / (n + 1)
        for i, t in enumerate(items):
            y = 5.1 - (i + 1) * gap - (i + 1) * h
            box(ax, x + 0.12, y, 1.86, h, t, fc="white", ec="#4b5563", fs=8.3)
            if i:
                arrow(ax, x + 1.05, y + h + gap, x + 1.05, y + h + 0.02)
        if x != xs[-1]:
            arrow(ax, x + 2.12, 3.05, x + 2.53, 3.05)
    save(fig, "fig_architecture.png")


def methodology():
    fig, ax = plt.subplots(figsize=(12, 3.0))
    ax.set_xlim(0, 12); ax.set_ylim(0, 3); ax.axis("off")
    steps = ["Data\nacquisition", "Cleaning &\nde-duplication", "Grouped\n5-fold split", "Class weights +\naugmentation",
             "Two-stage\ntransfer learning", "Checkpoint &\nthreshold (val)", "Out-of-fold\nevaluation", "Final model\n(folds 0-3)"]
    w, gap = 1.3, 0.18
    for i, s in enumerate(steps):
        x = 0.1 + i * (w + gap)
        fc = ["#f3f4f6", "#eef3fb", "#eef3fb", "#fff4e5", "#eaf7ef", "#eaf7ef", "#fdecea", "#fdecea"][i]
        box(ax, x, 0.9, w, 1.2, s, fc=fc, ec="#4b5563", fs=8.5)
        if i < len(steps) - 1:
            arrow(ax, x + w + 0.01, 1.5, x + w + gap - 0.01, 1.5)
    ax.text(6, 0.4, "Repeated for every fold k = 0..4; the test fold is never used for any training decision",
            ha="center", fontsize=8.5, style="italic", color=C["gray"])
    save(fig, "fig_methodology.png")


def model_diagram():
    fig, ax = plt.subplots(figsize=(12, 2.8))
    ax.set_xlim(0, 12); ax.set_ylim(0, 2.8); ax.axis("off")
    blocks = [("Input image\n224 x 224 x 3\n(grayscale x3)", "#f3f4f6"),
              ("Augmentation\n(train only)\nflip, rotation,\nbrightness,\ncontrast", "#fff4e5"),
              ("MobileNetV3-Large\nbackbone\n(ImageNet)\n7 x 7 x 960", "#eef3fb"),
              ("Global Average\nPooling\n960-d vector", "#eaf7ef"),
              ("Dropout\np = 0.3", "#eaf7ef"),
              ("Dense(1)\nsigmoid\nP(malignant)", "#fdecea"),
              ("Threshold t\n(Youden J on\nvalidation)", "#fdecea")]
    w, gap = 1.5, 0.2
    for i, (t, fc) in enumerate(blocks):
        x = 0.05 + i * (w + gap)
        box(ax, x, 0.55, w, 1.7, t, fc=fc, ec="#4b5563", fs=8.2)
        if i < len(blocks) - 1:
            arrow(ax, x + w + 0.01, 1.4, x + w + gap - 0.01, 1.4)
    ax.text(0.05 + 2 * (w + gap) + w / 2, 0.22, "frozen in stage A; top 50% unfrozen in stage B\n(BatchNorm always frozen)",
            ha="center", fontsize=7.8, style="italic", color=C["gray"])
    save(fig, "fig_model.png")


def fold_scheme():
    fig, ax = plt.subplots(figsize=(9, 3.6))
    ax.set_xlim(-1.6, 5.6); ax.set_ylim(-0.9, 5); ax.axis("off")
    col = {"train": "#cfe0f8", "val": "#fde3c2", "test": "#cdebd8"}
    for k in range(5):
        y = 4 - k
        ax.text(-1.5, y + 0.4, f"Run {k + 1}\n(test fold {k})", va="center", fontsize=8.5)
        for f in range(5):
            role = "test" if f == k else "val" if f == (k + 1) % 5 else "train"
            ax.add_patch(FancyBboxPatch((f, y + 0.08), 0.92, 0.7, boxstyle="round,pad=0.01,rounding_size=0.05",
                                        fc=col[role], ec="#6b7280", lw=0.8))
            ax.text(f + 0.46, y + 0.43, f"F{f}\n{role}", ha="center", va="center", fontsize=7.5)
    for i, (r, lab) in enumerate([("train", "training (3 folds)"), ("val", "validation (checkpoint, threshold)"),
                                  ("test", "test: reported result")]):
        ax.add_patch(FancyBboxPatch((i * 2.5 - 1.5, -0.75), 0.3, 0.3, boxstyle="square,pad=0", fc=col[r], ec="#6b7280"))
        ax.text(i * 2.5 - 1.1, -0.6, lab, va="center", fontsize=8)
    save(fig, "fig_folds.png")


# ------------------------------------------------------------------ data figures
def samples():
    order = ["normal_ovary", "simple_cyst", "chocolate_cyst", "teratoma", "serous_cystadenoma",
             "mucinous_cystadenoma", "theca_cell_tumor", "high_grade_serous"]
    fig, axes = plt.subplots(2, 4, figsize=(11, 6))
    for ax, c in zip(axes.ravel(), order):
        ax.imshow(Image.open(FIG / "samples" / f"{c}.png"), cmap="gray")
        ax.set_title(c.replace("_", " ").title() + ("\n(MALIGNANT)" if c == "high_grade_serous" else "\n(benign)"),
                     fontsize=9, color=C["red"] if c == "high_grade_serous" else "black")
        ax.axis("off")
    fig.tight_layout()
    save(fig, "fig_samples.png")


def class_distribution():
    rows = [("Chocolate cyst", 306), ("Teratoma", 304), ("Normal ovary", 243), ("Serous cystadenoma", 199),
            ("Mucinous cystadenoma", 96), ("Theca cell tumour", 82), ("Simple cyst", 63), ("High-grade serous", 50)]
    fig, ax = plt.subplots(figsize=(8, 3.8))
    names, vals = zip(*rows)
    colors = [C["train"]] * 7 + [C["red"]]
    b = ax.barh(names[::-1], vals[::-1], color=colors[::-1])
    for r, v in zip(b, vals[::-1]):
        ax.text(v + 4, r.get_y() + r.get_height() / 2, str(v), va="center", fontsize=8.5)
    ax.set_xlabel("images after cleaning (total 1,343)")
    ax.set_title("Class distribution: 1,293 benign (blue) vs 50 malignant (red), 3.7% positive", fontsize=10)
    save(fig, "fig_class_dist.png")


# ------------------------------------------------------------------ result figures
def logs():
    return {k: pd.read_csv(CV / f"fold{k}" / "epoch_log.csv") for k in range(5)}


def kept_epoch(metrics_path):
    m = json.loads(Path(metrics_path).read_text())
    return int(m["kept_checkpoint"].split("overall ")[1].rstrip(")"))


def curves_fold(L, k=0):
    h = L[k]
    kept = kept_epoch(CV / f"fold{k}" / "metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, met, title in [(axes[0], "loss", "Binary cross-entropy loss"), (axes[1], "auc", "ROC AUC")]:
        for split in ("train", "val", "test"):
            col = met if split == "train" else f"{split}_{met}"
            ax.plot(h["global_epoch"], h[col], color=C[split], lw=1.5,
                    label={"train": "training", "val": "validation", "test": "test (monitoring only)"}[split])
        ax.axvline(30.5, color=C["gray"], ls="--", lw=1, label="start of fine-tuning (stage B)")
        ax.axvline(kept, color=C["red"], ls=":", lw=1.4, label=f"kept checkpoint (epoch {kept})")
        ax.set_title(f"{title}, fold {k}", fontsize=10); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    axes[0].set_ylim(0, 1.5)
    fig.tight_layout()
    save(fig, f"fig_curves_fold{k}.png")


def val_auc_all(L):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for k, h in L.items():
        axes[0].plot(h["global_epoch"], h["auc"], color=FOLD_COLORS[k], lw=1.2, label=f"fold {k}")
        axes[1].plot(h["global_epoch"], h["val_auc"], color=FOLD_COLORS[k], lw=1.2, label=f"fold {k}")
        i = h["val_auc"].idxmax()
        axes[1].plot(h.loc[i, "global_epoch"], h.loc[i, "val_auc"], "*", color=FOLD_COLORS[k], ms=11)
    for ax, t in zip(axes, ["Training AUC, all folds", "Validation AUC, all folds (star = kept epoch)"]):
        ax.axvline(30.5, color=C["gray"], ls="--", lw=1)
        ax.set_title(t, fontsize=10); ax.set_xlabel("epoch"); ax.set_ylim(0.3, 1.02); ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5, loc="lower right")
    fig.tight_layout()
    save(fig, "fig_auc_allfolds.png")


def roc_and_cm():
    t = pd.concat([pd.read_csv(CV / f"fold{k}" / "test_predictions.csv") for k in range(5)], ignore_index=True)
    y, p, pred = t["label"].to_numpy(), t["prob"].to_numpy(), t["pred"].to_numpy()
    fig, ax = plt.subplots(figsize=(5.6, 5.2))
    for k, g in t.groupby("fold"):
        f, tp_, _ = roc_curve(g["label"], g["prob"])
        ax.plot(f, tp_, color=FOLD_COLORS[k], lw=0.9, alpha=0.55,
                label=f"fold {k} (AUC {roc_auc_score(g['label'], g['prob']):.3f})")
    f, tp_, _ = roc_curve(y, p)
    ax.plot(f, tp_, color="black", lw=2.2, label=f"pooled out-of-fold (AUC {roc_auc_score(y, p):.3f})")
    tp = int(((pred == 1) & (y == 1)).sum()); tn = int(((pred == 0) & (y == 0)).sum())
    ax.plot(1 - tn / (y == 0).sum(), tp / y.sum(), "o", color=C["red"], ms=8,
            label=f"operating point: {tp}/50 caught")
    ax.plot([0, 1], [0, 1], ":", color=C["gray"])
    ax.set_xlabel("false positive rate (1 - specificity)"); ax.set_ylabel("sensitivity")
    ax.set_title("ROC curves, 5-fold cross-validation", fontsize=10); ax.grid(alpha=0.3)
    ax.legend(fontsize=7.5, loc="lower right")
    save(fig, "fig_roc.png")

    fn = int(((pred == 0) & (y == 1)).sum()); fp = int(((pred == 1) & (y == 0)).sum())
    cm = np.array([[tn, fp], [fn, tp]])
    fig, ax = plt.subplots(figsize=(4.8, 4.2))
    ax.imshow(cm / cm.sum(axis=1, keepdims=True), cmap="Blues", vmin=0, vmax=1)
    for i in range(2):
        for j in range(2):
            frac = cm[i, j] / cm[i].sum()
            ax.text(j, i, f"{cm[i, j]}\n({frac:.1%})", ha="center", va="center", fontsize=12,
                    color="white" if frac > 0.5 else "black")
    ax.set_xticks([0, 1], ["predicted benign", "predicted malignant"])
    ax.set_yticks([0, 1], ["actual benign", "actual malignant"])
    ax.set_title("Pooled confusion matrix (1,343 images)", fontsize=10)
    for s in ax.spines.values():
        s.set_visible(False)
    save(fig, "fig_confusion.png")


def per_fold():
    ms = [json.loads((CV / f"fold{k}" / "metrics.json").read_text())["test"] for k in range(5)]
    x = np.arange(5); w = 0.26
    fig, ax = plt.subplots(figsize=(8.5, 3.8))
    for i, (key, lab, c) in enumerate([("roc_auc", "ROC AUC", "#374151"), ("sensitivity", "Sensitivity", C["red"]),
                                       ("specificity", "Specificity", C["train"])]):
        v = [m[key] for m in ms]
        b = ax.bar(x + (i - 1) * w, v, w, label=lab, color=c)
        for r, val in zip(b, v):
            ax.text(r.get_x() + r.get_width() / 2, val + 0.01, f"{val:.2f}", ha="center", fontsize=7.5)
    ax.set_xticks(x, [f"fold {k}\n({m['tp']}/{m['malignant']} caught)" for k, m in enumerate(ms)])
    ax.set_ylim(0, 1.1); ax.legend(fontsize=8, ncol=3, loc="upper center", bbox_to_anchor=(0.5, 1.12), frameon=False)
    ax.grid(axis="y", alpha=0.3)
    save(fig, "fig_perfold.png")


def final_curves():
    h = pd.read_csv(E0 / "outputs" / "final" / "epoch_log.csv")
    kept = kept_epoch(E0 / "outputs" / "final" / "metrics.json")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    for ax, met, title in [(axes[0], "loss", "Loss"), (axes[1], "auc", "ROC AUC")]:
        ax.plot(h["global_epoch"], h[met], color=C["train"], lw=1.5, label="training (folds 0-3)")
        ax.plot(h["global_epoch"], h[f"val_{met}"], color=C["val"], lw=1.5, label="validation (fold 4)")
        ax.axvline(30.5, color=C["gray"], ls="--", lw=1, label="start of fine-tuning")
        ax.axvline(kept, color=C["red"], ls=":", lw=1.4, label=f"kept checkpoint (epoch {kept})")
        ax.set_title(f"{title}, final model", fontsize=10); ax.set_xlabel("epoch"); ax.grid(alpha=0.3)
        ax.legend(fontsize=7.5)
    axes[0].set_ylim(0, 1.5)
    fig.tight_layout()
    save(fig, "fig_final_curves.png")


if __name__ == "__main__":
    architecture(); methodology(); model_diagram(); fold_scheme()
    samples(); class_distribution()
    L = logs()
    curves_fold(L, 0); val_auc_all(L); roc_and_cm(); per_fold(); final_curves()
    src = Path(__file__).resolve().parent / "source_figures"  # copied from the archived ovarian-twin run
    shutil.copy(src / "cleaning_before_after.png", FIG / "fig_cleaning.png")
    shutil.copy(src / "gradcam_E0_baseline_ep120.png", FIG / "fig_gradcam.png")
    print(sorted(p.name for p in FIG.glob("*.png")))

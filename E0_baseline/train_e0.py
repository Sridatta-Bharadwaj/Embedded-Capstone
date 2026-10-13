"""E0 baseline, isolated: MobileNetV3-Large, grayscale images, basic augmentation, class weights, 30 + 90 epochs.

This reproduces experiment E0_baseline_ep120 from ovarian-twin (src/model/crossval.py, run 3) as one standalone file.
Same folds (data/cv_folds.csv), same model, same augmentation, same schedule, same threshold rule.

Modes
  cv     fold k: test = fold k, val = fold (k+1) % 5, train = the other three folds.
         Running all 5 folds tests every one of the 50 malignant images exactly once (this is where "35/50" came from).
  final  deployable model: train = folds 0-3, val = fold 4 (picks the checkpoint and the threshold). No separate test set,
         same as the original final model.

Run from WSL (GPU only, it refuses to run on CPU):
  python -u train_e0.py --mode cv --fold 0          # one fold (~5 min on a GTX 1650)
  python -u train_e0.py --mode final
  python -u summarize_cv.py                         # pools the 5 folds into the 35/50-style result
Or just: bash run_e0.sh   (all 5 folds, then the summary, then the final model)

Every epoch: one summary line in the terminal, a row in epoch_log.csv, and every curve redrawn as its own PNG in curves/.
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
METRICS = ["loss", "accuracy", "auc", "pr_auc", "recall", "precision"]
TITLES = {"loss": "Loss (binary cross-entropy)", "accuracy": "Accuracy (threshold 0.5)", "auc": "ROC AUC",
          "pr_auc": "PR AUC", "recall": "Recall / sensitivity (threshold 0.5)", "precision": "Precision (threshold 0.5)"}


# ---------------------------------------------------------------- GPU guard
def require_gpu() -> str:
    import tensorflow as tf
    gpus = tf.config.list_physical_devices("GPU")
    if not gpus:
        sys.exit("[GPU GUARD] No GPU visible to TensorFlow. Refusing to train on CPU.\n"
                 "Run inside WSL with the ~/ovenv venv, and check `nvidia-smi` works there.")
    for g in gpus:
        tf.config.experimental.set_memory_growth(g, True)
    name = tf.config.experimental.get_device_details(gpus[0]).get("device_name", gpus[0].name)
    with tf.device("/GPU:0"):
        x = tf.random.uniform((256, 256)) @ tf.random.uniform((256, 256))
    print(f"[GPU GUARD] GPU: {name} | test matmul ran on: {x.device}", flush=True)
    if "GPU" not in x.device.upper():
        sys.exit("[GPU GUARD] Test op did not run on the GPU. Refusing to continue.")
    return name


# ---------------------------------------------------------------- data
def split_frames(folds: pd.DataFrame, data_root: Path, mode: str, k: int):
    df = folds.copy()
    df["abs_path"] = df["processed_path"].map(lambda p: str(data_root / p))
    if mode == "cv":
        test, val = df[df["fold"] == k], df[df["fold"] == (k + 1) % 5]
        train = df[~df["fold"].isin([k, (k + 1) % 5])]
    else:
        test, val, train = df.iloc[0:0], df[df["fold"] == 4], df[df["fold"] != 4]
    missing = [p for p in pd.concat([train, val, test])["abs_path"] if not Path(p).exists()]
    if missing:
        sys.exit(f"[DATA] {len(missing)} images missing, e.g. {missing[0]}\n"
                 f"Expected the processed 224x224 PNGs under {data_root}/data/processed/images/ (set --data-root).")
    return [d.reset_index(drop=True) for d in (train, val, test)]


def make_datasets(train, val, test, batch, seed):
    import tensorflow as tf
    auto = tf.data.AUTOTUNE

    def load(path, label):
        img = tf.io.decode_png(tf.io.read_file(path), channels=3)
        img = tf.ensure_shape(img, (224, 224, 3))
        return tf.cast(img, tf.float32), tf.cast(label, tf.float32)

    def base(d):
        return tf.data.Dataset.from_tensor_slices((d["abs_path"].to_numpy(), d["label"].to_numpy())) \
            .map(load, num_parallel_calls=auto).cache()

    # E0 augmentation (ovarian-twin build_augmenter): flip, rotation +/-15 deg, brightness/contrast +/-10%.
    aug = tf.keras.Sequential([
        tf.keras.layers.RandomFlip("horizontal", seed=seed),
        tf.keras.layers.RandomRotation(15.0 / 360.0, fill_mode="constant", fill_value=0.0, seed=seed),
        tf.keras.layers.RandomBrightness(0.10, value_range=(0.0, 255.0), seed=seed),
        tf.keras.layers.RandomContrast(0.10, seed=seed)], name="augment")
    tr = base(train).shuffle(len(train), seed=seed, reshuffle_each_iteration=True).repeat().batch(batch)
    tr = tr.map(lambda x, y: (tf.clip_by_value(aug(x, training=True), 0.0, 255.0), y), num_parallel_calls=auto)
    steps = int(np.ceil(len(train) / batch))
    ev = lambda d: base(d).batch(batch).prefetch(auto) if len(d) else None  # noqa: E731
    return tr.prefetch(auto), steps, ev(val), ev(test)


def class_weights(labels) -> dict:
    labels = np.asarray(labels).astype(int)
    n, counts = len(labels), np.bincount(labels, minlength=2)
    return {c: float(n / (2 * counts[c])) for c in (0, 1)}


# ---------------------------------------------------------------- model
def build_model(weights="imagenet"):
    import tensorflow as tf
    backbone = tf.keras.applications.MobileNetV3Large(weights=weights, include_top=False, input_shape=(224, 224, 3),
                                                      include_preprocessing=True)
    backbone.trainable = False
    inp = tf.keras.Input((224, 224, 3), name="image")
    x = backbone(inp, training=False)          # BatchNorm stays in inference mode, also after unfreezing
    x = tf.keras.layers.GlobalAveragePooling2D(name="gap")(x)
    x = tf.keras.layers.Dropout(0.3, name="dropout")(x)
    out = tf.keras.layers.Dense(1, activation="sigmoid", name="malignancy")(x)
    return tf.keras.Model(inp, out, name="ovarian_mnv3l")


def unfreeze_top(model, fraction):
    import tensorflow as tf
    backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    backbone.trainable = True
    cut = int(len(backbone.layers) * (1 - fraction))
    for i, layer in enumerate(backbone.layers):
        layer.trainable = i >= cut and not isinstance(layer, tf.keras.layers.BatchNormalization)
    return sum(l.trainable for l in backbone.layers)


def compile_model(model, lr):
    import tensorflow as tf
    model.compile(optimizer=tf.keras.optimizers.Adam(lr), loss="binary_crossentropy", metrics=[
        tf.keras.metrics.BinaryAccuracy(name="accuracy"), tf.keras.metrics.AUC(name="auc"),
        tf.keras.metrics.AUC(curve="PR", name="pr_auc"), tf.keras.metrics.Recall(name="recall"),
        tf.keras.metrics.Precision(name="precision")])


# ---------------------------------------------------------------- per-epoch logging + curves
def plot_curves(log_csv: Path, out_dir: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    h = pd.read_csv(log_csv)
    out_dir.mkdir(parents=True, exist_ok=True)
    b_start = h.loc[h["stage"] == "B", "global_epoch"].min() if (h["stage"] == "B").any() else None
    for m in METRICS:
        fig, ax = plt.subplots(figsize=(7, 4.2))
        for split, color in (("train", "#2a6fdb"), ("val", "#e07b00"), ("test", "#2e9e5b")):
            col = m if split == "train" else f"{split}_{m}"
            if col in h and h[col].notna().any():
                ax.plot(h["global_epoch"], h[col], label=split if split != "val" else "validation", color=color, lw=1.6)
        if m == "auc" and "val_auc" in h:
            i = h["val_auc"].idxmax()
            ax.plot(h.loc[i, "global_epoch"], h.loc[i, "val_auc"], "*", color="#c0392b", ms=13,
                    label=f"best val AUC (kept), epoch {int(h.loc[i, 'global_epoch'])}")
        if b_start is not None:
            ax.axvline(b_start - 0.5, color="gray", ls="--", lw=1, label="start of fine-tuning")
        ax.set_title(f"{TITLES[m]} | {title}", fontsize=10)
        ax.set_xlabel("epoch")
        ax.grid(alpha=0.3)
        ax.legend(fontsize=8)
        fig.tight_layout()
        fig.savefig(out_dir / f"{m}.png", dpi=110)
        plt.close(fig)


def make_epoch_logger(run_dir: Path, test_ds, title: str, total_epochs: int):
    import tensorflow as tf

    class EpochLogger(tf.keras.callbacks.Callback):
        """Test-set numbers are for watching only: checkpoint, early restore and threshold all use validation."""

        def __init__(self, stage, offset, stage_epochs):
            super().__init__()
            self.stage, self.offset, self.stage_epochs, self.t0 = stage, offset, stage_epochs, time.time()

        def on_epoch_begin(self, epoch, logs=None):
            self.t0 = time.time()

        def on_epoch_end(self, epoch, logs=None):
            logs = {k: float(v) for k, v in (logs or {}).items() if np.isscalar(v)}
            secs = time.time() - self.t0
            test = {}
            if test_ds is not None:
                test = {f"test_{k}": float(v) for k, v in
                        self.model.evaluate(test_ds, verbose=0, return_dict=True).items()}
            g = self.offset + epoch + 1
            lr = float(tf.keras.backend.get_value(self.model.optimizer.learning_rate))
            row = {"stage": self.stage, "global_epoch": g, "stage_epoch": epoch + 1, "seconds": round(secs, 1),
                   "lr": lr, **logs, **test}
            p = run_dir / "epoch_log.csv"
            prev = pd.read_csv(p) if p.exists() else pd.DataFrame()
            best_before = prev["val_auc"].max() if len(prev) else -1.0
            pd.concat([prev, pd.DataFrame([row])], ignore_index=True).to_csv(p, index=False)
            self.model.save(run_dir / "last.keras")
            try:
                plot_curves(p, run_dir / "curves", title)
            except Exception as e:  # a plotting hiccup must never stop training
                print(f"[WARN] plotting failed: {e}", flush=True)
            best = max(best_before, logs.get("val_auc", -1))
            f = lambda pre: (f"loss {row.get(pre + 'loss', np.nan):.3f} acc {row.get(pre + 'accuracy', np.nan):.3f} "  # noqa: E731
                             f"auc {row.get(pre + 'auc', np.nan):.3f} recall {row.get(pre + 'recall', np.nan):.2f}")
            print(f"[stage {self.stage}] epoch {epoch + 1:>3}/{self.stage_epochs} (overall {g:>3}/{total_epochs}) "
                  f"{secs:5.1f}s | TRAIN {f('')} | VAL {f('val_')}" + (f" | TEST {f('test_')}" if test else "") +
                  f" | best val_auc {best:.3f}{'  <- saved' if logs.get('val_auc', -1) > best_before else ''}",
                  flush=True)

    return EpochLogger


def make_restore_best(monitor, mode):
    import tensorflow as tf

    class RestoreBest(tf.keras.callbacks.Callback):
        """Run every epoch (no early stopping), then restore the best-`monitor` weights at the end (as in E0_ep120)."""

        def __init__(self):
            super().__init__()
            self.sign, self.best, self.weights = (1 if mode == "min" else -1), np.inf, None

        def on_epoch_end(self, epoch, logs=None):
            v = (logs or {}).get(monitor)
            if v is not None and self.sign * v < self.best:
                self.best, self.weights = self.sign * v, self.model.get_weights()

        def on_train_end(self, logs=None):
            if self.weights is not None:
                self.model.set_weights(self.weights)

    return RestoreBest()


# ---------------------------------------------------------------- threshold + metrics
def youden_threshold(y, p) -> float:
    from sklearn.metrics import roc_curve
    fpr, tpr, thr = roc_curve(y, p)
    j = tpr - fpr
    return float(np.clip(thr[np.flatnonzero(j == j.max())[-1]], 1e-6, 1.0))  # lowest among ties -> favour sensitivity


def summarize(y, p, thr) -> dict:
    from sklearn.metrics import average_precision_score, roc_auc_score
    y, pred = np.asarray(y).astype(int), (np.asarray(p) >= thr).astype(int)
    tp, fn = int(((pred == 1) & (y == 1)).sum()), int(((pred == 0) & (y == 1)).sum())
    tn, fp = int(((pred == 0) & (y == 0)).sum()), int(((pred == 1) & (y == 0)).sum())
    both = 0 < y.sum() < len(y)
    return {"threshold": thr, "n": len(y), "malignant": int(y.sum()), "tp": tp, "fn": fn, "tn": tn, "fp": fp,
            "sensitivity": tp / (tp + fn) if tp + fn else None, "specificity": tn / (tn + fp) if tn + fp else None,
            "roc_auc": float(roc_auc_score(y, p)) if both else None,
            "pr_auc": float(average_precision_score(y, p)) if both else None}


# ---------------------------------------------------------------- main
def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--mode", choices=["cv", "final"], required=True)
    ap.add_argument("--fold", type=int, default=0, help="cv mode: test fold 0-4")
    ap.add_argument("--data-root", default=os.path.expanduser("~/ovarian-twin"),
                    help="folder that contains data/processed/images/ (default ~/ovarian-twin)")
    ap.add_argument("--folds-csv", default=str(HERE / "data" / "cv_folds.csv"))
    ap.add_argument("--out", default=str(HERE / "outputs"))
    ap.add_argument("--head-epochs", type=int, default=30)
    ap.add_argument("--ft-epochs", type=int, default=90)
    ap.add_argument("--head-lr", type=float, default=1e-3)
    ap.add_argument("--ft-lr", type=float, default=1e-4)
    ap.add_argument("--unfreeze-fraction", type=float, default=0.5)
    ap.add_argument("--batch-size", type=int, default=32)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--force", action="store_true", help="retrain even if this run already finished")
    a = ap.parse_args(argv)

    gpu = require_gpu()
    import tensorflow as tf

    tag = f"fold{a.fold}" if a.mode == "cv" else "final"
    run_dir = Path(a.out) / ("cv" if a.mode == "cv" else "") / tag
    if (run_dir / "metrics.json").exists() and not a.force:
        print(f"[SKIP] {run_dir} already finished (use --force to retrain).")
        return
    run_dir.mkdir(parents=True, exist_ok=True)
    for f in ("epoch_log.csv", "best.keras", "last.keras"):
        (run_dir / f).unlink(missing_ok=True)

    seed = a.seed + (a.fold if a.mode == "cv" else 4)   # E0 used seed + fold index (final: val fold 4)
    tf.keras.utils.set_random_seed(seed)
    train, val, test = split_frames(pd.read_csv(a.folds_csv), Path(a.data_root), a.mode, a.fold)
    tr, steps, va, te = make_datasets(train, val, test, a.batch_size, seed)
    cw = class_weights(train["label"])
    title = f"E0 baseline, {tag}"
    print(f"[DATA] {tag} | train {len(train)} ({int(train.label.sum())} malignant) | val {len(val)} "
          f"({int(val.label.sum())}) | test {len(test)} ({int(test.label.sum())}) | class weights "
          f"{{0: {cw[0]:.3f}, 1: {cw[1]:.3f}}}", flush=True)

    total = a.head_epochs + a.ft_epochs
    Logger = make_epoch_logger(run_dir, te, title, total)
    best = run_dir / "best.keras"
    ckpt = lambda: tf.keras.callbacks.ModelCheckpoint(  # noqa: E731
        best, monitor="val_auc", mode="max", save_best_only=True,
        initial_value_threshold=(pd.read_csv(run_dir / "epoch_log.csv")["val_auc"].max()
                                 if (run_dir / "epoch_log.csv").exists() else None))
    t_start = time.time()

    # Stage A: frozen backbone, head only. Keeps the best-val-loss weights for stage B.
    model = build_model()
    compile_model(model, a.head_lr)
    print(f"\n===== Stage A: head only, {a.head_epochs} epochs, lr {a.head_lr} =====", flush=True)
    model.fit(tr, steps_per_epoch=steps, validation_data=va, epochs=a.head_epochs, class_weight=cw, verbose=0,
              callbacks=[make_restore_best("val_loss", "min"), ckpt(), Logger("A", 0, a.head_epochs)])

    # Stage B: unfreeze the top half of the backbone (BatchNorm stays frozen).
    n = unfreeze_top(model, a.unfreeze_fraction)
    compile_model(model, a.ft_lr)
    print(f"\n===== Stage B: fine-tuning, {n} backbone layers unfrozen, {a.ft_epochs} epochs, lr {a.ft_lr} =====",
          flush=True)
    model.fit(tr, steps_per_epoch=steps, validation_data=va, epochs=a.ft_epochs, class_weight=cw, verbose=0,
              callbacks=[make_restore_best("val_auc", "max"), ckpt(), Logger("B", a.head_epochs, a.ft_epochs)])
    model.save(run_dir / "final.keras")
    train_secs = time.time() - t_start

    # Evaluate the kept checkpoint (best val AUC over both stages). Threshold = Youden J on validation.
    model = tf.keras.models.load_model(best)
    log = pd.read_csv(run_dir / "epoch_log.csv")
    kept = log.loc[log["val_auc"].idxmax()]
    val_p = model.predict(va, verbose=0).ravel()
    thr = youden_threshold(val["label"], val_p)
    out = {"experiment": "E0_baseline_ep120 (isolated)", "mode": a.mode, "fold": a.fold if a.mode == "cv" else None,
           "gpu": gpu, "train_seconds": round(train_secs), "args": vars(a),
           "kept_checkpoint": f"stage {kept['stage']} epoch {int(kept['stage_epoch'])} (overall {int(kept['global_epoch'])})",
           "threshold_rule": "Youden J on the validation fold (lowest threshold among ties)",
           "note": "test curves are monitoring only; checkpoint and threshold use validation only",
           "val": summarize(val["label"], val_p, thr)}
    val.assign(prob=val_p, threshold=thr)[["filename", "class_name", "label", "fold", "prob", "threshold"]] \
        .to_csv(run_dir / "val_predictions.csv", index=False)
    (run_dir / "threshold.json").write_text(json.dumps({"threshold": thr, "chosen_on": f"validation ({tag})"}))
    if len(test):
        p = model.predict(te, verbose=0).ravel()
        out["test"] = summarize(test["label"], p, thr)
        test.assign(prob=p, threshold=thr, pred=(p >= thr).astype(int))[
            ["filename", "class_name", "label", "fold", "prob", "threshold", "pred"]] \
            .to_csv(run_dir / "test_predictions.csv", index=False)
    (run_dir / "last.keras").unlink(missing_ok=True)
    (run_dir / "metrics.json").write_text(json.dumps(out, indent=2, default=float))

    r = out.get("test", out["val"])
    which = "TEST fold" if "test" in out else "VALIDATION fold (also chose checkpoint + threshold, so optimistic)"
    print(f"\n===== {tag} done in {train_secs / 60:.1f} min | kept {out['kept_checkpoint']} | threshold {thr:.4f} =====")
    print(f"{which}: AUC {r['roc_auc']:.3f} | malignant caught {r['tp']}/{r['malignant']} | "
          f"false alarms {r['fp']}/{r['fp'] + r['tn']} benign (specificity {r['specificity']:.3f})")
    for f in sorted(run_dir.rglob("*")):
        if f.is_file():
            print(f"  saved {f} ({f.stat().st_size / 1e6:.2f} MB)")


if __name__ == "__main__":
    main()

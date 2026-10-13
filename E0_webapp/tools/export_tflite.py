"""Export the finalized E0 model to TFLite for the Raspberry Pi web app (run once, on the PC, in WSL with TensorFlow).

The exported graph takes one 224x224x3 float image (raw 0-255 pixels, same as training) and returns two outputs:
  prob  (1, 1)     malignancy probability (sigmoid), identical to the Keras model
  cam   (1, 7, 7)  Grad-CAM map for the malignancy logit, before the ReLU (applied on the Pi)

Why the CAM can live inside the graph: E0's head is GAP -> dropout (off at inference) -> Dense(1). The gradient of the
logit with respect to the last feature map F (7x7x960) is k_c / 49 at every position, so the Grad-CAM channel weights are
k_c / 49 and Grad-CAM = ReLU(sum_c k_c F_c) / 49. That is one dot product, so the Pi needs no gradients and no TensorFlow.
This script checks that claim against tf.GradientTape on real validation images.

    wsl ~/ovenv/bin/python /mnt/d/.../E0_webapp/tools/export_tflite.py
Writes E0_webapp/model/e0_fp32.tflite, e0_dynamic.tflite, model_info.json and export_check.json.
"""
import argparse
import json
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
APP = HERE.parent
PROJECT = APP.parent


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--keras", default=str(PROJECT / "E0_baseline/outputs/final/best.keras"))
    ap.add_argument("--threshold", default=str(PROJECT / "E0_baseline/outputs/final/threshold.json"))
    ap.add_argument("--folds", default=str(PROJECT / "E0_baseline/data/cv_folds.csv"))
    ap.add_argument("--data-root", default=str(Path.home() / "ovarian-twin"))
    ap.add_argument("--out", default=str(APP / "model"))
    args = ap.parse_args()

    import pandas as pd
    import tensorflow as tf
    from sklearn.metrics import roc_auc_score

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    model = tf.keras.models.load_model(args.keras)
    backbone = next(l for l in model.layers if isinstance(l, tf.keras.Model))
    dense = model.get_layer("malignancy")
    kernel = tf.constant(dense.kernel.numpy()[:, 0])
    bias = tf.constant(dense.bias.numpy()[0])

    @tf.function(input_signature=[tf.TensorSpec([1, 224, 224, 3], tf.float32, name="image")])
    def serve(x):
        feats = backbone(x, training=False)                       # (1, 7, 7, 960)
        z = tf.reduce_mean(feats, axis=(1, 2)) @ kernel[:, None] + bias
        cam = tf.tensordot(feats, kernel, axes=[[3], [0]]) / 49.0  # (1, 7, 7), ReLU applied on the Pi
        return {"prob": tf.sigmoid(z), "cam": cam}

    conc = serve.get_concrete_function()
    paths = {}
    for name, quant in (("fp32", False), ("dynamic", True)):
        conv = tf.lite.TFLiteConverter.from_concrete_functions([conc], serve)
        if quant:
            conv.optimizations = [tf.lite.Optimize.DEFAULT]
        p = out / f"e0_{name}.tflite"
        p.write_bytes(conv.convert())
        paths[name] = p
        print(f"[EXPORT] {p.name}: {p.stat().st_size / 2**20:.1f} MB", flush=True)

    # ---- checks on the final model's validation fold (fold 4: never trained on)
    folds = pd.read_csv(args.folds)
    val = folds[(folds["fold"] == 4) & (~folds["dropped"].astype(bool))].reset_index(drop=True)
    root = Path(args.data_root)
    xs = np.stack([tf.io.decode_png(tf.io.read_file(str(root / p)), channels=3).numpy() for p in val["processed_path"]])
    xs = xs.astype(np.float32)
    keras_p = model.predict(xs, batch_size=32, verbose=0)[:, 0]

    def gradcam_tape(x):
        with tf.GradientTape() as tape:
            f = backbone(x, training=False)
            tape.watch(f)
            zz = tf.reduce_mean(f, axis=(1, 2)) @ kernel[:, None] + bias
        g = tape.gradient(zz, f)
        w = tf.reduce_mean(g, axis=(1, 2), keepdims=True)
        return tf.nn.relu(tf.reduce_sum(w * f, axis=-1)).numpy()

    thr = json.loads(Path(args.threshold).read_text())["threshold"]
    labels = val["label"].to_numpy()
    report = {"n_val": len(val), "malignant": int(labels.sum()), "threshold": thr,
              "keras_auc": float(roc_auc_score(labels, keras_p))}
    for name, p in paths.items():
        it = tf.lite.Interpreter(model_path=str(p), num_threads=4)
        it.allocate_tensors()
        inp = it.get_input_details()[0]
        outs = {tuple(d["shape"]): d["index"] for d in it.get_output_details()}
        probs, cam_err = [], []
        for i, x in enumerate(xs):
            it.set_tensor(inp["index"], x[None])
            it.invoke()
            probs.append(float(it.get_tensor(outs[(1, 1)])[0, 0]))
            if i < 40:  # Grad-CAM equivalence on a subset (tape is slow-ish)
                cam = np.maximum(it.get_tensor(outs[(1, 7, 7)])[0], 0)
                ref = gradcam_tape(x[None])[0]
                n = lambda m: m / (m.max() + 1e-8)  # noqa: E731
                cam_err.append(float(np.abs(n(cam) - n(ref)).max()))
        probs = np.array(probs)
        report[name] = {
            "size_MB": round(p.stat().st_size / 2**20, 2),
            "auc": float(roc_auc_score(labels, probs)),
            "max_abs_prob_diff_vs_keras": float(np.abs(probs - keras_p).max()),
            "decisions_changed_vs_keras": int(((probs >= thr) != (keras_p >= thr)).sum()),
            "tp": int(((probs >= thr) & (labels == 1)).sum()),
            "fp": int(((probs >= thr) & (labels == 0)).sum()),
            "gradcam_max_abs_err_normalised": max(cam_err),
        }
        print(f"[CHECK] {name}: {json.dumps(report[name])}", flush=True)

    info = {"model": "E0_baseline_ep120 (final, best.keras)", "backbone": "MobileNetV3-Large", "input": [224, 224, 3],
            "input_range": "raw 0-255 grayscale replicated to 3 channels", "threshold": thr,
            "threshold_rule": "Youden J on validation fold 4", "positive_class": "high-grade serous carcinoma",
            "outputs": {"prob": [1, 1], "cam": [1, 7, 7]}, "keep_color": False, "blackout_tables": False}
    (out / "model_info.json").write_text(json.dumps(info, indent=2))
    (out / "export_check.json").write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()

"""E0 inference on the Pi: clean -> letterbox -> TFLite -> probability + Grad-CAM outline on the original frame.

numpy / opencv / a TFLite interpreter only (tflite_runtime, ai_edge_litert, or TensorFlow as a last resort).
The TFLite file comes from tools/export_tflite.py and has two outputs: prob (1, 1) and cam (1, 7, 7).
"""
import json
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path

import cv2
import numpy as np

from .clean import clean_image

SIZE = 224
OUTLINE_LEVEL = 0.5        # outline = CAM >= 50% of its maximum
MIN_REGION_FRACTION = 0.01  # ignore outline blobs smaller than 1% of the scan fan


def load_interpreter_module():
    try:
        import tflite_runtime.interpreter as mod
    except ImportError:
        try:
            import ai_edge_litert.interpreter as mod
        except ImportError:
            import tensorflow as tf
            mod = tf.lite
    return mod


def letterbox(img: np.ndarray, size: int = SIZE):
    """Same as training: pad to a centred square on black, resize to size x size, grey -> 3 channels.
    Returns the image plus (side, x0, y0) so the CAM can be mapped back."""
    h, w = img.shape[:2]
    side = max(h, w)
    canvas = np.zeros((side, side) + img.shape[2:], np.uint8)
    y0, x0 = (side - h) // 2, (side - w) // 2
    canvas[y0:y0 + h, x0:x0 + w] = img
    out = cv2.resize(canvas, (size, size), interpolation=cv2.INTER_AREA)
    if out.ndim == 2:
        out = np.repeat(out[:, :, None], 3, axis=2)
    return out, (side, x0, y0)


def centred_score(p: float, threshold: float) -> float:
    """Shift the logit so the decision threshold sits at 0.5. E0's raw probabilities are not calibrated (its threshold
    is about 0.11), so this is the scale shown to users and fed to the digital-twin risk model."""
    eps = 1e-6
    p, t = min(max(p, eps), 1 - eps), min(max(threshold, eps), 1 - eps)
    z = np.log(p / (1 - p)) - np.log(t / (1 - t))
    return float(1 / (1 + np.exp(-z)))


@dataclass
class Prediction:
    prob: float                  # raw model probability of high-grade serous carcinoma
    threshold: float
    score: float                 # threshold-centred malignancy score in [0, 1], 0.5 = decision boundary
    prediction: str              # 'malignant' | 'benign'
    confidence: float            # confidence in the stated prediction, in [0.5, 1]
    cam: np.ndarray              # float32 attention map in [0, 1], same H x W as the original frame
    fan_mask: np.ndarray         # bool, where the scan fan is in the original frame
    cam_peak: float              # un-normalised CAM maximum (0 = nothing pushed towards malignant)
    timings_ms: dict = field(default_factory=dict)

    @property
    def label(self) -> str:
        return "Malignant: high-grade serous carcinoma" if self.prediction == "malignant" else "Benign lesion"


class E0Model:
    def __init__(self, model_dir, threads: int = 4, model_file: str = "e0_fp32.tflite"):
        model_dir = Path(model_dir)
        self.info = json.loads((model_dir / "model_info.json").read_text())
        self.threshold = float(self.info["threshold"])
        self.path = model_dir / model_file
        self.version = self.path.stem
        mod = load_interpreter_module()
        self.backend = mod.__name__
        self.interp = mod.Interpreter(model_path=str(self.path), num_threads=threads)
        self.interp.allocate_tensors()
        self.inp = self.interp.get_input_details()[0]
        outs = {tuple(int(s) for s in d["shape"]): d["index"] for d in self.interp.get_output_details()}
        self.out_prob, self.out_cam = outs[(1, 1)], outs[(1, 7, 7)]
        self.lock = threading.Lock()  # one interpreter, many web threads

    def predict(self, bgr: np.ndarray) -> Prediction:
        t0 = time.perf_counter()
        cleaned, info = clean_image(bgr, keep_color=False, blackout_tables=False)
        x, (side, x0, y0) = letterbox(cleaned)
        t1 = time.perf_counter()
        with self.lock:
            self.interp.set_tensor(self.inp["index"], x[None].astype(np.float32))
            self.interp.invoke()
            prob = float(self.interp.get_tensor(self.out_prob)[0, 0])
            cam7 = np.maximum(self.interp.get_tensor(self.out_cam)[0], 0).astype(np.float32)
        t2 = time.perf_counter()

        cam, fan = self._cam_to_frame(cam7, bgr.shape[:2], info, side, x0, y0, cleaned)
        score = centred_score(prob, self.threshold)
        pred = "malignant" if prob >= self.threshold else "benign"
        t3 = time.perf_counter()
        return Prediction(prob=prob, threshold=self.threshold, score=score, prediction=pred,
                          confidence=score if pred == "malignant" else 1 - score, cam=cam, fan_mask=fan,
                          cam_peak=float(cam7.max()),
                          timings_ms={"preprocess": round(1000 * (t1 - t0), 1), "inference": round(1000 * (t2 - t1), 1),
                                      "attention_map": round(1000 * (t3 - t2), 1)})

    @staticmethod
    def _cam_to_frame(cam7, frame_hw, info, side, x0, y0, cleaned):
        """7x7 CAM -> 224x224 (bilinear, as in training-time Grad-CAM) -> letterbox square -> fan crop -> full frame."""
        H, W = frame_hw
        peak = cam7.max()
        cam224 = cv2.resize(cam7 / peak if peak > 0 else cam7, (SIZE, SIZE), interpolation=cv2.INTER_LINEAR)
        sq = cv2.resize(cam224, (side, side), interpolation=cv2.INTER_LINEAR)
        cx, cy, cw, ch = info["crop_x"], info["crop_y"], info["crop_w"], info["crop_h"]
        crop_cam = sq[y0:y0 + ch, x0:x0 + cw]
        fan_crop = (cleaned if cleaned.ndim == 2 else cleaned.max(axis=2)) > 0
        fan_crop = cv2.morphologyEx(fan_crop.astype(np.uint8), cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8)) > 0
        cam = np.zeros((H, W), np.float32)
        fan = np.zeros((H, W), bool)
        cam[cy:cy + ch, cx:cx + cw] = crop_cam * fan_crop
        fan[cy:cy + ch, cx:cx + cw] = fan_crop
        return cam, fan


# ---------------------------------------------------------------- rendering
def _thickness(shape) -> int:
    return max(2, round(max(shape[:2]) / 300))


def outline_contours(pred: Prediction):
    """Contours of the high-attention region (CAM >= OUTLINE_LEVEL), small specks dropped."""
    if pred.cam_peak <= 0:
        return []
    mask = (pred.cam >= OUTLINE_LEVEL).astype(np.uint8)
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    min_area = MIN_REGION_FRACTION * max(int(pred.fan_mask.sum()), 1)
    return [c for c in contours if cv2.contourArea(c) >= min_area]


def render_outline(bgr: np.ndarray, pred: Prediction) -> np.ndarray:
    out = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    colour = (60, 60, 235) if pred.prediction == "malignant" else (40, 190, 250)  # BGR: red / amber
    t = _thickness(out.shape)
    contours = outline_contours(pred)
    cv2.drawContours(out, contours, -1, (0, 0, 0), t + 2, cv2.LINE_AA)  # dark halo keeps it visible on bright tissue
    cv2.drawContours(out, contours, -1, colour, t, cv2.LINE_AA)
    if pred.cam_peak > 0:  # cross-hair on the single strongest point
        y, x = np.unravel_index(int(np.argmax(pred.cam)), pred.cam.shape)
        r = 4 * t
        cv2.drawMarker(out, (int(x), int(y)), (255, 255, 255), cv2.MARKER_CROSS, 2 * r, t, cv2.LINE_AA)
    return out


def render_heatmap(bgr: np.ndarray, pred: Prediction, alpha: float = 0.45) -> np.ndarray:
    base = bgr.copy() if bgr.ndim == 3 else cv2.cvtColor(bgr, cv2.COLOR_GRAY2BGR)
    heat = cv2.applyColorMap((pred.cam * 255).astype(np.uint8), cv2.COLORMAP_JET)
    blend = cv2.addWeighted(base, 1 - alpha, heat, alpha, 0)
    out = base.copy()
    out[pred.fan_mask] = blend[pred.fan_mask]
    return out


def decode_image(data: bytes) -> np.ndarray:
    arr = np.frombuffer(data, np.uint8)
    img = cv2.imdecode(arr, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("Not a readable image (use JPG, PNG or BMP).")
    if min(img.shape[:2]) < 64:
        raise ValueError("Image is too small to analyse.")
    return img


def save_jpeg(path: Path, img: np.ndarray, max_side: int = 1400) -> None:
    h, w = img.shape[:2]
    s = max_side / max(h, w)
    if s < 1:
        img = cv2.resize(img, (round(w * s), round(h * s)), interpolation=cv2.INTER_AREA)
    path.parent.mkdir(parents=True, exist_ok=True)
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
    if not ok:
        raise IOError(f"could not encode {path.name}")
    path.write_bytes(buf.tobytes())

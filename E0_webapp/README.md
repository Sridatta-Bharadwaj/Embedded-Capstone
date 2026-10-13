# E0 web app: ovarian ultrasound digital twin

A Flask web UI for the finalized **E0 baseline** model (MobileNetV3-Large, binary: high-grade serous carcinoma vs. benign),
built to run on a Raspberry Pi 4 with no TensorFlow and no internet.

For every uploaded ultrasound frame it shows:
- **Tumour type**: *Malignant (high-grade serous carcinoma)* or *Benign lesion*. E0 is binary, so it does not name benign subtypes.
- **Confidence**: distance from the decision threshold. Raw scores are shifted so the threshold (0.110) sits at 50%.
- **Outline**: a Grad-CAM attention region drawn on the original frame (red = malignant result, amber = benign), plus heatmap
  and original views. It shows *where the evidence came from*. It is not a segmentation of the tumour.
- **Digital twin**: each patient's scan history, a smoothed risk (EWMA), trend, and an alert when risk jumps by more than 15 points,
  crosses 50%, or the latest scan scores malignant. A time-scaled risk chart is included.

## Layout
```
app/__init__.py      routes (dashboard, scan, result, patient twin, model card, JSON API)
app/inference.py     clean -> letterbox -> TFLite -> probability + CAM -> outline/heatmap rendering
app/clean.py         overlay removal + fan crop (copy of ovarian-twin/src/data/clean.py)
app/twin.py          SQLite twin store, EWMA risk state, chart geometry
app/templates, app/static   server-rendered HTML/CSS, 6 held-out sample scans
model/e0_fp32.tflite the deployed model (11.3 MB) + model_info.json + export_check.json
tools/export_tflite.py      re-exports the model from E0_baseline/outputs/final/best.keras (needs TensorFlow)
tests/test_app.py    end-to-end tests on the real model
data/                created at run time: twin.db, scans/*.jpg (not committed)
```

## Run it (PC or Pi)
```
python -m venv .venv && . .venv/bin/activate
pip install -r requirements.txt
python run.py                 # http://<host>:8000
python -m pytest -q tests/    # 7 tests
```
Env vars: `E0_DATA_DIR` (default `./data`), `E0_MODEL_DIR` (default `./model`), `E0_THREADS` (default 4).

JSON API: `GET /api/health`, and `POST /api/predict` (multipart `image`, plus optional `patient_id` and `scan_date` to log the scan to a twin).

## Verified
- The TFLite fp32 model matches Keras on all 275 images of validation fold 4 (max difference 6e-6, 0 decisions changed).
  The dynamic-range int8 export was **rejected**: it changed 10 decisions and broke the attention map.
- The in-graph CAM equals TensorFlow Grad-CAM (max error 3e-5). For E0's GAP → Dense head, Grad-CAM reduces to
  ReLU(features · dense weights), so it costs no gradients on the Pi.
- The app's preprocessing of raw frames is pixel-identical to the training images (8/8 checked).
- On the PC: about 3 ms inference and 40–100 ms preprocessing per image. The Pi 4 will be slower; measure it after deployment.

## Notes
- No login: run it on a trusted network only. Use pseudonymous patient IDs.
- Research prototype, not a medical device.

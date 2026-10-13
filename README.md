# Embedded Capstone: ovarian tumour classification on the edge (E0 model)

Binary classification of 2D ovarian ultrasound (OTU_2D / MMOTU): high-grade serous carcinoma vs. benign, using a
fine-tuned MobileNetV3-Large (**E0 baseline**). The model is deployed as a TFLite web app with a patient digital twin on a Raspberry Pi 4.

| Folder | What it holds |
|---|---|
| [`E0_baseline/`](E0_baseline/README.md) | Training code, 5-fold grouped CV (pooled AUC 0.819, 34/50 cancers caught, specificity 0.879), final model |
| [`E0_report/`](E0_report/) | Project report (`E0_Project_Report.docx` / `.pdf`) and the scripts that build it |
| [`E0_webapp/`](E0_webapp/README.md) | Flask web UI: tumour type, confidence, Grad-CAM outline, per-patient digital twin. Runs on a Raspberry Pi 4 |
| [`DATA_REPORT.md`](DATA_REPORT.md) | Dataset description, cleaning, splits and limitations |

The dataset itself is not included. Download OTU_2D from the MMOTU authors (link in `DATA_REPORT.md`).

Research prototype. Not a medical device.

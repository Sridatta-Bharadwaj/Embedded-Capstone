"""End-to-end smoke tests on the real model: python -m pytest -q tests/"""
import io
import json
from pathlib import Path

import pytest

from app import create_app

SAMPLES = Path(__file__).resolve().parents[1] / "app" / "static" / "samples"


@pytest.fixture()
def client(tmp_path):
    app = create_app({"DATA_DIR": tmp_path, "TESTING": True})
    return app.test_client()


def sample(name):
    return (io.BytesIO((SAMPLES / name).read_bytes()), name)


def test_health(client):
    r = client.get("/api/health").get_json()
    assert r["ok"] and r["model"] == "e0_fp32"


def test_api_predict_samples(client):
    meta = json.loads((SAMPLES / "samples.json").read_text())
    for name, m in meta.items():
        r = client.post("/api/predict", data={"image": sample(name)}, content_type="multipart/form-data").get_json()
        assert r["prediction"] in ("benign", "malignant")
        assert 0.5 <= r["confidence"] <= 1
        assert (r["prediction"] == "malignant") == (r["raw_probability"] >= r["threshold"])
        print(f"{name:28s} truth={m['truth']:9s} pred={r['prediction']:9s} conf={r['confidence']:.3f} "
              f"p={r['raw_probability']:.4f} {r['timings_ms']}")


def test_rejects_non_image(client):
    r = client.post("/api/predict", data={"image": (io.BytesIO(b"not an image"), "x.jpg")},
                    content_type="multipart/form-data")
    assert r.status_code == 400


def test_full_ui_flow(client):
    assert client.get("/").status_code == 200
    r = client.post("/patients/new", data={"id": "PT-1", "name": "Demo", "birth_year": "1980"})
    assert r.status_code == 302
    for d, s in (("2026-01-10", "benign_simple_cyst.jpg"), ("2026-04-02", "malignant_hgsc_a.jpg")):
        r = client.post("/scan", data={"patient_id": "PT-1", "scan_date": d, "sample": s})
        assert r.status_code == 302 and "/visits/" in r.headers["Location"]
    page = client.get(r.headers["Location"])
    assert page.status_code == 200 and b"Digital twin update" in page.data
    img = client.get(page.data.decode().split('id="scanimg" src="')[1].split('"')[0])
    assert img.status_code == 200 and img.data[:2] == b"\xff\xd8"
    twin = client.get("/patients/PT-1")
    assert twin.status_code == 200 and b"<svg class=\"chart\"" in twin.data
    # upload path + new patient created on the fly
    r = client.post("/scan", data={"new_patient_id": "PT-2", "scan_date": "2026-05-01",
                                   "image": sample("normal_ovary.jpg")}, content_type="multipart/form-data")
    assert r.status_code == 302
    assert client.get("/model").status_code == 200
    assert client.get("/patients/nope").status_code == 404
    r = client.post("/patients/PT-1/delete")
    assert r.status_code == 302 and client.get("/patients/PT-1").status_code == 404


def test_bad_patient_id(client):
    r = client.post("/scan", data={"new_patient_id": "../etc", "sample": "normal_ovary.jpg"})
    assert r.status_code == 302 and r.headers["Location"].endswith("/scan")


def test_twin_state():
    from app.twin import twin_state
    assert twin_state([]).status == "none"
    s = twin_state([0.2, 0.3, 0.9])
    assert s.trend == "rising" and s.flag
    assert twin_state([0.8, 0.1]).trend == "falling"


def test_malignant_latest_scan_always_alerts():
    from app.twin import twin_state
    s = twin_state([0.02, 0.13, 0.62, 0.62])   # smoothed risk 0.48, rise < 15 points, never crossed 0.5
    assert s.flag and "latest scan scored malignant" in s.reason
    assert not twin_state([0.6, 0.2]).flag

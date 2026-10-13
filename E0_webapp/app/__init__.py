"""Flask web app: ovarian ultrasound digital twin running the finalized E0 model on a Raspberry Pi 4.

Pages: dashboard (all twins) - new scan - scan result (type, confidence, outlined image) - patient twin (timeline + risk
trend) - model card. JSON: GET /api/health, POST /api/predict (multipart 'image', optional 'patient_id').
Everything is served locally (no CDN), so it works on a Pi with no internet.
"""
import json
import os
import re
import secrets
import uuid
from datetime import date, datetime
from pathlib import Path

from flask import Flask, abort, flash, jsonify, redirect, render_template, request, send_from_directory, url_for

from . import inference as inf
from .twin import RISK_THRESHOLD, TwinStore, chart

APP_DIR = Path(__file__).resolve().parent
ROOT = APP_DIR.parent
PID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9_.-]{0,39}$")
ALLOWED_EXT = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}


def _secret_key(data_dir: Path) -> str:
    p = data_dir / "secret_key"
    if not p.exists():
        p.write_text(secrets.token_hex(32))
    return p.read_text().strip()


def create_app(test_config=None) -> Flask:
    app = Flask(__name__)
    data_dir = Path(os.environ.get("E0_DATA_DIR", ROOT / "data"))
    model_dir = Path(os.environ.get("E0_MODEL_DIR", ROOT / "model"))
    app.config.update(DATA_DIR=data_dir, MODEL_DIR=model_dir,
                      MAX_CONTENT_LENGTH=20 * 2**20, THREADS=int(os.environ.get("E0_THREADS", 4)))
    if test_config:
        app.config.update(test_config)
    app.config["DATA_DIR"] = Path(app.config["DATA_DIR"])
    app.config["SCANS_DIR"] = app.config["DATA_DIR"] / "scans"
    app.config["SCANS_DIR"].mkdir(parents=True, exist_ok=True)
    app.secret_key = _secret_key(app.config["DATA_DIR"])

    store = TwinStore(app.config["DATA_DIR"] / "twin.db")
    model = inf.E0Model(app.config["MODEL_DIR"], threads=app.config["THREADS"])
    check = model.path.parent / "export_check.json"
    model_check = json.loads(check.read_text()) if check.exists() else {}
    app.extensions["e0"] = {"store": store, "model": model}
    sample_dir = APP_DIR / "static" / "samples"
    samples = sorted(p.name for p in sample_dir.glob("*.jpg"))
    meta_file = sample_dir / "samples.json"
    sample_meta = json.loads(meta_file.read_text()) if meta_file.exists() else {}

    # ---------------------------------------------------------------- helpers
    def media(stem, kind):
        return url_for("media", filename=f"{stem}_{kind}.jpg")

    def analyse(raw: bytes, image_name: str, patient_id: str, scan_date: str, notes: str = ""):
        img = inf.decode_image(raw)
        pred = model.predict(img)
        stem = f"{datetime.now():%Y%m%d-%H%M%S}-{uuid.uuid4().hex[:8]}"
        d = app.config["SCANS_DIR"]
        inf.save_jpeg(d / f"{stem}_original.jpg", img)
        inf.save_jpeg(d / f"{stem}_outline.jpg", inf.render_outline(img, pred))
        inf.save_jpeg(d / f"{stem}_heatmap.jpg", inf.render_heatmap(img, pred))
        store.ensure_patient(patient_id)
        vid = store.add_visit(patient_id=patient_id, scan_date=scan_date,
                              created_at=datetime.now().isoformat(timespec="seconds"), prediction=pred.prediction,
                              prob=pred.prob, threshold=pred.threshold, score=pred.score, confidence=pred.confidence,
                              model_version=model.version, image_name=image_name[:120], file_stem=stem,
                              infer_ms=pred.timings_ms["inference"], notes=notes[:500] or None)
        return vid, pred

    def valid_date(s) -> str:
        try:
            return date.fromisoformat(s).isoformat()
        except (TypeError, ValueError):
            return date.today().isoformat()

    def remove_files(stems):
        for s in stems:
            for k in ("original", "outline", "heatmap"):
                (app.config["SCANS_DIR"] / f"{s}_{k}.jpg").unlink(missing_ok=True)

    @app.context_processor
    def globals_():
        return {"media": media, "boundary": RISK_THRESHOLD, "model_version": model.version}

    @app.template_filter("pct")
    def pct(v, digits=0):
        """Percent for display; never rounds a model output to a flat 0% or 100%."""
        if v is None:
            return "–"
        lo, hi = 0.5 * 10 ** -(digits + 2), 1 - 0.5 * 10 ** -(digits + 2)
        if v >= hi:
            return f">{100 - 10 ** -digits:.{digits}f}%"
        if v < lo:
            return f"<{10 ** -digits:.{digits}f}%"
        return f"{100 * v:.{digits}f}%"

    # ---------------------------------------------------------------- pages
    @app.get("/")
    def dashboard():
        twins = []
        for p in store.patients():
            st, visits = store.state(p["id"])
            twins.append({"p": p, "st": st, "last": visits[-1] if visits else None, "n": len(visits)})
        order = {"alert": 0, "high": 1, "low": 2, "none": 3}
        twins.sort(key=lambda t: (order[t["st"].status], -(t["st"].risk or 0)))
        return render_template("dashboard.html", twins=twins, counts=store.counts(), recent=store.recent_visits(6),
                               flagged=sum(t["st"].flag for t in twins))

    @app.route("/scan", methods=["GET", "POST"])
    def scan():
        if request.method == "GET":
            return render_template("scan.html", patients=store.patients(), today=date.today().isoformat(),
                                   samples=samples, sample_meta=sample_meta, preselect=request.args.get("patient", ""))
        f = request.form
        pid = (f.get("new_patient_id") or "").strip() or (f.get("patient_id") or "").strip()
        if not PID_RE.match(pid):
            flash("Choose a patient, or type a new patient ID (letters, digits, - _ . ; up to 40 characters).", "error")
            return redirect(url_for("scan"))
        sample = f.get("sample", "")
        upload = request.files.get("image")
        if upload and upload.filename:
            if Path(upload.filename).suffix.lower() not in ALLOWED_EXT:
                flash("Unsupported file type. Upload a JPG, PNG, BMP or TIFF ultrasound frame.", "error")
                return redirect(url_for("scan", patient=pid))
            raw, name = upload.read(), upload.filename
        elif sample in samples:
            raw, name = (sample_dir / sample).read_bytes(), f"sample {sample}"
        else:
            flash("Choose an ultrasound image to analyse.", "error")
            return redirect(url_for("scan", patient=pid))
        try:
            vid, _ = analyse(raw, name, pid, valid_date(f.get("scan_date")), (f.get("notes") or "").strip())
        except ValueError as e:
            flash(str(e), "error")
            return redirect(url_for("scan", patient=pid))
        return redirect(url_for("visit", vid=vid))

    @app.get("/visits/<int:vid>")
    def visit(vid):
        v = store.visit(vid) or abort(404)
        st, visits = store.state(v["patient_id"])
        idx = next(i for i, x in enumerate(visits) if x["id"] == vid)
        return render_template("visit.html", v=visits[idx], st=st, prev=visits[idx - 1] if idx > 0 else None,
                               n=len(visits), idx=idx, patient=store.patient(v["patient_id"]),
                               is_latest=idx == len(visits) - 1)

    @app.post("/visits/<int:vid>/delete")
    def delete_visit(vid):
        v = store.visit(vid) or abort(404)
        store.delete_visit(vid)
        remove_files([v["file_stem"]])
        flash(f"Scan from {v['scan_date']} deleted.", "ok")
        return redirect(url_for("patient", pid=v["patient_id"]))

    @app.route("/patients/new", methods=["GET", "POST"])
    def new_patient():
        if request.method == "GET":
            return render_template("patient_new.html", year=date.today().year)
        f = request.form
        pid = (f.get("id") or "").strip()
        if not PID_RE.match(pid):
            flash("Patient ID: letters, digits, - _ . only, up to 40 characters.", "error")
            return redirect(url_for("new_patient"))
        if store.patient(pid):
            flash(f"Patient {pid} already exists.", "error")
            return redirect(url_for("patient", pid=pid))
        by = (f.get("birth_year") or "").strip()
        by = int(by) if by.isdigit() and 1900 <= int(by) <= date.today().year else None
        store.add_patient(pid, (f.get("name") or "").strip()[:80], by, (f.get("notes") or "").strip()[:500])
        flash(f"Digital twin created for {pid}. Add the first scan.", "ok")
        return redirect(url_for("scan", patient=pid))

    @app.get("/patients/<pid>")
    def patient(pid):
        p = store.patient(pid) or abort(404)
        st, visits = store.state(pid)
        return render_template("patient.html", p=p, st=st, visits=visits, chart=chart(visits) if visits else None,
                               year=date.today().year)

    @app.post("/patients/<pid>/delete")
    def delete_patient(pid):
        store.patient(pid) or abort(404)
        remove_files(store.delete_patient(pid))
        flash(f"Patient {pid} and all their scans were deleted.", "ok")
        return redirect(url_for("dashboard"))

    @app.get("/model")
    def model_card():
        return render_template("model.html", info=model.info, check=model_check, backend=model.backend,
                               size_mb=model.path.stat().st_size / 2**20)

    @app.get("/media/<path:filename>", endpoint="media")
    def media_file(filename):
        return send_from_directory(app.config["SCANS_DIR"], filename, max_age=86400)

    # ---------------------------------------------------------------- JSON API
    @app.get("/api/health")
    def health():
        return jsonify(ok=True, model=model.version, backend=model.backend, threshold=model.threshold)

    @app.post("/api/predict")
    def api_predict():
        up = request.files.get("image")
        if not up:
            return jsonify(error="multipart field 'image' is required"), 400
        pid = (request.form.get("patient_id") or "").strip()
        if pid and not PID_RE.match(pid):
            return jsonify(error="invalid patient_id"), 400
        try:
            if pid:
                vid, p = analyse(up.read(), up.filename or "api", pid, valid_date(request.form.get("scan_date")))
            else:
                vid, p = None, model.predict(inf.decode_image(up.read()))
        except ValueError as e:
            return jsonify(error=str(e)), 400
        out = {"prediction": p.prediction, "label": p.label, "confidence": round(p.confidence, 4),
               "malignancy_score": round(p.score, 4), "raw_probability": round(p.prob, 6),
               "threshold": p.threshold, "timings_ms": p.timings_ms}
        if vid:
            st, _ = store.state(pid)
            out.update(visit_id=vid, url=url_for("visit", vid=vid, _external=True),
                       twin={"risk": round(st.risk, 4), "trend": st.trend, "flag": st.flag, "reason": st.reason})
        return jsonify(out)

    @app.errorhandler(404)
    def not_found(_):
        return render_template("error.html", code=404, msg="That page or record doesn't exist."), 404

    @app.errorhandler(413)
    def too_big(_):
        flash("That file is larger than 20 MB.", "error")
        return redirect(url_for("scan"))

    return app

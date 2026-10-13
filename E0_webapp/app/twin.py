"""Digital twin: per-patient visit history in SQLite + an EWMA malignancy-risk state with a rising-risk flag.

Same risk model as ovarian-twin/src/twin/risk_model.py:
  risk_t = alpha * score_t + (1 - alpha) * risk_{t-1}   (score = threshold-centred malignancy score, 0.5 = boundary)
  trend  = rising / falling / stable from the change between the last two visits
  flag   = risk rose by more than RISE_DELTA since the last visit, or crossed 0.5 upwards, or the latest scan itself
           scored >= 0.5 (so a malignant result is never shown as all-clear while the smoothed risk catches up)
"""
import sqlite3
import threading
from dataclasses import dataclass
from datetime import date, datetime
from pathlib import Path

ALPHA, RISE_DELTA, RISK_THRESHOLD, STABLE_BAND = 0.5, 0.15, 0.5, 0.05

SCHEMA = """
CREATE TABLE IF NOT EXISTS patients (
    id          TEXT PRIMARY KEY,
    name        TEXT,
    birth_year  INTEGER,
    notes       TEXT,
    created_at  TEXT NOT NULL
);
CREATE TABLE IF NOT EXISTS visits (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    patient_id     TEXT NOT NULL REFERENCES patients(id) ON DELETE CASCADE,
    scan_date      TEXT NOT NULL,          -- YYYY-MM-DD, entered by the user (defaults to today)
    created_at     TEXT NOT NULL,          -- ISO timestamp the scan was analysed
    prediction     TEXT NOT NULL,          -- 'benign' | 'malignant'
    prob           REAL NOT NULL,          -- raw model probability
    threshold      REAL NOT NULL,
    score          REAL NOT NULL,          -- threshold-centred malignancy score
    confidence     REAL NOT NULL,          -- confidence in the stated prediction
    model_version  TEXT NOT NULL,
    image_name     TEXT,                   -- original upload filename (display only)
    file_stem      TEXT NOT NULL,          -- data/scans/<file_stem>_{original,outline,heatmap}.jpg
    infer_ms       REAL,
    notes          TEXT
);
CREATE INDEX IF NOT EXISTS idx_visits_patient ON visits(patient_id, scan_date, id);
"""


@dataclass
class TwinState:
    risk: float | None
    previous_risk: float | None
    trend: str        # 'no scans' | 'baseline' | 'rising' | 'falling' | 'stable'
    flag: bool
    reason: str
    series: list      # risk after each visit, in visit order

    @property
    def status(self) -> str:
        if self.risk is None:
            return "none"
        if self.flag:
            return "alert"
        return "high" if self.risk >= RISK_THRESHOLD else "low"


def twin_state(scores: list[float]) -> TwinState:
    if not scores:
        return TwinState(None, None, "no scans", False, "", [])
    series, risk = [], None
    for s in scores:
        risk = s if risk is None else ALPHA * s + (1 - ALPHA) * risk
        series.append(risk)
    risk = series[-1]
    if len(series) == 1:
        high = risk >= RISK_THRESHOLD
        return TwinState(risk, None, "baseline", high, "first scan already above the decision boundary" if high else "",
                         series)
    prev = series[-2]
    delta = risk - prev
    trend = "rising" if delta > STABLE_BAND else "falling" if delta < -STABLE_BAND else "stable"
    reasons = []
    if delta > RISE_DELTA:
        reasons.append(f"risk rose {100 * delta:+.0f} points since the previous scan")
    if prev < RISK_THRESHOLD <= risk:
        reasons.append("risk crossed the decision boundary")
    if scores[-1] >= RISK_THRESHOLD:
        reasons.append("the latest scan scored malignant")
    return TwinState(risk, prev, trend, bool(reasons), "; ".join(reasons), series)


class TwinStore:
    def __init__(self, db_path):
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)
        self.conn = sqlite3.connect(str(db_path), check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self.conn.execute("PRAGMA foreign_keys = ON")
        self.conn.executescript(SCHEMA)
        self.lock = threading.Lock()

    def _q(self, sql, args=()):
        with self.lock:
            return [dict(r) for r in self.conn.execute(sql, args).fetchall()]

    def _x(self, sql, args=()):
        with self.lock:
            cur = self.conn.execute(sql, args)
            self.conn.commit()
            return cur.lastrowid

    # ---- patients
    def add_patient(self, pid, name=None, birth_year=None, notes=None):
        self._x("INSERT INTO patients(id, name, birth_year, notes, created_at) VALUES (?, ?, ?, ?, ?)",
                (pid, name or None, birth_year, notes or None, datetime.now().isoformat(timespec="seconds")))

    def ensure_patient(self, pid):
        if not self.patient(pid):
            self.add_patient(pid)

    def patient(self, pid):
        rows = self._q("SELECT * FROM patients WHERE id = ?", (pid,))
        return rows[0] if rows else None

    def patients(self):
        return self._q("SELECT * FROM patients ORDER BY created_at DESC")

    def delete_patient(self, pid):
        stems = [v["file_stem"] for v in self.visits(pid)]
        self._x("DELETE FROM patients WHERE id = ?", (pid,))
        return stems

    # ---- visits
    VISIT_COLUMNS = ("patient_id", "scan_date", "created_at", "prediction", "prob", "threshold", "score", "confidence",
                     "model_version", "image_name", "file_stem", "infer_ms", "notes")

    def add_visit(self, **v) -> int:
        cols = [c for c in self.VISIT_COLUMNS if c in v]
        return self._x(f"INSERT INTO visits({','.join(cols)}) VALUES ({','.join('?' * len(cols))})",
                       tuple(v[c] for c in cols))

    def visit(self, vid):
        rows = self._q("SELECT * FROM visits WHERE id = ?", (vid,))
        return rows[0] if rows else None

    def visits(self, pid):
        return self._q("SELECT * FROM visits WHERE patient_id = ? ORDER BY scan_date, id", (pid,))

    def recent_visits(self, limit=8):
        return self._q("SELECT * FROM visits ORDER BY created_at DESC, id DESC LIMIT ?", (limit,))

    def delete_visit(self, vid):
        self._x("DELETE FROM visits WHERE id = ?", (vid,))

    def counts(self):
        return self._q("SELECT (SELECT COUNT(*) FROM patients) AS patients, (SELECT COUNT(*) FROM visits) AS visits, "
                       "(SELECT COUNT(*) FROM visits WHERE prediction = 'malignant') AS malignant")[0]

    # ---- twin
    def state(self, pid) -> tuple[TwinState, list]:
        visits = self.visits(pid)
        st = twin_state([v["score"] for v in visits])
        for v, r in zip(visits, st.series):
            v["risk_after"] = r
        return st, visits


def chart(visits: list, width=720, height=240, pad_l=44, pad_r=16, pad_t=16, pad_b=34) -> dict:
    """Geometry for the server-side SVG twin chart: x = visit order, y = 0..1 (score and smoothed risk)."""
    n = len(visits)
    iw, ih = width - pad_l - pad_r, height - pad_t - pad_b
    days = [date.fromisoformat(v["scan_date"]).toordinal() for v in visits]
    span = days[-1] - days[0] if n else 0
    # x is proportional to time; fall back to even spacing when every scan is on the same day
    xs = [pad_l + (iw / 2 if n == 1 else iw * ((d - days[0]) / span if span else i / (n - 1)))
          for i, d in enumerate(days)]
    y = lambda v: pad_t + ih * (1 - v)  # noqa: E731
    pts = [{"x": round(x, 1), "ys": round(y(v["score"]), 1), "yr": round(y(v["risk_after"]), 1), "v": v}
           for x, v in zip(xs, visits)]
    last = None  # date labels: keep the last one, drop any that would sit within 64 px of a kept label
    for p in reversed(pts):
        p["label"] = last is None or last - p["x"] >= 64
        if p["label"]:
            last = p["x"]
    return {"w": width, "h": height, "pad_l": pad_l, "pad_r": pad_r, "pad_t": pad_t, "ih": ih, "iw": iw,
            "y_boundary": round(y(RISK_THRESHOLD), 1), "y0": round(y(0), 1),
            "ticks": [{"v": t, "y": round(y(t), 1)} for t in (0, 0.25, 0.5, 0.75, 1)],
            "risk_path": " ".join(f"{'M' if i == 0 else 'L'}{p['x']},{p['yr']}" for i, p in enumerate(pts)),
            "points": pts}

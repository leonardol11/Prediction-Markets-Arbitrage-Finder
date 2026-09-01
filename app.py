"""Web front end for the prediction-market arbitrage finder.

    python3 app.py            # then open http://127.0.0.1:5000

One scan runs at a time. The browser starts it, streams the log over
Server-Sent Events, then renders the finished sheets and can download the
same data as a styled .xlsx workbook.
"""

import io
import json
import os
import queue
import threading
import uuid
import webbrowser
from datetime import datetime, timezone

from flask import (Flask, Response, jsonify, render_template, request,
                   send_file, stream_with_context)

from matcher import (Cancelled, Reporter, ScanConfig, TOTAL_STEPS, build_sheets,
                     build_summary, parse_cutoff, run_scan, workbook_bytes)

app = Flask(__name__)
app.config["JSON_SORT_KEYS"] = False

CACHE_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), ".cache")
CACHE_FILE = os.path.join(CACHE_DIR, "last_scan.json")

_lock = threading.Lock()
_jobs = {}
_latest_done = None          # job id of the most recent successful scan


class Job:
    """One scan: its live log, its progress, and its finished sheets."""

    def __init__(self, cfg: ScanConfig):
        self.id = uuid.uuid4().hex[:12]
        self.config = cfg
        self.status = "running"           # running | done | error | cancelled
        self.created_at = datetime.now(timezone.utc)
        self.logs = []
        self.step = 0
        self.step_label = "Starting…"
        self.step_pct = None
        self.error = None
        self.sheets = None
        self.summary = None
        self._xlsx = None
        self._cancel = threading.Event()
        self._subscribers = []
        self._sub_lock = threading.Lock()

    # ── event fan-out ────────────────────────────────────────────────────────
    def subscribe(self) -> queue.Queue:
        q = queue.Queue()
        with self._sub_lock:
            self._subscribers.append(q)
        return q

    def unsubscribe(self, q):
        with self._sub_lock:
            if q in self._subscribers:
                self._subscribers.remove(q)

    def _emit(self, event: str, data: dict):
        payload = (event, data)
        with self._sub_lock:
            for q in list(self._subscribers):
                q.put(payload)

    # ── reporter hooks ───────────────────────────────────────────────────────
    def on_log(self, message: str):
        stamp = datetime.now(timezone.utc).strftime("%H:%M:%S")
        line = {"t": stamp, "m": message}
        self.logs.append(line)
        del self.logs[:-600]
        self._emit("log", line)

    def on_progress(self, step: int, label: str, pct=None):
        self.step, self.step_label, self.step_pct = step, label, pct
        self._emit("progress", {"step": step, "total": TOTAL_STEPS,
                                "label": label, "pct": pct})

    def cancel(self):
        self._cancel.set()

    @property
    def cancelled(self) -> bool:
        return self._cancel.is_set()

    def state(self) -> dict:
        return {
            "id": self.id, "status": self.status,
            "step": self.step, "total": TOTAL_STEPS,
            "label": self.step_label, "pct": self.step_pct,
            "error": self.error,
        }

    def xlsx(self) -> bytes:
        if self._xlsx is None and self.sheets:
            self._xlsx = workbook_bytes(self.sheets)
        return self._xlsx


def _save_cache(job: Job):
    """Keep the last finished scan on disk so a restart doesn't lose it."""
    try:
        os.makedirs(CACHE_DIR, exist_ok=True)
        tmp = CACHE_FILE + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"id": job.id, "summary": job.summary, "sheets": job.sheets}, fh)
        os.replace(tmp, CACHE_FILE)
    except OSError as exc:
        job.on_log(f"  (could not cache results: {exc})")


def _load_cache():
    """Re-register the last finished scan as a job so every route just works."""
    global _latest_done
    try:
        with open(CACHE_FILE) as fh:
            data = json.load(fh)
    except (OSError, ValueError):
        return
    job = Job(ScanConfig())
    job.id = data.get("id") or job.id
    job.status = "done"
    job.summary = data.get("summary")
    job.sheets = data.get("sheets")
    if not job.sheets:
        return
    _jobs[job.id] = job
    _latest_done = job.id


def _run_job(job: Job):
    global _latest_done
    rep = Reporter(on_log=job.on_log, on_progress=job.on_progress,
                   should_cancel=lambda: job.cancelled)
    try:
        result = run_scan(job.config, rep)
        job.sheets = build_sheets(result)
        job.summary = build_summary(result)
        job.status = "done"
        with _lock:
            _latest_done = job.id
        _save_cache(job)
        job._emit("done", {"summary": job.summary})
    except Cancelled:
        job.status = "cancelled"
        job.on_log("Scan cancelled.")
        job._emit("cancelled", {})
    except Exception as exc:                     # surface the failure in the UI
        job.status = "error"
        job.error = f"{type(exc).__name__}: {exc}"
        job.on_log(f"✕ {job.error}")
        job._emit("scan_error", {"error": job.error})
    finally:
        job._emit("end", {"status": job.status})


# ─────────────────────────────────── routes ───────────────────────────────────

@app.get("/")
def index():
    defaults = ScanConfig()
    return render_template(
        "index.html",
        defaults={
            "min_volume": int(defaults.min_volume),
            "cutoff": defaults.cutoff_date.strftime("%Y-%m-%d"),
            "similarity": defaults.match_display_threshold,
            "confidence": defaults.groq_confidence_cutoff,
            "min_margin": defaults.min_margin,
            "model": defaults.groq_model,
        },
        has_env_key=bool(os.environ.get("GROQ_API_KEY")),
    )


def _config_from_request(body: dict) -> ScanConfig:
    def f(key, default):
        try:
            v = body.get(key)
            return default if v in (None, "") else float(v)
        except (TypeError, ValueError):
            return default

    base = ScanConfig()
    threshold = f("similarity", base.match_display_threshold)
    return ScanConfig(
        min_volume=max(0.0, f("min_volume", base.min_volume)),
        cutoff_date=parse_cutoff(body.get("cutoff")
                                 or base.cutoff_date.strftime("%Y-%m-%d")),
        tfidf_threshold=min(threshold, base.tfidf_threshold),
        combined_threshold=threshold,
        match_display_threshold=threshold,
        use_llm=bool(body.get("use_llm", True)),
        groq_api_key=(body.get("groq_api_key") or "").strip(),
        groq_confidence_cutoff=f("confidence", base.groq_confidence_cutoff),
        min_margin=max(0.0, f("min_margin", base.min_margin)),
    )


@app.post("/api/scan")
def start_scan():
    body = request.get_json(silent=True) or {}
    with _lock:
        running = [j for j in _jobs.values() if j.status == "running"]
        if running:
            return jsonify({"error": "A scan is already running.",
                            "id": running[0].id}), 409
        try:
            cfg = _config_from_request(body)
        except ValueError as exc:
            return jsonify({"error": f"Invalid settings: {exc}"}), 400
        job = Job(cfg)
        _jobs[job.id] = job
        for old in sorted(_jobs.values(), key=lambda j: j.created_at)[:-6]:
            _jobs.pop(old.id, None)

    threading.Thread(target=_run_job, args=(job,), daemon=True).start()
    return jsonify({"id": job.id, "llm_enabled": cfg.llm_enabled})


@app.post("/api/scan/<job_id>/cancel")
def cancel_scan(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown scan."}), 404
    job.cancel()
    return jsonify(job.state())


@app.get("/api/scan/<job_id>/stream")
def stream(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown scan."}), 404

    def gen():
        q = job.subscribe()
        try:
            yield _sse("state", job.state())
            for line in job.logs[-200:]:
                yield _sse("log", line)
            if job.status != "running":
                yield _sse("end", {"status": job.status})
                return
            while True:
                try:
                    event, data = q.get(timeout=15)
                except queue.Empty:
                    yield ": ping\n\n"
                    continue
                yield _sse(event, data)
                if event == "end":
                    return
        finally:
            job.unsubscribe(q)

    return Response(stream_with_context(gen()), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache",
                             "X-Accel-Buffering": "no",
                             "Connection": "keep-alive"})


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


@app.get("/api/scan/<job_id>/results")
def results(job_id):
    job = _jobs.get(job_id)
    if not job:
        return jsonify({"error": "Unknown scan."}), 404
    if job.status == "running":
        return jsonify({"status": "running", **job.state()}), 202
    if not job.sheets:
        return jsonify({"status": job.status, "error": job.error}), 200
    return jsonify({"status": job.status, "summary": job.summary,
                    "sheets": job.sheets})


@app.get("/api/current")
def current():
    """Lets a reloaded page reattach to a scan that is still running."""
    job = next((j for j in _jobs.values() if j.status == "running"), None)
    return jsonify({"id": job.id, **job.state()} if job else {"id": None})


@app.get("/api/latest")
def latest():
    job = _jobs.get(_latest_done)
    if not job or not job.sheets:
        return jsonify({"status": "empty"})
    return jsonify({"status": "done", "id": job.id,
                    "summary": job.summary, "sheets": job.sheets})


@app.get("/api/scan/<job_id>/download")
def download(job_id):
    job = _jobs.get(job_id)
    if not job or not job.sheets:
        return jsonify({"error": "No results to download."}), 404
    stamp = (job.summary or {}).get("finished_at", "")[:10] or "results"
    return send_file(
        io.BytesIO(job.xlsx()),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        as_attachment=True,
        download_name=f"market-arbitrage-{stamp}.xlsx",
    )


_load_cache()


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    host = os.environ.get("HOST", "127.0.0.1")
    url = f"http://{'127.0.0.1' if host == '0.0.0.0' else host}:{port}"
    print(f"\n  Prediction-market arbitrage finder → {url}")
    print("  Press Ctrl-C to stop.\n")

    if not os.environ.get("NO_BROWSER") and not os.environ.get("WERKZEUG_RUN_MAIN"):
        threading.Timer(1.0, lambda: webbrowser.open(url)).start()

    app.run(host=host, port=port,
            debug=bool(os.environ.get("FLASK_DEBUG")), threaded=True)

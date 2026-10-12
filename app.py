"""Web front end for the prediction-market arbitrage finder.

    python3 app.py            # then open http://127.0.0.1:5000

The server keeps no state between requests, so it runs the same way locally
and as a serverless function on Vercel:

  POST /api/scan      runs one scan inside the request and streams its log as
                      Server-Sent Events; the final event carries the sheets
                      plus a compact repricing context.
  POST /api/refresh   reprices that context and returns the repriced sheets.
  POST /api/workbook  turns the displayed sheets into a styled .xlsx file.

The browser keeps the last finished scan, so a page reload brings it back.
Large request bodies may be gzip-compressed and flagged with the
X-Content-Encoding header, which keeps them under Vercel's request size limit.
"""

import json
import os
import queue
import threading
import time
import webbrowser
import zlib
from datetime import datetime, timezone

from flask import Flask, Response, jsonify, render_template, request

from matcher import (Cancelled, Reporter, ScanConfig, ScanResult, TOTAL_STEPS,
                     build_sheets, build_summary, parse_cutoff, refresh_scan,
                     run_scan, workbook_bytes)

ROOT = os.path.dirname(os.path.abspath(__file__))

# Assets live in public/ so Vercel serves them from its CDN; Flask serves the
# same folder at the same URL when running locally.
app = Flask(__name__, static_folder=os.path.join(ROOT, "public", "static"),
            static_url_path="/static")
app.config["JSON_SORT_KEYS"] = False
app.config["MAX_CONTENT_LENGTH"] = 32 * 1024 * 1024

MAX_INFLATED_BYTES = 256 * 1024 * 1024
REPRICED_SHEETS = ("arbitrage", "verified", "matches")
CONTEXT_VERSION = 2


def _scan_time_limit() -> float:
    """Seconds a scan may run before it stops itself; 0 means no limit.

    Vercel kills a function at its max duration (300s on Hobby, configured in
    vercel.json). Stopping a little earlier lets the scan report a clear error
    instead of the connection dropping mid-stream.
    """
    raw = os.environ.get("SCAN_TIME_LIMIT")
    if raw:
        try:
            return max(0.0, float(raw))
        except ValueError:
            pass
    return 270.0 if os.environ.get("VERCEL") else 0.0


# ─────────────────────────────── request helpers ──────────────────────────────

class BadRequest(ValueError):
    pass


def _json_body() -> dict:
    raw = request.get_data(cache=False)
    if request.headers.get("X-Content-Encoding", "").lower() == "gzip":
        inflater = zlib.decompressobj(16 + zlib.MAX_WBITS)
        try:
            raw = inflater.decompress(raw, MAX_INFLATED_BYTES)
        except zlib.error as exc:
            raise BadRequest(f"Invalid gzip body: {exc}") from exc
        if inflater.unconsumed_tail:
            raise BadRequest("Request body is too large.")
    if not raw:
        return {}
    try:
        data = json.loads(raw)
    except ValueError as exc:
        raise BadRequest("Request body is not valid JSON.") from exc
    if not isinstance(data, dict):
        raise BadRequest("Request body must be a JSON object.")
    return data


@app.errorhandler(BadRequest)
def _bad_request(exc):
    return jsonify({"error": str(exc)}), 400


@app.errorhandler(413)
def _too_large(_exc):
    return jsonify({"error": "Request body is too large."}), 413


def _sse(event: str, data: dict) -> str:
    return f"event: {event}\ndata: {json.dumps(data)}\n\n"


# ─────────────────────────────── repricing context ────────────────────────────

def _serialize_context(result: ScanResult) -> dict:
    """Everything refresh needs, and nothing more.

    Only markets referenced by a pair are kept, so the context stays small
    enough for the browser to send back. API keys are never included.
    """
    poly, kalshi, p_ix, k_ix = [], [], {}, {}

    def add(m, markets, index):
        if id(m) not in index:
            index[id(m)] = len(markets)
            markets.append(m)
        return index[id(m)]

    def pairs(rows):
        return [[add(p, poly, p_ix), add(k, kalshi, k_ix), sorted(shared), score]
                for p, k, shared, score in rows]

    matches, verified = pairs(result.matches), pairs(result.verified)
    matched_poly = [add(m, poly, p_ix) for m in result.matched_poly]
    matched_kalshi = [add(m, kalshi, k_ix) for m in result.matched_kalshi]
    cfg = result.config
    return {
        "version": CONTEXT_VERSION,
        "config": {
            "min_volume": cfg.min_volume,
            "cutoff_date": cfg.cutoff_date.isoformat(),
            "match_display_threshold": cfg.match_display_threshold,
            "groq_confidence_cutoff": cfg.groq_confidence_cutoff,
            "min_margin": cfg.min_margin,
            "orderbook_workers": cfg.orderbook_workers,
        },
        "poly": poly, "kalshi": kalshi,
        "poly_count": len(result.poly), "kalshi_count": len(result.kalshi),
        "matches": matches, "verified": verified,
        "matched_poly": matched_poly, "matched_kalshi": matched_kalshi,
        "started_at": result.started_at.isoformat(),
        "finished_at": (result.finished_at or result.started_at).isoformat(),
        "prices_updated_at": (
            result.prices_updated_at.isoformat() if result.prices_updated_at else None),
        "arb_source_label": result.arb_source_label, "llm_used": result.llm_used,
    }


def _restore_context(data):
    if not isinstance(data, dict) or data.get("version") != CONTEXT_VERSION:
        raise BadRequest("This snapshot can't be repriced. Run a new full scan.")
    try:
        settings = data["config"]
        cfg = ScanConfig(
            min_volume=float(settings["min_volume"]),
            cutoff_date=datetime.fromisoformat(settings["cutoff_date"]),
            match_display_threshold=float(settings["match_display_threshold"]),
            groq_confidence_cutoff=float(settings["groq_confidence_cutoff"]),
            min_margin=float(settings["min_margin"]),
            orderbook_workers=max(1, min(int(settings["orderbook_workers"]), 128)),
            use_llm=False,
        )
        # Refresh never calls Groq, even when the environment supplies a key.
        cfg.groq_api_key = ""
        result = ScanResult(cfg)
        result.poly, result.kalshi = list(data["poly"]), list(data["kalshi"])
        if not all(isinstance(m, dict) for m in result.poly + result.kalshi):
            raise ValueError("Invalid cached market")

        def market(markets, index):
            if type(index) is not int or not 0 <= index < len(markets):
                raise ValueError("Invalid cached market reference")
            return markets[index]

        def pairs(rows):
            return [(market(result.poly, pi), market(result.kalshi, ki),
                     set(shared), float(score))
                    for pi, ki, shared, score in rows]

        result.matches, result.verified = pairs(data["matches"]), pairs(data["verified"])
        result.matched_poly = [market(result.poly, i) for i in data["matched_poly"]]
        result.matched_kalshi = [market(result.kalshi, i) for i in data["matched_kalshi"]]
        result.started_at = datetime.fromisoformat(data["started_at"])
        result.finished_at = datetime.fromisoformat(data["finished_at"])
        stamp = data.get("prices_updated_at")
        result.prices_updated_at = datetime.fromisoformat(stamp) if stamp else None
        result.arb_source_label = str(data["arb_source_label"])
        result.llm_used = bool(data["llm_used"])
        counts = int(data["poly_count"]), int(data["kalshi_count"])
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise BadRequest(f"Invalid repricing context: {exc}") from exc
    return result, counts


def _payload(result: ScanResult, sheets, summary) -> dict:
    stamp = result.prices_updated_at
    return {
        "status": "done",
        "summary": summary,
        "sheets": sheets,
        "prices_updated_at": stamp.isoformat() if stamp else None,
        "refresh_available": bool(result.matched_poly or result.matched_kalshi),
        "context": _serialize_context(result),
    }


# ─────────────────────────────────── routes ───────────────────────────────────

@app.get("/")
def index():
    defaults = ScanConfig()
    limit = _scan_time_limit()
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
        scan_time_limit=int(limit),
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
def scan():
    """Run a scan for the lifetime of this request, streaming its progress."""
    body = _json_body()
    try:
        cfg = _config_from_request(body)
    except ValueError as exc:
        return jsonify({"error": f"Invalid settings: {exc}"}), 400

    events = queue.Queue()
    stop = threading.Event()
    limit = _scan_time_limit()
    deadline = time.monotonic() + limit if limit else None

    def timed_out() -> bool:
        return deadline is not None and time.monotonic() > deadline

    def on_log(message):
        events.put(("log", {"t": datetime.now(timezone.utc).strftime("%H:%M:%S"),
                            "m": message}))

    def on_progress(step, label, pct=None):
        events.put(("progress", {"step": step, "total": TOTAL_STEPS,
                                 "label": label, "pct": pct}))

    def work():
        status = "error"
        rep = Reporter(on_log=on_log, on_progress=on_progress,
                       should_cancel=lambda: stop.is_set() or timed_out())
        try:
            result = run_scan(cfg, rep)
            payload = _payload(result, build_sheets(result), build_summary(result))
            events.put(("done", payload))
            status = "done"
        except Cancelled:
            if timed_out():
                error = (f"The scan hit the {int(limit)}s server time limit. "
                         "Try a sooner 'Resolves before' date or a higher minimum "
                         "volume, or raise the limit (see README).")
                on_log(f"✕ {error}")
                events.put(("scan_error", {"error": error}))
            else:
                status = "cancelled"
                on_log("Scan cancelled.")
                events.put(("cancelled", {}))
        except Exception as exc:                 # surface the failure in the UI
            error = f"{type(exc).__name__}: {exc}"
            on_log(f"✕ {error}")
            events.put(("scan_error", {"error": error}))
        finally:
            events.put(("end", {"status": status}))

    def stream():
        threading.Thread(target=work, daemon=True).start()
        try:
            yield _sse("start", {"total": TOTAL_STEPS, "llm_enabled": cfg.llm_enabled,
                                 "time_limit": int(limit)})
            while True:
                try:
                    event, data = events.get(timeout=10)
                except queue.Empty:
                    yield ": ping\n\n"           # keeps proxies from closing the stream
                    continue
                yield _sse(event, data)
                if event == "end":
                    return
        finally:
            # Runs when the browser disconnects too, so an abandoned scan stops.
            stop.set()

    return Response(stream(), mimetype="text/event-stream",
                    headers={"Cache-Control": "no-cache, no-transform",
                             "X-Accel-Buffering": "no"})


@app.post("/api/refresh")
def refresh():
    body = _json_body()
    result, (poly_count, kalshi_count) = _restore_context(body.get("context"))
    try:
        refreshed = refresh_scan(result, Reporter(on_log=lambda _m: None))
    except Exception as exc:
        return jsonify({"error": f"Price refresh failed: {exc}"}), 502
    sheets = [s for s in build_sheets(refreshed) if s["key"] in REPRICED_SHEETS]
    summary = build_summary(refreshed)
    summary.update(poly_count=poly_count, kalshi_count=kalshi_count)
    payload = _payload(refreshed, sheets, summary)
    payload["context"].update(poly_count=poly_count, kalshi_count=kalshi_count)
    return jsonify(payload)


@app.post("/api/workbook")
def workbook():
    body = _json_body()
    sheets = body.get("sheets")
    if not isinstance(sheets, list) or not sheets:
        raise BadRequest("No results to download.")
    try:
        content = workbook_bytes(sheets)
    except (KeyError, TypeError, ValueError, IndexError, AttributeError) as exc:
        raise BadRequest(f"Invalid sheet data: {exc}") from exc
    stamp = str(body.get("stamp") or "")[:10]
    stamp = stamp if stamp.replace("-", "").isdigit() else "results"

    def chunks(size=256 * 1024):
        # Streamed so large workbooks aren't held to the buffered response limit.
        for i in range(0, len(content), size):
            yield content[i:i + size]

    return Response(
        chunks(),
        mimetype="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
        headers={"Content-Disposition":
                 f'attachment; filename="market-arbitrage-{stamp}.xlsx"'},
    )


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

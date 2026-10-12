/* Discovery streams over one POST request; live repricing updates the displayed
   workbook without discarding the reader's active sheet, filter, sort or position.
   The server is stateless (it runs on Vercel), so the browser keeps the latest
   snapshot, including the context the server needs to reprice it. */

const $ = (id) => document.getElementById(id);
const CHUNK = 120;                       // rows appended per render pass
const REFRESH_MS = 30_000;

// Storage can be unavailable in private or restricted browser contexts.
function stored(key, fallback = null) {
  try { return localStorage.getItem(key) ?? fallback; } catch (_) { return fallback; }
}
function persist(key, value) {
  try { localStorage.setItem(key, value); } catch (_) { /* Session still works. */ }
}

const state = {
  snapshot: null,                        // {summary, sheets, prices_updated_at, refresh_available, context}
  sheets: [],
  active: 0,
  sort: {},                              // sheetKey -> {col, dir}
  filter: "",
  rendered: 0,
  view: [],                              // filtered+sorted rows for active sheet
  abort: null,                           // AbortController of the running scan
  running: false,
  initializing: true,
  refreshing: false,
  cancelling: false,
  downloading: false,
  refreshAvailable: false,
  autoRefresh: stored("pmaf-auto-refresh", "true") !== "false",
  nextRefresh: Date.now() + REFRESH_MS,
  liveError: "",
};

/* ── snapshot storage (IndexedDB; results can exceed localStorage quotas) ── */
const DB_NAME = "pmaf", DB_STORE = "snapshots", DB_KEY = "latest";

function openDb() {
  return new Promise((resolve, reject) => {
    const req = indexedDB.open(DB_NAME, 1);
    req.onupgradeneeded = () => req.result.createObjectStore(DB_STORE);
    req.onsuccess = () => resolve(req.result);
    req.onerror = () => reject(req.error);
  });
}

async function loadSnapshot() {
  try {
    const db = await openDb();
    return await new Promise((resolve, reject) => {
      const req = db.transaction(DB_STORE).objectStore(DB_STORE).get(DB_KEY);
      req.onsuccess = () => resolve(req.result || null);
      req.onerror = () => reject(req.error);
    });
  } catch (_) { return null; }            // Private modes may block storage.
}

async function saveSnapshot(snapshot) {
  try {
    const db = await openDb();
    await new Promise((resolve, reject) => {
      const tx = db.transaction(DB_STORE, "readwrite");
      tx.objectStore(DB_STORE).put(snapshot, DB_KEY);
      tx.oncomplete = resolve;
      tx.onerror = () => reject(tx.error);
    });
  } catch (_) { /* The snapshot still works until the page closes. */ }
}

/* ── theme ─────────────────────────────────────────────────────────────── */
const savedTheme = stored("pmaf-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
else if (window.matchMedia("(prefers-color-scheme: light)").matches)
  document.documentElement.dataset.theme = "light";

$("theme-toggle").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  persist("pmaf-theme", next);
};

/* ── settings persistence ──────────────────────────────────────────────── */
const SETTING_IDS = ["min_volume", "cutoff", "similarity", "min_margin",
                     "confidence", "groq_api_key"];

(function restoreSettings() {
  let saved = {};
  try { saved = JSON.parse(stored("pmaf-settings", "{}")) || {}; } catch (_) { /* Ignore invalid preferences. */ }
  SETTING_IDS.forEach((id) => { if (saved[id] != null) $(id).value = saved[id]; });
  if (saved.use_llm != null) $("use_llm").checked = saved.use_llm;
})();

function saveSettings() {
  const out = { use_llm: $("use_llm").checked };
  SETTING_IDS.forEach((id) => (out[id] = $(id).value));
  persist("pmaf-settings", JSON.stringify(out));
}

$("settings-toggle").onclick = () => {
  $("settings").hidden = !$("settings").hidden;
  $("settings-toggle").setAttribute("aria-expanded", !$("settings").hidden);
};

/* ── log console ───────────────────────────────────────────────────────── */
function logLine({ t, m }) {
  const body = $("console-body");
  const near = body.scrollHeight - body.scrollTop - body.clientHeight < 60;
  const cls = /^✕|error|failed/i.test(m) ? "err"
            : m.includes("⚠") ? "warn"
            : /^\[\d|^Done/.test(m) ? "ok" : "";
  const el = document.createElement("div");
  el.innerHTML = `<span class="ts">${t}</span>  <span class="${cls}"></span>`;
  el.lastChild.textContent = m;
  body.appendChild(el);
  while (body.childElementCount > 500) body.removeChild(body.firstChild);
  if (near) body.scrollTop = body.scrollHeight;
}

$("console-toggle").onclick = () => {
  const body = $("console-body");
  const hidden = body.style.display === "none";
  body.style.display = hidden ? "" : "none";
  $("console-toggle").textContent = hidden ? "Hide" : "Show";
};

/* ── status ────────────────────────────────────────────────────────────── */
function setStatus(kind, text, detail) {
  const pill = $("status-pill");
  pill.className = "pill " + ({ running: "pill-run", done: "pill-done",
                                error: "pill-err", cancelled: "pill-err" }[kind] || "pill-idle");
  pill.textContent = text;
  $("status-detail").textContent = detail || "";
}

function setProgress(step, total, label, pct) {
  $("progress-track").hidden = false;
  const base = ((step - 1) / total) * 100;
  const span = (1 / total) * 100;
  $("progress-bar").style.width = `${Math.min(100, base + span * (pct ?? 0.35))}%`;
  setStatus("running", "Scanning", `Step ${step}/${total} · ${label}`);
}

/* ── live refresh / scan lifecycle ─────────────────────────────────────── */
// Gzip large JSON bodies: Vercel rejects request bodies over 4.5 MB.
async function jsonBody(value) {
  const blob = new Blob([JSON.stringify(value)], { type: "application/json" });
  const headers = { "Content-Type": "application/json" };
  if (typeof CompressionStream === "undefined") return { body: blob, headers };
  const zipped = await new Response(blob.stream().pipeThrough(new CompressionStream("gzip"))).blob();
  return { body: zipped, headers: { ...headers, "X-Content-Encoding": "gzip" } };
}

async function request(url, options = {}, timeout = 15_000, as = "json") {
  const controller = new AbortController();
  const timer = setTimeout(() => controller.abort(), timeout);
  try {
    const res = await fetch(url, { ...options, signal: controller.signal });
    if (res.ok && as === "blob") return await res.blob();
    let data;
    try { data = await res.json(); }
    catch (_) { throw new Error(`The server returned an unreadable response (${res.status}).`); }
    if (!res.ok) {
      const error = new Error(data.error || `Request failed (${res.status}).`);
      error.status = res.status;
      throw error;
    }
    return data;
  } catch (error) {
    if (error.name === "AbortError") throw new Error("Request timed out.");
    throw error;
  } finally {
    clearTimeout(timer);
  }
}

function syncControls() {
  const busy = state.initializing || state.running || state.refreshing;
  $("run-btn").disabled = busy;
  if ($("empty-run-btn")) $("empty-run-btn").disabled = busy;
  $("refresh-btn").disabled = busy || !state.refreshAvailable;
  $("refresh-btn").textContent = state.refreshing ? "Refreshing…" : "Refresh now";
  $("cancel-btn").hidden = !state.running;
  $("cancel-btn").disabled = state.cancelling;
  $("cancel-btn").textContent = state.cancelling ? "Stopping…" : "Stop";
  $("download-btn").disabled = !state.snapshot || state.downloading;
  $("auto-refresh").checked = state.autoRefresh;

  let mode = "idle";
  let message = "Run a full scan to discover matching markets and enable live prices.";
  let countdown = "Awaiting scan";
  if (state.initializing) {
    mode = "busy";
    message = "Loading your last saved snapshot.";
    countdown = "Loading…";
  } else if (state.running) {
    mode = "busy";
    message = state.snapshot
      ? "Discovering markets. The previous snapshot stays visible; live refresh resumes after the scan. Keep this tab open."
      : "Discovering markets and matching questions. Live prices start when this scan finishes. Keep this tab open.";
    countdown = "Scan in progress";
  } else if (state.refreshing) {
    mode = "busy";
    message = "Fetching matched orderbooks and recalculating arbitrage. Displaying the last successful snapshot.";
    countdown = "Refreshing…";
  } else if (state.refreshAvailable) {
    mode = state.autoRefresh ? "live" : "idle";
    message = state.autoRefresh
      ? "Auto refresh is on. Matched orderbooks refresh while this page is open."
      : "Auto refresh is paused. Prices are a saved snapshot; use Refresh now or turn auto refresh back on.";
    countdown = state.autoRefresh
      ? `Next in ${Math.max(0, Math.ceil((state.nextRefresh - Date.now()) / 1000))}s`
      : "Paused";
  } else if (state.snapshot) {
    message = "This snapshot has no refreshable matches. Run a new full scan to enable live refresh.";
    countdown = "New scan needed";
  }
  if (state.liveError) {
    mode = "stale";
    message = `${state.snapshot ? "Showing the last snapshot; prices may be stale. " : ""}${state.liveError}`;
    if (state.autoRefresh && state.refreshAvailable && !busy)
      message += " Automatic refresh will retry.";
  }
  $("live-indicator").dataset.mode = mode;
  $("live-message").dataset.mode = mode;
  if ($("live-message").textContent !== message) $("live-message").textContent = message;
  $("refresh-countdown").textContent = countdown;
}

function showError(message) {
  state.liveError = message;
  setStatus("error", state.snapshot ? "Stale snapshot" : "Error", message);
  syncControls();
}

function setTimestamp(id, value) {
  const el = $(id);
  const date = value ? new Date(value) : null;
  if (date && !Number.isNaN(date.getTime())) {
    el.dateTime = date.toISOString();
    el.textContent = date.toLocaleString();
    el.title = date.toISOString();
  } else {
    el.removeAttribute("datetime");
    el.textContent = state.snapshot ? "Unavailable" : "Not yet";
    el.removeAttribute("title");
  }
}

$("auto-refresh").onchange = () => {
  state.autoRefresh = $("auto-refresh").checked;
  persist("pmaf-auto-refresh", String(state.autoRefresh));
  state.nextRefresh = Date.now() + REFRESH_MS;
  syncControls();
};
$("refresh-btn").onclick = refreshPrices;

async function refreshPrices() {
  if (state.initializing || state.running || state.refreshing || !state.refreshAvailable) return;
  state.refreshing = true; // Guard before the first await, including manual clicks.
  state.nextRefresh = Date.now() + REFRESH_MS;
  syncControls();
  const base = state.snapshot;
  try {
    const data = await request("/api/refresh",
      { method: "POST", ...(await jsonBody({ context: base.context })) }, 120_000);
    if (data.status !== "done" || !data.sheets || !data.summary)
      throw new Error("The price refresh returned no usable results.");
    if (state.snapshot !== base) return;   // A new scan replaced this snapshot.
    // Refresh returns only repriced sheets; raw market tabs stay as discovered.
    const repriced = new Map(data.sheets.map((sheet) => [sheet.key, sheet]));
    const snapshot = { ...data, sheets: base.sheets.map((sheet) => repriced.get(sheet.key) || sheet) };
    applyResults(snapshot, true);
    saveSnapshot(snapshot);
  } catch (error) {
    showError(error.message || "Could not refresh prices.");
  } finally {
    state.refreshing = false;
    syncControls();
  }
}

setInterval(() => {
  if (Date.now() >= state.nextRefresh) {
    state.nextRefresh = Date.now() + REFRESH_MS;
    if (state.autoRefresh) refreshPrices(); // Busy ticks are skipped, never queued.
  }
  syncControls();
}, 1000);

$("run-btn").onclick = startScan;
$("empty-run-btn").onclick = startScan;
$("cancel-btn").onclick = () => {
  if (!state.running || state.cancelling) return;
  state.cancelling = true;
  setStatus("running", "Stopping", "Closing the scan connection.");
  syncControls();
  state.abort?.abort();                    // Disconnecting stops the scan on the server.
};

window.addEventListener("beforeunload", (event) => {
  if (state.running) { event.preventDefault(); event.returnValue = ""; }
});

const IDLE_CTA = $("empty-cta").innerHTML;

function markRunningCopy(running) {
  const cta = $("empty-cta");
  if (cta) cta.innerHTML = running
    ? "Scanning now. Results arrive when discovery finishes — follow the scan log below. Keep this tab open."
    : IDLE_CTA;
}

async function startScan() {
  if (state.initializing || state.running || state.refreshing) return;
  state.running = true;
  state.cancelling = false;
  state.liveError = "";
  state.abort = new AbortController();
  saveSettings();
  markRunningCopy(true);
  $("console").hidden = false;
  $("console-body").textContent = "";
  setStatus("running", "Starting", "Discovering new markets may take a few minutes.");
  setProgress(1, 5, "Discovering markets");
  syncControls();

  let outcome = { status: "error", error: "The scan connection closed before the scan finished." };
  try {
    const res = await fetch("/api/scan", {
      method: "POST",
      signal: state.abort.signal,
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({
        min_volume: $("min_volume").value,
        cutoff: $("cutoff").value,
        similarity: $("similarity").value,
        min_margin: (parseFloat($("min_margin").value) || 0) / 100,
        confidence: $("confidence").value,
        use_llm: $("use_llm").checked,
        groq_api_key: $("groq_api_key").value,
      }),
    });
    if (!res.ok || !res.body) {
      let message = `The server refused the scan (${res.status}).`;
      try { message = (await res.json()).error || message; } catch (_) { /* Keep the status text. */ }
      throw new Error(message);
    }
    outcome = await readScanStream(res.body, outcome);
  } catch (error) {
    outcome = state.cancelling
      ? { status: "cancelled" }
      : { status: "error", error: error.message || outcome.error };
  }
  finishScan(outcome);
}

// Parse the text/event-stream body: blocks of "event:" / "data:" lines.
async function readScanStream(body, fallback) {
  const reader = body.pipeThrough(new TextDecoderStream()).getReader();
  let buffer = "", outcome = fallback, done = false, hosted = false;
  while (!done) {
    const chunk = await reader.read();
    if (chunk.done) break;
    buffer += chunk.value;
    let cut;
    while ((cut = buffer.indexOf("\n\n")) >= 0) {
      const block = buffer.slice(0, cut);
      buffer = buffer.slice(cut + 2);
      let event = "message", data = "";
      for (const line of block.split("\n")) {
        if (line.startsWith("event:")) event = line.slice(6).trim();
        else if (line.startsWith("data:")) data += line.slice(5).trim();
      }
      if (!data) continue;               // Heartbeat comments carry no data.
      const payload = JSON.parse(data);
      if (event === "start") {
        hosted = payload.time_limit > 0;
        if (!payload.llm_enabled && $("use_llm").checked)
          logLine({ t: "--:--:--", m: "No Groq API key — running without LLM verification." });
      } else if (event === "log") logLine(payload);
      else if (event === "progress") setProgress(payload.step, payload.total, payload.label, payload.pct);
      else if (event === "done") outcome = { status: "done", data: payload };
      else if (event === "scan_error") outcome = { status: "error", error: payload.error };
      else if (event === "cancelled") outcome = { status: "cancelled" };
      else if (event === "end") done = true;
    }
  }
  if (!done && outcome === fallback && hosted)
    outcome = { status: "error", error: `${fallback.error} The host may have stopped it at its time limit; try a sooner 'Resolves before' date or a higher minimum volume.` };
  reader.cancel().catch(() => {});
  return outcome;
}

function finishScan({ status, data, error }) {
  try {
    if (status === "done") {
      if (!data?.sheets || !data?.summary) throw new Error("Completed scan results are unavailable.");
      applyResults(data);
      saveSnapshot(data);
    } else {
      state.liveError = status === "cancelled"
        ? "Discovery was stopped. No new discovery results were applied."
        : (error || "Discovery failed. No new discovery results were applied.");
      setStatus(status === "cancelled" ? "cancelled" : "error",
        status === "cancelled" ? "Stopped" : "Scan failed", state.liveError);
    }
  } catch (err) {
    showError(`Could not load completed results: ${err.message}`);
  } finally {
    state.running = false;
    state.cancelling = false;
    state.abort = null;
    state.nextRefresh = Date.now() + REFRESH_MS;
    markRunningCopy(false);
    $("progress-track").hidden = true;
    $("progress-bar").style.width = "0%";
    syncControls();
  }
}

/* ── results ───────────────────────────────────────────────────────────── */
function applyResults(data, preserve = false) {
  const wrap = $("grid-wrap");
  const context = {
    key: state.sheets[state.active]?.key,
    top: wrap.scrollTop, left: wrap.scrollLeft, rows: state.rendered,
    tabScroll: $("tabs").scrollLeft,
    focusedTab: Array.from($("tabs").children).indexOf(document.activeElement),
    focusedColumn: Array.from($("grid-head").children).indexOf(document.activeElement),
  };
  state.snapshot = data;
  state.sheets = data.sheets;
  state.refreshAvailable = data.refresh_available === true && !!data.context;
  state.liveError = "";
  if (preserve) {
    state.active = Math.max(0, state.sheets.findIndex((sheet) => sheet.key === context.key));
  } else {
    state.sort = {};
    state.active = 0;
    state.filter = "";
    $("filter").value = "";
  }
  renderSummary(data.summary);
  renderTabs();
  renderSheet(preserve ? context.rows : CHUNK);
  if (preserve) {
    wrap.scrollTop = context.top;
    wrap.scrollLeft = context.left;
    $("tabs").scrollLeft = context.tabScroll;
    if (context.focusedTab >= 0) $("tabs").children[context.focusedTab]?.focus({ preventScroll: true });
    if (context.focusedColumn >= 0) $("grid-head").children[context.focusedColumn]?.focus({ preventScroll: true });
  } else {
    wrap.scrollTop = 0;
    wrap.scrollLeft = 0;
    state.nextRefresh = Date.now() + REFRESH_MS;
  }
  setTimestamp("prices-updated", data.prices_updated_at);
  setTimestamp("discovery-updated", data.summary.finished_at);
  const s = data.summary;
  setStatus("done", s.arb_count ? `${s.arb_count} arb${s.arb_count > 1 ? "s" : ""}` : "No edge",
    `${s.poly_count.toLocaleString()} Polymarket · ${s.kalshi_count.toLocaleString()} Kalshi`);
  syncControls();
}

$("download-btn").onclick = async () => {
  const snapshot = state.snapshot;
  if (!snapshot || state.downloading) return;
  state.downloading = true;
  syncControls();
  try {
    const stamp = (snapshot.summary?.finished_at || "").slice(0, 10) || "results";
    const blob = await request("/api/workbook",
      { method: "POST", ...(await jsonBody({ sheets: snapshot.sheets, stamp })) }, 120_000, "blob");
    const url = URL.createObjectURL(blob);
    const link = document.createElement("a");
    link.href = url;
    link.download = `market-arbitrage-${stamp}.xlsx`;
    document.body.appendChild(link);
    link.click();
    link.remove();
    setTimeout(() => URL.revokeObjectURL(url), 10_000);
  } catch (error) {
    showError(`Could not build the Excel file: ${error.message}`);
  } finally {
    state.downloading = false;
    syncControls();
  }
};

function stat(label, value, sub, cls = "") {
  return `<div class="stat ${cls}">
      <div class="stat-label">${label}</div>
      <div class="stat-value">${value}</div>
      <div class="stat-sub">${sub || ""}</div>
    </div>`;
}

function renderSummary(s) {
  $("summary").hidden = false;
  $("summary").innerHTML = [
    stat("Best edge", s.arb_count ? s.best_margin_display : "—",
         s.arb_count ? "per $1.00 pair" : "no qualifying edge",
         s.arb_count ? "stat-hero" : "stat-zero"),
    stat("Arbitrage", s.arb_count.toLocaleString(), `from ${s.arb_source}`,
         s.arb_count ? "" : "stat-zero"),
    stat("LLM verified", s.verified_count.toLocaleString(),
         s.llm_used ? "same-outcome pairs" : "LLM off"),
    stat("Matches", s.match_count.toLocaleString(), "title pairs"),
    stat("Polymarket", s.poly_count.toLocaleString(), "eligible markets"),
    stat("Kalshi", s.kalshi_count.toLocaleString(), "eligible markets"),
    stat("Discovery time", `${s.duration}s`, `vol ≥ $${Number(s.min_volume).toLocaleString()}`),
  ].join("");
}

function renderTabs() {
  const tabs = $("tabs");
  tabs.hidden = false;
  $("sheet-toolbar").hidden = false;
  tabs.innerHTML = "";
  state.sheets.forEach((sheet, i) => {
    const b = document.createElement("button");
    b.className = "tab" + (sheet.key === "arbitrage" && sheet.rows.length ? " tab-hot" : "");
    b.role = "tab";
    b.setAttribute("aria-selected", i === state.active);
    b.innerHTML = `${sheet.name}<span class="tab-count">${sheet.rows.length.toLocaleString()}</span>`;
    b.setAttribute("aria-controls", "grid-wrap");
    b.onclick = () => {
      state.active = i;
      clearTimeout(filterTimer);
      $("filter").value = "";
      state.filter = "";
      renderTabs();
      renderSheet();
      $("grid-wrap").scrollTop = 0;
      $("tabs").children[i].focus({ preventScroll: true });
    };
    b.onkeydown = (event) => {
      let target;
      if (event.key === "ArrowRight") target = (i + 1) % state.sheets.length;
      if (event.key === "ArrowLeft") target = (i + state.sheets.length - 1) % state.sheets.length;
      if (event.key === "Home") target = 0;
      if (event.key === "End") target = state.sheets.length - 1;
      if (target !== undefined) {
        event.preventDefault();
        $("tabs").children[target].click();
      }
    };
    tabs.appendChild(b);
  });
}

const raw = (c) => (c && typeof c === "object" ? c.v : c);
const disp = (c) => {
  if (c === null || c === undefined) return "—";
  if (typeof c === "object") return c.d ?? "—";
  return String(c);
};

function renderSheet(minRows = CHUNK) {
  const sheet = state.sheets[state.active];
  if (!sheet) return;
  $("empty-state").hidden = true;
  $("grid").hidden = false;
  $("sheet-hint").textContent = sheet.hint || "";

  // header
  const head = $("grid-head");
  head.innerHTML = "";
  const sort = state.sort[sheet.key];
  sheet.columns.forEach((col, ci) => {
    const th = document.createElement("th");
    th.className = col.align === "right" ? "num" : col.align === "center" ? "center" : "";
    th.style.minWidth = col.width + "px";
    th.textContent = col.label;
    th.tabIndex = 0;
    th.scope = "col";
    th.setAttribute("aria-sort", sort?.col === ci ? (sort.dir > 0 ? "ascending" : "descending") : "none");
    th.setAttribute("aria-label", `${col.label}: activate to sort`);
    if (sort && sort.col === ci) {
      const a = document.createElement("span");
      a.className = "arrow";
      a.textContent = sort.dir > 0 ? "▲" : "▼";
      th.appendChild(a);
    }
    th.onclick = () => {
      const cur = state.sort[sheet.key];
      state.sort[sheet.key] = cur && cur.col === ci ? { col: ci, dir: -cur.dir }
                                                    : { col: ci, dir: -1 };
      renderSheet();
    };
    th.onkeydown = (event) => {
      if (event.key === "Enter" || event.key === " ") {
        event.preventDefault();
        th.click();
        $("grid-head").children[ci]?.focus({ preventScroll: true });
      }
    };
    head.appendChild(th);
  });

  // filter + sort
  let rows = sheet.rows;
  if (state.filter) {
    const q = state.filter.toLowerCase();
    rows = rows.filter((r) => r.some((c) => disp(c).toLowerCase().includes(q)));
  }
  if (sort) {
    const { col, dir } = sort;
    rows = rows.slice().sort((a, b) => {
      const x = raw(a[col]), y = raw(b[col]);
      const xn = x === null || x === undefined, yn = y === null || y === undefined;
      if (xn && yn) return 0;
      if (xn) return 1;
      if (yn) return -1;
      if (typeof x === "number" && typeof y === "number") return (x - y) * dir;
      return String(x).localeCompare(String(y)) * dir;
    });
  }

  state.view = rows;
  state.rendered = 0;
  $("grid-body").innerHTML = "";
  $("row-count").textContent =
    `${rows.length.toLocaleString()} of ${sheet.rows.length.toLocaleString()} rows`;

  if (!rows.length) {
    $("grid").hidden = true;
    $("empty-state").hidden = false;
    $("empty-state").innerHTML =
      `<h2>${state.filter ? "No matching rows" : "Nothing here"}</h2><p>${
        state.filter ? "Try a different filter." : (sheet.empty || "")}</p>`;
    return;
  }
  do { appendRows(); } while (state.rendered < Math.min(minRows, state.view.length));
}

function appendRows() {
  const sheet = state.sheets[state.active];
  const body = $("grid-body");
  const frag = document.createDocumentFragment();
  const end = Math.min(state.rendered + CHUNK, state.view.length);

  for (let i = state.rendered; i < end; i++) {
    const tr = document.createElement("tr");
    state.view[i].forEach((cell, ci) => {
      const col = sheet.columns[ci];
      const td = document.createElement("td");
      td.className = [
        col.align === "right" ? "num" : col.align === "center" ? "center" : "text",
        col.kind === "num" ? "mono" : "",
        "col-" + col.key,
      ].join(" ").trim();

      const text = disp(cell);
      if (col.kind === "side") {
        const span = document.createElement("span");
        span.className = "side " + (text === "YES" ? "side-yes" : "side-no");
        span.textContent = text;
        td.appendChild(span);
      } else if (cell && typeof cell === "object" && cell.u) {
        const a = document.createElement("a");
        a.href = cell.u; a.target = "_blank"; a.rel = "noopener";
        a.textContent = text;
        a.title = text;
        td.appendChild(a);
      } else if (text === "—") {
        td.innerHTML = '<span class="dash">—</span>';
      } else {
        td.textContent = text;
      }
      tr.appendChild(td);
    });
    frag.appendChild(tr);
  }
  body.appendChild(frag);
  state.rendered = end;
}

new IntersectionObserver((entries) => {
  if (entries[0].isIntersecting && state.rendered < state.view.length) appendRows();
}, { root: $("grid-wrap"), rootMargin: "400px" }).observe($("sentinel"));

/* ── filter + copy ─────────────────────────────────────────────────────── */
let filterTimer;
$("filter").oninput = (e) => {
  clearTimeout(filterTimer);
  filterTimer = setTimeout(() => { state.filter = e.target.value.trim(); renderSheet(); }, 140);
};

$("copy-btn").onclick = async () => {
  const sheet = state.sheets[state.active];
  if (!sheet) return;
  const esc = (v) => (/[",\n]/.test(v) ? `"${v.replace(/"/g, '""')}"` : v);
  const csv = [sheet.columns.map((c) => esc(c.label)).join(",")]
    .concat(state.view.map((r) => r.map((c) => esc(disp(c))).join(",")))
    .join("\n");
  try {
    await navigator.clipboard.writeText(csv);
    $("copy-btn").textContent = "Copied";
  } catch (_) {
    $("copy-btn").textContent = "Copy failed";
  }
  setTimeout(() => ($("copy-btn").textContent = "Copy CSV"), 1400);
};

/* Restore the snapshot this browser saved last. */
(async function boot() {
  syncControls();
  const snapshot = await loadSnapshot();
  if (snapshot?.sheets && snapshot?.summary) applyResults(snapshot);
  else setStatus("idle", "Ready", "Run scan to discover your first matches.");
  state.initializing = false;
  syncControls();
})();

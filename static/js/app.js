/* Prediction-market arbitrage finder — front end.
   Starts a scan, streams its log over SSE, renders the finished sheets. */

const $ = (id) => document.getElementById(id);
const CHUNK = 120;                       // rows appended per render pass

const state = {
  jobId: null,
  sheets: [],
  active: 0,
  sort: {},                              // sheetKey -> {col, dir}
  filter: "",
  rendered: 0,
  view: [],                              // filtered+sorted rows for active sheet
  es: null,
  running: false,
};

/* ── theme ─────────────────────────────────────────────────────────────── */
const savedTheme = localStorage.getItem("pmaf-theme");
if (savedTheme) document.documentElement.dataset.theme = savedTheme;
else if (window.matchMedia("(prefers-color-scheme: light)").matches)
  document.documentElement.dataset.theme = "light";

$("theme-toggle").onclick = () => {
  const next = document.documentElement.dataset.theme === "dark" ? "light" : "dark";
  document.documentElement.dataset.theme = next;
  localStorage.setItem("pmaf-theme", next);
};

/* ── settings persistence ──────────────────────────────────────────────── */
const SETTING_IDS = ["min_volume", "cutoff", "similarity", "min_margin",
                     "confidence", "groq_api_key"];

(function restoreSettings() {
  const saved = JSON.parse(localStorage.getItem("pmaf-settings") || "{}");
  SETTING_IDS.forEach((id) => { if (saved[id] != null) $(id).value = saved[id]; });
  if (saved.use_llm != null) $("use_llm").checked = saved.use_llm;
})();

function saveSettings() {
  const out = { use_llm: $("use_llm").checked };
  SETTING_IDS.forEach((id) => (out[id] = $(id).value));
  localStorage.setItem("pmaf-settings", JSON.stringify(out));
}

$("settings-toggle").onclick = () => {
  $("settings").hidden = !$("settings").hidden;
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

/* ── run / stop ────────────────────────────────────────────────────────── */
$("run-btn").onclick = startScan;
$("cancel-btn").onclick = async () => {
  if (state.jobId) await fetch(`/api/scan/${state.jobId}/cancel`, { method: "POST" });
};

async function startScan() {
  saveSettings();
  const body = {
    min_volume: $("min_volume").value,
    cutoff: $("cutoff").value,
    similarity: $("similarity").value,
    min_margin: (parseFloat($("min_margin").value) || 0) / 100,
    confidence: $("confidence").value,
    use_llm: $("use_llm").checked,
    groq_api_key: $("groq_api_key").value,
  };

  $("run-btn").disabled = true;
  $("console").hidden = false;
  $("console-body").textContent = "";
  setStatus("running", "Starting", "");

  let res;
  try {
    res = await fetch("/api/scan", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
  } catch (e) {
    $("run-btn").disabled = false;
    setStatus("error", "Error", "Could not reach the server.");
    return;
  }
  const data = await res.json();
  if (!res.ok) {
    $("run-btn").disabled = false;
    setStatus("error", "Error", data.error || "Failed to start.");
    return;
  }

  state.jobId = data.id;
  state.running = true;
  markRunningCopy(true);
  $("cancel-btn").hidden = false;
  $("download-btn").disabled = true;
  if (!data.llm_enabled && $("use_llm").checked)
    logLine({ t: "--:--:--", m: "⚠ No Groq API key — running without LLM verification." });

  listen(data.id);
}

const IDLE_CTA = $("empty-cta") && $("empty-cta").innerHTML;

function markRunningCopy(running) {
  const cta = $("empty-cta");
  if (!cta) return;
  cta.innerHTML = running
    ? "Scanning now. Results land here the moment the pass finishes — the log below shows where it is."
    : IDLE_CTA;
}

function listen(jobId) {
  if (state.es) state.es.close();
  const es = new EventSource(`/api/scan/${jobId}/stream`);
  state.es = es;

  es.addEventListener("log", (e) => logLine(JSON.parse(e.data)));
  es.addEventListener("progress", (e) => {
    const d = JSON.parse(e.data);
    setProgress(d.step, d.total, d.label, d.pct);
  });
  es.addEventListener("scan_error", (e) => {
    setStatus("error", "Failed", JSON.parse(e.data).error);
  });
  es.addEventListener("cancelled", () => setStatus("cancelled", "Stopped", ""));
  es.addEventListener("end", async (e) => {
    es.close();
    state.es = null;
    state.running = false;
    markRunningCopy(false);
    $("run-btn").disabled = false;
    $("cancel-btn").hidden = true;
    $("progress-bar").style.width = "100%";
    setTimeout(() => { $("progress-track").hidden = true;
                       $("progress-bar").style.width = "0%"; }, 700);
    const status = JSON.parse(e.data).status;
    if (status === "done") await loadResults(jobId);
  });
}

/* ── results ───────────────────────────────────────────────────────────── */
async function loadResults(jobId) {
  const res = await fetch(`/api/scan/${jobId}/results`);
  const data = await res.json();
  if (!data.sheets) return;
  state.jobId = jobId;
  applyResults(data);
}

function applyResults(data) {
  state.sheets = data.sheets;
  state.sort = {};
  state.active = 0;
  $("download-btn").disabled = false;
  renderSummary(data.summary);
  renderTabs();
  renderSheet();
  const s = data.summary;
  setStatus("done", s.arb_count ? `${s.arb_count} arb${s.arb_count > 1 ? "s" : ""}` : "Clean",
    `${s.poly_count.toLocaleString()} Polymarket · ${s.kalshi_count.toLocaleString()} Kalshi · ${s.duration}s`);
}

$("download-btn").onclick = () => {
  if (state.jobId) window.location = `/api/scan/${state.jobId}/download`;
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
         s.arb_count ? "per $1.00 pair" : "nothing above $1.00",
         s.arb_count ? "stat-hero" : "stat-zero"),
    stat("Arbitrage", s.arb_count.toLocaleString(), `from ${s.arb_source}`,
         s.arb_count ? "" : "stat-zero"),
    stat("LLM verified", s.verified_count.toLocaleString(),
         s.llm_used ? "same-outcome pairs" : "LLM off"),
    stat("Matches", s.match_count.toLocaleString(), "title pairs"),
    stat("Polymarket", s.poly_count.toLocaleString(), "eligible markets"),
    stat("Kalshi", s.kalshi_count.toLocaleString(), "eligible markets"),
    stat("Scan time", `${s.duration}s`, `vol ≥ $${Number(s.min_volume).toLocaleString()}`),
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
    b.onclick = () => { state.active = i; $("filter").value = ""; state.filter = "";
                        renderTabs(); renderSheet(); };
    tabs.appendChild(b);
  });
}

const raw = (c) => (c && typeof c === "object" ? c.v : c);
const disp = (c) => {
  if (c === null || c === undefined) return "—";
  if (typeof c === "object") return c.d ?? "—";
  return String(c);
};

function renderSheet() {
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
  appendRows();
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

/* ── boot: pick up the last completed scan, or a scan already running ───── */
(async function boot() {
  try {
    const res = await fetch("/api/latest");
    const data = await res.json();
    if (data.status === "done") {
      state.jobId = data.id;
      applyResults(data);
      setStatus("done", "Last scan", new Date(data.summary.finished_at).toLocaleString());
    }
  } catch (_) { /* first load, nothing cached */ }

  // A scan outlives a page reload — pick it back up instead of stranding it.
  try {
    const res = await fetch("/api/current");
    const cur = await res.json();
    if (cur.id) {
      state.jobId = cur.id;
      state.running = true;
      markRunningCopy(true);
      $("run-btn").disabled = true;
      $("cancel-btn").hidden = false;
      $("console").hidden = false;
      setProgress(cur.step || 1, cur.total || 5, cur.label || "Reconnecting…", cur.pct);
      listen(cur.id);
    }
  } catch (_) { /* nothing running */ }
})();

function $(sel) {
  return document.querySelector(sel);
}

const WORKFLOW_TABS = new Set(["pipeline", "generate", "score", "judge", "analyze"]);

function tabSetup() {
  const tabs = document.querySelectorAll(".tab");
  const panels = document.querySelectorAll(".panel");
  tabs.forEach((btn) => {
    btn.addEventListener("click", () => {
      const id = btn.getAttribute("data-tab");
      tabs.forEach((b) => b.classList.toggle("active", b === btn));
      panels.forEach((p) => p.classList.toggle("active", p.id === `tab-${id}`));
      if (id === "run-extract") pilotSyncSliders();
      if (WORKFLOW_TABS.has(id)) runnerRefreshLists();
      if (id === "results") {
        loadResponseFilters();
        loadRankings();
      }
    });
  });
}

function parseDimensionFromRangeId(rangeId) {
  const m = String(rangeId || "").match(/^(valence|arousal|dominance)_/);
  return m ? m[1] : null;
}

function fillDatalist(datalistEl, values) {
  datalistEl.innerHTML = "";
  values.forEach((v) => {
    const o = document.createElement("option");
    o.value = v;
    datalistEl.appendChild(o);
  });
}

async function loadResponseFilters() {
  const dlR = $("#resp-range-list");
  const dlRun = $("#resp-run-list");
  if (!dlR) return;
  try {
    const res = await fetch("/api/response-filters");
    if (!res.ok) throw new Error(await res.text());
    const data = await res.json();
    fillDatalist(dlR, data.range_ids || []);
    if (dlRun) fillDatalist(dlRun, data.run_ids || []);
  } catch (e) {
    console.warn("response-filters", e);
  }
}

async function loadVectors() {
  const ul = $("#vector-list");
  ul.innerHTML = "";
  try {
    const res = await fetch("/api/vectors");
    const items = await res.json();
    if (!items.length) {
      ul.innerHTML = "<li class='muted'>No .pt files in sweep/results/vectors yet.</li>";
      return;
    }
    items.forEach((v) => {
      const li = document.createElement("li");
      li.textContent = v.name;
      ul.appendChild(li);
    });
  } catch (e) {
    ul.innerHTML = `<li class='error'>${e}</li>`;
  }
}

let rankRows = [];
let rankByScenario = [];
let sortKey = "composite";
let sortDir = -1;

const RESULTS_COLLAPSE_LIMIT = 15;
let rankTableExpanded = false;
let respPlotsExpanded = false;

/** Match sweep/grids.py FULL32 preset — fixed matrix axes. */
const MATRIX_NUM_LAYERS = 32;
const MATRIX_LAST_LAYER = MATRIX_NUM_LAYERS - 1;
const FULL32_STARTS = [4, 6, 8, 10, 12, 14, 16, 18, 20, 22, 24];
const FULL32_WIDTHS = [1, 2, 4, 6, 8, 12];

function parseRangeId(rangeId) {
  const m = String(rangeId).match(/^(valence|arousal|dominance)_(\d+)_(\d+)$/);
  if (!m) return null;
  const start = parseInt(m[2], 10);
  const end = parseInt(m[3], 10);
  const width = end - start + 1;
  if (width < 1 || !Number.isFinite(start) || !Number.isFinite(end)) return null;
  return { dim: m[1], start, end, width };
}

function matrixCellValid(start, width) {
  return start + width - 1 <= MATRIX_LAST_LAYER;
}

function olsAvgByRangeRun(rows) {
  const acc = new Map();
  for (const r of rows) {
    const v = Number(r.ols_slope);
    if (!Number.isFinite(v)) continue;
    const k = `${r.run_id || ""}|${r.range_id}`;
    if (!acc.has(k)) acc.set(k, { sum: 0, n: 0 });
    const o = acc.get(k);
    o.sum += v;
    o.n += 1;
  }
  const out = new Map();
  for (const [k, { sum, n }] of acc) {
    if (n > 0) out.set(k, sum / n);
  }
  return out;
}

function collectCellMetrics(dim, meanRows, byScenario, metric) {
  const olsMap = metric === "ols_slope" ? olsAvgByRangeRun(byScenario) : null;
  const buckets = new Map();
  for (const row of meanRows) {
    const p = parseRangeId(row.range_id);
    if (!p || p.dim !== dim) continue;
    if (!matrixCellValid(p.start, p.width)) continue;
    const ck = `${p.start}|${p.width}`;
    let val = null;
    if (metric === "ols_slope") {
      const k = `${row.run_id || ""}|${row.range_id}`;
      val = olsMap != null ? olsMap.get(k) : null;
    } else {
      val = Number(row[metric]);
    }
    if (!Number.isFinite(val)) continue;
    if (!buckets.has(ck)) buckets.set(ck, { sum: 0, n: 0 });
    const b = buckets.get(ck);
    b.sum += val;
    b.n += 1;
  }
  const values = new Map();
  for (const [k, { sum, n }] of buckets) {
    if (n > 0) values.set(k, sum / n);
  }
  return values;
}

function matrixNonZeroRange(nums) {
  const nz = nums.filter((v) => Number.isFinite(v) && Math.abs(v) > 1e-12);
  if (nz.length === 0) return { vmin: null, vmax: null, hasScale: false };
  let vmin = Math.min(...nz);
  let vmax = Math.max(...nz);
  if (vmin === vmax) {
    const pad = Math.max(1e-6, Math.abs(vmin) * 1e-4);
    vmin -= pad;
    vmax += pad;
  }
  return { vmin, vmax, hasScale: true };
}

function matrixNormT(val, vmin, vmax, isZero) {
  if (!Number.isFinite(val)) return null;
  if (isZero) return 0;
  if (!Number.isFinite(vmin) || !Number.isFinite(vmax) || vmax === vmin) return 0.5;
  const t = (val - vmin) / (vmax - vmin);
  return Math.max(0, Math.min(1, t));
}

function matrixCellColor(metric, t) {
  if (t == null || !Number.isFinite(t)) return "";
  if (metric === "danger_score") {
    const L = 12 + t * 48;
    return `hsl(2, 82%, ${Math.round(L)}%)`;
  }
  const L = 10 + t * 52;
  return `hsl(206, 58%, ${Math.round(L)}%)`;
}

function matrixCellTextColor(_metric, t) {
  if (t == null || !Number.isFinite(t)) return "";
  return "#f2f2f2";
}

function hslToRgb(h, s, l) {
  s /= 100;
  l /= 100;
  const c = (1 - Math.abs(2 * l - 1)) * s;
  const hp = h / 60;
  const x = c * (1 - Math.abs((hp % 2) - 1));
  let r1 = 0;
  let g1 = 0;
  let b1 = 0;
  if (hp >= 0 && hp < 1) [r1, g1, b1] = [c, x, 0];
  else if (hp < 2) [r1, g1, b1] = [x, c, 0];
  else if (hp < 3) [r1, g1, b1] = [0, c, x];
  else if (hp < 4) [r1, g1, b1] = [0, x, c];
  else if (hp < 5) [r1, g1, b1] = [x, 0, c];
  else [r1, g1, b1] = [c, 0, x];
  const m = l - c / 2;
  return [
    Math.round((r1 + m) * 255),
    Math.round((g1 + m) * 255),
    Math.round((b1 + m) * 255),
  ];
}

function matrixCellFillRgb(metric, t) {
  if (t == null || !Number.isFinite(t)) return [255, 255, 255];
  if (metric === "danger_score") {
    const L = 12 + t * 48;
    return hslToRgb(2, 82, L);
  }
  const L = 10 + t * 52;
  return hslToRgb(206, 58, L);
}

function rankMatrixLegendText(metric, cellVals) {
  const nums = Array.from(cellVals.values()).filter((v) => Number.isFinite(v));
  const { vmin, vmax, hasScale } = matrixNonZeroRange(nums);
  if (!nums.length) return "No numeric values for this metric in loaded mean rows.";
  if (hasScale && vmin != null && vmax != null) {
    return `${metric}: min ${vmin.toFixed(4)} to max ${vmax.toFixed(4)}`;
  }
  const dataMin = Math.min(...nums);
  const dataMax = Math.max(...nums);
  return `${metric}: min ${dataMin.toFixed(4)} to max ${dataMax.toFixed(4)}`;
}

function renderRankMatrix() {
  const wrap = $("#rank-matrix-wrap");
  const leg = $("#rank-matrix-legend");
  const metricSel = $("#rank-matrix-metric");
  if (!wrap) return;
  const dim = $("#dim-select").value;
  const metric = metricSel ? metricSel.value : "composite";
  wrap.innerHTML = "";
  if (!rankRows.length) {
    if (leg) leg.textContent = "Load rankings to show the matrix.";
    return;
  }
  const cellVals = collectCellMetrics(dim, rankRows, rankByScenario, metric);
  const nums = Array.from(cellVals.values()).filter((v) => Number.isFinite(v));
  const { vmin, vmax, hasScale } = matrixNonZeroRange(nums);
  if (leg) leg.textContent = rankMatrixLegendText(metric, cellVals);

  const nX = FULL32_STARTS.length;
  const nY = FULL32_WIDTHS.length;
  const table = document.createElement("table");
  table.className = "rank-matrix-table";

  const thead = document.createElement("thead");
  const trTop = document.createElement("tr");
  const corner = document.createElement("th");
  corner.className = "rank-matrix-corner";
  corner.scope = "col";
  trTop.appendChild(corner);
  FULL32_STARTS.forEach((s) => {
    const th = document.createElement("th");
    th.scope = "col";
    th.textContent = String(s);
    th.title = `Layer start ${s}`;
    trTop.appendChild(th);
  });
  thead.appendChild(trTop);
  table.appendChild(thead);

  const tbody = document.createElement("tbody");
  FULL32_WIDTHS.forEach((w) => {
    const tr = document.createElement("tr");
    const rowHeader = document.createElement("th");
    rowHeader.scope = "row";
    rowHeader.textContent = String(w);
    rowHeader.title = `Width ${w} (layers)`;
    tr.appendChild(rowHeader);

    FULL32_STARTS.forEach((s) => {
      const td = document.createElement("td");
      td.className = "rank-matrix-cell";
      const ok = matrixCellValid(s, w);
      if (!ok) {
        td.classList.add("rank-matrix-invalid");
        td.textContent = "—";
        td.title = `Invalid span (end > ${MATRIX_LAST_LAYER})`;
        tr.appendChild(td);
        return;
      }
      const key = `${s}|${w}`;
      const val = cellVals.get(key);
      if (!Number.isFinite(val)) {
        td.classList.add("rank-matrix-nodata");
        td.textContent = "·";
        td.title = `${dim}_${s}_${s + w - 1} — no data`;
        tr.appendChild(td);
        return;
      }
      const isZero = Math.abs(val) < 1e-12;
      const t = hasScale ? matrixNormT(val, vmin, vmax, isZero) : (isZero ? 0 : 0.5);
      td.style.background = matrixCellColor(metric, t);
      td.style.color = matrixCellTextColor(metric, t);
      td.textContent =
        metric === "danger_score" ? String(Math.round(val)) : val.toFixed(2);
      td.title = `${dim}_${s}_${s + w - 1}: ${metric}=${val}`;
      tr.appendChild(td);
    });
    tbody.appendChild(tr);
  });
  table.appendChild(tbody);

  wrap.appendChild(table);
  const xlab = document.createElement("p");
  xlab.className = "muted small";
  xlab.style.marginTop = "0.4rem";
  xlab.textContent = "↑ width (layers) · → start (layer index)";
  wrap.appendChild(xlab);
}

function padAxisLabel(dim) {
  if (dim === "valence") return "P — valence";
  if (dim === "arousal") return "A — arousal";
  return "D — dominance";
}

function saveRankMatrixPdf() {
  if (!rankRows.length) {
    alert("Load rankings first so the matrix is displayed.");
    return;
  }
  const jsPDFObj = window.jspdf;
  if (!jsPDFObj || !jsPDFObj.jsPDF) {
    alert("jsPDF did not load (check network / CDN).");
    return;
  }

  const dim = $("#dim-select").value;
  const metricSel = $("#rank-matrix-metric");
  const metric = metricSel ? metricSel.value : "composite";
  const cellVals = collectCellMetrics(dim, rankRows, rankByScenario, metric);
  const nums = Array.from(cellVals.values()).filter((v) => Number.isFinite(v));
  const { vmin, vmax, hasScale } = matrixNonZeroRange(nums);
  const legendText = rankMatrixLegendText(metric, cellVals);

  const RGB_CORNER = [18, 21, 28];
  const RGB_HEADER = [26, 31, 42];
  const RGB_HEADER_TEXT = [154, 163, 178];
  const RGB_INVALID = [30, 34, 43];
  const RGB_NODATA = [37, 42, 53];
  const RGB_BORDER = [42, 49, 64];
  const RGB_CELL_TEXT = [242, 242, 242];
  const RGB_NOCELL_TEXT = [154, 163, 178];

  const nCol = FULL32_STARTS.length + 1;
  const nRow = FULL32_WIDTHS.length + 1;
  /** Accompanying blocks (title, blurb, legend, hint, footer): 1.5× original pt / line rhythm. */
  const PDF_BODY = 1.5;
  /** Matrix value + axis labels: 2× original cell font formulas. */
  const PDF_MATRIX = 2;
  /** ~max pt so glyphs fit inside a square cell of side cellMm (mm). */
  const maxFontPtForCell = (cellMmVal) => Math.max(6, ((cellMmVal * 72) / 25.4) * 0.58);

  try {
    const pdf = new jsPDFObj.jsPDF({ unit: "mm", format: "a4", orientation: "landscape" });
    const pageW = pdf.internal.pageSize.getWidth();
    const pageH = pdf.internal.pageSize.getHeight();
    const margin = 10;
    const contentW = pageW - 2 * margin;

    const title = `Layer-range matrix · ${padAxisLabel(dim)} · metric: ${metric}`;
    const sub =
      "Axes: top row = layer start index; first column = span width (layers). Grid: 32 layers (full32-style). Grey cells = invalid span; · = no data.";
    const axisHint = "Row headers: width (layers) · Column headers: layer start";

    let y = margin;
    pdf.setTextColor(0, 0, 0);
    pdf.setFont("helvetica", "bold");
    pdf.setFontSize(12.5 * PDF_BODY);
    const titleLines = pdf.splitTextToSize(title, contentW);
    pdf.text(titleLines, margin, y);
    y += titleLines.length * 5.8 * PDF_BODY + 2.5 * PDF_BODY;

    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(9 * PDF_BODY);
    const subLines = pdf.splitTextToSize(sub, contentW);
    pdf.text(subLines, margin, y);
    y += subLines.length * 4 * PDF_BODY + 1.5 * PDF_BODY;

    const legLines = pdf.splitTextToSize(legendText, contentW);
    pdf.text(legLines, margin, y);
    y += legLines.length * 4 * PDF_BODY + 1.5 * PDF_BODY;

    pdf.setFontSize(8.5 * PDF_BODY);
    const hintLines = pdf.splitTextToSize(axisHint, contentW);
    pdf.text(hintLines, margin, y);
    y += hintLines.length * 3.8 * PDF_BODY + 5 * PDF_BODY;

    const footerStr = `Exported ${new Date().toISOString()} · PAD steering sweep`;
    pdf.setFontSize(7.5 * PDF_BODY);
    const footLines = pdf.splitTextToSize(footerStr, contentW);
    const footerH = footLines.length * 3.6 * PDF_BODY + 2 * PDF_BODY;

    const availH = pageH - y - margin - footerH - 2;
    let cellMm = Math.min(contentW / nCol, availH / nRow);
    while (y + nRow * cellMm + footerH + 2 > pageH - margin && cellMm > 2.35) {
      cellMm -= 0.15;
    }
    cellMm = Math.max(2.35, cellMm);

    const gridW = cellMm * nCol;
    const gridH = cellMm * nRow;
    const x0 = (pageW - gridW) / 2;

    const cap = maxFontPtForCell(cellMm);
    const cellFont = Math.min(Math.max(4.2, Math.min(7, cellMm * 1.05)) * PDF_MATRIX, cap);
    const headFont = Math.min(Math.max(4.5, Math.min(7.5, cellMm * 1.1)) * PDF_MATRIX, cap);

    const drawCell = (x, yy, w, h, fillRgb, txt, fontPt, txtRgb, fontStyle = "normal") => {
      pdf.setFillColor(fillRgb[0], fillRgb[1], fillRgb[2]);
      pdf.rect(x, yy, w, h, "F");
      pdf.setDrawColor(RGB_BORDER[0], RGB_BORDER[1], RGB_BORDER[2]);
      pdf.setLineWidth(0.04);
      pdf.rect(x, yy, w, h, "S");
      if (txt !== "") {
        pdf.setFont("helvetica", fontStyle);
        pdf.setFontSize(fontPt);
        pdf.setTextColor(txtRgb[0], txtRgb[1], txtRgb[2]);
        pdf.text(String(txt), x + w / 2, yy + h / 2, { align: "center", baseline: "middle" });
      }
    };

    for (let r = 0; r < nRow; r++) {
      for (let c = 0; c < nCol; c++) {
        const x = x0 + c * cellMm;
        const yy = y + r * cellMm;
        if (r === 0 && c === 0) {
          drawCell(x, yy, cellMm, cellMm, RGB_CORNER, "", headFont, RGB_HEADER_TEXT);
        } else if (r === 0) {
          const s = FULL32_STARTS[c - 1];
          drawCell(x, yy, cellMm, cellMm, RGB_HEADER, String(s), headFont, RGB_HEADER_TEXT, "bold");
        } else if (c === 0) {
          const wv = FULL32_WIDTHS[r - 1];
          drawCell(x, yy, cellMm, cellMm, RGB_HEADER, String(wv), headFont, RGB_HEADER_TEXT, "bold");
        } else {
          const wv = FULL32_WIDTHS[r - 1];
          const s = FULL32_STARTS[c - 1];
          const ok = matrixCellValid(s, wv);
          if (!ok) {
            drawCell(x, yy, cellMm, cellMm, RGB_INVALID, "\u2014", cellFont, RGB_NOCELL_TEXT);
          } else {
            const key = `${s}|${wv}`;
            const val = cellVals.get(key);
            if (!Number.isFinite(val)) {
              drawCell(x, yy, cellMm, cellMm, RGB_NODATA, "\u00b7", cellFont, RGB_NOCELL_TEXT);
            } else {
              const isZero = Math.abs(val) < 1e-12;
              const tNorm = hasScale ? matrixNormT(val, vmin, vmax, isZero) : (isZero ? 0 : 0.5);
              const fillRgb = matrixCellFillRgb(metric, tNorm);
              const label = metric === "danger_score" ? String(Math.round(val)) : val.toFixed(2);
              drawCell(x, yy, cellMm, cellMm, fillRgb, label, cellFont, RGB_CELL_TEXT);
            }
          }
        }
      }
    }

    y += gridH + 4;
    pdf.setFont("helvetica", "normal");
    pdf.setFontSize(7.5 * PDF_BODY);
    pdf.setTextColor(40, 40, 40);
    pdf.text(footLines, margin, y);

    const stamp = new Date().toISOString().slice(0, 19).replace(/[:T]/g, "-");
    const safeMetric = String(metric).replace(/[^a-zA-Z0-9._-]+/g, "_");
    pdf.save(`rank_matrix_${dim}_${safeMetric}_${stamp}.pdf`);
  } catch (e) {
    console.error(e);
    alert(`PDF export failed: ${e}`);
  }
}

function renderResultsExpandBar(container, expanded, total, onExpand, onCollapse) {
  if (!container) return;
  if (total <= RESULTS_COLLAPSE_LIMIT) {
    container.hidden = true;
    container.innerHTML = "";
    return;
  }
  container.hidden = false;
  container.innerHTML = "";

  const btn = document.createElement("button");
  btn.type = "button";
  btn.className = "btn-lite";
  if (expanded) {
    btn.textContent = "Show less";
    btn.addEventListener("click", onCollapse);
  } else {
    const hidden = total - RESULTS_COLLAPSE_LIMIT;
    btn.textContent = `Show more (${hidden} more)`;
    btn.addEventListener("click", onExpand);
  }
  container.appendChild(btn);
}

function renderRankTable() {
  const tbody = $("#rank-table tbody");
  tbody.innerHTML = "";
  const rows = [...rankRows].sort((a, b) => {
    const av = a[sortKey];
    const bv = b[sortKey];
    if (av === bv) return 0;
    if (typeof av === "string") return sortDir * String(av).localeCompare(String(bv));
    return sortDir * (av < bv ? -1 : 1);
  });
  const collapse = rows.length > RESULTS_COLLAPSE_LIMIT && !rankTableExpanded;
  rows.forEach((r, idx) => {
    const tr = document.createElement("tr");
    if (collapse && idx >= RESULTS_COLLAPSE_LIMIT) tr.classList.add("results-row-collapsed");
    tr.innerHTML = `
      <td>${escapeHtml(r.run_id || "")}</td>
      <td>${escapeHtml(r.range_id)}</td>
      <td>${fmt(r.composite)}</td>
      <td>${fmt(r.pad_accuracy)}</td>
      <td>${fmt(r.coherence)}</td>
      <td>${fmt(r.llm_corr)}</td>
      <td>${fmt(r.llm_coherence)}</td>
      <td>${fmt(r.self_direction)}</td>
      <td>${fmt(r.emotional_range)}</td>
      <td>${Number.isFinite(Number(r.danger_score)) ? String(Math.round(Number(r.danger_score))) : ""}</td>
      <td>${r.n}</td>`;
    tbody.appendChild(tr);
  });

  const bar = document.getElementById("rank-table-expand-bar");
  renderResultsExpandBar(bar, rankTableExpanded, rows.length, () => {
    rankTableExpanded = true;
    renderRankTable();
  }, () => {
    rankTableExpanded = false;
    renderRankTable();
  });
}

function fmt(x) {
  if (x == null || Number.isNaN(x)) return "";
  return Number(x).toFixed(4);
}

function escapeHtml(s) {
  return String(s)
    .replace(/&/g, "&amp;")
    .replace(/</g, "&lt;")
    .replace(/>/g, "&gt;")
    .replace(/"/g, "&quot;");
}

async function loadRankings() {
  const dim = $("#dim-select").value;
  const err = $("#rank-error");
  err.hidden = true;
  rankTableExpanded = false;
  try {
    const [resMean, resFull] = await Promise.all([
      fetch(`/api/rankings_mean?dimension=${encodeURIComponent(dim)}`),
      fetch(`/api/rankings?dimension=${encodeURIComponent(dim)}`),
    ]);
    if (!resMean.ok) throw new Error(await resMean.text());
    if (!resFull.ok) throw new Error(await resFull.text());
    rankRows = await resMean.json();
    const full = await resFull.json();
    rankByScenario = Array.isArray(full.by_scenario) ? full.by_scenario : [];
    renderRankTable();
    renderRankMatrix();
  } catch (e) {
    rankRows = [];
    rankByScenario = [];
    renderRankTable();
    renderRankMatrix();
    err.textContent = String(e);
    err.hidden = false;
  }
}

function sortSetup() {
  document.querySelectorAll("#rank-table th[data-sort]").forEach((th) => {
    th.addEventListener("click", () => {
      const k = th.getAttribute("data-sort");
      if (sortKey === k) sortDir *= -1;
      else {
        sortKey = k;
        sortDir = k === "range_id" || k === "run_id" || k === "danger_score" ? 1 : -1;
      }
      renderRankTable();
    });
  });
}

function pickPadForDim(r, dim) {
  if (r.pad_v == null && r.pad_a == null && r.pad_d == null) return null;
  if (dim === "arousal") return r.pad_a;
  if (dim === "dominance") return r.pad_d;
  return r.pad_v;
}

/** Ordinary least squares y ≈ intercept + slope*x; RMSE/MAD vs fitted line. */
function linearTrend(xs, ys) {
  const pts = [];
  for (let i = 0; i < xs.length; i++) {
    const x = Number(xs[i]);
    const y = Number(ys[i]);
    if (!Number.isFinite(x) || !Number.isFinite(y)) continue;
    pts.push([x, y]);
  }
  const n = pts.length;
  if (n < 2) {
    const y0 = n === 1 ? pts[0][1] : 0;
    return { slope: 0, intercept: y0, rmse: 0, mad: 0 };
  }
  let sx = 0;
  let sy = 0;
  let sxx = 0;
  let sxy = 0;
  for (const [x, y] of pts) {
    sx += x;
    sy += y;
    sxx += x * x;
    sxy += x * y;
  }
  const denom = n * sxx - sx * sx;
  const slope = Math.abs(denom) < 1e-12 ? 0 : (n * sxy - sx * sy) / denom;
  const intercept = (sy - slope * sx) / n;
  let sse = 0;
  let sabs = 0;
  for (const [x, y] of pts) {
    const pred = intercept + slope * x;
    const e = y - pred;
    sse += e * e;
    sabs += Math.abs(e);
  }
  const rmse = Math.sqrt(sse / n);
  const mad = sabs / n;
  return { slope, intercept, rmse, mad };
}

function syncRespPadDimFromRange() {
  const el = $("#resp-pad-dim");
  const inp = $("#resp-range");
  if (!el || !inp) return;
  const m = inp.value.trim().match(/^(valence|arousal|dominance)_/);
  if (m) el.value = m[1];
}

async function loadAllScenarioCharts() {
  const rangeId = $("#resp-range").value.trim();
  const runIdEl = $("#resp-run-id");
  const runId = runIdEl && runIdEl.value.trim() ? runIdEl.value.trim() : "";
  const padDim = $("#resp-pad-dim").value;
  const stack = $("#resp-plots-stack");
  const dump = $("#resp-dump");
  dump.textContent = "";
  if (!rangeId) {
    dump.textContent = "Enter a range_id.";
    return;
  }
  if (typeof Plotly === "undefined") {
    dump.textContent = "Plotly failed to load.";
    return;
  }

  const scRes = await fetch("/api/scenarios");
  if (!scRes.ok) {
    dump.textContent = await scRes.text();
    return;
  }
  const scenarios = await scRes.json();
  if (!Array.isArray(scenarios) || !scenarios.length) {
    dump.textContent = "No scenarios in scenarios.json.";
    return;
  }

  stack.innerHTML = "";
  respPlotsExpanded = false;
  const summary = {};

  const fetched = await Promise.all(
    scenarios.map(async (sc, i) => {
      const sid = sc.id;
      let url = `/api/responses?range_id=${encodeURIComponent(rangeId)}&scenario_id=${encodeURIComponent(sid)}`;
      if (runId) url += `&run_id=${encodeURIComponent(runId)}`;
      const res = await fetch(url);
      let rows = [];
      let err = null;
      if (!res.ok) err = await res.text();
      else rows = await res.json();
      return { i, sid, rows, err, status: res.status };
    })
  );

  for (const { i, sid, rows, err, status } of fetched) {
    const cell = document.createElement("div");
    cell.className = "resp-plot-cell";
    const plotId = `resp-plot-${i}`;
    cell.id = plotId;
    stack.appendChild(cell);

    if (err) {
      cell.classList.add("muted-panel");
      cell.textContent = `${sid}: ${err}`;
      summary[sid] = { error: status };
      continue;
    }
    if (!rows.length) {
      cell.classList.add("muted-panel");
      cell.textContent = `${sid}: no rows for this range / run.`;
      summary[sid] = { n: 0 };
      continue;
    }

    const pairs = rows
      .map((r) => {
        const m = Number(r.multiplier);
        const p = pickPadForDim(r, padDim);
        if (!Number.isFinite(m) || p == null || !Number.isFinite(Number(p))) return null;
        return {
          m,
          p: Number(p),
          hover: (r.response_text || "").slice(0, 120),
        };
      })
      .filter(Boolean);
    pairs.sort((a, b) => a.m - b.m);
    const mult = pairs.map((o) => o.m);
    const padY = pairs.map((o) => o.p);
    const texts = pairs.map((o) => o.hover);

    const llmPairs = rows
      .map((r) => {
        const m = Number(r.multiplier);
        const v = Number(r.llm_axis);
        const c = Number(r.llm_coherence);
        if (!Number.isFinite(m) || !Number.isFinite(v)) return null;
        return { m, v, c };
      })
      .filter(Boolean);
    llmPairs.sort((a, b) => a.m - b.m);
    const llmX = llmPairs.map((o) => o.m);
    const llmY = llmPairs.map((o) => o.v);
    const llmText = llmPairs.map((o) => {
      const score = `LLM judge score ${o.v.toFixed(3)}`;
      const coh = Number.isFinite(o.c) ? ` · coherence ${o.c.toFixed(3)}` : "";
      return score + coh;
    });

    if (!mult.length) {
      cell.classList.add("muted-panel");
      cell.textContent = `${sid}: no scored PAD points for axis "${padDim}".`;
      summary[sid] = { n: 0, note: "no_pad" };
      continue;
    }

    const t = linearTrend(mult, padY);
    const mMin = Math.min(...mult);
    const mMax = Math.max(...mult);
    const lineX = mMin === mMax ? [mMin - 0.01, mMax + 0.01] : [mMin, mMax];
    const lineY = lineX.map((x) => t.intercept + t.slope * x);

    summary[sid] = {
      n: pairs.length,
      slope: t.slope,
      intercept: t.intercept,
      rmse: t.rmse,
      mad: t.mad,
    };
    if (llmPairs.length) {
      summary[sid].llm_n = llmPairs.length;
      summary[sid].llm_mean = llmY.reduce((a, b) => a + b, 0) / llmY.length;
    }

    const title = `${sid} · OLS slope ${t.slope.toFixed(4)} · RMSE ${t.rmse.toFixed(4)} · MAD ${t.mad.toFixed(4)}`;

    await Plotly.newPlot(
      plotId,
      [
        {
          x: mult,
          y: padY,
          mode: "markers",
          name: "PAD",
          marker: { size: 10, color: "#6cb3ff" },
          text: texts,
          hovertemplate: "%{text}<extra></extra>",
        },
        {
          x: lineX,
          y: lineY,
          mode: "lines",
          name: "Trend",
          line: { color: "#ffb86b", width: 2 },
        },
        ...(llmPairs.length
          ? [
              {
                x: llmX,
                y: llmY,
                mode: "markers",
                name: "LLM judge score",
                marker: { size: 9, color: "#b77bff", symbol: "diamond" },
                text: llmText,
                hovertemplate: "%{text}<extra>LLM judge</extra>",
              },
            ]
          : []),
      ],
      {
        title: { text: title, font: { size: 13, color: "#f0f0f0" } },
        paper_bgcolor: "#161616",
        plot_bgcolor: "#242424",
        font: { color: "#f0f0f0" },
        margin: { t: 52, l: 52, r: 20, b: 44 },
        xaxis: { title: "steering multiplier", gridcolor: "#2a3140", zerolinecolor: "#444" },
        yaxis: {
          title: `classifier ${padDim}`,
          gridcolor: "#2a3140",
          zerolinecolor: "#444",
        },
        showlegend: true,
        legend: { orientation: "h", y: 1.08, x: 0 },
      },
      { responsive: true, displayModeBar: true }
    );
  }

  dump.textContent = JSON.stringify({ range_id: rangeId, pad_axis: padDim, by_scenario: summary }, null, 2);
  applyRespPlotsCollapse();
}

function applyRespPlotsCollapse() {
  const stack = $("#resp-plots-stack");
  if (!stack) return;
  const cells = [...stack.querySelectorAll(".resp-plot-cell")];
  const collapse = cells.length > RESULTS_COLLAPSE_LIMIT && !respPlotsExpanded;
  cells.forEach((cell, idx) => {
    cell.classList.toggle("results-item-collapsed", collapse && idx >= RESULTS_COLLAPSE_LIMIT);
  });
  const bar = document.getElementById("resp-plots-expand-bar");
  renderResultsExpandBar(bar, respPlotsExpanded, cells.length, () => {
    respPlotsExpanded = true;
    applyRespPlotsCollapse();
  }, () => {
    respPlotsExpanded = false;
    applyRespPlotsCollapse();
  });
}

function pilotNumLayers() {
  const n = parseInt($("#pilot-nlayers").value, 10);
  if (Number.isNaN(n) || n < 1) return 32;
  return Math.min(128, n);
}

function pilotSyncSliders() {
  const last = pilotNumLayers() - 1;
  const startEl = $("#pilot-start");
  const endEl = $("#pilot-end");
  startEl.max = String(last);
  endEl.max = String(last);
  let s = parseInt(startEl.value, 10);
  let e = parseInt(endEl.value, 10);
  if (s > last) s = last;
  if (e > last) e = last;
  if (e < s) e = s;
  startEl.value = String(s);
  endEl.value = String(e);
  $("#pilot-start-val").textContent = String(s);
  $("#pilot-end-val").textContent = String(e);
  const dim = $("#pilot-dim").value;
  $("#pilot-range-preview").textContent = `${dim}_${s}_${e}`;
  $("#pilot-out-preview").textContent = `sweep/results/vectors/${dim}_${s}_${e}.pt`;
}

let pilotPollTimer = null;
let pilotActiveJob = null;

function pilotClearInterval() {
  if (pilotPollTimer) {
    clearInterval(pilotPollTimer);
    pilotPollTimer = null;
  }
}

function pilotSetBusy(busy) {
  const b = $("#pilot-extract-btn");
  b.disabled = busy;
  $("#pilot-status").textContent = busy ? "Running…" : "";
}

async function pilotPollOnce() {
  if (!pilotActiveJob) return;
  const res = await fetch(`/api/extract/${encodeURIComponent(pilotActiveJob)}`);
  if (!res.ok) {
    $("#pilot-error").textContent = await res.text();
    $("#pilot-error").hidden = false;
    pilotClearInterval();
    pilotActiveJob = null;
    pilotSetBusy(false);
    return;
  }
  const data = await res.json();
  $("#pilot-log").textContent = data.log_tail || "";
  const st = data.status;
  $("#pilot-status").textContent =
    st === "running"
      ? "Running…"
      : st === "done"
        ? `Done (exit ${data.exit_code}). output_exists=${data.output_exists}`
        : `Error (exit ${data.exit_code})`;
  if (st === "done" || st === "error") {
    pilotClearInterval();
    pilotActiveJob = null;
    pilotSetBusy(false);
    if (st === "done") loadVectors();
  }
}

function pilotResetJob() {
  pilotClearInterval();
  pilotActiveJob = null;
}

async function pilotStartExtract() {
  $("#pilot-error").hidden = true;
  $("#pilot-error").textContent = "";
  pilotResetJob();
  pilotSyncSliders();
  const dim = $("#pilot-dim").value;
  const start = parseInt($("#pilot-start").value, 10);
  const end = parseInt($("#pilot-end").value, 10);
  const num_layers = pilotNumLayers();
  if (end < start) {
    $("#pilot-error").textContent = "End layer must be >= start layer.";
    $("#pilot-error").hidden = false;
    return;
  }
  pilotSetBusy(true);
  $("#pilot-log").textContent = "";
  try {
    const res = await fetch("/api/extract", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ dimension: dim, start, end, num_layers }),
    });
    if (!res.ok) {
      const t = await res.text();
      throw new Error(t || res.statusText);
    }
    const j = await res.json();
    pilotActiveJob = j.job_id;
    $("#pilot-status").textContent = `Job ${j.job_id}…`;
    await pilotPollOnce();
    if (pilotActiveJob) {
      pilotPollTimer = setInterval(() => {
        pilotPollOnce().catch((e) => {
          $("#pilot-error").textContent = String(e);
          $("#pilot-error").hidden = false;
          pilotResetJob();
          pilotSetBusy(false);
        });
      }, 1200);
    }
  } catch (e) {
    $("#pilot-error").textContent = String(e);
    $("#pilot-error").hidden = false;
    pilotSetBusy(false);
  }
}

function pilotSetup() {
  ["pilot-dim", "pilot-nlayers", "pilot-start", "pilot-end"].forEach((id) => {
    $(`#${id}`).addEventListener("input", () => {
      const last = pilotNumLayers() - 1;
      const sEl = $("#pilot-start");
      const eEl = $("#pilot-end");
      let s = parseInt(sEl.value, 10);
      let e = parseInt(eEl.value, 10);
      if (e < s) eEl.value = String(s);
      if (parseInt(eEl.value, 10) > last) eEl.value = String(last);
      pilotSyncSliders();
    });
  });
  $("#pilot-extract-btn").addEventListener("click", () => {
    pilotStartExtract().catch((e) => {
      $("#pilot-error").textContent = String(e);
      $("#pilot-error").hidden = false;
      pilotSetBusy(false);
    });
  });
  pilotSyncSliders();
}

const RUNNER = {
  gen: { id: null, timer: null },
  score: { id: null, timer: null },
  llm: { id: null, timer: null },
  ana: { id: null, timer: null },
  pipe: { id: null, timer: null },
};

function runnerClear(which) {
  const s = RUNNER[which];
  if (!s) return;
  if (s.timer) {
    clearInterval(s.timer);
    s.timer = null;
  }
  s.id = null;
}

function guessFromVectorFile(name) {
  const m = String(name).match(/^(valence|arousal|dominance)_(\d+)_(\d+)\.pt$/);
  if (!m) return null;
  const dim = m[1];
  return { dimension: dim, range_id: `${dim}_${m[2]}_${m[3]}` };
}

async function runnerRefreshLists() {
  const sel = $("#gen-vector");
  const cur = sel.value;
  sel.innerHTML = '<option value="">— pick vector —</option>';
  try {
    const res = await fetch("/api/vectors");
    const items = await res.json();
    items.forEach((v) => {
      const o = document.createElement("option");
      o.value = v.name;
      o.textContent = v.name;
      sel.appendChild(o);
    });
    if (cur && [...sel.options].some((o) => o.value === cur)) sel.value = cur;
  } catch (e) {
    console.warn(e);
  }

  const fs = $("#score-file");
  const curF = fs.value;
  fs.innerHTML = '<option value="">— pick JSONL —</option>';
  const llmFs = $("#llm-file");
  const curL = llmFs ? llmFs.value : "";
  if (llmFs) llmFs.innerHTML = '<option value="">— pick JSONL —</option>';
  try {
    const res = await fetch("/api/response-files");
    const rows = await res.json();
    rows.forEach((r) => {
      const o = document.createElement("option");
      o.value = r.name;
      o.textContent = r.name;
      fs.appendChild(o);
      if (llmFs) {
        const o2 = document.createElement("option");
        o2.value = r.name;
        o2.textContent = r.name;
        llmFs.appendChild(o2);
      }
    });
    if (curF && [...fs.options].some((o) => o.value === curF)) fs.value = curF;
    if (llmFs && curL && [...llmFs.options].some((o) => o.value === curL)) llmFs.value = curL;
  } catch (e) {
    console.warn(e);
  }
  await pipelineRefreshStates();
}

async function pipelineRefreshStates() {
  const dl = $("#pipe-state-datalist");
  if (!dl) return;
  try {
    const res = await fetch("/api/pipeline/states?limit=80");
    if (!res.ok) return;
    const rows = await res.json();
    dl.innerHTML = "";
    rows.forEach((r) => {
      const o = document.createElement("option");
      o.value = r.name;
      dl.appendChild(o);
    });
  } catch (e) {
    console.warn("pipeline states", e);
  }
}

async function pipelineRun() {
  const err = $("#pipe-err");
  err.hidden = true;
  err.textContent = "";
  runnerClear("pipe");
  const state = $("#pipe-state").value.trim();
  if (!state || !state.endsWith(".json")) {
    err.textContent = "State must be a .json basename (e.g. pad_pipeline_state.json).";
    err.hidden = false;
    return;
  }
  const body = {
    state,
    preset: "full32",
    dimensions: $("#pipe-dims").value.trim() || "valence,arousal,dominance",
    phase: $("#pipe-phase").value,
    force_new_state: $("#pipe-force").checked,
    from_job: 0,
    num_layers: parseInt($("#pipe-nlayers").value, 10) || 32,
    skip_existing_vectors: $("#pipe-skip-vec").checked,
    cuda_empty_cache: $("#pipe-cuda-cache").checked,
    max_new_tokens: 100,
    do_sample: false,
    temperature: 0.7,
    top_p: 0.9,
  };

  const pipeBoxes = document.querySelectorAll("#pipe-scenarios input[type=checkbox]");
  if (pipeBoxes.length) {
    const nChecked = [...pipeBoxes].filter((b) => b.checked).length;
    if (nChecked === 0) {
      err.textContent = "Select at least one scenario for the pipeline, or reload scenarios.";
      err.hidden = false;
      return;
    }
    const ps = pipeScenarioSelection();
    if (ps && ps.length) body.scenario_ids = ps.join(",");
  }

  runnerSetPipeBusy(true);
  $("#pipe-log").textContent = "";
  $("#pipe-status").textContent = "";
  try {
    const res = await fetch("/api/pipeline/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) {
      throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j));
    }
    RUNNER.pipe.id = j.job_id;
    $("#pipe-status").textContent = `Job ${j.job_id} → ${j.state}`;
    await runnerPollOnce("pipe", $("#pipe-log"), $("#pipe-status"), $("#pipe-err"));
    if (RUNNER.pipe.id) {
      RUNNER.pipe.timer = setInterval(() => {
        runnerPollOnce("pipe", $("#pipe-log"), $("#pipe-status"), $("#pipe-err")).catch((e) => {
          $("#pipe-err").textContent = String(e);
          $("#pipe-err").hidden = false;
          runnerClear("pipe");
          runnerSetPipeBusy(false);
        });
      }, 2000);
    }
  } catch (e) {
    $("#pipe-err").textContent = String(e);
    $("#pipe-err").hidden = false;
    runnerSetPipeBusy(false);
  }
}

async function pipelinePause() {
  const err = $("#pipe-err");
  err.hidden = true;
  const state = $("#pipe-state").value.trim();
  try {
    const res = await fetch("/api/pipeline/pause", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state }),
    });
    const j = await res.json();
    if (!res.ok) throw new Error(await res.text());
    $("#pipe-status").textContent = `Paused (${j.pause_file})`;
  } catch (e) {
    err.textContent = String(e);
    err.hidden = false;
  }
}

async function pipelineResume() {
  const err = $("#pipe-err");
  err.hidden = true;
  const state = $("#pipe-state").value.trim();
  try {
    const res = await fetch("/api/pipeline/resume", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ state }),
    });
    const j = await res.json();
    if (!res.ok) throw new Error(await res.text());
    $("#pipe-status").textContent = j.removed ? "Pause file removed." : "(no pause file)";
  } catch (e) {
    err.textContent = String(e);
    err.hidden = false;
  }
}

let runnerScenariosLoaded = false;

function fillScenarioCheckboxes(containerEl, scenarios) {
  if (!containerEl) return;
  containerEl.innerHTML = "";
  scenarios.forEach((s) => {
    const id = s.id;
    const lab = document.createElement("label");
    const cb = document.createElement("input");
    cb.type = "checkbox";
    cb.value = id;
    cb.checked = true;
    lab.appendChild(cb);
    lab.appendChild(document.createTextNode(` ${id}`));
    containerEl.appendChild(lab);
  });
}

async function runnerLoadScenarios() {
  if (runnerScenariosLoaded) return;
  const res = await fetch("/api/scenarios");
  if (!res.ok) return;
  const sc = await res.json();
  fillScenarioCheckboxes($("#gen-scenarios"), sc);
  fillScenarioCheckboxes(document.getElementById("pipe-scenarios"), sc);
  runnerScenariosLoaded = true;
}

function runnerScenarioSelection() {
  const boxes = [...document.querySelectorAll("#gen-scenarios input[type=checkbox]")];
  if (!boxes.length) return null;
  const checked = boxes.filter((b) => b.checked).map((b) => b.value);
  if (checked.length === 0 || checked.length === boxes.length) return null;
  return checked;
}

function pipeScenarioSelection() {
  const boxes = [...document.querySelectorAll("#pipe-scenarios input[type=checkbox]")];
  if (!boxes.length) return null;
  const checked = boxes.filter((b) => b.checked).map((b) => b.value);
  if (checked.length === 0 || checked.length === boxes.length) return null;
  return checked;
}

function runnerSetGenBusy(busy) {
  $("#gen-run").disabled = busy;
}

function runnerSetScoreBusy(busy) {
  $("#score-run").disabled = busy;
}

function runnerSetLlmBusy(busy) {
  const b = document.getElementById("llm-run");
  const all = document.getElementById("llm-run-all");
  if (b) b.disabled = busy;
  if (all) all.disabled = busy;
}

function runnerSetAnaBusy(busy) {
  $("#ana-run").disabled = busy;
}

function runnerSetPipeBusy(busy) {
  const b = $("#pipe-run");
  if (b) b.disabled = busy;
}

async function runnerPollOnce(which, logEl, statusEl, errEl) {
  const id = RUNNER[which].id;
  if (!id) return;
  const res = await fetch(`/api/runner/${encodeURIComponent(id)}`);
  if (!res.ok) {
    errEl.textContent = await res.text();
    errEl.hidden = false;
    runnerClear(which);
    if (which === "gen") runnerSetGenBusy(false);
    if (which === "score") runnerSetScoreBusy(false);
    if (which === "llm") runnerSetLlmBusy(false);
    if (which === "ana") runnerSetAnaBusy(false);
    if (which === "pipe") runnerSetPipeBusy(false);
    return;
  }
  const data = await res.json();
  logEl.textContent = data.log_tail || "";
  const st = data.status;
  let msg = st === "running" ? "Running…" : st === "done" ? `Done (exit ${data.exit_code})` : `Error (exit ${data.exit_code})`;
  if (which === "gen" && data.output_jsonl) msg += ` — ${data.output_exists ? "output ok" : "output missing"}`;
  if (which === "pipe" && data.state_file) msg += ` — state: ${data.state_file}`;
  if (which === "llm" && Number.isFinite(Number(data.files_done)) && Number.isFinite(Number(data.files_total))) {
    msg += ` — ${data.files_done}/${data.files_total}`;
    if (data.current_file) msg += ` · ${data.current_file}`;
  }
  statusEl.textContent = msg;
  if (st === "done" || st === "error") {
    if (RUNNER[which].timer) {
      clearInterval(RUNNER[which].timer);
      RUNNER[which].timer = null;
    }
    RUNNER[which].id = null;
    if (which === "gen") runnerSetGenBusy(false);
    if (which === "score") runnerSetScoreBusy(false);
    if (which === "llm") runnerSetLlmBusy(false);
    if (which === "ana") runnerSetAnaBusy(false);
    if (which === "pipe") runnerSetPipeBusy(false);
    if (st === "done") runnerRefreshLists();
  }
}

async function runnerStartGen() {
  $("#gen-err").hidden = true;
  $("#gen-err").textContent = "";
  runnerClear("gen");
  const vector = $("#gen-vector").value;
  const rangeId = $("#gen-range-id").value.trim();
  const dimension = parseDimensionFromRangeId(rangeId);
  if (!vector || !rangeId) {
    $("#gen-err").textContent = "Choose a vector and set range_id.";
    $("#gen-err").hidden = false;
    return;
  }
  if (!dimension) {
    $("#gen-err").textContent = "range_id must start with valence_, arousal_, or dominance_.";
    $("#gen-err").hidden = false;
    return;
  }
  const body = {
    vector,
    range_id: rangeId,
    dimension,
    multipliers: $("#gen-mults").value.trim(),
    resume: $("#gen-resume").checked,
    max_new_tokens: parseInt($("#gen-max-tok").value, 10) || 100,
    do_sample: false,
    temperature: parseFloat($("#gen-temp").value) || 0.7,
    top_p: parseFloat($("#gen-topp").value) || 0.9,
  };
  const sc = runnerScenarioSelection();
  if (sc) body.scenario_ids = sc;

  runnerSetGenBusy(true);
  $("#gen-log").textContent = "";
  $("#gen-status").textContent = "";
  try {
    const res = await fetch("/api/generate", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    RUNNER.gen.id = j.job_id;
    $("#gen-status").textContent = `Job ${j.job_id} → ${j.output_jsonl}`;
    await runnerPollOnce("gen", $("#gen-log"), $("#gen-status"), $("#gen-err"));
    if (RUNNER.gen.id) {
      RUNNER.gen.timer = setInterval(() => {
        runnerPollOnce("gen", $("#gen-log"), $("#gen-status"), $("#gen-err")).catch((e) => {
          $("#gen-err").textContent = String(e);
          $("#gen-err").hidden = false;
          runnerClear("gen");
          runnerSetGenBusy(false);
        });
      }, 1200);
    }
  } catch (e) {
    $("#gen-err").textContent = String(e);
    $("#gen-err").hidden = false;
    runnerSetGenBusy(false);
  }
}

async function runnerStartScore() {
  $("#score-err").hidden = true;
  $("#score-err").textContent = "";
  runnerClear("score");
  const input_file = $("#score-file").value;
  if (!input_file) {
    $("#score-err").textContent = "Pick a JSONL file.";
    $("#score-err").hidden = false;
    return;
  }
  const body = { input_file, resume: $("#score-resume").checked };
  const emo = $("#score-emo").value.trim();
  if (emo) body.emotion_model = emo;

  runnerSetScoreBusy(true);
  $("#score-log").textContent = "";
  $("#score-status").textContent = "";
  try {
    const res = await fetch("/api/score", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    RUNNER.score.id = j.job_id;
    $("#score-status").textContent = `Job ${j.job_id}`;
    await runnerPollOnce("score", $("#score-log"), $("#score-status"), $("#score-err"));
    if (RUNNER.score.id) {
      RUNNER.score.timer = setInterval(() => {
        runnerPollOnce("score", $("#score-log"), $("#score-status"), $("#score-err")).catch((e) => {
          $("#score-err").textContent = String(e);
          $("#score-err").hidden = false;
          runnerClear("score");
          runnerSetScoreBusy(false);
        });
      }, 1200);
    }
  } catch (e) {
    $("#score-err").textContent = String(e);
    $("#score-err").hidden = false;
    runnerSetScoreBusy(false);
  }
}

async function runnerStartLlmJudge(all) {
  const err = $("#llm-err");
  err.hidden = true;
  err.textContent = "";
  runnerClear("llm");

  const seedRaw = $("#llm-seed").value.trim();
  const shuffle_seed = seedRaw ? parseInt(seedRaw, 10) : null;
  const overwrite = $("#llm-overwrite").checked;

  const body = { all: !!all, overwrite };
  const dimEl = document.getElementById("llm-dimension");
  const dim = dimEl && dimEl.value ? dimEl.value : "";
  if (dim) body.dimension = dim;
  if (!all) {
    const f = $("#llm-file").value;
    if (!f) {
      err.textContent = "Pick a JSONL file (or run on all).";
      err.hidden = false;
      return;
    }
    body.files = [f];
  }
  if (shuffle_seed != null && Number.isFinite(shuffle_seed)) body.shuffle_seed = shuffle_seed;

  runnerSetLlmBusy(true);
  $("#llm-log").textContent = "";
  $("#llm-status").textContent = "";
  try {
    const res = await fetch("/api/llm-judge/run", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify(body),
    });
    const j = await res.json().catch(() => ({}));
    if (!res.ok) throw new Error(typeof j.detail === "string" ? j.detail : JSON.stringify(j.detail ?? j));
    RUNNER.llm.id = j.job_id;
    $("#llm-status").textContent = `Job ${j.job_id}`;
    await runnerPollOnce("llm", $("#llm-log"), $("#llm-status"), $("#llm-err"));
    if (RUNNER.llm.id) {
      RUNNER.llm.timer = setInterval(() => {
        runnerPollOnce("llm", $("#llm-log"), $("#llm-status"), $("#llm-err")).catch((e) => {
          $("#llm-err").textContent = String(e);
          $("#llm-err").hidden = false;
          runnerClear("llm");
          runnerSetLlmBusy(false);
        });
      }, 1400);
    }
  } catch (e) {
    err.textContent = String(e);
    err.hidden = false;
    runnerSetLlmBusy(false);
  }
}

async function runnerStartAnalyze() {
  $("#ana-err").hidden = true;
  $("#ana-err").textContent = "";
  runnerClear("ana");
  runnerSetAnaBusy(true);
  $("#ana-log").textContent = "";
  $("#ana-status").textContent = "";
  try {
    const res = await fetch("/api/analyze", { method: "POST" });
    if (!res.ok) throw new Error(await res.text());
    const j = await res.json();
    RUNNER.ana.id = j.job_id;
    $("#ana-status").textContent = `Job ${j.job_id}`;
    await runnerPollOnce("ana", $("#ana-log"), $("#ana-status"), $("#ana-err"));
    if (RUNNER.ana.id) {
      RUNNER.ana.timer = setInterval(() => {
        runnerPollOnce("ana", $("#ana-log"), $("#ana-status"), $("#ana-err")).catch((e) => {
          $("#ana-err").textContent = String(e);
          $("#ana-err").hidden = false;
          runnerClear("ana");
          runnerSetAnaBusy(false);
        });
      }, 1200);
    }
  } catch (e) {
    $("#ana-err").textContent = String(e);
    $("#ana-err").hidden = false;
    runnerSetAnaBusy(false);
  }
}

function runnerOnVectorChange() {
  const name = $("#gen-vector").value;
  const g = guessFromVectorFile(name);
  if (g) $("#gen-range-id").value = g.range_id;
}

function runnerSetup() {
  $("#gen-vector").addEventListener("change", runnerOnVectorChange);
  $("#gen-sel-all").addEventListener("click", () => {
    document.querySelectorAll("#gen-scenarios input[type=checkbox]").forEach((b) => {
      b.checked = true;
    });
  });
  $("#gen-sel-none").addEventListener("click", () => {
    document.querySelectorAll("#gen-scenarios input[type=checkbox]").forEach((b) => {
      b.checked = false;
    });
  });
  $("#gen-run").addEventListener("click", () => runnerStartGen());
  $("#score-run").addEventListener("click", () => runnerStartScore());
  $("#score-refresh").addEventListener("click", () => runnerRefreshLists());
  const llmRun = document.getElementById("llm-run");
  if (llmRun) llmRun.addEventListener("click", () => runnerStartLlmJudge(false));
  const llmAll = document.getElementById("llm-run-all");
  if (llmAll) llmAll.addEventListener("click", () => runnerStartLlmJudge(true));
  const llmRef = document.getElementById("llm-refresh");
  if (llmRef) llmRef.addEventListener("click", () => runnerRefreshLists());
  $("#ana-run").addEventListener("click", () => runnerStartAnalyze());
  const psa = document.getElementById("pipe-sel-all");
  if (psa) {
    psa.addEventListener("click", () => {
      document.querySelectorAll("#pipe-scenarios input[type=checkbox]").forEach((b) => {
        b.checked = true;
      });
    });
  }
  const psn = document.getElementById("pipe-sel-none");
  if (psn) {
    psn.addEventListener("click", () => {
      document.querySelectorAll("#pipe-scenarios input[type=checkbox]").forEach((b) => {
        b.checked = false;
      });
    });
  }
  const pipeBindings = [
    ["pipe-refresh-states", () => pipelineRefreshStates()],
    ["pipe-run", () => pipelineRun()],
    ["pipe-pause", () => pipelinePause()],
    ["pipe-resume", () => pipelineResume()],
  ];
  pipeBindings.forEach(([id, fn]) => {
    const el = document.getElementById(id);
    if (!el) {
      console.warn(`[sweep UI] missing #${id} — reload the page (hard refresh) if pipeline controls were added recently.`);
      return;
    }
    el.addEventListener("click", () => {
      try {
        const ret = fn();
        if (ret != null && typeof ret.catch === "function") {
          ret.catch((e) => console.error(`[sweep UI #${id}]`, e));
        }
      } catch (e) {
        console.error(`[sweep UI #${id}]`, e);
      }
    });
  });
}

document.addEventListener("DOMContentLoaded", () => {
  tabSetup();
  sortSetup();
  pilotSetup();
  runnerSetup();
  runnerLoadScenarios().catch(console.warn);
  $("#load-rankings").addEventListener("click", loadRankings);
  const rmm = $("#rank-matrix-metric");
  if (rmm) rmm.addEventListener("change", () => renderRankMatrix());
  const pdfBtn = document.getElementById("rank-matrix-pdf");
  if (pdfBtn) pdfBtn.addEventListener("click", () => saveRankMatrixPdf());
  const dimSel = $("#dim-select");
  if (dimSel) {
    dimSel.addEventListener("change", () => {
      const panel = $("#tab-results");
      if (panel && panel.classList.contains("active")) loadRankings();
    });
  }
  $("#load-responses").addEventListener("click", () => {
    loadAllScenarioCharts().catch((e) => {
      $("#resp-dump").textContent = String(e);
    });
  });
  const rrf = $("#resp-refresh-filters");
  if (rrf) rrf.addEventListener("click", () => loadResponseFilters());
  const respRange = $("#resp-range");
  if (respRange) {
    respRange.addEventListener("change", syncRespPadDimFromRange);
    respRange.addEventListener("blur", syncRespPadDimFromRange);
  }
  loadResponseFilters();
});

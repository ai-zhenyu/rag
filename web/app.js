// Document Q&A front end. Everything shown from the documents or the model is escaped before display.

const $ = (sel) => document.querySelector(sel);
const esc = (s) => String(s ?? "").replace(/[&<>"']/g, (c) => ({ "&": "&amp;", "<": "&lt;", ">": "&gt;", '"': "&quot;", "'": "&#39;" }[c]));
const fmt = (n, d = 3) => (n === null || n === undefined ? "–" : Number(n).toFixed(d));
const pct = (n, d) => (d ? `${Math.round((100 * n) / d)}%` : "–");

const EXAMPLES = [
  "What was NVIDIA's revenue for the quarter ended July 30, 2023?",
  "Compare NVIDIA's and AMD's revenue last quarter.",
  "How much stock was sitting in NVIDIA's warehouses at the end of the quarter?",
  "Why did NVIDIA's gross margin fall compared with the previous quarter?",
  "Who won the 2022 FIFA World Cup?",
];

let settings = null;   // from /api/documents
let lastResult = null; // last /api/ask response

// ---------- tabs ----------
document.querySelectorAll(".tab").forEach((tab) =>
  tab.addEventListener("click", () => {
    document.querySelectorAll(".tab").forEach((t) => t.classList.toggle("active", t === tab));
    document.querySelectorAll(".tab-panel").forEach((p) => (p.hidden = p.id !== `tab-${tab.dataset.tab}`));
    if (tab.dataset.tab === "eval") loadBenchmark();
  })
);

// ---------- ask ----------
const questionBox = $("#question");
questionBox.addEventListener("input", () => ($("#char-count").textContent = questionBox.value.length));
questionBox.addEventListener("keydown", (e) => {
  if (e.key === "Enter" && !e.shiftKey) {
    e.preventDefault();
    $("#ask-form").requestSubmit();
  }
});
$("#ask-form").addEventListener("submit", (e) => {
  e.preventDefault();
  ask(questionBox.value);
});

$("#examples").innerHTML = EXAMPLES.map((q) => `<button type="button" class="chip">${esc(q)}</button>`).join("");
$("#examples").addEventListener("click", (e) => {
  if (!e.target.classList.contains("chip")) return;
  questionBox.value = e.target.textContent;
  questionBox.dispatchEvent(new Event("input"));
  ask(questionBox.value);
});

async function ask(question) {
  if (!question.trim()) return questionBox.focus();
  $("#ask-btn").disabled = true;
  $("#result").hidden = true;
  $("#error").hidden = true;
  $("#loading").hidden = false;
  const steps = ["Searching the documents…", "Re-ranking the candidates…", "Writing the answer…"];
  let i = 0;
  $("#loading-text").textContent = steps[0];
  const ticker = setInterval(() => ($("#loading-text").textContent = steps[Math.min(++i, steps.length - 1)]), 1400);
  try {
    const res = await fetch("/api/ask", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ question }),
    });
    if (!res.ok) throw new Error(`Server error ${res.status}`);
    lastResult = await res.json();
    renderResult(lastResult);
  } catch (err) {
    $("#error").textContent = `Something went wrong: ${err.message}. Is the server still running?`;
    $("#error").hidden = false;
  } finally {
    clearInterval(ticker);
    $("#loading").hidden = true;
    $("#ask-btn").disabled = false;
  }
}

const STATUS = {
  answered: ["Answered from the documents", "good"],
  not_found: ["Not in the documents", "warn"],
  blocked: ["Reply blocked by the output check", "bad"],
  rejected: ["Question not accepted", "bad"],
};

function renderResult(r) {
  const [label, cls] = STATUS[r.status] || ["", ""];
  $("#status-badge").className = `badge ${cls}`;
  $("#status-badge").textContent = label;
  const t = r.timings || {};
  const tok = r.tokens || {};
  $("#answer-meta").textContent = t.total !== undefined
    ? `${t.total.toFixed(1)} s · ${(tok.rerank + tok.answer).toLocaleString()} tokens`
    : "";
  $("#answer-text").innerHTML = formatAnswer(r.answer);
  $("#blocked-note").hidden = !r.blocked_reason;
  $("#blocked-note").textContent = r.blocked_reason
    ? `The model's reply was replaced because it ${r.blocked_reason.startsWith("no") ? "had " : ""}${r.blocked_reason}.`
    : "";

  renderPipeline(r);
  renderSources(r.sent || []);
  renderCandidates(r.candidates || []);
  $("#result").hidden = false;
}

// Bold (**x**) and clickable citations like "(nvidia-1.pdf, pp. 26-27)". Text is escaped first.
const bold = (text) => esc(text).replace(/\*\*(.+?)\*\*/g, "<strong>$1</strong>");

function formatAnswer(text) {
  let html = bold(text);
  html = html.replace(/\(([^()]*?\b(?:pp?|secs?)\.\s*\d[^()]*)\)/g,
    (m, inner) => `<button type="button" class="cite" data-cite="${inner}">(${inner})</button>`);
  return html;
}

$("#answer-text").addEventListener("click", (e) => {
  const btn = e.target.closest(".cite");
  if (!btn || !lastResult) return;
  const cited = parseCitation(btn.dataset.cite, lastResult.sent);
  const cards = [...document.querySelectorAll(".source")].filter((card) => {
    const c = lastResult.sent[Number(card.dataset.index)];
    return cited.some((x) => (!x.file || x.file === c.source) && x.n >= c.page_start && x.n <= c.page_end);
  });
  cards.forEach((card) => {
    card.classList.add("flash");
    setTimeout(() => card.classList.remove("flash"), 1800);
  });
  if (cards[0]) cards[0].scrollIntoView({ behavior: "smooth", block: "center" });
});

// "nvidia-1.pdf, p. 3; pp. 26-27" -> [{file, n}, ...]; a location without a file uses the last file named.
function parseCitation(inner, sent) {
  const out = [];
  let file = sent.length && new Set(sent.map((c) => c.source)).size === 1 ? sent[0].source : null;
  const re = /([^\s(),;]+\.(?:pdf|docx))|\b(?:pp?|secs?)\.\s*([\d\s,–-]+)/gi;
  let m;
  while ((m = re.exec(inner.replace(/&amp;/g, "&")))) {
    if (m[1]) { file = m[1]; continue; }
    for (const part of m[2].split(",")) {
      const [a, b] = part.split(/[–-]/).map((s) => parseInt(s, 10));
      if (Number.isNaN(a)) continue;
      for (let n = a; n <= (Number.isNaN(b) || b === undefined ? a : b); n++) out.push({ file, n });
    }
  }
  return out;
}

function renderPipeline(r) {
  const c = r.counts || { candidates: 0, kept: 0, sent: 0 };
  const t = r.timings || {};
  const tok = r.tokens || {};
  const docs = new Set((r.candidates || []).map((x) => x.source)).size;
  const steps = [
    ["Search", c.candidates, `candidates from ${docs} document${docs === 1 ? "" : "s"} · vector, keyword, generated questions`, t.search],
    ["Thresholds", c.kept, `passed a score threshold`, null],
    ["Re-rank", c.sent, `chosen by the re-ranker${tok.rerank ? ` · ${tok.rerank.toLocaleString()} tokens` : ""}`, t.rerank],
    ["Answer", r.status === "answered" ? "✓" : "–", tok.answer ? `${tok.answer.toLocaleString()} tokens` : "model not called", t.answer],
  ];
  $("#pipeline").innerHTML = steps.map(([label, value, detail, secs]) => `
    <li class="step">
      <div class="step-label">${label}${secs !== null && secs !== undefined ? ` · ${secs.toFixed(1)} s` : ""}</div>
      <div class="step-value">${esc(value)}</div>
      <div class="step-detail">${esc(detail)}</div>
    </li>`).join("");
}

const FOUND_LABELS = { vector: "vector", keyword: "keyword", question: "gen. question" };

function scoreSpans(c) {
  const th = settings?.thresholds || {};
  const span = (label, v, limit, d) =>
    `<span class="${v !== null && limit !== undefined && v >= limit ? "pass" : ""}" title="${label}">${label} ${fmt(v, d)}</span>`;
  return span("sim", c.similarity, th.similarity, 3) + span("bm25", c.bm25, th.bm25, 1) +
    span("q-sim", c.question_similarity, th.question_similarity, 3);
}

function chunkBody(c) {
  return c.type === "table" ? c.text : c.text.replace(/\s*\n\s*/g, " ");
}

function renderSources(sent) {
  if (!sent.length) {
    $("#sources").innerHTML = `<p class="empty-note">No excerpt was sent to the answer model, so it wasn't called.</p>`;
    return;
  }
  $("#sources").innerHTML = sent.map((c, i) => `
    <div class="source" data-index="${i}">
      <div class="source-head">
        <span class="rank">#${c.picked_rank}</span>
        <span class="source-name">${esc(c.source)}</span>
        <span class="muted">${esc(c.location)}</span>
        <span class="badge">${esc(c.type)}</span>
        ${c.found_by.map((f) => `<span class="badge accent">${esc(FOUND_LABELS[f] || f)}</span>`).join("")}
        <span class="scores">${scoreSpans(c)}</span>
      </div>
      ${c.section_titles ? `<div class="matched">Section: <b>${esc(c.section_titles)}</b></div>` : ""}
      ${c.matched_question ? `<div class="matched">Closest generated question: <b>${esc(c.matched_question)}</b></div>` : ""}
      <div class="chunk-text ${c.type === "table" ? "table" : ""}">${esc(chunkBody(c))}</div>
      <button type="button" class="link-btn" data-toggle>Show full text</button>
    </div>`).join("");
}

$("#sources").addEventListener("click", (e) => {
  const btn = e.target.closest("[data-toggle]");
  if (!btn) return;
  const body = btn.previousElementSibling;
  body.classList.toggle("open");
  btn.textContent = body.classList.contains("open") ? "Show less" : "Show full text";
});

function renderCandidates(cands) {
  const th = settings?.thresholds || {};
  const order = (c) => (c.picked_rank ? c.picked_rank : c.kept ? 100 : 200);
  const rows = [...cands].sort((a, b) => order(a) - order(b) || b.similarity - a.similarity);
  const cell = (v, limit, d) => `<td class="num ${v !== null && v >= limit ? "pass" : ""}">${fmt(v, d)}</td>`;
  $("#candidates-table tbody").innerHTML = rows.map((c, i) => {
    const status = c.picked_rank ? `<span class="badge good">sent #${c.picked_rank}</span>`
      : c.kept ? `<span class="badge">kept, not chosen</span>` : `<span class="badge">below thresholds</span>`;
    return `
      <tr class="row ${c.kept ? "" : "dropped"}" data-row="${i}">
        <td>${status}</td><td>${esc(c.source)}</td><td>${esc(c.location)}</td><td>${esc(c.type)}</td>
        ${cell(c.similarity, th.similarity, 3)}${cell(c.bm25, th.bm25, 1)}${cell(c.question_similarity, th.question_similarity, 3)}
        <td>${c.found_by.map((f) => esc(FOUND_LABELS[f] || f)).join(", ")}</td>
      </tr>
      <tr class="detail" data-detail="${i}" hidden><td colspan="8">
        ${c.matched_question ? `<div class="matched">Closest generated question: <b>${esc(c.matched_question)}</b></div>` : ""}
        <pre>${esc(c.text)}</pre>
      </td></tr>`;
  }).join("");
  const kept = cands.filter((c) => c.kept).length;
  $("#candidates-summary").textContent = `${cands.length} found · ${kept} passed thresholds · ${cands.filter((c) => c.picked_rank).length} sent`;
  $("#threshold-legend").textContent = settings
    ? `A chunk passes if any score reaches its threshold (shown in green): similarity ≥ ${th.similarity}, BM25 ≥ ${th.bm25}, or generated-question similarity ≥ ${th.question_similarity}. Click a row to see its text.`
    : "";
}

$("#candidates-table tbody").addEventListener("click", (e) => {
  const row = e.target.closest("tr.row");
  if (!row) return;
  const detail = $(`#candidates-table tr[data-detail="${row.dataset.row}"]`);
  detail.hidden = !detail.hidden;
});

// ---------- documents ----------
async function loadDocuments() {
  try {
    settings = await (await fetch("/api/documents")).json();
  } catch {
    return;
  }
  $("#char-max").textContent = settings.max_question_chars;
  questionBox.maxLength = settings.max_question_chars;
  const docs = settings.documents;
  $("#brand-sub").textContent = `${docs.length} document${docs.length === 1 ? "" : "s"} · ${settings.settings["Chat model"]}`;
  $("#docs").innerHTML = docs.length ? docs.map((d) => `
    <div class="card">
      <div class="doc-name">${esc(d.name)}</div>
      <span class="badge">${d.name.toLowerCase().endsWith(".docx") ? "Word" : "PDF"}</span>
      <div class="doc-stats">
        <div><b>${d.units}</b>${d.unit === "section" ? "sections" : "pages"} with text</div>
        <div><b>${d.chunks}</b>chunks</div>
        <div><b>${d.tables}</b>table chunks</div>
        <div><b>${d.questions.toLocaleString()}</b>generated questions</div>
      </div>
    </div>`).join("") : `<div class="card muted">No documents indexed. Put PDFs or Word files in data/ and run python ingest.py.</div>`;
  $("#settings").innerHTML = Object.entries(settings.settings)
    .map(([k, v]) => `<dt>${esc(k)}</dt><dd>${esc(v)}</dd>`).join("");
}

// ---------- evaluation ----------
let benchmarkLoaded = false;
let evalFilter = "all";

async function loadBenchmark() {
  if (benchmarkLoaded) return;
  const data = await (await fetch("/api/benchmark")).json();
  benchmarkLoaded = true;
  if (!data.results) { $("#eval-empty").hidden = false; return; }
  $("#eval-content").hidden = false;
  $("#eval-meta").textContent = `${data.mode} · answers by ${data.answer_model}, judged by ${data.judge_model} · ${data.run_at}`;
  renderBenchmark(data.results);
}

function renderBenchmark(results) {
  const answerable = results.filter((r) => r.evidence.length);
  const unanswerable = results.filter((r) => r.category === "unanswerable");
  const adversarial = results.filter((r) => r.category === "adversarial");
  const count = (rs, f) => rs.filter(f).length;
  const sent = results.filter((r) => r.n_chunks);
  const mrr = answerable.reduce((s, r) => s + (r.rank ? 1 / r.rank : 0), 0) / (answerable.length || 1);
  const tiles = [
    ["Correct", count(answerable, (r) => r.correct), answerable.length, "answerable questions"],
    ["Answer retrieved", count(answerable, (r) => r.rank), answerable.length, `MRR ${mrr.toFixed(2)}`],
    ["Right page cited", count(answerable, (r) => r.citation), answerable.length, "answerable questions"],
    ["Declined", count(unanswerable, (r) => r.correct), unanswerable.length, "unanswerable questions"],
    ["Handled safely", count(adversarial, (r) => r.correct), adversarial.length, "prompt-injection attempts"],
    ["Faithful", count(results, (r) => r.faithful), results.length, "no unsupported claims"],
  ];
  const avgTokens = sent.length ? Math.round(sent.reduce((s, r) => s + (r.context_tokens || 0) + (r.rerank_tokens || 0), 0) / sent.length) : 0;
  $("#tiles").innerHTML = tiles.map(([label, n, d, sub]) => `
    <div class="tile"><div class="tile-value">${pct(n, d)}</div><div class="tile-label">${label} · ${n}/${d}</div><div class="tile-sub">${sub}</div></div>`).join("") +
    `<div class="tile"><div class="tile-value">${avgTokens.toLocaleString()}</div><div class="tile-label">tokens per question</div><div class="tile-sub">re-ranker + context excerpts</div></div>`;

  const cats = {};
  results.forEach((r) => (cats[r.category] ||= []).push(r));
  $("#cat-table tbody").innerHTML = Object.entries(cats).map(([cat, rs]) => `
    <tr><td>${esc(cat)}</td><td class="num">${rs.length}</td>
    <td class="num">${rs[0].evidence.length ? `${count(rs, (r) => r.rank)}/${rs.length}` : "–"}</td>
    <td class="num">${count(rs, (r) => r.correct)}/${rs.length}</td></tr>`).join("");

  const filters = [["all", "All"], ["fail", `Failures (${count(results, (r) => !r.correct || !r.faithful)})`],
    ...Object.keys(cats).map((c) => [c, c])];
  $("#eval-filters").innerHTML = filters.map(([key, label]) =>
    `<button type="button" class="chip ${key === evalFilter ? "active" : ""}" data-filter="${key}">${esc(label)}</button>`).join("");
  $("#eval-filters").onclick = (e) => {
    const key = e.target.dataset.filter;
    if (!key) return;
    evalFilter = key;
    renderBenchmark(results);
  };

  const shown = results.filter((r) => evalFilter === "all" ? true
    : evalFilter === "fail" ? !r.correct || !r.faithful : r.category === evalFilter);
  $("#eval-list").innerHTML = shown.map((r) => {
    const ok = r.correct && r.faithful;
    const retrieval = r.evidence.length ? (r.rank ? `answer chunk sent at rank ${r.rank}` : "answer chunk not sent") : "";
    return `
      <details class="q-row">
        <summary>
          <span class="badge ${ok ? "good" : "bad"}">${ok ? "pass" : "fail"}</span>
          <span class="q-id">#${r.id}</span>
          <span class="q-text" title="${esc(r.question)}">${esc(r.question)}</span>
          <span class="badge">${esc(r.category)}</span>
        </summary>
        <div class="q-body">
          <div><div class="label">Answer</div><div class="value">${bold(r.answer)}</div></div>
          ${r.question.length > 160 ? `<div><div class="label">Question</div><div class="value">${esc(r.question)}</div></div>` : ""}
          <div><div class="label">Expected</div><div class="value">${esc(r.expected)}</div></div>
          ${r.reason ? `<div><div class="label">Judge</div><div class="value">${esc(r.reason)}</div></div>` : ""}
          <div class="muted small">${[retrieval, r.evidence.length ? `citation ${r.citation ? "includes a right page" : "misses the right page"}` : "",
            r.faithful ? "" : "unsupported claim", `${r.n_chunks} excerpt${r.n_chunks === 1 ? "" : "s"} sent`].filter(Boolean).join(" · ")}</div>
          ${r.sources?.length ? `<div class="muted small">Sources: ${esc(r.sources.join("; "))}</div>` : ""}
        </div>
      </details>`;
  }).join("");
}

loadDocuments();

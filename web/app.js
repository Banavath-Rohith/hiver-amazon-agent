/* ═══════════════════════════════════════════════════════════════
   AmazonHelp AI Agent — Interactive Application Logic
   ═══════════════════════════════════════════════════════════════ */

'use strict';

// ─── Tab Navigation ───────────────────────────────────────────────────────────

document.querySelectorAll('.tab-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    const tabId = btn.dataset.tab;
    document.querySelectorAll('.tab-btn').forEach(b => b.classList.remove('active'));
    document.querySelectorAll('.tab-panel').forEach(p => p.classList.remove('active'));
    btn.classList.add('active');
    document.getElementById(`panel-${tabId}`).classList.add('active');

    // Lazy-load tab data on first open
    if (tabId === 'metrics' && !window._metricsLoaded) { loadMetrics(); window._metricsLoaded = true; }
    if (tabId === 'golden'  && !window._goldenLoaded)  { loadGolden();  window._goldenLoaded  = true; }
    if (tabId === 'failures'&& !window._failuresLoaded){ loadFailures(); window._failuresLoaded= true; }
  });
});

// ─── Playground ───────────────────────────────────────────────────────────────

const msgInput      = document.getElementById('msg-input');
const charCount     = document.getElementById('char-count');
const btnAnalyze    = document.getElementById('btn-analyze');
const loadingBar    = document.getElementById('loading-bar');
const pipeline      = document.getElementById('pipeline');

msgInput.addEventListener('input', () => {
  charCount.textContent = `${msgInput.value.length}/280`;
});

// Preset buttons
document.querySelectorAll('.preset-btn').forEach(btn => {
  btn.addEventListener('click', () => {
    msgInput.value = btn.dataset.msg;
    charCount.textContent = `${msgInput.value.length}/280`;
    msgInput.focus();
  });
});

btnAnalyze.addEventListener('click', analyzeMessage);

msgInput.addEventListener('keydown', e => {
  if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) analyzeMessage();
});

async function analyzeMessage() {
  const message = msgInput.value.trim();
  if (!message) {
    msgInput.focus();
    return;
  }

  // Show loading
  btnAnalyze.disabled = true;
  loadingBar.classList.add('active');
  pipeline.style.display = 'none';

  try {
    const resp   = await fetch('/api/classify', {
      method:  'POST',
      headers: { 'Content-Type': 'application/json' },
      body:    JSON.stringify({ message }),
    });
    const data = await resp.json();

    if (data.error) throw new Error(data.error);

    renderPipeline(data);
    pipeline.style.display = 'flex';
    pipeline.scrollIntoView({ behavior: 'smooth', block: 'nearest' });

  } catch (err) {
    alert('Agent error: ' + err.message + '\n\nMake sure web/server.py is running.');
  } finally {
    btnAnalyze.disabled = false;
    loadingBar.classList.remove('active');
  }
}

// ─── Pipeline Render ──────────────────────────────────────────────────────────

function renderPipeline(data) {
  // Step 1: Intent
  const intentBadge     = document.getElementById('intent-badge');
  const confidenceFill  = document.getElementById('confidence-fill');
  const confidenceVal   = document.getElementById('confidence-val');
  const keywordChips    = document.getElementById('keyword-chips');

  intentBadge.textContent = data.intent.replace(/_/g, ' ');

  const pct = Math.round((data.confidence || 0) * 100);
  setTimeout(() => { confidenceFill.style.width = pct + '%'; }, 50);
  confidenceVal.textContent = pct + '%';

  keywordChips.innerHTML = '';
  (data.triggered || []).forEach(kw => {
    const chip = document.createElement('span');
    chip.className = 'keyword-chip';
    chip.textContent = kw;
    keywordChips.appendChild(chip);
  });

  if (!data.triggered || data.triggered.length === 0) {
    const chip = document.createElement('span');
    chip.className = 'keyword-chip';
    chip.style.opacity = '0.4';
    chip.textContent = 'none';
    keywordChips.appendChild(chip);
  }

  // Step 2: Evidence
  const evidenceCards    = document.getElementById('evidence-cards');
  const evidenceCountBadge = document.getElementById('evidence-count-badge');
  const evidence         = data.evidence || [];

  evidenceCountBadge.textContent = `${evidence.length} case${evidence.length !== 1 ? 's' : ''}`;
  evidenceCards.innerHTML = '';

  if (evidence.length === 0) {
    evidenceCards.innerHTML = '<p style="color:var(--text-muted);font-size:13px">No similar historical cases found.</p>';
  } else {
    evidence.forEach((ev, i) => {
      const card = document.createElement('div');
      card.className = 'evidence-card';
      card.innerHTML = `
        <div class="evidence-header">
          <span class="evidence-rank">Case #${i+1}</span>
          <span class="evidence-sim">sim=${(ev.similarity * 100).toFixed(1)}%</span>
        </div>
        <div class="evidence-label">Customer message</div>
        <div class="evidence-msg">${escHtml(ev.message || '')}</div>
        <div class="evidence-label">AmazonHelp historical response</div>
        <div class="evidence-resp">${escHtml(ev.support_response || '(no response)')}</div>
      `;
      evidenceCards.appendChild(card);
    });
  }

  // Step 3: Reply
  const replyBox = document.getElementById('reply-box');
  replyBox.textContent = data.reply || '(no reply generated)';

  // Step 4: Escalation
  const actionBadge = document.getElementById('action-badge');
  const reasonBox   = document.getElementById('reason-box');

  actionBadge.textContent = data.action || '—';
  actionBadge.dataset.action = data.action || '';
  actionBadge.className = 'step-badge action-badge';

  reasonBox.textContent   = data.reason || '';
  reasonBox.dataset.action = data.action || '';
  reasonBox.className = 'reason-box';
}

// ─── Metrics Tab ──────────────────────────────────────────────────────────────

async function loadMetrics() {
  try {
    const resp = await fetch('/api/baselines');
    const data = await resp.json();
    renderComparison(data);
    renderPerClassChart(data.agent);
    renderEscalation(data.agent);
    renderConfusionMatrix(data.agent);
  } catch (err) {
    document.getElementById('comparison-tbody').innerHTML =
      `<tr><td colspan="4" class="loading-cell">Error loading metrics: ${err.message}</td></tr>`;
  }
}

function renderComparison(data) {
  const trivial = data.trivial || {};
  const simple  = data.simple  || {};
  const agent   = data.agent   || {};

  const ti = trivial.intent || {};
  const si = simple.intent  || {};
  const ai = agent.intent_metrics || {};

  const ta = trivial.action || {};
  const sa = simple.action  || {};
  const ae = agent.escalation_metrics || {};

  const rows = [
    ['Intent Accuracy',    fmt(ti.accuracy),    fmt(si.accuracy),    fmt(ai.accuracy)],
    ['Intent Macro F1',    fmt(ti.macro_f1),    fmt(si.macro_f1),    fmt(ai.macro_f1)],
    ['Intent Macro Prec',  fmt(ti.macro_precision), fmt(si.macro_precision), fmt(ai.macro_precision)],
    ['Intent Macro Recall',fmt(ti.macro_recall), fmt(si.macro_recall), fmt(ai.macro_recall)],
    ['Action Accuracy',    fmt(ta.accuracy),    fmt(sa.accuracy),    fmt(ae.accuracy)],
    ['Escalation F1',      fmt(ta.macro_f1),    fmt(sa.macro_f1),    fmt(ae.f1)],
    ['False Escalation',   '—',                 '—',                 fmtPct(ae.false_escalation_rate)],
    ['Missed Escalation',  '—',                 '—',                 fmtPct(ae.missed_escalation_rate)],
  ];

  const tbody = document.getElementById('comparison-tbody');
  tbody.innerHTML = rows.map(([label, t, s, a]) => `
    <tr>
      <td>${label}</td>
      <td>${t}</td>
      <td>${s}</td>
      <td class="highlight-col">${a}</td>
    </tr>
  `).join('');
}

function renderPerClassChart(agent) {
  const pc   = (agent && agent.intent_metrics && agent.intent_metrics.per_class) || {};
  const chart = document.getElementById('per-class-chart');
  chart.innerHTML = '';

  const shortLabel = {
    delivery_or_order:   'Delivery / Order',
    other_or_unclear:    'Other / Unclear',
    account_or_login:    'Account / Login',
    payment_or_refund:   'Payment / Refund',
    product_or_service:  'Product / Service',
    technical_issue:     'Technical Issue',
  };

  Object.entries(pc).sort((a,b) => b[1].f1 - a[1].f1).forEach(([label, vals]) => {
    const pct = Math.round((vals.f1 || 0) * 100);
    const div = document.createElement('div');
    div.className = 'bar-row';
    div.innerHTML = `
      <span class="bar-label">${shortLabel[label] || label}</span>
      <div class="bar-track"><div class="bar-fill" style="width:0%"></div></div>
      <span class="bar-val">${pct}%</span>
    `;
    chart.appendChild(div);
    // Animate
    setTimeout(() => { div.querySelector('.bar-fill').style.width = pct + '%'; }, 100);
  });
}

function renderEscalation(agent) {
  const e    = (agent && agent.escalation_metrics) || {};
  const grid = document.getElementById('esc-grid');
  const cells = [
    { val: fmtPct(e.accuracy),               label: 'Accuracy',          cls: 'neutral' },
    { val: fmtPct(e.f1),                     label: 'F1 Score',          cls: 'neutral' },
    { val: fmtPct(e.false_escalation_rate),  label: 'False Escalation',  cls: 'good'    },
    { val: fmtPct(e.missed_escalation_rate), label: 'Missed Escalation', cls: 'neutral' },
  ];
  grid.innerHTML = cells.map(c => `
    <div class="esc-cell ${c.cls}">
      <div class="esc-val">${c.val}</div>
      <div class="esc-label">${c.label}</div>
    </div>
  `).join('');
}

function renderConfusionMatrix(agent) {
  const cm     = (agent && agent.confusion_matrix) || {};
  const labels = Object.keys(cm);
  if (!labels.length) return;

  const shortL = {
    delivery_or_order:   'delivery',
    other_or_unclear:    'other',
    account_or_login:    'account',
    payment_or_refund:   'payment',
    product_or_service:  'product',
    technical_issue:     'technical',
  };

  const wrap = document.getElementById('confusion-matrix');
  let html = '<table class="cm-table"><thead><tr><th>True \\ Pred</th>';
  labels.forEach(l => { html += `<th>${shortL[l] || l}</th>`; });
  html += '</tr></thead><tbody>';

  labels.forEach(trueL => {
    html += `<tr><td class="cm-row-label">${shortL[trueL] || trueL}</td>`;
    labels.forEach(predL => {
      const val = (cm[trueL] && cm[trueL][predL]) || 0;
      const isDiag = trueL === predL;
      html += `<td class="${isDiag ? 'cm-diag' : (val === 0 ? 'cm-zero' : '')}">${val}</td>`;
    });
    html += '</tr>';
  });

  html += '</tbody></table>';
  wrap.innerHTML = html;
}

// ─── Golden Set Tab ───────────────────────────────────────────────────────────

let _allGoldenRows = [];

async function loadGolden() {
  try {
    const resp = await fetch('/api/golden');
    _allGoldenRows = await resp.json();
    renderGolden(_allGoldenRows);
    setupGoldenFilters();
  } catch (err) {
    document.getElementById('golden-tbody').innerHTML =
      `<tr><td colspan="5" class="loading-cell">Error: ${err.message}</td></tr>`;
  }
}

const INTENT_CLASS = {
  delivery_or_order:  'intent-delivery',
  account_or_login:   'intent-account',
  payment_or_refund:  'intent-payment',
  product_or_service: 'intent-product',
  technical_issue:    'intent-technical',
  other_or_unclear:   'intent-other',
};

function renderGolden(rows) {
  const tbody = document.getElementById('golden-tbody');
  document.getElementById('golden-count').textContent = `${rows.length} rows`;

  if (!rows.length) {
    tbody.innerHTML = '<tr><td colspan="5" class="loading-cell">No matching rows.</td></tr>';
    return;
  }

  tbody.innerHTML = rows.slice(0, 100).map((r, i) => `
    <tr>
      <td>${i + 1}</td>
      <td class="td-msg">${escHtml((r.message || '').substring(0, 120))}${r.message && r.message.length > 120 ? '…' : ''}</td>
      <td><span class="intent-tag ${INTENT_CLASS[r.intent] || 'intent-other'}">${r.intent || '—'}</span></td>
      <td><span class="action-tag ${r.expected_action === 'AUTO' ? 'action-auto' : 'action-escalate'}">${r.expected_action || '—'}</span></td>
      <td class="td-reason">${escHtml((r.expected_reason || '—').substring(0, 90))}${(r.expected_reason||'').length>90?'…':''}</td>
    </tr>
  `).join('');
}

function setupGoldenFilters() {
  const search  = document.getElementById('golden-search');
  const intent  = document.getElementById('golden-intent-filter');
  const action  = document.getElementById('golden-action-filter');

  function applyFilters() {
    const q = search.value.toLowerCase();
    const i = intent.value;
    const a = action.value;
    const filtered = _allGoldenRows.filter(r => {
      const matchQ = !q || (r.message || '').toLowerCase().includes(q);
      const matchI = !i || r.intent === i;
      const matchA = !a || r.expected_action === a;
      return matchQ && matchI && matchA;
    });
    renderGolden(filtered);
  }

  search.addEventListener('input',  applyFilters);
  intent.addEventListener('change', applyFilters);
  action.addEventListener('change', applyFilters);
}

// ─── Failures Tab ─────────────────────────────────────────────────────────────

async function loadFailures() {
  try {
    const resp  = await fetch('/api/failures');
    const rows  = await resp.json();
    const cards = document.getElementById('failure-cards');

    if (!rows.length) {
      cards.innerHTML = '<p class="loading-cell">No failures found. Run src/generate_failure_analysis.py first.</p>';
      return;
    }

    cards.innerHTML = rows.map((r, i) => `
      <div class="failure-card">
        <div class="failure-type">${r.failure_type || 'Unknown failure'}</div>
        <div class="failure-row">
          <div class="failure-field">
            <label>Customer message</label>
            <p class="failure-msg">${escHtml((r.customer_message || '').substring(0, 200))}</p>
          </div>
          <div class="failure-field">
            <label>Expected intent</label>
            <p><span class="intent-tag ${INTENT_CLASS[r.expected_intent] || 'intent-other'}">${r.expected_intent || '—'}</span></p>
            <label style="margin-top:8px">Predicted intent</label>
            <p><span class="intent-tag ${INTENT_CLASS[r.predicted_intent] || 'intent-other'}">${r.predicted_intent || '—'}</span></p>
          </div>
        </div>
        <div class="failure-hyp-label">Root-cause hypothesis</div>
        <p class="failure-hyp">${escHtml(r.hypothesis || '—')}</p>
      </div>
    `).join('');

  } catch (err) {
    document.getElementById('failure-cards').innerHTML =
      `<p class="loading-cell">Error loading failures: ${err.message}</p>`;
  }
}

// ─── Utils ────────────────────────────────────────────────────────────────────

function fmt(val) {
  if (val === undefined || val === null || val === '') return '—';
  return (val * 100).toFixed(1) + '%';
}

function fmtPct(val) {
  if (val === undefined || val === null) return '—';
  return (val * 100).toFixed(1) + '%';
}

function escHtml(str) {
  return (str || '').replace(/&/g,'&amp;').replace(/</g,'&lt;').replace(/>/g,'&gt;').replace(/"/g,'&quot;');
}

// ─── Load metrics on page load if already on metrics tab ─────────────────────
window.addEventListener('load', () => {
  // Auto-load metrics in background for stat pills
  fetch('/api/metrics').then(r => r.json()).then(data => {
    const ai = data.intent_metrics || {};
    const esc = data.escalation_metrics || {};
    const acc = document.getElementById('stat-accuracy');
    const f1  = document.getElementById('stat-f1');
    const esc_el = document.getElementById('stat-esc');
    if (acc && ai.accuracy !== undefined) acc.textContent = (ai.accuracy * 100).toFixed(1) + '%';
    if (f1  && ai.macro_f1 !== undefined) f1.textContent  = (ai.macro_f1  * 100).toFixed(1) + '%';
    if (esc_el && esc.false_escalation_rate !== undefined)
      esc_el.textContent = (esc.false_escalation_rate * 100).toFixed(1) + '%';
  }).catch(() => {});
});

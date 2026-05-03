/* Affective Events Review — single-page client.
 *
 * Architecture: a tiny state object + pure render functions. The DOM is the
 * source of truth for transient form input; `state` is the source of truth
 * for everything else. Auto-save triggers on debounced form change.
 */

(() => {
'use strict';

// ---------------------------------------------------------------------------
// State
// ---------------------------------------------------------------------------

const state = {
    session: null,        // /api/session payload
    signals: null,        // {arousal: [...], valence: [...], dominance: [...]}
    signalsMeta: null,    // {arousal: {hop_sec, window_sec, n_frames}, ...}
    waveform: null,       // {min, max, n_peaks, sample_rate, duration_sec}
    windowedWaveform: null, // high-res waveform for current view (from /api/waveform?t0=&t1=)
    eventsById: {},       // id -> event
    filtered: [],         // ordered list of event ids after filter+sort
    currentIndex: 0,      // index into filtered
    filter: { signal: '', type: '', minPeakZ: 0, unlabeledOnly: false },
    sort: 'time',
    contextZoom: 30,      // view window in seconds (10, 30, 60, 120, 300), 0 = full
    playheadSec: 0,
    lastEventEnteredId: null,
    knownTags: new Set(),
    panels: null,         // built by buildPanelConfig()
    signalNames: [],      // ordered signal names
};

const VERDICTS = ['tp', 'fp', 'unclear', 'partial'];
const REVIEW_AUDIO_PAD_SEC = 8;

// ---------------------------------------------------------------------------
// Dynamic panel config — derived from loaded signals at runtime
// ---------------------------------------------------------------------------

const KNOWN_ORDER = ['arousal', 'valence', 'dominance'];
const PALETTE = ['#2563eb', '#ea580c', '#16a34a', '#8b5cf6', '#0891b2', '#db2777'];
const WAVEFORM_FRAC = 0.22;
const PANEL_GAP = 0.02;

function resolveSignalNames(signalsMeta, loadedSignals) {
    const metaKeys = new Set(Object.keys(signalsMeta));
    const loadedKeys = new Set(Object.keys(loadedSignals));
    const valid = [...metaKeys].filter(k => {
        if (!loadedKeys.has(k)) { console.warn(`signal "${k}" in meta but missing from data — skipped`); return false; }
        return true;
    });
    for (const k of loadedKeys) {
        if (!metaKeys.has(k)) console.warn(`signal "${k}" in data but missing from meta — skipped`);
    }
    const known = KNOWN_ORDER.filter(k => valid.includes(k));
    const extras = valid.filter(k => !KNOWN_ORDER.includes(k)).sort();
    return [...known, ...extras];
}

function buildPanelConfig(signalNames) {
    const n = signalNames.length;
    if (n === 0) {
        return { _waveform: { domain: [0, 1], axis: 'y', color: '#888' } };
    }
    const availableHeight = 1 - WAVEFORM_FRAC - PANEL_GAP;
    const panelH = (availableHeight - PANEL_GAP * Math.max(0, n - 1)) / n;

    const panels = {};
    panels._waveform = { domain: [0, WAVEFORM_FRAC], axis: 'y', color: '#888' };

    const reversed = [...signalNames].reverse();
    reversed.forEach((name, i) => {
        const lo = WAVEFORM_FRAC + PANEL_GAP + i * (panelH + PANEL_GAP);
        panels[name] = {
            domain: [lo, lo + panelH],
            axis: `y${i + 2}`,
            color: PALETTE[signalNames.indexOf(name) % PALETTE.length],
        };
    });
    return panels;
}

// ---------------------------------------------------------------------------
// View-local y-axis ranges — computed from visible data for each render
// ---------------------------------------------------------------------------

function computeLocalSignalRanges(t0View, t1View) {
    const ranges = {};
    for (const name of state.signalNames) {
        const meta = state.signalsMeta[name];
        const arr = state.signals[name];
        if (!meta || !arr || !arr.length) { ranges[name] = [0, 1]; continue; }

        const dt = meta.hop_sec;
        const offset = meta.window_sec / 2;
        const i0 = Math.max(0, Math.floor((t0View - offset) / dt));
        const i1 = Math.min(arr.length - 1, Math.ceil((t1View - offset) / dt));

        let lo = Infinity, hi = -Infinity;
        for (let i = i0; i <= i1; i++) {
            const v = arr[i];
            if (!Number.isFinite(v)) continue;
            if (v < lo) lo = v;
            if (v > hi) hi = v;
        }
        if (!Number.isFinite(lo)) { ranges[name] = [0, 1]; continue; }

        const span = hi - lo;
        const margin = Math.max(span * 0.15, 0.02);
        ranges[name] = [lo - margin, hi + margin];
    }
    return ranges;
}

// ---------------------------------------------------------------------------
// API
// ---------------------------------------------------------------------------

async function apiGet(path) {
    const r = await fetch(path);
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
}

async function apiPost(path, body) {
    const r = await fetch(path, {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify(body),
    });
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
}

async function apiDelete(path) {
    const r = await fetch(path, { method: 'DELETE' });
    if (!r.ok) throw new Error(`${path}: ${r.status}`);
    return r.json();
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

const fmtTime = (sec) => {
    if (!isFinite(sec)) return '–';
    const m = Math.floor(sec / 60), s = sec - 60 * m;
    return `${m}:${s.toFixed(1).padStart(4, '0')}`;
};

const fmtNum = (v, d = 2) => (v == null ? '–' : Number(v).toFixed(d));

function signedPeakZ(ev) {
    if (!ev || ev.event_type === 'joint') return ev ? ev.peak_z : null;
    return ev.direction === '-' ? -ev.peak_z : ev.peak_z;
}

function eventContextStart(ev) {
    return Math.max(0, (ev ? ev.start_sec : 0) - REVIEW_AUDIO_PAD_SEC);
}

function blockIds(ev) {
    const ids = ev && ev.extra ? ev.extra.block_ids : null;
    return Array.isArray(ids) ? ids.join(',') : '';
}

function debounce(fn, ms) {
    let t = null;
    return function (...args) {
        clearTimeout(t);
        t = setTimeout(() => fn.apply(this, args), ms);
    };
}

function uniqueSorted(items) {
    return Array.from(new Set(items)).sort();
}

function frameTimes(meta) {
    const out = new Float64Array(meta.n_frames);
    const dt = meta.hop_sec, half = meta.window_sec / 2;
    for (let i = 0; i < meta.n_frames; i++) out[i] = i * dt + half;
    return out;
}

// ---------------------------------------------------------------------------
// Filter & sort
// ---------------------------------------------------------------------------

function applyFilterSort() {
    const events = state.session.events;
    const labels = state.session.labels || {};
    const f = state.filter;

    let list = events.filter(e => {
        if (f.signal && e.signal_name !== f.signal) return false;
        if (f.type && e.event_type !== f.type) return false;
        if ((e.peak_z || 0) < f.minPeakZ) return false;
        const lbl = labels[e.event_id];
        if (f.unlabeledOnly && lbl && lbl.verdict) return false;
        return true;
    });

    const cmp = {
        time:       (a, b) => a.start_sec - b.start_sec,
        peak_z:     (a, b) => (b.peak_z || 0) - (a.peak_z || 0),
        duration:   (a, b) => (b.duration_sec || 0) - (a.duration_sec || 0),
    }[state.sort] || ((a, b) => a.start_sec - b.start_sec);
    list.sort(cmp);

    state.filtered = list.map(e => e.event_id);
    if (state.currentIndex >= state.filtered.length) {
        state.currentIndex = Math.max(0, state.filtered.length - 1);
    }
}

function currentEvent() {
    const id = state.filtered[state.currentIndex];
    return id ? state.eventsById[id] : null;
}

// ---------------------------------------------------------------------------
// Plotly traces & shapes
// ---------------------------------------------------------------------------

function buildSignalTraces() {
    const traces = [];
    for (const name of state.signalNames) {
        const meta = state.signalsMeta[name];
        if (!meta) continue;
        const panel = state.panels[name];
        if (!panel) continue;
        const ts = frameTimes(meta);
        traces.push({
            type: 'scattergl',
            mode: 'lines',
            name,
            x: Array.from(ts),
            y: state.signals[name],
            xaxis: 'x',
            yaxis: panel.axis,
            line: { color: panel.color, width: 1.4 },
            hovertemplate: `${name}: %{y:.3f}<br>t=%{x:.2f}s<extra></extra>`,
            showlegend: false,
        });
    }
    return traces;
}

function buildWaveformTrace() {
    const wf = state.windowedWaveform || state.waveform;
    if (!wf || !wf.n_peaks) return null;
    const tStart = wf.t0_sec || 0;
    const dt = (wf.duration_sec || wf.duration_sec) / wf.n_peaks;
    const n = wf.n_peaks;
    const x = new Array(2 * n);
    const y = new Array(2 * n);
    for (let i = 0; i < n; i++) {
        const t = tStart + i * dt + dt / 2;
        x[i] = t;             y[i] = wf.max[i];
        x[2 * n - 1 - i] = t; y[2 * n - 1 - i] = wf.min[i];
    }
    const cs = getComputedStyle(document.documentElement);
    return {
        type: 'scatter',
        mode: 'lines',
        x, y,
        fill: 'toself',
        fillcolor: cs.getPropertyValue('--waveform-fill').trim() || 'rgba(120,120,120,0.25)',
        line: { color: cs.getPropertyValue('--waveform-line').trim() || 'rgba(100,100,100,0.5)', width: 0.5 },
        xaxis: 'x',
        yaxis: 'y',
        hoverinfo: 'skip',
        showlegend: false,
    };
}

function getThemeColors() {
    const cs = getComputedStyle(document.documentElement);
    const v = (name, fallback) => cs.getPropertyValue(name).trim() || fallback;
    return {
        eventCurrentFill:   v('--event-current-fill',   'rgba(251,146,60,0.18)'),
        eventCurrentBand:   v('--event-current-band',   'rgba(251,146,60,0.32)'),
        eventCurrentBorder: v('--event-current-border', 'rgba(234,88,12,0.8)'),
        eventCurrentEdge:   v('--event-current-edge',   'rgba(234,88,12,0.6)'),
        eventOtherFill:     v('--event-other-fill',     'rgba(160,160,180,0.10)'),
        eventPlayhead:      v('--event-playhead',       'rgba(0,0,0,0.6)'),
        eventSpeechBlock:   v('--event-speech-block',   'rgba(34,197,94,0.08)'),
        eventDetectionTint: v('--event-detection-tint', 'rgba(59,130,246,0.07)'),
        plotBg:             v('--plot-bg',    '#ffffff'),
        plotGrid:           v('--plot-grid',  '#eee'),
        plotZero:           v('--plot-zero',  '#ddd'),
        plotTick:           v('--plot-tick',  '#666'),
        plotTitle:          v('--plot-title', '#333'),
        plotHoverBg:        v('--plot-hover-bg', '#fff'),
        plotHoverFg:        v('--plot-hover-fg', '#333'),
    };
}

function buildShapes(ev) {
    const shapes = [];
    if (!ev) return shapes;

    const tc = getThemeColors();

    // (1) Highlight detection signal panel — full-width tinted rect.
    const detPanel = state.panels[ev.signal_name];
    if (detPanel) {
        shapes.push({
            type: 'rect',
            xref: 'paper', yref: 'paper',
            x0: 0, x1: 1,
            y0: detPanel.domain[0], y1: detPanel.domain[1],
            fillcolor: tc.eventDetectionTint,
            line: { width: 0 },
            layer: 'below',
        });
    }

    // (2) Speech-block bands.
    const viewRange = getViewRange(ev);
    for (const b of state.session.blocks) {
        if (b.end_sec < viewRange[0] || b.start_sec > viewRange[1]) continue;
        shapes.push({
            type: 'rect',
            xref: 'x', yref: 'paper',
            x0: b.start_sec, x1: b.end_sec,
            y0: 0, y1: 1,
            fillcolor: tc.eventSpeechBlock,
            line: { width: 0 },
            layer: 'below',
        });
    }

    // (3) Other events — gray, on their own signal panel only.
    for (const other of state.session.events) {
        if (other.event_id === ev.event_id) continue;
        if (other.event_type === 'joint' || other.event_type === 'affective_episode') continue;
        if (other.end_sec < viewRange[0] || other.start_sec > viewRange[1]) continue;
        const otherPanel = state.panels[other.signal_name];
        if (!otherPanel) continue;
        shapes.push({
            type: 'rect',
            xref: 'x', yref: 'paper',
            x0: other.start_sec, x1: other.end_sec,
            y0: otherPanel.domain[0], y1: otherPanel.domain[1],
            fillcolor: tc.eventOtherFill,
            line: { width: 0 },
            layer: 'below',
        });
    }

    // (4) Current event — amber band across full height + bold on signal panel.
    const eDomain = detPanel ? detPanel.domain : [0, 1];
    shapes.push({
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: ev.start_sec, x1: ev.end_sec,
        y0: 0, y1: 1,
        fillcolor: tc.eventCurrentFill,
        line: { width: 0 },
        layer: 'below',
    });
    shapes.push({
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: ev.start_sec, x1: ev.end_sec,
        y0: eDomain[0], y1: eDomain[1],
        fillcolor: tc.eventCurrentBand,
        line: { color: tc.eventCurrentBorder, width: 2 },
        layer: 'below',
    });
    // Vertical edge lines at event boundaries.
    for (const xEdge of [ev.start_sec, ev.end_sec]) {
        shapes.push({
            type: 'line',
            xref: 'x', yref: 'paper',
            x0: xEdge, x1: xEdge,
            y0: 0, y1: 1,
            line: { color: tc.eventCurrentEdge, width: 1.5, dash: 'dot' },
            layer: 'below',
        });
    }

    // (5) Playhead — thin vertical line spanning full plot.
    shapes.push({
        type: 'line',
        xref: 'x', yref: 'paper',
        x0: state.playheadSec, x1: state.playheadSec,
        y0: 0, y1: 1,
        line: { color: tc.eventPlayhead, width: 1.5 },
        layer: 'above',
    });
    return shapes;
}

function getViewRange(ev) {
    const dur = state.session.audio_duration_sec || 60;
    if (!ev) return [0, dur];
    const windowSec = state.contextZoom;
    if (windowSec === 0) return [0, dur];
    const center = (ev.start_sec + ev.end_sec) / 2;
    const half = windowSec / 2;
    return [Math.max(0, center - half), Math.min(dur, center + half)];
}

function buildLayout(ev) {
    const [t0, t1] = getViewRange(ev);
    const tc = getThemeColors();
    const localRanges = computeLocalSignalRanges(t0, t1);

    const baseAxis = (title, domain, range) => {
        const ax = {
            domain,
            title: { text: title, standoff: 4, font: { size: 11, color: tc.plotTitle } },
            zerolinecolor: tc.plotZero,
            gridcolor: tc.plotGrid,
            tickfont: { size: 10, color: tc.plotTick },
            fixedrange: true,
        };
        if (range) { ax.range = range; ax.autorange = false; }
        return ax;
    };

    const layout = {
        margin: { t: 14, b: 32, l: 56, r: 14 },
        showlegend: false,
        plot_bgcolor: tc.plotBg,
        paper_bgcolor: tc.plotBg,
        hovermode: 'x',
        dragmode: false,
        hoverlabel: { bgcolor: tc.plotHoverBg, font: { color: tc.plotHoverFg, size: 12 } },
        xaxis: {
            domain: [0, 1],
            range: [t0, t1],
            autorange: false,
            fixedrange: true,
            title: { text: 'time (s)', font: { size: 11, color: tc.plotTitle } },
            tickfont: { size: 10, color: tc.plotTick },
            gridcolor: tc.plotGrid,
        },
        yaxis: baseAxis('audio', state.panels._waveform.domain, undefined),
        shapes: buildShapes(ev),
    };

    for (const name of state.signalNames) {
        const panel = state.panels[name];
        if (!panel) continue;
        const axisKey = panel.axis === 'y' ? 'yaxis' : `yaxis${panel.axis.slice(1)}`;
        layout[axisKey] = baseAxis(name, panel.domain, localRanges[name]);
    }

    return layout;
}

// ---------------------------------------------------------------------------
// Render
// ---------------------------------------------------------------------------

function renderTopbar() {
    const sess = state.session;
    document.getElementById('recording-id').textContent = sess.recording_id;
    document.getElementById('config-hash').textContent  = `cfg=${sess.config_hash}`;
}

function renderEventMeta(ev) {
    const labels = state.session.labels || {};
    const lbl = ev ? labels[ev.event_id] : null;
    const pos = document.getElementById('event-position');
    const meta = document.getElementById('event-meta');

    if (!ev) {
        pos.textContent = '0 / 0';
        meta.innerHTML = '<span class="muted">no events match the current filter</span>';
        return;
    }

    pos.textContent = `${state.currentIndex + 1} / ${state.filtered.length}`;

    meta.innerHTML = [
        `<span class="pill signal">${ev.signal_name}</span>`,
        `<span class="pill">${ev.event_type}</span>`,
        `<span class="pill">${ev.direction}</span>`,
        `t=${fmtNum(ev.start_sec, 2)}–${fmtNum(ev.end_sec, 2)}s (${fmtNum(ev.duration_sec, 1)}s)`,
        `peak_z=${fmtNum(ev.peak_z, 2)}`,
        `signed_z=${fmtNum(signedPeakZ(ev), 2)}`,
        `peak_t=${fmtNum(ev.peak_time_sec, 2)}s`,
        `Δ=${fmtNum(ev.delta, 3)}`,
        `baseline=${fmtNum(ev.baseline_at_peak, 3)}`,
        `scale=${fmtNum(ev.scale_at_peak, 3)}`,
        `frames=${ev.frame_start}–${ev.frame_end}`,
        blockIds(ev) ? `block=${blockIds(ev)}` : '',
        ev.parent_id ? `parent=<code>${ev.parent_id}</code>` : '',
        ev.children && ev.children.length ? `children=${ev.children.length}` : '',
        lbl && lbl.inherited_from ? `<span class="pill" style="border-color:#7c3aed;color:#7c3aed">inherited (${fmtNum(lbl.inherited_match_score, 2)})</span>` : '',
    ].filter(Boolean).join(' &middot; ');
}

function renderLabelForm(ev) {
    const labels = state.session.labels || {};
    const lbl = ev ? labels[ev.event_id] : null;

    document.querySelectorAll('#verdict-group button').forEach(b => {
        b.classList.toggle('active', !!lbl && lbl.verdict === b.dataset.verdict);
    });
    document.getElementById('tags').value    = lbl && lbl.tags ? lbl.tags.join(', ') : '';
    document.getElementById('comment').value = lbl && lbl.comment ? lbl.comment : '';

    const meta = document.getElementById('label-meta');
    if (lbl && lbl.labeled_at) {
        let txt = `last labeled ${lbl.labeled_at}`;
        if (lbl.inherited_from) txt += ` · inherited from ${lbl.inherited_from}`;
        meta.textContent = txt;
    } else {
        meta.textContent = '';
    }
}

function renderMinimap() {
    const mm = document.getElementById('minimap');
    mm.innerHTML = '';
    const N = state.filtered.length;
    if (!N) return;
    const labels = state.session.labels || {};
    const w = mm.clientWidth || 1;
    const stepPx = w / Math.max(N, 1);
    const stride = N > 600 ? Math.ceil(N / 600) : 1;  // cap DOM size

    for (let i = 0; i < N; i += stride) {
        const ev = state.eventsById[state.filtered[i]];
        const lbl = labels[ev.event_id];
        const tick = document.createElement('div');
        tick.className = 'tick' + (lbl && lbl.verdict ? ' ' + lbl.verdict : '');
        tick.style.left = `${(i / N) * 100}%`;
        if (i === state.currentIndex) tick.classList.add('current');
        tick.dataset.index = i;
        tick.title = `#${i + 1} ${ev.signal_name} ${ev.event_type} t=${fmtNum(ev.start_sec, 1)}s` + (lbl && lbl.verdict ? ` [${lbl.verdict}]` : '');
        mm.appendChild(tick);
    }
}

function renderPlot(ev) {
    const traces = [
        buildWaveformTrace(),
        ...buildSignalTraces(),
    ].filter(Boolean);

    Plotly.react('plot', traces, buildLayout(ev), {
        displaylogo: false,
        responsive: true,
        scrollZoom: false,
        modeBarButtonsToRemove: [
            'select2d', 'lasso2d', 'autoScale2d',
            'zoomIn2d', 'zoomOut2d', 'zoom2d', 'pan2d', 'resetScale2d',
        ],
    });
}

function renderAll() {
    const ev = currentEvent();
    state.windowedWaveform = null;
    renderEventMeta(ev);
    renderLabelForm(ev);
    renderMinimap();
    renderPlot(ev);
    debouncedWaveformFetch();

    if (ev) {
        const audio = document.getElementById('audio');
        const target = eventContextStart(ev);
        // Avoid re-triggering buffering on tiny diffs
        if (Math.abs(audio.currentTime - target) > 0.5) {
            audio.currentTime = target;
        }
        state.playheadSec = audio.currentTime;
        state.lastEventEnteredId = null;  // re-arm entry flash for this event
    }
}

// ---------------------------------------------------------------------------
// Playhead update (called from <audio>.timeupdate)
// ---------------------------------------------------------------------------

let pendingPlayheadFrame = null;
function schedulePlayheadUpdate(t) {
    state.playheadSec = t;
    if (pendingPlayheadFrame !== null) return;
    pendingPlayheadFrame = requestAnimationFrame(() => {
        pendingPlayheadFrame = null;
        const shapes = (Plotly.d3 && document.getElementById('plot').layout && document.getElementById('plot').layout.shapes) || null;
        // Use Plotly.relayout with a precise shape index. The playhead is the last shape.
        const layout = document.getElementById('plot').layout;
        if (!layout || !layout.shapes || !layout.shapes.length) return;
        const idx = layout.shapes.length - 1;
        Plotly.relayout('plot', {
            [`shapes[${idx}].x0`]: state.playheadSec,
            [`shapes[${idx}].x1`]: state.playheadSec,
        });
    });
}

function checkEventEntry(audioTime) {
    const ev = currentEvent();
    if (!ev) return;
    if (audioTime >= ev.start_sec && audioTime <= ev.end_sec
        && state.lastEventEnteredId !== ev.event_id) {
        state.lastEventEnteredId = ev.event_id;
        const plot = document.getElementById('plot');
        plot.classList.remove('event-entered');
        // Force reflow so animation restarts
        // eslint-disable-next-line no-unused-expressions
        void plot.offsetWidth;
        plot.classList.add('event-entered');
    } else if (audioTime < ev.start_sec) {
        state.lastEventEnteredId = null;
    }
}

// ---------------------------------------------------------------------------
// Filter UI
// ---------------------------------------------------------------------------

function populateFilterUI() {
    const events = state.session.events;
    const signalNames = uniqueSorted(events.map(e => e.signal_name));
    const eventTypes  = uniqueSorted(events.map(e => e.event_type));
    const sigSel = document.getElementById('filter-signal');
    const typSel = document.getElementById('filter-type');
    sigSel.innerHTML = '<option value="">all</option>' + signalNames.map(n => `<option value="${n}">${n}</option>`).join('');
    typSel.innerHTML = '<option value="">all</option>' + eventTypes.map(n => `<option value="${n}">${n}</option>`).join('');

    sigSel.addEventListener('change', () => { state.filter.signal = sigSel.value; refresh(); });
    typSel.addEventListener('change', () => { state.filter.type   = typSel.value; refresh(); });

    const peakZ = document.getElementById('filter-peak-z');
    const peakZDisplay = document.getElementById('filter-peak-z-display');
    peakZ.addEventListener('input', () => {
        state.filter.minPeakZ = parseFloat(peakZ.value);
        peakZDisplay.textContent = state.filter.minPeakZ.toFixed(2);
        refresh();
    });

    const ulOnly = document.getElementById('filter-unlabeled');
    ulOnly.addEventListener('change', () => { state.filter.unlabeledOnly = ulOnly.checked; refresh(); });

    const sortSel = document.getElementById('sort-by');
    sortSel.addEventListener('change', () => { state.sort = sortSel.value; refresh(); });

    const zoomSel = document.getElementById('context-zoom');
    if (zoomSel) {
        zoomSel.addEventListener('change', () => {
            state.contextZoom = parseInt(zoomSel.value, 10);
            renderPlot(currentEvent());
            debouncedWaveformFetch();
        });
    }
}

function refresh() {
    applyFilterSort();
    document.getElementById('filter-count').textContent = `${state.filtered.length} events`;
    renderAll();
}

// ---------------------------------------------------------------------------
// Label save (debounced auto-save on form change)
// ---------------------------------------------------------------------------

function setSaveStatus(cls, text) {
    const el = document.getElementById('save-indicator');
    el.className = cls;
    el.textContent = text;
}

async function saveCurrentLabel() {
    const ev = currentEvent();
    if (!ev) return;

    const verdictBtn = document.querySelector('#verdict-group button.active');
    const verdict = verdictBtn ? verdictBtn.dataset.verdict : '';
    const tagsRaw = document.getElementById('tags').value;
    const tags = tagsRaw.split(',').map(s => s.trim()).filter(Boolean);
    const comment = document.getElementById('comment').value;

    // No-op if there's nothing to save.
    const existing = (state.session.labels || {})[ev.event_id];
    if (!verdict && !tags.length && !comment && (!existing || !existing.verdict && !(existing.tags || []).length && !existing.comment)) {
        return;
    }

    setSaveStatus('saving', 'saving…');
    try {
        const payload = await apiPost(`/api/label/${encodeURIComponent(ev.event_id)}`, { verdict, tags, comment });
        state.session.labels = state.session.labels || {};
        state.session.labels[ev.event_id] = payload;
        setSaveStatus('saved', 'saved');
        for (const t of tags) state.knownTags.add(t);
        refreshTagSuggestions();
        renderMinimap();
        renderEventMeta(ev);
    } catch (e) {
        console.error(e);
        setSaveStatus('error', 'save failed');
    }
}

const debouncedSave = debounce(saveCurrentLabel, 350);

function setVerdict(v) {
    document.querySelectorAll('#verdict-group button').forEach(b => {
        b.classList.toggle('active', b.dataset.verdict === v);
    });
    debouncedSave();
}

async function clearVerdict() {
    setVerdict('');
    const ev = currentEvent();
    if (!ev) return;
    const tags = document.getElementById('tags').value.trim();
    const comment = document.getElementById('comment').value.trim();
    if (!tags && !comment) {
        // Truly empty → DELETE on server
        try {
            await apiDelete(`/api/label/${encodeURIComponent(ev.event_id)}`);
            if (state.session.labels) delete state.session.labels[ev.event_id];
            setSaveStatus('saved', 'cleared');
            renderMinimap();
            renderEventMeta(ev);
        } catch (e) {
            console.error(e); setSaveStatus('error', 'clear failed');
        }
    } else {
        debouncedSave();
    }
}

function refreshTagSuggestions() {
    const dl = document.getElementById('tag-suggestions');
    dl.innerHTML = Array.from(state.knownTags).sort().map(t => `<option value="${t}">`).join('');
}

// ---------------------------------------------------------------------------
// Bindings
// ---------------------------------------------------------------------------

function bindLabelForm() {
    document.querySelectorAll('#verdict-group button').forEach(b => {
        b.addEventListener('click', () => {
            const v = b.dataset.verdict;
            if (v === '') return clearVerdict();
            setVerdict(v);
        });
    });
    document.getElementById('tags').addEventListener('input', debouncedSave);
    document.getElementById('comment').addEventListener('input', debouncedSave);
}

function bindAudio() {
    const audio = document.getElementById('audio');
    audio.src = '/api/audio';
    audio.addEventListener('timeupdate', () => {
        schedulePlayheadUpdate(audio.currentTime);
        checkEventEntry(audio.currentTime);
    });
    document.getElementById('play-event').addEventListener('click', () => {
        const ev = currentEvent(); if (!ev) return;
        audio.currentTime = ev.start_sec; audio.play();
    });
    document.getElementById('play-context').addEventListener('click', () => {
        const ev = currentEvent(); if (!ev) return;
        audio.currentTime = eventContextStart(ev); audio.play();
    });
    document.getElementById('back-2s').addEventListener('click', () => { audio.currentTime = Math.max(0, audio.currentTime - 2); });
    document.getElementById('fwd-2s').addEventListener('click',  () => { audio.currentTime = audio.currentTime + 2; });
}

function bindNavigation() {
    document.getElementById('prev').addEventListener('click', () => navigate(-1));
    document.getElementById('next').addEventListener('click', () => navigate(+1));

    // Click on minimap → jump
    document.getElementById('minimap').addEventListener('click', (e) => {
        const mm = e.currentTarget;
        const x = e.clientX - mm.getBoundingClientRect().left;
        const frac = x / mm.clientWidth;
        const idx = Math.min(state.filtered.length - 1, Math.max(0, Math.floor(frac * state.filtered.length)));
        if (idx !== state.currentIndex) {
            saveCurrentLabel().finally(() => { state.currentIndex = idx; renderAll(); });
        }
    });

    // Click on plot → seek audio. (Plotly augments the div with .on() after the first render.)
}

function bindPlotClick() {
    const plot = document.getElementById('plot');
    if (typeof plot.on !== 'function') return;
    plot.on('plotly_click', (e) => {
        if (!e || !e.points || !e.points.length) return;
        document.getElementById('audio').currentTime = e.points[0].x;
    });
}

function navigate(delta) {
    if (!state.filtered.length) return;
    const next = Math.min(state.filtered.length - 1, Math.max(0, state.currentIndex + delta));
    if (next === state.currentIndex) return;
    // Flush pending save before moving on
    saveCurrentLabel().finally(() => { state.currentIndex = next; renderAll(); });
}

function bindKeyboard() {
    document.addEventListener('keydown', (e) => {
        const tag = (document.activeElement || {}).tagName;
        const inField = tag === 'INPUT' || tag === 'TEXTAREA' || tag === 'SELECT';
        if (inField && e.key !== 'Escape') return;

        switch (e.key) {
            case 'j': case 'ArrowDown': navigate(+1); e.preventDefault(); break;
            case 'k': case 'ArrowUp':   navigate(-1); e.preventDefault(); break;
            case '1': setVerdict('tp');      e.preventDefault(); break;
            case '2': setVerdict('fp');      e.preventDefault(); break;
            case '3': setVerdict('unclear'); e.preventDefault(); break;
            case '4': setVerdict('partial'); e.preventDefault(); break;
            case '0': clearVerdict();        e.preventDefault(); break;
            case ' ': {
                const a = document.getElementById('audio');
                a.paused ? a.play() : a.pause();
                e.preventDefault();
                break;
            }
            case 'e': {
                const ev = currentEvent(); if (!ev) break;
                const a = document.getElementById('audio');
                a.currentTime = ev.start_sec; a.play();
                e.preventDefault();
                break;
            }
            case 'Escape': document.activeElement && document.activeElement.blur(); break;
        }
    });
}

// ---------------------------------------------------------------------------
// Theme
// ---------------------------------------------------------------------------

function initTheme() {
    const saved = localStorage.getItem('ae-review-theme') || 'dark';
    document.documentElement.dataset.theme = saved;
}

function toggleTheme() {
    const next = document.documentElement.dataset.theme === 'dark' ? 'light' : 'dark';
    document.documentElement.dataset.theme = next;
    localStorage.setItem('ae-review-theme', next);
    renderPlot(currentEvent());
}

// ---------------------------------------------------------------------------
// Multi-resolution waveform
// ---------------------------------------------------------------------------

let waveformAbort = null;

async function fetchWindowedWaveform(t0, t1) {
    if (waveformAbort) waveformAbort.abort();
    waveformAbort = new AbortController();
    try {
        const url = `/api/waveform?t0=${t0.toFixed(2)}&t1=${t1.toFixed(2)}&n_peaks=2000`;
        const r = await fetch(url, { signal: waveformAbort.signal });
        if (!r.ok) return;
        const wf = await r.json();
        state.windowedWaveform = wf;
        renderPlot(currentEvent());
    } catch (e) {
        if (e.name !== 'AbortError') console.error('waveform fetch:', e);
    }
}

const debouncedWaveformFetch = debounce(() => {
    const ev = currentEvent();
    if (!ev) return;
    const [t0, t1] = getViewRange(ev);
    const dur = state.session.audio_duration_sec || 1;
    if ((t1 - t0) / dur > 0.5) {
        state.windowedWaveform = null;
        renderPlot(currentEvent());
        return;
    }
    fetchWindowedWaveform(t0, t1);
}, 200);

// ---------------------------------------------------------------------------
// Init
// ---------------------------------------------------------------------------

async function init() {
    initTheme();

    const themeBtn = document.getElementById('theme-toggle');
    if (themeBtn) themeBtn.addEventListener('click', toggleTheme);

    if (typeof Plotly === 'undefined') {
        await new Promise((resolve) => {
            const check = setInterval(() => {
                if (typeof Plotly !== 'undefined') { clearInterval(check); resolve(); }
            }, 30);
        });
    }

    setSaveStatus('muted', 'loading…');
    try {
        const [session, signalsResp, waveform] = await Promise.all([
            apiGet('/api/session'),
            apiGet('/api/signals'),
            apiGet('/api/waveform'),
        ]);
        state.session = session;
        state.signals = signalsResp.signals;
        state.signalsMeta = signalsResp.meta;
        state.waveform = waveform;
        state.eventsById = Object.fromEntries(session.events.map(e => [e.event_id, e]));

        state.signalNames = resolveSignalNames(state.signalsMeta, state.signals);
        state.panels = buildPanelConfig(state.signalNames);

        for (const lbl of Object.values(session.labels || {})) {
            for (const t of (lbl.tags || [])) state.knownTags.add(t);
        }
        refreshTagSuggestions();

        renderTopbar();
        populateFilterUI();
        bindLabelForm();
        bindAudio();
        bindNavigation();
        bindKeyboard();

        refresh();
        bindPlotClick();

        window.addEventListener('resize', debounce(() => {
            Plotly.Plots.resize('plot');
        }, 150));

        setSaveStatus('muted', '');
    } catch (e) {
        console.error(e);
        setSaveStatus('error', 'failed to load session');
    }
}

if (document.readyState === 'loading') {
    document.addEventListener('DOMContentLoaded', init);
} else {
    init();
}

})();

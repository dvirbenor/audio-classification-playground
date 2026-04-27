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
    eventsById: {},       // id -> event
    filtered: [],         // ordered list of event ids after filter+sort
    currentIndex: 0,      // index into filtered
    filter: { signal: '', type: '', minConf: 0, unlabeledOnly: false },
    sort: 'time',
    playheadSec: 0,
    lastEventEnteredId: null,
    knownTags: new Set(),
};

// Panel y-axis layout (top→bottom): arousal, valence, dominance, waveform.
// Maps signal name → axis suffix used by Plotly (yaxis, yaxis2, ...).
const PANEL_DOMAINS = {
    arousal:   [0.78, 1.00],
    valence:   [0.55, 0.76],
    dominance: [0.32, 0.53],
    waveform:  [0.00, 0.28],
};
const PANEL_AXIS = {
    arousal:   'y4',
    valence:   'y3',
    dominance: 'y2',
    waveform:  'y',
};
const SIGNAL_COLORS = {
    arousal:   '#2563eb',  // blue
    valence:   '#ea580c',  // orange
    dominance: '#16a34a',  // green
};
const VERDICTS = ['tp', 'fp', 'unclear', 'partial'];

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
        if ((e.confidence || 0) < f.minConf) return false;
        const lbl = labels[e.event_id];
        if (f.unlabeledOnly && lbl && lbl.verdict) return false;
        return true;
    });

    const cmp = {
        time:       (a, b) => a.start_sec - b.start_sec,
        strength:   (a, b) => (b.strength || 0) - (a.strength || 0),
        confidence: (a, b) => (b.confidence || 0) - (a.confidence || 0),
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
    for (const name of Object.keys(state.signals)) {
        const meta = state.signalsMeta[name];
        const ts = frameTimes(meta);
        traces.push({
            type: 'scattergl',
            mode: 'lines',
            name,
            x: Array.from(ts),
            y: state.signals[name],
            xaxis: 'x',
            yaxis: PANEL_AXIS[name] || 'y2',
            line: { color: SIGNAL_COLORS[name] || '#666', width: 1.4 },
            hovertemplate: `${name}: %{y:.3f}<br>t=%{x:.2f}s<extra></extra>`,
            showlegend: false,
        });
    }
    return traces;
}

function buildWaveformTrace() {
    const wf = state.waveform;
    if (!wf || !wf.n_peaks) return null;
    const dt = wf.duration_sec / wf.n_peaks;
    const x = new Array(wf.n_peaks * 3);
    const y = new Array(wf.n_peaks * 3);
    for (let i = 0; i < wf.n_peaks; i++) {
        const t = i * dt + dt / 2;
        x[3 * i]     = t; y[3 * i]     = wf.min[i];
        x[3 * i + 1] = t; y[3 * i + 1] = wf.max[i];
        x[3 * i + 2] = null; y[3 * i + 2] = null;
    }
    return {
        type: 'scattergl',
        mode: 'lines',
        x, y,
        xaxis: 'x',
        yaxis: 'y',
        line: { color: '#888', width: 1 },
        hoverinfo: 'skip',
        showlegend: false,
    };
}

function buildShapes(ev) {
    const shapes = [];
    if (!ev) return shapes;

    // (1) Highlight detection signal panel — full-width tinted rect.
    const detDomain = PANEL_DOMAINS[ev.signal_name];
    if (detDomain) {
        shapes.push({
            type: 'rect',
            xref: 'paper', yref: 'paper',
            x0: 0, x1: 1,
            y0: detDomain[0], y1: detDomain[1],
            fillcolor: 'rgba(37, 99, 235, 0.07)',
            line: { width: 0 },
            layer: 'below',
        });
    }

    // (2) Speech-block bands — visible behind data lines on every panel.
    const t0 = ev.review_audio_start_sec;
    const t1 = ev.review_audio_end_sec;
    for (const b of state.session.blocks) {
        if (b.end_sec < t0 || b.start_sec > t1) continue;
        shapes.push({
            type: 'rect',
            xref: 'x', yref: 'paper',
            x0: b.start_sec, x1: b.end_sec,
            y0: 0, y1: 1,
            fillcolor: 'rgba(34, 197, 94, 0.08)',
            line: { width: 0 },
            layer: 'below',
        });
    }

    // (3) Other events in the visible window (pale, so the user sees them).
    for (const other of state.session.events) {
        if (other.event_id === ev.event_id) continue;
        if (other.event_type === 'joint' || other.event_type === 'affective_episode') continue;
        if (other.end_sec < t0 || other.start_sec > t1) continue;
        const otherDomain = PANEL_DOMAINS[other.signal_name];
        if (!otherDomain) continue;
        shapes.push({
            type: 'rect',
            xref: 'x', yref: 'paper',
            x0: other.start_sec, x1: other.end_sec,
            y0: otherDomain[0], y1: otherDomain[1],
            fillcolor: 'rgba(220, 50, 50, 0.10)',
            line: { width: 0 },
            layer: 'below',
        });
    }

    // (4) THE event — bold rect on the detection panel + thin band across all.
    const eDomain = PANEL_DOMAINS[ev.signal_name] || [0, 1];
    shapes.push({
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: ev.start_sec, x1: ev.end_sec,
        y0: 0, y1: 1,
        fillcolor: 'rgba(220, 50, 50, 0.12)',
        line: { color: 'rgba(220, 50, 50, 0.4)', width: 1 },
        layer: 'below',
    });
    shapes.push({
        type: 'rect',
        xref: 'x', yref: 'paper',
        x0: ev.start_sec, x1: ev.end_sec,
        y0: eDomain[0], y1: eDomain[1],
        fillcolor: 'rgba(220, 50, 50, 0.20)',
        line: { color: 'rgba(220, 50, 50, 0.7)', width: 1.5 },
        layer: 'below',
    });

    // (5) Playhead — thin vertical line spanning full plot. Updated separately.
    shapes.push({
        type: 'line',
        xref: 'x', yref: 'paper',
        x0: state.playheadSec, x1: state.playheadSec,
        y0: 0, y1: 1,
        line: { color: 'rgba(0, 0, 0, 0.6)', width: 1.5 },
        layer: 'above',
    });
    return shapes;
}

const PLAYHEAD_SHAPE_KEY = 'playhead';

function buildLayout(ev) {
    const t0 = ev ? ev.review_audio_start_sec : 0;
    const t1 = ev ? ev.review_audio_end_sec : (state.session.audio_duration_sec || 60);

    const baseAxis = (title, domain) => ({
        domain,
        title: { text: title, standoff: 4, font: { size: 11 } },
        zerolinecolor: '#ddd',
        gridcolor: '#eee',
        tickfont: { size: 10 },
    });

    return {
        margin: { t: 14, b: 32, l: 56, r: 14 },
        showlegend: false,
        plot_bgcolor: '#ffffff',
        paper_bgcolor: '#ffffff',
        hovermode: 'x',
        dragmode: 'pan',
        xaxis: {
            domain: [0, 1],
            range: [t0, t1],
            title: { text: 'time (s)', font: { size: 11 } },
            tickfont: { size: 10 },
            gridcolor: '#eee',
        },
        yaxis:  baseAxis('audio',     PANEL_DOMAINS.waveform),
        yaxis2: baseAxis('dominance', PANEL_DOMAINS.dominance),
        yaxis3: baseAxis('valence',   PANEL_DOMAINS.valence),
        yaxis4: baseAxis('arousal',   PANEL_DOMAINS.arousal),
        shapes: buildShapes(ev),
    };
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

    const baselineLabel = ev.baseline_source === 'local'
        ? `local baseline (ctx ${fmtNum(ev.baseline_context_speech_sec, 1)}s)`
        : `global baseline (sparse ctx ${fmtNum(ev.baseline_context_speech_sec, 1)}s)`;

    meta.innerHTML = [
        `<span class="pill signal">${ev.signal_name}</span>`,
        `<span class="pill">${ev.event_type}</span>`,
        `<span class="pill">${ev.direction}</span>`,
        `t=${fmtNum(ev.start_sec, 2)}–${fmtNum(ev.end_sec, 2)}s (${fmtNum(ev.duration_sec, 1)}s)`,
        `Δ=${fmtNum(ev.delta, 3)} (z=${fmtNum(ev.delta_z, 2)})`,
        `strength=${fmtNum(ev.strength, 2)}`,
        `conf=${fmtNum(ev.confidence, 2)}`,
        `cov=${fmtNum(ev.mean_speech_coverage, 2)}`,
        `block=${ev.block_ids.join(',')}`,
        baselineLabel,
        ev.near_block_start ? '<span class="pill" style="border-color:#d97706;color:#d97706">near-start</span>' : '',
        ev.near_block_end   ? '<span class="pill" style="border-color:#d97706;color:#d97706">near-end</span>'   : '',
        ev.parent_id ? `parent=<code>${ev.parent_id}</code>` : '',
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

    Plotly.react('plot', traces, buildLayout(ev), { displaylogo: false, responsive: true });
}

function renderAll() {
    const ev = currentEvent();
    renderEventMeta(ev);
    renderLabelForm(ev);
    renderMinimap();
    renderPlot(ev);

    if (ev) {
        const audio = document.getElementById('audio');
        const target = ev.review_audio_start_sec;
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

    const conf = document.getElementById('filter-conf');
    const confDisplay = document.getElementById('filter-conf-display');
    conf.addEventListener('input', () => {
        state.filter.minConf = parseFloat(conf.value);
        confDisplay.textContent = state.filter.minConf.toFixed(2);
        refresh();
    });

    const ulOnly = document.getElementById('filter-unlabeled');
    ulOnly.addEventListener('change', () => { state.filter.unlabeledOnly = ulOnly.checked; refresh(); });

    const sortSel = document.getElementById('sort-by');
    sortSel.addEventListener('change', () => { state.sort = sortSel.value; refresh(); });
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
        audio.currentTime = ev.review_audio_start_sec; audio.play();
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
// Init
// ---------------------------------------------------------------------------

async function init() {
    // Wait for Plotly (loaded with `defer`) before rendering.
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

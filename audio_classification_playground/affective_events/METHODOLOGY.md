# Affective Events — Methodology and Implementation Plan

This document specifies the new affective-events detection algorithm and the
plan to replace the current implementation. It is the canonical reference;
where it disagrees with `ALGORITHM.md`, this document wins. `ALGORITHM.md`
will be retired once the new module ships.

---

# Part I — Methodology

## 1. Goal and design philosophy

**Goal.** Given continuous A/V/D speech-emotion predictions on a regular hop
grid plus a VAD timeline, emit discrete events: time intervals where one
signal departs notably from its local context in a sustained, single-direction
way. An "event" should match what a human picks out by eye when scanning the
A/V/D plot — typically a U-shaped or inverted-U-shaped lobe lasting a few
seconds, that visibly stands out from the surrounding speech-region wiggle.

**Single principle.** An event is a contiguous interval where the signal,
measured against a baseline computed from *surrounding speech blocks*,
deviates by more than a threshold and stays out long enough to be real.

Everything in this design follows from that one sentence. There are no
sub-categories of events at detection time, no model-shape selection, no
post-hoc validation/aggregation passes. The structural taxonomy ("block
deviation" vs "within-block excursion" vs "regime shift" vs "ramp") that the
old design imposed is replaced by *one* event type whose duration and shape
are descriptive metadata.

**Why one detector works for both block-level and within-block events.** The
baseline is computed from *other* speech blocks (the candidate's own block is
excluded entirely). So:

- A whole block elevated above its neighbors yields a flat-but-elevated
  *z*-score across the block's interior — detected as one long event.
- A peak inside an otherwise-normal block yields a localized *z*-spike —
  detected as one short event.

Same operation, different durations. The categorical distinction in the old
design was an artifact of using a *moving* baseline that could be tricked by
sustained events; it disappears once the baseline is anchored on the
candidate's neighbors instead.

---

## 2. Inputs and outputs

### 2.1 Inputs

| Input | Type | Description |
|---|---|---|
| `signals` | `dict[str, np.ndarray]` | One 1-D array per signal (e.g. `arousal`, `valence`, `dominance`) on a regular hop grid. Frame *i* summarizes audio in `[i·hop, i·hop + window]`. Values are bounded model outputs in roughly `[0, 1]`. |
| `vad` | `Sequence[(start_sec, end_sec)]` | Voice-activity intervals, sorted, non-overlapping. |
| `hop_sec` | `float` | Spacing between consecutive frames (e.g. 0.25). |
| `window_sec` | `float` | Receptive-field length each frame summarizes (e.g. 3.5). |

### 2.2 Output — `Event`

A flat list of records:

| Field | Meaning |
|---|---|
| `event_id` | unique string id |
| `signal_name` | `"arousal"` / `"valence"` / `"dominance"` for leaves, `"joint"` for parents |
| `event_type` | `"deviation"` for leaves, `"joint"` for parents |
| `start_sec`, `end_sec`, `duration_sec` | event extent in seconds |
| `frame_start`, `frame_end` | inclusive/exclusive frame indices |
| `direction` | `"+"` or `"-"` (sign of *z* at peak) for leaves; signature like `"A+ V- D0"` for joints |
| `peak_z` | maximum `\|z\|` reached inside the event — the **strength** score |
| `peak_time_sec` | absolute time of the peak (frame center) |
| `baseline_at_peak` | block-aware baseline at the peak frame |
| `scale_at_peak` | block-aware scale at the peak frame |
| `delta` | `signal[peak] − baseline_at_peak` (raw difference, for human readability) |
| `parent_id` | id of a `joint` parent if the leaf belongs to one |
| `children` | child ids on `joint` parents |

**No** other event types: no `regime_shift`, `ramp`, `block_deviation`,
`excursion`, `short_gap_block_transition`, `affective_episode`. A long event
spanning a whole VAD block and a short event inside one are both
`deviation`; the distinction is descriptive (duration, peak position), not
categorical.

### 2.3 Strength versus confidence

`peak_z` is the **strength** of the event. It is a comparable, signal-agnostic
score (because *z* is computed against block-aware MAD), well-suited for
ranking. It is **not** a calibrated probability and should not be called
"confidence" in the API or schema.

If a UI ever needs a `[0, 1]`-bounded score for display, a sigmoid mapping
(e.g. `1 / (1 + exp(−1.5·(peak_z − 1.5)))`) computed in the UI layer is fine —
but the underlying field is `strength = peak_z`.

---

## 3. Pre-processing

### 3.1 Build analysis blocks

Take the raw VAD and merge it:

- Bridge gaps `≤ vad_merge_gap_sec` (default 0.5 s).
- Drop intervals shorter than `min_speech_block_sec` (default 0.75 s).

The result is a list of `Block` records each with `(block_id, start_sec,
end_sec)`. These are the "speech blocks" referred to throughout.

### 3.2 Per-frame block assignment (interior-only mask)

For each frame `i` with center `c_i = i·hop + window/2`:

```
frame_block[i] = bid    if c_i − window/2 >= block[bid].start
                         and c_i + window/2 <= block[bid].end
                 -1     otherwise (silence, or block edge)
```

A frame is **interior** iff its *entire* receptive window falls inside a
merged speech block. Only interior frames participate in detection.

Why this strict criterion: speech-edge frames have receptive windows that
straddle silence, and the upstream A/V/D model produces transitional values
on those frames that look like deviations but are artifacts. Excluding them
structurally — instead of flagging them post-hoc with a `boundary_margin`
penalty — eliminates a whole class of false positives.

The user concern that "fully interior is too strict" was tested empirically
(notebook §A): going from coverage ≥ 0.6 to coverage = 1.0 loses only 7% of
valid frames (66.9% → 60.3%) and the count of contiguous valid runs ≥ 5 s is
essentially unchanged (70 → 71). The gain of cleaner edges far outweighs the
loss of usable frames.

### 3.3 Per-signal global stats

Compute once per signal over interior frames:

```
global_median[s] = median(signal[s][interior])
global_mad[s]    = 1.4826 · median(|signal[s][interior] − global_median[s]|)
```

These are used as fallback for the block-aware baseline/scale and as a floor
for the local scale.

---

## 4. Block-aware baseline and scale

This is the load-bearing structural choice of the design. Per signal:

### 4.1 Definition

For each interior frame `i` belonging to block `B`, with frame center `t_i`:

```
context = { interior frames j  :  frame_block[j] != B
                                  AND  |frame_center[j] − t_i| <= radius_sec }

if total_seconds_in(context) < min_context_sec:
    baseline[i] = global_median[signal]
    scale[i]    = global_mad[signal]
else:
    ctx_vals    = signal[context]
    baseline[i] = median(ctx_vals)
    scale[i]    = max( 1.4826·MAD(ctx_vals),
                       scale_floor_frac · global_mad[signal] )
```

For non-interior frames: `baseline[i] = scale[i] = NaN`.

### 4.2 What this replaces

This single mechanism replaces **three** components of the old design:

| Old component | Replaced by |
|---|---|
| `local_baseline(start, end, ctx_radius, exclude_radius)` — per-candidate baseline excluding a fixed radius around the candidate region | Block-aware baseline excludes the candidate's *whole block* by definition; no `exclude_candidate_radius_sec` parameter. |
| Two-pass baseline (detect → NaN-out events → recompute → re-detect) | Block-level exclusion does the same job in one pass. The two-pass approach was empirically near-identical to one-pass on this recording (notebook §B: 99 vs 101 events) and structurally redundant with block-aware exclusion. |
| Local moving MAD for the *z*-score scale | Block-aware MAD (same context as the baseline). The local moving MAD was over-firing in calm regions because the candidate's own bumps inflated/deflated the denominator; excluding the own block fixes this. |

### 4.3 Why this is structural, not heuristic

- **No self-bias.** A sustained event cannot pull its own baseline up — the
  *entire* block containing it is excluded from the median. The "sustained
  event poisons its own baseline" failure mode of moving-median detectors is
  eliminated by construction.
- **Block-level and within-block events unify.** No special detector for
  "this whole block is elevated"; the *z*-score across the block's interior
  becomes flat-but-high, and the same prominence detector picks it up as one
  long event.
- **Adapts to non-stationarity.** The scale comes from the surrounding
  ~minutes of speech. Noisier sections demand bigger excursions; calmer ones
  accept smaller. The `scale_floor_frac · global_mad` floor prevents
  pathologically small scales in flat regions (which would otherwise make
  the detector fire on noise).
- **No per-candidate computation.** The baseline/scale at frame *i* depend
  only on which block *i* belongs to and its absolute time — not on any
  detection state. So baseline/scale can be computed once, in isolation,
  before any detection logic runs.

### 4.4 Implementation note: per-frame versus per-block context

The conceptually clean version selects context using each frame's own
center time. For an efficient implementation, per-block computation
(using each block's center) is an acceptable approximation **as long as
no block is longer than `radius_sec`**. For blocks longer than that, the
per-block approximation drops context near the block's edges; per-frame
selection must be used.

This recording's longest VAD interval is 135 s; with `radius_sec = 120 s`
that one block needs per-frame computation. The implementation should:

1. Compute per-block context for blocks shorter than `radius_sec`.
2. Compute per-frame context inside any block longer than `radius_sec`.

Both produce identical output for short blocks; the split exists only for
performance.

---

## 5. Z-score timeline

```
for each interior frame i in signal s:
    z[s][i] = (signal[s][i] − baseline[s][i]) / scale[s][i]
for non-interior frames:
    z[s][i] = 0
```

The `z[s]` array is the input to detection. Same units across signals (each
signal's *z* is normalized by *its own* block-aware MAD), so a single
threshold parameter applies uniformly.

---

## 6. Detection — prominence with seed width and shoulder bound

Run **independently per signal**.

### 6.1 The detection rule

For each sign `σ ∈ {+1, −1}` (positive and negative deviations are detected
independently and tagged with their direction):

```
sz = σ · z                                   # signed z
seed_mask[i] = interior[i] AND (sz[i] >= z_seed)

for each contiguous run [r_start, r_end) of seed_mask = True:
    if (r_end − r_start) < seed_min_width_frames:
        continue                              # seed cluster too thin → noise

    peak_i = argmax(sz[r_start : r_end])
    peak_z = sz[peak_i]

    # LEFT shoulder: extend while we're still above the return threshold
    left = peak_i
    while left > 0 and interior[left − 1] and sz[left − 1] >= z_return:
        left −= 1

    # RIGHT shoulder: same
    right = peak_i
    while right < N − 1 and interior[right + 1] and sz[right + 1] >= z_return:
        right += 1

    record event (left, right + 1, sign=σ, peak_z=peak_z, peak_i=peak_i)
```

### 6.2 What each piece does, and why these specifically

- **Seed criterion (`z_seed` + `seed_min_width`).** A peak qualifies for
  detection only if `|z|` stays above the seed threshold for at least
  `seed_min_width_sec`. This is the structural fix for the over-firing we
  saw with z-thresholding alone (notebook §C → §E: 237 events → 51 events
  on arousal). Without `seed_min_width`, a single-frame crossing of `z_seed`
  could be inflated into a multi-second event by the shoulder extension.
- **Peak from seed cluster.** The peak is the local extremum *inside* the
  seed cluster, not the first frame above threshold. This anchors the
  shoulder trace at the actual event center.
- **Shoulder bound (`z_return`).** Trace left/right while `|z| ≥ z_return`.
  This gives event extents that visually trace the U-shape of the lobe — we
  evaluated a parameter-free alternative (zero-crossing of *z*, capped by
  prominence base) in notebook §H and found it cut events tighter than the
  human eye expects. `z_return = 0.5` keeps the shoulders on the natural
  fall-off of the lobe.
- **Interior gating in shoulder trace.** Hitting a non-interior frame stops
  the shoulder. Events never extend across silence or block edges.
- **Per-sign detection.** Positive and negative deviations are detected
  separately. A positive lobe and a nearby negative lobe are two distinct
  events with opposite `direction`.

### 6.3 Post-detection filtering (per signal)

After the per-sign detection produces a list of (overlapping or adjacent)
events:

1. **Sort** events by `frame_start`.
2. **Merge same-sign events** whose frame gap `≤ merge_gap_frames`. The
   merged event spans `[min_start, max_end]` and carries `peak_z =
   max(child.peak_z)` and the corresponding `peak_i`.
3. **Drop events** with `(frame_end − frame_start) < min_duration_frames`.

Merging across opposite signs is **never** done — a positive and a negative
lobe at close range are two events.

### 6.4 Per-event metadata

For each surviving event, attach:

```
event.peak_time_sec    = frame_center[peak_i]
event.baseline_at_peak = baseline[peak_i]
event.scale_at_peak    = scale[peak_i]
event.delta            = signal[peak_i] − baseline[peak_i]
event.start_sec        = frame_center[frame_start]      # first frame's center
event.end_sec          = frame_center[frame_end - 1]    # last frame's center
event.duration_sec     = end_sec − start_sec
```

---

## 7. Cross-signal joint events

A simple post-pass over leaves from all signals:

```
build undirected graph G:
    nodes = leaves
    edge (u, v) iff
        u.signal_name != v.signal_name
        AND temporal overlap between u and v >= cross_signal_min_overlap_sec

for each connected component C with >= 2 distinct signals:
    emit a joint parent:
        signal_name = "joint"
        event_type  = "joint"
        start_sec   = min over C of leaf.start_sec
        end_sec     = max over C of leaf.end_sec
        peak_z      = sqrt( sum over C of leaf.peak_z² )    # quadrature
        direction   = "A{+/-/0} V{+/-/0} D{+/-/0}"
                      where each axis is + if any leaf in that signal has
                      direction "+" and peak_z >= 0.5, − if any has "−" and
                      peak_z >= 0.5, otherwise 0
        children    = [leaf.event_id for leaf in C]

set leaf.parent_id = joint.event_id for all leaves in C
```

There is **no** per-signal episode aggregation. A signal's events span what
they span; if two same-signal events stayed separate after the merge step in
§6.3, they are genuinely separate events.

---

## 8. Parameters

Six in detection, three in pre-processing, two in baseline/scale, one in
fusion. Every parameter has an intuitive unit and a default that survived
empirical verification on the test recording.

| Group | Parameter | Default | Type | Meaning |
|---|---|---|---|---|
| Pre-processing | `vad_merge_gap_sec` | `0.5` | `float` | Bridge VAD gaps shorter than this |
|  | `min_speech_block_sec` | `0.75` | `float` | Drop blocks shorter than this |
| Baseline & scale | `radius_sec` | `120.0` | `float` | Half-width of the context window over neighbor blocks |
|  | `min_context_sec` | `5.0` | `float` | Minimum total context speech-time before falling back to global stats |
|  | `scale_floor_frac` | `0.5` | `float` | Floor `scale` at this fraction of `global_mad` |
| Detection | `z_seed` | `1.75` | `float \| dict[str, float]` | Threshold for an event to seed |
|  | `seed_min_width_sec` | `1.0` | `float \| dict[str, float]` | Seed cluster must span at least this long at `z_seed` |
|  | `z_return` | `0.5` | `float \| dict[str, float]` | Shoulder bound: extend event while `\|z\| ≥ z_return` |
|  | `min_duration_sec` | `2.5` | `float \| dict[str, float]` | Drop events shorter than this |
|  | `merge_gap_sec` | `0.5` | `float \| dict[str, float]` | Merge same-sign events with frame gap ≤ this |
| Cross-signal | `cross_signal_min_overlap_sec` | `1.0` | `float` | Minimum temporal overlap to link leaves into a joint |

### 8.1 Per-signal overrides

Detection parameters accept either a scalar (uniform across signals) or a
`dict[signal_name, value]` for per-signal overrides. **Default is uniform**:
the block-aware MAD already normalizes per-signal scale, so a single
threshold serves all signals.

The one signal that drifts off uniform is dominance — at `z_seed = 2.0` it
produces 64 events vs arousal's 51 (notebook §G), because dominance has the
lowest global MAD and small bumps look proportionally bigger in *z*-units.
If a downstream user observes "dominance over-detects on this corpus," the
clean override is `z_seed = {"dominance": 2.0, "_default": 1.75}`. **Do
not** populate this dict by default — start uniform and override only if
evidence demands it.

The pre-processing, baseline-radius, and fusion parameters stay scalar:
they describe data geometry (receptive field, VAD structure, cross-signal
coincidence), not detection trade-offs.

### 8.2 Parameter sensitivity (from the notebook)

- `radius_sec` — low sensitivity. Notebook §6 showed 30/60/120/240 s windows
  produce nearly identical deviation distributions.
- `merge_gap_sec` — low sensitivity. Notebook §D showed event counts change
  by ≤5% across 0.25–1.5 s.
- `seed_min_width_sec` — moderate sensitivity. Going from 0 → 1 s drops
  arousal events from 78 → 44 at `z_seed = 2.0` (notebook §E table). This
  is the parameter that does the real work of suppressing brief noise.
- `z_seed` — high sensitivity by design. This is the only "trade-off"
  parameter; everything else has a structural justification.
- `min_duration_sec` — moderate sensitivity. The shoulder rule already
  controls extent, so `min_duration` is mainly an outlier filter.

---

## 9. Edge cases and verification

### 9.1 Long blocks with internal heterogeneity

**The case.** A speech block significantly longer than `radius_sec` (e.g.
135 s when `radius_sec = 120 s`), where the block as a whole is tonally
shifted from its neighbors AND contains one or more shorter excursions on
top of that shift.

**Current behavior.** The block-aware baseline computed at the *block*'s
center reflects the block's neighbors. *z* across the whole block will be
flat-but-high (block-level component). The within-block excursion adds an
extra bump on top. The shoulder rule extends until *z* drops below
`z_return`. Result: **one long event** spanning the full block, peak at
the within-block excursion location.

**Is this correct?** Under the "one deviation event" philosophy, yes. The
block IS one sustained deviation; the within-block excursion is its peak.
Two separate events would imply we believe in distinct categories
("block-level" vs "within-block"), which we explicitly rejected.

**However, this is the case to verify on real data.** If reviewers
consistently report wanting the within-block peak as a *separate, finer*
event sitting inside the long block, the design needs an extension:

- Run a *second* detector pass with a within-block local moving median as
  the baseline (no exclusion), only on blocks longer than some threshold,
  with results layered as "sub-events" inside the long event.

This extension is **not in the v1 scope**. We ship the unified detector
first and add the within-block sub-pass only if reviewers say it's needed.

### 9.2 Short blocks with no qualifying interior frames

A block whose duration is `< window_sec` produces zero interior frames. It
contributes nothing to detection. This is correct: a sub-window block
doesn't have enough audio to make a confident deviation claim.

### 9.3 Recording with very few blocks

If `min_context_sec` worth of context cannot be assembled from neighbor
blocks (e.g., a 30-second clip with two 5-second blocks), every frame falls
back to global stats. Detection still runs but loses the per-region
adaptation. This is correct degradation.

### 9.4 Verifiable invariants for tests

- An event's `frame_start` is interior, and `frame_end - 1` is interior
  (since `frame_end` is exclusive).
- An event never contains a non-interior frame in `[frame_start,
  frame_end)`.
- An event's `peak_z` is `≥ z_seed`.
- The seed cluster underlying each event has width `≥ seed_min_width_sec`.
- Both shoulder endpoints have `|z| ≥ z_return`; the next frame outward
  (if interior) has `|z| < z_return`.
- Same-sign events on the same signal are separated by `frame_gap >
  merge_gap_frames`.
- Every event has `duration_sec ≥ min_duration_sec`.
- A `joint` parent has children from `≥ 2` distinct signals.
- For a `joint` parent and any pair of its children, the temporal overlap
  is `≥ cross_signal_min_overlap_sec`.
- Every leaf has `parent_id` set iff it is a child of some `joint`.

---

## 10. What we removed (and why)

| Removed | Reason |
|---|---|
| Block-level model selection (M0/M1/M2 with BIC) | Answered the wrong question (parametric description) instead of "is this notable?". Block-aware baseline subsumes its detection role. |
| `block_deviation`, `regime_shift`, `ramp`, `excursion`, `short_gap_block_transition`, `affective_episode` event types | All collapse into one `deviation` type; differences are descriptive (peak_z, duration), not categorical. |
| Two-pass exclusion baseline | Empirically near-identical to one-pass on this recording (99 vs 101 events) and structurally subsumed by block-aware exclusion. |
| `boundary_margin_sec` confidence component | Replaced by the interior-only frame mask. Edge frames don't enter the math. |
| Local moving-MAD scale | Made the detector over-fire (notebook §C). Block-aware MAD adapts locally without self-bias. |
| 7-component weighted confidence (`strength + duration + coverage + context + shape + boundary + cross_block`) | `peak_z` is the single principled strength score. Other components either fall out of structural choices or were compensating-weights soup. |
| `cross_block_validate` | Block-aware baseline already grounds each event against context. No need for a post-hoc neighbor check. |
| `min_absolute_effect: dict` | Subsumed by `z_seed` per-signal overrides. |
| `affective_episode` parents | Same-signal events that should be one event are merged in §6.3; ones that aren't merged are genuinely separate. |

The configuration shrinks from ~35 scalar fields + 7 confidence weights +
one `dict[str, float]` per-signal-effect override to **~12 scalar
parameters + 5 optional per-signal overrides**.

---

# Part II — Implementation Plan

## 11. Module layout

Replace the current `affective_events/` package contents (excluding
`review/`) with the following:

```
affective_events/
├── __init__.py          # public API: extract_events, Event, Config
├── types.py             # Signal, Vad, Block, Event (frozen dataclasses)
├── config.py            # Config (frozen dataclass + presets)
├── preprocessing.py     # build_blocks, frame_block_assignment, global_stats
├── baseline.py          # block_aware_baseline_scale
├── detector.py          # detect_prominence (seed + shoulder + post-filter)
├── fusion.py            # merge_cross_signal
├── pipeline.py          # extract_events end-to-end orchestration
├── METHODOLOGY.md       # this document
├── ALGORITHM.md         # retired; will be deleted in the cleanup commit
└── review/              # untouched
```

Total expected line count for the algorithmic core (excluding tests,
docstrings, `types.py`, `__init__.py`, `review/`):
**~400 lines**, down from ~1100 lines today. Per-file budget:

| File | Approx LOC |
|---|---|
| `types.py` | 80 |
| `config.py` | 60 |
| `preprocessing.py` | 70 |
| `baseline.py` | 80 |
| `detector.py` | 100 |
| `fusion.py` | 50 |
| `pipeline.py` | 60 |

## 12. Function signatures

### 12.1 Public API (`__init__.py`)

```python
def extract_events(
    signals: Sequence[Signal],
    vad: Vad,
    config: Config | None = None,
    *,
    diagnostics: bool = False,
) -> list[Event] | tuple[list[Event], pd.DataFrame]:
    """Top-level entry point. See METHODOLOGY.md."""
```

Same signature as today — call sites in `review/` keep working unchanged.

### 12.2 `types.py`

`Signal`, `Vad`, `Block` keep their current shapes. `Event` is rewritten:

```python
@dataclass(frozen=True)
class Event:
    event_id: str
    signal_name: str            # signal name or "joint"
    event_type: str             # "deviation" or "joint"

    start_sec: float
    end_sec: float
    duration_sec: float
    frame_start: int
    frame_end: int              # exclusive

    direction: str              # "+" / "-" for leaves, "A+ V- D0"-style for joints

    peak_z: float               # the strength score
    peak_time_sec: float
    baseline_at_peak: float
    scale_at_peak: float
    delta: float

    parent_id: str | None = None
    children: tuple[str, ...] = ()

    extra: dict = field(default_factory=dict)   # detector-specific evidence
```

### 12.3 `preprocessing.py`

```python
def build_blocks(vad: Vad, config: Config) -> list[Block]: ...

def assign_frame_blocks(
    n_frames: int, hop_sec: float, window_sec: float, blocks: list[Block]
) -> np.ndarray:
    """Per-frame block id; -1 for non-interior frames."""

def global_stats(values: np.ndarray, interior: np.ndarray) -> tuple[float, float]:
    """Return (median, scaled_mad) over interior frames."""
```

### 12.4 `baseline.py`

```python
def block_aware_baseline_scale(
    values: np.ndarray,
    frame_block: np.ndarray,
    blocks: list[Block],
    hop_sec: float,
    window_sec: float,
    config: Config,
    *,
    global_median: float,
    global_mad: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Return (baseline, scale) of length n_frames. NaN on non-interior frames.

    Implementation:
      - For each block shorter than config.radius_sec, compute context once
        per block (using block center) and broadcast to all interior frames.
      - For each block longer than config.radius_sec, compute per-frame.
    """
```

### 12.5 `detector.py`

```python
def detect_prominence(
    z: np.ndarray,
    interior: np.ndarray,
    config: Config,
    signal_name: str,
) -> list[dict]:
    """Run prominence detection on signed z-score timeline.
    Return a list of dicts with frame_start, frame_end, sign, peak_i, peak_z.
    Implements seeds (with min width), shoulder bounds, merge, min duration.
    """
```

### 12.6 `fusion.py`

```python
def merge_cross_signal(
    leaves: list[Event],
    config: Config,
    id_counter: Iterator[int],
) -> list[Event]:
    """Build temporal-overlap graph and emit joint parents."""

def attach_parent_ids(leaves: list[Event], parents: list[Event]) -> list[Event]:
    """Set leaf.parent_id from parents.children. Return new leaves list."""
```

### 12.7 `pipeline.py`

```python
def extract_events(...): ...    # orchestration only; ~50 lines
```

## 13. Tests

Test file layout:

```
tests/affective_events/
├── conftest.py             # synthetic-signal fixtures
├── test_preprocessing.py   # build_blocks, interior mask invariants
├── test_baseline.py        # block-aware baseline correctness
├── test_detector.py        # prominence detection invariants
├── test_fusion.py          # joint event construction
├── test_pipeline.py        # end-to-end on synthetic data
└── test_invariants.py      # the §9.4 invariants on real-data fixture
```

**Mandatory test cases**:

1. **`test_baseline.py` — block exclusion**: A signal with two blocks
   identical except block A is uniformly elevated by `+0.2`. Block-aware
   baseline at A's interior must equal block B's median (within
   floating-point tolerance), not A's own median.

2. **`test_baseline.py` — long block per-frame**: A signal with one block of
   length `2 · radius_sec`. Verify per-frame computation produces different
   baseline values at the block's start vs end (because their context windows
   differ), while per-block computation would not.

3. **`test_detector.py` — seed width filter**: A *z*-trace with a 1-frame
   crossing of `z_seed` and a 2-second sustained crossing. With
   `seed_min_width_sec = 1.0`, only the sustained crossing produces an event.

4. **`test_detector.py` — shoulder boundary**: A clean inverted-U lobe with
   known peak and shoulder positions. Detected event extent matches
   shoulder positions to within one frame.

5. **`test_detector.py` — interior gating in shoulder**: A lobe whose
   shoulder would extend across a block boundary. Detected event must end
   at the last interior frame of the block.

6. **`test_detector.py` — same-sign merge, opposite-sign split**: Two
   close positive lobes → one merged event. One positive lobe followed by
   one negative lobe with no gap → two distinct events.

7. **`test_fusion.py` — joint construction**: Three overlapping leaves on
   different signals → one joint parent with all three as children.

8. **`test_pipeline.py` — golden recording**: Run `extract_events` on the
   `jamespiper-2026-2-9` recording used in the notebook. Assert the marked
   events are detected (each `MARKED_EVENTS` entry has at least one event
   in the output whose extent overlaps the marked range by ≥50%) and the
   total event count per signal stays within `[30, 80]` (the empirically
   verified range).

9. **`test_pipeline.py` — long-block edge case** (the §9.1 scenario):
   Synthetic signal with a 200-s block tonally shifted by `+0.15` and
   containing a 5-s peak inside it. Verify v1 behavior emits **one**
   long event spanning the full block; this test pins down the §9.1
   contract so a future change is intentional.

10. **`test_invariants.py`**: For every event in the golden-recording
    output, assert all invariants from §9.4.

## 14. Migration path

A four-step sequence, each step a separate PR.

### Step 1 — Add new module alongside the old one

Create `affective_events/v2/` with the new implementation. The public API
is `affective_events.v2.extract_events`. Old `affective_events.extract_events`
continues to work unchanged.

Tests in `tests/affective_events/v2/`. CI runs both test trees.

**Acceptance**: all v2 tests pass; v1 tests untouched.

### Step 2 — Wire v2 into the review tool, side-by-side with v1

In `review/`, add a config toggle that selects v1 or v2 events. Run both on
a labeled recording set; produce a side-by-side comparison HTML showing
where v1 and v2 disagree.

Manual review pass: do v2's events match human eyeball labels at least as
well as v1's? Document divergences.

**Acceptance**: review shows v2 ≥ v1 on the labeled set; remaining
divergences are understood and documented.

### Step 3 — Switch defaults to v2

Change `affective_events.extract_events` to call v2 internally. Keep
`affective_events.v1.extract_events` available for one release cycle in case
of rollback.

Update any caller in the codebase that accessed v1-specific event types
(`block_deviation`, `regime_shift`, etc.) — those callers now see only
`deviation` and `joint`.

**Acceptance**: all tests green, all callers updated.

### Step 4 — Delete v1

Remove `affective_events/v1/`, the old `detectors.py`, `validation.py`, and
the parts of `fusion.py`, `baseline.py`, `preprocessing.py`, `scoring.py`,
`config.py` that no longer have callers. Move v2 contents up to the package
root. Delete `ALGORITHM.md` (replaced by this `METHODOLOGY.md`).

**Acceptance**: package line count drops to ~400 LOC for the algorithmic
core. No regressions in review-tool acceptance tests.

## 15. Effort estimate

Rough estimates, assuming familiarity with the current codebase:

| Step | Effort |
|---|---|
| 1. New module + unit tests | ~2 days |
| 2. Side-by-side review wiring + manual review | ~1 day |
| 3. Switch defaults + caller updates | ~0.5 day |
| 4. Delete v1 + cleanup | ~0.5 day |
| **Total** | **~4 days** |

The dominant uncertainty is step 2 — if v2 produces qualitatively
different events from v1 in ways the review pass dislikes, we may iterate
on detection parameters or revisit §9.1 (within-block sub-pass for long
blocks). Budget one extra day for that contingency.

## 16. Open questions for the design review

These should be resolved before step 1 starts. None are blockers for the
algorithm; they are interface choices.

1. **Should `Event.extra` carry the signal-specific evidence (peak position
   inside event, shoulder *z* values, seed-cluster width) or stay empty?**
   Recommendation: carry it. It's small, free, and makes downstream review
   tooling much easier.

2. **Should `joint.peak_z` use quadrature or simple sum of children's
   `peak_z`?** Recommendation: quadrature (`sqrt(sum(z²))`) — matches
   "combined effect size across orthogonal dimensions" semantically.

3. **Should the cross-signal direction signature use `0.5` z as the
   threshold for `+ / − / 0`?** Recommendation: yes, matches the magnitude
   below which a child contributes nothing meaningful to the signature.

4. **Should we keep `Config` presets (`balanced` / `exploratory` /
   `conservative`)?** Recommendation: yes, but only varying `z_seed`. The
   other parameters reflect data geometry, not recall/precision tradeoff.
   Concretely:
   - `balanced()`: `z_seed=1.75` (the default)
   - `exploratory()`: `z_seed=1.5`
   - `conservative()`: `z_seed=2.25`

---

*End of methodology and implementation plan.*

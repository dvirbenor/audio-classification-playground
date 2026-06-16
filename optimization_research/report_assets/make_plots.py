"""Generate plots for OPTIMIZATION_REPORT.md (meeting-facing).
Run: uv run python optimization_research/report_assets/make_plots.py
All numbers sourced from optimization_research/*.md + baseline_results/*.
"""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.ticker import FuncFormatter

plt.rcParams.update({
    "font.size": 13,
    "axes.titlesize": 16,
    "axes.titleweight": "bold",
    "axes.labelsize": 13,
    "figure.dpi": 130,
    "savefig.bbox": "tight",
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.axisbelow": True,
})

OUT = "optimization_research/report_assets"
BLUE = "#2563eb"
GREEN = "#16a34a"
ORANGE = "#ea580c"
PURPLE = "#7c3aed"
GREY = "#94a3b8"


def annot(ax, bars, fmt="{:.2f}x", dy=0):
    for b in bars:
        h = b.get_height()
        ax.text(b.get_x() + b.get_width() / 2, h + dy, fmt.format(h),
                ha="center", va="bottom", fontweight="bold", fontsize=12)


# ---------------------------------------------------------------------------
# Plot 1 — GPU compound speed-up (cumulative, vs production "compiled fp32")
#          bars 1-4 = clean-benchmark projection; final bar = MEASURED in prod.
# ---------------------------------------------------------------------------
labels = ["Production\n(compiled fp32)", "+ FP16", "+ VAD gating", "+ GB202\nhardware",
          "+ MPS\nco-location"]
vals = [1.0, 1.7, 3.1, 11.0, 17.0]            # blended task-throughput / GPU; final = measured
colors = [GREY, BLUE, GREEN, PURPLE, ORANGE]
fig, ax = plt.subplots(figsize=(10, 5.4))
ax.axvspan(3.55, 4.45, color="#fef3c7", alpha=0.6, zorder=0)   # +MPS = per-GPU packing step
bars = ax.bar(labels, vals, color=colors, edgecolor="white", linewidth=1.5)
for b in bars[:4]:
    ax.text(b.get_x() + b.get_width() / 2, b.get_height() + 0.2, f"{b.get_height():.1f}x",
            ha="center", va="bottom", fontweight="bold", fontsize=12)
ax.text(bars[4].get_x() + bars[4].get_width() / 2, 17.0 + 0.2, "~17x\nmeasured",
        ha="center", va="bottom", fontweight="bold", fontsize=13, color="#7c2d12")
ax.text(2.55, 14.8, "MPS packing\n(+1.5x, per-GPU)", ha="center", fontsize=9, color="#92400e",
        style="italic")
ax.annotate("", xy=(3.62, 16.6), xytext=(3.1, 15.4),
            arrowprops=dict(arrowstyle="->", color="#92400e"))

ax.set_ylabel("throughput per GPU  (x vs old A10G fleet)")
ax.set_title("GPU inference: compounded speed-up per lever")
ax.set_ylim(0, 20)
fig.text(0.01, -0.02, "Blended task-throughput per GPU: measured 967 vs 56 jobs/GPU/h (steady-state). "
         "Bars 1-4 attribute it across levers; clean-benchmark ceiling ~23x. Per-task detail below.",
         fontsize=8.5, color="#64748b")
fig.savefig(f"{OUT}/gpu_compound_speedup.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 2 — Cross-GPU throughput + price/performance (affect, fp16+compile)
# ---------------------------------------------------------------------------
gpus = ["A10G\n$1.624/h", "L40S\n$2.242/h", "GB202 (RTX6000)\n$3.363/h"]
winps = [291, 447, 1039]
perdollar = [179, 199, 309]
fig, (a1, a2) = plt.subplots(1, 2, figsize=(12, 5))
b1 = a1.bar(gpus, winps, color=[GREY, BLUE, ORANGE], edgecolor="white", linewidth=1.5)
annot(a1, b1, fmt="{:.0f}", dy=8)
a1.set_ylabel("windows / sec   (affect, fp16+compile)")
a1.set_title("Raw throughput")
a1.set_ylim(0, 1220)
a1.text(1.52, 980, "3.6x A10G", ha="right", color=ORANGE, fontweight="bold")

b2 = a2.bar(gpus, perdollar, color=[GREY, BLUE, ORANGE], edgecolor="white", linewidth=1.5)
annot(a2, b2, fmt="{:.0f}", dy=3)
a2.set_ylabel("windows / sec  per  $/h")
a2.set_title("Price-performance  (higher = cheaper per unit work)")
a2.set_ylim(0, 360)
a2.text(1.52, 300, "1.73x A10G", ha="right", color=ORANGE, fontweight="bold")
fig.suptitle("Hardware comparison — GB202 is fastest AND cheapest per unit of work",
             fontsize=16, fontweight="bold", y=1.04)
fig.savefig(f"{OUT}/gpu_hardware_priceperf.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 3 — VAD CPU backfill throughput ramp (per pod)
# ---------------------------------------------------------------------------
stages = ["Old\n(threads, 1 core)", "1 process\nper core", "+ pin threads,\ntrim prefetch"]
rate = [53, 343, 660]
endlbl = ["53/h", "343/h   (6.5x)", "660/h   (+1.9x)"]
fig, ax = plt.subplots(figsize=(8.5, 3.6))
bars = ax.barh(stages, rate, color=[GREY, BLUE, GREEN], edgecolor="white", linewidth=1.5, height=0.62)
ax.invert_yaxis()   # progression reads top -> bottom
for b, r, lbl in zip(bars, rate, endlbl):
    ax.text(r + 14, b.get_y() + b.get_height() / 2, lbl, va="center", fontweight="bold", fontsize=11)
ax.set_title("VAD (CPU) backfill: 53 -> 660 arc/h/pod  (~12.5x)", fontsize=14)
ax.set_xlim(0, 900)
ax.tick_params(axis="y", labelsize=10.5)
fig.text(0.5, -0.12, "Silero VAD is GIL-bound -> 1 process per core (not threads); then pin math threads "
         "+ thin the decode prefetch\n(fewer workers / shallower lookahead, which also fixed a 20-pod OOM).",
         fontsize=8.5, color="#64748b", ha="center")
fig.savefig(f"{OUT}/vad_multiproc_ramp.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 4 — Per-task speed-up by lever (granularity, every per-task step)
# ---------------------------------------------------------------------------
tasks = ["affect\n(WavLM)", "disfluency\n(WavLM)", "emotion\n(e2v)"]
levers = [
    ("Compile",      [1.17, 1.18, 0.97], GREY),
    ("FP16",         [1.85, 1.88, 1.40], BLUE),
    ("VAD gating",   [2.2,  2.3,  2.1],  GREEN),
    ("GB202 (hw)",   [3.6,  3.6,  3.6],  ORANGE),
]
x = range(len(tasks))
n = len(levers)
w = 0.20
fig, ax = plt.subplots(figsize=(11, 5.4))
for k, (name, vals_, col) in enumerate(levers):
    off = (k - (n - 1) / 2) * w
    bars = ax.bar([i + off for i in x], vals_, w, label=name, color=col,
                  edgecolor="white", linewidth=1.0)
    annot(ax, bars, dy=0.03)
ax.set_xticks(list(x))
ax.set_xticklabels(tasks)
ax.set_ylabel("speed-up  (x)")
ax.set_title("Per-task speed-up by lever — each measured independently")
ax.set_ylim(0, 4.3)
ax.legend(loc="upper left", ncol=4, fontsize=11)
ax.axhline(1.0, color="#475569", lw=1, ls="--", alpha=0.6)
fig.text(0.01, -0.02, "Compile helps WavLM only (emotion ~1.0x); FP16 helps all 3; gating scales with "
         "silence (mean 2.34x); GB202 is a uniform ~3.6x. MPS co-location (x1.49) is a per-GPU packing "
         "win, not per-task. All steps event-safe.", fontsize=9, color="#64748b")
fig.savefig(f"{OUT}/per_task_levers.png")
plt.close(fig)

# ---------------------------------------------------------------------------
# Plot 5 — Per-task COMPOUND (cumulative across levers, measured endpoints)
# ---------------------------------------------------------------------------
stages = ["Production\n(compiled fp32)", "+ FP16", "+ VAD\ngating", "+ GB202\nhardware"]
series = [
    ("affect (WavLM)",     [1.0, 1.85, 4.1, 14.0], BLUE,   "o"),
    ("disfluency (WavLM)", [1.0, 1.88, 4.3, 14.0], GREEN,  "s"),
    ("emotion (e2v)",      [1.0, 1.40, 2.9,  9.0], ORANGE, "^"),
]
xs = list(range(len(stages)))
fig, ax = plt.subplots(figsize=(10, 5.4))
for name, vals_, col, mk in series:
    ax.plot(xs, vals_, marker=mk, color=col, lw=2.6, markersize=8, label=name)
ax.annotate("~14x  (WavLM)", (3, 14.0), xytext=(8, 0), textcoords="offset points", color=BLUE,
            fontweight="bold", fontsize=12, va="center")
ax.annotate("~9x  (emotion)", (3, 9.0), xytext=(8, 0), textcoords="offset points", color=ORANGE,
            fontweight="bold", fontsize=12, va="center")
ax.text(1.5, 10.6, "MPS then packs all 3 tasks onto one GPU\n->  ~17x throughput PER GPU (measured)",
        fontsize=9.5, color="#92400e", style="italic",
        bbox=dict(boxstyle="round", fc="#fef3c7", ec="#f59e0b", alpha=0.9))
ax.set_xticks(xs)
ax.set_xticklabels(stages)
ax.set_ylabel("speed-up vs old A10G dedicated  (x)")
ax.set_title("Per-task compound — dedicated-GPU-equivalent (modeled)")
ax.set_ylim(0, 16)
ax.set_xlim(-0.3, 3.95)
ax.legend(loc="upper left")
fig.text(0.5, -0.05, "Steps 1-3 = validated benchmark multipliers; +GB202 anchored to measured production "
         "(per-task dedicated-equivalent).\nEmotion gains less (FP16 1.4x, no compile). Deployed: MPS packs these into ~17x per GPU.",
         fontsize=8.5, color="#64748b", ha="center")
fig.savefig(f"{OUT}/per_task_compound.png")
plt.close(fig)

print("wrote 5 plots to", OUT)

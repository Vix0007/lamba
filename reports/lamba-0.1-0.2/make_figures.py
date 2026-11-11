"""Builds every figure in the Lamba technical report.

All numbers were recomputed from bundle/lamba_report_bundle/outputs_eval_results/
on 2026-10-02/03 (loop stats with the tail_stats_cmp.py logic, eval30 with
rumination_metric_v2_2_FROZEN.py) or read from the bundle's summary files.
Run: python make_figures.py   (writes figures/*.pdf)
"""
import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

OUT = os.path.join(os.path.dirname(os.path.abspath(__file__)), "figures")
os.makedirs(OUT, exist_ok=True)

plt.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "font.size": 8.5,
    "axes.titlesize": 9,
    "axes.labelsize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "axes.spines.top": False,
    "axes.spines.right": False,
    "axes.edgecolor": "#888782",
    "axes.linewidth": 0.6,
    "xtick.color": "#555450",
    "ytick.color": "#555450",
    "grid.color": "#e3e2dc",
    "grid.linewidth": 0.6,
    "pdf.fonttype": 42,
})
OURS = "#2a78d6"
OURS2 = "#86b6ef"
OTHER = "#c9c8c0"
BASE = "#898781"
INK = "#222220"

# ---------------- data ----------------
# loop test (150 prompts, greedy, 4,000 tokens): rates in %
loop = {
    "base":  dict(L=39.3, any=37.3, heavy=18.0, B=53.3),
    "c10":   dict(L=4.0,  any=2.7,  heavy=0.0,  B=39.3),
    "dpov2": dict(L=2.7,  any=2.7,  heavy=0.0,  B=35.3),
    "rp10":  dict(L=14.0, any=10.0, heavy=2.0,  B=35.3),
    "qwen":  dict(L=34.0, any=22.0, heavy=1.3,  B=70.7),
    "tcap":  dict(L=18.7, any=13.3, heavy=2.0,  B=49.3),
    "ds":    dict(L=26.0, any=17.3, heavy=0.0,  B=60.0),
    "glm":   dict(L=30.0, any=26.0, heavy=8.0,  B=64.7),
    "ilmu":  dict(L=19.3, any=18.7, heavy=15.3, B=22.7),
    "gem":   dict(L=None, any=None, heavy=None, B=10.7),
    "son":   dict(L=0.0,  any=0.0,  heavy=0.0,  B=4.7),
    "opus":  dict(L=None, any=None, heavy=None, B=None),
}
e30 = {  # correct %, redundant-after-verification %
    "dpov2": (98.8, 8.8), "c10": (98.8, 5.5), "qwen": (96.2, 3.9), "tcap": (100, 5.9),
    "ds": (100, 10.1), "glm": (91.2, 3.2), "ilmu": (75.0, 30.4), "gem": (100, 4.1),
    "son": (100, 0.0), "opus": (100, 1.7),
}
short = {"dpov2": "L0.2", "c10": "L0.1", "qwen": "Qwen", "tcap": "TCap", "ds": "DS",
         "glm": "GLM", "ilmu": "ILMU", "gem": "Gem.", "son": "Son.", "opus": "Opus"}
order = ["dpov2", "c10", "qwen", "tcap", "ds", "glm", "ilmu", "gem", "son", "opus"]
inv = lambda v: None if v is None else 100 - v
n_done = {"dpov2": 160, "c10": 183, "qwen": 203, "tcap": 205, "ds": 208, "glm": 188,
          "ilmu": 92, "gem": 197, "son": 57, "opus": 59}


def wilson(pct, n, z=1.96):
    """95% Wilson interval (in %) for a rate given in % over n trials."""
    k = round(pct / 100 * n)
    p = k / n
    den = 1 + z * z / n
    c = (p + z * z / (2 * n)) / den
    h = z * ((p * (1 - p) / n + z * z / (4 * n * n)) ** 0.5) / den
    return 100 * max(0.0, c - h), 100 * min(1.0, c + h)


def save(fig, name):
    fig.savefig(os.path.join(OUT, name), bbox_inches="tight")
    plt.close(fig)


# ---------------- Fig: version trajectory ----------------
def fig_trajectory():
    vers = ["base", "c10", "dpov2"]
    xl = ["Base", "0.1", "0.2"]
    panels = [("Loop-free (100 $-$ L)", "L"), ("Clean endings (100 $-$ R$_{any}$)", "any"),
              ("Within budget (100 $-$ B)", "B")]
    fig, axes = plt.subplots(1, 4, figsize=(7.0, 2.2))
    for ax, (title, k) in zip(axes[:3], panels):
        refs = sorted([(o, inv(loop[o][k])) for o in order[2:] if loop[o][k] is not None],
                      key=lambda t: -t[1])
        for o, v in refs:
            ax.axhline(v, color=BASE, lw=0.6, ls=(0, (2, 2)), alpha=0.7, zorder=1)
        # merge labels of lines closer than 4 points, then keep >= 8 units between labels
        groups = []
        for o, v in refs:
            if groups and groups[-1][1] - v < 4:
                groups[-1][0].append(short[o])
            else:
                groups.append([[short[o]], v])
        last = 1e9
        for names, v in groups:
            y = min(v, last - 8)
            last = y
            ax.text(2.28, y, "/".join(names), fontsize=5.8, color="#6d6c66", va="center")
        ys = [inv(loop[v][k]) for v in vers]
        ax.plot(range(3), ys, color=OURS, lw=1.6, zorder=3)
        ax.scatter(range(3), ys, color=[BASE, OURS, OURS], s=18, zorder=4, edgecolor="white", lw=0.8)
        ax.scatter([2.12], [inv(loop["rp10"][k])], facecolor="white", edgecolor=OURS, s=16, lw=1.1, zorder=4)
        for i, y in enumerate(ys):
            ax.text(i, y + 4, f"{y:.1f}", ha="center", fontsize=6.5, color=INK)
        ax.set_xlim(-0.3, 2.75)
        ax.set_ylim(0, 108)
        ax.set_xticks(range(3), xl)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=7.5)
    ax = axes[3]
    gp = [151 / 1.98, 153 / 1.98, 157 / 1.98]
    mm = [76.74, 77.71, 77.91]
    ax.plot(range(3), gp, color=OURS, lw=1.6, marker="o", ms=4, mfc=OURS, mec="white", label="GPQA-D")
    ax.plot(range(3), mm, color=OURS, lw=1.2, ls="--", marker="s", ms=3.5, mfc=OURS, mec="white", label="MalayMMLU")
    ax.plot([1], [147 / 1.98], marker="o", ms=4, mfc="white", mec=OURS, ls="none", label="GPQA-D rerun")
    for i, (g, m) in enumerate(zip(gp, mm)):
        ax.text(i, g + 0.6, f"{g:.1f}", ha="center", fontsize=6, color=INK)
    ax.set_ylim(70, 85)
    ax.set_xlim(-0.3, 2.3)
    ax.set_xticks(range(3), xl)
    ax.set_yticks([70, 75, 80, 85])
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.set_title("Capability (accuracy %, axis 70–85)", fontsize=7.5)
    ax.legend(frameon=False, fontsize=6, loc="upper left")
    fig.tight_layout(w_pad=0.6)
    save(fig, "trajectory.pdf")


# ---------------- Fig: cross-model bars ----------------
def fig_bars():
    g150 = lambda o: 150
    panels = [
        ("Loop-free", lambda o: inv(loop[o]["L"]), g150),
        ("Clean endings", lambda o: inv(loop[o]["any"]), g150),
        ("No runaway loops", lambda o: inv(loop[o]["heavy"]), g150),
        ("Finishes within budget", lambda o: inv(loop[o]["B"]), g150),
        ("eval30 accuracy", lambda o: e30[o][0], lambda o: 240),
        ("Stops after verifying", lambda o: 100 - e30[o][1], lambda o: n_done[o]),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(7.0, 3.9))
    for ax, (title, get, nfun) in zip(axes.flat, panels):
        for i, o in enumerate(order):
            v = get(o)
            c = OURS if o == "dpov2" else OURS2 if o == "c10" else OTHER
            if v is None:
                ax.text(i, 2, "n/a", ha="center", fontsize=5.5, color="#6d6c66", rotation=90, va="bottom")
                continue
            ax.bar(i, v, width=0.78, color=c, zorder=2)
            lo, hi = wilson(v, nfun(o))
            ax.plot([i, i], [lo, hi], color="#3d3c38", lw=0.7, zorder=3)
            ax.plot([i - 0.15, i + 0.15], [lo, lo], color="#3d3c38", lw=0.7, zorder=3)
            ax.plot([i - 0.15, i + 0.15], [hi, hi], color="#3d3c38", lw=0.7, zorder=3)
            lab = "100" if v == 100 else f"{v:.1f}"
            ax.text(i, hi + 1.5, lab, ha="center", fontsize=5.2, color=INK if o in ("dpov2", "c10") else "#555450",
                    rotation=90, va="bottom")
        ax.set_ylim(0, 122)
        ax.set_yticks([0, 25, 50, 75, 100])
        ax.set_xticks(range(len(order)), [short[o] for o in order], rotation=90, fontsize=6.3)
        ax.grid(axis="y")
        ax.set_axisbelow(True)
        ax.set_title(title, fontsize=8)
    fig.tight_layout(h_pad=0.8, w_pad=0.6)
    save(fig, "crossmodel.pdf")


# ---------------- Fig: repetition-penalty sweep ----------------
def fig_rp():
    rp = [1.0, 1.05, 1.1, 1.15, 1.2]
    pre = [22.5, 26.3, 31.2, 3.8, -62.5]    # pre-registered: exact repetition OR budget hit
    liq = [23.1, 28.6, 39.6, 15.4, -42.9]   # Liquid-style loop OR budget hit
    fig, ax = plt.subplots(figsize=(3.3, 2.2))
    ax.axhline(30, color="#d03b3b", lw=0.8, ls="--")
    ax.text(1.2, 33, "30% bar", ha="right", fontsize=6.5, color="#d03b3b")
    ax.axhline(0, color="#888782", lw=0.6)
    ax.plot(rp, pre, color=OURS, lw=1.6, marker="o", ms=4, mec="white", label="pre-registered detector")
    ax.plot(rp, liq, color=BASE, lw=1.2, ls="--", marker="s", ms=3.5, mec="white", label="Liquid-style detector")
    for x, y in zip(rp, pre):
        ax.text(x, y - 10, f"{y:.1f}", ha="center", fontsize=6.3, color=OURS)
    ax.legend(frameon=False, fontsize=6, loc="lower left")
    ax.set_xticks(rp)
    ax.set_xlabel("repetition penalty")
    ax.set_ylabel("relative reduction vs base (%)")
    ax.set_ylim(-75, 55)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    save(fig, "rp_sweep.pdf")


# ---------------- Fig: letter-bias checkpoint oscillation ----------------
def fig_letter():
    steps = [10, 20, 30, 40, 50, 60, 90]
    gap = [-1.92, 11.65, -4.03, -4.66, -8.43, -13.7, -7.84]
    fig, ax = plt.subplots(figsize=(3.3, 2.1))
    ax.axhspan(-2, 2, color="#0ca30c", alpha=0.12, lw=0)
    ax.text(92, 2.6, "target $|$gap$|\\leq$2", ha="right", fontsize=6.5, color="#006300")
    ax.axhline(-11.23, color=BASE, lw=0.8, ls="--")
    ax.text(75, -12.9, "base $-$11.2", ha="center", fontsize=6.5, color="#555450")
    ax.axhline(0, color="#888782", lw=0.5)
    ax.plot(steps, gap, color=OURS, lw=1.3, marker="o", ms=4, mec="white")
    ax.scatter([10], [-1.92], s=55, facecolor="none", edgecolor="#006300", lw=1.1, zorder=5)
    for x, y in zip(steps, gap):
        if x == 10:
            ax.text(13, y - 0.4, f"{y:+.2f} (shipped)", ha="left", va="center", fontsize=6.2, color="#006300")
        else:
            ax.text(x, y + (1.6 if y > -10 else -3.2), f"{y:+.1f}", ha="center", fontsize=6.2)
    ax.set_xlabel("training step (same run)")
    ax.set_ylabel("A$-$B accuracy gap (pt)")
    ax.set_ylim(-17, 16)
    ax.set_xticks(steps)
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    save(fig, "letter_checkpoints.pdf")


# ---------------- Fig: post-correct rumination, integral family ----------------
def fig_pcr():
    # original (import-blocking) sandbox, 2026-09-29 runs; sandbox-B re-run 2026-10-03 (70 runs each)
    rows = [("Lamba 0.1", 16.0, 50, OURS), ("Stock Qwen3.8", 32.9, 70, OTHER),
            ("seed 42", 36.0, 50, "#b9b8b0"), ("seed 1337", 58.0, 50, "#9e9d95"),
            ("Lamba 0.1", 0.0, 70, OURS), ("Lamba 0.2", 1.4, 70, OURS)]
    ys = [7, 6, 5, 4, 1.5, 0.5]
    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    for (name, v, k, c), y in zip(rows, ys):
        ax.barh(y, v, color=c, height=0.7, zorder=2)
        lo, hi = wilson(v, k)
        ax.plot([lo, hi], [y, y], color="#3d3c38", lw=0.7, zorder=3)
        ax.text(max(hi, v) + 1.2, y, f"{v:.1f}%  (n={k})", va="center", fontsize=6.6)
    ax.set_yticks(ys, [r[0] for r in rows], fontsize=7.2)
    ax.text(79, 7.75, "original sandbox (imports blocked)", ha="right", fontsize=6.6, color="#555450")
    ax.text(79, 2.25, "sandbox-B (imports allowed)", ha="right", fontsize=6.6, color="#555450")
    ax.axhline(2.9, color="#c3c2b7", lw=0.6)
    ax.set_xlim(0, 80)
    ax.set_ylim(-0.2, 8.2)
    ax.set_xlabel("post-correct rumination rate (%)")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    save(fig, "pcr_integral.pdf")


# ---------------- Fig: how an AntiDoom training pair is built ----------------
def fig_method():
    """Real pair from antidoom_data/all_ftpo_pairs_antidoom8.jsonl (source_sample_id kssm_0163-03):
    base-model output on a Malay prompt, repeated unit n_occ=21, finish=length,
    rejected ' mengatakan' (base log p = -1.13), 16 chosen alternatives from the base model's top-20."""
    from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
    RED, GREEN, GREY, ORANGE = "#d03b3b", "#0f8a3c", "#e6e5df", "#f4a582"
    fig = plt.figure(figsize=(7.0, 3.35))
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 100); ax.set_ylim(0, 48); ax.axis("off")

    def box(x, y, w, h, fc, ec="none", lw=0.0, r=0.6):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=f"round,pad=0,rounding_size={r}",
                                    fc=fc, ec=ec, lw=lw))

    # (a) the looping output
    ax.text(1, 46, "(a) A base-model output that loops (Malay prompt; schematic, not to scale)",
            fontsize=8, weight="bold", color=INK, va="top")
    y0, h = 36.5, 4.5
    box(1, y0, 26, h, GREY); ax.text(14, y0 + h / 2, "reasoning", ha="center", va="center", fontsize=7, color="#555450")
    box(27.5, y0, 11, h, ORANGE); ax.text(33, y0 + h / 2, "occurrence 1", ha="center", va="center", fontsize=6.6)
    box(39, y0, 17, h, GREY); ax.text(47.5, y0 + h / 2, "more reasoning", ha="center", va="center", fontsize=7, color="#555450")
    box(56.5, y0, 11, h, ORANGE, ec=RED, lw=1.3); ax.text(62, y0 + h / 2, "occurrence 2", ha="center", va="center", fontsize=6.6)
    for j in range(4):
        box(68 + j * 5.2, y0, 4.8, h, ORANGE)
    ax.text(89.3, y0 + h / 2, "… 21 in total", ha="left", va="center", fontsize=6.8)
    ax.text(98.5, y0 - 1.6, "stopped at the 4,000-token cap", ha="right", va="top", fontsize=6.4, color="#555450")
    ax.add_patch(FancyArrowPatch((56.6, y0 + h + 3.2), (56.6, y0 + h + 0.2), arrowstyle="-|>", mutation_scale=8, color=RED, lw=1.1))
    ax.text(57.4, y0 + h + 2.4, "training position: first token of the 2nd occurrence", fontsize=6.6, color=RED, va="center")

    renderer = fig.canvas.get_renderer()

    def right(t, gap=0.6):
        bb = t.get_window_extent(renderer)
        return ax.transData.inverted().transform((bb.x1, bb.y0))[0] + gap

    def cursor(x, y):
        # a red insertion bar drawn as a line (no font glyph needed)
        ax.plot([x, x], [y - 1.4, y + 1.4], color=RED, lw=1.6, solid_capstyle="butt")

    # (b) the context at the training position
    ax.text(1, 30, "(b) Text before the training position", fontsize=8, weight="bold", color=INK, va="top")
    t = ax.text(1.5, 25.2, "…Tetapi konteks di sini", fontsize=7, va="center")
    ax.text(right(t, 0.9), 25.2, "mengatakan \"Perkhidmatan Feri Malaysia\", mungkin merujuk…", fontsize=7,
            va="center", bbox=dict(boxstyle="round,pad=0.15", fc=ORANGE, ec="none"))
    ax.text(1.5, 21.6, "“…But the context here says ‘Malaysia Ferry Service’, possibly referring to…”  (occurrence 1)",
            fontsize=6.3, style="italic", color="#555450", va="center")
    t = ax.text(1.5, 16.4, "…dimansuhkan pada tahun 1996.  Tetapi soalan", fontsize=7, va="center")
    cursor(right(t, 0.8), 16.4)
    t = ax.text(1.5, 12.8, "“…was discontinued in 1996.  But the question", fontsize=6.3, style="italic",
                color="#555450", va="center")
    cursor(right(t, 0.7), 12.8)
    ax.text(1.5, 6.6, "Here the model wrote “mengatakan” (says), starting occurrence 2,\n"
                      "and went on to repeat the sentence 21 times in total until the token cap.",
            fontsize=6.5, color="#555450", va="center")

    # (c) the preference pair
    ax.text(55, 30, "(c) The training pair at that position", fontsize=8, weight="bold", color=INK, va="top")
    ax.text(55.5, 24.8, "rejected", fontsize=6.6, color=RED, va="center", weight="bold")
    t = ax.text(64, 24.8, "mengatakan", fontsize=7.2, va="center",
                bbox=dict(boxstyle="round,pad=0.25", fc="#fbe3e3", ec=RED, lw=0.8))
    ax.text(right(t, 1.6), 24.8, "‘says’: starts the repeat (base p = 0.32)", fontsize=6.2, color="#555450", va="center")
    ax.text(55.5, 18.6, "chosen", fontsize=6.6, color=GREEN, va="center", weight="bold")
    xs = 64
    for tok, gl in [("bertanya", "asks"), ("menyebut", "mentions"), ("ini", "this"), ("tidak", "not")]:
        t = ax.text(xs, 18.6, tok, fontsize=7.2, va="center",
                    bbox=dict(boxstyle="round,pad=0.25", fc="#ddf1e3", ec=GREEN, lw=0.8))
        ax.text(xs, 15.0, gl, fontsize=6, color="#555450", va="center")
        xs = right(t, 2.4)
    ax.text(xs, 18.6, "+12 more", fontsize=6.4, color="#555450", va="center")
    ax.text(55.5, 9.6, "FTPO raises each chosen logit above the rejected one by a margin,\n"
                       "while a tether keeps every other logit close to the base model.\n"
                       "Chosen tokens are the base model's own top-20 alternatives.",
            fontsize=6.6, color=INK, va="center")
    save(fig, "method.pdf")


# ---------------- Fig 1: training lineage on g150 ----------------
def fig_lineage():
    # rates (%) on g150, recomputed from outputs_eval_results/loop_*_g150.jsonl
    stages = ["Base", "v8", "v9b", "v9c", "malay-\nfinal", "Lamba\n0.1", "Lamba\n0.2"]
    what = ["", "AntiDoom\npairs", "own-output\nround", "+ letter\ntie", "Malay\nround", "+ letter stage\n+ rp 1.1", "+ DPO\n(tools)"]
    L = [39.3, 26.0, 24.0, 22.7, 16.7, 4.0, 2.7]
    Ra = [37.3, 24.0, 18.0, 16.7, 12.7, 2.7, 2.7]
    Rh = [18.0, 12.7, 4.7, 2.7, 0.7, 0.0, 0.0]
    B = [53.3, 48.7, 41.3, 41.3, 41.3, 39.3, 35.3]
    x = list(range(len(stages)))
    fig, ax = plt.subplots(figsize=(7.0, 2.9))
    ax.axvspan(4.5, 6.5, color="#eaf2fc", lw=0, zorder=0)
    ax.text(5.5, 57.5, "served at repetition penalty 1.1", ha="center", fontsize=7, color="#1c5cab")
    ax.text(2.0, 57.5, "training only (repetition penalty 1.0)", ha="center", fontsize=7, color="#555450")
    series = [("budget hit $B$", B, "#9e9d95", "--"), ("Liquid-style loop $L$", L, "#eb6834", "-"),
              ("tail repetition $R_{any}$", Ra, OURS, "-"), ("runaway loop $R_{heavy}$", Rh, "#1baf7a", "-")]
    for name, ys, c, ls in series:
        if ls == "-":
            band = [wilson(v, 150) for v in ys]
            ax.fill_between(x, [b[0] for b in band], [b[1] for b in band], color=c, alpha=0.12, lw=0, zorder=1)
        ax.plot(x, ys, color=c, lw=1.7 if ls == "-" else 1.2, ls=ls, marker="o", ms=3.8, mec="white", mew=0.8, zorder=3)
        ax.text(6.18, ys[-1] + {"Liq": 6.0, "tai": 1.8, "run": -3.2, "bud": 0}[name[:3]],
                name, fontsize=6.8, color=c, va="center")
    for i, v in enumerate(Rh):
        ax.text(i, v - 3.6, f"{v:.1f}", ha="center", fontsize=6, color="#13805a")
    # penalty-only reference: the untouched base model at repetition penalty 1.1 (reviewer pack, 2026-10-03)
    for v, c in [(8.7, "#eb6834"), (4.7, OURS), (36.0, "#9e9d95")]:
        ax.plot([4.55, 6.1], [v, v], color=c, lw=1.0, ls=(0, (1, 1.5)), zorder=2)
    ax.text(4.58, 9.9, "base + penalty only", fontsize=6.2, color="#555450", va="bottom")
    ax.set_xticks(x, [s if not w else f"{s}" for s, w in zip(stages, what)], fontsize=7.2)
    for i, w in enumerate(what):
        ax.text(i, -17.5, w, ha="center", va="top", fontsize=5.9, color="#6d6c66")
    ax.set_xlim(-0.4, 7.6)
    ax.set_ylim(-6, 61)
    ax.set_yticks([0, 10, 20, 30, 40, 50])
    ax.set_ylabel("% of g150 answers (lower is better)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    save(fig, "lineage.pdf")


if __name__ == "__main__":
    fig_method()
    fig_lineage()
    fig_trajectory()
    fig_bars()
    fig_rp()
    fig_letter()
    fig_pcr()
    print("wrote", sorted(os.listdir(OUT)))

#!/usr/bin/env python3
"""Builds the Lamba 0.4 comparison charts (SVG, real text) and Markdown tables from one data table.

Usage: python build_charts.py   (writes the .svg files next to this script and tables.md)
"""
import os
import textwrap
from xml.sax.saxutils import escape

OUT = os.path.dirname(os.path.abspath(__file__))

# ---------------------------------------------------------------- data (single source of truth)
# [label, Lamba 0.4, Lamba 0.3, note, stock Qwen or None, stock note]
VS = [
    ("0.4 ahead", [
        ("Ibu negara / negeri questions correct", 96.1, 61.7, "128 samples each", None, ""),
        ('Does not self-introduce on "X tu apa?" (100 minus rate)', 97.4, 68.4, "self-intro 2.6% vs 31.6%, n=76", None, ""),
        ("Calls a search tool when it should (first turn)", 99.2, 72.6, "248 held-out prompts, first-turn subset n=124", None, ""),
        ("Refuses harmful prompts, with constitution", 93.3, 77.6, "StrongREJECT, n=313", 83.1, "stock Qwen 83.1%"),
        ("Malaysian facts correct", 81.9, 73.9, "348 samples each", 80.2, "stock Qwen 80.2%, same held-out facts"),
        ("Avoids needless searches (100 minus over-search)", 98.8, 94.6, "over-search 1.2% vs 5.4%, 760 samples", None, ""),
        ("MalayMMLU generative, 5k", 84.6, 82.5, "0.4 is the mean of two runs (84.74, 84.50)", None, ""),
    ]),
    ("Level, within noise", [
        ("GPQA-Diamond (160 vs 155 of 198)", 80.8, 78.3, "noise about plus or minus 6 questions", None, ""),
        ("Finishes within token budget", 74.8, 72.9, "g150; most budget hits are long legitimate answers", 34.0, "stock Qwen at penalty 1.1; 29.3 with no penalty"),
        ("Refuses harmful prompts, no system prompt", 96.2, 95.5, "StrongREJECT, n=313", 95.5, "stock Qwen 95.5%"),
        ("Clean endings (100 minus tail repetition)", 97.0, 97.3, "g150 at penalty 1.05", 90.0, "stock Qwen at penalty 1.1; 78.0 with no penalty"),
    ]),
    ("0.3 ahead", [
        ('Says "Lamba" on held-out identity items', 96.2, 99.7, '0.3 partly answers "Lamba" even when another model is only mentioned', None, ""),
        ("Loop-free (100 minus Liquid-style loop rate)", 93.5, 96.0, "g150 at penalty 1.05", 78.0, "stock Qwen at penalty 1.1; 66.0 with no penalty"),
    ]),
]
KPIS = [("99.2%", ["calls a search tool when", "it should (first turn)"], "+26.6 pts vs 0.3", "0.3: 72.6%"),
        ("84.6%", ["MalayMMLU,", "generative 5k"], "+2.1 pts vs 0.3", "0.3: 82.5%"),
        ("8.0/10", ["chat tone: friendly,", "blind judge"], "about +2.3 vs 0.3", "0.3: about 5.7"),
        ("93.3%", ["refuses harmful prompts", "(with the constitution)"], "+15.7 pts vs 0.3", "0.3: 77.6%")]

MODELS = [("v04", "Lamba 0.4", "Lamba 0.4"), ("v03", "Lamba 0.3", "Lamba 0.3"), ("qwen", "Qwen3.8-27B (stock)", "Stock Qwen3.8-27B"),
          ("tcap", "ThinkingCap-Qwen3.8-27B", "ThinkingCap"), ("ds", "DeepSeek v4.1 Flash", "DeepSeek v4.1 Flash"),
          ("glm", "GLM-5.3 Flash", "GLM-5.3 Flash"), ("ilmu", "ILMU v3.1", "ILMU v3.1"), ("gem", "Gemini 3.8 Flash", "Gemini 3.8 Flash"),
          ("son", "Claude Sonnet 5.5", "Claude Sonnet 5.5"), ("opus", "Claude Opus 5.5", "Claude Opus 5.5")]
RAW = {"v04": dict(L=6.5, any=3.0, B=25.2, acc=99.6, red=4.1), "v03": dict(L=4.0, any=2.7, B=27.1, acc=99.6, red=4.9),
       "qwen": dict(L=34.0, any=22.0, B=70.7, acc=96.2, red=3.9), "tcap": dict(L=18.7, any=13.3, B=49.3, acc=100, red=5.9),
       "ds": dict(L=26.0, any=17.3, B=60.0, acc=100, red=10.1), "glm": dict(L=30.0, any=26.0, B=64.7, acc=91.2, red=3.2),
       "ilmu": dict(L=19.3, any=18.7, B=22.7, acc=75.0, red=30.4), "gem": dict(L=None, any=None, B=10.7, acc=100, red=4.1),
       "son": dict(L=0.0, any=0.0, B=4.7, acc=100, red=0.0), "opus": dict(L=None, any=None, B=None, acc=100, red=1.7)}
inv = lambda v: None if v is None else 100 - v
METS = [("L", "Loop-free", "100 minus the share with a 60-character block repeated 4 or more times", lambda r: inv(r["L"])),
        ("any", "Clean endings", "100 minus the share with any repetition in the last 2,500 characters", lambda r: inv(r["any"])),
        ("B", "Finishes within budget", "100 minus the share that hit the 4,000-token cap", lambda r: inv(r["B"])),
        ("acc", "Tool maths correct", "share of 240 eval30 runs that were correct", lambda r: r["acc"]),
        ("red", "Stops after verifying", "100 minus the share of solved runs that kept calling tools after verifying", lambda r: inv(r["red"]))]
CAP = [("Lamba base", "151", "8,900", "83.60", "369"), ("Coder390 base", "157", "n/a", "83.84", "n/a"),
       ("Lamba 0.1", "162", "8,950", "83.26", "377"), ("Lamba 0.2", "154", "8,741", "82.84", "372"),
       ("Lamba 0.3", "155", "8,654", "82.50", "422"), ("Lamba 0.4", "160", "7,260", "84.62", "278")]

f1 = lambda v: f"{round(v + 1e-9, 1):.1f}"
sg = lambda v: ("+" if v >= 0 else "−") + f1(abs(v))
ordn = lambda n: f"{n}{'th' if 10 <= n % 100 <= 20 else {1: 'st', 2: 'nd', 3: 'rd'}.get(n % 10, 'th')}"
esc = escape

STYLE = """
:root{--bg:#ffffff;--t1:#141413;--t2:#52514e;--t3:#898781;--rule:#e3e2dc;--tile:#f5f4f0;--c4:#2a78d6;--c3:#85b7eb;--cx:#9a988f}
@media (prefers-color-scheme:dark){:root{--bg:#1a1a19;--t1:#f0efec;--t2:#c3c2b7;--t3:#8f8d86;--rule:#33332f;--tile:#242422;--c4:#3987e5;--c3:#9ec1ea;--cx:#7a7872}}
text{font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Helvetica,Arial,sans-serif;font-size:12px;fill:var(--t2)}
.t1{fill:var(--t1)}.t3{fill:var(--t3)}.s{font-size:11px}.h{font-size:15px;font-weight:600;fill:var(--t1)}.m{font-weight:600}
.big{font-family:Georgia,"Times New Roman",serif;font-size:30px;fill:var(--c4)}.end{text-anchor:end}
.c4{fill:var(--c4)}.c3{fill:var(--c3)}.cx{fill:var(--cx)}.tile{fill:var(--tile)}.rule{stroke:var(--rule);stroke-width:1}
"""


def lines(x, y, parts, cls="", lh=15):
    return "".join(f'<text x="{x}" y="{y + i * lh}" class="{cls}">{esc(p)}</text>' for i, p in enumerate(parts))


def svg_open(w, h, title, desc):
    return (f'<svg xmlns="http://www.w3.org/2000/svg" width="{w}" height="{h}" viewBox="0 0 {w} {h}" role="img" aria-labelledby="ttl dsc">'
            f'<title id="ttl">{esc(title)}</title><desc id="dsc">{esc(desc)}</desc><style>{STYLE}</style>'
            f'<rect width="{w}" height="{h}" rx="12" fill="var(--bg)"/>')


# ---------------------------------------------------------------- chart 1: 0.4 vs 0.3 vs stock Qwen
def chart_vs():
    W = 760
    total = sum(len(g[1]) for g in VS)
    take = f"Lamba 0.4 leads Lamba 0.3 on {len(VS[0][1])} of {total} tests, is level on {len(VS[1][1])} and trails on {len(VS[2][1])}."
    o = []
    y = 34
    o.append(f'<text x="24" y="{y}" class="h">{esc(take)}</text>')
    y += 16
    tw, gap = 172, 10
    for i, (big, lab, gain, was) in enumerate(KPIS):
        x = 24 + i * (tw + gap)
        o.append(f'<rect x="{x}" y="{y}" width="{tw}" height="104" rx="8" class="tile"/>')
        o.append(f'<text x="{x + 14}" y="{y + 38}" class="big">{esc(big)}</text>')
        o.append(lines(x + 14, y + 56, lab, "s", 13))
        o.append(f'<text x="{x + 14}" y="{y + 88}" class="m t1">{esc(gain)}</text>')
        o.append(f'<text x="{x + 14}" y="{y + 101}" class="s t3">{esc(was)}</text>')
    y += 104 + 34
    o.append(f'<line x1="24" x2="{W - 24}" y1="{y - 18}" y2="{y - 18}" class="rule"/>')
    o.append(f'<text x="24" y="{y}" class="h">Lamba 0.4 against Lamba 0.3 and stock Qwen</text>')
    y += 18
    sub = ('Every bar is "higher is better" (rates where lower is better are shown as 100 minus the rate). Rows are sorted by the '
           "gap between 0.4 and 0.3, shown on the right. Stock Qwen3.8-27B appears only where we ran it. Lamba 0.4's base model is "
           "different from 0.3's, so the comparison mixes training and base.")
    o.append(lines(24, y, textwrap.wrap(sub, 118), "s", 14))
    y += 14 * len(textwrap.wrap(sub, 118)) + 8
    lx = 24
    for cls, name, wd in [("c4", "Lamba 0.4", 84), ("c3", "Lamba 0.3", 84), ("cx", "Stock Qwen3.8-27B", 140)]:
        o.append(f'<rect x="{lx}" y="{y - 9}" width="10" height="10" rx="2" class="{cls}"/><text x="{lx + 15}" y="{y}" class="s">{name}</text>')
        lx += wd
    y += 12
    X0, SC = 280, 3.5
    for gname, items in VS:
        y += 10
        o.append(f'<line x1="24" x2="{W - 24}" y1="{y}" y2="{y}" class="rule"/>')
        y += 18
        o.append(f'<text x="24" y="{y}" class="m">{esc(gname)} · {len(items)}</text>')
        y += 6
        for lab, a, b, note, q, qn in sorted(items, key=lambda r: -abs(r[1] - r[2])):
            ls = textwrap.wrap(lab, 38)
            bh = 11 + 3 + 7 + ((3 + 7) if q is not None else 0)
            rh = max(len(ls) * 14, bh) + 14
            top = y
            ly = top + (rh - len(ls) * 14) / 2 + 11
            by = top + (rh - bh) / 2
            tip = f"{lab}. Lamba 0.4: {f1(a)}. Lamba 0.3: {f1(b)}. " + (f"Stock Qwen: {f1(q)}. " if q is not None else "Stock Qwen: not run. ") + f"Gap: {sg(a - b)} points. {note}." + (f" {qn}." if qn else "")
            o.append(f"<g><title>{esc(tip)}</title>")
            o.append(lines(24, ly, ls, "", 14))
            o.append(f'<rect x="{X0}" y="{by}" width="{a * SC:.1f}" height="11" rx="3" class="c4"/><text x="{X0 + a * SC + 6:.1f}" y="{by + 9.5}" class="s t1">{f1(a)}</text>')
            o.append(f'<rect x="{X0}" y="{by + 14}" width="{b * SC:.1f}" height="7" rx="3" class="c3"/><text x="{X0 + b * SC + 6:.1f}" y="{by + 21.5}" class="s">{f1(b)}</text>')
            if q is not None:
                o.append(f'<rect x="{X0}" y="{by + 24}" width="{q * SC:.1f}" height="7" rx="3" class="cx"/><text x="{X0 + q * SC + 6:.1f}" y="{by + 31.5}" class="s">{f1(q)}</text>')
            o.append(f'<text x="{W - 24}" y="{top + rh / 2 + 4}" class="m t1 end">{sg(a - b)}</text></g>')
            y += rh
    y += 22
    foot = ("Lamba 0.1 to 0.4 loop results are like-for-like runs at repetition penalty 1.05. Stock Qwen is shown at penalty 1.1 (no-penalty "
            "values are in each row's hover text) and favours stock Qwen relative to its provider default. Own test sets and harnesses on "
            "one machine, not independently reproduced. Lamba 0.4 is on a different base than 0.1 to 0.3.")
    fl = textwrap.wrap(foot, 126)
    o.append(lines(24, y, fl, "s t3", 14))
    h = y + 14 * len(fl) + 14
    desc = take + " Stock Qwen3.8-27B is shown where a run exists."
    return svg_open(W, h, "Lamba 0.4 against Lamba 0.3 and stock Qwen", desc) + "".join(o) + "</svg>\n"


# ---------------------------------------------------------------- chart 2: against other models (small multiples)
def chart_others():
    W, PW, CG = 760, 344, 24
    o, y = [], 34
    o.append(f'<text x="24" y="{y}" class="h">Lamba 0.4 against other models</text>')
    y += 8
    note = ("Higher is better on every panel. Rows highlighted in the tile colour are Lamba 0.4. Lamba runs are at repetition penalty 1.05; "
            "the other models ran at provider defaults with no penalty.")
    o.append(lines(24, y + 14, textwrap.wrap(note, 126), "s", 14))
    y += 14 * len(textwrap.wrap(note, 126)) + 20
    short = {k: s for k, _, s in MODELS}
    colors = {"v04": "c4", "v03": "c3"}
    rank_text = []
    # panel height = header (title, rank line, wrapped definition) + one 20px row per model that has data
    def panel_h(m):
        n = sum(1 for k, _, _ in MODELS if m[3](RAW[k]) is not None)
        return 29 + 13 * len(textwrap.wrap(m[2], 58)) + 6 + 20 * n + 26
    nrows = (len(METS) + 1) // 2
    row_h = [max(panel_h(METS[i]) for i in range(r * 2, min(r * 2 + 2, len(METS)))) for r in range(nrows)]
    row_y = [y + sum(row_h[:r]) for r in range(nrows)]
    for idx, (mid, title, sdef, fn) in enumerate(METS):
        col, row = idx % 2, idx // 2
        px = 24 + col * (PW + CG)
        py = row_y[row]
        vals = [(k, fn(RAW[k])) for k, _, _ in MODELS]
        ok = sorted([(k, v) for k, v in vals if v is not None], key=lambda t: -t[1])
        rank = 1 + sum(1 for k, v in ok if v > fn(RAW["v04"]))
        rank_text.append(f"{title}: {ordn(rank)} of {len(ok)}")
        o.append(f'<text x="{px}" y="{py}" class="m t1">{esc(title)}</text>')
        o.append(f'<text x="{px}" y="{py + 15}" class="s t1 m">Lamba 0.4 ranks {ordn(rank)} of {len(ok)}</text>')
        dl = textwrap.wrap(sdef, 58)
        o.append(lines(px, py + 29, dl, "s t3", 13))
        ry = py + 29 + 13 * len(dl) + 6
        for k, v in ok:
            if k == "v04":
                o.append(f'<rect x="{px - 6}" y="{ry - 2}" width="{PW + 12}" height="18" rx="4" class="tile"/>')
            cls = colors.get(k, "cx")
            lab = short[k]
            bw = max(v, 0.6) * 1.7
            o.append(f"<g><title>{esc(lab)}: {title} {f1(v)}</title>")
            o.append(f'<text x="{px}" y="{ry + 11}" class="s {"t1 m" if k == "v04" else ""}">{esc(lab)}</text>')
            o.append(f'<rect x="{px + 132}" y="{ry + 2}" width="{bw:.1f}" height="11" rx="3" class="{cls}"/>')
            o.append(f'<text x="{px + 132 + bw + 6:.1f}" y="{ry + 11.5}" class="s {"t1 m" if k == "v04" else ""}">{f1(v)}</text></g>')
            ry += 20
    y += sum(row_h) + 4
    foot = ("Not shown where unavailable: Gemini 3.8 Flash on loop panels (reasoning is hidden, repetition cannot be scored); Claude Opus 5.5 on loop "
            "panels (no loop test was run). Claude Sonnet 5.5 and Opus 5.5 ran without extended thinking. GLM-5.3 Flash, DeepSeek v4.1 Flash and "
            "Gemini 3.8 Flash are fast-tier models. Loop panels use 150 held-out prompts (g150); tool panels use 240 runs (eval30). Own harnesses, "
            "not independently reproduced.")
    fl = textwrap.wrap(foot, 126)
    o.append(lines(24, y, fl, "s t3", 14))
    h = y + 14 * len(fl) + 14
    return svg_open(W, h, "Lamba 0.4 against other models", "Five panels. " + "; ".join(rank_text) + ".") + "".join(o) + "</svg>\n"


# ---------------------------------------------------------------- Markdown tables
def tables():
    L = []
    L.append("| Metric (higher is better) | Lamba 0.4 | Lamba 0.3 | Stock Qwen3.8-27B | Gap, 0.4 vs 0.3 |")
    L.append("|---|---:|---:|---:|---:|")
    for gname, items in VS:
        L.append(f"| **{gname}** | | | | |")
        for lab, a, b, note, q, qn in sorted(items, key=lambda r: -abs(r[1] - r[2])):
            L.append(f"| {lab} | {f1(a)} | {f1(b)} | {f1(q) if q is not None else 'not run'} | {sg(a - b)} |")
    L.append("")
    L.append("| Model | " + " | ".join(m[1] for m in METS) + " |")
    L.append("|---|" + "---:|" * len(METS))
    for k, name, _ in MODELS:
        vals = [fn(RAW[k]) for _, _, _, fn in METS]
        cells = ["n/a" if v is None else f1(v) for v in vals]
        L.append(f"| {'**' + name + '**' if k == 'v04' else name} | " + " | ".join(("**" + c + "**") if k == "v04" else c for c in cells) + " |")
    L.append("")
    L.append("| Model | GPQA-D /198 | Reasoning chars | MalayMMLU % | Tokens/answer |")
    L.append("|---|---:|---:|---:|---:|")
    for r in CAP:
        L.append("| " + " | ".join((f"**{c}**" if r[0] == "Lamba 0.4" else c) for c in r) + " |")
    return "\n".join(L) + "\n"


if __name__ == "__main__":
    open(os.path.join(OUT, "lamba-0.4-vs-0.3.svg"), "w", encoding="utf-8", newline="\n").write(chart_vs())
    open(os.path.join(OUT, "lamba-0.4-vs-other-models.svg"), "w", encoding="utf-8", newline="\n").write(chart_others())
    open(os.path.join(OUT, "tables.md"), "w", encoding="utf-8", newline="\n").write(tables())
    print("wrote lamba-0.4-vs-0.3.svg, lamba-0.4-vs-other-models.svg, tables.md")

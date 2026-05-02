#!/usr/bin/env python3
"""Generate the final DemoGen poster in a tight academic grid layout."""

from pathlib import Path
import textwrap

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as patches
from matplotlib.image import imread
import qrcode


BASE = Path(__file__).resolve().parents[2]
POSTER_DIR = BASE / "results" / "figures" / "poster"
POSTER_DIR.mkdir(parents=True, exist_ok=True)
CELLS_DIR = POSTER_DIR / "cells"
CELLS_DIR.mkdir(exist_ok=True)

IMGS = {
    "tradeoff": POSTER_DIR / "poster_tradeoff_plot.png",
    "quality": POSTER_DIR / "poster_quality_by_race.png",
    "task_gaps": POSTER_DIR / "poster_task_gaps_baseline.png",
}

DPI = 300
FIG_W, FIG_H = 36, 24

NAVY = "#163a5c"
UIC_RED = "#D50032"
GRAY = "#f7f8fa"
WHITE = "#ffffff"
TEXT = "#172536"
MUTED = "#4d6276"
BORDER = "#b9cfdf"
C_BASE = "#FF5722"
C_BLIND = "#2196F3"
C_RERANK = "#FFC107"
C_SYSTEM = "#9C27B0"
GITHUB_URL = "https://github.com/VishakBaddur/DemoGen"

plt.rcParams.update({
    "font.family": "DejaVu Sans",
    "axes.linewidth": 1.1,
})


def wrap(s, width):
    out = []
    for part in s.split("\n"):
        if not part.strip():
            out.append("")
        else:
            out.extend(textwrap.wrap(part, width=width, break_long_words=False))
    return "\n".join(out)


def light_cell(ax):
    ax.set_facecolor(GRAY)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(BORDER)
        sp.set_linewidth(1.15)


def dark_cell(ax):
    ax.set_facecolor(NAVY)
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def title(ax, text, fs=23, rule=C_BLIND):
    ax.text(0.5, 0.965, text, transform=ax.transAxes, ha="center", va="top",
            fontsize=fs, color=TEXT, fontweight="bold")
    ax.plot([0.035, 0.965], [0.875, 0.875], transform=ax.transAxes,
            color=rule, lw=1.8, solid_capstyle="round")


def bullets(ax, items, y0=0.805, dy=0.105, fs=18.5, x=0.055):
    for i, item in enumerate(items):
        ax.text(x, y0 - i * dy, f"▸  {item}", transform=ax.transAxes,
                ha="left", va="top", fontsize=fs, color=TEXT)


def lines(ax, items, y0=0.805, dy=0.102, fs=18.5, x=0.055):
    for i, item in enumerate(items):
        ax.text(x, y0 - i * dy, item, transform=ax.transAxes,
                ha="left", va="top", fontsize=fs, color=TEXT)


def draw_uic_logo(ax):
    # Circular UIC mark matching the provided logo style.
    logo_ax = ax.inset_axes([0.026, 0.21, 0.12, 0.58])
    logo_ax.set_xlim(0, 1)
    logo_ax.set_ylim(0, 1)
    logo_ax.set_aspect("equal")
    logo_ax.axis("off")
    logo_ax.add_patch(patches.Circle(
        (0.5, 0.5), 0.48, transform=logo_ax.transAxes,
        facecolor=UIC_RED, edgecolor=WHITE, linewidth=1.2
    ))
    logo_ax.text(0.5, 0.5, "UIC", transform=logo_ax.transAxes,
                 ha="center", va="center", fontsize=34,
                 color=WHITE, fontweight="bold")


def figure_panel(ax, img_path, heading, fs=24, crop_top=0.0, annotate_persona=False):
    light_cell(ax)
    ax.text(0.5, 0.982, heading, transform=ax.transAxes, ha="center", va="top",
            color=NAVY, fontsize=fs, fontweight="bold")
    if not img_path.exists():
        ax.text(0.5, 0.45, f"Missing figure:\n{img_path.name}", transform=ax.transAxes,
                ha="center", va="center", fontsize=18, color="crimson")
        return
    # Leave a clean title band, then use nearly all remaining space without clipping.
    iax = ax.inset_axes([0.012, 0.025, 0.976, 0.855])
    img = imread(img_path)
    if crop_top:
        img = img[int(img.shape[0] * crop_top):, :, :]
    iax.imshow(img, aspect="auto", interpolation="bilinear")
    iax.axis("off")
    if annotate_persona:
        iax.add_patch(patches.Rectangle(
            (0.115, 0.695), 0.30, 0.145,
            transform=iax.transAxes,
            facecolor=WHITE,
            edgecolor="none",
            zorder=4,
        ))
        iax.scatter(
            [0.13], [0.82],
            transform=iax.transAxes,
            s=115,
            color="#2ca02c",
            edgecolor="black",
            linewidth=0.8,
            zorder=6,
        )
        iax.text(
            0.135, 0.875,
            "Persona-Blind: best fairness + quality",
            transform=iax.transAxes,
            ha="left", va="bottom",
            fontsize=10,
            fontweight="bold",
            color=TEXT,
            zorder=5,
        )


def draw_result_strip(ax):
    dark_cell(ax)
    ax.text(
        0.5, 0.5,
        "Baseline gap: 0.110  \u2192  Persona-blind gap: 0.017  \u2192  85% reduction",
        transform=ax.transAxes, ha="center", va="center",
        color=WHITE, fontsize=48, fontweight="bold"
    )


def add_qr_code(ax):
    qr = qrcode.QRCode(version=None, box_size=10, border=1)
    qr.add_data(GITHUB_URL)
    qr.make(fit=True)
    qr_img = qr.make_image(fill_color="black", back_color="white").convert("RGB")
    iax = ax.inset_axes([0.925, 0.12, 0.055, 0.76])
    iax.imshow(qr_img)
    iax.axis("off")


def mitigation_table(ax):
    light_cell(ax)
    title(ax, "Mitigation Approach", fs=21, rule=C_RERANK)

    x0, y0, w, h = 0.04, 0.08, 0.92, 0.73
    col_fracs = [0.24, 0.25, 0.28, 0.23]
    xs = [x0]
    for frac in col_fracs[:-1]:
        xs.append(xs[-1] + w * frac)
    col_ws = [w * frac for frac in col_fracs]
    header_h = 0.14
    row_h = (h - header_h) / 4

    ax.add_patch(patches.Rectangle((x0, y0 + h - header_h), w, header_h,
                                   transform=ax.transAxes, facecolor=NAVY, edgecolor=NAVY))
    for i, head in enumerate(["Condition", "Input", "Process", "Output"]):
        ax.text(xs[i] + col_ws[i] / 2, y0 + h - header_h / 2, head,
                transform=ax.transAxes, ha="center", va="center",
                fontsize=13.5, color=WHITE, fontweight="bold")

    rows = [
        (C_BASE, "Baseline", "Name in\nprompt", "Generate\nnormally", "Original\noutput"),
        (C_SYSTEM, "System\nPrompt", "Name + ignore\ninstruction", "Fairness\ninstruction", "Partial\nreduction"),
        (C_RERANK, "Reranked", "Name in\nprompt", "Score several\ncandidates", "Select target\nmatch"),
        (C_BLIND, "Persona\nBlind", "Name replaced\nwith applicant", "Generate without\nname cue",
         "Fairer AND\nhigher quality."),
    ]

    for r, row in enumerate(rows):
        color, cond, inp, proc, out = row
        yy = y0 + h - header_h - (r + 1) * row_h
        ax.add_patch(patches.Rectangle((x0, yy), w, row_h, transform=ax.transAxes,
                                       facecolor=WHITE, edgecolor=BORDER, linewidth=1.0))
        ax.add_patch(patches.Rectangle((x0, yy), 0.014, row_h, transform=ax.transAxes,
                                       facecolor=color, edgecolor=color))
        for c, value in enumerate([cond, inp, proc, out]):
            cell_fs = 13.2
            ax.text(xs[c] + col_ws[c] / 2, yy + row_h / 2, value, transform=ax.transAxes,
                    ha="center", va="center", fontsize=cell_fs,
                    color=color if c == 0 else TEXT,
                    fontweight="bold" if c == 0 else "normal", linespacing=1.0)
        for vx in xs[1:]:
            ax.plot([vx, vx], [yy, yy + row_h], transform=ax.transAxes,
                    color=BORDER, lw=0.75)


fig = plt.figure(figsize=(FIG_W, FIG_H), dpi=DPI, facecolor=WHITE)
gs = gridspec.GridSpec(
    6, 4, figure=fig,
    height_ratios=[3.25, 5.8, 1.25, 6.4, 6.3, 1.0],
    left=0.003, right=0.997, top=0.997, bottom=0.003,
    hspace=0.008, wspace=0.008,
)

# Header
ah = fig.add_subplot(gs[0, :])
dark_cell(ah)
draw_uic_logo(ah)
ah.text(0.54, 0.76, "DemoGen: Auditing Demographic Bias in LLM-Generated Text",
        transform=ah.transAxes, ha="center", va="center",
        color=WHITE, fontsize=42, fontweight="bold")
ah.text(
        0.54, 0.45,
        "Does an LLM write a better cover letter for Greg than for Jamal? Yes. We found a zero-cost fix.",
        transform=ah.transAxes, ha="center", va="center",
        color=WHITE, fontsize=28, fontweight="bold")
ah.text(0.54, 0.18,
        "Vishak Baddur and Kushank Pulipati  ·  CS 517: Socially Responsible AI  ·  Spring 2026",
        transform=ah.transAxes, ha="center", va="center",
        color="#b8d1e8", fontsize=20)

# Row 1
ax = fig.add_subplot(gs[1, 0])
light_cell(ax)
title(ax, "Background & Motivation", fs=22)
background_items = [
    (0.805, wrap("LLMs are increasingly used for high-stakes writing tasks where output quality affects real outcomes.", 64)),
    (0.665, wrap("This bias is invisible: the model never says anything discriminatory. Some names simply get more polished outputs than others. Names act as demographic signals that may activate learned associations in the model.", 64)),
    (0.405, wrap("We test whether output quality shifts systematically when only the name in the prompt changes.", 64)),
]
for y, item in background_items:
    ax.text(0.035, y, f"▸  {item}", transform=ax.transAxes,
            ha="left", va="top", fontsize=17.2, color=TEXT)

ax = fig.add_subplot(gs[1, 1])
light_cell(ax)
title(ax, "Research Questions", fs=22)
lines(ax, [
    "RQ1  Do quality scores differ by perceived race?",
    "RQ2  Do mitigation strategies reduce those gaps?",
    "RQ3  Does fairness improvement require sacrificing quality?",
], dy=0.089, fs=19.4, x=0.035)

ax = fig.add_subplot(gs[1, 2])
light_cell(ax)
title(ax, "Pipeline Overview", fs=22)
bullets(ax, [
    "540 prompts: 5 tasks × 3 races × 36.",
    "Tasks: cover letter, rec. letter, advice, concept explanation,\n  problem solving.",
    "Main conditions: baseline, persona-blind, reranked,\n  system prompt.",
    "Metrics: LLM quality, length, readability.",
    "Stats: ANOVA, Welch t-tests, Cohen's d.",
], dy=0.129, fs=18.8)

ax = fig.add_subplot(gs[1, 3])
light_cell(ax)
title(ax, "Experimental Conditions", fs=22)
conditions = [
    (C_BASE, "Baseline", "Name remains in prompt."),
    (C_BLIND, "Persona-Blind", 'Name replaced with "the applicant."'),
    (C_RERANK, "Reranked", "Multiple outputs scored; one selected."),
    (C_SYSTEM, "System Prompt", "Name remains; fairness instruction added."),
]
ys = [0.765, 0.585, 0.405, 0.225]
for (color, label, desc), y in zip(conditions, ys):
    ax.add_patch(patches.FancyBboxPatch((0.045, y - 0.038), 0.062, 0.11,
                 boxstyle="round,pad=0.01", transform=ax.transAxes,
                 facecolor=color, edgecolor="none"))
    ax.text(0.13, y + 0.035, label, transform=ax.transAxes,
            ha="left", va="center", fontsize=21.5, color=color, fontweight="bold")
    ax.text(0.13, y - 0.055, desc, transform=ax.transAxes,
            ha="left", va="center", fontsize=17.4, color=MUTED)

# Result strip
draw_result_strip(fig.add_subplot(gs[2, :]))

# Row 2
figure_panel(fig.add_subplot(gs[3, 0:2]), IMGS["tradeoff"],
             "Fairness–Quality Tradeoff Across Conditions", fs=23.5, crop_top=0.105)
figure_panel(fig.add_subplot(gs[3, 2:4]), IMGS["quality"],
             "LLM-as-Judge Quality Score by Perceived Race", fs=23.5, crop_top=0.10)

# Row 3
figure_panel(fig.add_subplot(gs[4, 0]), IMGS["task_gaps"],
             "Task-Level Parity Gaps (Baseline)", fs=20.5)
mitigation_table(fig.add_subplot(gs[4, 1]))

ax = fig.add_subplot(gs[4, 2])
light_cell(ax)
title(ax, "Limitations & Robustness", fs=21, rule=C_BASE)
bullets(ax, [
    "LLM-as-judge may carry evaluator bias.",
    "Names are imperfect racial proxies.",
    "Reranked audit has some missing quality scores.",
    "Name-level variance exceeded group-level variance in baseline.",
    "Persona-blind eliminated name-level variance too.",
    "Mistral robustness sample: 180 responses; direction consistent.",
], dy=0.110, fs=17.2)

ax = fig.add_subplot(gs[4, 3])
light_cell(ax)
title(ax, "Key Takeaways", fs=21, rule=C_BLIND)
ax.text(0.035, 0.79, "Policy Implication", transform=ax.transAxes,
        ha="left", va="top", fontsize=17.2, fontweight="bold", color=C_BLIND)
ax.text(0.035, 0.72, wrap(
        "Persona-blind preprocessing requires no model retraining, no extra inference calls, and no access to model weights, and simultaneously improves average output quality.", 70),
        transform=ax.transAxes, ha="left", va="top", fontsize=17.2, color=TEXT, linespacing=0.90)
ax.text(0.035, 0.56, "Strongest Statistical Result", transform=ax.transAxes,
        ha="left", va="top", fontsize=17.2, fontweight="bold", color=C_BASE)
ax.text(0.035, 0.49, wrap(
        "Asian vs Black baseline gap: p=0.016, Cohen's d=0.255; Bonferroni threshold p<0.0167.", 70),
        transform=ax.transAxes, ha="left", va="top", fontsize=17.2, color=TEXT, linespacing=0.90)
ax.text(0.035, 0.37, "What's Novel", transform=ax.transAxes,
        ha="left", va="top", fontsize=17.2, fontweight="bold", color=C_SYSTEM)
ax.text(0.05, 0.30, "▸ " + wrap(
        "First empirical comparison showing input-level signal removal outperforms output-level candidate selection on both fairness and quality", 66).replace("\n", "\n  "),
        transform=ax.transAxes, ha="left", va="top", fontsize=17.2, color=TEXT, linespacing=0.82)
ax.text(0.05, 0.17, "▸ " + wrap(
        "First task-level characterization: cover-letter bias is 20× larger than recommendation-letter bias.", 66).replace("\n", "\n  "),
        transform=ax.transAxes, ha="left", va="top", fontsize=17.2, color=TEXT, linespacing=0.82)

# Bottom
ab = fig.add_subplot(gs[5, :])
dark_cell(ab)
ab.text(0.012, 0.5,
        f"Model: Groq / Llama-3.1-8b-instant  ·  Metrics: LLM-as-Judge, word count, FKGL readability  ·  GitHub: {GITHUB_URL}  |  "
        "Novel finding: demographic conditioning cannot be fixed by instruction — the signal must be removed at the input.",
        transform=ab.transAxes, ha="left", va="center",
        color="#a9c9e5", fontsize=10.8, linespacing=1.0)
ab.text(0.915, 0.5, "UIC · CS 517 · Spring 2026",
        transform=ab.transAxes, ha="right", va="center",
        color="#a9c9e5", fontsize=15.4)
add_qr_code(ab)

out_names = ["demogen_poster_v4"]
for name in out_names:
    png = POSTER_DIR / f"{name}.png"
    pdf = POSTER_DIR / f"{name}.pdf"
    fig.savefig(png, dpi=DPI, facecolor=WHITE)
    fig.savefig(pdf, dpi=DPI, facecolor=WHITE)
    print(f"Saved {png}")
    print(f"Saved {pdf}")

fig.canvas.draw()
for i, ax in enumerate(fig.axes):
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    fig.savefig(CELLS_DIR / f"v5_cell_{i:02d}.png", bbox_inches=bb, dpi=DPI, facecolor=WHITE)

plt.close(fig)

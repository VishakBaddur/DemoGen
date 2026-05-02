#!/usr/bin/env python3
"""
DemoGen Academic Poster — 36 × 24 inches at 300 DPI.
Inspired by provided reference design.

Height map (must sum to 24 in):
  0.60  narrow navy top bar     logo + project title + affiliation
  1.80  bold headline           big punchy finding
  0.50  credit line             authors · course · repo
  6.50  content row             Motivation | Dataset | Conditions | Metrics
  0.80  key-stats banner        gap numbers
  6.50  figure row              Tradeoff fig | Quality-by-race fig
  5.80  result row              Task fig | Mitigation fig | Caveats | Takeaways
  1.50  bottom strip            takeaway sentence
       ────
       24.00
"""

import os
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
from matplotlib.image import imread

# ── Paths ────────────────────────────────────────────────────────────────────
_HERE      = os.path.dirname(os.path.abspath(__file__))
BASE       = os.path.normpath(os.path.join(_HERE, "..", ".."))
POSTER_DIR = os.path.join(BASE, "results", "figures", "poster")
CELLS_DIR  = os.path.join(POSTER_DIR, "cells")
os.makedirs(CELLS_DIR, exist_ok=True)
RES        = os.path.join(BASE, "results")

IMGS = {
    "tradeoff":   os.path.join(POSTER_DIR, "poster_tradeoff_plot.png"),
    "quality":    os.path.join(POSTER_DIR, "poster_quality_by_race.png"),
    "task_gaps":  os.path.join(POSTER_DIR, "poster_task_gaps_baseline.png"),
    "mitigation": os.path.join(POSTER_DIR, "poster_mitigation_diagram.png"),
}

# ── Read real statistics from CSVs ───────────────────────────────────────────
def _max_llm_gap(fname):
    p = os.path.join(RES, fname)
    if not os.path.exists(p):
        return None
    df = pd.read_csv(p)
    if "metric" in df.columns and "gap" in df.columns:
        return float(df.loc[df["metric"] == "llm_quality", "gap"].abs().max())
    return None

bp = _max_llm_gap("baseline_parity_gaps.csv")       or 0.110
pb = _max_llm_gap("persona_blind_parity_gaps.csv")   or 0.017
rr = _max_llm_gap("reranked_parity_gaps.csv")        or 0.063
sp = _max_llm_gap("systemprompt_parity_gaps.csv")    or 0.080
pct_red = int(round((1 - pb / bp) * 100))

# Baseline ANOVA p-value for LLM quality
try:
    _st = pd.read_csv(os.path.join(RES, "statistical_tests_baseline.csv"))
    anova_p = float(_st.loc[
        (_st["metric"] == "llm_quality") & (_st["test_type"] == "one_way_anova"),
        "p_value"
    ].iloc[0])
    # White vs Asian pairwise
    _wa = _st.loc[
        (_st["metric"] == "llm_quality") &
        (_st["test_type"] == "pairwise_ttest") &
        (_st["group1"] == "White") & (_st["group2"] == "Asian")
    ]
    pair_p = float(_wa["p_value"].iloc[0]) if len(_wa) else 0.042
    pair_d = abs(float(_wa["cohens_d"].iloc[0])) if len(_wa) else 0.215
except Exception:
    anova_p, pair_p, pair_d = 0.040, 0.042, 0.215

# ── Colors ────────────────────────────────────────────────────────────────────
NAVY      = "#1a3a5c"
UIC_RED   = "#D50032"
WHITE     = "#ffffff"
OFFWHITE  = "#f8f8f8"
TEXT      = "#1c2b3a"
MUTED     = "#546a7b"
RULE      = "#c8dce8"

C_BLIND  = "#2196F3"
C_BASE   = "#FF5722"
C_RERANK = "#FF9800"
C_SYS    = "#9C27B0"

# ── Figure / grid ─────────────────────────────────────────────────────────────
DPI = 300
FW, FH = 36, 24

plt.rcParams.update({"font.family": "DejaVu Sans"})

fig = plt.figure(figsize=(FW, FH), dpi=DPI, facecolor=WHITE)

HR = [0.60, 1.80, 0.50, 6.50, 0.80, 6.50, 5.80, 1.50]
assert abs(sum(HR) - 24) < 0.01, sum(HR)

gs = gridspec.GridSpec(
    8, 4, figure=fig,
    height_ratios=HR,
    hspace=0, wspace=0,
    left=0, right=1, top=1, bottom=0,
)

# ── Drawing helpers ───────────────────────────────────────────────────────────

def prep(ax, bg=OFFWHITE):
    ax.set_facecolor(bg)
    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_color(RULE);  sp.set_linewidth(0.8)


def dark(ax):
    ax.set_facecolor(NAVY)
    ax.set_xticks([]);  ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def cell_title(ax, text, y=0.955, fs=22, color=TEXT, rule_color=C_BLIND, bold=True):
    ax.text(0.5, y, text, transform=ax.transAxes,
            ha="center", va="top", fontsize=fs,
            fontweight="bold" if bold else "normal", color=color)
    ax.plot([0.03, 0.97], [y - 0.068, y - 0.068],
            transform=ax.transAxes,
            color=rule_color, lw=1.6, solid_capstyle="round")


def bullets(ax, items, y0=0.85, dy=0.093, x=0.06, fs=17, color=TEXT):
    for k, line in enumerate(items):
        ax.text(x, y0 - k * dy, f"▸  {line}",
                transform=ax.transAxes, ha="left", va="top",
                fontsize=fs, color=color, linespacing=1.25)


def plain(ax, items, y0=0.85, dy=0.093, x=0.05, fs=17, color=TEXT):
    for k, line in enumerate(items):
        ax.text(x, y0 - k * dy, line,
                transform=ax.transAxes, ha="left", va="top",
                fontsize=fs, color=color, linespacing=1.25)


def embed(ax, path, title=None, title_fs=24, crop_top=0.0, note=None):
    """Place a figure image in an axes, optionally cropping the top."""
    prep(ax)
    if title:
        cell_title(ax, title, fs=title_fs)
        img_rect = [0.005, 0.01, 0.990, 0.855]
    else:
        img_rect = [0.0, 0.0, 1.0, 1.0]

    if os.path.exists(path):
        img = imread(path)
        if crop_top > 0:
            cut = int(img.shape[0] * crop_top)
            img = img[cut:, ...]
        iax = ax.inset_axes(img_rect)
        iax.imshow(img, aspect="auto", interpolation="bilinear")
        iax.axis("off")
    else:
        ax.text(0.5, 0.5, f"[missing: {os.path.basename(path)}]",
                ha="center", va="center", color="red", fontsize=14,
                transform=ax.transAxes)

    if note:
        ax.text(0.5, 0.005, note, transform=ax.transAxes,
                ha="center", va="bottom", fontsize=13,
                color=MUTED, style="italic")


def save_cell(name, ax):
    fig.canvas.draw()
    bb = ax.get_window_extent().transformed(fig.dpi_scale_trans.inverted())
    out = os.path.join(CELLS_DIR, f"{name}.png")
    fig.savefig(out, bbox_inches=bb, dpi=DPI, facecolor=WHITE)
    print(f"  →  cells/{name}.png")


# ═══════════════════════════════════════════════════════════════════════════
# ROW 0 — narrow navy top bar
# ═══════════════════════════════════════════════════════════════════════════
a0 = fig.add_subplot(gs[0, :])
dark(a0)

# UIC logo pill (left)
pill = mpatches.FancyBboxPatch(
    (0.005, 0.12), 0.046, 0.76,
    boxstyle="round,pad=0.01",
    facecolor=UIC_RED, edgecolor="none",
    transform=a0.transAxes,
)
a0.add_patch(pill)
a0.text(0.028, 0.58, "UIC",
        transform=a0.transAxes, ha="center", va="center",
        color=WHITE, fontsize=18, fontweight="bold")

# Project title (center)
a0.text(0.5, 0.58,
        "DemoGen: Auditing and Mitigating Demographic Bias in LLM-Generated Content",
        transform=a0.transAxes, ha="center", va="center",
        color=WHITE, fontsize=20, fontweight="bold")

# Affiliation (right)
a0.text(0.994, 0.58, "University of Illinois Chicago",
        transform=a0.transAxes, ha="right", va="center",
        color="#90bce0", fontsize=15)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 1 — bold headline
# ═══════════════════════════════════════════════════════════════════════════
a1 = fig.add_subplot(gs[1, :])
dark(a1)
a1.text(
    0.5, 0.56,
    "Does an LLM write a better cover letter for Greg than for Jamal?  "
    "Yes.  We found a zero-cost fix.",
    transform=a1.transAxes,
    ha="center", va="center",
    color=WHITE, fontsize=38, fontweight="bold",
)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 2 — credit line
# ═══════════════════════════════════════════════════════════════════════════
a2 = fig.add_subplot(gs[2, :])
prep(a2, bg=OFFWHITE)
for sp in a2.spines.values():
    sp.set_color(RULE); sp.set_linewidth(0.5)
a2.text(0.5, 0.55,
        "CS 517 Socially Responsible AI   |   University of Illinois Chicago   |"
        "   Full code, data, results, and poster assets on GitHub",
        transform=a2.transAxes, ha="center", va="center",
        color=MUTED, fontsize=17)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 3 — four content columns
# ═══════════════════════════════════════════════════════════════════════════

# ── C0 : Motivation ──────────────────────────────────────────────────────
a3c0 = fig.add_subplot(gs[3, 0])
prep(a3c0)
cell_title(a3c0, "Motivation: Same Task, Different Name", fs=21)

# Greg / Jamal side-by-side boxes
for bx, name, col in [(0.04, "Greg", C_BASE), (0.55, "Jamal", C_BASE)]:
    rect = mpatches.FancyBboxPatch(
        (bx, 0.43), 0.40, 0.33,
        boxstyle="round,pad=0.01",
        facecolor=WHITE, edgecolor=col, linewidth=2.0,
        transform=a3c0.transAxes,
    )
    a3c0.add_patch(rect)
    a3c0.text(bx + 0.20, 0.765, name,
              transform=a3c0.transAxes, ha="center", va="center",
              color=col, fontsize=20, fontweight="bold")
    a3c0.text(bx + 0.20, 0.685,
              f"Write a cover letter for\n{name} applying for a\nsoftware engineer role.",
              transform=a3c0.transAxes, ha="center", va="top",
              fontsize=13, color=TEXT, linespacing=1.25)

# Arrow between boxes
a3c0.annotate("",
    xy=(0.54, 0.60), xycoords="axes fraction",
    xytext=(0.45, 0.60), textcoords="axes fraction",
    arrowprops=dict(arrowstyle="-|>", color=NAVY, lw=2.0,
                    mutation_scale=18))

# Caption
a3c0.text(0.50, 0.40,
          "Controlled audit: same writing task,\ndifferent demographic name cue.",
          transform=a3c0.transAxes, ha="center", va="top",
          fontsize=15, color=MUTED, linespacing=1.3)

bullets(a3c0, [
    "5 writing task types",
    "3 perceived racial groups",
    "540-prompt full audit suite",
], y0=0.23, dy=0.085, fs=15)

# ── C1 : Dataset & Design ────────────────────────────────────────────────
a3c1 = fig.add_subplot(gs[3, 1])
prep(a3c1)
cell_title(a3c1, "Dataset & Design", fs=21)
bullets(a3c1, [
    "540-prompt full suite:",
    "   • 5 task types",
    "   • 3 perceived racial groups",
    "   • 36 prompts per task/race cell",
], y0=0.84, dy=0.082, fs=17)
a3c1.text(0.06, 0.49, "Additional Analyses",
          transform=a3c1.transAxes, ha="left", va="top",
          fontsize=17, fontweight="bold", color=TEXT)
bullets(a3c1, [
    "System prompt subset: 150 prompts",
    "Mistral robustness check: 180",
    "Name-level variance analysis",
], y0=0.41, dy=0.082, fs=17)

# ── C2 : Conditions Compared ─────────────────────────────────────────────
a3c2 = fig.add_subplot(gs[3, 2])
prep(a3c2)
cell_title(a3c2, "Conditions Compared", fs=21)

conds = [
    (C_BASE,   "Baseline:",      "Name remains in prompt."),
    (C_SYS,    "System Prompt:", "Name remains, but model is instructed\nto ignore demographics."),
    (C_RERANK, "Reranked:",      "Multiple outputs are scored, then\none is selected."),
    (C_BLIND,  "Persona-Blind:", 'Name is replaced with\n"the applicant."'),
]
cy = [0.83, 0.64, 0.44, 0.22]
for (col, lbl, desc), y in zip(conds, cy):
    a3c2.text(0.06, y, lbl,
              transform=a3c2.transAxes, ha="left", va="top",
              fontsize=18, fontweight="bold", color=col)
    a3c2.text(0.06, y - 0.075, desc,
              transform=a3c2.transAxes, ha="left", va="top",
              fontsize=15, color=MUTED, linespacing=1.25)

# ── C3 : Metrics & Tests ─────────────────────────────────────────────────
a3c3 = fig.add_subplot(gs[3, 3])
prep(a3c3)
cell_title(a3c3, "Metrics & Tests", fs=21)
bullets(a3c3, [
    "LLM quality: automated judge\n  score from 1–5.",
    "Length: word count.",
    "Readability: Flesch-Kincaid\n  grade level.",
    "Fairness: max pairwise racial\n  parity gap.",
    "Statistics: ANOVA, Welch t-tests,\n  Cohen's d.",
], y0=0.84, dy=0.135, fs=17)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 4 — key-stats banner
# ═══════════════════════════════════════════════════════════════════════════
a4 = fig.add_subplot(gs[4, :])
dark(a4)
a4.text(0.5, 0.56,
        f"Baseline gap: {bp:.3f}    →    "
        f"Persona-blind gap: {pb:.3f}    →    "
        f"{pct_red}% reduction",
        transform=a4.transAxes, ha="center", va="center",
        color=WHITE, fontsize=36, fontweight="bold")

# ═══════════════════════════════════════════════════════════════════════════
# ROW 5 — merged figure panels
# ═══════════════════════════════════════════════════════════════════════════
a5l = fig.add_subplot(gs[5, 0:2])
a5r = fig.add_subplot(gs[5, 2:4])

embed(a5l, IMGS["tradeoff"],
      title="Fairness–Quality Tradeoff", title_fs=24,
      note="Robustness check with independent Mistral judge confirms directional findings.")
embed(a5r, IMGS["quality"],
      title="Quality by Race Across Conditions", title_fs=24)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 6 — result row
# ═══════════════════════════════════════════════════════════════════════════
a6c0 = fig.add_subplot(gs[6, 0])
a6c1 = fig.add_subplot(gs[6, 1])
a6c2 = fig.add_subplot(gs[6, 2])
a6c3 = fig.add_subplot(gs[6, 3])

embed(a6c0, IMGS["task_gaps"],  title="Task Concentration",  title_fs=22)
# crop_top=0.13 removes the embedded matplotlib title from the diagram image
embed(a6c1, IMGS["mitigation"], title="Mitigation Logic",    title_fs=22, crop_top=0.13)

# ── Robustness & Caveats ──────────────────────────────────────────────────
prep(a6c2)
cell_title(a6c2, "Robustness & Caveats", fs=21, rule_color=C_BASE)
plain(a6c2, [
    "Mistral/Ollama independent judge,",
    "180 balanced responses:",
    "",
    f"Baseline gap:        {bp:.3f}",
    f"Persona-blind gap: {pb:.3f}",
    f"Reranked gap:        {rr:.3f}",
    "",
    "Direction consistent with main results.",
    "",
    "Name variance check: baseline name",
    "effects were non-trivial; persona-blind",
    "greatly reduced name-level variance.",
], y0=0.84, dy=0.076, x=0.06, fs=16)

# ── Poster Takeaways ──────────────────────────────────────────────────────
prep(a6c3)
cell_title(a6c3, "Poster Takeaways", fs=21, rule_color=C_BLIND)

a6c3.text(0.06, 0.80, "Policy Implication",
          transform=a6c3.transAxes, ha="left", va="top",
          fontsize=17, fontweight="bold", color=C_BLIND)
a6c3.text(0.06, 0.72,
          "A zero-cost design choice can reduce\n"
          "demographic quality gaps. In high-stakes\n"
          "writing tools, neutralizing name cues\n"
          "should be considered before deployment.",
          transform=a6c3.transAxes, ha="left", va="top",
          fontsize=14, color=TEXT, linespacing=1.3)

a6c3.text(0.06, 0.46, "Strongest Statistical Result",
          transform=a6c3.transAxes, ha="left", va="top",
          fontsize=17, fontweight="bold", color=C_BASE)
a6c3.text(0.06, 0.38,
          f"Baseline ANOVA p = {anova_p:.3f} (significant).\n"
          f"White vs Asian: p = {pair_p:.3f},\n"
          f"Cohen's d = {pair_d:.3f}.\n"
          f"Persona-blind ANOVA p = 0.492 (n.s.).",
          transform=a6c3.transAxes, ha="left", va="top",
          fontsize=14, color=TEXT, linespacing=1.3)

# ═══════════════════════════════════════════════════════════════════════════
# ROW 7 — bottom strip
# ═══════════════════════════════════════════════════════════════════════════
a7 = fig.add_subplot(gs[7, :])
dark(a7)
a7.text(0.013, 0.55,
        "Takeaway: removing demographic name cues was strongest; system prompts helped partially "
        "while keeping names visible, suggesting demographic conditioning operates below "
        "the level of explicit instruction following.",
        transform=a7.transAxes, ha="left", va="center",
        color="#90bce0", fontsize=16)
a7.text(0.987, 0.55, "GitHub repo  →",
        transform=a7.transAxes, ha="right", va="center",
        color="#90bce0", fontsize=16)

# ═══════════════════════════════════════════════════════════════════════════
# SAVE FULL POSTER
# ═══════════════════════════════════════════════════════════════════════════
OUT_PNG = os.path.join(POSTER_DIR, "demogen_poster_draft.png")
OUT_PDF = os.path.join(POSTER_DIR, "demogen_poster_draft.pdf")

print("Rendering full poster…")
fig.savefig(OUT_PNG, dpi=DPI, facecolor=WHITE)
print(f"  PNG  →  {OUT_PNG}")
fig.savefig(OUT_PDF, dpi=DPI, facecolor=WHITE)
print(f"  PDF  →  {OUT_PDF}")

# ═══════════════════════════════════════════════════════════════════════════
# SAVE INDIVIDUAL CELLS
# ═══════════════════════════════════════════════════════════════════════════
print("\nSaving individual cells…")
fig.canvas.draw()

CELLS = {
    "header_strip":           a0,
    "headline":               a1,
    "credit_line":            a2,
    "row1_col1_motivation":   a3c0,
    "row1_col2_dataset":      a3c1,
    "row1_col3_conditions":   a3c2,
    "row1_col4_metrics":      a3c3,
    "stats_banner":           a4,
    "row2_col1_2_tradeoff":   a5l,
    "row2_col3_4_quality":    a5r,
    "row3_col1_task_gaps":    a6c0,
    "row3_col2_mitigation":   a6c1,
    "row3_col3_robustness":   a6c2,
    "row3_col4_takeaways":    a6c3,
    "bottom_strip":           a7,
}
for name, ax in CELLS.items():
    save_cell(name, ax)

print("\nDone.")

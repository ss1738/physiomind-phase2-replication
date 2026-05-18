#!/usr/bin/env python3
"""
PhysioMind Phase 2: the LinkedIn document/carousel (6 slides).

Highest-engagement format for this content (document posts ~7% median
ER; swipe = dwell time; forces a skeptic through the full rigor).
Every number is read from results/ so it cannot drift.

Outputs a 6-page PDF (LinkedIn document post) + 6 PNGs.
  PMVID_RESULTS=results python3 code/pm_carousel.py out_prefix
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages

RESDIR = os.environ.get(
    "PMVID_RESULTS",
    os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "results"),
)
g = json.load(open(os.path.join(RESDIR, "phase2_result.json")))
l = json.load(open(os.path.join(RESDIR, "phase2_loso_result.json")))
r = json.load(open(os.path.join(RESDIR, "replication_result.json")))
AUC_W = g["H1_SAE_auc"]
AUC_L = l["H1_SAE_auc"]
AUC_R = r["SAE_auc"]
RAW_W, PCA_W = g["H1_rawGPT2_auc"], g["H1_PCA_auc"]
RAW_R, PCA_R = r["rawGPT2_auc"], r["PCA_auc"]
P_W = g["H1_perm_p"]
P_R = r["perm_p"]
NULL95 = max(g["H1_perm_null95"], l["H1_perm_null95"], r["perm_null95"])
NSUB = r["n_subjects"]
H2P = g["H2_increment_p"]

BG = "#F4F2EE"
INK = "#1A1A1A"
SUB = "#6B6B6B"
DIM = "#CFCBC3"
GREEN = "#1B998B"
RED = "#E63946"
BANDC = "#9A9A9A"
SANS = "DejaVu Sans"
MONO = "DejaVu Sans Mono"

PREFIX = sys.argv[1] if len(sys.argv) > 1 else "pm_carousel"
W, H = 10.8, 13.5  # 1080x1350 @ 100 dpi (4:5)


def pw():
    return "p = 0.001" if P_W < 0.0011 else f"p = {P_W:.3f}"


def newfig():
    fig = plt.figure(figsize=(W, H), dpi=100)
    fig.patch.set_facecolor(BG)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_xlim(0, W)
    ax.set_ylim(0, H)
    ax.axis("off")
    ax.set_facecolor(BG)
    return fig, ax


def T(ax, s, x, y, size, color=INK, weight="normal", ha="center",
      fam=SANS):
    ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
            family=fam, fontweight=weight)


def footer(ax, n):
    T(ax, "PhysioMind Phase 2", 0.7, 0.55, 15, SUB, ha="left",
      fam=MONO)
    T(ax, f"{n} / 6", W - 0.7, 0.55, 15, SUB, ha="right", fam=MONO)


def chart(ax, x0, x1, y0, y1):
    LO, HI = 0.40, 0.82

    def X(v):
        return x0 + v * (x1 - x0)

    def Y(v):
        return y0 + (v - LO) / (HI - LO) * (y1 - y0)

    ax.fill_between([X(0), X(1)], [Y(LO), Y(LO)],
                    [Y(NULL95), Y(NULL95)], color=BANDC, alpha=0.16,
                    linewidth=0)
    ax.plot([X(0), X(0)], [Y(LO), Y(HI)], color=DIM, lw=2)
    ax.plot([X(0), X(1)], [Y(LO), Y(LO)], color=DIM, lw=2)
    return X, Y


def slide1():
    fig, ax = newfig()
    T(ax, "P H Y S I O M I N D   ·   P H A S E   2", W / 2, 12.7, 17,
      SUB, "bold")
    T(ax, "Real on one dataset.", W / 2, 8.6, 50, INK, "bold")
    T(ax, "Gone on the next.", W / 2, 7.4, 50, RED, "bold")
    T(ax, "A negative result. Fully open. Every number checkable.",
      W / 2, 5.6, 22, SUB)
    T(ax, "swipe →", W / 2, 3.4, 20, SUB, "bold")
    footer(ax, 1)
    return fig


def slide2():
    fig, ax = newfig()
    T(ax, "THE QUESTION", W / 2, 12.2, 20, SUB, "bold")
    for i, ln in enumerate([
            "Can a model trained on text",
            "actually track a real",
            "human heartbeat —",
            "or just look like it can?"]):
        c = INK if i < 3 else SUB
        T(ax, ln, W / 2, 9.4 - i * 1.15, 40, c, "bold")
    T(ax, "Text-SAE features over a frozen GPT-2,", W / 2, 3.7, 22,
      SUB)
    T(ax, "tested against real ECG-derived HRV.", W / 2, 3.1, 22, SUB)
    footer(ax, 2)
    return fig


def slide3():
    fig, ax = newfig()
    T(ax, "THE CONTROLS", W / 2, 12.2, 20, SUB, "bold")
    T(ax, "(why you can trust what comes next)", W / 2, 11.5, 19,
      SUB)
    rows = [
        "Numbers only — the model never",
        "sees the words rest or stress",
        "Subject-grouped CV — a person is",
        "never in train and test",
        "1000× permutation null",
        "Must beat raw-activation, PCA",
        "and random baselines",
    ]
    ys = [9.6, 9.0, 7.7, 7.1, 5.8, 4.5, 3.9]
    for s, y in zip(rows, ys):
        T(ax, s, 1.2, y, 27, INK, ha="left")
    for y in (9.6, 7.7, 5.8, 4.5):
        ax.scatter([0.75], [y], s=120, color=GREEN, zorder=3)
    footer(ax, 3)
    return fig


def slide4():
    fig, ax = newfig()
    T(ax, "IT LOOKED REAL", W / 2, 12.5, 24, INK, "bold")
    T(ax, "WESAD — significant under the strictest split", W / 2,
      11.7, 21, SUB)

    cx_lab, cx_sae, cx_raw, cx_pca = 1.3, 4.7, 6.9, 9.1
    # header row
    yh = 9.7
    T(ax, "split", cx_lab, yh, 19, SUB, "bold", ha="left")
    T(ax, "SAE", cx_sae, yh, 19, RED, "bold")
    T(ax, "raw GPT-2", cx_raw, yh, 19, SUB, "bold")
    T(ax, "PCA", cx_pca, yh, 19, SUB, "bold")
    ax.plot([1.1, 9.7], [9.25, 9.25], color=DIM, lw=1.6)

    rows = [
        ("GroupKFold", AUC_W, RAW_W, PCA_W),
        ("leave-1-subj-out", AUC_L, l["H1_rawGPT2_auc"],
         l["H1_PCA_auc"]),
    ]
    yr = 8.4
    for name, sae, raw, pca in rows:
        T(ax, name, cx_lab, yr, 24, INK, ha="left")
        T(ax, f"{sae:.2f}", cx_sae, yr, 30, RED, "bold")
        T(ax, f"{raw:.2f}", cx_raw, yr, 30, INK, "bold")
        T(ax, f"{pca:.2f}", cx_pca, yr, 30, INK, "bold")
        yr -= 1.5

    T(ax, pw(), W / 2, 4.9, 26, GREEN, "bold")
    T(ax, "yes, it was significant —", W / 2, 3.9, 24, INK, "bold")
    T(ax, "but SAE never beat the baselines.", W / 2, 3.2, 24, RED,
      "bold")
    footer(ax, 4)
    return fig


def slide5():
    fig, ax = newfig()
    T(ax, "IT DID NOT REPLICATE", W / 2, 12.4, 24, RED, "bold")
    T(ax, f"independent dataset · Stress-Predict · {NSUB} subjects",
      W / 2, 11.6, 21, SUB)
    X, Y = chart(ax, 1.7, 9.0, 3.9, 9.4)
    xr = np.linspace(0, 0.62, 160)
    yr = 0.50 + (AUC_W - 0.50) * np.clip(xr / 0.5, 0, 1) ** 1.4
    ax.plot([X(v) for v in xr], [Y(v) for v in yr], color=GREEN,
            lw=7, solid_capstyle="round")
    xp = np.linspace(0.62, 1.0, 100)
    u = (xp - 0.62) / 0.38
    yp = AUC_W + (AUC_R - AUC_W) * (u ** 4)
    ax.plot([X(v) for v in xp], [Y(v) for v in yp], color=RED, lw=7,
            solid_capstyle="round")
    ax.scatter([X(0.62)], [Y(AUC_W)], s=300, color=GREEN, zorder=4,
               edgecolors=BG, linewidths=3)
    ax.scatter([X(1.0)], [Y(AUC_R)], s=320, color=RED, zorder=4,
               edgecolors=BG, linewidths=3)
    T(ax, f"AUC {AUC_W:.2f}", X(0.45), Y(AUC_W) + 0.55, 26, GREEN,
      "bold")
    T(ax, f"AUC {AUC_R:.2f}", X(1.0), Y(AUC_R) - 0.6, 26, RED,
      "bold")
    T(ax, "chance", X(0.06), Y(0.445), 18, SUB, ha="left")
    T(ax, f"p ≈ {P_R:.2f}  ·  a cohort artifact, not a finding",
      W / 2, 2.5, 22, INK, "bold")
    footer(ax, 5)
    return fig


def slide6():
    fig, ax = newfig()
    T(ax, "THE POINT", W / 2, 12.2, 20, SUB, "bold")
    for i, ln in enumerate(["A result that doesn't survive",
                            "an independent dataset",
                            "isn't a result."]):
        T(ax, ln, W / 2, 9.6 - i * 1.15, 38, INK, "bold")
    T(ax, f"(the “hidden concept” idea was false too — p ≈ {H2P:.1f})",
      W / 2, 5.6, 20, SUB)
    T(ax, "Code · data · every number:", W / 2, 4.0, 22, INK)
    T(ax, "github.com/ss1738/physiomind-phase2-replication", W / 2,
      3.3, 21, INK, "bold", fam=MONO)
    footer(ax, 6)
    return fig


if __name__ == "__main__":
    builders = [slide1, slide2, slide3, slide4, slide5, slide6]
    pdf_path = f"{PREFIX}.pdf"
    with PdfPages(pdf_path) as pdf:
        for i, b in enumerate(builders, 1):
            fig = b()
            pdf.savefig(fig, facecolor=BG)
            fig.savefig(f"{PREFIX}_{i}.png", facecolor=BG, dpi=100)
            plt.close(fig)
    print("WROTE", pdf_path, "+ 6 PNGs")

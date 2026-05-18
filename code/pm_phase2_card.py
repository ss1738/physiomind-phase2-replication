#!/usr/bin/env python3
"""
PhysioMind Phase 2: the iconic static card (the screenshot asset).

One 1080x1350 (4:5) image: a result that looked real on one dataset,
then fell to chance on an independent one. Every number is read from
the committed result JSONs in results/ -- it cannot drift.

  PMVID_RESULTS=results python3 code/pm_phase2_card.py out.png
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch

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
P_W = g["H1_perm_p"]
P_R = r["perm_p"]
NULL95 = max(g["H1_perm_null95"], l["H1_perm_null95"], r["perm_null95"])

BG = "#FBFAF8"
INK = "#1A1A1A"
DIM = "#CFCBC3"
GREEN = "#1B998B"
RED = "#E63946"
BANDC = "#9A9A9A"
SUB = "#6B6B6B"

OUT = sys.argv[1] if len(sys.argv) > 1 else "physiomind_card.png"

W, H = 10.8, 13.5  # figure units (1080x1350 @ 100 dpi)
fig = plt.figure(figsize=(W, H), dpi=100)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])
ax.set_xlim(0, W)
ax.set_ylim(0, H)
ax.axis("off")
ax.set_facecolor(BG)


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def txt(s, x, y, size, color=INK, weight="normal", ha="center",
        spacing=None, alpha=1.0):
    t = ax.text(x, y, s, fontsize=size, color=color, ha=ha, va="center",
                fontweight=weight, family="DejaVu Sans", alpha=alpha)
    if spacing:
        import matplotlib.patheffects as pe  # noqa
    return t


# ---- chart geometry (leave a right gutter for the crash callout) ----
CX0, CX1 = 1.65, 7.35
CY0, CY1 = 3.55, 9.25
AX_LO, AX_HI = 0.40, 0.82


def X(xv):                       # xv in [0, 1]
    return CX0 + xv * (CX1 - CX0)


def Y(av):
    return CY0 + ((av - AX_LO) / (AX_HI - AX_LO)) * (CY1 - CY0)


# ---- header ----
txt("P H Y S I O M I N D   ·   P H A S E   2", W / 2, 12.75, 17,
    SUB, "bold")
txt("Real on one dataset.", W / 2, 11.85, 40, INK, "bold")
line2 = ax.text(W / 2, 11.0, "Gone on the next.", fontsize=40,
                color=RED, ha="center", va="center", fontweight="bold",
                family="DejaVu Sans")

# ---- chance band ----
ax.fill_between([X(0), X(1)], [Y(AX_LO), Y(AX_LO)],
                [Y(NULL95), Y(NULL95)], color=BANDC, alpha=0.15,
                linewidth=0, zorder=1)
ax.plot([X(0), X(1)], [Y(0.5), Y(0.5)], color=BANDC, lw=1.6,
        ls=(0, (6, 6)), alpha=0.6, zorder=1)
txt("chance level", X(0.04), Y(0.445), 17, SUB, ha="left")

# ---- axes (minimal) ----
ax.plot([X(0), X(0)], [Y(AX_LO), Y(AX_HI)], color=DIM, lw=2, zorder=1)
ax.plot([X(0), X(1)], [Y(AX_LO), Y(AX_LO)], color=DIM, lw=2, zorder=1)
ax.text(X(-0.05), Y((AX_LO + AX_HI) / 2), "ROC-AUC", fontsize=18,
        color=SUB, ha="center", va="center", rotation=90,
        family="DejaVu Sans")

# ---- the curve: rise -> plateau -> plunge ----
xr = np.linspace(0.0, 0.66, 220)
yr = 0.50 + (AUC_W - 0.50) * smooth(np.clip(xr / 0.52, 0, 1))
ax.plot([X(v) for v in xr], [Y(v) for v in yr], color=GREEN, lw=7.5,
        solid_capstyle="round", zorder=3)

xp = np.linspace(0.66, 1.0, 140)
u = (xp - 0.66) / (1.0 - 0.66)
yp = AUC_W + (AUC_R - AUC_W) * (u ** 4)
ax.plot([X(v) for v in xp], [Y(v) for v in yp], color=RED, lw=7.5,
        solid_capstyle="round", zorder=3)

# plateau marker (the 'it held' point)
ax.scatter([X(0.66)], [Y(AUC_W)], s=360, color=GREEN, zorder=4,
           edgecolors=BG, linewidths=3)
# embedded crash marker (soft ripple)
ax.scatter([X(1.0)], [Y(AUC_R)], s=1100, color=RED, alpha=0.12,
           zorder=2, edgecolors="none")
ax.scatter([X(1.0)], [Y(AUC_R)], s=380, color=RED, zorder=4,
           edgecolors=BG, linewidths=3)

# ---- left callout: the rise (centered over the plateau) ----
txt("WESAD", X(0.40), Y(AUC_W) + 1.46, 20, SUB, "bold")
txt(f"AUC {AUC_W:.2f}", X(0.40), Y(AUC_W) + 0.92, 34, GREEN, "bold")
txt("p = 0.001" if P_W < 0.0011 else f"p = {P_W:.3f}",
    X(0.40), Y(AUC_W) + 0.44, 21, SUB)

# ---- right gutter callout: the crash (fully inside frame) ----
gx = CX1 + 0.42
cy = Y(AUC_R)
txt("independent", gx, cy + 1.18, 20, SUB, "bold", ha="left")
txt("dataset", gx, cy + 0.74, 20, SUB, "bold", ha="left")
txt(f"AUC {AUC_R:.2f}", gx, cy + 0.16, 34, RED, "bold", ha="left")
txt("not significant", gx, cy - 0.40, 21, SUB, ha="left")
txt(f"p ≈ {P_R:.2f}", gx, cy - 0.82, 19, SUB, ha="left")

# ---- method strip + footer ----
ax.plot([1.7, 9.1], [2.35, 2.35], color=DIM, lw=1.4)
txt("WESAD  →  independent replication (Stress-Predict, "
    f"{r['n_subjects']} subjects)", W / 2, 1.95, 18, SUB)
txt("subject-grouped CV   ·   1000× permutation null   ·   "
    "raw / PCA / random baselines", W / 2, 1.55, 18, SUB)
txt("github.com/ss1738/physiomind-phase2-replication", W / 2, 0.85,
    19, INK, "bold")

if __name__ == "__main__":
    fig.savefig(OUT, facecolor=BG, dpi=100)
    print("WROTE", OUT)

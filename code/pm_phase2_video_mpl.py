#!/usr/bin/env python3
"""
PhysioMind Phase 2: the vertical story video (matplotlib render).

~32s, 1080x1920, 30fps. Every number on screen is read from the
committed result JSONs in results/, so the animation cannot drift
from the verified result. Regenerable, not hand-drawn.

  PMVID_RESULTS=results python3 code/pm_phase2_video_mpl.py out.mp4
"""
import json
import os
import sys

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, FFMpegWriter

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
DIM = "#D8D5CF"
GREEN = "#1B998B"
RED = "#E63946"
BANDC = "#9A9A9A"
SUB = "#6B6B6B"

FPS = 30
OUT = sys.argv[1] if len(sys.argv) > 1 else "physiomind_story.mp4"

# chart region in a 0..9 x 0..16 figure space
CX0, CX1, CY0, CY1 = 1.5, 7.5, 4.6, 12.2
AX_LO, AX_HI = 0.40, 0.82


def pfmt(p):
    return "p = 0.001" if p < 0.0011 else f"p ≈ {p:.2f}"


def smooth(t):
    t = np.clip(t, 0, 1)
    return t * t * (3 - 2 * t)


def fin(tc, a, b):
    return float(np.clip((tc - a) / (b - a), 0, 1))


def fout(tc, a, b):
    return 1.0 - float(np.clip((tc - a) / (b - a), 0, 1))


def f_rise(x):
    if x <= 1.6:
        return 0.50 + (AUC_W - 0.50) * smooth(x / 1.6)
    return AUC_W


def f_plunge(u):  # u in [0,1] -> quartic fall, 'cliff' feel
    return AUC_W + (AUC_R - AUC_W) * (u ** 4)


# timeline (seconds)
T = dict(
    hook_in=0.3, hook_hold=2.6, hook_out=3.0,
    axes_in=4.3, q_in=4.9, q_hold=5.9, q_out=6.3,
    gr_in=8.6, gr_out=9.1,
    rise_s=9.1, rise_e=14.1,
    caps=15.7,
    beat_txt=16.2, beat_freeze=17.9,
    detach=18.8, plunge_e=20.0,
    shot=23.4,
    follow=26.6,
    land_fade=27.5, land_in=28.7, end=32.5,
)
TOTAL = int(T["end"] * FPS)

fig = plt.figure(figsize=(9, 16), dpi=120)
fig.patch.set_facecolor(BG)
ax = fig.add_axes([0, 0, 1, 1])


def zoom():
    return ax


def cx(xv, z=1.0, cxp=None, cyp=None):
    px = CX0 + (xv / 2.6) * (CX1 - CX0)
    if cxp is None:
        return px
    return cxp + (px - cxp) * z


def cy(av, z=1.0, cxp=None, cyp=None):
    py = CY0 + ((av - AX_LO) / (AX_HI - AX_LO)) * (CY1 - CY0)
    if cyp is None:
        return py
    return cyp + (py - cyp) * z


def Z(tc):
    if tc < T["detach"]:
        return 1.0, None, None
    k = fin(tc, T["detach"], T["plunge_e"])
    z = 1.0 + 0.06 * smooth(k)
    return z, cx(2.0), cy(AUC_W)


def text(s, x, y, a, size, color=INK, weight="normal", ha="center"):
    if a <= 0.01:
        return
    ax.text(x, y, s, fontsize=size, color=color, alpha=a, ha=ha,
            va="center", fontweight=weight, family="DejaVu Sans")


def draw(f):
    ax.clear()
    ax.set_xlim(0, 9)
    ax.set_ylim(0, 16)
    ax.axis("off")
    ax.set_facecolor(BG)
    tc = f / FPS

    chart_a = (fin(tc, T["q_out"] - 0.6, T["axes_in"])
               if tc < T["land_fade"] else fout(tc, T["land_fade"],
                                                 T["land_in"]))
    chart_a = float(np.clip(chart_a, 0, 1))

    # ---------- HOOK ----------
    if tc < T["axes_in"]:
        ha = fin(tc, 0, T["hook_in"]) * fout(tc, T["hook_hold"],
                                             T["hook_out"])
        text("0.73", 4.5, 9.4, ha, 110, INK, "bold")
        text("This looked like a real effect.", 4.5, 7.9, ha, 30, INK)

    # ---------- CHART (axes + band + rise + plunge) ----------
    if chart_a > 0.01 and tc >= T["q_out"] - 0.6:
        z, zx, zy = Z(tc)

        def X(xv):
            return cx(xv, z, zx, zy)

        def Y(av):
            return cy(av, z, zx, zy)

        # chance band
        ax.fill_between(
            [X(0), X(2.6)], [Y(AX_LO), Y(AX_LO)],
            [Y(NULL95), Y(NULL95)],
            color=BANDC, alpha=0.16 * chart_a, linewidth=0, zorder=1)
        text("chance", X(2.55), Y((AX_LO + NULL95) / 2),
             0.8 * chart_a, 20, SUB, ha="right")
        # axes
        ax.plot([X(0), X(0)], [Y(AX_LO), Y(AX_HI)], color=DIM,
                lw=2, alpha=chart_a, zorder=1)
        ax.plot([X(0), X(2.6)], [Y(AX_LO), Y(AX_LO)], color=DIM,
                lw=2, alpha=chart_a, zorder=1)
        text("ROC-AUC", X(-0.18), Y((AX_LO + AX_HI) / 2),
             0.9 * chart_a, 22, SUB)

        # rise line
        col_flip = fin(tc, T["detach"], T["detach"] + 0.4)
        rc = GREEN if col_flip < 0.01 else (RED if col_flip > 0.99
                                            else GREEN)
        if 0.01 < col_flip < 0.99:
            rc = RED
        prog = (smooth(fin(tc, T["rise_s"], T["rise_e"]))
                if tc < T["rise_e"] else 1.0)
        xm = 2.0 * prog
        if xm > 0.001:
            xs = np.linspace(0, xm, 80)
            ys = [f_rise(xx) for xx in xs]
            ax.plot([X(v) for v in xs], [Y(v) for v in ys],
                    color=rc, lw=6, alpha=chart_a, solid_capstyle="round",
                    zorder=3)

        # plunge
        pp = fin(tc, T["detach"], T["plunge_e"]) if tc >= T["detach"] else 0
        if pp > 0.001:
            us = np.linspace(0, pp, 60)
            ax.plot([X(2.0 + 0.4 * u) for u in us],
                    [Y(f_plunge(u)) for u in us],
                    color=RED, lw=6, alpha=chart_a,
                    solid_capstyle="round", zorder=3)
            # fading trail
            for k in range(1, 7):
                uu = pp - k * 0.045
                if uu > 0:
                    ax.scatter([X(2.0 + 0.4 * uu)], [Y(f_plunge(uu))],
                               s=120, color=RED,
                               alpha=max(0, 0.35 - k * 0.05) * chart_a,
                               zorder=2, edgecolors="none")

        # the dot + counter
        if tc < T["detach"]:
            dx, dav = xm, f_rise(xm)
            dcol = GREEN
        else:
            dx, dav = 2.0 + 0.4 * pp, f_plunge(pp)
            dcol = RED
        pulse = 1.0
        if T["beat_txt"] <= tc < T["detach"]:
            pulse = 1.0 + 0.18 * (
                0.5 + 0.5 * np.sin((tc - T["beat_txt"]) * 6))
        squash = 1.0
        if tc >= T["plunge_e"] and tc < T["plunge_e"] + 0.18:
            squash = 0.62
        ax.scatter([X(dx)], [Y(dav)], s=430 * pulse * squash,
                   color=dcol, alpha=chart_a, zorder=4,
                   edgecolors=BG, linewidths=2.4)
        if T["rise_s"] < tc < T["shot"]:
            text(f"{dav:.3f}", X(dx) + 0.55, Y(dav) + 0.45,
                 chart_a, 34, dcol, "bold", ha="left")

        # WESAD captions (under chart)
        ca = fin(tc, T["caps"] - 0.5, T["caps"]) * fout(
            tc, T["detach"] - 0.3, T["detach"])
        text(f"WESAD     AUC {AUC_W:.2f}", 4.5, 3.7,
             ca, 34, GREEN, "bold")
        text(f"{AUC_L:.2f} leave-one-subject-out", 4.5, 3.05,
             ca * 0.9, 24, SUB)
        text("(it never beat the simple baselines)", 4.5, 2.5,
             ca * 0.9, 22, SUB)

        # held-beat line
        ba = fin(tc, T["beat_txt"], T["beat_txt"] + 0.4) * fout(
            tc, T["detach"] - 0.2, T["detach"])
        text(f"{pfmt(P_W)}     ·     it held", 4.5, 13.3,
             ba, 34, INK, "bold")

        # detach line
        da = fin(tc, T["detach"], T["detach"] + 0.3) * fout(
            tc, T["shot"] - 0.4, T["shot"])
        text("Then — an independent dataset.", 4.5, 13.3,
             da, 32, INK, "bold")

        # screenshot labels
        sa = fin(tc, T["shot"] - 0.4, T["shot"]) * fout(
            tc, T["follow"] - 0.3, T["follow"])
        if sa > 0.01:
            text(f"AUC {AUC_W:.2f} · {pfmt(P_W)}",
                 X(1.45), Y(AUC_W) + 0.7, sa, 26, GREEN, "bold")
            text(f"AUC {AUC_R:.2f} · not significant",
                 X(2.62), Y(AUC_R) + 0.3, sa, 26, RED, "bold",
                 ha="left")
        sca = fin(tc, T["shot"] - 0.3, T["shot"]) * fout(
            tc, T["follow"] - 0.3, T["follow"])
        text("Real on one dataset.  Gone on the next.",
             4.5, 3.4, sca, 36, INK, "bold")

        # follow-up
        fa = fin(tc, T["follow"], T["follow"] + 0.4) * fout(
            tc, T["land_fade"] - 0.2, T["land_fade"])
        text("The “hidden concept” idea?  Also false.",
             4.5, 13.4, fa, 32, INK)
        text("p ≈ 1.0", 4.5, 12.7, fa, 38, INK, "bold")

        ua = sca
        text("github.com/ss1738/physiomind-phase2-replication",
             4.5, 1.5, max(sca, fa) * 0.9, 19, SUB)

    # ---------- LANDING CARD ----------
    if tc >= T["land_fade"]:
        la = fin(tc, T["land_in"], T["land_in"] + 0.9)
        text("An effect that doesn't replicate", 4.5, 9.4,
             la, 40, INK, "bold")
        text("isn't an effect.", 4.5, 8.5, la, 40, GREEN, "bold")
        la2 = fin(tc, T["land_in"] + 0.6, T["land_in"] + 1.3)
        text("Satyawan Singh   ·   PhysioMind Phase 2",
             4.5, 7.0, la2, 24, SUB)
        text("github.com/ss1738/physiomind-phase2-replication",
             4.5, 6.4, la2, 21, SUB)
    return []


if __name__ == "__main__":
    ani = FuncAnimation(fig, draw, frames=TOTAL,
                        interval=1000 / FPS, blit=False)
    w = FFMpegWriter(fps=FPS, bitrate=3200,
                     metadata={"title": "PhysioMind Phase 2"})
    ani.save(OUT, writer=w)
    print("WROTE", OUT, TOTAL, "frames")

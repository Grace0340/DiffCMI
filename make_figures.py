#!/usr/bin/env python3
"""Generate all paper figures from the real results.json (IEEE-compliant, vector PDF)."""
import json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib import rcParams

# IEEE-style: Times-like serif, vector output, modest font sizes
rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif", "Times New Roman"]
rcParams["mathtext.fontset"] = "dejavuserif"
rcParams["axes.linewidth"] = 0.8
rcParams["pdf.fonttype"] = 42   # embed real fonts (TrueType), not Type-3 bitmaps
rcParams["ps.fonttype"] = 42

R = json.load(open("results.json"))
COL = {"zero": "#4C72B0", "mean": "#55A868", "mmin": "#C44E52", "diffcmi": "#8172B3"}
MK = {"zero": "o", "mean": "s", "mmin": "^", "diffcmi": "D"}
NAME = {"zero": "Zero-Imp", "mean": "Mean-Imp", "mmin": "MMIN", "diffcmi": "DiffCMI (ours)"}

# ----------------------------------------------------------------------------
# Figure 1: Missing-rate robustness on CMU-MOSEI (Acc-2 and MAE vs rate)
# ----------------------------------------------------------------------------
def fig_missing_rate():
    rates = [10, 30, 50, 70]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(7.0, 2.9))
    handles_labels = None
    for m in ["zero", "mean", "mmin", "diffcmi"]:
        acc = [R[f"mosei_rate{r}_{m}"]["acc2"] * 100 for r in rates]
        mae = [R[f"mosei_rate{r}_{m}"]["mae"] for r in rates]
        ax1.plot(rates, acc, marker=MK[m], color=COL[m], label=NAME[m],
                 linewidth=1.4, markersize=5)
        ax2.plot(rates, mae, marker=MK[m], color=COL[m], label=NAME[m],
                 linewidth=1.4, markersize=5)
    ax1.set_xlabel("Missing rate (%)"); ax1.set_ylabel("Binary Accuracy (%)")
    ax2.set_xlabel("Missing rate (%)"); ax2.set_ylabel("MAE")
    ax1.set_xticks(rates); ax2.set_xticks(rates)
    ax1.grid(alpha=0.3, linewidth=0.5); ax2.grid(alpha=0.3, linewidth=0.5)
    ax1.set_title("(a) Accuracy vs. missing rate", fontsize=9)
    ax2.set_title("(b) MAE vs. missing rate", fontsize=9)
    # single shared legend below both panels -> never overlaps any curve
    handles, labels = ax1.get_legend_handles_labels()
    fig.legend(handles, labels, fontsize=8, loc="lower center",
               ncol=4, frameon=False, bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    fig.savefig("fig_missing_rate.pdf", bbox_inches="tight")
    plt.close(fig)
    print("OK fig_missing_rate.pdf")

# ----------------------------------------------------------------------------
# Figure 2: Selective prediction curves (the flagship result) -- 3 datasets
# ----------------------------------------------------------------------------
def fig_selective():
    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.5))
    panels = [("mosi_random50_diffcmi", "(a) CMU-MOSI"),
              ("mosei_random50_diffcmi", "(b) CMU-MOSEI"),
              ("chsims_random50_diffcmi", "(c) CH-SIMS v2")]
    for ax, (key, title) in zip(axes, panels):
        curve = R[key]["uncertainty"]["selective_curve"]
        cov = [c["coverage"] * 100 for c in curve]
        acc = [c["acc2"] * 100 for c in curve]
        ax.plot(cov, acc, marker="D", color=COL["diffcmi"], linewidth=1.6, markersize=5)
        # reference line: accuracy at full coverage
        ax.axhline(acc[0], color="gray", linestyle="--", linewidth=0.8, alpha=0.7)
        ax.set_xlabel("Coverage (%)")
        ax.set_title(title, fontsize=9)
        ax.grid(alpha=0.3, linewidth=0.5)
        ax.invert_xaxis()  # high confidence (low coverage) on the right
        gain = acc[-1] - acc[0]
        ax.annotate(f"+{gain:.1f}", xy=(cov[-1], acc[-1]),
                    xytext=(cov[-1] + 8, acc[-1] - 2.5), fontsize=8,
                    color=COL["mmin"], fontweight="bold")
    axes[0].set_ylabel("Binary Accuracy (%)")
    fig.tight_layout()
    fig.savefig("fig_selective.pdf", bbox_inches="tight")
    plt.close(fig)
    print("? fig_selective.pdf")

# ----------------------------------------------------------------------------
# Figure 3: Uncertainty-error correlation across settings (bar chart)
# ----------------------------------------------------------------------------
def fig_unc_corr():
    settings = [("mosi_random50_diffcmi", "MOSI@0.5"),
                ("mosei_random50_diffcmi", "MOSEI@0.5"),
                ("chsims_random50_diffcmi", "SIMS@0.5"),
                ("mosei_rate30_diffcmi", "MOSEI@0.3"),
                ("mosei_rate10_diffcmi", "MOSEI@0.1"),
                ("mosei_rate70_diffcmi", "MOSEI@0.7")]
    labels = [s[1] for s in settings]
    sp = [R[s[0]]["uncertainty"]["unc_err_spearman"] for s in settings]
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    colors = [COL["diffcmi"] if v > 0 else "#999999" for v in sp]
    bars = ax.bar(range(len(sp)), sp, color=colors, width=0.65, edgecolor="black", linewidth=0.5)
    ax.axhline(0, color="black", linewidth=0.7)
    ax.set_xticks(range(len(sp)))
    ax.set_xticklabels(labels, rotation=40, ha="right", fontsize=7)
    ax.set_ylabel("Spearman $\\rho$ (unc. vs. error)")
    ax.grid(alpha=0.3, axis="y", linewidth=0.5)
    fig.tight_layout()
    fig.savefig("fig_unc_corr.pdf", bbox_inches="tight")
    plt.close(fig)
    print("? fig_unc_corr.pdf")

# ----------------------------------------------------------------------------
# Figure 4: Selective MAE improvement under different missing rates (MOSEI)
# ----------------------------------------------------------------------------
def fig_selective_rate():
    fig, ax = plt.subplots(figsize=(3.5, 2.6))
    for r, lab in [(10, "10%"), (30, "30%"), (50, "50%")]:
        curve = R[f"mosei_rate{r}_diffcmi"]["uncertainty"]["selective_curve"]
        cov = [c["coverage"] * 100 for c in curve]
        mae = [c["mae"] for c in curve]
        ax.plot(cov, mae, marker="o", linewidth=1.4, markersize=4, label=f"miss={lab}")
    ax.set_xlabel("Coverage (%)"); ax.set_ylabel("MAE")
    ax.invert_xaxis()
    ax.legend(fontsize=7); ax.grid(alpha=0.3, linewidth=0.5)
    fig.tight_layout()
    fig.savefig("fig_selective_rate.pdf", bbox_inches="tight")
    plt.close(fig)
    print("? fig_selective_rate.pdf")

if __name__ == "__main__":
    fig_missing_rate()
    fig_selective()
    fig_unc_corr()
    fig_selective_rate()
    print("All figures generated.")

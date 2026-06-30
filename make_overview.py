#!/usr/bin/env python3
"""DiffCMI architecture overview - precise edge-to-edge connections, no broken lines."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
from matplotlib import rcParams

rcParams["font.family"] = "serif"
rcParams["font.serif"] = ["DejaVu Serif"]
rcParams["pdf.fonttype"] = 42

fig, ax = plt.subplots(figsize=(7.6, 2.5))
ax.set_xlim(0, 15); ax.set_ylim(0.8, 6.0); ax.axis("off")

# Store box edges for precise connections
boxes = {}
def box(name, x, y, w, h, text, fc, fs=8):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.02,rounding_size=0.08",
                 linewidth=1.0, edgecolor="#333333", facecolor=fc, zorder=3))
    ax.text(x + w/2, y + h/2, text, ha="center", va="center", fontsize=fs, zorder=5)
    boxes[name] = dict(x=x, y=y, w=w, h=h, cx=x+w/2, cy=y+h/2,
                       L=(x, y+h/2), R=(x+w, y+h/2), T=(x+w/2, y+h), B=(x+w/2, y))

def arrow(p1, p2, color="#444444"):
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=11,
                 linewidth=1.1, color=color, zorder=2,
                 shrinkA=0, shrinkB=0))

def poly(pts, color):
    """Draw connected line segments through points, arrow at the end."""
    for i in range(len(pts)-2):
        ax.plot([pts[i][0], pts[i+1][0]], [pts[i][1], pts[i+1][1]],
                color=color, linewidth=1.1, zorder=2, solid_capstyle="round")
    arrow(pts[-2], pts[-1], color)

# ===== Boxes =====
box("text",  0.3, 4.55, 1.5, 0.7, "Text",  "#D6E4F0")
box("audio", 0.3, 3.15, 1.5, 0.7, "Audio", "#D6E4F0")
box("vision",0.3, 1.5,  1.5, 0.7, "Vision", "#F2D6D6", fs=7.5)
ax.text(1.05, 1.28, "(missing)", ha="center", fontsize=6, color="#A03030")

box("et", 2.5, 4.55, 1.3, 0.7, "Enc$_t$", "#E8E8E8")
box("ea", 2.5, 3.15, 1.3, 0.7, "Enc$_a$", "#E8E8E8")

box("ccma", 4.5, 3.35, 1.7, 1.7, "Contrastive\nAlignment", "#DCECDC", fs=7.5)
box("ctx",  6.9, 4.0,  1.3, 0.7, "context $c$", "#DCECDC", fs=7)
box("diff", 6.7, 1.9,  1.8, 1.4, "Latent\nDiffusion\nImputer", "#E5DCEF", fs=7.5)
box("vimp", 9.1, 1.95, 1.5, 0.7, "Vision$'$", "#E5DCEF", fs=7.5)
box("fuse", 11.3, 2.7, 1.9, 1.7, "Availability-\nAware\nFusion", "#FBEAD2", fs=7.5)
box("pred", 13.7, 3.6, 1.1, 0.7, "pred $\\bar y$", "#FAD9D9", fs=8)
box("unc",  13.7, 2.4, 1.1, 0.7, "unc. $u$", "#FAD9D9", fs=8)

BLUE="#3C6FB0"; PURPLE="#7A5BA6"; GRAY="#555555"

# ===== Connections (all edge-to-edge) =====
# inputs -> encoders
arrow(boxes["text"]["R"],  boxes["et"]["L"])
arrow(boxes["audio"]["R"], boxes["ea"]["L"])

# encoders -> CCMA
arrow(boxes["et"]["R"], (boxes["ccma"]["x"], 4.55))
arrow(boxes["ea"]["R"], (boxes["ccma"]["x"], 3.50))

# CCMA -> context
arrow(boxes["ccma"]["R"], boxes["ctx"]["L"])

# context -> diffusion imputer (vertical down, both centered at x=7.55 area)
poly([boxes["ctx"]["B"], (boxes["ctx"]["cx"], 3.55), (boxes["diff"]["cx"]+0.4, 3.55),
      (boxes["diff"]["cx"]+0.4, boxes["diff"]["y"]+boxes["diff"]["h"])], PURPLE)
ax.text(7.95, 3.55, "cond.", fontsize=6, color=PURPLE, va="bottom", ha="left")

# diffusion -> imputed vision
arrow(boxes["diff"]["R"], boxes["vimp"]["L"], PURPLE)
ax.text(7.6, 1.65, "$\\times N$ samples", fontsize=6.5, ha="center",
        style="italic", color=PURPLE)

# imputed vision -> fusion
poly([boxes["vimp"]["R"], (10.85, 2.3), (10.85, 3.05), (boxes["fuse"]["x"], 3.05)], PURPLE)

# genuine modalities skip-path: encoders -> top lane -> down into fusion
# Text skip (top lane y=5.6)
poly([boxes["et"]["T"], (boxes["et"]["cx"], 5.6), (11.0, 5.6),
      (11.0, 3.95), (boxes["fuse"]["x"], 3.95)], BLUE)
# Audio skip (lane y=5.15)
poly([boxes["ea"]["T"], (boxes["ea"]["cx"], 5.15), (10.7, 5.15),
      (10.7, 3.55), (boxes["fuse"]["x"], 3.55)], BLUE)
ax.text(5.5, 5.78, "genuine modalities (skip path)", fontsize=6.2, color=BLUE)

# fusion -> outputs
poly([boxes["fuse"]["R"], (13.35, 3.55), (13.35, 3.95), boxes["pred"]["L"]], GRAY)
poly([boxes["fuse"]["R"], (13.35, 3.55), (13.35, 2.75), boxes["unc"]["L"]], GRAY)

fig.tight_layout()
fig.savefig("fig_overview.pdf", bbox_inches="tight", pad_inches=0.02)
print("OK fig_overview.pdf (precise connections)")

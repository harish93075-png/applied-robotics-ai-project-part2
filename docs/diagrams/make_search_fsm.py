"""Draw the search-mode state machine of epuck_controller.py (docs/diagrams/search_fsm.png)."""
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

OUT = Path(__file__).with_name("search_fsm.png")
TEAL, INK, MUTED, BG = "#0F766E", "#1F2937", "#55606E", "#F1F5F4"

states = {  # name: (x, y, description)
    "SEARCH": (1.6, 3.0, "wander ~5 s, spin to scan\nlook for any target colour"),
    "APPROACH": (5.6, 3.0, "lock on to the colour\nsteer to keep it centred"),
    "RECORD": (9.6, 3.0, '"I see <colour>"\n+ summary so far\nrepaint block (found)'),
    "AVOID": (5.6, 0.7, "reverse, turn by a\nrandom angle"),
    "DONE": (9.6, 0.7, "final summary\n(all found or 10 min)"),
}
W, H = 2.9, 1.45

fig, ax = plt.subplots(figsize=(11.5, 4.6), dpi=200)
ax.set_xlim(0, 11.2)
ax.set_ylim(-0.3, 4.1)
ax.axis("off")
for name, (x, y, desc) in states.items():
    ax.add_patch(FancyBboxPatch((x - W / 2, y - H / 2), W, H, boxstyle="round,pad=0.05,rounding_size=0.18",
                                fc=BG, ec=TEAL, lw=2))
    ax.text(x, y + 0.38, name, ha="center", va="center", fontsize=13, fontweight="bold", color=TEAL)
    ax.text(x, y - 0.2, desc, ha="center", va="center", fontsize=9.5, color=INK, linespacing=1.25)


def arrow(a, b, label, rad=0.0, start=None, end=None, lx=0, ly=0):
    (x1, y1, _), (x2, y2, _) = states[a], states[b]
    p1 = start or (x1, y1)
    p2 = end or (x2, y2)
    ax.add_patch(FancyArrowPatch(p1, p2, arrowstyle="-|>", mutation_scale=16, lw=1.6, color=INK,
                                 connectionstyle=f"arc3,rad={rad}", shrinkA=0, shrinkB=0))
    ax.text((p1[0] + p2[0]) / 2 + lx, (p1[1] + p2[1]) / 2 + ly, label, ha="center", va="center", fontsize=8.8,
            color=MUTED, bbox=dict(fc="white", ec="none", pad=1.5))


arrow("SEARCH", "APPROACH", "blob ≥ 1% of view", start=(3.05, 3.15), end=(4.15, 3.15), ly=0.22)
arrow("APPROACH", "SEARCH", "lost for ~3 s", start=(4.15, 2.75), end=(3.05, 2.75), ly=-0.22)
arrow("APPROACH", "RECORD", "blob ≥ 35% or bumper", start=(7.05, 3.0), end=(8.15, 3.0), ly=0.22)
arrow("RECORD", "AVOID", "block recorded", start=(8.5, 2.27), end=(6.6, 1.43), lx=0.35, ly=0.12)
arrow("SEARCH", "AVOID", "obstacle ahead", start=(1.6, 2.27), end=(4.15, 0.7), rad=0.25, lx=-0.55, ly=-0.25)
arrow("AVOID", "SEARCH", "turn done, blocks left", start=(4.15, 0.95), end=(2.3, 2.27), rad=0.15, lx=0.55, ly=0.25)
arrow("AVOID", "DONE", "all blocks found", start=(7.05, 0.7), end=(8.15, 0.7), ly=0.22)
ax.text(0.1, 4.0, "Start: blocks moved to random, non-overlapping positions", fontsize=9.5, color=MUTED, va="top")
fig.tight_layout()
fig.savefig(OUT, facecolor="white")
print("wrote", OUT)

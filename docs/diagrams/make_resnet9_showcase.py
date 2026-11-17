"""Showcase poster of the final ResNet-9, for the title/architecture slide and the report.

Layer shapes and parameter counts come from make_cnn_diagrams.resnet9_spec() (checked against
Keras' 6,577,610). Results are the real numbers from cnn/outputs/results.json, and the input/
output panel uses my own car photo and ResNet-9's actual top-3 prediction for it.

Run:  python docs/diagrams/make_resnet9_showcase.py   ->  docs/diagrams/resnet9_showcase.png
"""
import json
from math import log2
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch, FancyBboxPatch, Circle
from PIL import Image

from make_cnn_diagrams import resnet9_spec, fmt_shape

HERE = Path(__file__).parent
ROOT = HERE.parent.parent
RES = json.loads((ROOT / "cnn/outputs/results.json").read_text())
R9 = RES["resnet9"]
CAR = next(p for p in RES["predictions_resnet9"]["own_crop"] if p["image"] == "my_car.jpg")
INT8 = RES["optimisation_resnet9"]["tflite_int8"]

BG0, BG1 = "#06181A", "#0E3431"
WHITE, MUTED, LIME = "#EAF6F2", "#8FB3AB", "#C6F23A"
NEON = {"conv": "#33E6C4", "res": "#8C7CFF", "in": "#D9E8E4", "out": LIME}
plt.rcParams["font.family"] = ["Segoe UI", "DejaVu Sans"]


def shade(hex_colour, k):
    r, g, b = (int(hex_colour[i:i + 2], 16) / 255 for i in (1, 3, 5))
    if k < 1:
        return (r * k, g * k, b * k)
    return (r + (1 - r) * (k - 1), g + (1 - g) * (k - 1), b + (1 - b) * (k - 1))


def glow_poly(ax, pts, colour, z):
    for lw, a in [(14, .04), (9, .07), (5, .12)]:
        ax.add_patch(Polygon(pts, closed=True, fill=False, ec=colour, lw=lw, alpha=a, zorder=z,
                             joinstyle="round"))


def slab(ax, x, cy, h, t, colour, z=5):
    d = h * 0.45
    top, bot = cy + h / 2, cy - h / 2
    front = [(x, bot), (x + t, bot), (x + t, top), (x, top)]
    topf = [(x, top), (x + t, top), (x + t + d, top + d * .55), (x + d, top + d * .55)]
    side = [(x + t, bot), (x + t + d, bot + d * .55), (x + t + d, top + d * .55), (x + t, top)]
    outline = [(x, bot), (x + t, bot), (x + t + d, bot + d * .55), (x + t + d, top + d * .55),
               (x + d, top + d * .55), (x, top)]
    glow_poly(ax, outline, colour, z)
    for face, k, a in [(side, .35, .95), (topf, 1.25, .95), (front, .62, .95)]:
        ax.add_patch(Polygon(face, closed=True, fc=shade(colour, k), ec=colour, lw=1.3, alpha=a, zorder=z + 1))
    # feature-map grid lines on the front face: one line per 4 rows, hints at spatial size
    return x + t + d, top + d * .55


def glow_line(ax, xs, ys, colour, z=3, lw=2.2):
    for w, a in [(lw * 6, .05), (lw * 3, .10), (lw, .9)]:
        ax.plot(xs, ys, color=colour, lw=w, alpha=a, solid_capstyle="round", zorder=z)


def main():
    layers = resnet9_spec()
    total = sum(l[2] for l in layers)
    assert total == R9["params"] == 6_577_610

    fig = plt.figure(figsize=(24, 13.5), dpi=160)
    ax = fig.add_axes([0, 0, 1, 1]); ax.set_xlim(0, 2400); ax.set_ylim(0, 1350); ax.axis("off")

    # background: vertical gradient + radial light + dot grid
    grad = np.linspace(0, 1, 512)[:, None]
    c0, c1 = np.array(shade(BG0, 1.0)), np.array(shade(BG1, 1.0))
    yy, xx = np.mgrid[0:1:512j, 0:1:512j]
    radial = np.exp(-(((xx - .45) / .45) ** 2 + ((yy - .55) / .35) ** 2))[..., None]
    img = c0 * (1 - grad[..., None] * .4) + (c1 - c0) * radial * .9
    ax.imshow(np.clip(img, 0, 1), extent=[0, 2400, 0, 1350], aspect="auto", zorder=0, origin="lower")
    gx, gy = np.meshgrid(np.arange(30, 2400, 48), np.arange(30, 1350, 48))
    ax.scatter(gx, gy, s=1.2, color=LIME, alpha=.10, zorder=1)

    # title block
    ax.text(90, 1245, "ResNet-9", fontsize=58, fontweight="bold", color=WHITE, zorder=10)
    ax.text(92, 1178, "A 9-layer residual CNN trained from scratch on CIFAR-10", fontsize=21,
            color=MUTED, zorder=10)
    chips = [(f"{R9['test_accuracy']:.1%}", "test accuracy"),
             (f"{R9['test_accuracy_tta']:.1%}", "with flip TTA"),
             (f"{total / 1e6:.2f} M", "parameters"),
             (f"{INT8['size_kb'] / 1024:.1f} MB · {INT8['latency_ms']:.0f} ms", "TFLite int8, 1 CPU thread")]
    cx = 1170
    for val, lab in chips:
        w = 60 + 21 * len(val)
        w = max(w, 14 * len(lab) + 40)
        ax.add_patch(FancyBboxPatch((cx, 1160), w, 128, boxstyle="round,pad=0,rounding_size=18",
                                    fc=(1, 1, 1, .045), ec=(0.78, 0.95, 0.23, .35), lw=1.4, zorder=9))
        ax.text(cx + 24, 1240, val, fontsize=27, fontweight="bold", color=LIME, zorder=10)
        ax.text(cx + 24, 1186, lab, fontsize=13.5, color=MUTED, zorder=10)
        cx += w + 22

    # input photo (pixelated 32x32, exactly what the network sees)
    cy = 640
    car32 = Image.open(ROOT / "cnn/my_images/my_car.jpg").convert("RGB")
    W, H = car32.size
    car32 = car32.crop((int(.11 * W), int(.41 * H), int(.89 * W), int(.85 * H)))
    s = min(car32.size); l, t = (car32.size[0] - s) // 2, (car32.size[1] - s) // 2
    car32 = car32.crop((l, t, l + s, t + s)).resize((32, 32), Image.BILINEAR)
    ax.imshow(np.asarray(car32), extent=[60, 230, cy - 85, cy + 85], interpolation="nearest", zorder=6)
    glow_poly(ax, [(60, cy - 85), (230, cy - 85), (230, cy + 85), (60, cy + 85)], NEON["in"], 5)
    ax.text(145, cy + 128, "Input", ha="center", fontsize=17, fontweight="bold", color=WHITE, zorder=10)
    ax.text(145, cy - 135, "32×32×3", ha="center", fontsize=17, family="monospace",
            fontweight="bold", color=LIME, zorder=10)
    ax.text(145, cy - 168, "my car photo", ha="center", fontsize=12.5, color=MUTED, zorder=10)

    # stages
    stages = layers[1:]
    x0, x_end = 330, 1930
    gap = (x_end - x0) / (len(stages) - 0.4)
    mids, prev_right = [], 236
    for i, (name, shape, params, _, colour) in enumerate(stages):
        x = x0 + i * gap
        kind = "res" if name.startswith("Residual") else ("out" if name.startswith("Dense") else
                                                          "in" if "Pooling" in name else "conv")
        neon = NEON[kind]
        if isinstance(shape, tuple):
            h = 60 + 7.2 * shape[0]
            tth = 10 + 13 * log2(shape[2] / 16 + 1)
            right, top = slab(ax, x, cy, h, tth, neon)
            mid = x + (right - x) / 2
        else:
            h = 300 if shape > 10 else 150
            for w_, a in [(26, .06), (16, .12)]:
                ax.add_patch(FancyBboxPatch((x - w_ / 2 + 12, cy - h / 2 - w_ / 2), 24 + w_ - 24 + 0, h + w_,
                                            boxstyle="round,pad=0,rounding_size=10", fc=neon, alpha=a, ec="none", zorder=4))
            ax.add_patch(FancyBboxPatch((x, cy - h / 2), 24, h, boxstyle="round,pad=0,rounding_size=6",
                                        fc=shade(neon, .7), ec=neon, lw=1.5, zorder=6))
            right, top, mid = x + 24, cy + h / 2, x + 12
        glow_line(ax, [prev_right + 14, x - 14], [cy, cy], "#5FA79A", z=3, lw=1.8)
        ax.add_patch(FancyArrowPatch((x - 30, cy), (x - 10, cy), arrowstyle="-|>", mutation_scale=16,
                                     color="#7FC9BA", lw=0, zorder=4))
        prev_right = right
        mids.append(mid)
        tops = locals().setdefault("tops", [])
        tops.append(top)
        label = name.replace("\n", "\n", 1)
        ax.text(mid, 905, label, ha="center", va="bottom", fontsize=14, fontweight="bold",
                color=WHITE, linespacing=1.25, zorder=10)
        ax.text(mid, 405, fmt_shape(shape), ha="center", fontsize=16, family="monospace",
                fontweight="bold", color=LIME, zorder=10)
        ax.text(mid, 372, f"{params:,} params" if params else "no weights", ha="center",
                fontsize=12.5, color=MUTED, zorder=10)

        if kind == "res":   # glowing skip connection with the add node
            a, b = mids[-2], mid
            y0, y1 = tops[-2] + 8, top + 30            # leave the skipped block, land on the add node
            arc_y = 845
            t_ = np.linspace(0, np.pi, 80)
            xs = a + (b - a) * (1 - np.cos(t_)) / 2
            base = y0 + (y1 - y0) * (1 - np.cos(t_)) / 2
            ys = base + (arc_y - (y0 + y1) / 2) * np.sin(t_)
            glow_line(ax, xs, ys, NEON["res"], z=7, lw=2.4)
            ax.add_patch(Circle((b, y1), 17, fc=BG0, ec=NEON["res"], lw=2.4, zorder=8))
            ax.text(b, y1 - 1, "+", ha="center", va="center", fontsize=20, fontweight="bold",
                    color=NEON["res"], zorder=9)
            ax.text((a + b) / 2, arc_y + 14, "skip  x", ha="center", fontsize=12.5, color=NEON["res"],
                    fontweight="bold", zorder=9)

    # output: real top-3 for my car
    ox = 2025
    ax.text(ox, 905, "Prediction", fontsize=14, fontweight="bold", color=WHITE, va="bottom", zorder=10)
    for k, (cls, p) in enumerate(CAR["top3"]):
        yb = cy + 70 - k * 78
        ax.add_patch(FancyBboxPatch((ox, yb), 300, 34, boxstyle="round,pad=0,rounding_size=8",
                                    fc=(1, 1, 1, .06), ec="none", zorder=6))
        ax.add_patch(FancyBboxPatch((ox, yb), max(8, 300 * p), 34, boxstyle="round,pad=0,rounding_size=8",
                                    fc=LIME if k == 0 else "#4E8F84", ec="none", zorder=7))
        ax.text(ox, yb + 46, f"{cls}", fontsize=14, color=WHITE if k == 0 else MUTED,
                fontweight="bold" if k == 0 else "normal", zorder=10)
        ax.text(ox + 300, yb + 46, f"{p:.1%}", ha="right", fontsize=14,
                color=LIME if k == 0 else MUTED, fontweight="bold", zorder=10)
    ax.text(ox, cy - 168, "softmax of the 10 logits", fontsize=12.5, color=MUTED, zorder=10)

    # parameter budget bar
    by, bx, bw = 205, 90, 2220
    ax.text(bx, by + 75, "Where the 6,577,610 parameters live", fontsize=16, fontweight="bold",
            color=WHITE, zorder=10)
    acc = bx
    for name, shape, params, _, colour in stages:
        if not params:
            continue
        frac = params / total
        kind = "res" if name.startswith("Residual") else ("out" if name.startswith("Dense") else "conv")
        w = bw * frac
        ax.add_patch(FancyBboxPatch((acc + 2, by), max(w - 4, 3), 40, boxstyle="round,pad=0,rounding_size=6",
                                    fc=shade(NEON[kind], .85 if kind != "res" else .9), ec="none", zorder=6))
        if frac > 0.10:
            ax.text(acc + w / 2, by + 13, f"{name.split(chr(10))[0]} · {frac:.0%}", ha="center",
                    fontsize=12.5, color=BG0, fontweight="bold", zorder=10)
        acc += w
    small = {n.split(chr(10))[0]: p / total for n, _, p, _, _ in stages if p and p / total <= 0.10}
    ax.text(bx, by - 40, "Smaller slices: " + " · ".join(f"{k} {v:.1%}" if v >= 0.001 else f"{k} <0.1%" for k, v in small.items()) +
            ".  Most of the capacity sits in the last residual block, at 4×4×512.",
            fontsize=13, color=MUTED, zorder=10)

    # footer
    ax.text(90, 70, f"Trained {R9['epochs']} epochs in {R9['train_minutes']} min on a Colab T4 · "
            "SGD Nesterov one-cycle (peak 0.1) · random crop + flip + Cutout · label smoothing 0.1 · L2 5e-4 · "
            "mixed precision", fontsize=13, color=MUTED, zorder=10)
    ax.text(2310, 70, "Harish A · Applied Robotics & AI Project Part 2", ha="right", fontsize=13,
            color=MUTED, zorder=10)

    out = HERE / "resnet9_showcase.png"
    fig.savefig(out, facecolor=BG0)
    plt.close(fig)
    print("saved", out)


if __name__ == "__main__":
    main()

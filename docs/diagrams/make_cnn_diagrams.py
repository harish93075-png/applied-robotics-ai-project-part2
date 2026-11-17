"""Draw architecture diagrams of the two CIFAR-10 CNNs in cnn/CNN_Project_Part2.ipynb.

Every output shape and parameter count is computed here from the layer
definitions (Keras rules: 'valid' conv shrinks by k-1, 'same' keeps size,
2x2 max-pool floors the halving, BatchNorm has 4 params per channel of which
2 are trainable). The totals are checked against the numbers Keras reports
for the notebook models, so the picture cannot drift from the code.

Run:  python docs/diagrams/make_cnn_diagrams.py
Out:  docs/diagrams/baseline_cnn.png/.svg, improved_cnn.png/.svg
"""
from math import log2
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Polygon, FancyArrowPatch

OUT = Path(__file__).parent
INK, MUTED, PAPER = "#102624", "#4E625F", "#F3F6F1"
CONV, POOL, DENSE, OUTC, INP = "#0E7C6B", "#E8573F", "#12403D", "#B8E62E", "#9DB3AD"
RES = "#2F6FB0"
CLASSES = ["airplane", "automobile", "bird", "cat", "deer",
           "dog", "frog", "horse", "ship", "truck"]


# ---------------------------------------------------------------- shape maths
def conv(shape, filters, k=3, padding="valid", bias=True):
    h, w, c = shape
    if padding == "valid":
        h, w = h - (k - 1), w - (k - 1)
    params = k * k * c * filters + (filters if bias else 0)
    return (h, w, filters), params


def pool(shape):
    h, w, c = shape
    return (h // 2, w // 2, c), 0


def bn(shape):
    return shape, 4 * shape[2]          # gamma, beta trainable; mean, var not


def dense(n_in, units):
    return units, n_in * units + units


# ------------------------------------------------------------ model specs
def baseline_spec():
    s = (32, 32, 3)
    layers = [("Input", s, 0, "32×32 RGB\n÷ 255", INP)]
    s, p = conv(s, 32); layers.append(("Conv2D 32\n3×3 · ReLU", s, p, None, CONV))
    s, p = pool(s);     layers.append(("MaxPool\n2×2", s, p, None, POOL))
    s, p = conv(s, 64); layers.append(("Conv2D 64\n3×3 · ReLU", s, p, None, CONV))
    s, p = pool(s);     layers.append(("MaxPool\n2×2", s, p, None, POOL))
    s, p = conv(s, 64); layers.append(("Conv2D 64\n3×3 · ReLU", s, p, None, CONV))
    flat = s[0] * s[1] * s[2]
    layers.append(("Flatten", flat, 0, None, INP))
    n, p = dense(flat, 64); layers.append(("Dense 64\nReLU", n, p, None, DENSE))
    n, p = dense(n, 10);    layers.append(("Dense 10\nlogits", n, p, None, OUTC))
    return layers


def improved_spec():
    s = (32, 32, 3)
    layers = [("Input", s, 0, "32×32 RGB\n÷ 255", INP)]
    blocks = []
    for filters, drop in [(32, 0.2), (64, 0.3), (128, 0.4)]:
        total = 0
        for _ in range(2):
            s, p = conv(s, filters, padding="same", bias=False); total += p
            s, p = bn(s); total += p
        layers.append((f"2 × [Conv2D {filters}\n3×3 · BN · ReLU]", s, total, None, CONV))
        s, _ = pool(s)
        layers.append((f"MaxPool 2×2\nDropout {drop}", s, 0, None, POOL))
        blocks.append(total)
    n = s[2]
    layers.append(("Global Avg\nPooling", n, 0, None, INP))
    n2, p = dense(n, 256); layers.append(("Dense 256 · ReLU\nDropout 0.5", n2, p, None, DENSE))
    n3, p = dense(n2, 10); layers.append(("Dense 10\nlogits", n3, p, None, OUTC))
    return layers


def resnet9_spec():
    """ResNet-9 (Page, 2018): conv blocks have no bias, BatchNorm, ReLU; residual = x + F(x)."""
    def cbr(shape, f):
        shape, p1 = conv(shape, f, padding="same", bias=False)
        shape, p2 = bn(shape)
        return shape, p1 + p2

    s = (32, 32, 3)
    layers = [("Input", s, 0, "32×32 RGB\n÷ 255", INP)]
    s, p = cbr(s, 64);  layers.append(("Prep\nConv 64", s, p, None, CONV))
    s, p = cbr(s, 128); s, _ = pool(s)
    layers.append(("Layer 1\nConv 128\n+ MaxPool", s, p, None, CONV))
    s1, p1 = cbr(s, 128); s1, p2 = cbr(s1, 128)
    layers.append(("Residual 1\n2× Conv 128\nx + F(x)", s1, p1 + p2, None, RES))
    s, p = cbr(s1, 256); s, _ = pool(s)
    layers.append(("Layer 2\nConv 256\n+ MaxPool", s, p, None, CONV))
    s, p = cbr(s, 512); s, _ = pool(s)
    layers.append(("Layer 3\nConv 512\n+ MaxPool", s, p, None, CONV))
    s1, p1 = cbr(s, 512); s1, p2 = cbr(s1, 512)
    layers.append(("Residual 2\n2× Conv 512\nx + F(x)", s1, p1 + p2, None, RES))
    layers.append(("Global Max\nPooling", s1[2], 0, None, INP))
    n, p = dense(s1[2], 10); layers.append(("Dense 10\nlogits", n, p, None, OUTC))
    return layers


# ---------------------------------------------------------------- drawing
def cuboid(ax, x, cy, h, t, colour):
    """Feature map: front face t wide (channels) x h tall (rows), depth h*0.42."""
    d = h * 0.42
    top, bot = cy + h / 2, cy - h / 2
    faces = [
        [(x, bot), (x + t, bot), (x + t, top), (x, top)],                       # front
        [(x, top), (x + t, top), (x + t + d, top + d * .6), (x + d, top + d * .6)],  # top
        [(x + t, bot), (x + t + d, bot + d * .6), (x + t + d, top + d * .6), (x + t, top)],  # side
    ]
    shades = [1.0, 1.25, 0.8]
    for f, k in zip(faces, shades):
        ax.add_patch(Polygon(f, closed=True, fc=_shade(colour, k), ec=INK, lw=1.1))
    return x + t + d   # right-most x


def _shade(hex_colour, k):
    r, g, b = (int(hex_colour[i:i + 2], 16) for i in (1, 3, 5))
    f = (lambda v: min(255, int(v * k))) if k <= 1 else (lambda v: int(v + (255 - v) * (k - 1)))
    return "#%02x%02x%02x" % (f(r), f(g), f(b))


def fmt_shape(s):
    return f"{s[0]}×{s[1]}×{s[2]}" if isinstance(s, tuple) else f"{s:,}"


def draw(layers, total_expected, title, fname, note):
    total = sum(l[2] for l in layers)
    assert total == total_expected, f"{fname}: {total} != {total_expected}"

    fig_w, fig_h = 21.6, 7.0
    fig, ax = plt.subplots(figsize=(fig_w, fig_h), dpi=150)
    fig.patch.set_facecolor(PAPER); ax.set_facecolor(PAPER)
    ax.set_xlim(0, 2160); ax.set_ylim(-30, 660); ax.axis("off")
    cy, gap = 330, 1980 / (len(layers) + 0.2)

    prev_right = None
    for i, (name, shape, params, extra, colour) in enumerate(layers):
        x = 40 + i * gap
        if isinstance(shape, tuple):
            h = 34 + 5.2 * shape[0]                # rows -> height
            t = 8 + 11 * log2(shape[2] / 3 + 1)    # channels -> thickness
            right = cuboid(ax, x, cy, h, t, colour)
            mid = (x + right) / 2
        else:                                      # 1-D vector
            h = 60 + 42 * log2(shape) ** 0.9 if shape > 10 else 150
            h = min(h, 330)
            ax.add_patch(Polygon([(x, cy - h / 2), (x + 26, cy - h / 2), (x + 26, cy + h / 2), (x, cy + h / 2)],
                                 closed=True, fc=colour, ec=INK, lw=1.1))
            right, mid = x + 26, x + 13
        if prev_right is not None:
            ax.add_patch(FancyArrowPatch((prev_right + 6, cy), (x - 6, cy), arrowstyle="-|>",
                                         mutation_scale=14, color=MUTED, lw=1.4))
        if name.startswith("Residual"):            # skip connection: x is added to F(x)
            ax.add_patch(FancyArrowPatch((prev_mid, cy + 105), (mid, cy + 105), arrowstyle="-|>",
                                         connectionstyle="arc3,rad=-0.45", mutation_scale=14,
                                         color=RES, lw=1.8, ls="--"))
            ax.text((prev_mid + mid) / 2, cy + 172, "skip: + x", ha="center", fontsize=9,
                    color=RES, fontweight="bold")
        prev_right, prev_mid = right, mid
        ax.text(mid, 572, name, ha="center", va="center", fontsize=10, fontweight="bold", color=INK, linespacing=1.3)
        ax.text(mid, 120, fmt_shape(shape), ha="center", va="center", fontsize=11.5,
                family="monospace", fontweight="bold", color=INK)
        ax.text(mid, 88, f"{params:,} params" if params else "no params", ha="center",
                va="center", fontsize=9.5, color=MUTED)
        if extra:
            ax.text(mid, 200, extra, ha="center", va="center", fontsize=9, color=MUTED)

    # class list beside the output vector
    ax.text(prev_right + 30, cy, "\n".join(CLASSES), ha="left", va="center", fontsize=8.5,
            color=MUTED, family="monospace", linespacing=1.25)
    ax.text(40, 625, title, ha="left", va="center", fontsize=14, fontweight="bold", color=INK)
    ax.text(2150, 625, f"Total {total:,} parameters", ha="right", va="center",
            fontsize=13, fontweight="bold", color=POOL)
    ax.text(40, -12, note, ha="left", va="center", fontsize=9.5, color=MUTED)

    # legend: only the kinds of layer this diagram uses
    used = {l[4] for l in layers}
    items = [(lab, c) for lab, c in [("convolution", CONV), ("pooling", POOL), ("residual block", RES),
             ("dense", DENSE), ("output logits", OUTC), ("input / reshape", INP)] if c in used]
    for k, (lab, c) in enumerate(items):
        lx = 40 + k * 190
        ax.add_patch(Polygon([(lx, 24), (lx + 18, 24), (lx + 18, 40), (lx, 40)], fc=c, ec=INK, lw=.8))
        ax.text(lx + 26, 32, lab, va="center", fontsize=9, color=INK)

    for ext in ("png", "svg"):
        fig.savefig(OUT / f"{fname}.{ext}", facecolor=PAPER, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    print(f"{fname}: {total:,} params, {len(layers)} stages")
    for name, shape, params, *_ in layers:
        print(f"   {name.replace(chr(10), ' '):<34}{fmt_shape(shape):>12}{params:>10,}")


if __name__ == "__main__":
    draw(baseline_spec(), 122_570, "Baseline – Session 8 CNN_v3 (all layers)", "baseline_cnn",
         "'valid' 3×3 convolutions shrink each side by 2; 2×2 max-pooling halves it (rounded down). "
         "Output = 10 logits; softmax applied only to read probabilities.")
    draw(improved_spec(), 323_946, "Improved model – BatchNorm, Dropout, Global Average Pooling",
         "improved_cnn",
         "'same' padding keeps size; conv layers have no bias (BatchNorm adds the shift). "
         "BN params include 896 non-trainable moving statistics. Augmentation runs in the input pipeline.")
    draw(resnet9_spec(), 6_577_610, "ResNet-9 – final model (residual connections)", "resnet9_cnn",
         "'same' padding, no conv bias (BatchNorm), ReLU after every conv. Residual blocks output x + F(x) "
         "at the same shape. Trained with crop/flip/Cutout, label smoothing 0.1, SGD one-cycle.")

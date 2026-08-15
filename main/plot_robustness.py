#!/usr/bin/env python3
"""
Render the shaping-robustness figure from the committed results_drl/ sweeps.

The headline the four-arm benchmark actually supports is not parameter
efficiency -- once classical-small is sized to just exceed the VQC's parameter
count, the MLP matches or beats the circuit at every depth *with* reward
shaping. The result that survives is what happens when shaping is removed: the
circuit keeps solving, the parameter-matched MLP largely stops.

    ./plot_robustness.py                    # writes both light and dark PNGs
    ./plot_robustness.py --out-dir docs/

Palette: slots 2-3 of the validated categorical set (orange/aqua), which clear
the all-pairs CVD and normal-vision floors in both modes (worst CVD dE 9.2
light / 9.4 dark). Aqua sits at 2.74:1 on the light surface, below the 3:1 bar,
so the relief rule applies and every bar carries a visible value label.
"""

import argparse
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch

DEPTHS = (1, 2, 3, 5)

# The oversized `classical` baseline is deliberately absent: it solves nearly
# everything in both conditions, so it flattens the axis and buries the only
# comparison this figure exists to make. Its numbers stay in the README tables.
ARMS = ("classical-small", "quantum")
LABEL = {
    "classical-small": "classical-small  (MLP, parameter-matched)",
    "quantum": "quantum  (VQC)",
}

# Hues are slots 2 and 3 of the validated categorical set -- the same steps
# these two arms carried when `classical` (slot 1) was also plotted. Dropping a
# series must never repaint the survivors: colour follows the entity, not its
# rank in the current chart.
THEME = {
    "light": dict(surface="#fcfcfb", ink="#0b0b0b", ink2="#52514e", grid="#e3e2df",
                  series=("#eb6834", "#1baf7a")),
    "dark":  dict(surface="#1a1a19", ink="#ffffff", ink2="#c3c2b7", grid="#3a3a38",
                  series=("#d95926", "#199e70")),
}


def load(root: Path):
    """solved counts and parameter counts per (shaping, depth, arm)."""
    data, params = {}, {}
    for shaped, tag in ((True, ""), (False, "-no-rwshp")):
        for d in DEPTHS:
            f = root / f"nlayer{d}{tag}" / "benchmark.json"
            rows = {r["arm"]: r for r in json.load(open(f))["summary"]}
            for a in ARMS:
                data[(shaped, d, a)] = rows[a]["solved"]
                params[(d, a)] = rows[a]["params"]
    return data, params


def rounded_bar(ax, x, w, h, color, r_px=4.0):
    """A bar with a 4px rounded top, anchored to the baseline.

    matplotlib's bar() has square ends, so each bar is a FancyBboxPatch. The
    corner radius is specified in pixels and converted to the axes' own x/y
    data scales, which differ -- mutation_aspect carries that ratio so the
    corner comes out circular on screen rather than stretched.
    """
    if h <= 0:
        # A zero bar still needs a mark, or the category reads as missing data
        # rather than as a measured zero.
        ax.plot([x - w / 2, x + w / 2], [0.035, 0.035], color=color, lw=2.5,
                solid_capstyle="round", zorder=3)
        return
    # pixels -> data units, per axis
    px = ax.transData.inverted().transform
    x0, y0 = px((0, 0))
    rx, ry = px((r_px, r_px)) - [x0, y0]
    rx, ry = min(abs(rx), w / 2), min(abs(ry), h / 2)
    if ry <= 0 or rx <= 0:
        ax.add_patch(plt.Rectangle((x - w / 2, 0), w, h, linewidth=0,
                                   facecolor=color, zorder=3))
        return
    ax.add_patch(FancyBboxPatch(
        (x - w / 2 + rx, ry), w - 2 * rx, max(h - 2 * ry, 1e-6),
        boxstyle=f"round,pad=0,rounding_size={rx}",
        linewidth=0, facecolor=color, zorder=3, mutation_aspect=ry / rx))


def render(data, params, mode, out_path):
    t = THEME[mode]
    fig, axes = plt.subplots(1, 2, figsize=(11.4, 4.6), dpi=200, sharey=True)
    fig.patch.set_facecolor(t["surface"])
    # Bars are drawn in pixel-derived coordinates, so the layout has to be
    # settled before any patch is placed.
    fig.subplots_adjust(left=0.075, right=0.985, top=0.775, bottom=0.175, wspace=0.07)

    width, gap = 0.32, 0.03
    centre = (len(ARMS) - 1) / 2          # keeps the group centred on its tick
    for ax, shaped, title in ((axes[0], True, "Reward shaping ON"),
                              (axes[1], False, "Reward shaping OFF")):
        ax.set_facecolor(t["surface"])
        # Limits first: rounded_bar reads transData to size its corners.
        ax.set_ylim(0, 11.4)
        ax.set_xlim(-0.6, len(DEPTHS) - 0.4)

        for k, arm in enumerate(ARMS):
            off = (k - centre) * (width + gap)
            for i, d in enumerate(DEPTHS):
                v = data[(shaped, d, arm)]
                rounded_bar(ax, i + off, width, v, t["series"][k])
                ax.text(i + off, v + 0.28, str(v), ha="center", va="bottom",
                        fontsize=8.5, color=t["ink2"], zorder=4)

        ax.set_title(title, color=t["ink"], fontsize=12, pad=10, loc="left")
        ax.set_xticks(range(len(DEPTHS)))
        ax.set_xticklabels(
            [f"{d}\n{params[(d,'quantum')]}p vs {params[(d,'classical-small')]}p"
             for d in DEPTHS], color=t["ink2"], fontsize=9)
        ax.set_xlabel("VQC depth  (circuit params vs matched-MLP params)",
                      color=t["ink2"], fontsize=9.5, labelpad=6)
        ax.set_yticks(range(0, 11, 2))
        ax.tick_params(colors=t["ink2"], length=0)
        ax.grid(axis="y", color=t["grid"], lw=0.8, zorder=0)
        ax.set_axisbelow(True)
        for s in ("top", "right", "left"):
            ax.spines[s].set_visible(False)
        ax.spines["bottom"].set_color(t["grid"])

    axes[0].set_ylabel("Seeds solved  (of 10)", color=t["ink2"], fontsize=10)

    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=t["series"][k], linewidth=0)
               for k in range(len(ARMS))]
    leg = fig.legend(handles, [LABEL[a] for a in ARMS], loc="upper center",
                     bbox_to_anchor=(0.5, 0.915), ncol=len(ARMS), frameon=False,
                     fontsize=9.5, handlelength=1.1, handleheight=1.1,
                     columnspacing=2.0)
    for txt in leg.get_texts():
        txt.set_color(t["ink2"])

    fig.suptitle("Remove the hand-designed reward and the circuit keeps solving; "
                 "the size-matched network does not",
                 color=t["ink"], fontsize=13.5, y=0.985, x=0.5)
    fig.savefig(out_path, facecolor=t["surface"])
    plt.close(fig)
    print(f"wrote {out_path}")


def main():
    p = argparse.ArgumentParser()
    here = Path(__file__).resolve().parent
    p.add_argument("--results", default=str(here.parent / "results_drl"))
    p.add_argument("--out-dir", default=str(here.parent / "results_drl"))
    args = p.parse_args()

    data, params = load(Path(args.results))
    out = Path(args.out_dir)
    out.mkdir(parents=True, exist_ok=True)
    for mode in ("light", "dark"):
        render(data, params, mode, out / f"shaping_robustness-{mode}.png")


if __name__ == "__main__":
    main()

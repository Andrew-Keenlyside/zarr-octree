"""Figures for ``bench_bounds.py``: object sizes and query times, uniform grids against bounded octree runs.

    python bench/figures_bounds.py [--results bench/results/bench_bounds.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.lines import Line2D

from figures import COLOR, HERE, INK2, MUTED, SURFACE

UNI, OCT = COLOR["coarse"], COLOR["octree"]
MARK = dict(markersize=7, markeredgewidth=1.4)
FAMILY_ORDER = ["random", "compact", "cluster", "curve", "box-dense", "box-uniform"]


def label(name: str) -> str:
    if name.startswith("u"):
        g = int(name[1:])
        return f"uniform {g}³ grid ({1000 / g:.1f} u)"
    g, lo, hi = name[1:].split("-")
    return f"octree {g}³ leaves, {lo}–{hi} vtx"


def lat_name(lat: float, bw: float | None) -> str:
    return "local disk" if not lat else f"{lat * 1e3:.0f} ms + {bw / 1e6:.0f} MB/s per request"


def medians(q: list) -> dict:
    """``{(latency, reader, layout, kind, size): median seconds}`` over queries that returned points."""
    groups: dict = {}
    for x in q:
        if x["points"] > 0:
            groups.setdefault((x["latency"], x["reader"], x["layout"], x["kind"], x["size"]), []).append(x["seconds"])
    return {k: float(np.median(v)) for k, v in groups.items()}


def scores(res: dict) -> dict:
    """``{(latency, reader, layout): geometric mean over query families of the median time}``."""
    med = medians(res["queries"])
    fams = sorted({k[3:] for k in med})
    out = {}
    for lat, reader, layout in {k[:3] for k in med}:
        vals = [med.get((lat, reader, layout, *f)) for f in fams]
        if all(v is not None for v in vals):
            out[lat, reader, layout] = float(np.exp(np.mean(np.log(vals))))
    return out


def layouts(res: dict) -> list[str]:
    return list(res["objects"])


def fig_objects(res: dict, out: Path) -> None:
    names = layouts(res)
    fig, (ax, ax2) = plt.subplots(1, 2, figsize=(14, 0.42 * len(names) + 1.6), sharey=True, gridspec_kw={"width_ratios": [3, 1]})
    y = np.arange(len(names))[::-1]
    for yi, n in zip(y, names):
        o = res["objects"][n]
        c = UNI if n.startswith("u") else OCT
        ax.plot([max(o["min"], 0.5), o["max"]], [yi, yi], color=c, linewidth=1, alpha=0.5)
        ax.plot([max(o["min"], 0.5), o["p99"]], [yi, yi], color=c, linewidth=3, solid_capstyle="round")
        ax.plot(o["median"], yi, "o", color=c, markeredgecolor=SURFACE, **MARK)
        ax2.barh(yi, o["objects"], color=c, height=0.6)
        ax2.text(o["objects"] * 1.15, yi, f"{o['objects']:,}", va="center", fontsize=8, color=INK2)
    ax.set_xscale("log")
    ax.set_yticks(y, [label(n) for n in names])
    ax.grid(axis="y", visible=False)
    ax.set_xlabel("vertices per stored object  (dot: median; thick: min–p99; thin: to max)")
    ax.set_title("Object sizes")
    ax2.set_xscale("log")
    ax2.set_xlabel("stored objects")
    ax2.set_title("Object count")
    ax2.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_scores(res: dict, out: Path) -> None:
    sc = scores(res)
    lats = sorted({k[0] for k in sc})
    names = layouts(res)
    bw = res["setup"].get("bandwidth")
    fig, axes = plt.subplots(1, len(lats), figsize=(5.0 * len(lats), 0.42 * len(names) + 2.0), sharey=True, squeeze=False)
    y = np.arange(len(names))[::-1]
    for ax, lat in zip(axes[0], lats):
        for reader, filled in (("direct", True), ("zarr", False)):
            for yi, n in zip(y, names):
                v = sc.get((lat, reader, n))
                if v is None:
                    continue
                c = UNI if n.startswith("u") else OCT
                ax.plot(v * 1e3, yi, "o" if filled else "s", color=c, markerfacecolor=c if filled else SURFACE, **MARK)
            have = [(sc[lat, reader, n], n) for n in names if (lat, reader, n) in sc]
            if not have:
                continue
            best = min(have)
            ax.axvline(best[0] * 1e3, color=MUTED, linewidth=0.8, linestyle=":" if not filled else "-")
        ax.set_xscale("log")
        ax.set_yticks(y, [label(n) for n in names])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("geometric mean of median query time (ms)")
        ax.set_title(lat_name(lat, bw))
    handles = [
        Line2D([], [], color=UNI, marker="o", linestyle="none", label="uniform grid", **MARK),
        Line2D([], [], color=OCT, marker="o", linestyle="none", label="octree runs", **MARK),
        Line2D([], [], color=INK2, marker="o", linestyle="none", label="read directly (filled)", **MARK),
        Line2D([], [], color=INK2, marker="s", markerfacecolor=SURFACE, linestyle="none", label="read through zarr (hollow)", **MARK),
    ]
    fig.legend(handles=handles, loc="upper center", ncol=4, fontsize=8.5)
    fig.suptitle("All queries: random vertices, objects, boxes — lower is better; vertical line = best layout for that reader",
                 y=0.93, fontsize=9, color=INK2)
    fig.tight_layout(rect=(0, 0, 1, 0.9))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_families(res: dict, out: Path, reader: str) -> None:
    """Per query family: best uniform grid, best octree runs and the finest grid, for one reader."""
    med = medians(res["queries"])
    sc = scores(res)
    lats = sorted({k[0] for k in sc if k[1] == reader})
    bw = res["setup"].get("bandwidth")
    names = layouts(res)
    fams = [f for f in FAMILY_ORDER if any(k[3] == f for k in med)]
    fig, axes = plt.subplots(len(lats), len(fams), figsize=(3.3 * len(fams), 3.1 * len(lats)), squeeze=False)
    for i, lat in enumerate(lats):
        pick = lambda pre: min((sc[lat, reader, n], n) for n in names if n.startswith(pre) and (lat, reader, n) in sc)[1]
        shown = [(pick("u"), UNI, "-", "best uniform"), (pick("o"), OCT, "-", "best octree"), ("u128", UNI, ":", "finest uniform")]
        for j, fam in enumerate(fams):
            ax = axes[i, j]
            sizes = sorted({k[4] for k in med if k[3] == fam and k[0] == lat})
            for n, c, ls, _ in shown:
                ys = [med.get((lat, reader, n, fam, s), np.nan) * 1e3 for s in sizes]
                ax.plot(range(len(sizes)), ys, color=c, linestyle=ls, marker="o", markeredgecolor=SURFACE, label=label(n), **MARK)
            ax.set_xticks(range(len(sizes)), [f"{s:g}" for s in sizes])
            ax.set_yscale("log")
            ax.set_title(f"{fam} — {lat_name(lat, bw)}" if j == 0 else fam, fontsize=9)
            ax.set_xlabel("vertices" if fam in ("random", "compact", "cluster", "curve") else "box edge (units)")
            if j == 0:
                ax.set_ylabel("median ms")
            if j == len(fams) - 1:
                ax.legend(fontsize=7, loc="upper left")
    fig.suptitle(f"Read {'directly' if reader == 'direct' else 'through zarr'}", y=1.0, fontsize=10)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "bench_bounds.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figures")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fig_objects(res, args.out / "bounds_objects.png")
    fig_scores(res, args.out / "bounds_scores.png")
    fig_families(res, args.out / "bounds_families_direct.png", "direct")
    fig_families(res, args.out / "bounds_families_zarr.png", "zarr")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

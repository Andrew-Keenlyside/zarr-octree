"""Figures for ``bench_index.py``: spatial and vertex-index query times, octree against regular chunks.

    python bench/figures_index.py [--results bench/results/bench_index.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from figures import COLOR, HERE, INK2, LABEL, MARKER, MUTED, ORDER, SURFACE, box_axis

OBJECTS = [("compact", 1000), ("compact", 10000), ("cluster", 1000), ("cluster", 10000), ("curve", 1000), ("curve", 10000)]
MARK = dict(markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)


def med(qs: list, key: str = "seconds") -> float:
    """Median over the queries that returned points -- the same queries for every layout.

    (An empty box can still cost a request in the coarse layout, so filtering on
    requests would compare different boxes across layouts.)
    """
    qs = [q for q in qs if q["points"] > 0]
    return float(np.median([q[key] for q in qs])) if qs else np.nan


def lat_name(lat: float, bw: float | None) -> str:
    if not lat:
        return "local disk (no added latency)"
    return f"{lat * 1e3:.0f} ms + {bw / 1e6:.0f} MB/s per request" if bw else f"{lat * 1e3:.0f} ms per request"


def k_axis(ax, ks: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(ks, [f"{k:,}" for k in ks])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("vertices requested (random ids)")


def fig_times(res: dict, out: Path) -> None:
    v, s = res["vertex"], res["spatial"]
    lats = res["setup"]["latencies"]
    bw = res["setup"].get("bandwidth")
    sel = lambda qs, **kw: [x for x in qs if all(x[k] == val for k, val in kw.items())]
    fig, axes = plt.subplots(len(lats), 4, figsize=(17, 3.9 * len(lats)), squeeze=False)
    for i, lat in enumerate(lats):
        name = lat_name(lat, bw)
        ax = axes[i, 0]
        ks = sorted({x["size"] for x in sel(v, kind="random", latency=lat)})
        for layout in ORDER:
            ax.plot(ks, [med(sel(v, kind="random", size=k, layout=layout, latency=lat)) * 1e3 for k in ks],
                    color=COLOR[layout], marker=MARKER[layout], label=LABEL[layout], **MARK)
        k_axis(ax, ks)
        ax.set_title(f"Random vertices — {name}")

        ax = axes[i, 1]
        x = np.arange(len(OBJECTS))
        for j, layout in enumerate(ORDER):
            ys = [med(sel(v, kind=kind, size=k, layout=layout, latency=lat)) * 1e3 for kind, k in OBJECTS]
            ax.plot(x + (j - 1.5) * 0.12, ys, linestyle="none", color=COLOR[layout], marker=MARKER[layout], **MARK)
        ax.set_xticks(x, [f"{kind}\n{k:,}" for kind, k in OBJECTS])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.grid(axis="x", visible=False)
        ax.set_xlabel("object kind and vertex count (read via its manifest)")
        ax.set_title(f"Whole objects — {name}")

        for col, placement in ((2, "dense"), (3, "uniform")):
            ax = axes[i, col]
            sizes = sorted({x["size"] for x in sel(s, placement=placement, latency=lat)})
            for layout in ORDER:
                ax.plot(sizes, [med(sel(s, placement=placement, size=b, layout=layout, latency=lat)) * 1e3 for b in sizes],
                        color=COLOR[layout], marker=MARKER[layout], **MARK)
            box_axis(ax, sizes)
            where = "centred on a vertex (dense)" if placement == "dense" else "placed uniformly (mostly sparse)"
            ax.set_title(f"Boxes {where}")
        for ax in axes[i]:
            ax.set_yscale("log")
        axes[i, 0].set_ylabel("median query time (ms)")
    axes[0, 0].legend(loc="upper left", fontsize=8)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_ratio(res: dict, out: Path) -> None:
    """Octree time ÷ each regular layout's time, for every query family at every latency."""
    v, s = res["vertex"], res["spatial"]
    lats = res["setup"]["latencies"]
    bw = res["setup"].get("bandwidth")
    sel = lambda qs, **kw: [x for x in qs if all(x[k] == val for k, val in kw.items())]
    rows: list[tuple[str, list, dict]] = []
    for k in sorted({x["size"] for x in sel(v, kind="random")}):
        rows.append((f"random vertices, {k:,}", v, dict(kind="random", size=k)))
    for kind, k in OBJECTS:
        rows.append((f"object: {kind}, {k:,}", v, dict(kind=kind, size=k)))
    for placement in ("dense", "uniform"):
        for b in sorted({x["size"] for x in sel(s, placement=placement)}):
            rows.append((f"box {b}³, {placement}", s, dict(placement=placement, size=b)))
    others = [l for l in ORDER if l != "octree"]
    fig, axes = plt.subplots(1, len(lats), figsize=(5.2 * len(lats), 0.32 * len(rows) + 1.6), sharey=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, lat in zip(axes[0], lats):
        ax.axvline(1, color=MUTED, linewidth=1)
        for j, other in enumerate(others):
            r = []
            for _, qs, kw in rows:
                a = med(sel(qs, layout="octree", latency=lat, **kw))
                b = med(sel(qs, layout=other, latency=lat, **kw))
                r.append(a / b if b and b > 0 else np.nan)
            ax.plot(r, y + (1 - j) * 0.22, linestyle="none", color=COLOR[other], marker=MARKER[other],
                    label=f"vs {LABEL[other]}", **MARK)
        ax.set_xscale("log")
        ax.set_xlim(1 / 40, 40)
        ax.set_xticks([1 / 30, 1 / 10, 1 / 3, 1, 3, 10, 30], ["1/30", "1/10", "1/3", "1", "3", "10", "30"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y, [r[0] for r in rows])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("octree time ÷ other layout's time  (< 1: octree faster)")
        ax.set_title(lat_name(lat, bw))
        ax.text(0.02, -0.07 - 0.25 / len(rows), "← octree faster", transform=ax.transAxes, color=INK2, fontsize=8, va="top")
        ax.text(0.98, -0.07 - 0.25 / len(rows), "octree slower →", transform=ax.transAxes, color=INK2, fontsize=8, va="top", ha="right")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, fontsize=8.5)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.5 / (0.32 * len(rows) + 1.6)))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "bench_index.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figures")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fig_times(res, args.out / "index_queries.png")
    fig_ratio(res, args.out / "index_ratio.png")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

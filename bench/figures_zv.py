"""Figures for ``bench_zv.py``: prototype B (``zarr_octree_sharding``) against ``sharding_indexed`` in zarr-vectors.

    python bench/figures_zv.py [--results bench/results/bench_zv.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np

from figures import HERE, INK2, MUTED, SURFACE

COLOR = {"indexed": "#eb6834", "octree": "#2a78d6", "octree-zarr": "#86b6ef"}
MARKER = {"indexed": "s", "octree": "o", "octree-zarr": "D"}
LABEL = {
    "indexed": "sharding_indexed (zarr-vectors today)",
    "octree": "zarr_octree_sharding",
    "octree-zarr": "zarr_octree_sharding, pooled reader off",
}
KINDS = [("box-dense", "Boxes centred on a vertex"), ("box-uniform", "Boxes placed uniformly"), ("objects", "Objects (~1,000 vertices each)")]
MARK = dict(markersize=7, markeredgecolor=SURFACE, markeredgewidth=1.2)


def split(layout: str) -> tuple[str, int]:
    kind, shard = layout.rsplit("-", 1)
    return kind, int(shard)


def lat_name(lat: float, bw: float) -> str:
    return "local store (zarr-vectors reads shard files)" if not lat else f"{lat * 1e3:.0f} ms + {bw / 1e6:.0f} MB/s per request"


def medians(q: list) -> dict:
    groups: dict = {}
    for x in q:
        groups.setdefault((x["latency"], x["layout"], x["kind"], x["size"]), []).append(x["seconds"])
    return {k: float(np.median(v)) for k, v in groups.items()}


def fig_storage(res: dict, out: Path) -> None:
    b = res["build"]
    names = list(b)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.6))
    y = np.arange(len(names))[::-1]
    for ax, key, scale, title in (
        (axes[0], "disk_bytes", 1e6, "Whole store (MB)"),
        (axes[1], "vertex_bytes", 1e6, "Vertex array shards (MB)"),
        (axes[2], "write_seconds", 1, "Write: create + add_points (s)"),
    ):
        for yi, n in zip(y, names):
            kind, _ = split(n)
            v = b[n][key] / scale
            ax.barh(yi, v, color=COLOR[kind], height=0.6)
            ax.text(v * 1.01, yi, f"{v:,.1f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [f"{LABEL[split(n)[0]]}, {split(n)[1]}³-cell shards" for n in names] if ax is axes[0] else [])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(b[n][key] / scale for n in names) * 1.18)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_queries(res: dict, out: Path, shard: int) -> None:
    med = medians(res["queries"])
    lats = sorted({k[0] for k in med})
    bw = res["setup"]["bandwidth"]
    fig, axes = plt.subplots(len(lats), len(KINDS), figsize=(14, 3.5 * len(lats)), squeeze=False)
    for i, lat in enumerate(lats):
        for j, (kind, title) in enumerate(KINDS):
            ax = axes[i, j]
            sizes = sorted({k[3] for k in med if k[2] == kind})
            for variant in ("indexed", "octree", "octree-zarr"):
                layout = f"{variant}-{shard}"
                ys = [med.get((lat, layout, kind, s), np.nan) * 1e3 for s in sizes]
                ax.plot(range(len(sizes)), ys, color=COLOR[variant], marker=MARKER[variant], label=LABEL[variant], **MARK)
            ax.set_xticks(range(len(sizes)), [f"{s:g}" for s in sizes])
            ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
            ax.set_xlabel("objects per query" if kind == "objects" else "box edge (units of 1000)")
            ax.set_yscale("log")
            ax.set_title(f"{title} — {lat_name(lat, bw)}" if j == 0 else title, fontsize=9)
            if j == 0:
                ax.set_ylabel("median ms")
    axes[0, 0].legend(fontsize=8, loc="upper left")
    fig.suptitle(f"Queries through zarr-vectors, {shard}³-cell shards", y=1.0, fontsize=10, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)


def fig_ratio(res: dict, out: Path) -> None:
    med = medians(res["queries"])
    lats = sorted({k[0] for k in med})
    bw = res["setup"]["bandwidth"]
    rows = [(kind, s) for kind, _ in KINDS for s in sorted({k[3] for k in med if k[2] == kind})]
    fig, axes = plt.subplots(1, len(lats), figsize=(4.8 * len(lats), 0.36 * len(rows) + 1.8), sharey=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    for ax, lat in zip(axes[0], lats):
        ax.axvline(1, color=MUTED, linewidth=1)
        for k, (shard, variant) in enumerate([(16, "octree"), (32, "octree"), (16, "octree-zarr"), (32, "octree-zarr")]):
            r = [med.get((lat, f"{variant}-{shard}", kind, s), np.nan) / med.get((lat, f"indexed-{shard}", kind, s), np.nan) for kind, s in rows]
            ax.plot(r, y + (1.5 - k) * 0.17, linestyle="none", color=COLOR[variant],
                    marker="o" if shard == 16 else "^", label=f"{LABEL[variant]}, {shard}³ shards", **MARK)
        ax.set_xscale("log")
        ax.set_xlim(1 / 3, 3)
        ax.set_xticks([1 / 3, 1 / 2, 1, 2, 3], ["1/3", "1/2", "1", "2", "3"])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y, [f"{kind} {s:g}" for kind, s in rows])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("time ÷ sharding_indexed  (< 1: faster)")
        ax.set_title(lat_name(lat, bw), fontsize=9)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.9 / (0.36 * len(rows) + 1.8)))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "bench_zv.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figures")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    fig_storage(res, args.out / "zv_storage.png")
    fig_queries(res, args.out / "zv_queries_16.png", 16)
    fig_ratio(res, args.out / "zv_ratio.png")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

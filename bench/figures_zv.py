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

from figures import HERE, INK2, MUTED

# Fixed order, validated (dataviz validate_palette.js, light): colour follows the reader everywhere;
# the marker (square: sharding_indexed, circle: octree) is the second cue.
READERS = ["indexed", "indexed-batched", "indexed-gap", "octree-zarr", "octree-cells", "octree-runs"]
COLOR = dict(zip(READERS, ["#eb6834", "#6250d6", "#eda100", "#1baf7a", "#e87ba4", "#2a78d6"]))
MARKER = {r: "s" if r.startswith("indexed") else "o" for r in READERS}
LABEL = {
    "indexed": "sharding_indexed, read as today",
    "indexed-batched": "sharding_indexed, pooled: one request per cell",
    "indexed-gap": "sharding_indexed, pooled: gaps ≤ 64 KiB merged",
    "octree-zarr": "octree, read as today (through zarr)",
    "octree-cells": "octree, pooled: one request per cell",
    "octree-runs": "octree, pooled: one request per run",
}
STORE_COLOR = {"indexed": COLOR["indexed"], "octree": COLOR["octree-runs"]}
STORE_LABEL = {"indexed": "sharding_indexed", "octree": "zarr_octree_sharding"}
KINDS = [("box-dense", "Boxes centred on a vertex"), ("box-uniform", "Boxes placed uniformly"), ("objects", "Objects (~1,000 vertices each)")]
MARK = dict(markersize=7, markeredgecolor="#fcfcfb", markeredgewidth=1.2)


def split(layout: str) -> tuple[str, int]:
    kind, shard = layout.rsplit("-", 1)
    return kind, int(shard)


def lat_name(lat: float, bw: float) -> str:
    return "local store" if not lat else f"{lat * 1e3:.0f} ms + {bw / 1e6:.0f} MB/s per request"


def medians(q: list, field: str = "seconds") -> dict:
    groups: dict = {}
    for x in q:
        groups.setdefault((x["latency"], x["layout"], x["kind"], x["size"]), []).append(x[field])
    return {k: float(np.median(v)) for k, v in groups.items()}


def geomean(med: dict, lat: float, layout: str) -> float:
    v = [x for (l, lay, *_), x in med.items() if l == lat and lay == layout]
    return float(np.exp(np.mean(np.log(np.maximum(v, 1e-9))))) if v else np.nan


def readers_in(res: dict, lat: float) -> list[str]:
    have = {split(q["layout"])[0] for q in res["queries"] if q["latency"] == lat}
    return [r for r in READERS if r in have]


def fig_storage(res: dict, out: Path) -> None:
    b = res["build"]
    names = list(b)
    fig, axes = plt.subplots(1, 3, figsize=(14, 3.0))
    y = np.arange(len(names))[::-1]
    for ax, key, scale, title in (
        (axes[0], "disk_bytes", 1e6, "Whole store (MB)"),
        (axes[1], "vertex_bytes", 1e6, "Vertex array shards (MB)"),
        (axes[2], "write_seconds", 1, "Write: create + add_points (s)"),
    ):
        for yi, n in zip(y, names):
            kind, _ = split(n)
            v = b[n][key] / scale
            ax.barh(yi, v, color=STORE_COLOR[kind], height=0.6)
            ax.text(v * 1.01, yi, f"{v:,.1f}", va="center", fontsize=8, color=INK2)
        ax.set_yticks(y, [f"{STORE_LABEL[split(n)[0]]}, {split(n)[1]}³-cell shards" for n in names] if ax is axes[0] else [])
        ax.set_title(title)
        ax.grid(axis="y", visible=False)
        ax.set_xlim(0, max(b[n][key] / scale for n in names) * 1.18)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_summary(res: dict, out: Path) -> None:
    """Geometric-mean query time, and requests and bytes per query, by reader."""
    med = medians(res["queries"])
    req = medians(res["queries"], "requests")
    mb = medians(res["queries"], "bytes_read")
    lats = sorted({k[0] for k in med})
    bw = res["setup"]["bandwidth"]
    remote = [l for l in lats if l]
    def short(l: float) -> str:
        return "local" if not l else f"{l * 1e3:.0f} ms latency"

    panels = [(f"Query time (ms), {short(l)}", med, l, 1e3) for l in lats]
    if remote:
        panels += [(f"Requests, {short(remote[0])}", req, remote[0], 1),
                   (f"MB read, {short(remote[0])}", mb, remote[0], 1e-6)]
    fig, axes = plt.subplots(2, len(panels), figsize=(3.3 * len(panels) + 2.6, 6.2), squeeze=False)
    for row, shard in enumerate((16, 32)):
        for col, (title, m, lat, scale) in enumerate(panels):
            ax = axes[row, col]
            rs = [r for r in READERS if any(r in readers_in(res, l) for l in lats)]  # the same rows in every panel
            y = np.arange(len(rs))[::-1]
            vals = [geomean(m, lat, f"{r}-{shard}") * scale for r in rs]
            ref = vals[0]
            for yi, r, v in zip(y, rs, vals):
                if np.isnan(v):
                    ax.text(0, yi, "  same reads as above, locally", va="center", fontsize=7.5, color=MUTED)
                    continue
                ax.barh(yi, v, color=COLOR[r], height=0.62)
                ax.plot([v], [yi], marker=MARKER[r], color=COLOR[r], linestyle="none", **MARK)
                tag = f"{v:,.0f}" if v >= 10 else f"{v:,.1f}"
                if r != rs[0] and ref:
                    tag += f"  ({v / ref:.2f}×)"
                ax.text(v * 1.04, yi, tag, va="center", fontsize=7.5, color=INK2)
            ax.set_yticks(y, [LABEL[r] for r in rs] if col == 0 else [""] * len(rs), fontsize=8)
            ax.set_xlim(0, np.nanmax(vals) * 1.55)
            ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(4))
            ax.grid(axis="y", visible=False)
            ax.set_title(f"{title}\n{shard}³-cell shards" if row == 0 else f"{shard}³-cell shards", fontsize=8.5)
    fig.suptitle(f"Geometric mean over all queries ({lat_name(remote[0], bw) if remote else 'local store'} for requests and bytes)",
                 x=0.01, ha="left", fontsize=9, color=INK2)
    fig.tight_layout()
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_ratio(res: dict, out: Path, shard: int) -> None:
    """Each query's time over ``sharding_indexed`` read as today."""
    med = medians(res["queries"])
    lats = sorted({k[0] for k in med})
    bw = res["setup"]["bandwidth"]
    rows = [(kind, s) for kind, _ in KINDS for s in sorted({k[3] for k in med if k[2] == kind})]
    h = 0.42 * len(rows) + 1.7
    fig, axes = plt.subplots(1, len(lats), figsize=(4.8 * len(lats), h), sharey=True, squeeze=False)
    y = np.arange(len(rows))[::-1]
    ratios = [med[k] / med[(k[0], f"indexed-{shard}", k[2], k[3])] for k in med
              if k[1].endswith(f"-{shard}") and (k[0], f"indexed-{shard}", k[2], k[3]) in med]
    lo = min([1 / 3] + [r for r in ratios if r > 0]) / 1.3
    hi = max([3] + ratios) * 1.3
    ticks = [t for t in (1 / 100, 1 / 30, 1 / 10, 1 / 3, 1, 3, 10) if lo <= t <= hi]
    for ax, lat in zip(axes[0], lats):
        ax.axvline(1, color=MUTED, linewidth=1)
        rs = [r for r in readers_in(res, lat) if r != "indexed"]
        for k, r in enumerate(rs):
            v = [med.get((lat, f"{r}-{shard}", kind, s), np.nan) / med.get((lat, f"indexed-{shard}", kind, s), np.nan) for kind, s in rows]
            ax.plot(v, y + ((len(rs) - 1) / 2 - k) * 0.14, linestyle="none", color=COLOR[r], marker=MARKER[r], label=LABEL[r], **MARK)
        ax.set_xscale("log")
        ax.set_xlim(lo, hi)
        ax.set_xticks(ticks, [f"1/{1 / t:.0f}" if t < 1 else f"{t:g}" for t in ticks])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_yticks(y, [f"{kind} {s:g}" for kind, s in rows])
        ax.grid(axis="y", visible=False)
        ax.set_xlabel("time ÷ sharding_indexed read as today  (< 1: faster)")
        ax.set_title(f"{shard}³-cell shards — {lat_name(lat, bw)}", fontsize=9)
    handles, labels = {}, {}
    for ax in axes[0]:
        for hd, lb in zip(*ax.get_legend_handles_labels()):
            handles.setdefault(lb, hd)
    order = [LABEL[r] for r in READERS if LABEL[r] in handles]
    fig.legend([handles[l] for l in order], order, loc="upper center", ncol=3, fontsize=8)
    fig.tight_layout(rect=(0, 0, 1, 1 - 0.6 / h))
    fig.savefig(out, dpi=160)
    plt.close(fig)


def fig_zarr_api(res: dict, out: Path, shard: int = 16) -> None:
    """The README's results figure: the octree codec against constant chunks (``sharding_indexed``), both read
    through zarr's API; a custom reader (shard index read once, neighbouring ranges merged) only as a marker."""
    med, mb = medians(res["queries"]), medians(res["queries"], "bytes_read")
    lats = sorted({k[0] for k in med})
    rows = [  # (store, reader through zarr, custom reader, label, colour): colours as in concept.png
        ("indexed", "indexed", "indexed-gap", "constant chunks\n(sharding_indexed)", "#1baf7a"),
        ("octree", "octree-zarr", "octree-runs", "octree codec\n(zarr_octree_sharding)", "#6250d6"),
    ]
    panels = [(f"Query time (ms), {'local' if not l else f'{l * 1e3:.0f} ms latency'}", med, l, 1e3, True) for l in lats]
    panels += [("MB read per query (20 ms)", mb, 0.02, 1e-6, False), ("Store size (MB)", None, None, 1e-6, False)]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.55 * len(panels) + 1.4, 2.3), sharey=True)
    y = np.arange(len(rows))[::-1]
    for ax, (title, m, lat, scale, custom) in zip(axes, panels):
        vals = [res["build"][f"{store}-{shard}"]["disk_bytes"] * scale if m is None else geomean(m, lat, f"{zr}-{shard}") * scale
                for store, zr, *_ in rows]
        for yi, v, (store, _, cr, _, colour) in zip(y, vals, rows):
            ax.barh(yi, v, color=colour, height=0.58)
            ax.text(v * 1.03, yi, f"{v:,.0f}" if v >= 10 else f"{v:,.1f}", va="center", fontsize=8, color=INK2)
            if custom:
                c = geomean(m, lat, f"{cr}-{shard}") * scale
                ax.plot([c], [yi - 0.42], marker="v", color=INK2, markersize=5, linestyle="none")
        ax.set_xlim(0, max(vals) * 1.3)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=8.5)
    axes[0].set_yticks(y, [label for *_, label, _ in rows], fontsize=8.5)
    axes[0].set_ylim(-0.75, len(rows) - 0.4)
    axes[0].plot([], [], marker="v", color=INK2, markersize=5, linestyle="none",
                 label="same store, custom reader (each shard index read once, neighbouring ranges merged)")
    fig.legend(loc="lower left", bbox_to_anchor=(0.0, -0.08), fontsize=7.5, frameon=False)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


STORE_ROWS = [  # (store, label) for bench_zv_readers.json
    ("indexed-16", "constant chunks\n(sharding_indexed)"),
    ("octree10k-16", "octree, 10 kB target"),
    ("octree320k-16", "octree, 320 kB target"),
    ("runs-16", "runs, 160–640 kB"),
]
READER_COLOR = {"zarr": "#2a78d6", "optimised": "#eb6834"}
READER_LABEL = {"zarr": "through zarr's API only", "optimised": "zarr-vectors' optimised reader"}


def fig_readers(res: dict, out: Path) -> None:
    """Each store (sharding codec and layout) read both ways: query time at each latency, MB read at 20 ms, store size."""
    med, mb = medians(res["queries"]), medians(res["queries"], "bytes_read")
    lats = sorted({k[0] for k in med})
    panels = [(f"Query time (ms), {'local' if not l else f'{l * 1e3:.0f} ms latency'}", med, l, 1e3) for l in lats]
    panels += [("MB read per query (20 ms)", mb, 0.02, 1e-6), ("Store size (MB)", None, None, 1e-6)]
    fig, axes = plt.subplots(1, len(panels), figsize=(2.45 * len(panels) + 1.6, 3.4), sharey=True)
    y = np.arange(len(STORE_ROWS))[::-1].astype(float)
    h = 0.36
    for ax, (title, m, lat, scale) in zip(axes, panels):
        top = 0.0
        for yi, (store, _) in zip(y, STORE_ROWS):
            if m is None:
                v = res["build"][store]["disk_bytes"] * scale
                ax.barh(yi, v, color=MUTED, height=2 * h * 0.9)
                ax.text(v * 1.03, yi, f"{v:,.0f}" if v >= 10 else f"{v:,.1f}", va="center", fontsize=7.5, color=INK2)
                top = max(top, v)
                continue
            for k, mode in enumerate(READER_COLOR):
                v = geomean(m, lat, f"{store}/{mode}") * scale
                yy = yi + (h / 2 if k == 0 else -h / 2)
                ax.barh(yy, v, color=READER_COLOR[mode], height=h * 0.9)
                ax.text(v * 1.03, yy, f"{v:,.0f}" if v >= 10 else f"{v:,.1f}", va="center", fontsize=7, color=INK2)
                top = max(top, v)
        ax.set_xlim(0, top * 1.35)
        ax.xaxis.set_major_locator(matplotlib.ticker.MaxNLocator(3))
        ax.grid(axis="y", visible=False)
        ax.set_title(title, fontsize=8.5)
    axes[0].set_yticks(y, [label for _, label in STORE_ROWS], fontsize=8.5)
    axes[0].set_ylim(-0.6, len(STORE_ROWS) - 0.4)
    from matplotlib.patches import Patch

    fig.legend(handles=[Patch(color=c, label=READER_LABEL[m]) for m, c in READER_COLOR.items()],
               loc="lower left", bbox_to_anchor=(0.0, -0.04), ncol=2, fontsize=8, frameon=False)
    fig.tight_layout(w_pad=0.8, rect=(0, 0.06, 1, 1))
    fig.savefig(out, dpi=160, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "bench_zv.json")
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figures")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    args.out.mkdir(parents=True, exist_ok=True)
    if any("/" in q["layout"] for q in res["queries"]):  # bench_zv_readers.json: store/reader
        fig_readers(res, args.out / "zv_readers.png")
        print(f"-> {args.out}")
        return
    fig_storage(res, args.out / "zv_storage.png")
    fig_zarr_api(res, args.out / "zv_zarr_api.png")
    fig_summary(res, args.out / "zv_summary.png")
    for shard in (16, 32):
        fig_ratio(res, args.out / f"zv_ratio_{shard}.png", shard)
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

"""Figures for the octree benchmark: the point cloud and its octree, object sizes, queries.

    python bench/figures.py [--results bench/results/bench.json]
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker
import numpy as np
from matplotlib.colors import ListedColormap
from matplotlib.patches import Rectangle
from mpl_toolkits.mplot3d.art3d import Line3DCollection
from zarr.storage import LocalStore

from zarr_octree import OctreeStore

HERE = Path(__file__).parent

# Reference palette (dataviz skill), light mode.  Colour follows the layout everywhere.
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK2 = "#52514e"
MUTED = "#8a8985"
GRID_C = "#e4e3df"
COLOR = {"octree": "#2a78d6", "sharded": "#eb6834", "leaf": "#1baf7a", "coarse": "#eda100"}
MARKER = {"octree": "o", "sharded": "s", "leaf": "^", "coarse": "D"}
LABEL = {
    "octree": "octree (this extension)",
    "sharded": "leaf chunks + sharding_indexed",
    "leaf": "regular leaf chunks",
    "coarse": "regular coarse chunks",
}
ORDER = ["octree", "sharded", "leaf", "coarse"]
BLUE = ["#cde2fb", "#b7d3f6", "#9ec5f4", "#86b6ef", "#6da7ec", "#5598e7", "#3987e5", "#2a78d6", "#256abf", "#1c5cab", "#184f95", "#104281", "#0d366b"]
SEQ = ListedColormap(BLUE[3:], name="blue_seq")

plt.rcParams.update({
    "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
    "axes.edgecolor": MUTED, "axes.labelcolor": INK2, "xtick.color": INK2, "ytick.color": INK2,
    "text.color": INK, "axes.grid": True, "grid.color": GRID_C, "grid.linewidth": 0.6,
    "axes.spines.top": False, "axes.spines.right": False, "font.size": 9, "axes.titlesize": 10,
    "axes.titleweight": "semibold", "axes.titlelocation": "left", "lines.linewidth": 2,
    "legend.frameon": False, "axes.axisbelow": True,
})


def depth_colors(max_depth: int) -> list[str]:
    """An ordinal ramp, shallow (light) to deep (dark), every step >= 2:1 on the surface."""
    idx = np.linspace(3, 12, max_depth + 1).round().astype(int)
    return [BLUE[i] for i in idx]


def fig_points(pts: np.ndarray, setup: dict, stores: Path, out: Path) -> None:
    grid, leaf, D = setup["grid"], setup["leaf"], setup["max_depth"]
    nodes = OctreeStore(LocalStore(str(stores / "octree"), read_only=True)).nodes("pts")
    cell = np.clip(np.floor(pts / leaf).astype(int), 0, grid - 1)
    lin = (cell[:, 0] * grid + cell[:, 1]) * grid + cell[:, 2]
    counts = np.bincount(lin, minlength=grid**3)
    dens = counts[lin]
    rng = np.random.default_rng(0)
    # Sample uniformly over *cells*-ish: keep sparse regions visible next to the dense ones.
    keep = rng.random(len(pts)) < np.minimum(1.0, 6.0 / np.sqrt(dens)) * 0.5
    sub, sd = pts[keep], np.log10(dens[keep])
    order = np.argsort(sd)
    sub, sd = sub[order], sd[order]
    dc = depth_colors(D)

    fig = plt.figure(figsize=(15, 5.4))
    ax = fig.add_subplot(1, 3, 1, projection="3d")
    sc = ax.scatter(*sub.T, c=sd, cmap=SEQ, s=0.15, alpha=0.6, linewidths=0, rasterized=True)
    style_3d(ax)
    ax.set_title(f"densMAP point cloud, {setup['points'] / 1e6:.1f} M points", pad=-4)
    cb = fig.colorbar(sc, ax=ax, shrink=0.55, pad=0.02)
    cb.set_label("points in its leaf cell (log10)")
    cb.outline.set_visible(False)
    cb.solids.set(alpha=1)

    ax = fig.add_subplot(1, 3, 2, projection="3d")
    ax.scatter(*sub[::4].T, c=MUTED, s=0.05, alpha=0.25, linewidths=0, rasterized=True)
    for d in range(D + 1):
        segs = []
        for (nd, c), _ in nodes.items():
            if nd != d:
                continue
            size = leaf * 2 ** (D - d)
            lo = np.array(c) * size
            segs.extend(box_edges(lo, size))
        if segs:
            ax.add_collection3d(Line3DCollection(segs, colors=dc[d], linewidths=0.35 + 0.15 * d, alpha=0.9))
    style_3d(ax)
    ax.set_title(f"its octree: {len(nodes)} stored nodes", pad=-4)
    for d in range(D + 1):
        n = sum(1 for k in nodes if k[0] == d)
        ax.plot([], [], color=dc[d], label=f"depth {d}: {n} nodes, {2 ** (D - d)}³ leaves")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, 0.06), ncol=2, fontsize=7.5, handlelength=1.2)

    # A plane through the densest region: the nodes it cuts, drawn as squares.
    z0 = float(np.median(pts[dens > np.percentile(dens, 99), 2]))
    slab = np.abs(pts[:, 2] - z0) < leaf
    sp = pts[slab][:: max(1, slab.sum() // 150_000)]
    ax = fig.add_subplot(1, 3, 3)
    ax.scatter(sp[:, 0], sp[:, 1], s=0.15, c=INK2, alpha=0.4, linewidths=0, rasterized=True)
    for (d, c), _ in nodes.items():
        size = leaf * 2 ** (D - d)
        lo = np.array(c) * size
        if not lo[2] <= z0 < lo[2] + size:
            continue
        ax.add_patch(Rectangle(lo[:2], size, size, fill=False, edgecolor=dc[d], linewidth=0.5 + 0.25 * d))
    ax.set_title(f"plane z = {z0:.0f}: nodes it cuts, points within one leaf")
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_aspect("equal")
    ax.grid(False)
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def box_edges(lo: np.ndarray, s: float) -> list:
    c = np.array([[x, y, z] for x in (0, s) for y in (0, s) for z in (0, s)]) + lo
    pairs = [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]
    return [(c[a], c[b]) for a, b in pairs]


def style_3d(ax) -> None:
    ax.set_xlim(0, 1000)
    ax.set_ylim(0, 1000)
    ax.set_zlim(0, 1000)
    ax.set_box_aspect((1, 1, 1))
    ax.view_init(elev=22, azim=-58)
    for a in (ax.xaxis, ax.yaxis, ax.zaxis):
        a.set_pane_color((1, 1, 1, 0))
        a._axinfo["grid"]["color"] = GRID_C
        a.set_ticklabels([])
    ax.tick_params(length=0)


def fig_objects(res: dict, out: Path) -> None:
    fig, axes = plt.subplots(1, 3, figsize=(15, 5.2), gridspec_kw={"width_ratios": [1.6, 1, 1]})
    ax = axes[0]
    stats = res["objects"]
    for layout in ORDER:
        s = np.load(out.parent.parent / f"objects_{layout}.npy")
        x = np.sort(s)
        note = " — same contents as coarse" if layout == "sharded" else ""
        ax.plot(x, np.arange(1, len(x) + 1) / len(x), color=COLOR[layout], label=f"{LABEL[layout]} ({len(x):,} objects){note}",
                linestyle=(0, (4, 3)) if layout == "sharded" else "-", zorder=3 if layout == "sharded" else 2)
    ax.axvline(res["setup"]["target_bytes"] / 12, color=MUTED, linewidth=1, linestyle=(0, (3, 3)))
    ax.text(res["setup"]["target_bytes"] / 12 * 1.1, 0.04, "octree target", color=INK2, fontsize=8)
    ax.set_xscale("log")
    ax.set_xlabel("points per stored object")
    ax.set_ylabel("fraction of objects ≤ x")
    ax.set_title("Size of every stored object")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.2), fontsize=8, ncol=1)

    ax = axes[1]
    y = np.arange(len(ORDER))
    cv = [stats[l]["cv"] for l in ORDER]
    ax.barh(y, cv, color=[COLOR[l] for l in ORDER], height=0.6)
    for i, v in enumerate(cv):
        ax.text(v, i, f"  {v:.2f}", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [LABEL[l] for l in ORDER])
    ax.invert_yaxis()
    ax.set_xlabel("coefficient of variation of points per object")
    ax.set_title("Spread of object size (lower = more even)")
    ax.grid(axis="y", visible=False)
    ax.set_xlim(0, max(cv) * 1.25)

    ax = axes[2]
    eff = [stats[l]["gpu_batch_efficiency"] * 100 for l in ORDER]
    ax.barh(y, eff, color=[COLOR[l] for l in ORDER], height=0.6)
    for i, v in enumerate(eff):
        ax.text(v, i, f"  {v:.0f}%", va="center", color=INK, fontsize=8.5)
    ax.set_yticks(y, [""] * len(ORDER))
    ax.invert_yaxis()
    ax.set_xlim(0, 115)
    ax.set_xlabel("useful work, batches of 64 padded to their largest (%)")
    ax.set_title("GPU batch efficiency")
    ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def med(qs: list, key: str = "seconds") -> float:
    """Median over the queries that touched any data (an empty box costs nothing anywhere)."""
    qs = [q for q in qs if q.get("requests", 1) > 0]
    return float(np.median([q[key] for q in qs])) if qs else np.nan


def box_axis(ax, sizes: list[int]) -> None:
    ax.set_xscale("log")
    ax.set_xticks(sizes, [f"{s}³" for s in sizes])
    ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
    ax.set_xlabel("box edge, in leaf cells (leaf = 7.8 units)")


def common_queries(q: list) -> list:
    """The boxes run at every latency (latency 0 runs more), so panels compare the same boxes."""
    n = min(max(x["query"] for x in q if x["latency"] == lat) + 1 for lat in {x["latency"] for x in q})
    return [x for x in q if x["query"] < n]


def fig_queries(res: dict, out: Path) -> None:
    q = common_queries(res["queries"])
    lats = sorted({x["latency"] for x in q})
    sizes = sorted({x["size"] for x in q})
    leaf = res["setup"]["leaf"]
    sel = lambda **kw: [x for x in q if all(x[k] == v for k, v in kw.items())]
    fig, axes = plt.subplots(2, max(3, len(lats)), figsize=(15, 8.2))
    for j, lat in enumerate(lats):
        ax = axes[0, j]
        for layout in ORDER:
            ys = [med(sel(layout=layout, latency=lat, size=s, placement="dense", mode="occupied")) * 1e3 for s in sizes]
            ax.plot(sizes, ys, color=COLOR[layout], marker=MARKER[layout], markersize=6,
                    markeredgecolor=SURFACE, markeredgewidth=1.2, label=LABEL[layout])
        box_axis(ax, sizes)
        ax.set_yscale("log")
        ax.set_ylabel("median query time (ms)")
        ax.set_title(f"Box queries in dense regions, {lat * 1e3:.0f} ms per request")
        if j == 0:
            ax.legend(loc="upper left", fontsize=8)

    ax = axes[1, 0]
    for layout in ORDER:
        ys = [med(sel(layout=layout, latency=lats[0], size=s, placement="dense", mode="occupied"), "requests") for s in sizes]
        ax.plot(sizes, ys, color=COLOR[layout], marker=MARKER[layout], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    box_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("median requests per query")
    ax.set_title("Requests per query (dense regions)")

    ax = axes[1, 1]
    for layout in ORDER:
        ys = []
        for s in sizes:
            qs = sel(layout=layout, latency=lats[0], size=s, placement="dense", mode="occupied")
            ys.append(np.median([x["bytes_read"] / (12 * x["points"]) for x in qs if x["points"]]) if qs else np.nan)
        ax.plot(sizes, ys, color=COLOR[layout], marker=MARKER[layout], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    box_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("stored bytes read ÷ raw bytes in the box")
    ax.set_title("Read amplification (dense; zstd stores ≈0.89 of raw)")

    ax = axes[1, 2]
    lat = lats[min(1, len(lats) - 1)]
    for layout in ORDER:
        ys = [med(sel(layout=layout, latency=lat, size=s, placement="uniform", mode="occupied")) * 1e3 for s in sizes]
        ax.plot(sizes, ys, color=COLOR[layout], marker=MARKER[layout], markersize=6, markeredgecolor=SURFACE, markeredgewidth=1.2)
    box_axis(ax, sizes)
    ax.set_yscale("log")
    ax.set_ylabel("median query time (ms)")
    ax.set_title(f"Boxes placed uniformly (mostly sparse), {lat * 1e3:.0f} ms")
    for ax in axes.flat[len(lats) : 3]:
        ax.set_visible(False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_slice_cold(res: dict, out: Path) -> None:
    q = common_queries(res["queries"])
    lats = sorted({x["latency"] for x in q})
    lat = lats[min(1, len(lats) - 1)]
    sizes = sorted({x["size"] for x in q if x["mode"] == "slice"})
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.2))
    ax = axes[0]
    w = 0.8 / len(ORDER)
    for i, layout in enumerate(ORDER):
        ys = [med([x for x in q if x["layout"] == layout and x["latency"] == lat and x["size"] == s and x["mode"] == "slice" and x["placement"] == "dense"]) * 1e3 for s in sizes]
        ax.bar(np.arange(len(sizes)) + (i - 1.5) * w, ys, width=w * 0.9, color=COLOR[layout], label=LABEL[layout])
    ax.set_yscale("log")
    ax.set_xticks(np.arange(len(sizes)), [f"{s}³ leaves" for s in sizes])
    ax.set_ylabel("median query time (ms)")
    ax.set_title(f"Plain zarr slicing arr[box], {lat * 1e3:.0f} ms per request\n(zarr asks for every chunk in the box, empty or not)")
    ax.legend(fontsize=8, loc="upper left")
    ax.grid(axis="x", visible=False)

    ax = axes[1]
    cold = res["cold"]
    clats = sorted({x["latency"] for x in cold})
    w = 0.8 / len(ORDER)
    for i, layout in enumerate(ORDER):
        ys = [med([x for x in cold if x["layout"] == layout and x["latency"] == l], "open_and_query_seconds") * 1e3 for l in clats]
        ax.bar(np.arange(len(clats)) + (i - 1.5) * w, ys, width=w * 0.9, color=COLOR[layout])
    ax.set_xticks(np.arange(len(clats)), [f"{l * 1e3:.0f} ms per request" for l in clats])
    ax.set_yscale("log")
    ax.set_ylabel("open + first query (ms)")
    ax.set_title("Cold start\n(open the array, then one dense 16³-leaf box)")
    ax.grid(axis="x", visible=False)
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def fig_sweep(res: dict, out: Path) -> None:
    sw = res["sweep"]
    depths = sorted({s["max_depth"] for s in sw})
    picks = np.linspace(4, 12, len(depths)).round().astype(int)  # well-separated ordinal steps
    dc = {d: BLUE[i] for d, i in zip(depths, picks)}
    targets = sorted({s["target_bytes"] // 1024 for s in sw})
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    for d in depths:
        rows = sorted([s for s in sw if s["max_depth"] == d], key=lambda s: s["target_bytes"])
        t = [s["target_bytes"] / 1024 for s in rows]
        c = dc[d]
        axes[0].plot(t, [s["cv"] for s in rows], color=c, marker="o", markersize=5, markeredgecolor=SURFACE, label=f"max depth {d} (root {2**d}³ leaves)")
        axes[1].plot(t, [s["objects"] for s in rows], color=c, marker="o", markersize=5, markeredgecolor=SURFACE)
    for ax in axes:
        ax.set_xscale("log")
        ax.set_xticks(targets, [str(t) for t in targets])
        ax.xaxis.set_minor_locator(matplotlib.ticker.NullLocator())
        ax.set_xlabel("target node size (KiB)")
    axes[0].set_ylabel("coefficient of variation of points per object")
    axes[0].set_title("Evenness of the partition")
    axes[0].legend(fontsize=8)
    axes[1].set_yscale("log")
    axes[1].set_ylabel("stored objects")
    axes[1].set_title("Number of stored objects")
    fig.tight_layout()
    fig.savefig(out, dpi=170)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", type=Path, default=HERE / "results" / "bench.json")
    ap.add_argument("--data", type=Path, default=HERE / "_data" / "densmap_points.npz")
    ap.add_argument("--stores", type=Path, default=HERE / "_stores")
    args = ap.parse_args()
    res = json.loads(args.results.read_text())
    figs = args.results.parent / "figures"
    figs.mkdir(parents=True, exist_ok=True)
    pts = np.clip(np.load(args.data)["points"], 0, 999.999)
    fig_points(pts, res["setup"], args.stores, figs / "points_and_octree.png")
    fig_objects(res, figs / "objects.png")
    fig_queries(res, figs / "queries.png")
    fig_slice_cold(res, figs / "slice_and_cold.png")
    fig_sweep(res, figs / "sweep.png")
    print("figures ->", figs)


if __name__ == "__main__":
    main()

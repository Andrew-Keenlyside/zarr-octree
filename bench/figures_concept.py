"""The README's explanatory figure: constant chunks against an octree, on a synthetic point cloud.

    python bench/figures_concept.py

Three clusters with broad (Student-t) tails, so the density falls smoothly from dense cores to a
sparse background.  Every layout is tuned to the same mean object size (``MEAN`` points), so they
differ only in how evenly the points are spread over objects:

* constant chunks: the number of chunks per axis;
* octree, hard limit: a node is split while it holds more than the limit;
* octree, target: a node is split when its children are closer to the target (in log size) than it
  is, or when it holds more than 4 x the target;
* Z-order runs (what zarr-octree stores): runs of the octree's Z-order between max / 4 and max.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from figures import HERE, INK2, MUTED
from zarr_octree.tree import build_runs, morton

N = 128  # leaf chunks per axis: the octree's finest level
MEAN = 10_000  # the mean object size every layout is tuned to, in points
CAP = 4  # the target scheme's hard cap, as a multiple of its target
COLOR = {"constant": "#1baf7a", "limit": "#2a78d6", "target": "#eb6834", "runs": "#6250d6"}


def cluster_points(seed: int = 0) -> np.ndarray:
    """Three anisotropic clusters with Student-t tails (3 degrees of freedom), in the unit cube."""
    rng = np.random.default_rng(seed)
    parts = []
    for centre, scales, n in [((0.32, 0.38, 0.40), (0.07, 0.05, 0.04), 1_200_000),
                              ((0.70, 0.62, 0.58), (0.05, 0.08, 0.05), 800_000),
                              ((0.62, 0.25, 0.75), (0.03, 0.03, 0.04), 300_000)]:
        rot, _ = np.linalg.qr(rng.normal(size=(3, 3)))
        t = rng.standard_t(3, size=(n, 3)) * scales
        parts.append(centre + t @ rot.T)
    p = np.concatenate(parts)
    return p[((p >= 0) & (p < 1)).all(1)]


def counts(p: np.ndarray, n: int) -> np.ndarray:
    """Points per chunk on an ``n``³ grid, indexed ``[x, y, z]``."""
    ijk = np.floor(p * n).astype(int)
    c = np.zeros((n, n, n), dtype=np.int64)
    np.add.at(c, tuple(ijk.T), 1)
    return c


Node = tuple[tuple[int, int, int], int, int]  # (corner, size, points), in leaf units


def octree(c: np.ndarray, split_if: Callable[[int, np.ndarray], bool]) -> list[Node]:
    """The occupied leaves of an octree over ``c``, splitting a node when ``split_if(points, children's points)``."""
    out: list[Node] = []

    def visit(x: int, y: int, z: int, size: int) -> None:
        w = int(c[x:x + size, y:y + size, z:z + size].sum())
        if w == 0:
            return
        if size > 1:
            h = size // 2
            kids = [(x + dx, y + dy, z + dz) for dx in (0, h) for dy in (0, h) for dz in (0, h)]
            kw = np.array([c[a:a + h, b:b + h, d:d + h].sum() for a, b, d in kids])
            if split_if(w, kw[kw > 0]):
                for k in kids:
                    visit(*k, h)
                return
        out.append(((x, y, z), size, w))

    visit(0, 0, 0, len(c))
    return out


def octree_limit(c: np.ndarray, limit: float) -> list[Node]:
    return octree(c, lambda w, kw: w > limit)


def octree_target(c: np.ndarray, target: float) -> list[Node]:
    def split_if(w: int, kw: np.ndarray) -> bool:
        return w > CAP * target or (w > target and np.mean(np.abs(np.log(kw / target))) < abs(np.log(w / target)))
    return octree(c, split_if)


def z_runs(c: np.ndarray, max_w: float) -> np.ndarray:
    """Points per run when the occupied leaves, in Z-order, are cut into runs of ``max_w / 4`` to ``max_w``."""
    xyz = np.argwhere(c > 0)
    codes = morton(xyz, int(np.log2(len(c))))
    order = np.argsort(codes)
    w = c[tuple(xyz[order].T)]
    cum = np.concatenate([[0], np.cumsum(w)])
    return np.array([cum[b] - cum[a] for _, a, b in build_runs(codes[order], w, 3, max_w / 4, max_w)])


def tune(sizes: Callable[[float], np.ndarray], lo: float, hi: float) -> tuple[float, np.ndarray]:
    """The parameter in ``[lo, hi]`` whose objects' mean size is closest to ``MEAN`` (mean grows with it)."""
    best = None
    for _ in range(40):
        mid = np.sqrt(lo * hi)
        w = sizes(mid)
        if best is None or abs(np.log(w.mean() / MEAN)) < abs(np.log(best[1].mean() / MEAN)):
            best = (mid, w)
        lo, hi = (mid, hi) if w.mean() < MEAN else (lo, mid)
    return best


def cube_edges(cubes: list[tuple[tuple[float, float, float], float]]) -> list:
    """The edges of axis-aligned cubes ``(corner, size)``, each drawn once."""
    seen = set()
    for (x, y, z), s in cubes:
        v = [(x + a, y + b, z + d) for a in (0, s) for b in (0, s) for d in (0, s)]
        for i, j in [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]:
            seen.add((v[i], v[j]))
    return [list(e) for e in seen]


def per_decade(weights: np.ndarray, grid: np.ndarray, lower: float = 1, upper: float | None = None,
               bw: float = 0.12) -> np.ndarray:
    """Object sizes as a smooth curve: a Gaussian kernel density over log10(points), as a share of the objects
    per decade.  The kernels are reflected at the bounds, so no density spills past them; the curve is zero
    outside them."""
    x = np.log10(weights)[:, None]
    lo, hi = np.log10(lower), np.log10(upper) if upper else None
    mirrors = [x, 2 * lo - x] + ([2 * hi - x] if hi is not None else [])
    y = sum(np.exp(-0.5 * ((grid[None] - m) / bw) ** 2).sum(0) for m in mirrors) / (bw * np.sqrt(2 * np.pi) * len(weights))
    inside = (grid >= lo) & ((grid <= hi) if hi is not None else True)
    return np.where(inside, y, 0.0)


def style(ax) -> None:
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_zlim(0, N)
    ax.set_box_aspect((1, 1, 1), zoom=1.05)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()


def fig_concept(out: Path) -> None:
    p = cluster_points()
    c = counts(p, N)
    k = min(range(2, 33), key=lambda k: abs(np.log(len(p) / (counts(p, k) > 0).sum() / MEAN)))
    ck = counts(p, k)
    limit, _ = tune(lambda x: np.array([w for *_, w in octree_limit(c, x)]), 1e3, 1e6)
    target, _ = tune(lambda x: np.array([w for *_, w in octree_target(c, x)]), 1e3, 1e6)
    max_w, runs = tune(lambda x: z_runs(c, x), 1e3, 1e6)
    nodes = octree_limit(c, limit)
    sizes = {
        "constant": (f"constant chunks, {k}³", ck[ck > 0], 1, None),
        "limit": (f"octree, limit {limit / 1e3:.0f}k", np.array([w for *_, w in nodes]), 1, limit),
        "target": (f"octree, target {target / 1e3:.0f}k (cap {CAP}×)",
                   np.array([w for *_, w in octree_target(c, target)]), 1, CAP * target),
        "runs": (f"Z-order runs, {max_w / 4e3:.1f}k–{max_w / 1e3:.1f}k (zarr-octree)", runs, max_w / 4, max_w),
    }

    ijk = np.floor(p * N).astype(int)
    keep = 1 / np.sqrt(c[tuple(ijk.T)])  # thin dense chunks more than sparse ones, so both stay visible
    show = p[np.random.default_rng(1).random(len(p)) < keep * 25_000 / keep.sum()] * N
    s = N / k
    panels = [
        ("constant", f"Constant chunks ({k}³)", cube_edges([(tuple(v * s for v in xyz), s) for xyz in np.argwhere(ck > 0)])),
        ("limit", f"Octree (nodes ≤ {limit / 1e3:.0f}k points)", cube_edges([(xyz, sz) for xyz, sz, _ in nodes])),
    ]
    fig = plt.figure(figsize=(11, 8.2))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.5], hspace=0.1, wspace=0.08)
    for col, (key, title, segs) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.scatter(*show.T, s=0.8, color=INK2, alpha=0.45, linewidths=0, rasterized=True, depthshade=False)
        ax.add_collection3d(Line3DCollection(segs, colors=COLOR[key], linewidths=0.6, alpha=0.6))
        ax.add_collection3d(Line3DCollection(cube_edges([((0, 0, 0), N)]), colors=MUTED, linewidths=0.6, alpha=0.6))
        style(ax)
        ax.set_title(f"{title}: {len(sizes[key][1])} objects", fontsize=10.5, pad=0)

    h = fig.add_subplot(gs[1, :])
    grid = np.linspace(0, np.log10(max(w.max() for _, w, _, _ in sizes.values())) + 0.2, 800)
    for key, (label, w, lower, upper) in sizes.items():
        h.plot(10 ** grid, per_decade(w, grid, lower, upper), color=COLOR[key],
               label=f"{label}: {len(w)} objects, {w.min():,}–{w.max():,} points, CV {w.std() / w.mean():.2f}")
    h.set_xscale("log")
    h.set_xlim(1, 10 ** grid[-1])
    h.set_ylim(0, None)
    h.yaxis.set_major_locator(MaxNLocator(4))
    h.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
    h.grid(axis="x", visible=False)
    h.set_xlabel("points per object")
    h.set_ylabel("share of objects\n(per decade)")
    h.set_title(f"Object sizes, every layout tuned to a mean of {MEAN:,} points per object",
                fontsize=9.5, color=INK2, fontweight="normal")
    h.legend(loc="upper left", fontsize=8)
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)
    for key, (label, w, _, _) in sizes.items():
        print(f"{label:45s} {len(w):4d} objects  mean {w.mean():7,.0f}  median {np.median(w):7,.0f}  CV {w.std() / w.mean():.2f}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_concept(args.out / "concept.png")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

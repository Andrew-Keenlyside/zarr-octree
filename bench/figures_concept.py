"""The README's explanatory figure: constant chunks against an octree, on a synthetic point cloud.

    python bench/figures_concept.py

Three clusters with broad (Student-t) tails, so the density falls smoothly from dense cores to a
sparse background.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.ticker import MaxNLocator, PercentFormatter
from mpl_toolkits.mplot3d.art3d import Line3DCollection

from figures import COLOR, HERE, INK2, MUTED

N = 128  # leaf chunks per axis: the octree's finest level
CONST = 16  # constant chunks per axis
MAX_W = 32_000  # the most points a node may hold


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


def counts(p: np.ndarray) -> np.ndarray:
    """Points per chunk, indexed ``[x, y, z]``."""
    ijk = np.floor(p * N).astype(int)
    c = np.zeros((N, N, N), dtype=np.int64)
    np.add.at(c, tuple(ijk.T), 1)
    return c


def octree_leaves(c: np.ndarray) -> list[tuple[tuple[int, int, int], int, int]]:
    """Split the octree until no node holds more than ``MAX_W`` points: ``[(corner, size, points)]``."""
    out = []

    def split(x: int, y: int, z: int, size: int) -> None:
        w = int(c[x:x + size, y:y + size, z:z + size].sum())
        if w > MAX_W and size > 1:
            h = size // 2
            for dx in (0, h):
                for dy in (0, h):
                    for dz in (0, h):
                        split(x + dx, y + dy, z + dz, h)
        else:
            out.append(((x, y, z), size, w))

    split(0, 0, 0, N)
    return out


def cube_edges(cubes: list[tuple[tuple[int, int, int], int]]) -> list:
    """The edges of axis-aligned cubes ``(corner, size)``, each drawn once."""
    seen = set()
    for (x, y, z), s in cubes:
        v = [(x + a, y + b, z + d) for a in (0, s) for b in (0, s) for d in (0, s)]
        for i, j in [(0, 1), (0, 2), (0, 4), (1, 3), (1, 5), (2, 3), (2, 6), (3, 7), (4, 5), (4, 6), (5, 7), (6, 7)]:
            seen.add((v[i], v[j]))
    return [list(e) for e in seen]


def per_decade(weights: np.ndarray, grid: np.ndarray, bw: float = 0.18) -> np.ndarray:
    """Object sizes as a smooth curve: a Gaussian kernel density over log10(points), as a share of the objects per decade."""
    x = np.log10(weights)[:, None]
    return np.exp(-0.5 * ((grid[None] - x) / bw) ** 2).sum(0) / (bw * np.sqrt(2 * np.pi) * len(weights))


def style(ax) -> None:
    ax.set_xlim(0, N)
    ax.set_ylim(0, N)
    ax.set_zlim(0, N)
    ax.set_box_aspect((1, 1, 1), zoom=1.05)
    ax.view_init(elev=20, azim=-60)
    ax.set_axis_off()


def fig_concept(out: Path) -> None:
    p = cluster_points()
    c = counts(p)
    leaves = octree_leaves(c)
    ijk = np.floor(p * N).astype(int)
    dens = c[tuple(ijk.T)]
    keep = 1 / np.sqrt(dens)  # thin dense chunks more than sparse ones, so both stay visible
    show = p[np.random.default_rng(1).random(len(p)) < keep * 25_000 / keep.sum()] * N
    b = N // CONST
    cc = c.reshape(CONST, b, CONST, b, CONST, b).sum((1, 3, 5))
    occupied = [(tuple(int(v) * b for v in xyz), b) for xyz in np.argwhere(cc > 0)]
    panels = [
        ("leaf", f"Constant chunks ({CONST}³)", cube_edges(occupied), cc[cc > 0], 0.25, 0.25),
        ("octree", f"Octree on {N}³ leaves (nodes ≤ {MAX_W // 1000}k points)",
         cube_edges([(xyz, s) for xyz, s, w in leaves if w > 0]), np.array([w for *_, w in leaves if w > 0]), 0.5, 0.5),
    ]
    fig = plt.figure(figsize=(11, 7.6))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.1], hspace=0.12, wspace=0.08)
    grid = np.linspace(0, np.log10(max(cc.max(), MAX_W)) + 0.4, 400)
    curves = {key: per_decade(w, grid) for key, _, _, w, _, _ in panels}
    top = max(y.max() for y in curves.values()) * 1.2
    for col, (key, title, segs, weights, lw, alpha) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col], projection="3d")
        ax.scatter(*show.T, s=0.8, color=INK2, alpha=0.45, linewidths=0, rasterized=True, depthshade=False)
        ax.add_collection3d(Line3DCollection(segs, colors=COLOR[key], linewidths=lw, alpha=alpha))
        ax.add_collection3d(Line3DCollection(cube_edges([((0, 0, 0), N)]), colors=MUTED, linewidths=0.6, alpha=0.6))
        style(ax)
        ax.set_title(title, fontsize=10.5, pad=0)
        h = fig.add_subplot(gs[1, col])
        h.fill_between(10 ** grid, curves[key], color=COLOR[key], alpha=0.15, linewidth=0)
        h.plot(10 ** grid, curves[key], color=COLOR[key])
        h.set_xscale("log")
        h.set_xlim(10 ** grid[0], 10 ** grid[-1])
        h.set_ylim(0, top)
        h.yaxis.set_major_locator(MaxNLocator(4))
        h.yaxis.set_major_formatter(PercentFormatter(1, decimals=0))
        if key == "octree":
            h.axvline(MAX_W, color=INK2, linewidth=1, linestyle=(0, (3, 2)))
            h.text(MAX_W * 0.9, top, f"{MAX_W // 1000}k limit", ha="right", va="top", fontsize=8, color=INK2)
        h.grid(axis="x", visible=False)
        h.set_xlabel("points per object")
        h.set_ylabel("share of objects\n(per decade)" if col == 0 else "")
        h.set_title(f"{len(weights):,} objects, {weights.min():,} to {weights.max():,} points each",
                    fontsize=9, color=INK2, fontweight="normal")
    fig.savefig(out, dpi=160, bbox_inches="tight", pad_inches=0.15)
    plt.close(fig)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, default=HERE / "results" / "figures")
    args = ap.parse_args()
    args.out.mkdir(parents=True, exist_ok=True)
    fig_concept(args.out / "concept.png")
    print(f"-> {args.out}")


if __name__ == "__main__":
    main()

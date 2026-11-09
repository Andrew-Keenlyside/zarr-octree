"""The README's explanatory figure: constant chunks against octree runs, on a 2-D toy.

    python bench/figures_concept.py

In 2-D the octree is a quadtree; the 3-D case is the same with eight children a node.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.collections import LineCollection
from matplotlib.ticker import MaxNLocator, PercentFormatter

from figures import COLOR, GRID_C, HERE, INK2, MUTED

N = 32  # chunks per axis
MAX_W = 8_000  # the most points a node may hold


def synthetic_points(seed: int = 0) -> np.ndarray:
    """Clustered points in the unit square: blobs of many sizes and densities, curved filaments, a ring,
    a dense core inside a halo, and a sparse background."""
    rng = np.random.default_rng(seed)
    parts = []
    for centre, sigma, n in [((0.22, 0.72), 0.05, 50_000), ((0.70, 0.28), 0.08, 40_000), ((0.82, 0.84), 0.022, 22_000),
                             ((0.47, 0.47), 0.012, 9_000), ((0.12, 0.30), 0.03, 6_000), ((0.58, 0.88), 0.035, 4_000),
                             ((0.90, 0.55), 0.015, 3_000), ((0.35, 0.15), 0.02, 2_000)]:
        parts.append(rng.normal(centre, sigma, (n, 2)))
    parts.append(rng.normal((0.70, 0.28), 0.006, (12_000, 2)))  # a dense core in the halo above
    t = rng.uniform(0, 1, 18_000)
    parts.append(np.c_[0.05 + 0.55 * t, 0.08 + 0.3 * t + 0.07 * np.sin(7 * t)] + rng.normal(0, 0.008, (18_000, 2)))
    t = rng.uniform(0, 1, 10_000)
    parts.append(np.c_[0.30 + 0.5 * t, 0.58 + 0.12 * np.sin(5 * t + 1)] + rng.normal(0, 0.006, (10_000, 2)))
    a = rng.uniform(0, 2 * np.pi, 8_000)
    parts.append(np.c_[0.50 + 0.10 * np.cos(a), 0.76 + 0.10 * np.sin(a)] + rng.normal(0, 0.007, (8_000, 2)))
    parts.append(rng.uniform(0, 1, (400, 2)))
    p = np.concatenate(parts)
    return p[((p >= 0) & (p < 1)).all(1)]


def counts(p: np.ndarray) -> np.ndarray:
    """Points per chunk, indexed ``[y, x]``."""
    ij = np.floor(p * N).astype(int)
    c = np.zeros((N, N), dtype=np.int64)
    np.add.at(c, (ij[:, 1], ij[:, 0]), 1)
    return c


def octree_leaves(c: np.ndarray) -> list[tuple[int, int, int, int]]:
    """Split the octree (here a quadtree) until no node holds more than ``MAX_W`` points: ``[(x, y, size, points)]``."""
    out = []

    def split(x: int, y: int, size: int) -> None:
        w = int(c[y:y + size, x:x + size].sum())
        if w > MAX_W and size > 1:
            h = size // 2
            for dy in (0, h):
                for dx in (0, h):
                    split(x + dx, y + dy, h)
        else:
            out.append((x, y, size, w))

    split(0, 0, N)
    return out


def per_decade(weights: np.ndarray, grid: np.ndarray, bw: float = 0.18) -> np.ndarray:
    """Object sizes as a smooth curve: a Gaussian kernel density over log10(points), as a share of the objects per decade."""
    x = np.log10(weights)[:, None]
    return np.exp(-0.5 * ((grid[None] - x) / bw) ** 2).sum(0) / (bw * np.sqrt(2 * np.pi) * len(weights))


def fig_concept(out: Path) -> None:
    p = synthetic_points()
    c = counts(p)
    leaves = octree_leaves(c)
    ij = np.floor(p * N).astype(int)
    sparse = c[ij[:, 1], ij[:, 0]] < 50  # every point of a sparse chunk; a sample of the rest
    show = p[sparse | (np.random.default_rng(1).random(len(p)) < 12_000 / len(p))] * N
    cells = [[(x, y), (x + 1, y), (x + 1, y + 1), (x, y + 1), (x, y)] for y, x in np.argwhere(c > 0)]
    nodes = [[(x, y), (x + s, y), (x + s, y + s), (x, y + s), (x, y)] for x, y, s, _ in leaves]
    panels = [
        ("leaf", "Constant chunks", cells, c[c > 0]),
        ("octree", f"Octree (split while a node holds > {MAX_W // 1000}k points)", nodes,
         np.array([w for *_, w in leaves if w > 0])),
    ]
    fig = plt.figure(figsize=(10.4, 7.0))
    gs = fig.add_gridspec(2, 2, height_ratios=[4, 1.15], hspace=0.25, wspace=0.12)
    grid = np.linspace(0, np.log10(max(c.max(), MAX_W)) + 0.4, 400)
    curves = {key: per_decade(w, grid) for key, _, _, w in panels}
    top = max(y.max() for y in curves.values()) * 1.2
    for col, (key, title, outlines, weights) in enumerate(panels):
        ax = fig.add_subplot(gs[0, col])
        ax.add_collection(LineCollection([[(i, 0), (i, N)] for i in range(N + 1)] + [[(0, i), (N, i)] for i in range(N + 1)],
                                         colors=GRID_C, linewidths=0.6))
        ax.scatter(show[:, 0], show[:, 1], s=1.0, color=MUTED, alpha=0.6, linewidths=0)
        ax.add_collection(LineCollection(outlines, colors=COLOR[key], linewidths=0.7 if key == "leaf" else 1.4))
        ax.set_xlim(0, N)
        ax.set_ylim(0, N)
        ax.set_aspect("equal")
        ax.set_xticks([])
        ax.set_yticks([])
        ax.grid(False)
        for sp in ax.spines.values():
            sp.set_visible(False)
        ax.set_title(title, fontsize=10.5)
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
        h.set_title(f"{len(weights)} objects, {weights.min():,} to {weights.max():,} points each",
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

"""A 3-D point cloud with strongly varied density: a densMAP embedding of synthetic data.

Synthetic high-dimensional data only: Gaussian clusters with power-law sizes
and spreads, a few curved one-dimensional manifolds, and a uniform
background.  densMAP (UMAP with ``densmap=True``) is used rather than plain
UMAP because UMAP evens out local density, which is the property this
benchmark needs to keep.  The embedding is then resampled to ``--points``
points by jittering each point within its local neighbour distance, so the
density contrast survives the upsampling.

    python bench/make_points.py --embed 150000 --points 4000000
"""

from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np


def synthetic_highdim(n: int, dim: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    n_clusters = 24
    weights = 1.0 / np.arange(1, n_clusters + 1) ** 1.3  # a few big clusters, many small
    weights /= weights.sum()
    n_bg = n // 50
    n_curve = n // 8
    sizes = rng.multinomial(n - n_bg - n_curve, weights)
    parts, labels = [], []
    for k, m in enumerate(sizes):
        centre = rng.normal(0, 6, dim)
        spread = 0.15 + 2.0 * rng.random() ** 2  # tight and loose clusters
        axes = rng.normal(0, 1, (dim, dim)) * (rng.random(dim) ** 3)  # anisotropic
        parts.append(centre + spread * rng.normal(0, 1, (m, dim)) @ axes.T / np.sqrt(dim) * 4)
        labels.append(np.full(m, k))
    for k in range(3):
        t = rng.uniform(0, 3 * np.pi, n_curve // 3)
        basis = rng.normal(0, 1, (3, dim))
        curve = np.stack([np.cos(t), np.sin(t), t / 3], 1) @ basis * 4 + rng.normal(0, 6, dim)
        parts.append(curve + rng.normal(0, 0.1, (len(t), dim)))
        labels.append(np.full(len(t), n_clusters + k))
    parts.append(rng.uniform(-15, 15, (n - sum(len(p) for p in parts), dim)))
    labels.append(np.full(len(parts[-1]), -1))
    return np.vstack(parts).astype(np.float32), np.concatenate(labels)


def upsample(emb: np.ndarray, labels: np.ndarray, n: int, rng: np.random.Generator) -> tuple[np.ndarray, np.ndarray]:
    from sklearn.neighbors import NearestNeighbors

    dist, _ = NearestNeighbors(n_neighbors=9).fit(emb).kneighbors(emb)
    scale = dist[:, 1:].mean(1)
    pick = rng.integers(0, len(emb), n)
    pts = emb[pick] + rng.normal(0, 1, (n, 3)) * scale[pick, None] * 0.5
    return pts.astype(np.float32), labels[pick]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--embed", type=int, default=150_000, help="points embedded with densMAP")
    ap.add_argument("--points", type=int, default=4_000_000, help="points after resampling")
    ap.add_argument("--dim", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", type=Path, default=Path(__file__).parent / "_data" / "densmap_points.npz")
    args = ap.parse_args()

    import umap

    rng = np.random.default_rng(args.seed)
    x, labels = synthetic_highdim(args.embed, args.dim, rng)
    t0 = time.time()
    emb = umap.UMAP(n_components=3, densmap=True, n_neighbors=30, min_dist=0.05, n_jobs=-1).fit_transform(x)
    print(f"densMAP on {len(x)} x {args.dim}: {time.time() - t0:.0f} s", flush=True)
    pts, lab = upsample(emb, labels, args.points, rng)
    lo, hi = np.percentile(pts, [0.05, 99.95], axis=0)
    keep = ((pts >= lo) & (pts <= hi)).all(1)
    pts, lab = pts[keep], lab[keep]
    span = (hi - lo).max()
    pts = (pts - lo) / span * 1000.0  # a 1000-unit box, longest axis
    args.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(args.out, points=pts.astype(np.float32), labels=lab.astype(np.int16), embedding=((emb - lo) / span * 1000).astype(np.float32))
    print(f"{len(pts)} points -> {args.out}; extent {pts.max(0).round(1)}")


if __name__ == "__main__":
    main()

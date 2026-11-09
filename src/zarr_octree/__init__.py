"""Octree chunk packing for Zarr v3, as a store-layer extension.

See ``SPEC.md`` for the on-disk format.
"""

from zarr_octree.store import EXTENSION, OctreeConfig, OctreeStore, convert
from zarr_octree.tree import build_octree, build_runs, decode_manifest, decode_node, encode_manifest, encode_node

__all__ = [
    "EXTENSION",
    "OctreeConfig",
    "OctreeStore",
    "build_octree",
    "build_runs",
    "convert",
    "decode_manifest",
    "decode_node",
    "encode_manifest",
    "encode_node",
]
__version__ = "0.2.0"

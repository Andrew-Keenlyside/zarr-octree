"""Octree chunk packing for Zarr v3, as a store-layer extension.

See ``SPEC.md`` for the on-disk format.
"""

from zarr_octree.store import EXTENSION, OctreeConfig, OctreeStore, convert
from zarr_octree.tree import build_partition, decode_manifest, decode_node, encode_manifest, encode_node

__all__ = [
    "EXTENSION",
    "OctreeConfig",
    "OctreeStore",
    "build_partition",
    "convert",
    "decode_manifest",
    "decode_node",
    "encode_manifest",
    "encode_node",
]
__version__ = "0.1.0"

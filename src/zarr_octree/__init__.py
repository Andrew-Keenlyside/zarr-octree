"""``octree``: a Zarr v3 sharding codec with a sparse index and octree runs.

The format is specified in ``registry/codecs/octree/README.md``.
"""

from zarr_octree.codec import CODEC_NAME, InvalidIndex, OctreeShardingCodec, register_weigher
from zarr_octree.tree import build_octree, build_runs

__all__ = ["CODEC_NAME", "InvalidIndex", "OctreeShardingCodec", "build_octree", "build_runs", "register_weigher"]
__version__ = "0.4.0"

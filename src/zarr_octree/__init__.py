"""``zarr_octree_sharding``: a Zarr v3 sharding codec with a sparse index and octree runs.

See ``SPEC.md`` for the shard format.
"""

from zarr_octree.codec import CODEC_NAME, OctreeShardingCodec, register_weigher
from zarr_octree.tree import build_octree, build_runs

__all__ = ["CODEC_NAME", "OctreeShardingCodec", "build_octree", "build_runs", "register_weigher"]
__version__ = "0.3.0"

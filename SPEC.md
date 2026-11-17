# The `octree` shard format

The specification is the codec's entry for the Zarr extensions registry:
[`registry/codecs/octree/README.md`](registry/codecs/octree/README.md), with
its JSON schema beside it.  It covers the metadata, the shard format, what a
reader must check, and the legacy `zarr_octree_sharding` name and index
version 2 that zarr-octree 0.3 and earlier wrote.

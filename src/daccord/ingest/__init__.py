"""Tier-4 full-corpus PDF→markdown ingest package.

`marker_runner` wraps Marker for whole-document parsing; `manifest` defines the
on-disk schema for `data/ingest/manifest.jsonl`. The tier-2D bake-off code in
`daccord.bakeoff` operates on per-page rasterized PDFs and is left untouched —
this package owns the production parse path going forward.
"""

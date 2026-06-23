"""I/O for stereo-fate.

Loaders return a standard AnnData with spatial coordinates in ``adata.obsm['spatial']``.
We work from *processed* bin/cell-level Stereo-seq data (GEF via stereopy, or .h5ad) --
never from raw FASTQ. We do NOT invoke SAW.
"""

from __future__ import annotations

from .loaders import (
    bundled_subsample_path,
    download_mosta,
    load_adata,
    load_bundled_subsample,
    load_gef,
    load_h5ad,
)

__all__ = [
    "load_adata",
    "load_h5ad",
    "load_gef",
    "load_bundled_subsample",
    "bundled_subsample_path",
    "download_mosta",
]

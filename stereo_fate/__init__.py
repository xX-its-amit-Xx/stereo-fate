"""stereo-fate: nominate the TF + signaling "ingredients" of cell-fate decisions
from Stereo-seq spatial transcriptomics, and prioritize them with functional-genomics
(perturbation) evidence.

The package is anndata-native and built on the standard spatial-omics stack
(stereopy, scanpy, squidpy, liana, decoupler) so its outputs drop straight back
into that ecosystem.

Pipeline
--------
1. ``io``            load processed Stereo-seq (GEF via stereopy, or .h5ad) -> AnnData
2. ``spatial``       QC, normalize, Leiden spatial domains, marker-based cell types
3. ``fate``          order cells along a differentiation / commitment axis
4. ``tf_activity``   intrinsic ingredients: TF activity (decoupler + CollecTRI)
5. ``communication`` extrinsic ingredients: spatial L-R signaling (liana + squidpy)
6. ``recipe``        combine intrinsic + extrinsic into one ranked ingredient list
7. ``funcgen``       cross-reference perturbation data -> causal vs correlative flag

Every entrypoint calls :func:`stereo_fate.resources.check_resources` first.
"""

from __future__ import annotations

from . import communication, fate, funcgen, io, recipe, spatial, tf_activity
from .resources import ResourceReport, check_resources

__version__ = "0.1.0"

__all__ = [
    "io",
    "spatial",
    "fate",
    "tf_activity",
    "communication",
    "recipe",
    "funcgen",
    "check_resources",
    "ResourceReport",
    "__version__",
]

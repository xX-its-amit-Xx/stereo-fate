"""Shared helper for the cookbook notebooks.

Computes the stereo-fate pipeline **once** on the bundled MOSTA-like subsample and
caches the annotated AnnData to ``cookbook/_cache/result.h5ad`` so every notebook can
load it cheaply (important on memory-tight machines). Always runs ``check_resources``
first, per the standing requirement.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

import stereo_fate as sf

HERE = Path(__file__).resolve().parent
CACHE = HERE / "_cache"
CACHE.mkdir(exist_ok=True)
RESULT = CACHE / "result.h5ad"
NET = HERE.parent / "stereo_fate" / "data" / "collectri_mouse_subset.csv"

TARGET_FATE = "Cardiac"
PROGENITOR = "Mesenchyme"


def build_result(rebuild: bool = False):
    """Run (or load cached) the full pipeline; return the annotated AnnData."""
    sf.check_resources(verbose=True)
    if RESULT.exists() and not rebuild:
        return sf.io.load_h5ad(RESULT, check=False)

    adata = sf.io.load_bundled_subsample()
    params = sf.spatial.SpatialParams(n_top_genes=800, leiden_resolution=0.6,
                                      spatial_weight=0.3)
    adata = sf.spatial.qc_filter(adata, params)
    adata = sf.spatial.normalize(adata, params, inplace=True)
    adata = sf.spatial.spatial_domains(adata, params, inplace=True)
    adata = sf.spatial.annotate_cell_types(adata, inplace=True)
    adata = sf.fate.fate_axis(adata, target_fate=TARGET_FATE,
                              progenitor_fate=PROGENITOR, inplace=True)

    net = pd.read_csv(NET)
    sf.tf_activity.infer_tf_activity(adata, net=net, inplace=True)
    sf.tf_activity.rank_tfs_along_axis(adata)

    sf.communication.infer_communication(adata, n_perms=100, n_jobs=1, inplace=True)
    sf.communication.rank_lr_by_fate_coincidence(adata)

    rank = adata.uns["stereo_fate"]["tf_ranking"]
    lr = adata.uns["stereo_fate"]["lr_ranking"]
    recipe = sf.recipe.build_recipe(adata, tf_ranking=rank, lr_ranking=lr,
                                    target_fate=TARGET_FATE)
    sf.funcgen.prioritize(recipe, fate=TARGET_FATE, adata=adata)

    adata.uns.get("stereo_fate", {}).pop("domain_scores", None)
    adata.write_h5ad(RESULT)
    return adata

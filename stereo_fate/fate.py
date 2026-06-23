"""Fate axis: order cells along a differentiation gradient.

Two regimes, matching the spec:

* **pseudotime** -- when a developmental timecourse / clear progenitor population is
  present, use diffusion pseudotime (``scanpy.tl.dpt``) rooted at the progenitor.
* **committed-vs-progenitor contrast** -- otherwise, define the fate axis as a
  continuous commitment score: similarity to the committed target fate minus
  similarity to the progenitor compartment.

Either way the result is a single per-cell scalar in ``adata.obs['fate_axis']``
(0 = progenitor / uncommitted, 1 = committed to the target fate), plus a boolean
``adata.obs['is_target_fate']``.
"""

from __future__ import annotations

import numpy as np
import scanpy as sc
from scipy import sparse
from sklearn.preprocessing import minmax_scale

from .resources import check_resources


def _mean_expr(adata, mask, genes):
    genes = [g for g in genes if g in adata.var_names]
    if not genes or mask.sum() == 0:
        return None
    sub = adata[mask, genes].X
    sub = sub.toarray() if sparse.issparse(sub) else np.asarray(sub)
    return sub.mean()


def fate_axis_pseudotime(
    adata,
    *,
    root_group: str,
    groupby: str = "cell_type",
    n_dcs: int = 15,
    inplace: bool = False,
):
    """Diffusion pseudotime rooted in a progenitor group.

    Requires a neighbors graph (run :func:`stereo_fate.spatial.spatial_domains`
    first). Writes ``adata.obs['fate_axis']`` (min-max scaled DPT).
    """
    if not inplace:
        adata = adata.copy()
    check_resources(adata.n_obs, adata.n_vars, verbose=False)
    if "neighbors" not in adata.uns:
        sc.pp.neighbors(adata, use_rep="X_pca" if "X_pca" in adata.obsm else None)
    if groupby not in adata.obs or root_group not in set(adata.obs[groupby]):
        raise ValueError(f"root_group {root_group!r} not found in obs['{groupby}'].")

    sc.tl.diffmap(adata, n_comps=n_dcs)
    # root = the cell of the progenitor group closest to its centroid in diffusion space
    root_mask = (adata.obs[groupby] == root_group).to_numpy()
    dm = adata.obsm["X_diffmap"]
    centroid = dm[root_mask].mean(0)
    root_idx = np.where(root_mask)[0][np.argmin(((dm[root_mask] - centroid) ** 2).sum(1))]
    adata.uns["iroot"] = int(root_idx)
    sc.tl.dpt(adata)

    dpt = np.asarray(adata.obs["dpt_pseudotime"])
    dpt[~np.isfinite(dpt)] = np.nanmax(dpt[np.isfinite(dpt)]) if np.isfinite(dpt).any() else 0.0
    adata.obs["fate_axis"] = minmax_scale(dpt).astype(np.float32)
    adata.uns.setdefault("stereo_fate", {})["fate_method"] = "pseudotime"
    adata.uns["stereo_fate"]["fate_root"] = root_group
    return adata


def fate_axis_contrast(
    adata,
    *,
    target_fate: str,
    progenitor_fate: str | None = None,
    groupby: str = "cell_type",
    signature: dict[str, list[str]] | None = None,
    inplace: bool = False,
):
    """Committed-vs-progenitor commitment score (no timecourse needed).

    The axis is ``score(target) - score(progenitor)``, min-max scaled to [0, 1].
    If ``signature`` (fate -> marker genes) is given, gene-signature scores are
    used; otherwise the axis is built from PCA-space proximity to each group's
    centroid. Writes ``adata.obs['fate_axis']`` and ``adata.obs['is_target_fate']``.
    """
    if not inplace:
        adata = adata.copy()
    check_resources(adata.n_obs, adata.n_vars, verbose=False)
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not in obs; annotate cell types first.")
    if target_fate not in set(adata.obs[groupby]):
        raise ValueError(f"target_fate {target_fate!r} not in obs['{groupby}'].")

    if signature is not None:
        tgt_genes = signature.get(target_fate, [])
        sc.tl.score_genes(adata, [g for g in tgt_genes if g in adata.var_names],
                          score_name="_tgt_score", ctrl_size=50)
        tgt = np.asarray(adata.obs["_tgt_score"])
        if progenitor_fate and progenitor_fate in signature:
            prog_genes = signature[progenitor_fate]
            sc.tl.score_genes(adata, [g for g in prog_genes if g in adata.var_names],
                              score_name="_prog_score", ctrl_size=50)
            prog = np.asarray(adata.obs["_prog_score"])
            adata.obs.drop(columns=["_prog_score"], inplace=True)
        else:
            prog = 0.0
        axis = tgt - prog
        adata.obs.drop(columns=["_tgt_score"], inplace=True)
    else:
        rep = "X_pca" if "X_pca" in adata.obsm else None
        if rep is None:
            sc.tl.pca(adata, n_comps=min(30, adata.n_vars - 1))
            rep = "X_pca"
        emb = adata.obsm[rep]
        groups = adata.obs[groupby]
        tgt_c = emb[(groups == target_fate).to_numpy()].mean(0)
        d_tgt = np.linalg.norm(emb - tgt_c, axis=1)
        if progenitor_fate and progenitor_fate in set(groups):
            prog_c = emb[(groups == progenitor_fate).to_numpy()].mean(0)
            d_prog = np.linalg.norm(emb - prog_c, axis=1)
        else:
            d_prog = np.full(adata.n_obs, np.median(d_tgt))
        # closer to target & farther from progenitor => higher commitment
        axis = (d_prog - d_tgt)

    adata.obs["fate_axis"] = minmax_scale(axis).astype(np.float32)
    adata.obs["is_target_fate"] = (adata.obs[groupby] == target_fate).to_numpy()
    res = adata.uns.setdefault("stereo_fate", {})
    res["fate_method"] = "contrast"
    res["target_fate"] = target_fate
    res["progenitor_fate"] = progenitor_fate
    return adata


def fate_axis(
    adata,
    *,
    target_fate: str,
    progenitor_fate: str | None = None,
    method: str = "auto",
    groupby: str = "cell_type",
    signature: dict[str, list[str]] | None = None,
    inplace: bool = False,
):
    """Build the fate axis, dispatching on ``method``.

    ``method='auto'`` uses pseudotime when a ``progenitor_fate`` is supplied and a
    neighbors graph exists, otherwise the committed-vs-progenitor contrast.
    """
    if method == "pseudotime" or (
        method == "auto" and progenitor_fate is not None and "neighbors" in adata.uns
    ):
        return fate_axis_pseudotime(
            adata, root_group=progenitor_fate, groupby=groupby, inplace=inplace
        )
    return fate_axis_contrast(
        adata,
        target_fate=target_fate,
        progenitor_fate=progenitor_fate,
        groupby=groupby,
        signature=signature,
        inplace=inplace,
    )

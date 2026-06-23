"""Extrinsic ingredients: spatially-resolved cell-cell communication.

We nominate the *niche cues* of a fate: ligand-receptor (L-R) pairs whose signaling
**spatially coincides with fate commitment**. Two ingredients combine:

* **liana** ranks candidate L-R pairs by expression/specificity across cell types
  (cluster-level communication);
* **squidpy** builds the spatial neighborhood graph, on which we compute a *local*
  interaction score per bin -- ligand at a bin x receptor in its spatial neighborhood --
  and score how strongly that local signaling tracks the fate axis.

A pair that lights up exactly where cells commit is a candidate extrinsic driver.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import squidpy as sq
from scipy import sparse, stats

from .resources import cap_n_jobs, check_resources


def spatial_neighbors(adata, *, n_neighs: int = 6, coord_type: str = "generic",
                      inplace: bool = False):
    """Build the squidpy spatial neighborhood graph (``obsp['spatial_connectivities']``)."""
    if not inplace:
        adata = adata.copy()
    if "spatial" not in adata.obsm:
        raise KeyError("obsm['spatial'] required for the spatial graph.")
    sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type=coord_type)
    return adata


def infer_communication(
    adata,
    *,
    groupby: str = "cell_type",
    resource_name: str = "mouseconsensus",
    use_raw: bool = False,
    expr_prop: float = 0.1,
    n_perms: int = 100,
    n_jobs: int = 1,
    inplace: bool = False,
):
    """Cluster-level L-R inference with liana's rank-aggregate consensus.

    Stores the L-R table in ``adata.uns['liana_res']``.

    Notes
    -----
    ``n_jobs`` defaults to **1**: liana's permutation workers use process ``spawn``
    on Windows, and each worker re-imports the full stack (~300 MB). On memory-tight
    machines spawning ``nproc-1`` of them will thrash/OOM, so we keep it serial unless
    the caller explicitly opts into more (the value is still capped at ``nproc-1``).
    ``n_perms`` is bounded for the same reason; raise both for production runs with
    headroom.
    """
    import liana as li

    if not inplace:
        adata = adata.copy()
    check_resources(adata.n_obs, adata.n_vars, verbose=False)
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not in obs; annotate cell types first.")

    li.mt.rank_aggregate(
        adata, groupby=groupby, resource_name=resource_name, use_raw=use_raw,
        expr_prop=expr_prop, n_perms=n_perms, verbose=False, n_jobs=cap_n_jobs(n_jobs),
    )
    return adata


def _gene_vec(adata, gene, layer="lognorm"):
    if gene not in adata.var_names:
        return None
    X = adata[:, gene].layers[layer] if layer in adata.layers else adata[:, gene].X
    X = X.toarray().ravel() if sparse.issparse(X) else np.asarray(X).ravel()
    return X.astype(np.float64)


def _complex_vec(adata, complex_name, layer, agg=np.minimum):
    """Expression of a (possibly multi-subunit) complex; subunits split on '_'."""
    subs = str(complex_name).split("_")
    vecs = [_gene_vec(adata, s, layer) for s in subs]
    vecs = [v for v in vecs if v is not None]
    if not vecs:
        return None
    out = vecs[0]
    for v in vecs[1:]:
        out = agg(out, v)  # min across subunits = complex limited by scarcest subunit
    return out


def rank_lr_by_fate_coincidence(
    adata,
    *,
    fate_key: str = "fate_axis",
    layer: str = "lognorm",
    top_n_candidates: int = 200,
    n_neighs: int = 6,
):
    """Rank L-R pairs by spatial coincidence of signaling with fate commitment.

    For each candidate pair (from ``adata.uns['liana_res']``) we form a per-bin local
    interaction score ``L_i * mean_{j in N(i)} R_j`` over the spatial graph, then score:

    * ``spatial_coincidence`` -- Spearman correlation of the local score with ``fate_axis``;
    * ``effect_size`` -- standardized difference of the local score between committed
      (top-quartile axis) and progenitor (bottom-quartile) bins;
    * ``liana_specificity`` / ``liana_magnitude`` -- the cluster-level liana ranks.

    Returns a DataFrame (sorted by spatial coincidence) and stores it in
    ``adata.uns['stereo_fate']['lr_ranking']``.
    """
    if "liana_res" not in adata.uns:
        raise KeyError("Run infer_communication first (adata.uns['liana_res']).")
    if fate_key not in adata.obs:
        raise KeyError(f"'{fate_key}' missing; build the fate axis first.")
    if "spatial_connectivities" not in adata.obsp:
        sq.gr.spatial_neighbors(adata, n_neighs=n_neighs, coord_type="generic")

    lr = adata.uns["liana_res"].copy()
    # rank columns differ slightly across liana versions; pick what exists
    spec_col = next((c for c in ("specificity_rank", "magnitude_rank") if c in lr), None)
    mag_col = "magnitude_rank" if "magnitude_rank" in lr else spec_col
    if spec_col is not None:
        lr = lr.sort_values(spec_col)
    # unique ligand/receptor complex pairs, capped for tractability
    pairs = (
        lr[["ligand_complex", "receptor_complex"]]
        .drop_duplicates()
        .head(top_n_candidates)
    )

    S = adata.obsp["spatial_connectivities"].astype(np.float64)
    rowsum = np.asarray(S.sum(1)).ravel()
    rowsum[rowsum == 0] = 1.0

    fate = np.asarray(adata.obs[fate_key], dtype=float)
    n = len(fate)
    k = max(1, n // 4)
    order = np.argsort(fate)
    lo_idx, hi_idx = order[:k], order[-k:]

    # best (smallest) liana rank per pair for annotation
    agg = {}
    if spec_col:
        agg[spec_col] = "min"
    if mag_col and mag_col != spec_col:
        agg[mag_col] = "min"
    lr_best = lr.groupby(["ligand_complex", "receptor_complex"], observed=True).agg(agg)

    rows = []
    for lig, rec in pairs.itertuples(index=False):
        L = _complex_vec(adata, lig, layer)
        R = _complex_vec(adata, rec, layer)
        if L is None or R is None:
            continue
        R_nbr = (S @ R) / rowsum            # mean receptor in spatial neighborhood
        local = L * R_nbr                    # local sending x neighborhood receiving
        if local.std() == 0:
            continue
        rho, _ = stats.spearmanr(local, fate)
        hi, lo = local[hi_idx], local[lo_idx]
        pooled = np.sqrt((hi.var(ddof=1) + lo.var(ddof=1)) / 2) or 1e-9
        d = (hi.mean() - lo.mean()) / pooled
        try:
            _, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        except ValueError:
            p = 1.0
        srank = lr_best.loc[(lig, rec), spec_col] if spec_col else np.nan
        mrank = lr_best.loc[(lig, rec), mag_col] if mag_col and mag_col != spec_col else srank
        rows.append((lig, rec, f"{lig}->{rec}",
                     rho if np.isfinite(rho) else 0.0, d, p,
                     float(srank), float(mrank)))

    df = pd.DataFrame(
        rows,
        columns=["ligand", "receptor", "interaction", "spatial_coincidence",
                 "effect_size", "pval", "liana_specificity", "liana_magnitude"],
    )
    if len(df):
        from .tf_activity import _bh_fdr

        df["fdr"] = _bh_fdr(df["pval"].to_numpy())
        df = df.sort_values("spatial_coincidence", ascending=False).reset_index(drop=True)
        df["rank"] = np.arange(1, len(df) + 1)
    adata.uns.setdefault("stereo_fate", {})["lr_ranking"] = df
    return df

"""Intrinsic ingredients: transcription-factor activity along the fate axis.

We infer per-cell TF activity with **decoupler** (univariate linear model, ULM)
over the **CollecTRI** regulon resource, then rank TFs by how strongly their
activity *rises along the target-fate axis*. A TF whose activity climbs as cells
commit is a candidate intrinsic driver of that fate.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy import stats

from .resources import check_resources

TF_ACTIVITY_KEY = "tf_activity"


def get_regulons(organism: str = "mouse", net=None):
    """Return a TF->target regulon network (CollecTRI) as a DataFrame.

    Pass ``net`` to use a bundled / custom regulon table offline (CI uses this to
    avoid a network call); otherwise CollecTRI is fetched via decoupler/omnipath.
    """
    if net is not None:
        return net
    import decoupler as dc

    return dc.get_collectri(organism=organism, split_complexes=False)


def infer_tf_activity(
    adata,
    *,
    organism: str = "mouse",
    net=None,
    min_n: int = 5,
    use_raw: bool = False,
    inplace: bool = False,
):
    """Infer per-cell TF activity with decoupler ULM over CollecTRI.

    Stores the activity matrix (cells x TFs) in ``adata.obsm['tf_activity']`` and
    the regulon network in ``adata.uns['stereo_fate']['regulon_net']``.
    """
    import decoupler as dc

    if not inplace:
        adata = adata.copy()
    check_resources(adata.n_obs, adata.n_vars, verbose=False)
    net = get_regulons(organism=organism, net=net)

    dc.run_ulm(
        mat=adata, net=net, source="source", target="target", weight="weight",
        min_n=min_n, use_raw=use_raw, verbose=False,
    )
    # decoupler writes obsm['ulm_estimate'] (DataFrame: cells x TFs) and ulm_pvals
    acts = adata.obsm["ulm_estimate"]
    adata.obsm[TF_ACTIVITY_KEY] = acts
    res = adata.uns.setdefault("stereo_fate", {})
    res["regulon_source"] = "CollecTRI" if net is None else "custom"
    res["n_tfs"] = int(acts.shape[1])
    return adata


def rank_tfs_along_axis(
    adata,
    *,
    fate_key: str = "fate_axis",
    activity_key: str = TF_ACTIVITY_KEY,
    top_frac: float = 0.25,
):
    """Rank TFs by association of activity with the fate axis.

    For each TF we compute:

    * ``spearman`` -- rank correlation of TF activity vs ``fate_axis`` (direction +
      monotonic strength);
    * ``effect_size`` -- standardized mean difference (Cohen's d) of activity between
      the most-committed cells (top ``top_frac`` of the axis) and the least-committed
      (bottom ``top_frac``);
    * ``pval`` / ``fdr`` -- Mann-Whitney U test between those two groups, BH-corrected.

    Returns a DataFrame sorted by descending effect size (TFs rising along the axis
    first) and stores it in ``adata.uns['stereo_fate']['tf_ranking']``.
    """
    if activity_key not in adata.obsm:
        raise KeyError(f"'{activity_key}' missing; run infer_tf_activity first.")
    if fate_key not in adata.obs:
        raise KeyError(f"'{fate_key}' missing; build the fate axis first.")

    acts = adata.obsm[activity_key]
    if isinstance(acts, pd.DataFrame):
        tf_names = list(acts.columns)
        A = acts.to_numpy()
    else:
        tf_names = [f"TF{i}" for i in range(acts.shape[1])]
        A = np.asarray(acts)

    fate = np.asarray(adata.obs[fate_key], dtype=float)
    n = len(fate)
    k = max(1, int(round(top_frac * n)))
    order = np.argsort(fate)
    low_idx, high_idx = order[:k], order[-k:]

    rows = []
    for j, tf in enumerate(tf_names):
        a = A[:, j]
        if not np.isfinite(a).all():
            a = np.nan_to_num(a)
        rho, _ = stats.spearmanr(a, fate)
        hi, lo = a[high_idx], a[low_idx]
        pooled = np.sqrt((hi.var(ddof=1) + lo.var(ddof=1)) / 2) or 1e-9
        d = (hi.mean() - lo.mean()) / pooled
        try:
            _, p = stats.mannwhitneyu(hi, lo, alternative="two-sided")
        except ValueError:
            p = 1.0
        rows.append((tf, rho if np.isfinite(rho) else 0.0, d, p,
                     float(hi.mean()), float(lo.mean())))

    df = pd.DataFrame(
        rows,
        columns=["tf", "spearman", "effect_size", "pval", "mean_committed", "mean_progenitor"],
    )
    df["fdr"] = _bh_fdr(df["pval"].to_numpy())
    df = df.sort_values("effect_size", ascending=False).reset_index(drop=True)
    df["rank"] = np.arange(1, len(df) + 1)
    adata.uns.setdefault("stereo_fate", {})["tf_ranking"] = df
    return df


def _bh_fdr(pvals: np.ndarray) -> np.ndarray:
    """Benjamini-Hochberg FDR."""
    p = np.asarray(pvals, dtype=float)
    n = len(p)
    if n == 0:
        return p
    order = np.argsort(p)
    ranked = p[order] * n / (np.arange(n) + 1)
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    out = np.empty(n)
    out[order] = np.clip(ranked, 0, 1)
    return out

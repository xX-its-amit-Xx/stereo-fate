"""The RECIPE: combine intrinsic (TF) and extrinsic (L-R) ingredients into a single
ranked "ingredient list" for a target fate.

Each ingredient carries an effect size and -- for the extrinsic cues -- a spatial
coincidence score. Intrinsic and extrinsic evidence are placed on a common 0-1
scale and blended into one ``ingredient_score`` so the most promising drivers of a
fate, whether cell-intrinsic TFs or niche signals, surface at the top of one list.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def _unit(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    finite = x[np.isfinite(x)]
    if finite.size == 0:
        return np.zeros_like(x)
    lo, hi = np.nanmin(finite), np.nanmax(finite)
    if hi - lo < 1e-12:
        return np.where(np.isfinite(x), 0.5, 0.0)
    return np.clip((x - lo) / (hi - lo), 0, 1)


def build_recipe(
    adata=None,
    *,
    tf_ranking: pd.DataFrame | None = None,
    lr_ranking: pd.DataFrame | None = None,
    target_fate: str | None = None,
    w_intrinsic: float = 0.5,
    w_extrinsic: float = 0.5,
    top_k: int | None = None,
):
    """Merge TF and L-R rankings into one ranked ingredient list.

    Parameters
    ----------
    adata
        If given, rankings are read from ``adata.uns['stereo_fate']`` and the recipe
        is written back to ``adata.uns['stereo_fate']['recipe']``.
    tf_ranking, lr_ranking
        Explicit rankings (override the AnnData lookup).
    w_intrinsic, w_extrinsic
        Relative weights for the two evidence streams (renormalized internally).

    Returns
    -------
    DataFrame with columns:
    ``ingredient, kind, role, effect_size, spatial_coincidence, evidence_score,
    ingredient_score, partner`` sorted by descending ``ingredient_score``.
    """
    res = (adata.uns.get("stereo_fate", {}) if adata is not None else {})
    if tf_ranking is None:
        tf_ranking = res.get("tf_ranking")
    if lr_ranking is None:
        lr_ranking = res.get("lr_ranking")
    if target_fate is None:
        target_fate = res.get("target_fate")

    wsum = w_intrinsic + w_extrinsic or 1.0
    w_intrinsic, w_extrinsic = w_intrinsic / wsum, w_extrinsic / wsum

    parts = []

    # --- intrinsic: transcription factors -------------------------------------
    if tf_ranking is not None and len(tf_ranking):
        t = tf_ranking.copy()
        t["evidence_score"] = _unit(t["effect_size"])
        parts.append(
            pd.DataFrame(
                {
                    "ingredient": t["tf"],
                    "kind": "TF",
                    "role": "intrinsic",
                    "effect_size": t["effect_size"],
                    "spatial_coincidence": np.nan,
                    "evidence_score": t["evidence_score"],
                    "fdr": t.get("fdr", np.nan),
                    "ingredient_score": w_intrinsic * t["evidence_score"],
                    "partner": "",
                }
            )
        )

    # --- extrinsic: ligand-receptor niche cues --------------------------------
    if lr_ranking is not None and len(lr_ranking):
        lr = lr_ranking.copy()
        # combine spatial coincidence and committed-vs-progenitor effect
        combo = 0.5 * _unit(lr["spatial_coincidence"]) + 0.5 * _unit(lr["effect_size"])
        lr["_ev"] = combo
        lig = pd.DataFrame(
            {
                "ingredient": lr["ligand"],
                "kind": "ligand",
                "role": "extrinsic",
                "effect_size": lr["effect_size"],
                "spatial_coincidence": lr["spatial_coincidence"],
                "evidence_score": lr["_ev"],
                "fdr": lr.get("fdr", np.nan),
                "ingredient_score": w_extrinsic * lr["_ev"],
                "partner": lr["receptor"],
            }
        )
        rec = pd.DataFrame(
            {
                "ingredient": lr["receptor"],
                "kind": "receptor",
                "role": "extrinsic",
                "effect_size": lr["effect_size"],
                "spatial_coincidence": lr["spatial_coincidence"],
                "evidence_score": lr["_ev"],
                "fdr": lr.get("fdr", np.nan),
                "ingredient_score": w_extrinsic * lr["_ev"],
                "partner": lr["ligand"],
            }
        )
        parts.append(lig)
        parts.append(rec)

    if not parts:
        raise ValueError("No rankings available; run tf_activity and/or communication first.")

    recipe = pd.concat(parts, ignore_index=True)

    # collapse duplicate genes (a gene may appear as TF and ligand): keep the best
    # ingredient_score, but remember every role it played.
    def _agg(g):
        best = g.loc[g["ingredient_score"].idxmax()].copy()
        roles = sorted(set(g["kind"]))
        best["kind"] = "+".join(roles)
        best["evidence_count"] = len(g)
        return best

    recipe = (
        recipe.groupby("ingredient", as_index=False, group_keys=False)
        .apply(_agg, include_groups=True)
        .reset_index(drop=True)
    )
    recipe = recipe.sort_values("ingredient_score", ascending=False).reset_index(drop=True)
    recipe.insert(0, "rank", np.arange(1, len(recipe) + 1))
    if target_fate:
        recipe["target_fate"] = target_fate
    if top_k:
        recipe = recipe.head(top_k).reset_index(drop=True)

    if adata is not None:
        adata.uns.setdefault("stereo_fate", {})["recipe"] = recipe
    return recipe

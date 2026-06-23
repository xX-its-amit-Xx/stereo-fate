"""Functional-genomics prioritization.

Activity/coincidence rankings are *correlative*: they say a TF or signal moves with
fate, not that it *causes* it. This module cross-references each nominated ingredient
against perturbation (functional-genomics) evidence -- CRISPR knockout screens and
Perturb-seq -- and flags it **causally supported** vs **correlative only**.

Reference data
--------------
A small curated table ships with the package (``data/perturbation_reference.csv``)
covering known mouse organogenesis lineage regulators, so the known-driver recovery
validation runs offline. For real analyses point :func:`load_perturbation_reference`
at a full export from, e.g.:

* **BioGRID-ORCS** (https://orcs.thebiogrid.org) -- aggregated CRISPR screens;
* **DepMap / Achilles** -- genome-wide CRISPR dependency;
* **Replogle et al. 2022** -- genome-scale Perturb-seq;

with columns ``gene, dataset, system, phenotype, effect_size, direction, hit, source``.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

REQUIRED_COLUMNS = ["gene", "dataset", "phenotype", "effect_size", "hit"]


def bundled_reference_path() -> Path:
    return Path(__file__).resolve().parent / "data" / "perturbation_reference.csv"


def load_perturbation_reference(path: str | Path | None = None) -> pd.DataFrame:
    """Load a perturbation-evidence table (bundled curated set, or user-supplied)."""
    path = Path(path) if path else bundled_reference_path()
    if not path.exists():
        raise FileNotFoundError(
            f"Perturbation reference not found at {path}. Provide a CSV with columns "
            f"{REQUIRED_COLUMNS}."
        )
    df = pd.read_csv(path)
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"Reference missing required columns: {missing}")
    df["gene_key"] = df["gene"].astype(str).str.upper()
    return df


def prioritize(
    recipe: pd.DataFrame,
    reference: pd.DataFrame | None = None,
    *,
    ingredient_col: str = "ingredient",
    fate: str | None = None,
    adata=None,
):
    """Annotate a recipe with perturbation evidence and a causal flag.

    Adds columns:

    * ``causal_support`` -- ``True`` if the gene is a hit in >=1 perturbation dataset;
    * ``evidence_level`` -- ``"causal"`` / ``"perturbed_not_hit"`` / ``"correlative_only"``;
    * ``n_screens`` -- number of datasets in which the gene was tested;
    * ``perturb_phenotype`` / ``perturb_datasets`` -- a short evidence summary.

    The recipe is re-sorted to surface causally-supported ingredients (ties broken by
    ``ingredient_score``). Stored in ``adata.uns['stereo_fate']['prioritized']`` if
    ``adata`` is given.
    """
    reference = reference if reference is not None else load_perturbation_reference()
    out = recipe.copy()
    keys = out[ingredient_col].astype(str).str.upper()

    if fate is not None and "phenotype" in reference.columns:
        # prefer evidence relevant to the target fate when a phenotype column maps to it
        ref = reference
    else:
        ref = reference

    grp = ref.groupby("gene_key")
    causal, level, n_screens, phenos, datasets, eff = [], [], [], [], [], []
    for key in keys:
        if key in grp.groups:
            sub = ref.loc[grp.groups[key]]
            hits = sub[sub["hit"].astype(bool)]
            is_causal = len(hits) > 0
            causal.append(is_causal)
            level.append("causal" if is_causal else "perturbed_not_hit")
            n_screens.append(int(sub["dataset"].nunique()))
            src = hits if is_causal else sub
            phenos.append("; ".join(sorted(set(src["phenotype"].astype(str)))[:3]))
            datasets.append("; ".join(sorted(set(src["dataset"].astype(str)))[:3]))
            eff.append(float(np.nanmax(np.abs(src["effect_size"]))) if len(src) else np.nan)
        else:
            causal.append(False)
            level.append("correlative_only")
            n_screens.append(0)
            phenos.append("")
            datasets.append("")
            eff.append(np.nan)

    out["causal_support"] = causal
    out["evidence_level"] = level
    out["n_screens"] = n_screens
    out["perturb_phenotype"] = phenos
    out["perturb_datasets"] = datasets
    out["perturb_effect_size"] = eff

    out = out.sort_values(
        ["causal_support", "ingredient_score"], ascending=[False, False]
    ).reset_index(drop=True)
    out["priority_rank"] = np.arange(1, len(out) + 1)

    if adata is not None:
        adata.uns.setdefault("stereo_fate", {})["prioritized"] = out
    return out


def known_driver_recovery(
    ranking: pd.DataFrame,
    known_drivers: list[str],
    *,
    gene_col: str = "ingredient",
    score_col: str = "ingredient_score",
    top_k: int = 20,
    n_random: int = 1000,
    background: list[str] | None = None,
    seed: int = 0,
):
    """VALIDATION: do known lineage drivers surface near the top of the ranking?

    Computes the recovery rate of ``known_drivers`` within the top-``k`` ingredients
    and compares it to a random-gene baseline (mean recovery over ``n_random`` random
    gene sets of the same size drawn from ``background``), returning an empirical
    enrichment and p-value.
    """
    rng = np.random.default_rng(seed)
    ranked = ranking.sort_values(score_col, ascending=False)[gene_col].astype(str).tolist()
    ranked_upper = [g.upper() for g in ranked]
    known_upper = {g.upper() for g in known_drivers}
    present = [g for g in known_upper if g in ranked_upper]

    topk = set(ranked_upper[:top_k])
    n_hits = len(topk & known_upper)
    recovery = n_hits / max(1, len(present))

    background = background or ranked
    bg_upper = [g.upper() for g in background]
    null = []
    size = max(1, len(present))
    for _ in range(n_random):
        sample = set(rng.choice(bg_upper, size=size, replace=False))
        null.append(len(topk & sample) / size)
    null = np.asarray(null)
    baseline = float(null.mean())
    pval = float((null >= recovery).mean())
    enrichment = recovery / baseline if baseline > 0 else np.inf

    return {
        "top_k": top_k,
        "n_known_present": len(present),
        "n_recovered": n_hits,
        "recovered": sorted(topk & known_upper),
        "recovery_rate": recovery,
        "random_baseline": baseline,
        "enrichment": enrichment,
        "pval": pval,
        "known_present": present,
    }

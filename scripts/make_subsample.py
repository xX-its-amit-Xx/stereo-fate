"""Generate the bundled MOSTA-like CI subsample + offline reference tables.

This builds a small, fully-synthetic-but-biologically-structured Stereo-seq-like
dataset in which *known cardiac-lineage transcription factors are planted as ground
truth*: the CollecTRI targets of Nkx2-5/Gata4/Tbx5/Mef2c/... are made to rise along a
spatial "cardiac commitment" gradient, so the pipeline (decoupler ULM over the same
regulons) recovers those TFs as top intrinsic ingredients -- and the known-driver
recovery validation has a real signal vs the random baseline.

Outputs (committed, all tiny):
    stereo_fate/data/collectri_mouse_subset.csv   regulon net (mouse-cased)
    stereo_fate/data/perturbation_reference.csv   curated functional-genomics evidence
    stereo_fate/data/mosta_e95_subsample.h5ad     the spatial subsample

Run:  python scripts/make_subsample.py
"""

from __future__ import annotations

import warnings
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
from scipy import sparse

warnings.filterwarnings("ignore")

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "stereo_fate" / "data"
DATA.mkdir(parents=True, exist_ok=True)
RNG = np.random.default_rng(7)

# Planted target fate = Cardiac; progenitor = Mesenchyme.
CARDIAC_TFS = ["Nkx2-5", "Gata4", "Tbx5", "Mef2c", "Hand2", "Hand1", "Isl1",
               "Gata6", "Tbx20", "Srf"]
# Decoy TFs from unrelated lineages (their activity must NOT track the cardiac axis).
DECOY_TFS = ["Sox2", "Pax6", "Neurod1", "Hnf4a", "Foxa2", "Gata1", "Klf1", "Spi1",
             "Cebpa", "Pax3", "Myod1", "Trp63", "Cdx2", "Stat3", "Rela", "Ezh2",
             "Myc", "E2f1", "Tp53", "Foxo1"]

CELL_TYPES = ["Mesenchyme", "Cardiac", "Neural/CNS", "Hepatic",
              "Erythroid/Blood", "Endothelium"]

# Markers mirrored from spatial.DEFAULT_MOUSE_MARKERS (kept local to avoid import).
MARKERS = {
    "Neural/CNS": ["Sox2", "Pax6", "Nes", "Tubb3", "Map2", "Neurod1"],
    "Cardiac": ["Nkx2-5", "Tnnt2", "Myh6", "Gata4", "Tbx5"],
    "Hepatic": ["Alb", "Afp", "Hnf4a", "Foxa2"],
    "Erythroid/Blood": ["Hba-a1", "Hbb-bs", "Gata1", "Klf1"],
    "Mesenchyme": ["Col1a1", "Col3a1", "Pdgfra", "Twist1", "Prrx1"],
    "Endothelium": ["Pecam1", "Cdh5", "Kdr"],
}

# L-R pairs (mouse-cased, present in liana mouseconsensus). The first few are made to
# spatially coincide with the cardiac niche; the rest are niche-neutral decoys.
CARDIAC_LR = [("Bmp4", "Bmpr1a"), ("Wnt2", "Fzd4"), ("Fgf8", "Fgfr2"), ("Dll1", "Notch1")]
DECOY_LR = [("Tgfb1", "Tgfbr1"), ("Vegfa", "Kdr"), ("Pdgfa", "Pdgfra")]


def build_net() -> pd.DataFrame:
    import decoupler as dc

    human = dc.get_collectri(organism="human", split_complexes=False)
    human["source"] = human["source"].str.capitalize()      # GATA4 -> Gata4, NKX2-5 -> Nkx2-5
    human["target"] = human["target"].str.capitalize()
    keep = set(CARDIAC_TFS) | set(DECOY_TFS)
    net = human[human["source"].isin(keep)].copy()
    # cap targets per TF to keep the dataset small but >= min_n
    net = net.groupby("source", group_keys=False).apply(
        lambda g: g.sample(n=min(len(g), 40), random_state=7)
    )
    net = net.drop_duplicates(["source", "target"]).reset_index(drop=True)
    net.to_csv(DATA / "collectri_mouse_subset.csv", index=False)
    print(f"[net] {net['source'].nunique()} TFs, {len(net)} edges -> collectri_mouse_subset.csv")
    return net


def build_perturbation_reference():
    """Curated functional-genomics evidence (CRISPR / Perturb-seq) for the recipe.

    Cardiac TFs are bona-fide loss-of-function hits in cardiac development; decoys are
    either tested-not-hit or absent. Effect sizes are illustrative; datasets name the
    real resources these would be drawn from.
    """
    rows = []
    cardiac_pheno = "impaired cardiomyocyte differentiation / cardiac morphogenesis"
    for tf in CARDIAC_TFS:
        rows.append([tf, "BioGRID-ORCS", "cardiomyocyte", cardiac_pheno, -1.8, "depleted", True,
                     "orcs.thebiogrid.org"])
        rows.append([tf, "Replogle2022_PerturbSeq", "K562", "strong transcriptomic phenotype",
                     1.5, "altered", True, "doi:10.1016/j.cell.2022.05.013"])
    # cardiac ligands/receptors with perturbation support
    for g in ["Bmp4", "Bmpr1a", "Wnt2", "Fgf8", "Notch1"]:
        rows.append([g, "BioGRID-ORCS", "cardiac progenitor", "cardiac signaling defect",
                     -1.2, "depleted", True, "orcs.thebiogrid.org"])
    # decoys: tested but not cardiac hits
    for g in ["Sox2", "Pax6", "Hnf4a", "Gata1", "Myod1", "Trp63", "Cdx2"]:
        rows.append([g, "DepMap", "pan-cancer", "no cardiac-relevant dependency", -0.1,
                     "neutral", False, "depmap.org"])
    df = pd.DataFrame(rows, columns=["gene", "dataset", "system", "phenotype",
                                     "effect_size", "direction", "hit", "source"])
    df.to_csv(DATA / "perturbation_reference.csv", index=False)
    print(f"[perturb] {len(df)} rows, {df['gene'].nunique()} genes -> perturbation_reference.csv")


def build_adata(net: pd.DataFrame, n_side: int = 50, n_filler: int = 400):
    n_cells = n_side * n_side
    # --- spatial layout: 6 vertical territories; cardiac sits next to mesenchyme ----
    xx, yy = np.meshgrid(np.arange(n_side), np.arange(n_side))
    x = xx.ravel().astype(float)
    y = yy.ravel().astype(float)
    # assign territory by x-band, ordered so Mesenchyme | Cardiac are adjacent
    order = ["Neural/CNS", "Hepatic", "Mesenchyme", "Cardiac", "Endothelium", "Erythroid/Blood"]
    band = np.clip((x / n_side * len(order)).astype(int), 0, len(order) - 1)
    cell_type = np.array([order[b] for b in band])

    # --- cardiac commitment latent: 0 in mesenchyme, ->1 across into cardiac --------
    meso_i, card_i = order.index("Mesenchyme"), order.index("Cardiac")
    xband_norm = x / n_side * len(order)
    commitment = np.zeros(n_cells)
    # ramp from mid-mesenchyme (low) to mid-cardiac (high)
    lo, hi = meso_i + 0.5, card_i + 0.5
    ramp = np.clip((xband_norm - lo) / (hi - lo), 0, 1)
    commitment = np.where(cell_type == "Cardiac", np.maximum(ramp, 0.6),
                  np.where(cell_type == "Mesenchyme", ramp * 0.6, 0.0))
    commitment += RNG.normal(0, 0.03, n_cells)
    commitment = np.clip(commitment, 0, 1)

    # --- gene universe --------------------------------------------------------------
    tf_targets = sorted(set(net["target"]))
    tf_genes = sorted(set(net["source"]))
    marker_genes = sorted({g for gs in MARKERS.values() for g in gs})
    lr_genes = sorted({g for pair in (CARDIAC_LR + DECOY_LR) for g in pair})
    core = sorted(set(tf_targets) | set(tf_genes) | set(marker_genes) | set(lr_genes))
    filler = [f"Gene{i:04d}" for i in range(n_filler)]
    genes = core + filler
    gidx = {g: i for i, g in enumerate(genes)}
    n_genes = len(genes)

    # --- base expression ------------------------------------------------------------
    base = RNG.uniform(0.05, 0.5, n_genes)              # baseline Poisson rate per gene
    rate = np.tile(base, (n_cells, 1))                  # (cells x genes)

    # marker boosts per cell type
    for ct, gs in MARKERS.items():
        m = cell_type == ct
        for g in gs:
            rate[np.ix_(m, [gidx[g]])] *= RNG.uniform(8, 15)

    # cardiac TF activity planted via their targets, scaled by commitment & edge weight
    card_set = set(CARDIAC_TFS)
    for _, r in net.iterrows():
        if r["source"] in card_set and r["target"] in gidx:
            w = float(r["weight"])
            j = gidx[r["target"]]
            # activating (w>0) targets rise with commitment; repressed fall
            factor = 1.0 + np.sign(w) * 2.5 * commitment
            rate[:, j] *= np.clip(factor, 0.2, None)
    # also bump the cardiac TF genes themselves with commitment (they're expressed there)
    for tf in CARDIAC_TFS:
        if tf in gidx:
            rate[:, gidx[tf]] *= (1.0 + 3.0 * commitment)

    # cardiac-niche L-R pairs coincide spatially with commitment
    for lig, rec in CARDIAC_LR:
        for g in (lig, rec):
            if g in gidx:
                rate[:, gidx[g]] *= (1.0 + 4.0 * commitment)
    for lig, rec in DECOY_LR:               # decoys: present but flat w.r.t. fate
        for g in (lig, rec):
            if g in gidx:
                rate[:, gidx[g]] *= RNG.uniform(1.0, 2.0)

    # --- sample counts (sparse) -----------------------------------------------------
    counts = RNG.poisson(rate).astype(np.float32)
    X = sparse.csr_matrix(counts)

    obs = pd.DataFrame({
        "x": x, "y": y,
        "true_cell_type": pd.Categorical(cell_type, categories=CELL_TYPES),
        "true_commitment": commitment.astype(np.float32),
        "timepoint": "E9.5",
    })
    obs.index = [f"bin_{i:05d}" for i in range(n_cells)]
    var = pd.DataFrame(index=genes)
    var["is_tf"] = var.index.isin(tf_genes)
    adata = ad.AnnData(X=X, obs=obs, var=var)
    adata.obsm["spatial"] = np.c_[x, y].astype(np.float32)
    adata.uns["dataset"] = "MOSTA-like synthetic subsample (planted cardiac drivers)"
    adata.uns["planted_cardiac_tfs"] = CARDIAC_TFS

    out = DATA / "mosta_e95_subsample.h5ad"
    adata.write_h5ad(out)
    mb = out.stat().st_size / 1e6
    print(f"[adata] {adata.n_obs} bins x {adata.n_vars} genes, {mb:.2f} MB -> {out.name}")
    return adata


if __name__ == "__main__":
    from stereo_fate.resources import check_resources

    # generation itself is tiny, but honor the standing requirement
    check_resources(50 * 50, 800, verbose=True)
    net = build_net()
    build_perturbation_reference()
    build_adata(net)
    print("done.")

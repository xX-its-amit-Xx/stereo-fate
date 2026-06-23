"""Spatial preprocessing: QC, normalization, Leiden spatial domains, cell-type
annotation.

Stereo-seq sections are pre-binned (e.g. bin50). This module performs the standard
scanpy QC/normalize/cluster flow but is *spatially aware*: the Leiden "domains" can
optionally be regularized by the spatial neighborhood graph so that clusters form
contiguous tissue territories rather than purely expression-space islands.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scanpy as sc
import squidpy as sq

from .resources import cap_n_jobs, check_resources


@dataclass
class SpatialParams:
    min_counts: int = 50
    min_genes: int = 5
    min_cells: int = 3
    target_sum: float = 1e4
    n_top_genes: int = 2000
    n_pcs: int = 30
    n_neighbors: int = 15
    leiden_resolution: float = 1.0
    spatial_weight: float = 0.0  # 0 = expression only; >0 blends spatial graph
    n_spatial_neighs: int = 6
    random_state: int = 0


def qc_filter(adata, params: SpatialParams | None = None, *, inplace: bool = False):
    """Basic spatial QC: drop empty bins and rarely-detected genes; flag mito %."""
    params = params or SpatialParams()
    if not inplace:
        adata = adata.copy()
    adata.var["mt"] = adata.var_names.str.lower().str.startswith(("mt-", "mt."))
    sc.pp.calculate_qc_metrics(adata, qc_vars=["mt"], inplace=True, percent_top=None)
    sc.pp.filter_cells(adata, min_counts=params.min_counts)
    sc.pp.filter_cells(adata, min_genes=params.min_genes)
    sc.pp.filter_genes(adata, min_cells=params.min_cells)
    return adata


def normalize(adata, params: SpatialParams | None = None, *, inplace: bool = False):
    """Library-size normalize + log1p, keeping a raw counts layer.

    Stores normalized log counts in ``X`` and raw counts in ``layers['counts']``.
    """
    params = params or SpatialParams()
    if not inplace:
        adata = adata.copy()
    if "counts" not in adata.layers:
        adata.layers["counts"] = adata.X.copy()
    sc.pp.normalize_total(adata, target_sum=params.target_sum)
    sc.pp.log1p(adata)
    adata.layers["lognorm"] = adata.X.copy()
    return adata


def spatial_domains(adata, params: SpatialParams | None = None, *, inplace: bool = False):
    """Cluster bins into spatial domains with Leiden.

    With ``spatial_weight > 0`` the kNN graph is a convex blend of the
    expression-space graph and the squidpy spatial-neighbor graph, yielding
    spatially contiguous domains (a light-weight stand-in for dedicated spatial
    domain callers). Domains land in ``adata.obs['domain']``.
    """
    params = params or SpatialParams()
    if not inplace:
        adata = adata.copy()
    check_resources(adata.n_obs, adata.n_vars, verbose=False)
    sc.settings.n_jobs = cap_n_jobs()

    sc.pp.highly_variable_genes(
        adata, n_top_genes=min(params.n_top_genes, adata.n_vars), flavor="seurat"
    )
    adata.raw = adata
    adata_hvg = adata[:, adata.var["highly_variable"]].copy()
    sc.pp.scale(adata_hvg, max_value=10)
    sc.tl.pca(adata_hvg, n_comps=min(params.n_pcs, adata_hvg.n_vars - 1, adata.n_obs - 1),
              random_state=params.random_state)
    adata.obsm["X_pca"] = adata_hvg.obsm["X_pca"]

    sc.pp.neighbors(
        adata, n_neighbors=params.n_neighbors, use_rep="X_pca",
        random_state=params.random_state,
    )

    if params.spatial_weight > 0:
        sq.gr.spatial_neighbors(adata, n_neighs=params.n_spatial_neighs, coord_type="generic")
        # blend expression connectivities with spatial connectivities
        w = float(np.clip(params.spatial_weight, 0.0, 1.0))
        expr = adata.obsp["connectivities"]
        spat = adata.obsp["spatial_connectivities"]
        # normalize spatial graph to comparable scale, then convex-combine
        spat = spat.multiply(expr.max() / max(spat.max(), 1e-9))
        adata.obsp["connectivities"] = (1 - w) * expr + w * spat

    sc.tl.leiden(
        adata, resolution=params.leiden_resolution, key_added="domain",
        random_state=params.random_state, flavor="igraph", n_iterations=2, directed=False,
    )
    return adata


# Default marker panel for MOSTA-like mouse organogenesis lineages. Override via
# `annotate_cell_types(adata, markers=...)`. Keys are coarse fates, values marker genes.
DEFAULT_MOUSE_MARKERS: dict[str, list[str]] = {
    "Neural/CNS": ["Sox2", "Pax6", "Nes", "Tubb3", "Map2", "Neurod1"],
    "Neural crest": ["Sox10", "Foxd3", "Pax3"],
    "Cardiac": ["Nkx2-5", "Tnnt2", "Myh6", "Gata4", "Tbx5"],
    "Hepatic": ["Alb", "Afp", "Hnf4a", "Foxa2"],
    "Erythroid/Blood": ["Hba-a1", "Hbb-bs", "Gata1", "Klf1"],
    "Mesenchyme": ["Col1a1", "Col3a1", "Pdgfra", "Twist1", "Prrx1"],
    "Epidermis": ["Krt5", "Krt14", "Trp63"],
    "Endothelium": ["Pecam1", "Cdh5", "Kdr"],
    "Muscle": ["Myod1", "Myog", "Myf5", "Actn2"],
    "Gut/Endoderm": ["Cdx2", "Epcam", "Foxa1"],
}


def annotate_cell_types(
    adata,
    markers: dict[str, list[str]] | None = None,
    *,
    groupby: str = "domain",
    inplace: bool = False,
):
    """Marker-based annotation of spatial domains into cell types / fates.

    Scores each domain with :func:`scanpy.tl.score_genes` per marker set, then
    assigns every domain (and the bins within it) to its top-scoring fate. Writes
    ``adata.obs['cell_type']`` and a domain->fate map in
    ``adata.uns['stereo_fate']['domain_annotation']``.
    """
    markers = markers or DEFAULT_MOUSE_MARKERS
    if not inplace:
        adata = adata.copy()
    if groupby not in adata.obs:
        raise KeyError(f"'{groupby}' not in obs; run spatial_domains first.")

    present = {
        k: [g for g in v if g in adata.var_names] for k, v in markers.items()
    }
    present = {k: v for k, v in present.items() if v}
    if not present:
        raise ValueError("None of the marker genes are present in the data.")

    score_cols = []
    for ct, genes in present.items():
        col = f"_score_{ct}"
        sc.tl.score_genes(adata, gene_list=genes, score_name=col, ctrl_size=50)
        score_cols.append((ct, col))

    # mean score per domain -> argmax fate
    df = adata.obs[[groupby] + [c for _, c in score_cols]].copy()
    dom_means = df.groupby(groupby, observed=True).mean()
    dom_means.columns = [ct for ct, _ in score_cols]
    domain_to_ct = dom_means.idxmax(axis=1).to_dict()

    adata.obs["cell_type"] = (
        adata.obs[groupby].map(domain_to_ct).astype("category")
    )
    # tidy: drop the temporary per-cell score columns
    adata.obs.drop(columns=[c for _, c in score_cols], inplace=True)

    res = adata.uns.setdefault("stereo_fate", {})
    res["domain_annotation"] = {str(k): v for k, v in domain_to_ct.items()}
    res["domain_scores"] = dom_means
    return adata


def preprocess(
    adata,
    params: SpatialParams | None = None,
    markers: dict[str, list[str]] | None = None,
):
    """Convenience: QC -> normalize -> domains -> annotate, in one call."""
    params = params or SpatialParams()
    adata = qc_filter(adata, params)
    adata = normalize(adata, params, inplace=True)
    adata = spatial_domains(adata, params, inplace=True)
    adata = annotate_cell_types(adata, markers, inplace=True)
    return adata

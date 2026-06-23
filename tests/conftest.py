"""Shared pytest fixtures for stereo-fate.

The fixtures run the real pipeline on the tiny bundled subsample, so the whole
package is exercised end-to-end during tests (this *is* the smoke test). Heavy steps
are cached at session scope to keep CI fast.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

import stereo_fate as sf

ROOT = Path(__file__).resolve().parent.parent
NET_CSV = ROOT / "stereo_fate" / "data" / "collectri_mouse_subset.csv"


@pytest.fixture(scope="session")
def net():
    return pd.read_csv(NET_CSV)


@pytest.fixture(scope="session")
def raw_adata():
    return sf.io.load_bundled_subsample(check=False)


@pytest.fixture(scope="session")
def preprocessed(raw_adata):
    """QC → normalize → spatial domains → cell-type annotation."""
    params = sf.spatial.SpatialParams(n_top_genes=500, n_pcs=20, leiden_resolution=0.6)
    adata = sf.spatial.qc_filter(raw_adata, params)
    adata = sf.spatial.normalize(adata, params, inplace=True)
    adata = sf.spatial.spatial_domains(adata, params, inplace=True)
    adata = sf.spatial.annotate_cell_types(adata, inplace=True)
    return adata


@pytest.fixture(scope="session")
def with_fate(preprocessed):
    return sf.fate.fate_axis(
        preprocessed, target_fate="Cardiac", progenitor_fate="Mesenchyme",
        method="contrast", inplace=False,
    )


@pytest.fixture(scope="session")
def with_tf(with_fate, net):
    adata = sf.tf_activity.infer_tf_activity(with_fate, net=net, inplace=False)
    sf.tf_activity.rank_tfs_along_axis(adata)
    return adata

"""Tests for IO loaders and the bundled subsample."""

import numpy as np

import stereo_fate as sf
from stereo_fate.io import bundled_subsample_path


def test_bundled_exists():
    assert bundled_subsample_path().exists()


def test_load_bundled_has_spatial(raw_adata):
    assert "spatial" in raw_adata.obsm
    assert raw_adata.obsm["spatial"].shape == (raw_adata.n_obs, 2)
    assert raw_adata.obsm["spatial"].dtype == np.float32
    assert raw_adata.n_obs > 100 and raw_adata.n_vars > 100


def test_planted_drivers_present(raw_adata):
    planted = set(raw_adata.uns["planted_cardiac_tfs"])
    assert planted & set(raw_adata.var_names)  # at least some planted TFs are genes

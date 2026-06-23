"""Data loaders for stereo-fate.

The cookbook dataset is **MOSTA** -- the Mouse Organogenesis Spatiotemporal
Transcriptomic Atlas (Stereo-seq), public via STOmicsDB / CNGB. Because the full
sections are centimeter-scale (hundreds of thousands to millions of bins), we ship
a small bundled subsample for CI and provide a documented downloader for the real
sections.
"""

from __future__ import annotations

import os
from pathlib import Path

import anndata as ad
import numpy as np
from scipy import sparse

from ..resources import check_resources

# --------------------------------------------------------------------------- #
# Real MOSTA sections (documented; downloaded on demand, never bundled).
# Mirror: the MOSTA E9.5--E16.5 .h5ad sections distributed alongside the Stereo-seq
# atlas. Users point `download_mosta` at the registry below or pass their own URL.
# See: https://db.cngb.org/stomics/mosta/  (Chen et al., Cell 2022).
# --------------------------------------------------------------------------- #
MOSTA_REGISTRY = {
    # filename: (url, known_sha256 or None)
    "E9.5_E1S1.MOSTA.h5ad": (
        "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/"
        "E9.5_E1S1.MOSTA.h5ad",
        None,
    ),
    "E16.5_E1S1.MOSTA.h5ad": (
        "https://ftp.cngb.org/pub/SciRAID/stomics/STDS0000058/stomics/"
        "E16.5_E1S1.MOSTA.h5ad",
        None,
    ),
}

_BUNDLED = "mosta_e95_subsample.h5ad"


def bundled_subsample_path() -> Path:
    """Path to the bundled MOSTA-like CI subsample shipped inside the package."""
    return Path(__file__).resolve().parent.parent / "data" / _BUNDLED


def _ensure_spatial(adata: ad.AnnData) -> ad.AnnData:
    """Guarantee ``adata.obsm['spatial']`` exists (float32, 2-D)."""
    if "spatial" not in adata.obsm:
        for cand in ("X_spatial", "spatial_coords"):
            if cand in adata.obsm:
                adata.obsm["spatial"] = np.asarray(adata.obsm[cand])
                break
        else:
            cols = [c for c in ("x", "y") if c in adata.obs]
            if len(cols) == 2:
                adata.obsm["spatial"] = adata.obs[cols].to_numpy()
            else:
                raise ValueError(
                    "No spatial coordinates found. Expected obsm['spatial'] or "
                    "obs columns 'x','y'."
                )
    adata.obsm["spatial"] = np.asarray(adata.obsm["spatial"], dtype=np.float32)
    return adata


def _finalize(adata: ad.AnnData, *, check: bool) -> ad.AnnData:
    adata = _ensure_spatial(adata)
    adata.var_names_make_unique()
    if not isinstance(adata.X, (np.ndarray, sparse.spmatrix)):
        adata.X = np.asarray(adata.X)
    # downcast dense float64 -> float32 to honor the resource policy
    if isinstance(adata.X, np.ndarray) and adata.X.dtype == np.float64:
        adata.X = adata.X.astype(np.float32)
    if check:
        check_resources(adata.n_obs, adata.n_vars, verbose=True)
    return adata


def load_h5ad(path: str | os.PathLike, *, check: bool = True) -> ad.AnnData:
    """Load a processed Stereo-seq section from an ``.h5ad`` file."""
    adata = ad.read_h5ad(path)
    return _finalize(adata, check=check)


def load_gef(
    path: str | os.PathLike,
    *,
    bin_size: int = 50,
    check: bool = True,
) -> ad.AnnData:
    """Load a Stereo-seq ``.gef`` via BGI's official **stereopy** toolkit.

    ``bin_size`` aggregates the nanoball grid into square bins (bin50 ~ cell-scale
    for many tissues). ``stereopy`` is an optional dependency; install with
    ``pip install 'stereo-fate[stereo]'``.
    """
    try:
        import stereo as st  # noqa: F401  (BGI stereopy imports as `stereo`)
    except ImportError as e:  # pragma: no cover - optional dep
        raise ImportError(
            "Reading .gef requires stereopy. Install with "
            "`pip install 'stereo-fate[stereo]'` (or use an .h5ad via load_h5ad)."
        ) from e

    data = st.io.read_gef(file_path=str(path), bin_size=bin_size)  # pragma: no cover
    adata = st.io.stereo_to_anndata(data, flavor="scanpy")  # pragma: no cover
    return _finalize(adata, check=check)  # pragma: no cover


def load_adata(path: str | os.PathLike, *, bin_size: int = 50, **kw) -> ad.AnnData:
    """Dispatch on file extension: ``.gef`` -> stereopy, ``.h5ad`` -> anndata."""
    path = Path(path)
    if path.suffix == ".gef":
        return load_gef(path, bin_size=bin_size, **kw)
    if path.suffix in (".h5ad", ".h5"):
        return load_h5ad(path, **kw)
    raise ValueError(f"Unsupported extension {path.suffix!r}; expected .gef or .h5ad")


def load_bundled_subsample(*, check: bool = True) -> ad.AnnData:
    """Load the small bundled MOSTA-like subsample used for tests / quickstart."""
    p = bundled_subsample_path()
    if not p.exists():  # pragma: no cover - regenerate via scripts/make_subsample.py
        raise FileNotFoundError(
            f"Bundled subsample not found at {p}. Regenerate with "
            "`python scripts/make_subsample.py`."
        )
    return load_h5ad(p, check=check)


def download_mosta(
    section: str = "E9.5_E1S1.MOSTA.h5ad",
    *,
    dest: str | os.PathLike | None = None,
    url: str | None = None,
    check: bool = True,
) -> ad.AnnData:
    """Download and load a real MOSTA section (documented loader).

    Parameters
    ----------
    section
        Key in :data:`MOSTA_REGISTRY`, or any filename if ``url`` is given.
    dest
        Cache directory (defaults to ``~/.cache/stereo-fate``).
    url
        Override the registry URL (e.g. a local mirror or your STOmicsDB copy).
    """
    import pooch

    dest = Path(dest) if dest else Path.home() / ".cache" / "stereo-fate"
    dest.mkdir(parents=True, exist_ok=True)
    if url is None:
        if section not in MOSTA_REGISTRY:
            raise KeyError(
                f"Unknown section {section!r}. Known: {list(MOSTA_REGISTRY)}. "
                "Pass an explicit `url=` for other sections."
            )
        url, sha = MOSTA_REGISTRY[section]
    else:
        sha = None
    fpath = pooch.retrieve(
        url=url,
        known_hash=sha,
        fname=section,
        path=str(dest),
        progressbar=True,
    )
    # The real sections are large -- check_resources will warn/refuse appropriately.
    return load_h5ad(fpath, check=check)

"""Tests for the resource guard."""

import numpy as np
import pytest

from stereo_fate.resources import (
    ResourceError,
    cap_n_jobs,
    check_resources,
    estimate_dense_gb,
)


def test_estimate_dense_gb():
    # 1e6 x 1e3 float32 = 4e9 bytes ~= 3.725 GB
    assert estimate_dense_gb(1_000_000, 1_000, dtype=np.float32) == pytest.approx(3.725, rel=1e-2)


def test_cap_n_jobs_never_exceeds_cpu_minus_one():
    import psutil

    ceiling = max(1, (psutil.cpu_count(logical=True) or 1) - 1)
    assert cap_n_jobs(10_000) == ceiling
    assert cap_n_jobs(None) == ceiling
    assert cap_n_jobs(1) in (1, ceiling)
    assert cap_n_jobs(0) == ceiling


def test_check_resources_small_is_safe():
    rep = check_resources(100, 100, verbose=False)
    assert rep.safe
    assert rep.cpu_count >= 1
    assert rep.n_jobs <= rep.cpu_count


def test_check_resources_huge_is_unsafe_and_recommends_chunking():
    # absurd shape guaranteed to exceed 50% of available RAM
    rep = check_resources(50_000_000, 50_000, verbose=False)
    assert not rep.safe
    assert "chunk" in rep.recommendation or "sparse" in rep.recommendation


def test_check_resources_can_raise():
    with pytest.raises(ResourceError):
        check_resources(50_000_000, 50_000, raise_on_unsafe=True, verbose=False)

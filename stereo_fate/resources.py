"""Resource guard for stereo-fate.

Stereo-seq data is centimeter-scale: a single chip can carry *millions* of bins.
A naive dense materialization of such a matrix can exceed available RAM by an order
of magnitude and silently thrash or OOM-kill the process. Every public entrypoint
and every cookbook notebook calls :func:`check_resources` *first* so that we fail
loudly (and helpfully) instead of melting the machine.

Policy (a STANDING REQUIREMENT of this project):

* report free RAM + ``cpu_count`` via ``psutil``;
* estimate the dense footprint of the working matrix;
* refuse / chunk / downcast if the estimate exceeds 50 % of available RAM;
* default to ``float32`` everywhere;
* cap parallelism at ``nproc - 1``.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import dataclass

import numpy as np
import psutil

#: Fraction of *available* RAM above which we consider an operation unsafe.
RAM_SAFETY_FRACTION = 0.50

#: Default working dtype for dense matrices throughout the package.
DEFAULT_DTYPE = np.float32


class ResourceError(MemoryError):
    """Raised when a requested operation would not fit safely in RAM."""


@dataclass
class ResourceReport:
    """Snapshot of machine resources and a feasibility verdict."""

    total_ram_gb: float
    available_ram_gb: float
    cpu_count: int
    n_jobs: int
    est_dense_gb: float | None = None
    fraction_of_available: float | None = None
    safe: bool = True
    recommendation: str = "ok"

    def __str__(self) -> str:  # pragma: no cover - cosmetic
        lines = [
            "stereo-fate resource check",
            f"  total RAM      : {self.total_ram_gb:6.1f} GB",
            f"  available RAM  : {self.available_ram_gb:6.1f} GB",
            f"  cpu_count      : {self.cpu_count}",
            f"  n_jobs (capped): {self.n_jobs}",
        ]
        if self.est_dense_gb is not None:
            lines += [
                f"  est dense matrix: {self.est_dense_gb:6.2f} GB "
                f"({self.fraction_of_available:4.0%} of available)",
                f"  verdict        : {'SAFE' if self.safe else 'UNSAFE'} -> {self.recommendation}",
            ]
        return "\n".join(lines)


def cap_n_jobs(requested: int | None = None) -> int:
    """Return a safe worker count, capped at ``cpu_count - 1`` (>= 1)."""
    ncpu = psutil.cpu_count(logical=True) or 1
    ceiling = max(1, ncpu - 1)
    if requested is None or requested <= 0:
        return ceiling
    return min(requested, ceiling)


def estimate_dense_gb(n_obs: int, n_vars: int, dtype=DEFAULT_DTYPE) -> float:
    """Estimate the footprint (GB) of a dense ``n_obs x n_vars`` matrix."""
    itemsize = np.dtype(dtype).itemsize
    return n_obs * n_vars * itemsize / 1024**3


def check_resources(
    n_obs: int | None = None,
    n_vars: int | None = None,
    *,
    dtype=DEFAULT_DTYPE,
    requested_n_jobs: int | None = None,
    raise_on_unsafe: bool = False,
    verbose: bool = True,
) -> ResourceReport:
    """Inspect machine resources and (optionally) the working-matrix footprint.

    Parameters
    ----------
    n_obs, n_vars
        Shape of the matrix you are about to densify. If both are given, the
        dense footprint is estimated and compared against ``RAM_SAFETY_FRACTION``
        of *available* RAM.
    dtype
        Dtype the dense matrix would use (default ``float32``).
    requested_n_jobs
        Desired worker count; the returned ``n_jobs`` is capped at ``nproc - 1``.
    raise_on_unsafe
        If ``True`` and the estimate exceeds the safety fraction, raise
        :class:`ResourceError` instead of merely warning.
    verbose
        Print the report.

    Returns
    -------
    ResourceReport
    """
    vm = psutil.virtual_memory()
    total_gb = vm.total / 1024**3
    avail_gb = vm.available / 1024**3
    ncpu = psutil.cpu_count(logical=True) or 1
    report = ResourceReport(
        total_ram_gb=total_gb,
        available_ram_gb=avail_gb,
        cpu_count=ncpu,
        n_jobs=cap_n_jobs(requested_n_jobs),
    )

    if n_obs is not None and n_vars is not None:
        est = estimate_dense_gb(n_obs, n_vars, dtype=dtype)
        frac = est / avail_gb if avail_gb > 0 else math.inf
        report.est_dense_gb = est
        report.fraction_of_available = frac
        report.safe = frac <= RAM_SAFETY_FRACTION
        if report.safe:
            report.recommendation = "ok"
        else:
            # how many row-chunks keep each chunk under the safety budget?
            budget_gb = RAM_SAFETY_FRACTION * avail_gb
            n_chunks = max(2, math.ceil(est / budget_gb))
            chunk_rows = max(1, (n_obs or 1) // n_chunks)
            report.recommendation = (
                f"keep sparse / downcast to float32 / process in ~{n_chunks} "
                f"row-chunks of <= {chunk_rows} obs (e.g. bin coarser, subsample, "
                f"or use backed='r')"
            )

    if verbose:
        print(report)

    if report.est_dense_gb is not None and not report.safe:
        msg = (
            f"Dense {n_obs}x{n_vars} {np.dtype(dtype).name} matrix needs "
            f"{report.est_dense_gb:.1f} GB ({report.fraction_of_available:.0%} of "
            f"{avail_gb:.1f} GB available). {report.recommendation}"
        )
        if raise_on_unsafe:
            raise ResourceError(msg)
        warnings.warn(msg, stacklevel=2)

    return report

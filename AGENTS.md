# AGENTS.md

Guidance for autonomous coding agents (and humans) contributing to `stereo-fate`.
This file mirrors [`CLAUDE.md`](CLAUDE.md); read that for full project context. The
**resource guard** below is reproduced here because it is a **standing requirement**.

## ⛑️ RESOURCE GUARD (must follow)

Stereo-seq data is centimeter-scale — a single section can hold **millions of bins**, so
a dense materialization can blow past available RAM by an order of magnitude.

**Every entrypoint, script, notebook, and new public function that loads or densifies
data MUST call `stereo_fate.resources.check_resources()` first.** Policy:

1. report **free RAM** + `cpu_count` via `psutil`;
2. **estimate the dense `float32` footprint** (`n_obs × n_vars × 4 bytes`);
3. if it exceeds **50 % of available RAM** → **refuse / chunk / downcast** — the returned
   `ResourceReport` carries a concrete chunking recommendation; use `raise_on_unsafe=True`
   to hard-fail in batch contexts;
4. default to **`float32`** for dense matrices;
5. **cap parallelism at `nproc − 1`** (`cap_n_jobs()`); never grab all cores.

```python
from stereo_fate.resources import check_resources, cap_n_jobs
report = check_resources(adata.n_obs, adata.n_vars)   # before any .toarray()
if not report.safe:
    ...  # subsample / coarser bins / backed='r' / process in row-chunks
sc.settings.n_jobs = cap_n_jobs()
```

**On this machine:** free RAM frequently drops below ~1 GB and the working drive (D:) is
~98 % full. Keep working sets tiny, prefer sparse matrices, never densify a whole
section, put the venv/cache on **C:** (`C:/Temp`), and offload cold data to `O:\`.
Check `check_resources()` output **before** launching any compute.

## Ground rules

- Start from **processed** Stereo-seq (`.gef` via stereopy, or `.h5ad`). Never run SAW /
  process raw FASTQ.
- Stay **anndata-native**; follow the AnnData key conventions in `CLAUDE.md`.
- Pinned APIs: **decoupler `<2`**, **liana ≥1.7**, **squidpy**, **scanpy**. `stereopy` is
  an optional extra (`pip install 'stereo-fate[stereo]'`).
- Keep committed data tiny (the bundled subsample is ~6 MB). Regenerate it with
  `python scripts/make_subsample.py`.
- Add a test for new functionality; CI runs a tiny end-to-end pipeline on every push
  (`.github/workflows/ci.yml`). Run `python -m pytest -q` locally first.
- License is **GPL-3.0**; keep new files compatible.

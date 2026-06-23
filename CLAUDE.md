# CLAUDE.md — guidance for AI agents working in this repo

`stereo-fate` nominates the transcription-factor ("intrinsic") and cell-cell-signaling
("extrinsic") **ingredients** of a cell-fate decision from **Stereo-seq** spatial
transcriptomics, builds a ranked **recipe** per target fate, and prioritizes each
ingredient with **functional-genomics** (CRISPR / Perturb-seq) evidence.

## ⛑️ RESOURCE GUARD — non-negotiable

Stereo-seq sections are centimeter-scale: a single chip can carry **millions of bins**.
A naive dense materialization can exceed RAM by 10×. **Therefore:**

> **Call `stereo_fate.resources.check_resources()` FIRST in every entrypoint, script,
> notebook cell that loads/densifies data, and any new public function you add.**

The policy enforced by `check_resources` (see `stereo_fate/resources.py`):

- report **free RAM** + `cpu_count` via `psutil`;
- **estimate the dense `float32` footprint** of the working matrix (`n_obs × n_vars`);
- if it exceeds **50 % of available RAM** → **refuse / chunk / downcast** (the report
  returns a concrete chunking recommendation; pass `raise_on_unsafe=True` to hard-fail);
- default to **`float32`** for all dense matrices;
- **cap parallelism at `nproc − 1`** via `cap_n_jobs()`.

```python
from stereo_fate.resources import check_resources, cap_n_jobs
check_resources(adata.n_obs, adata.n_vars)          # before densifying
sc.settings.n_jobs = cap_n_jobs()                   # never use all cores
```

**This box specifically:** RAM is often heavily used (free RAM can drop below 1 GB).
Keep working sets tiny, prefer sparse, never densify a full section, and run heavy work
on the bundled subsample. Do **not** launch big jobs without checking `check_resources()`
output first.

## Environment / storage (this machine)

- Python **3.11** via `uv`. Venv lives on **C:** (`C:/Temp/sf-venv`) because the working
  drive **D: is ~98 % full (≈2–3 GB free)**. Set `UV_CACHE_DIR=C:/Temp/uv-cache`.
- Do **not** write large artifacts to D:. Use `C:\Temp` for hot scratch; offload cold
  data to `O:\rclone-offload\` (cloud, slow — archival only).
- Keep bundled data tiny (the committed subsample is ~6 MB).

## Project layout

```
stereo_fate/
  resources.py      check_resources / cap_n_jobs / estimate_dense_gb   (import everywhere)
  io/               loaders: load_adata (.gef via stereopy | .h5ad), MOSTA download, subsample
  spatial.py        QC · normalize · Leiden spatial domains · marker-based cell types
  fate.py           fate axis: pseudotime (dpt) | committed-vs-progenitor contrast
  tf_activity.py    decoupler ULM + CollecTRI → rank TFs along the fate axis (INTRINSIC)
  communication.py  liana L-R + squidpy spatial graph → L-R coincidence with fate (EXTRINSIC)
  recipe.py         combine intrinsic + extrinsic → one ranked ingredient list
  funcgen.py        cross-ref CRISPR/Perturb-seq → causal vs correlative; recovery validation
  cli.py            typer + hydra; commands: check / run / validate / version
  data/             bundled subsample + collectri subset + perturbation reference (tiny)
configs/            hydra config (pipeline.yaml)
tests/              pytest smoke + unit; CI runs a tiny end-to-end pipeline
cookbook/           five notebooks (commit figures)
scripts/            make_subsample.py (regenerates bundled data)
```

## AnnData conventions (the shared contract)

- `obsm['spatial']` — coordinates (float32). `layers['counts']` raw; `layers['lognorm']` log-norm.
- `obs['domain']` — Leiden spatial domains; `obs['cell_type']` — marker annotation.
- `obs['fate_axis']` — commitment scalar 0→1; `obs['is_target_fate']`.
- `obsm['tf_activity']` — per-cell TF activity (decoupler ULM); `uns['liana_res']` — L-R table.
- `uns['stereo_fate']` — results dict: `tf_ranking`, `lr_ranking`, `recipe`, `prioritized`, …

## Key library APIs (pinned versions)

- **decoupler `<2`** (1.9.x): `dc.get_collectri(organism=...)`, `dc.run_ulm(...)` →
  `obsm['ulm_estimate']`. (2.0 changed the API; we use the 1.x surface.)
- **liana ≥1.7**: `li.mt.rank_aggregate(adata, groupby=...)` → `uns['liana_res']`;
  resources via `li.rs.select_resource('mouseconsensus')`.
- **squidpy**: `sq.gr.spatial_neighbors` → `obsp['spatial_connectivities']`.
- ⚠️ **Mouse CollecTRI** ortholog download is currently 404 upstream; we build the mouse
  regulon by `.capitalize()`-casing human CollecTRI (see `scripts/make_subsample.py`) and
  bundle the subset used for the demo so CI is offline.

## Workflow notes

- Install: `UV_CACHE_DIR=C:/Temp/uv-cache uv pip install --python C:/Temp/sf-venv/Scripts/python.exe -e '.[dev]'`
- Run venv binaries directly: `C:/Temp/sf-venv/Scripts/stereo-fate.exe ...` or `.../python.exe -m pytest`.
- Smoke test the whole thing: `stereo-fate run` (bundled) then `stereo-fate validate`.
- `data/` is committed (tiny); after big runs, offload cold outputs to `O:\rclone-offload\`.
- We do **not** process raw FASTQ / run SAW; start from processed `.gef`/`.h5ad`.

# stereo-fate 🧬🍳

**Read the spatial transcriptome of a developing tissue as a recipe: nominate the
transcription-factor and cell-cell-signaling "ingredients" that drive a cell-fate
decision, and prioritize them with functional-genomics (perturbation) evidence.**

`stereo-fate` is an anndata-native Python package built on the standard spatial-omics
stack — [stereopy](https://github.com/STOmics/Stereopy) (BGI's official Stereo-seq
toolkit), [scanpy](https://scanpy.readthedocs.io), [squidpy](https://squidpy.readthedocs.io),
[liana](https://liana-py.readthedocs.io), and [decoupler](https://decoupler-py.readthedocs.io)
— so everything it produces drops straight back into your existing analysis.

---

## 1. First principles — what is this, and why?

### 1.1 Stereo-seq (the data)

**Stereo-seq** is a spatial transcriptomics technology (BGI/STOmics) that captures
mRNA on a patterned array of DNA nanoballs (DNBs) at ~500 nm spot pitch, over
capture areas measured in **centimeters**. Each spot carries a spatial barcode, so
every transcript keeps its `(x, y)` tissue coordinate. Because the spots are so small
and the field so large, a single chip yields **hundreds of thousands to millions of
"bins"** (spots aggregated into square bins, e.g. bin50 ≈ cell-scale) — far larger
than a typical single-cell experiment. *This scale is why `stereo-fate` guards every
entrypoint with a resource check (see §6).*

We work from **processed** bin/cell-level data — a `.gef` file (read via stereopy) or
an `.h5ad` — never from raw FASTQ. We do **not** invoke SAW (the upstream basecaller).

### 1.2 Cell-fate decisions (the biology)

During development, multipotent **progenitor** cells **commit** to specialized
**fates** (cardiomyocyte, neuron, hepatocyte, …). A commitment is driven by two kinds
of "ingredient":

- **Intrinsic** — **transcription factors (TFs)**. A small set of lineage-specifying
  TFs switches on the fate's gene-expression program (e.g. *Nkx2-5*, *Gata4*, *Tbx5*
  for heart). We don't measure TF *protein activity* directly, but an active TF leaves
  a fingerprint: its target genes move together. **TF activity inference** reads that
  fingerprint.
- **Extrinsic** — **cell-cell signaling**. Cells in a **niche** talk via
  **ligand–receptor (L-R)** pairs (e.g. *Bmp4*→*Bmpr1a*). A fate often commits only
  where the right signal is present, so the *spatial* coincidence of signaling with
  commitment is itself evidence.

### 1.3 Functional genomics (the causality check)

Activity and coincidence are **correlative** — they tell you a TF/signal *moves with*
a fate, not that it *causes* it. **Functional genomics** supplies the causal arm:
**CRISPR knockout screens** and **Perturb-seq** directly perturb a gene and read out
the phenotype. `stereo-fate` cross-references every nominated ingredient against this
perturbation evidence and flags it **causally supported** vs **correlative only**.

### 1.4 The "ingredient / recipe" framing

> A fate is a dish. Its **ingredients** are the TFs (what the cell turns on inside)
> and the signaling cues (what the niche feeds it). The **recipe** is the ranked,
> weighted ingredient list — and **functional genomics is the taste-test** that tells
> you which ingredients actually matter.

`stereo-fate` produces exactly that: one ranked ingredient list per target fate, each
ingredient carrying an effect size, a spatial-coincidence score, and a causal flag.

---

## 2. The pipeline

```
 .gef / .h5ad
     │  io/          load processed Stereo-seq → AnnData  (check_resources first!)
     ▼
 spatial.py         QC · normalize · Leiden spatial domains · marker-based cell types
     ▼
 fate.py            order cells along a differentiation / commitment axis
     ▼
 ┌─────────────────────────────┬──────────────────────────────────────┐
 │ tf_activity.py  (INTRINSIC) │ communication.py  (EXTRINSIC)         │
 │ decoupler ULM + CollecTRI   │ liana L-R  +  squidpy spatial graph   │
 │ → TFs rising along the axis │ → L-R pairs coinciding with the niche │
 └─────────────────────────────┴──────────────────────────────────────┘
     ▼
 recipe.py          combine intrinsic + extrinsic → one ranked ingredient list
     ▼
 funcgen.py         cross-reference CRISPR / Perturb-seq → causal vs correlative flag
```

1. **Load** processed Stereo-seq (GEF via stereopy, or `.h5ad`) into AnnData.
2. **Spatial** QC, normalization, Leiden **spatial domains** (optionally regularized by
   the squidpy neighborhood graph for contiguous territories), marker-based cell types.
3. **Fate axis** — diffusion **pseudotime** rooted in a progenitor when a timecourse is
   present, otherwise a **committed-vs-progenitor contrast** score (0 → 1).
4. **Intrinsic ingredients** — per-cell **TF activity** (decoupler ULM over CollecTRI);
   rank TFs whose activity **rises along the target-fate axis**.
5. **Extrinsic ingredients** — spatially-resolved **cell-cell communication** (liana +
   squidpy graph); rank L-R pairs whose local signaling **spatially coincides** with
   commitment (the niche cues).
6. **Recipe** — combine intrinsic + extrinsic into a single ranked ingredient list with
   effect sizes and spatial-coincidence scores.
7. **Functional-genomics prioritization** — cross-reference against CRISPR/Perturb-seq;
   flag each ingredient **causally supported** vs **correlative only**.

---

## 3. Install

```bash
# requires Python 3.11–3.12
pip install stereo-fate                  # core (works on .h5ad)
pip install 'stereo-fate[stereo]'        # + stereopy for native .gef I/O
pip install 'stereo-fate[dev]'           # + test / notebook tooling
```

Development install (with [uv](https://docs.astral.sh/uv/)):

```bash
uv venv --python 3.11 && uv pip install -e '.[dev]'
```

> **stereopy is optional.** The full pipeline runs on `.h5ad`; stereopy is only needed
> to read raw `.gef` files and its wheels are platform-fragile.

---

## 4. Quickstart

```bash
# resource guard — always available, always runs first
stereo-fate check

# run the whole pipeline on the bundled MOSTA-like subsample
stereo-fate run

# on your own section, choosing the target fate and its progenitor
stereo-fate run input.path=E9.5_E1S1.MOSTA.h5ad fate.target=Cardiac fate.progenitor=Mesenchyme

# run + known-driver recovery validation vs a random baseline
stereo-fate validate
```

Python API:

```python
import stereo_fate as sf

sf.check_resources()                                   # RAM/CPU guard
adata = sf.io.load_bundled_subsample()                 # or load_adata("section.h5ad")
adata = sf.spatial.preprocess(adata)                   # QC → domains → cell types
adata = sf.fate.fate_axis(adata, target_fate="Cardiac",
                          progenitor_fate="Mesenchyme", inplace=True)

# intrinsic
sf.tf_activity.infer_tf_activity(adata, inplace=True)
tf_rank = sf.tf_activity.rank_tfs_along_axis(adata)

# extrinsic
sf.communication.infer_communication(adata, inplace=True)
lr_rank = sf.communication.rank_lr_by_fate_coincidence(adata)

# recipe + causal prioritization
recipe = sf.recipe.build_recipe(adata, tf_ranking=tf_rank, lr_ranking=lr_rank)
prioritized = sf.funcgen.prioritize(recipe)
print(prioritized.head(15))
```

Outputs (`outputs/`): `recipe.csv`, `prioritized_recipe.csv`, `tf_ranking.csv`,
`lr_ranking.csv`, and an annotated `stereo_fate_result.h5ad`.

---

## 5. The cookbook dataset — MOSTA

The reference dataset is **MOSTA**, the *Mouse Organogenesis Spatiotemporal
Transcriptomic Atlas* (Stereo-seq; Chen *et al.*, **Cell** 2022), public via
[STOmicsDB / CNGB](https://db.cngb.org/stomics/mosta/). It spans mouse embryos from
E9.5–E16.5 — a real developmental **differentiation** dataset, ideal for nominating
fate ingredients.

- **Documented loader:** `sf.io.download_mosta("E9.5_E1S1.MOSTA.h5ad")` fetches a real
  section on demand (these are large — the resource guard will warn/refuse/chunk).
- **Bundled subsample:** a small MOSTA-like section ships inside the package
  (`stereo_fate/data/mosta_e95_subsample.h5ad`) for CI and the quickstart. It has
  *planted* cardiac-lineage drivers so the recovery validation has a known answer.
  Regenerate with `python scripts/make_subsample.py`.

See [`cookbook/`](cookbook/) for five end-to-end notebooks with committed figures.

---

## 6. Resource guard (a standing requirement)

Stereo-seq matrices can dwarf RAM. **Every entrypoint and notebook calls
`check_resources()` first.** Policy:

- report free RAM + `cpu_count` (via `psutil`);
- estimate the **dense float32 footprint** of the working matrix;
- if it exceeds **50 % of available RAM** → refuse / chunk / downcast (with a concrete
  chunking recommendation);
- default to **float32** everywhere;
- cap parallelism at **`nproc − 1`**.

```python
from stereo_fate import check_resources
check_resources(n_obs=2_000_000, n_vars=30_000)   # estimate before you densify
```

This policy is documented for agents in [`CLAUDE.md`](CLAUDE.md) and
[`AGENTS.md`](AGENTS.md).

---

## 7. Validation

`stereo-fate validate` recovers **known organ/lineage-specifying TFs** as top
ingredients for the corresponding fate and reports the **recovery rate vs a random-gene
baseline** (empirical enrichment + p-value over 1,000 random gene sets). On the bundled
cardiac subsample the planted regulators (*Nkx2-5*, *Gata4*, *Tbx5*, *Mef2c*, *Hand2*, …)
recover at the top with strong enrichment over the random baseline.

---

## 8. Results

Generated figures live in [`figures/`](figures/) and are produced by the cookbook
notebooks: spatial-domain maps, TF-activity spatial maps along the fate axis,
ligand-receptor niche maps, the ranked recipe, and the functional-genomics
prioritization with known-driver recovery.

---

## License

[GNU GPL v3.0](LICENSE).

## Citation

If you use `stereo-fate`, please cite this repository and the underlying tools
(stereopy, scanpy, squidpy, liana, decoupler, CollecTRI) and the MOSTA atlas
(Chen *et al.*, Cell 2022).

"""Command-line interface for stereo-fate (typer + hydra).

Every entrypoint calls :func:`check_resources` first (a STANDING REQUIREMENT).

Examples
--------
    stereo-fate check
    stereo-fate run                       # runs on the bundled subsample
    stereo-fate run input.path=my.h5ad fate.target=Cardiac fate.progenitor=Mesenchyme
    stereo-fate validate
"""

from __future__ import annotations

import json
from pathlib import Path

import typer
from omegaconf import DictConfig, OmegaConf

from . import communication, funcgen, io, spatial, tf_activity
from . import fate as fate_mod
from . import recipe as recipe_mod
from .resources import check_resources

app = typer.Typer(add_completion=False, help="stereo-fate: ingredients & recipes of cell fate.")

CONFIG_DIR = Path(__file__).resolve().parent.parent / "configs"


def _load_cfg(overrides: list[str] | None) -> DictConfig:
    """Compose the hydra config from configs/ with CLI dotlist overrides."""
    from hydra import compose, initialize_config_dir

    overrides = overrides or []
    with initialize_config_dir(version_base=None, config_dir=str(CONFIG_DIR)):
        cfg = compose(config_name="pipeline", overrides=overrides)
    return cfg


@app.command()
def version():
    """Print the package version."""
    from . import __version__

    typer.echo(__version__)


@app.command()
def check(
    n_obs: int = typer.Option(None, help="rows of the matrix you intend to densify"),
    n_vars: int = typer.Option(None, help="cols (genes)"),
):
    """Resource guard: report RAM / CPU and (optionally) a dense-matrix footprint."""
    check_resources(n_obs, n_vars, verbose=True)


def run_pipeline(cfg: DictConfig) -> dict:
    """End-to-end pipeline: load -> spatial -> fate -> TF -> L-R -> recipe -> prioritize.

    Returns a dict of output paths. All heavy steps are guarded by check_resources.
    """
    outdir = Path(cfg.output.dir)
    outdir.mkdir(parents=True, exist_ok=True)
    typer.echo(OmegaConf.to_yaml(cfg))

    # 0. resource guard up-front -------------------------------------------------
    check_resources(verbose=True)

    # 1. load -------------------------------------------------------------------
    if cfg.input.path in (None, "", "bundled"):
        adata = io.load_bundled_subsample()
    else:
        adata = io.load_adata(cfg.input.path, bin_size=cfg.input.bin_size)
    typer.echo(f"[load] {adata.n_obs} bins x {adata.n_vars} genes")

    # 2. spatial preprocessing --------------------------------------------------
    p = spatial.SpatialParams(**OmegaConf.to_container(cfg.spatial, resolve=True))
    adata = spatial.qc_filter(adata, p)
    adata = spatial.normalize(adata, p, inplace=True)
    adata = spatial.spatial_domains(adata, p, inplace=True)
    markers = OmegaConf.to_container(cfg.markers, resolve=True) if cfg.get("markers") else None
    adata = spatial.annotate_cell_types(adata, markers, inplace=True)
    typer.echo(f"[spatial] domains={adata.obs['domain'].nunique()} "
               f"cell_types={sorted(adata.obs['cell_type'].unique())}")

    # 3. fate axis --------------------------------------------------------------
    adata = fate_mod.fate_axis(
        adata, target_fate=cfg.fate.target, progenitor_fate=cfg.fate.get("progenitor"),
        method=cfg.fate.method, signature=markers, inplace=True,
    )
    typer.echo(f"[fate] method={adata.uns['stereo_fate']['fate_method']} "
               f"target={cfg.fate.target}")

    # 4. intrinsic: TF activity -------------------------------------------------
    net = None
    if cfg.tf.get("net_csv"):
        import pandas as pd
        net = pd.read_csv(cfg.tf.net_csv)
    adata = tf_activity.infer_tf_activity(adata, organism=cfg.tf.organism, net=net, inplace=True)
    tf_rank = tf_activity.rank_tfs_along_axis(adata)
    tf_rank.to_csv(outdir / "tf_ranking.csv", index=False)
    typer.echo(f"[tf] top: {', '.join(tf_rank['tf'].head(5))}")

    # 5. extrinsic: spatial L-R -------------------------------------------------
    try:
        adata = communication.infer_communication(
            adata, groupby="cell_type", resource_name=cfg.comm.resource,
            n_perms=cfg.comm.get("n_perms", 100), n_jobs=cfg.comm.get("n_jobs", 1),
            inplace=True,
        )
        lr_rank = communication.rank_lr_by_fate_coincidence(
            adata, top_n_candidates=cfg.comm.top_n_candidates, n_neighs=cfg.comm.n_neighs
        )
        lr_rank.to_csv(outdir / "lr_ranking.csv", index=False)
        typer.echo(f"[comm] top: {', '.join(lr_rank['interaction'].head(5))}")
    except Exception as e:  # communication can fail on tiny/degenerate data
        typer.echo(f"[comm] skipped ({type(e).__name__}: {e})")
        lr_rank = None

    # 6. recipe -----------------------------------------------------------------
    rec = recipe_mod.build_recipe(
        adata, tf_ranking=tf_rank, lr_ranking=lr_rank, target_fate=cfg.fate.target,
        w_intrinsic=cfg.recipe.w_intrinsic, w_extrinsic=cfg.recipe.w_extrinsic,
    )
    rec.to_csv(outdir / "recipe.csv", index=False)
    typer.echo(f"[recipe] {len(rec)} ingredients; top: {', '.join(rec['ingredient'].head(5))}")

    # 7. functional-genomics prioritization -------------------------------------
    ref = funcgen.load_perturbation_reference(cfg.funcgen.get("reference_csv"))
    prioritized = funcgen.prioritize(rec, ref, fate=cfg.fate.target, adata=adata)
    prioritized.to_csv(outdir / "prioritized_recipe.csv", index=False)
    n_causal = int(prioritized["causal_support"].sum())
    typer.echo(f"[funcgen] {n_causal}/{len(prioritized)} ingredients causally supported")

    # persist the annotated object
    h5_path = outdir / "stereo_fate_result.h5ad"
    if cfg.output.save_h5ad:
        # uns DataFrames are fine for h5ad; drop the heavy score frame if present
        adata.uns.get("stereo_fate", {}).pop("domain_scores", None)
        adata.write_h5ad(h5_path)

    outputs = {
        "recipe": str(outdir / "recipe.csv"),
        "prioritized": str(outdir / "prioritized_recipe.csv"),
        "tf_ranking": str(outdir / "tf_ranking.csv"),
        "h5ad": str(h5_path) if cfg.output.save_h5ad else None,
    }
    (outdir / "outputs.json").write_text(json.dumps(outputs, indent=2))
    return outputs


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def run(ctx: typer.Context):
    """Run the full pipeline. Pass hydra-style overrides, e.g. ``fate.target=Cardiac``."""
    cfg = _load_cfg(ctx.args)
    outputs = run_pipeline(cfg)
    typer.echo("\nOutputs:")
    for k, v in outputs.items():
        typer.echo(f"  {k}: {v}")


@app.command(context_settings={"allow_extra_args": True, "ignore_unknown_options": True})
def validate(ctx: typer.Context):
    """Run the pipeline then report known-driver recovery vs a random baseline."""
    cfg = _load_cfg(ctx.args)
    outputs = run_pipeline(cfg)
    import pandas as pd

    rec = pd.read_csv(outputs["recipe"])
    known = list(cfg.validation.known_drivers)
    report = funcgen.known_driver_recovery(
        rec, known, top_k=cfg.validation.top_k, n_random=cfg.validation.n_random
    )
    typer.echo("\n=== Known-driver recovery ===")
    typer.echo(json.dumps({k: v for k, v in report.items() if k != "known_present"}, indent=2))


if __name__ == "__main__":  # pragma: no cover
    app()

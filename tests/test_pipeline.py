"""End-to-end pipeline tests on the bundled subsample (the CI smoke test).

These assert not just that the pipeline *runs*, but that it *recovers the planted
cardiac drivers* — i.e. the biology is wired up correctly.
"""

import numpy as np

import stereo_fate as sf


def test_spatial_domains_and_celltypes(preprocessed):
    assert "domain" in preprocessed.obs
    assert "cell_type" in preprocessed.obs
    # cardiac territory should be recovered as a cell type
    assert "Cardiac" in set(preprocessed.obs["cell_type"])


def test_fate_axis_tracks_true_commitment(with_fate):
    assert "fate_axis" in with_fate.obs
    fa = np.asarray(with_fate.obs["fate_axis"])
    assert fa.min() >= 0 and fa.max() <= 1
    # the recovered axis should correlate with the planted commitment latent
    from scipy.stats import spearmanr

    rho, _ = spearmanr(fa, with_fate.obs["true_commitment"])
    assert abs(rho) > 0.3


def test_tf_ranking_recovers_cardiac_drivers(with_tf):
    rank = with_tf.uns["stereo_fate"]["tf_ranking"]
    top = list(rank["tf"].head(8))
    planted = set(with_tf.uns["planted_cardiac_tfs"])
    # at least 2 planted cardiac TFs in the top 8 by activity-along-axis
    assert len(planted & set(top)) >= 2


def test_recipe_and_prioritization(with_tf):
    rank = with_tf.uns["stereo_fate"]["tf_ranking"]
    recipe = sf.recipe.build_recipe(with_tf, tf_ranking=rank, target_fate="Cardiac")
    assert {"ingredient", "ingredient_score", "rank"} <= set(recipe.columns)
    assert recipe["ingredient_score"].is_monotonic_decreasing

    prioritized = sf.funcgen.prioritize(recipe)
    assert "causal_support" in prioritized.columns
    assert prioritized["causal_support"].sum() >= 1  # some cardiac genes are causal hits


def test_known_driver_recovery_beats_random(with_tf):
    rank = with_tf.uns["stereo_fate"]["tf_ranking"]
    recipe = sf.recipe.build_recipe(with_tf, tf_ranking=rank)
    known = with_tf.uns["planted_cardiac_tfs"]
    report = sf.funcgen.known_driver_recovery(
        recipe, known, top_k=10, n_random=500,
        gene_col="ingredient", score_col="ingredient_score",
    )
    assert report["recovery_rate"] >= report["random_baseline"]
    assert report["enrichment"] >= 1.0

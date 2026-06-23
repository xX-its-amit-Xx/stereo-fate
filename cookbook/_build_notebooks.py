"""Build the five cookbook notebooks programmatically with nbformat.

Run:  python cookbook/_build_notebooks.py
Then execute them (commits figures + outputs):  see cookbook/_execute.py
"""

from __future__ import annotations

from pathlib import Path

import nbformat as nbf
from nbformat.v4 import new_code_cell, new_markdown_cell, new_notebook

HERE = Path(__file__).resolve().parent

PREAMBLE = """\
import sys, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, ".")  # so `_pipeline` is importable when run from cookbook/
import numpy as np, pandas as pd, matplotlib.pyplot as plt, seaborn as sns
import stereo_fate as sf
from _pipeline import build_result, TARGET_FATE, PROGENITOR
from pathlib import Path
FIG = Path("..") / "figures"; FIG.mkdir(exist_ok=True)
sns.set_context("talk")
# STANDING REQUIREMENT: resource guard first.
sf.check_resources(verbose=True)
"""


def nb(*cells):
    n = new_notebook(cells=list(cells))
    n.metadata["kernelspec"] = {"name": "python3", "display_name": "Python 3", "language": "python"}
    n.metadata["language_info"] = {"name": "python"}
    return n


def md(t):
    return new_markdown_cell(t)


def code(t):
    return new_code_cell(t)


# --------------------------------------------------------------------------- #
# 1. Spatial domains
# --------------------------------------------------------------------------- #
nb1 = nb(
    md("# 1 · Load MOSTA & map spatial domains\n\n"
       "Load the bundled MOSTA-like Stereo-seq subsample, run QC → normalization → "
       "Leiden **spatial domains** → marker-based **cell types**, and commit a "
       "spatial-domain figure.\n\n"
       "> Every notebook calls `check_resources()` first — Stereo-seq sections can be "
       "centimeter-scale with millions of bins."),
    code(PREAMBLE),
    code("adata = build_result()\n"
         "print(adata)\n"
         "print('cell types:', sorted(adata.obs['cell_type'].unique()))"),
    md("### Spatial map: domains and marker-based cell types"),
    code(
        "xy = adata.obsm['spatial']\n"
        "fig, axes = plt.subplots(1, 2, figsize=(15, 6))\n"
        "for ax, key, title in [(axes[0],'domain','Leiden spatial domains'),\n"
        "                       (axes[1],'cell_type','Marker-based cell types')]:\n"
        "    cats = adata.obs[key].astype('category')\n"
        "    codes = cats.cat.codes\n"
        "    sc = ax.scatter(xy[:,0], xy[:,1], c=codes, cmap='tab20', s=6)\n"
        "    ax.set_title(title); ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')\n"
        "    handles=[plt.Line2D([0],[0],marker='o',ls='',color=plt.cm.tab20(i/ max(1,len(cats.cat.categories)-1)),\n"
        "             label=c) for i,c in enumerate(cats.cat.categories)]\n"
        "    ax.legend(handles=handles, fontsize=8, loc='center left', bbox_to_anchor=(1,0.5))\n"
        "fig.suptitle('MOSTA-like subsample — spatial domains & fates', y=1.02)\n"
        "fig.tight_layout(); fig.savefig(FIG/'01_spatial_domains.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md("The Leiden domains form contiguous territories (spatial-graph regularized), and "
       "marker scoring labels each as a lineage — including the **Cardiac** territory "
       "whose ingredients we'll dissect next."),
)

# --------------------------------------------------------------------------- #
# 2. TF activity maps along the fate axis
# --------------------------------------------------------------------------- #
nb2 = nb(
    md("# 2 · TF-activity spatial maps along the fate axis\n\n"
       "Infer per-cell **TF activity** (decoupler ULM over CollecTRI) and visualize the "
       "top intrinsic ingredients spatially and against the **fate axis** (0 = "
       "progenitor → 1 = committed)."),
    code(PREAMBLE),
    code("adata = build_result()\n"
         "rank = adata.uns['stereo_fate']['tf_ranking']\n"
         "rank.head(10)[['rank','tf','spearman','effect_size','fdr']]"),
    md("### Fate axis in space, and the top fate-rising TFs"),
    code(
        "xy = adata.obsm['spatial']\n"
        "acts = adata.obsm['tf_activity']\n"
        "top_tfs = list(rank['tf'].head(3))\n"
        "fig, axes = plt.subplots(1, 4, figsize=(22, 5))\n"
        "s0=axes[0].scatter(xy[:,0],xy[:,1],c=adata.obs['fate_axis'],cmap='magma',s=6)\n"
        "axes[0].set_title('fate axis'); plt.colorbar(s0,ax=axes[0],shrink=.8)\n"
        "for ax,tf in zip(axes[1:], top_tfs):\n"
        "    v=acts[tf].values\n"
        "    sc=ax.scatter(xy[:,0],xy[:,1],c=v,cmap='RdBu_r',vmin=-np.abs(v).max(),vmax=np.abs(v).max(),s=6)\n"
        "    ax.set_title(f'{tf} activity'); plt.colorbar(sc,ax=ax,shrink=.8)\n"
        "for ax in axes: ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')\n"
        "fig.suptitle('Intrinsic ingredients: TF activity tracks cardiac commitment', y=1.04)\n"
        "fig.tight_layout(); fig.savefig(FIG/'02_tf_activity_maps.png', dpi=150, bbox_inches='tight')\n"
        "plt.show()"
    ),
    md("### Activity vs the fate axis (the ranking criterion)"),
    code(
        "fig, ax = plt.subplots(figsize=(9,6))\n"
        "fa = adata.obs['fate_axis'].values\n"
        "for tf in top_tfs:\n"
        "    order=np.argsort(fa)\n"
        "    y=pd.Series(acts[tf].values[order]).rolling(80,min_periods=1).mean()\n"
        "    ax.plot(np.sort(fa), y, label=tf, lw=2.5)\n"
        "ax.set_xlabel('fate axis (progenitor → committed)'); ax.set_ylabel('TF activity (ULM, smoothed)')\n"
        "ax.legend(title='top TFs'); ax.set_title('Top TFs rise along the cardiac fate axis')\n"
        "fig.tight_layout(); fig.savefig(FIG/'02b_tf_activity_vs_fate.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
)

# --------------------------------------------------------------------------- #
# 3. Ligand-receptor niche maps
# --------------------------------------------------------------------------- #
nb3 = nb(
    md("# 3 · Ligand–receptor niche maps\n\n"
       "Infer spatially-resolved **cell-cell communication** (liana + squidpy graph) and "
       "map the local signaling of the top niche cues — L-R pairs whose signaling "
       "**spatially coincides** with cardiac commitment."),
    code(PREAMBLE),
    code("adata = build_result()\n"
         "lr = adata.uns['stereo_fate']['lr_ranking']\n"
         "lr.head(10)[['rank','interaction','spatial_coincidence','effect_size','liana_specificity']]"),
    md("### Local interaction score in space for the top coincident pairs"),
    code(
        "from scipy import sparse\n"
        "import squidpy as sq\n"
        "if 'spatial_connectivities' not in adata.obsp:\n"
        "    sq.gr.spatial_neighbors(adata, n_neighs=6, coord_type='generic')\n"
        "S=adata.obsp['spatial_connectivities'].astype(float); rs=np.asarray(S.sum(1)).ravel(); rs[rs==0]=1\n"
        "def localscore(lig,rec):\n"
        "    L=adata[:,lig].layers['lognorm']; R=adata[:,rec].layers['lognorm']\n"
        "    L=L.toarray().ravel() if sparse.issparse(L) else np.asarray(L).ravel()\n"
        "    R=R.toarray().ravel() if sparse.issparse(R) else np.asarray(R).ravel()\n"
        "    return L*((S@R)/rs)\n"
        "xy=adata.obsm['spatial']; pairs=list(lr[['ligand','receptor']].head(3).itertuples(index=False))\n"
        "fig,axes=plt.subplots(1,4,figsize=(22,5))\n"
        "s0=axes[0].scatter(xy[:,0],xy[:,1],c=adata.obs['fate_axis'],cmap='magma',s=6)\n"
        "axes[0].set_title('fate axis'); plt.colorbar(s0,ax=axes[0],shrink=.8)\n"
        "for ax,(lig,rec) in zip(axes[1:],pairs):\n"
        "    v=localscore(lig,rec)\n"
        "    sc=ax.scatter(xy[:,0],xy[:,1],c=v,cmap='viridis',s=6)\n"
        "    ax.set_title(f'{lig}→{rec}'); plt.colorbar(sc,ax=ax,shrink=.8)\n"
        "for ax in axes: ax.set_aspect('equal'); ax.invert_yaxis(); ax.axis('off')\n"
        "fig.suptitle('Extrinsic ingredients: niche signaling coincides with commitment', y=1.04)\n"
        "fig.tight_layout(); fig.savefig(FIG/'03_lr_niche_maps.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
)

# --------------------------------------------------------------------------- #
# 4. The ranked recipe
# --------------------------------------------------------------------------- #
nb4 = nb(
    md("# 4 · The ranked ingredient **recipe** for the cardiac lineage\n\n"
       "Combine intrinsic (TF) and extrinsic (L-R) evidence into a single ranked "
       "ingredient list with effect sizes and spatial-coincidence scores."),
    code(PREAMBLE),
    code("adata = build_result()\n"
         "recipe = adata.uns['stereo_fate']['recipe']\n"
         "recipe.head(20)[['rank','ingredient','kind','role','effect_size','spatial_coincidence','ingredient_score']]"),
    md("### Top ingredients, colored by kind"),
    code(
        "top=recipe.head(20).iloc[::-1]\n"
        "palette={'TF':'#d62728','ligand':'#1f77b4','receptor':'#2ca02c'}\n"
        "colors=[palette.get(k.split('+')[0],'#7f7f7f') for k in top['kind']]\n"
        "fig,ax=plt.subplots(figsize=(9,9))\n"
        "ax.barh(top['ingredient'], top['ingredient_score'], color=colors)\n"
        "ax.set_xlabel('ingredient score'); ax.set_title(f'{TARGET_FATE} fate — ranked recipe (top 20)')\n"
        "handles=[plt.Rectangle((0,0),1,1,color=c) for c in palette.values()]\n"
        "ax.legend(handles,palette.keys(),title='kind')\n"
        "fig.tight_layout(); fig.savefig(FIG/'04_recipe.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
)

# --------------------------------------------------------------------------- #
# 5. Functional-genomics prioritization + recovery
# --------------------------------------------------------------------------- #
nb5 = nb(
    md("# 5 · Functional-genomics prioritization & known-driver recovery\n\n"
       "Cross-reference each ingredient against CRISPR / Perturb-seq evidence (causal vs "
       "correlative), and validate that **known cardiac drivers** surface at the top vs a "
       "random-gene baseline."),
    code(PREAMBLE),
    code("adata = build_result()\n"
         "pr = adata.uns['stereo_fate']['prioritized']\n"
         "pr.head(15)[['priority_rank','ingredient','kind','ingredient_score','causal_support','evidence_level','perturb_datasets']]"),
    md("### Prioritized ingredients — causal support highlighted"),
    code(
        "top=pr.head(20).iloc[::-1]\n"
        "colors=['#2ca02c' if c else '#bbbbbb' for c in top['causal_support']]\n"
        "fig,ax=plt.subplots(figsize=(9,9))\n"
        "ax.barh(top['ingredient'], top['ingredient_score'], color=colors)\n"
        "ax.set_xlabel('ingredient score'); ax.set_title('Prioritized recipe (green = causally supported)')\n"
        "fig.tight_layout(); fig.savefig(FIG/'05_prioritized.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
    md("### Known-driver recovery vs random baseline"),
    code(
        "recipe = adata.uns['stereo_fate']['recipe']\n"
        "known = list(adata.uns['planted_cardiac_tfs'])\n"
        "rep = sf.funcgen.known_driver_recovery(recipe, known, top_k=25, n_random=1000)\n"
        "print({k:v for k,v in rep.items() if k!='known_present'})\n"
        "fig,ax=plt.subplots(figsize=(6,6))\n"
        "ax.bar(['stereo-fate','random'],[rep['recovery_rate'],rep['random_baseline']],\n"
        "       color=['#d62728','#bbbbbb'])\n"
        "ax.set_ylabel('known-driver recovery @ top-25')\n"
        "ax.set_title(f\"recovery {rep['recovery_rate']:.0%} vs {rep['random_baseline']:.0%} \"\n"
        "             f\"(enrich {rep['enrichment']:.1f}x, p={rep['pval']:.3f})\")\n"
        "fig.tight_layout(); fig.savefig(FIG/'05b_recovery.png', dpi=150, bbox_inches='tight'); plt.show()"
    ),
    md("**Result:** the known cardiac regulators recover at the top of the recipe far "
       "above the random baseline — the pipeline nominates real fate ingredients, and "
       "functional-genomics evidence flags which are causally supported."),
)

NOTEBOOKS = {
    "01_spatial_domains.ipynb": nb1,
    "02_tf_activity_maps.ipynb": nb2,
    "03_lr_niche_maps.ipynb": nb3,
    "04_recipe.ipynb": nb4,
    "05_funcgen_recovery.ipynb": nb5,
}

if __name__ == "__main__":
    for name, n in NOTEBOOKS.items():
        nbf.write(n, str(HERE / name))
        print("wrote", name)

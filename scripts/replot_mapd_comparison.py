"""Regenerate the MAPD comparison figure from its archived values.

Reads ``Opt_ES/analysis/AT-0/ms_val_opt/MAPD_comp_bestFF.csv`` and rewrites
``MAPD_comp_bestFF.png`` next to it. This is a plotting path only: it loads no GP
models, needs no signac workspace, and contains no absolute paths, so the figure
can be reproduced from a fresh clone of this repository.

The values themselves are computed by ``make_GP_vs_sim_and_sens.py`` (and the
equivalent block in ``create_analysis_figs.ipynb``), which require the full
simulation environment. Those remain the computation path; this script only
redraws what they archived.

What the axes mean
------------------
The two axes are adjacent links in a chain that shares the FF simulation as its
middle term::

    GP  <-- x -->  FF simulation  <-- y -->  experiment

The x-axis is the MAPD between the GP posterior mean and the *simulated*
property at the GP-Optimized FF's parameter set, i.e. how well the surrogate
reproduces the simulation. The y-axis is the FF's MAPD against *experiment*.
They are not two estimates of the same quantity, so no parity line is drawn.

Run from the repository root::

    python scripts/replot_mapd_comparison.py
"""

import os

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(REPO_ROOT, "Opt_ES", "analysis", "AT-0", "ms_val_opt")
CSV_PATH = os.path.join(DATA_DIR, "MAPD_comp_bestFF.csv")
PNG_PATH = os.path.join(DATA_DIR, "MAPD_comp_bestFF.png")

X_LABEL = "GP prediction vs. FF simulation, MAPD/%"
Y_LABEL = "FF simulation vs. experiment, MAPD/%"


def main():
    data_df = pd.read_csv(CSV_PATH, index_col=0)
    # Panel y-limit matches the computation path, which tracks the largest of the
    # two MAPD values across every molecule and property.
    max_mapd = np.maximum(data_df["GP_MAPD"], data_df["FF_MAPD"]).max()

    fig, axes = plt.subplots(1, 2, figsize=(12, 6))
    for i, property_name in enumerate(data_df["Property Name"].unique()):
        label_type = "o" if i == 0 else "s"
        prop_data = data_df[data_df["Property Name"] == property_name]
        colors = plt.cm.tab10.colors
        for j in range(len(prop_data)):
            axes[i].scatter(
                prop_data["GP_MAPD"].iloc[j],
                prop_data["FF_MAPD"].iloc[j],
                label=prop_data["Molecule"].iloc[j],
                color=colors[j % len(colors)],
                marker=label_type,
                s=150,
                alpha=0.5,
            )
        axes[i].set_title(
            f"{property_name.split('/')[0]} MAPD Comparison", fontsize=24
        )
        if i == 1:
            axes[i].set_ylim(0, max_mapd * 1.05)
        else:
            axes[i].legend(loc="lower right", fontsize=18, ncol=2)
        axes[i].tick_params("y", direction="inout", which="both", length=7)
        axes[i].tick_params("y", which="major", length=14)
        axes[i].tick_params("x", pad=15)
        axes[i].tick_params(axis="both", which="major", labelsize=14)
    fig.supylabel(Y_LABEL, fontsize=18)
    fig.supxlabel(X_LABEL, fontsize=18)

    plt.tight_layout()
    fig.savefig(PNG_PATH)
    print(f"wrote {PNG_PATH} from {os.path.basename(CSV_PATH)} ({len(data_df)} rows)")


if __name__ == "__main__":
    main()

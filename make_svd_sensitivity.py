"""SVD-based local sensitivity analysis of the GP surrogates.

For each molecule and each training property this script

1. builds the Jacobian of the GP posterior mean with respect to the GP inputs
   (the LJ parameters and temperature), evaluated at every row of the
   corresponding ``all_results.csv``;
2. takes the singular value decomposition of that Jacobian; and
3. writes the singular values, the right singular vectors, the combined
   decomposition, and the :math:`q_j` parameter ranking to
   ``Build_GPs/analysis/<MOL>/<iters>/sens_approx/``.

Inputs are bounds-scaled before the GP is evaluated
------------------------------------------------------------------
The GP surrogates are trained on inputs scaled to the unit interval: the LJ
parameters are scaled by ``param_bounds`` (already expressed in nm and kJ/mol,
matching ``all_results.csv``) and temperature is scaled by
``temperature_bounds``.  The Jacobian is therefore evaluated in the same scaled
space, so that the GP is queried inside its training domain and the resulting
gradients are dimensionless and directly comparable between sigma and epsilon.

Two variants are produced in a single pass: ``w_temp`` retains the temperature
column of the Jacobian, while ``wo_temp`` drops it so that only the LJ
parameters are ranked.

Run from the repository root::

    python make_svd_sensitivity.py

This module deliberately depends only on numpy/pandas/tensorflow/gpflow (via the
pickled models) and the repository's own helpers, so the analysis can be
reproduced without the simulation-workflow dependencies used elsewhere.
"""

import glob
import os
import pickle

import numpy as np
import pandas as pd
import tensorflow as tf

from fffit.fffit.utils import values_real_to_scaled
from utils.molec_class_files import esolvs

REPO_ROOT = os.path.dirname(os.path.abspath(__file__))

MOL_NAMES = ["DEC", "DMF", "DMSO", "EG", "Gly", "MeOH"]
PROPERTIES = ["liq_density", "surf_tens"]
MODES = ["w_temp", "wo_temp"]

# Which iteration directory and kernel supply the GP for each property. These
# match the selection made in Opt_ES/utilsOpt/opt_atom_types.get_gp_data_from_pkl.
PROPERTY_SOURCE = {
    "liq_density": {"iters": "ld_iters", "kernel": "RQ", "gp_key": "sim_liq_density"},
    "surf_tens": {"iters": "vle_iters", "kernel": "Matern32", "gp_key": "sim_surf_tens"},
}


def load_gp_model(mol_name, prop_name, repo_root=REPO_ROOT):
    """Return the GP model used for ``prop_name`` and the matching results file."""
    source = PROPERTY_SOURCE[prop_name]
    iters = source["iters"]
    pattern = os.path.join(
        repo_root, "Build_GPs", "analysis", mol_name, iters, "iter-*", "gp_models.pkl"
    )
    model_files = sorted(glob.glob(pattern))
    if not model_files:
        raise FileNotFoundError(f"No GP models found for {mol_name} under {pattern}")
    # Use the most recent iteration
    with open(model_files[-1], "rb") as handle:
        gp_models, _best_labels = pickle.load(handle)
    gp_model = gp_models[source["gp_key"]][source["kernel"]]
    results_csv = os.path.join(
        repo_root, "Build_GPs", "analysis", mol_name, iters, "all_results.csv"
    )
    return gp_model, results_csv, model_files[-1]


def sensitivity_jacobian(gp_model, results_csv, mol_data, prop_name):
    """Jacobian of the GP posterior mean w.r.t. the (scaled) GP inputs."""
    param_names = list(mol_data.param_names)
    param_bounds = np.asarray(mol_data.param_bounds)
    temperature_bounds = mol_data.temperature_bounds(f"expt_{prop_name}")

    results = pd.read_csv(results_csv)
    real_params = results[param_names].values
    real_temperature = results["temperature"].values.reshape(-1, 1)

    # Scale to the unit interval the GP was trained on
    scaled_params = values_real_to_scaled(real_params, param_bounds)
    scaled_temperature = values_real_to_scaled(real_temperature, temperature_bounds)
    x_test = np.hstack((scaled_params, scaled_temperature))

    x_test_tf = tf.Variable(x_test, dtype=tf.float64)
    with tf.GradientTape() as tape:
        tape.watch(x_test_tf)
        mean, _variance = gp_model.predict_f(x_test_tf)
    jacobian = tape.gradient(mean, x_test_tf).numpy()
    return jacobian, param_names + ["temperature"], len(results)


def decompose(jacobian, input_names, mode):
    """SVD of the Jacobian, plus the q_j ranking, for one temperature mode."""
    if mode == "wo_temp":
        jacobian = jacobian[:, :-1]
        names = input_names[:-1]
    else:
        names = list(input_names)

    S, _U, Vt = tf.linalg.svd(tf.constant(jacobian))
    sing_vals = S.numpy()
    # tf.linalg.svd returns v (not v^H). Transposing puts one right singular
    # vector per ROW, with columns ordered like ``names``; row i pairs with
    # sing_vals[i], and singular values come back in descending order.
    right_vecs = Vt.numpy().T

    # q_j = sum_i s_i |v_{i,j}|
    q = sing_vals @ np.abs(right_vecs)
    order = np.argsort(q)[::-1]
    return {
        "names": names,
        "sing_vals": sing_vals,
        "right_vecs": right_vecs,
        "q": q,
        "ranked_names": [names[i] for i in order],
        "ranked_q": q[order],
    }


def output_dir(mol_name, prop_name, mode, repo_root=REPO_ROOT):
    iters = PROPERTY_SOURCE[prop_name]["iters"]
    suffix = "_wo_temp" if mode == "wo_temp" else ""
    path = os.path.join(
        repo_root,
        "Build_GPs",
        "analysis",
        mol_name,
        iters,
        "sens_approx",
        "sens_approx" + suffix,
    )
    os.makedirs(path, exist_ok=True)
    return path


def write_results(result, mol_name, prop_name, mode, repo_root=REPO_ROOT):
    """Write singular values, right singular vectors, ranking, and full SVD."""
    sens_dir = output_dir(mol_name, prop_name, mode, repo_root)
    names = result["names"]

    pd.DataFrame(result["sing_vals"], columns=["sing_value"]).to_csv(
        os.path.join(sens_dir, f"sing_val_{prop_name}.csv"), index=False
    )
    pd.DataFrame(result["right_vecs"], columns=names).to_csv(
        os.path.join(sens_dir, f"basis_vec_{prop_name}.csv"), index=False
    )
    pd.DataFrame(
        {"param_rank": result["ranked_names"], "sensitivity_score": result["ranked_q"]}
    ).to_csv(os.path.join(sens_dir, f"param_rank_{prop_name}.csv"), index=False)

    # Full decomposition in one table: each row is a singular triplet, pairing a
    # singular value with the components of its right singular vector, ordered
    # from largest to smallest singular value. This keeps the small-singular-value
    # directions inspectable next to the singular values they belong to.
    svd_full = pd.DataFrame(result["right_vecs"], columns=names)
    svd_full.insert(0, "sing_value", result["sing_vals"])
    svd_full.insert(0, "mode_index", np.arange(len(result["sing_vals"])))
    svd_full.to_csv(os.path.join(sens_dir, f"svd_full_{prop_name}.csv"), index=False)
    return sens_dir


def run_sensitivity_analysis(mol_names=None, repo_root=REPO_ROOT, verbose=True):
    """Run the full analysis and write every output file. Returns the results."""
    mol_names = list(mol_names or MOL_NAMES)
    molec_dict = esolvs.make_dict(mol_names)
    all_results = {}

    for mol_name in mol_names:
        mol_data = molec_dict[mol_name]
        ranks = {mode: {} for mode in MODES}
        q_vals = {mode: {} for mode in MODES}

        for prop_name in PROPERTIES:
            gp_model, results_csv, model_file = load_gp_model(
                mol_name, prop_name, repo_root
            )
            jacobian, input_names, n_rows = sensitivity_jacobian(
                gp_model, results_csv, mol_data, prop_name
            )
            for mode in MODES:
                result = decompose(jacobian, input_names, mode)
                write_results(result, mol_name, prop_name, mode, repo_root)
                ranks[mode][prop_name] = result["ranked_names"]
                q_vals[mode][prop_name] = result["ranked_q"]
                all_results[(mol_name, prop_name, mode)] = result
            if verbose:
                print(
                    f"{mol_name:5s} {prop_name:12s} N={n_rows:5d} "
                    f"GP={os.path.relpath(model_file, repo_root)}"
                )

        # Per-molecule summary table of the rankings for both properties
        analysis_dir = os.path.join(repo_root, "Build_GPs", "analysis", mol_name)
        for mode in MODES:
            suffix = "_wo_temp" if mode == "wo_temp" else ""
            rank_table = pd.DataFrame(
                {
                    "Rank": list(range(1, len(ranks[mode]["liq_density"]) + 1)),
                    r"\rho_l - LD Data": ranks[mode]["liq_density"],
                    "q_j (rho_l)": q_vals[mode]["liq_density"],
                    r"\gamma - ST Data": ranks[mode]["surf_tens"],
                    "q_j (gamma)": q_vals[mode]["surf_tens"],
                }
            )
            rank_table.to_csv(
                os.path.join(analysis_dir, f"param_rank_table{suffix}.csv"), index=False
            )

    return all_results


if __name__ == "__main__":
    run_sensitivity_analysis()

# SVD sensitivity analysis of the GP surrogates

This document describes how to reproduce the singular value decomposition (SVD)
sensitivity analysis of the Gaussian process (GP) surrogate models.

## Running the analysis

From the repository root:

```bash
python make_svd_sensitivity.py
```

This regenerates every output described below for all six molecules and both
training properties. The same analysis is invoked from
`make_GP_vs_sim_and_sens.py`, which calls `run_sensitivity_analysis()` from this
module; `make_svd_sensitivity.py` is the single implementation and can be run on
its own, without the simulation-workflow dependencies (`signac`, `pymser`) that
the larger script requires.

### Requirements

`numpy`, `pandas`, `tensorflow`, `gpflow`, `unyt`, `thermo`, and the vendored
`fffit` package in this repository. The pickled GP models are loaded directly,
so no models are retrained and results are reproducible from the committed
artifacts. The analysis has been run with GPflow 2.9.1 / TensorFlow 2.15.

## What is computed

For each molecule and each training property, the Jacobian of the GP posterior
mean is evaluated with respect to the GP inputs at every row of the
corresponding `all_results.csv`, and that Jacobian is decomposed with an SVD.

### Input scaling

The GP surrogates are trained on inputs scaled to the unit interval: the LJ
parameters are scaled by `param_bounds` (in nm and kJ/mol, matching
`all_results.csv`) and temperature is scaled by `temperature_bounds`. The
Jacobian is therefore evaluated in the same scaled space, so the GP is queried
inside its training domain. A useful side effect is that the resulting
derivatives are dimensionless, so sigma and epsilon sensitivities are directly
comparable rather than depending on the choice of length and energy units.

### Structure of the sensitivity matrix

Each **row** of the Jacobian is one prediction, i.e. one (parameter set,
temperature) combination taken from `all_results.csv`; each **column** is one GP
input. The row count is therefore (number of parameter sets) x (number of
temperatures) — for example 125 = 25 sets x 5 temperatures for the surface
tension iterations, and 2000-6983 rows for the liquid density iterations.

Two variants are produced in one pass:

- **`wo_temp`** drops the temperature column, so the columns are exactly the LJ
  parameters and temperature enters only by indexing rows (predictions). This is
  the variant reported in the manuscript.
- **`w_temp`** retains the temperature column among the inputs.

### Ranking metric

Parameters are ranked by

$$q_j = \sum_i s_i \left| v_{i,j} \right|$$

where $s_i$ is the $i$-th singular value and $v_{i,j}$ is the $j$-th component
of the $i$-th right singular vector.

## Output files

Written to `Build_GPs/analysis/<MOL>/<ld_iters|vle_iters>/sens_approx/sens_approx[_wo_temp]/`:

| File | Contents |
|------|----------|
| `sing_val_<prop>.csv` | Singular values, in descending order |
| `basis_vec_<prop>.csv` | Right singular vectors, one per row, columns labeled by input |
| `param_rank_<prop>.csv` | Parameters ordered by $q_j$, with the $q_j$ values |
| `svd_full_<prop>.csv` | Full decomposition: each row pairs a singular value with the components of its right singular vector |

`svd_full_<prop>.csv` presents the singular values and right singular vectors
together, ordered from largest to smallest singular value, so that the
directions associated with the smallest singular values — the parameter
combinations the data constrain least — can be read off alongside the singular
values they belong to.

A per-molecule summary of both properties is written to
`Build_GPs/analysis/<MOL>/param_rank_table[_wo_temp].csv`.

## GP models used

The most recent iteration is used for each property: liquid density from
`ld_iters` (RQ kernel) and surface tension from `vle_iters` (Matern32 kernel),
matching the selection in `Opt_ES/utilsOpt/opt_atom_types.get_gp_data_from_pkl`.
The script prints the exact model file used for each molecule and property when
it runs.

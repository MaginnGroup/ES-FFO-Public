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

### Mean functions, and a known exception for DMF

The liquid-density GPs use a plain `gpflow` `Linear` mean function. Five of the
six surface-tension GPs — MeOH, EG, Gly, DMSO and DEC — instead use
`MaskedLinear` (built as the `'Custom'` mean function in
`Build_GPs/utils/models.py`), which masks the sigma inputs so the linear mean
acts only on the epsilon parameters and temperature.

**The archived DMF surface-tension GP uses a plain `Linear` mean function, not
`MaskedLinear`.** It is the one exception among the six, and it is the model
behind the published DMF surface-tension results.

This is deliberate, not a stale artifact. In the development history, the commit
that introduced the masked mean function updated the models for the other five
molecules and explicitly left DMF's alone; its message reads *"Updated GP models
of all molecules except DMF to exlude linear dependence of sigma in the mean
functions"* (private-history commit `de6a765`, 2025-12-17; the supporting code
change is `a2aeecb`, the same day). DMF's surface-tension pickle dates from the
previous day's model update and was not rewritten by that commit. **No reason for
the exception is recorded anywhere in the history, and none should be inferred.**

**Re-running the pipeline as committed will not reproduce the archived DMF
model.** `Build_GPs/utils/models.py` computes an Eötvös scale for every molecule
in the `vle_iters` branch, and sets `mean_function='Custom'` whenever that scale
is not `None`, so current code masks DMF along with the other five. Anyone
regenerating the surface-tension GPs should expect a `MaskedLinear` DMF model
that differs from the archived one. The archived models are used as-is by the
analysis scripts here; nothing is retrained.

Two clarifications, because "DMF is different" invites broader inferences than
the evidence supports:

- **Only the mean function differs.** Eötvös noise scaling was applied to all six
  surface-tension GPs, DMF included: the archived white-kernel variance matches
  the Eötvös scaled variance to seven digits for DMF (0.1620487) and for DMSO
  (0.0863983, checked as a control). DMF's likelihood and noise settings are the
  same as the other five.
- **Accuracy does not motivate changing it.** Retraining DMF's surface-tension GP
  with the mask and the same Eötvös noise gives a slightly *worse* test-set fit
  (2.347% MAPD, 0.9947 mN/m RMSE) than the archived model (2.194%, 0.9223). The
  difference is small enough to sit within restart-to-restart variation, so this
  is a reason not to bother changing the archived model rather than evidence
  about why the exception was made.

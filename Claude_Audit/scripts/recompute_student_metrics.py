"""
Worker Task 03 (WP-C1), Part 2: independent, from-scratch recompute of student-FF
(GP-Optimized & Base) MAPD/MAE/MSE/MPD, using the SAME experimental reference values
the repo used. Does NOT import utils.prep_ms_data -- every formula is reimplemented
here so this is a genuine independent check, not a re-run of the audited code.

Two aggregation levels are computed for rho_l/rho_v/Pvap/gamma (see AUDIT_PLAN Task 03
"Replicate/aggregation note"):
  - raw_replicates : matches the repo's own prepare_df_errors, which operates on all
    raw ms_data.csv rows (3 replicate restarts per temperature, expt value repeated).
  - per_T_mean      : uses the Task 02 per-temperature MEAN sim value
    (audit_data/student_ff_values.csv) instead of raw replicates.
These will differ slightly because MAPD/MAE of raw points != MAPD/MAE of point means
in general (Jensen's-inequality-type effect) -- this is an aggregation-choice artifact,
not a bug, and is called out explicitly in the findings.

For Hvap, three separate comparisons are computed:
  - direct_sim_grid_raw_replicates : ms_data.csv's own sim_Hvap vs expt_Hvap columns,
    at the SIMULATED temperature grid, over all raw replicate rows.
  - direct_sim_grid_per_T_mean     : same, but averaged per temperature first.
  - CC_expt_grid_reimplemented     : an independent reimplementation of the repo's
    Clausius-Clapeyron bracket-search estimate (utils/prep_ms_data.py:estimate_hvaps),
    evaluated at the EXPERIMENTAL Hvap temperature grid (which is often a different,
    non-overlapping grid -- see findings/02). This lets us verify the repo's own
    Hvap_estimates.csv values from first principles without importing that function.

Tc/rho_c: single fitted value per (molecule, ff); reported as method=single_value.
"""
import os
import csv
import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error

# REPO = "/Users/adowling/DowlingLab/ES-FFO-Public"
# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
REPO = "/groups/ed/group_members/Montana_Carlozo/ES-FFO"
AT00 = f"{REPO}/Opt_ES/analysis/at_00"

MOLECULES = ["MeOH", "EG", "Gly", "DMF", "DMSO", "DEC"]
FF_DIRS = {"GP-Optimized": "ms_val_opt", "Base": "ms_val_no_opt"}

# Molecular weights (g/mol), read directly from the live esolvs.py module (not hand-copied)
import sys
sys.path.insert(0, REPO)
from utils.molec_class_files import esolvs  # noqa: E402
_molec_dict = esolvs.make_dict(MOLECULES)
MOL_WT = {m: _molec_dict[m].molecular_weight for m in MOLECULES}

R_KJ_PER_MOL_K = 8.314e-3  # kJ/(mol*K) -- same constant as utils/prep_ms_data.py:36

PROP_COLS = {"rho_l": "liq_density", "rho_v": "vap_density", "Pvap": "Pvap", "gamma": "surf_tens"}


def calc_metrics(expt, sim):
    """From-scratch MAPD (%), MAE, MSE, MPD (%) -- reimplemented, not imported."""
    expt = np.asarray(expt, dtype=float)
    sim = np.asarray(sim, dtype=float)
    finite = np.isfinite(expt) & np.isfinite(sim)
    n = int(finite.sum())
    if n == 0:
        return dict(mapd=np.nan, mae=np.nan, mse=np.nan, mpd=np.nan, n_points=0)
    e, s = expt[finite], sim[finite]
    mse = mean_squared_error(e, s)
    mapd = mean_absolute_percentage_error(e, s) * 100.0
    mae = mean_absolute_error(e, s)
    mpd = float(np.mean((s - e) / e * 100.0))
    return dict(mapd=mapd, mae=mae, mse=mse, mpd=mpd, n_points=n)


def get_hvap_est_reimplemented(sim_pvap_by_T, sim_hvap_by_T, Texp, mol_wt):
    """Independent reimplementation of utils/prep_ms_data.py:estimate_hvaps.get_hvap_est
    for a single experimental temperature. Returns (value, method)."""
    if Texp in sim_hvap_by_T and not np.isnan(sim_hvap_by_T[Texp]):
        return sim_hvap_by_T[Texp], "direct"
    temps = np.array(sorted(sim_pvap_by_T.keys()))
    higher = temps[temps > Texp]
    lower = temps[temps < Texp]
    if len(higher) == 0 or len(lower) == 0:
        return np.nan, "skipped"
    T_hi, T_lo = higher.min(), lower.max()
    P_hi, P_lo = sim_pvap_by_T.get(T_hi, np.nan), sim_pvap_by_T.get(T_lo, np.nan)
    if np.isnan(P_hi) or np.isnan(P_lo):
        return np.nan, "skipped"
    H_est_molar = -np.log(P_hi / P_lo) * R_KJ_PER_MOL_K / (1 / T_hi - 1 / T_lo)  # kJ/mol
    H_est_kg = H_est_molar / mol_wt * 1000.0  # kJ/kg
    return H_est_kg, "estimated"


def main():
    student_values = pd.read_csv(f"{AUDIT}/audit_data/student_ff_values.csv")
    out_rows = []

    for mol in MOLECULES:
        for ff_label, ff_dir in FF_DIRS.items():
            raw_path = f"{AT00}/{mol}/ExpVal/opt_res/{ff_dir}/ms_data.csv"
            raw = pd.read_csv(raw_path, index_col=0)

            # ---- rho_l, rho_v, Pvap, gamma ----
            for prop, repo_col in PROP_COLS.items():
                sim_col, expt_col = f"sim_{repo_col}", f"expt_{repo_col}"
                if sim_col not in raw.columns:
                    continue

                # (a) raw_replicates: every restart row, exactly matching prepare_df_errors's input granularity
                m = calc_metrics(raw[expt_col], raw[sim_col])
                out_rows.append({
                    "molecule": mol, "ff": ff_label, "property": prop, "method": "raw_replicates",
                    **m, "notes": f"n={m['n_points']} raw replicate rows (of {len(raw)} total rows in ms_data.csv)",
                })

                # (b) per_T_mean: Task 02's per-temperature mean sim value vs the recorded expt value
                sub = student_values[(student_values.molecule == mol) & (student_values.ff == ff_label)
                                      & (student_values.property == prop) & (student_values.temperature_K != "")]
                m2 = calc_metrics(sub["expt_reference_value"], sub["sim_value"])
                out_rows.append({
                    "molecule": mol, "ff": ff_label, "property": prop, "method": "per_T_mean",
                    **m2, "notes": f"n={m2['n_points']} temperature points (of {len(sub)} in student_ff_values.csv)",
                })

            # ---- Tc, rho_c: single value per (molecule, ff) ----
            for prop in ["Tc", "rho_c"]:
                sub = student_values[(student_values.molecule == mol) & (student_values.ff == ff_label)
                                      & (student_values.property == prop)]
                if len(sub) == 0:
                    continue
                m = calc_metrics(sub["expt_reference_value"], sub["sim_value"])
                out_rows.append({
                    "molecule": mol, "ff": ff_label, "property": prop, "method": "single_value",
                    **m, "notes": "single fitted critical-point value" if m["n_points"] else
                                  "sim value is NaN (critical-point fit failed -- no vapor-phase sim data)",
                })

            # ---- Hvap: direct-at-sim-grid (raw + per-T mean) and CC-at-expt-grid (reimplemented) ----
            if "sim_Hvap" in raw.columns:
                # (a) direct comparison at the simulated grid, raw replicate rows
                m = calc_metrics(raw["expt_Hvap"], raw["sim_Hvap"])
                out_rows.append({
                    "molecule": mol, "ff": ff_label, "property": "Hvap", "method": "direct_sim_grid_raw_replicates",
                    **m, "notes": f"n={m['n_points']} raw rows where expt_Hvap dict key coincided with a simulated T",
                })

                # (b) direct comparison at the simulated grid, per-T mean
                grp = raw.groupby("temperature").agg(sim_Hvap=("sim_Hvap", "mean"), expt_Hvap=("expt_Hvap", "first"))
                m2 = calc_metrics(grp["expt_Hvap"], grp["sim_Hvap"])
                out_rows.append({
                    "molecule": mol, "ff": ff_label, "property": "Hvap", "method": "direct_sim_grid_per_T_mean",
                    **m2, "notes": f"n={m2['n_points']} simulated temperatures with a coinciding expt_Hvap key",
                })

                # (c) Clausius-Clapeyron reimplementation at the EXPERIMENTAL Hvap grid
                pvap_grp = raw.groupby("temperature")["sim_Pvap"].mean()
                hvap_grp = raw.groupby("temperature")["sim_Hvap"].mean()
                sim_pvap_by_T = pvap_grp.to_dict()
                sim_hvap_by_T = hvap_grp.to_dict()

                hvap_expt_rows = student_values[(student_values.molecule == mol) & (student_values.ff == ff_label)
                                                 & (student_values.property == "Hvap")]
                cc_ests, cc_expts, methods = [], [], []
                for _, r in hvap_expt_rows.iterrows():
                    Texp = float(r["temperature_K"])
                    val, method = get_hvap_est_reimplemented(sim_pvap_by_T, sim_hvap_by_T, Texp, MOL_WT[mol])
                    cc_ests.append(val)
                    cc_expts.append(r["expt_reference_value"])
                    methods.append(method)
                m3 = calc_metrics(cc_expts, cc_ests)
                method_counts = pd.Series(methods).value_counts().to_dict()
                out_rows.append({
                    "molecule": mol, "ff": ff_label, "property": "Hvap", "method": "CC_expt_grid_reimplemented",
                    **m3, "notes": f"independent reimpl. of Clausius-Clapeyron bracket search; method counts: {method_counts}",
                })

                # cross-check against the repo's own Hvap_estimates.csv-derived values (already in student_ff_values.csv)
                repo_vals = hvap_expt_rows["sim_value"].astype(float).values
                mine_vals = np.array(cc_ests, dtype=float)
                both_finite = np.isfinite(repo_vals) & np.isfinite(mine_vals)
                if both_finite.sum() > 0:
                    max_abs_diff = float(np.nanmax(np.abs(repo_vals[both_finite] - mine_vals[both_finite])))
                else:
                    max_abs_diff = np.nan
                out_rows.append({
                    "molecule": mol, "ff": ff_label, "property": "Hvap", "method": "CC_vs_repo_crosscheck",
                    "mapd": np.nan, "mae": max_abs_diff, "mse": np.nan, "mpd": np.nan,
                    "n_points": int(both_finite.sum()),
                    "notes": f"max abs diff (kJ/kg) between my reimplemented CC estimate and the repo's "
                             f"Hvap_estimates.csv value, over {int(both_finite.sum())} overlapping points "
                             f"where both are finite -- should be ~0 if my reimplementation is faithful",
                })

    fields = ["molecule", "ff", "property", "method", "mapd", "mae", "mse", "mpd", "n_points", "notes"]
    out_path = f"{AUDIT}/audit_data/student_ff_recompute.csv"
    with open(out_path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {len(out_rows)} rows to {out_path}")


if __name__ == "__main__":
    main()

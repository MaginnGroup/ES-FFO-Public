"""
Worker Task 02 (WP-A2): extract student-FF (GP-Optimized & Base) simulated values
and the repo's own reported error metrics.

Reads from the repo /groups/ed/group_members/Montana_Carlozo/ES-FFO and writes to
/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit.

Property name mapping (repo column suffix -> audit property):
    liq_density -> rho_l
    vap_density -> rho_v
    Pvap        -> Pvap
    Hvap        -> Hvap   (see note below: two distinct grids)
    surf_tens   -> gamma
    Tc          -> Tc
    rhoc        -> rho_c

FF folder mapping:
    ms_val_opt    -> GP-Optimized
    ms_val_no_opt -> Base

Hvap has two distinct data sources in the repo:
  (a) ms_data.csv's own sim_Hvap column, on the SIMULATED temperature grid (direct MD
      liquid-enthalpy minus vapor-enthalpy calculation at each simulated state point).
  (b) Hvap_estimates.csv, on the EXPERIMENTAL temperature grid (which is often a
      different, denser/sparser grid than the simulated one -- e.g. DMSO expt Hvap at
      308/318/320/340/368 K vs simulated at 313.15/318.15/323.15/328.15/333.15 K).
      This file estimates Hvap at the experimental T via Clausius-Clapeyron using the
      simulated Pvap curve when the experimental T doesn't coincide with a simulated T
      (H_method = "estimated"), uses the direct simulated value when it does coincide
      (H_method = "direct"), or gives up entirely when no nearby simulated data exists
      to interpolate from (H_method = "skipped", est_Hvap = NaN).
This is the manuscript's actual Table 7 Hvap comparison mechanism (AUDIT_PLAN F1/F4), so
Hvap rows in student_ff_values.csv are built from Hvap_estimates.csv (grid b), not from
ms_data.csv's direct sim_Hvap column. The direct sim_Hvap-vs-simulated-T values are
still captured for the record in a `direct_sim_hvap` sidecar block (see NOTES in the
findings doc) but are not primary rows to avoid double-counting -- see findings/02.
"""
import os
import csv
import math
import pandas as pd

# REPO = "/Users/adowling/DowlingLab/ES-FFO-Public"
# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
REPO = "/groups/ed/group_members/Montana_Carlozo/ES-FFO"
AT00 = f"{REPO}/Opt_ES/analysis/at_00"

MOLECULES = ["MeOH", "EG", "Gly", "DMF", "DMSO", "DEC"]
FF_DIRS = {"GP-Optimized": "ms_val_opt", "Base": "ms_val_no_opt"}

PROP_MAP = {
    "liq_density": ("rho_l", "kg/m3"),
    "vap_density": ("rho_v", "kg/m3"),
    "Pvap": ("Pvap", "bar"),
    "surf_tens": ("gamma", "mN/m"),
}
HVAP_UNIT = "kJ/kg"
CRIT_UNITS = {"Tc": "K", "rho_c": "kg/m3"}

HMETHOD_MAP = {"direct": "direct", "estimated": "CC", "skipped": "na"}


def rel(path):
    return os.path.relpath(path, REPO)


def extract_ms_data_rows(mol, ff_label, ff_dir):
    path = f"{AT00}/{mol}/ExpVal/opt_res/{ff_dir}/ms_data.csv"
    rows = []
    if not os.path.exists(path):
        return rows, None
    df = pd.read_csv(path, index_col=0)

    # per-temperature properties (rho_l, rho_v, Pvap, gamma) -- average over restarts
    for repo_prop, (prop, unit) in PROP_MAP.items():
        sim_col = f"sim_{repo_prop}"
        expt_col = f"expt_{repo_prop}"
        if sim_col not in df.columns:
            continue
        grp = df.groupby("temperature")
        for T, g in grp:
            sim_vals = g[sim_col].dropna()
            expt_vals = g[expt_col].dropna() if expt_col in g.columns else pd.Series(dtype=float)
            n_restart = len(g)
            n_sim_valid = len(sim_vals)
            if n_sim_valid == 0:
                sim_mean, sim_std = float("nan"), float("nan")
                note = f"all {n_restart} restarts NaN for {sim_col} at T={T}"
            else:
                sim_mean = sim_vals.mean()
                sim_std = sim_vals.std() if n_sim_valid > 1 else 0.0
                note = f"mean of {n_sim_valid}/{n_restart} restarts (std={sim_std:.4g})"
            expt_val = expt_vals.iloc[0] if len(expt_vals) else float("nan")
            rows.append({
                "molecule": mol, "ff": ff_label, "property": prop,
                "temperature_K": T, "sim_value": sim_mean, "sim_unit": unit,
                "expt_reference_value": expt_val, "expt_unit": unit,
                "hvap_method": "na", "source_file": rel(path), "notes": note,
            })

    # critical points: single value per ff/molecule (should be constant across all rows)
    for repo_prop, prop in [("Tc", "Tc"), ("rhoc", "rho_c")]:
        sim_col = f"sim_{repo_prop}"
        expt_col = f"expt_{repo_prop}"
        if sim_col not in df.columns:
            continue
        sim_unique = df[sim_col].dropna().unique()
        expt_unique = df[expt_col].dropna().unique() if expt_col in df.columns else []
        sim_val = sim_unique[0] if len(sim_unique) else float("nan")
        expt_val = expt_unique[0] if len(expt_unique) else float("nan")
        note = "single fitted critical-point value (constant across all restarts/temperatures)"
        if len(sim_unique) > 1:
            note += f" -- WARNING: {len(sim_unique)} distinct sim_{repo_prop} values found, used first: {sim_unique}"
        rows.append({
            "molecule": mol, "ff": ff_label, "property": prop,
            "temperature_K": "", "sim_value": sim_val, "sim_unit": CRIT_UNITS[prop],
            "expt_reference_value": expt_val, "expt_unit": CRIT_UNITS[prop],
            "hvap_method": "na", "source_file": rel(path), "notes": note,
        })

    return rows, df


def extract_hvap_estimates_rows(mol, ff_label, ff_dir):
    path = f"{AT00}/{mol}/ExpVal/opt_res/{ff_dir}/Hvap_estimates.csv"
    rows = []
    if not os.path.exists(path):
        return rows
    df = pd.read_csv(path, index_col=0)
    for _, r in df.iterrows():
        method_raw = r["H_method"]
        method = HMETHOD_MAP.get(method_raw, method_raw)
        sim_val = r["est_Hvap"]
        note = f"H_method(repo)={method_raw}"
        if method_raw == "skipped":
            note += "; no simulated Pvap data close enough to interpolate -- sim_value is NaN"
        elif method_raw == "estimated":
            note += "; Clausius-Clapeyron estimate from simulated Pvap curve, evaluated at expt T"
        elif method_raw == "direct":
            note += "; expt T coincided with a simulated T, direct MD sim_Hvap used"
        rows.append({
            "molecule": mol, "ff": ff_label, "property": "Hvap",
            "temperature_K": r["t_exp"], "sim_value": sim_val, "sim_unit": HVAP_UNIT,
            "expt_reference_value": r["expt_Hvap"], "expt_unit": HVAP_UNIT,
            "hvap_method": method, "source_file": rel(path), "notes": note,
        })
    return rows


def load_comp_err_data():
    path = f"{REPO}/Opt_ES/analysis/comp_err_data.csv"
    df = pd.read_csv(path, index_col=0)
    return df, path


def load_per_molecule_error_data(mol, ff_dir):
    path = f"{AT00}/{mol}/ExpVal/opt_res/{ff_dir}/error_data.csv"
    if not os.path.exists(path):
        return None, path
    df = pd.read_csv(path, index_col=0)
    return df, path


def count_n_points(mol, ff_dir, repo_prop):
    """Count valid (non-NaN) data points actually used for a given property's error calc."""
    if repo_prop == "Hvap":
        path = f"{AT00}/{mol}/ExpVal/opt_res/{ff_dir}/Hvap_estimates.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        return int((df["H_method"] != "skipped").sum())
    elif repo_prop in ("Tc", "rhoc"):
        path = f"{AT00}/{mol}/ExpVal/opt_res/{ff_dir}/ms_data.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        col = f"expt_{repo_prop}"
        return 1 if (col in df.columns and df[col].notna().any()) else 0
    else:
        path = f"{AT00}/{mol}/ExpVal/opt_res/{ff_dir}/ms_data.csv"
        if not os.path.exists(path):
            return None
        df = pd.read_csv(path, index_col=0)
        expt_col = f"expt_{repo_prop}"
        sim_col = f"sim_{repo_prop}"
        if expt_col not in df.columns or sim_col not in df.columns:
            return None
        valid = df.groupby("temperature").apply(
            lambda g: g[expt_col].notna().any() and g[sim_col].notna().any()
        )
        return int(valid.sum())


def main():
    all_value_rows = []
    all_error_rows = []

    comp_err_df, comp_err_path = load_comp_err_data()
    ref_name_map = {"GP-Optimized": "Opt FF", "Base": "IFT FF"}
    error_prop_map = [
        ("liq_density", "rho_l"), ("vap_density", "rho_v"), ("Pvap", "Pvap"),
        ("Hvap", "Hvap"), ("surf_tens", "gamma"), ("Tc", "Tc"), ("rhoc", "rho_c"),
    ]

    for mol in MOLECULES:
        for ff_label, ff_dir in FF_DIRS.items():
            ms_rows, ms_df = extract_ms_data_rows(mol, ff_label, ff_dir)
            hvap_rows = extract_hvap_estimates_rows(mol, ff_label, ff_dir)
            all_value_rows.extend(ms_rows)
            all_value_rows.extend(hvap_rows)

            # errors: prefer comp_err_data.csv (consolidated, ref_name-labeled)
            ref_name = ref_name_map[ff_label]
            sub = comp_err_df[(comp_err_df["molecule"] == mol) & (comp_err_df["ref_name"] == ref_name)]
            per_mol_err_df, per_mol_err_path = load_per_molecule_error_data(mol, ff_dir)

            if len(sub) == 1:
                row = sub.iloc[0]
                source = rel(comp_err_path)
            elif per_mol_err_df is not None:
                row = per_mol_err_df.iloc[0]
                source = rel(per_mol_err_path)
            else:
                row = None
                source = "NOT FOUND"

            for repo_prop, prop in error_prop_map:
                mapd_col, mae_col, mse_col, mpd_col = (
                    f"mapd_{repo_prop}", f"mae_{repo_prop}", f"mse_{repo_prop}", f"mpd_{repo_prop}",
                )
                if row is None:
                    mapd = mae = mse = mpd = ""
                    note_source = "NOT FOUND"
                else:
                    def g(col):
                        if col in row.index:
                            v = row[col]
                            return "" if (isinstance(v, float) and math.isnan(v)) else v
                        return ""
                    mapd, mae, mse, mpd = g(mapd_col), g(mae_col), g(mse_col), g(mpd_col)
                    note_source = source
                n_pts = count_n_points(mol, ff_dir, repo_prop)
                all_error_rows.append({
                    "molecule": mol, "ff": ff_label, "property": prop,
                    "mapd": mapd, "mae": mae, "mse": mse, "mpd": mpd,
                    "n_points_used": n_pts if n_pts is not None else "",
                    "source_file": note_source,
                })

    # write student_ff_values.csv
    val_fields = ["molecule", "ff", "property", "temperature_K", "sim_value", "sim_unit",
                  "expt_reference_value", "expt_unit", "hvap_method", "source_file", "notes"]
    with open(f"{AUDIT}/audit_data/student_ff_values.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=val_fields)
        w.writeheader()
        for r in all_value_rows:
            w.writerow(r)

    # write student_ff_errors_repo.csv
    err_fields = ["molecule", "ff", "property", "mapd", "mae", "mse", "mpd", "n_points_used", "source_file"]
    with open(f"{AUDIT}/audit_data/student_ff_errors_repo.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=err_fields)
        w.writeheader()
        for r in all_error_rows:
            w.writerow(r)

    print(f"wrote {len(all_value_rows)} rows to student_ff_values.csv")
    print(f"wrote {len(all_error_rows)} rows to student_ff_errors_repo.csv")


if __name__ == "__main__":
    main()

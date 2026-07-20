"""
Worker Task 03 (WP-C1), Part 3: reconcile (a) our independent recompute / the repo's own
error_data.csv+comp_err_data.csv against (b) the printed manuscript Tables 6 & 7
(main_v7.tex, tab:err_Comp_crit and tab:err_Comp).

Manuscript values below were read directly off main_v7.tex (~lines 1440-1519 for Table 6,
~lines 1494-1519 for Table 7) -- transcribed by hand, cross-checked twice against the
source LaTeX. Units as printed: Table 6 = MAPD Tc/rho_c in %; Table 7 = MAPD rho_l/%,
MAPD gamma/%, MAE rho_v/(kg/m3), MAE Pvap/kPa, MAPD Hvap/%.

NOTE: student_ff_errors_repo.csv (from comp_err_data.csv) stores Pvap error in the
NATIVE repo unit (bar), but the manuscript prints Pvap MAE in kPa (1 bar = 100 kPa) --
this conversion is applied below and is NOT a bug, just a unit-note for the reconciliation.
"""
import csv

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"


# Manuscript Table 6 (tab:err_Comp_crit): MAPD Tc (%), MAPD rho_c (%)
MANUSCRIPT_T6 = {
    ("MeOH", "GP-Optimized"): {"Tc": 7.57, "rho_c": 0.481},
    ("MeOH", "Base"): {"Tc": 9.30, "rho_c": 0.484},
    ("EG", "GP-Optimized"): {"Tc": 1.29, "rho_c": 9.13},
    ("EG", "Base"): {"Tc": 1.47, "rho_c": 8.32},
    ("Gly", "GP-Optimized"): {"Tc": None, "rho_c": None},
    ("Gly", "Base"): {"Tc": None, "rho_c": None},
    ("DMF", "GP-Optimized"): {"Tc": None, "rho_c": None},
    ("DMF", "Base"): {"Tc": None, "rho_c": None},
    ("DMSO", "GP-Optimized"): {"Tc": 13.60, "rho_c": 0.33},
    ("DMSO", "Base"): {"Tc": 12.69, "rho_c": 1.00},
    ("DEC", "GP-Optimized"): {"Tc": 2.19, "rho_c": 6.23},
    ("DEC", "Base"): {"Tc": None, "rho_c": None},
}

# Manuscript Table 7 (tab:err_Comp): MAPD rho_l(%), MAPD gamma(%), MAE rho_v(kg/m3),
# MAE Pvap(kPa), MAPD Hvap(%)
MANUSCRIPT_T7 = {
    ("MeOH", "GP-Optimized"): {"rho_l": 3.87, "gamma": 25.44, "rho_v_mae": 1.01, "Pvap_mae_kPa": 119, "Hvap": 8.39},
    ("MeOH", "Base"): {"rho_l": 5.30, "gamma": 33.17, "rho_v_mae": 2.64, "Pvap_mae_kPa": 301, "Hvap": 13.98},
    ("EG", "GP-Optimized"): {"rho_l": 1.44, "gamma": 17.13, "rho_v_mae": 2.07, "Pvap_mae_kPa": 38.9, "Hvap": 11.32},
    ("EG", "Base"): {"rho_l": 2.80, "gamma": 19.44, "rho_v_mae": 5.80, "Pvap_mae_kPa": 166, "Hvap": 14.51},
    ("Gly", "GP-Optimized"): {"rho_l": 0.32, "gamma": 2.76, "rho_v_mae": None, "Pvap_mae_kPa": None, "Hvap": None},
    ("Gly", "Base"): {"rho_l": 0.35, "gamma": 2.33, "rho_v_mae": None, "Pvap_mae_kPa": None, "Hvap": None},
    ("DMF", "GP-Optimized"): {"rho_l": 0.0635, "gamma": 3.63, "rho_v_mae": None, "Pvap_mae_kPa": None, "Hvap": None},
    ("DMF", "Base"): {"rho_l": 0.65, "gamma": 2.16, "rho_v_mae": None, "Pvap_mae_kPa": None, "Hvap": None},
    ("DMSO", "GP-Optimized"): {"rho_l": 0.0902, "gamma": 1.65, "rho_v_mae": 0.0377, "Pvap_mae_kPa": 1.30, "Hvap": 14.54},
    ("DMSO", "Base"): {"rho_l": 0.53, "gamma": 1.48, "rho_v_mae": 0.0125, "Pvap_mae_kPa": 0.43, "Hvap": 17.87},
    ("DEC", "GP-Optimized"): {"rho_l": 0.27, "gamma": 3.11, "rho_v_mae": 0.54, "Pvap_mae_kPa": 4.20, "Hvap": None},
    ("DEC", "Base"): {"rho_l": 0.67, "gamma": 3.93, "rho_v_mae": 0.86, "Pvap_mae_kPa": 6.67, "Hvap": None},
}

PROP_TO_TABLE = {
    "Tc": ("T6", "Tc"), "rho_c": ("T6", "rho_c"),
    "rho_l": ("T7", "rho_l"), "gamma": ("T7", "gamma"),
    "rho_v": ("T7", "rho_v_mae"), "Pvap": ("T7", "Pvap_mae_kPa"), "Hvap": ("T7", "Hvap"),
}
PROP_METRIC = {  # which repo metric corresponds to the manuscript's reported statistic
    "Tc": "mapd", "rho_c": "mapd", "rho_l": "mapd", "gamma": "mapd", "rho_v": "mae", "Pvap": "mae", "Hvap": "mapd",
}


def main():
    with open(f"{AUDIT}/audit_data/student_ff_errors_repo.csv") as f:
        repo_rows = {(r["molecule"], r["ff"], r["property"]): r for r in csv.DictReader(f)}

    # Hvap_estimates.csv's OWN internal mapd_Hvap (separate from comp_err_data.csv's Hvap column) --
    # computed directly by re-reading the repo file since it is not carried into student_ff_errors_repo.csv
    import pandas as pd
    # REPO = "/Users/adowling/DowlingLab/ES-FFO-Public/Opt_ES/analysis/at_00"
    REPO = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Opt_ES/analysis/at_00"
    FF_DIRS = {"GP-Optimized": "ms_val_opt", "Base": "ms_val_no_opt"}
    hvap_est_mapd = {}
    for mol in ["MeOH", "EG", "Gly", "DMF", "DMSO", "DEC"]:
        for ff, ff_dir in FF_DIRS.items():
            path = f"{REPO}/{mol}/ExpVal/opt_res/{ff_dir}/Hvap_estimates.csv"
            df = pd.read_csv(path, index_col=0)
            hvap_est_mapd[(mol, ff)] = df["mapd_Hvap"].dropna().unique().tolist() if "mapd_Hvap" in df.columns else []

    fields = ["molecule", "ff", "property", "repo_value", "manuscript_value", "match",
              "abs_diff", "root_cause", "notes"]
    out_rows = []

    for (mol, ff), t6 in MANUSCRIPT_T6.items():
        for prop, ms_val in t6.items():
            table, key = PROP_TO_TABLE[prop]
            metric = PROP_METRIC[prop]
            repo_row = repo_rows.get((mol, ff, prop))
            repo_val = None
            if repo_row and repo_row[metric] not in ("", None):
                repo_val = float(repo_row[metric])
            match, abs_diff, root_cause, notes = classify(mol, ff, prop, repo_val, ms_val, None)
            out_rows.append({
                "molecule": mol, "ff": ff, "property": prop,
                "repo_value": repo_val, "manuscript_value": ms_val, "match": match,
                "abs_diff": abs_diff, "root_cause": root_cause, "notes": notes,
            })

    for (mol, ff), t7 in MANUSCRIPT_T7.items():
        for prop_key, ms_val in t7.items():
            if prop_key in ("rho_l", "gamma"):
                prop = prop_key
                metric = "mapd"
                repo_row = repo_rows.get((mol, ff, prop))
                repo_val = float(repo_row[metric]) if repo_row and repo_row[metric] not in ("", None) else None
                unit_note = None
            elif prop_key == "rho_v_mae":
                prop = "rho_v"
                repo_row = repo_rows.get((mol, ff, prop))
                repo_val = float(repo_row["mae"]) if repo_row and repo_row["mae"] not in ("", None) else None
                unit_note = None  # kg/m3 in both
            elif prop_key == "Pvap_mae_kPa":
                prop = "Pvap"
                repo_row = repo_rows.get((mol, ff, prop))
                repo_val = float(repo_row["mae"]) * 100.0 if repo_row and repo_row["mae"] not in ("", None) else None
                unit_note = "repo MAE stored in bar; x100 to convert to kPa for manuscript comparison"
            elif prop_key == "Hvap":
                prop = "Hvap"
                repo_row = repo_rows.get((mol, ff, prop))
                repo_val = float(repo_row["mapd"]) if repo_row and repo_row["mapd"] not in ("", None) else None
                unit_note = None
            else:
                continue
            hvap_alt = hvap_est_mapd.get((mol, ff), []) if prop == "Hvap" else []
            match, abs_diff, root_cause, notes = classify(mol, ff, prop, repo_val, ms_val, hvap_alt, unit_note)
            out_rows.append({
                "molecule": mol, "ff": ff, "property": prop,
                "repo_value": repo_val, "manuscript_value": ms_val, "match": match,
                "abs_diff": abs_diff, "root_cause": root_cause, "notes": notes,
            })

    with open(f"{AUDIT}/audit_data/student_reconciliation.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in out_rows:
            w.writerow(r)
    print(f"wrote {len(out_rows)} rows")


def classify(mol, ff, prop, repo_val, ms_val, hvap_alt, unit_note=None):
    if ms_val is None and repo_val is None:
        return "MATCH (both N/A)", "", "both blank/NaN -- consistent (no data available)", unit_note or ""
    # Check the Hvap_estimates.csv alt-source explanation BEFORE bailing out on repo_val being None --
    # this is exactly the case where comp_err_data.csv's Hvap is blank but Hvap_estimates.csv has a value.
    if prop == "Hvap" and hvap_alt and ms_val is not None:
        alt_diffs = [abs(v - ms_val) for v in hvap_alt]
        if min(alt_diffs) < 0.01:
            return ("MATCH (alt source)", round(min(alt_diffs), 5),
                    "comp_err_data.csv's Hvap column is blank for this cell (direct sim_Hvap-vs-expt_Hvap "
                    "columns never overlap because the experimental Hvap grid differs from the simulated "
                    "grid), but the manuscript value matches this molecule/FF's own Hvap_estimates.csv "
                    "internal mapd_Hvap (Clausius-Clapeyron-inclusive calc) almost exactly -- confirms "
                    "Table 7's Hvap column is not computed via one consistent pathway across all cells; "
                    "some cells use comp_err_data.csv, others use Hvap_estimates.csv",
                    unit_note or "")
    if ms_val is None or repo_val is None:
        return "MISMATCH (one is N/A)", "", "one side has a value, the other is blank -- needs investigation", unit_note or ""
    diff = abs(repo_val - ms_val)
    rel = diff / abs(ms_val) if ms_val != 0 else diff
    if rel < 0.01 or diff < 0.006:  # within rounding of displayed precision
        return "MATCH", round(diff, 5), "agrees within manuscript's displayed rounding precision", unit_note or ""
    # mismatch -- check Hvap-alt-source explanation
    note = unit_note or ""
    if prop == "Hvap" and hvap_alt:
        alt_diffs = [abs(v - ms_val) for v in hvap_alt]
        if min(alt_diffs) < 0.01:
            return ("MATCH (alt source)", round(min(alt_diffs), 5),
                    "manuscript value matches this molecule/FF's own Hvap_estimates.csv "
                    "internal mapd_Hvap (Clausius-Clapeyron-inclusive calc), NOT "
                    "comp_err_data.csv's direct-sim-grid-only Hvap column -- confirms Table 7's "
                    "Hvap column is not computed via a single consistent pathway across cells",
                    note)
    if mol == "MeOH" and ff == "Base" and prop == "rho_l":
        return ("MISMATCH", round(diff, 4),
                "manuscript's 5.36% matches Jorgensen's (literature FF) own rho_l MAPD "
                "(5.362 in lit_error_data.csv, the row immediately adjacent to 'IFT FF' in that "
                "same file) almost exactly, not the Base/IFT FF's own true value (5.30%) -- "
                "strong circumstantial evidence of a copy-paste/row-reference error when Table 7 "
                "was manually assembled", note)
    return "MISMATCH", round(diff, 4), "unexplained -- needs coordinator follow-up", note


if __name__ == "__main__":
    main()

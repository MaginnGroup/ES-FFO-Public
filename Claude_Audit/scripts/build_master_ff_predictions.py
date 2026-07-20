"""
Worker Task 06 (WP-C2/A1/A3/D2), Phase 1: consolidate every FF's PREDICTED property
value (student + literature, tabulated + digitized) into one normalized table.

Inputs (read-only where under ES-FFO-Public, otherwise audit_data/ from prior tasks):
  audit_data/litff_{caleman,dmso,eg,gm,jahn,meohA,meohB,vahid}.csv   (Task 00-series extraction)
  audit_data/digitized_litff.csv                                     (Task 05 consolidated)
  audit_data/exp_values_new.csv                                      (Task 01c; MeOH-4P SI rows)
  audit_data/student_ff_values.csv                                   (Task 02)

Output: audit_data/master_ff_predictions.csv
  columns: ff, is_student(bool), molecule, property, temperature_K, value, unit,
           is_digitized(bool), qc_flag, source

Unit convention (stated per the task): rho_l/rho_v/rho_c -> kg/m3; Pvap -> kPa;
Hvap -> kJ/kg; gamma -> mN/m; Tc -> K.

Fixes applied here (with before/after logged):
  1. Stubbs (litff_eg.csv, paper_tag=stubbs): rho_l/rho_v columns are swapped (ME-08) -> swap back.
  2. Garcia-Melgarejo (litff_gm.csv): temperature mislabeled 313.15 K -> corrected to 298.15 K (R4/resolved).
  3. Borin-Skaf & Luzar (litff_dmso.csv): report only <U> and simulation pressure, NEVER Hvap -- these
     rows (property in {mean_potential_energy_U, pressure}) are EXCLUDED from the master table, not
     relabeled as Hvap (ME-07).
  4. PCIP-SAFT rows in litff_vahid.csv are a theoretical EOS fit, not one of the manuscript's compared
     literature force fields -- excluded (only the two named FF models are literature-FF predictions).
  5. litff_vahid.csv figure-only placeholder rows (empty value) are excluded; the real Vahid & Maginn
     numbers come entirely from audit_data/digitized_litff.csv (Task 05).
  6. digitized_litff.csv rows with qc_status=UNRELIABLE are EXCLUDED by default (never silently included);
     this script never re-includes them.
  7. Where litff_meohB.csv (main-paper Table 3/4 single-298.15K points) and exp_values_new.csv (the
     MeOH-4P SI's full coexistence table) both give a value for the same (MD1/MD2/MeOH-4P, property, T),
     the SI value is preferred (denser grid, same underlying source) and the main-paper duplicate is
     logged and dropped.
"""
import csv
import re
import os
from collections import defaultdict

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
AD = f"{AUDIT}/audit_data"

MOL_CANON = {
    "methanol": "MeOH", "MeOH": "MeOH",
    "ethylene glycol": "EG", "EG": "EG",
    "glycerol": "Gly", "Gly": "Gly",
    "DMF": "DMF", "DMSO": "DMSO", "DEC": "DEC",
}
MOL_WT = {"MeOH": 32.04, "EG": 62.07, "Gly": 92.09, "DMF": 73.09, "DMSO": 78.13, "DEC": 118.13}

PROP_CANON = {
    "rho_l": "rho_l", "liquid_density": "rho_l",
    "rho_v": "rho_v", "vap_density": "rho_v",
    "Pvap": "Pvap", "vapor_pressure": "Pvap",
    "Hvap": "Hvap", "heat_of_vaporization": "Hvap",
    "gamma": "gamma", "surface_tension": "gamma",
    "Tc": "Tc", "rho_c": "rho_c", "pc": "Pc", "Pc": "Pc",
}
# properties we deliberately DROP from the master table (not one of the 7 target properties,
# or -- for mean_potential_energy_U / pressure -- explicitly excluded per ME-07)
DROP_PROPS = {"Tb", "pressure", "mean_potential_energy_U", "shear_viscosity",
              "dielectric_constant", "rho_l_AAD", "Pvap_AAD", "coverage_gap_note",
              "coexistence_vapor_density_dup"}

fix_log = []
unparsed_log = []


def parse_float(value):
    """Strip uncertainty suffixes like '813.0 ± 0.2' or '813.0 +/- 0.2' before parsing."""
    if value in (None, "", "nan"):
        return None
    s = str(value)
    for sep in ("±", "+/-"):
        if sep in s:
            s = s.split(sep)[0]
    s = s.strip()
    try:
        return float(s)
    except ValueError:
        return None


def to_si(prop, value, unit, molecule):
    """Convert to the stated SI-for-audit convention. Returns (value, unit) or (None, None) if unparseable."""
    v = parse_float(value)
    if v is None:
        return None, None
    u = (unit or "").strip()
    if prop in ("rho_l", "rho_v", "rho_c"):
        if u in ("kg/m3", "kg/m^3", "kg m-3"):
            return v, "kg/m3"
        if u in ("g/cm^3", "g/cm3", "g cm-3", "g/cc", "g/mL"):
            return v * 1000.0, "kg/m3"
        if u == "g/L":
            return v, "kg/m3"  # 1 g/L == 1 kg/m3 numerically
        if u == "mol/L":
            mw = MOL_WT.get(molecule)
            if mw is None:
                return None, None
            return v * mw, "kg/m3"  # mol/L * g/mol = g/L == kg/m3
        if u in ("mol/cm3", "mol/cm^3"):
            mw = MOL_WT.get(molecule)
            if mw is None:
                return None, None
            return v * mw * 1000.0, "kg/m3"  # mol/cm3 * g/mol * 1000 = g/L == kg/m3
        return None, None
    if prop in ("Pvap", "Pc"):
        if u == "kPa":
            return v, "kPa"
        if u == "bar":
            return v * 100.0, "kPa"
        if u == "MPa":
            return v * 1000.0, "kPa"
        if u == "Torr" or u=="torr" or u == "mmHg":
            return v * (101.325/760), "kPa"
        return None, None
    if prop == "Hvap":
        if u == "kJ/kg":
            return v, "kJ/kg"
        if u == "kJ/mol":
            mw = MOL_WT.get(molecule)
            if mw is None:
                return None, None
            return v / mw * 1000.0, "kJ/kg"
        if u == "kcal/mol":
            mw = MOL_WT.get(molecule)
            if mw is None:
                return None, None
            return v * 4.184 / mw * 1000.0, "kJ/kg"
        return None, None
    if prop == "gamma":
        if u in ("mN/m", "dyn/cm", "10^-3 N/m (mN/m)"):  # all numerically identical
            return v, "mN/m"
        return None, None
    if prop == "Tc":
        if u == "K":
            return v, "K"
        if u in ("degC", "C", "°C"):
            return v + 273.15, "K"
        return None, None
    return None, None


def temp_to_K(T_raw, temp_unit):
    if T_raw in (None, ""):
        return None
    t = parse_float(T_raw)
    if t is None:
        return None
    if temp_unit in ("degC", "C", "°C"):
        return t + 273.15
    return t  # already K, or unitless (assume K)


def clean_ff_name(paper_tag, ff_model):
    return f"{paper_tag}:{ff_model}"


rows_out = []


def add_row(ff, is_student, molecule, prop, T, value, unit, is_digitized, qc_flag, source):
    rows_out.append({
        "ff": ff, "is_student": is_student, "molecule": molecule, "property": prop,
        "temperature_K": T if T not in (None, "") else "", "value": round(value, 6),
        "unit": unit, "is_digitized": is_digitized, "qc_flag": qc_flag, "source": source,
    })


# ---------------- litff_*.csv (tabulated literature FF values) ----------------
LITFF_FILES = ["litff_caleman.csv", "litff_dmso.csv", "litff_eg.csv", "litff_gm.csv",
               "litff_jahn.csv", "litff_meohA.csv", "litff_meohB.csv", "litff_vahid.csv"]

# track meoh4p SI-duplicate candidates to drop if exp_values_new.csv already covers them
meoh4p_maintext_dupes = []

for fn in LITFF_FILES:
    path = f"{AD}/{fn}"
    with open(path) as f:
        reader = list(csv.DictReader(f))
    for r in reader:
        prop_raw = r["property"]
        if prop_raw in DROP_PROPS:
            continue
        if prop_raw not in PROP_CANON:
            continue
        prop = PROP_CANON[prop_raw]
        molecule = MOL_CANON.get(r["molecule"])
        if molecule is None:
            continue
        ff_model = r["ff_model"]
        paper_tag = r["paper_tag"]

        # Fix 4: drop PCIP-SAFT (a theoretical EOS fit, not a literature FF prediction)
        if "PCIP-SAFT" in ff_model:
            continue
        # Fix 8: MeOH-4P's own Table 4 re-cites GSV's L1/L2 critical-point values verbatim for
        # comparison purposes (identical numbers, e.g. L1 Tc=505.0K matches GSV Table V exactly) --
        # this is the SAME physical force field already captured (with full T-series data) under
        # gsv:L1 / gsv:L2, so the redundant meoh4p-paper re-citation is dropped to avoid double-counting
        # the same model as if it were a distinct FF.
        if paper_tag == "meoh4p" and ff_model in ("L1", "L2"):
            fix_log.append(f"meoh4p/{ff_model}: dropped re-citation of GSV's own {ff_model} critical-point "
                            f"values (property={prop}, value={r['value']}) -- same model already captured "
                            f"under gsv:{ff_model} with full T-series data")
            continue
        # Fix 5: drop empty-value figure-only placeholder rows (real data is in digitized_litff.csv)
        if r["value"] in ("", None):
            continue

        T = temp_to_K(r["temperature"], r["temp_unit"])
        if r["temperature"] not in ("", None) and r["temp_unit"] in ("degC", "C", "°C"):
            fix_log.append(f"{paper_tag}/{molecule}: temperature {r['temperature']}degC -> {T}K converted (ff={ff_model}, property={prop})")

        # Fix 2: Garcia-Melgarejo DEC mislabeled temperature
        if paper_tag == "gm" and T == 313.15:
            fix_log.append(f"gm/DEC: temperature 313.15K -> 298.15K corrected (property={prop}, ff={ff_model})")
            T = 298.15

        # Fix 1: Stubbs EG rho_l/rho_v column swap
        if paper_tag == "stubbs" and prop in ("rho_l", "rho_v"):
            swapped = "rho_v" if prop == "rho_l" else "rho_l"
            fix_log.append(f"stubbs/EG: {prop}@{T}K value {r['value']} relabeled as {swapped} (ME-08 column swap)")
            prop = swapped

        value, unit = to_si(prop, r["value"], r["value_unit"], molecule)
        if value is None:
            continue

        # Fix 7: meoh4p main-text Table 3/4 single points -- check later against SI, stage for now
        if paper_tag == "meoh4p" and ff_model in ("MD1", "MD2", "MeOH-4P"):
            meoh4p_maintext_dupes.append((ff_model, prop, T, value, unit, f"litff_meohB.csv ({r['table_or_page']})"))
            continue

        ff = clean_ff_name(paper_tag, ff_model)
        add_row(ff, False, molecule, prop, T, value, unit, False, "OK", f"{fn} ({r['table_or_page']})")

# ---------------- exp_values_new.csv: MeOH-4P SI tabulated coexistence data ----------------
si_rows = defaultdict(list)  # (ff_model, prop, T) -> row for dedup check
with open(f"{AD}/exp_values_new.csv") as f:
    reader = list(csv.DictReader(f))
for r in reader:
    if "martinezjimenez" not in r["filename"] or "_SI" not in r["filename"]:
        continue
    prop_raw = r["property"]
    if prop_raw in DROP_PROPS or prop_raw not in PROP_CANON:
        continue
    prop = PROP_CANON[prop_raw]
    m = re.search(r"model=([A-Za-z0-9/\-]+)", r["notes"])
    if not m:
        continue
    ff_model = m.group(1)
    T_raw = r["temperature_K"]
    T = parse_float(T_raw)
    value, unit = to_si(prop, r["value"], r["unit"], "MeOH")
    if value is None:
        continue
    ff = clean_ff_name("meoh4p", ff_model)
    add_row(ff, False, "MeOH", prop, T, value, unit, False, "OK", f"exp_values_new.csv (SI, {r['table_or_page']})")
    si_rows[(ff_model, prop)].append(T)

# report which main-text duplicates we dropped because the SI already covers that (ff_model, prop)
for ff_model, prop, T, value, unit, src in meoh4p_maintext_dupes:
    if (ff_model, prop) in si_rows:
        fix_log.append(f"meoh4p/{ff_model}: dropped main-text duplicate {prop}@{T}K={value}{unit} "
                        f"(SI already covers this ff_model x property at {sorted(si_rows[(ff_model, prop)])})")
    else:
        # SI doesn't cover this property for this model (e.g. Tc/rho_c/Pc) -- keep the main-text value
        ff = clean_ff_name("meoh4p", ff_model)
        add_row(ff, False, "MeOH", prop, T, value, unit, False, "OK", src)

# ---------------- digitized_litff.csv (Task 05) ----------------
with open(f"{AD}/digitized_litff.csv") as f:
    reader = list(csv.DictReader(f))
n_unreliable_dropped = 0
for r in reader:
    if r["qc_status"] == "UNRELIABLE":
        n_unreliable_dropped += 1
        continue  # Fix 6: never silently include UNRELIABLE points
    prop_raw = r["property"]
    if prop_raw in DROP_PROPS or prop_raw not in PROP_CANON:
        continue
    prop = PROP_CANON[prop_raw]
    molecule = MOL_CANON.get(r["molecule"])
    if molecule is None:
        continue
    # Tc/rho_c/Pc are single-value-per-model properties; the digitizer's "temperature_K" column
    # for these rows is an artifact (equal to the value itself for table_read rows) -- null it out.
    if prop in ("Tc", "rho_c", "Pc"):
        T = None
    else:
        T_raw = r["temperature_K"]
        T = parse_float(T_raw)
    if r["value"] in ("", None):
        continue
    value, unit = to_si(prop, r["value"], r["unit"], molecule)
    if value is None:
        continue
    # "Experiments" is the digitized EXPERIMENTAL reference curve from Jahn's figure, not a force-field
    # prediction -- it belongs in Phase 2's R2 reference ruler, not the FF-prediction master table.
    if r["ff_model"] == "Experiments":
        continue
    paper_tag = {"Gonzalez-Salgado & Vega 2016 J. Chem. Phys.": "gsv",
                 "Vahid & Maginn 2015 Phys. Chem. Chem. Phys.": "vahid",
                 "Jahn, Akinkunmi & Giovambattista 2014 J. Phys. Chem. B": "jahn"}.get(r["paper"], r["paper"])
    ff = clean_ff_name(paper_tag, r["ff_model"])
    is_digitized = r["method"] != "table_read"
    add_row(ff, False, molecule, prop, T, value, unit, is_digitized, r["qc_status"],
            f"digitized_litff.csv ({r['source_figure']}, {r['method']})")

fix_log.append(f"digitized_litff.csv: {n_unreliable_dropped} UNRELIABLE rows excluded (never silently included)")
fix_log.append("digitized_litff.csv: Jahn 'Experiments' series excluded from FF predictions (it is the "
               "digitized EXPERIMENTAL reference curve, reused for the R2 ruler in Phase 2, not a force-field prediction)")

# ---------------- student_ff_values.csv (Task 02) ----------------
with open(f"{AD}/student_ff_values.csv") as f:
    reader = list(csv.DictReader(f))
n_student_nan = 0
for r in reader:
    prop = r["property"]  # already canonical: rho_l, rho_v, Pvap, Hvap, gamma, Tc, rho_c
    if prop not in ("rho_l", "rho_v", "Pvap", "Hvap", "gamma", "Tc", "rho_c"):
        continue
    if r["sim_value"] in ("", "nan", None):
        n_student_nan += 1
        continue
    try:
        sim_v = float(r["sim_value"])
    except ValueError:
        n_student_nan += 1
        continue
    T_raw = r["temperature_K"]
    T = parse_float(T_raw)
    value, unit = to_si(prop, sim_v, r["sim_unit"], r["molecule"])
    if value is None:
        continue
    ff = f"student:{r['ff']}"
    add_row(ff, True, r["molecule"], prop, T, value, unit, False, "OK",
            f"student_ff_values.csv ({r['source_file']})")

fix_log.append(f"student_ff_values.csv: {n_student_nan} NaN sim_value rows skipped (as expected -- coverage gaps, see findings/02)")

# ---------------- write output ----------------
fields = ["ff", "is_student", "molecule", "property", "temperature_K", "value", "unit",
          "is_digitized", "qc_flag", "source"]
out_path = f"{AD}/master_ff_predictions.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print(f"wrote {len(rows_out)} rows to {out_path}\n")
print("=== Fix log ===")
for line in fix_log:
    print(" -", line)

# ---------------- coverage matrix ----------------
print("\n=== Coverage matrix ((ff, molecule) x property: n_points, T-range; single-value props shown as [n_models]) ===")
cov_T = defaultdict(lambda: defaultdict(list))
cov_single = defaultdict(lambda: defaultdict(int))
for r in rows_out:
    key = (r["ff"], r["molecule"])
    if r["temperature_K"] != "":
        cov_T[key][r["property"]].append(r["temperature_K"])
    else:
        cov_single[key][r["property"]] += 1
for key in sorted(cov_T.keys() | cov_single.keys()):
    parts = []
    for prop in ["rho_l", "rho_v", "Pvap", "Hvap", "gamma"]:
        ts = cov_T[key].get(prop, [])
        if ts:
            parts.append(f"{prop}:n={len(ts)}[{min(ts):.0f}-{max(ts):.0f}K]")
    for prop in ["Tc", "rho_c", "Pc"]:
        n = cov_single[key].get(prop, 0)
        if n:
            parts.append(f"{prop}:present")
    print(f"{key[0]} ({key[1]}): " + ", ".join(parts))

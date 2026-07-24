"""
Worker Task 08: visualize the R1 baseline (CoolProp for MeOH; `thermo` DIPPR correlations for
EG/Gly/DMF/DMSO/DEC) against every experimental data point gathered in this audit, to judge
whether R1 is a fair common ruler before committing to it for Tables 6/7.

Reuses the exact R1 evaluator and unit-conversion conventions validated/fixed in
scripts/build_reference_values_all.py and scripts/recompute_metrics_all.py (WP-C3): rho in
kg/m3, Pvap in kPa, Hvap in kJ/kg, gamma in mN/m, T in K.

Data sources overlaid (each a distinct legend entry):
  - audit_data/exp_values_new.csv       -- primary-source measurements we downloaded/digitized from PDFs
  - audit_data/litexp_*.csv             -- each FF paper's OWN cited experimental data
  - audit_data/montana_reference_values.csv -- esolvs.py's reference values (the manuscript's own ruler)
  - refprop_output/*.txt                -- Montana's raw REFPROP saturation tables (MeOH, EG only)

Output: figures/baseline_vs_exp_<MOL>.pdf/.png (one multi-panel figure per molecule) and
audit_data/baseline_agreement_stats.csv (the quantitative baseline-vs-experiment agreement table).
"""
import csv
import math
import warnings
import re
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

warnings.filterwarnings("ignore")
from thermo import Chemical
from CoolProp.CoolProp import PropsSI

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"

AD = f"{AUDIT}/audit_data"
FIG = f"{AUDIT}/figures"
import os
os.makedirs(FIG, exist_ok=True)

MOLECULES = ["MeOH", "EG", "Gly", "DMF", "DMSO", "DEC"]
PROPS = ["rho", "Pvap", "Hvap", "gamma"]  # "rho" panel shows both rho_l and rho_v
MOL_WT = {"MeOH": 32.04, "EG": 62.07, "Gly": 92.09, "DMF": 73.09, "DMSO": 78.13, "DEC": 118.13}
TC = {"MeOH": 512.5, "EG": 719.6, "Gly": 850.0, "DMF": 649.6, "DMSO": 718.0, "DEC": 576.0}
RHOC = {"MeOH": 273.846, "EG": 391.9405, "Gly": None, "DMF": 279.204, "DMSO": 366.0, "DEC": 341.42}
CAS = {"MeOH": "67-56-1", "EG": "107-21-1", "Gly": "56-81-5", "DMF": "68-12-2",
       "DMSO": "67-68-5", "DEC": "105-58-8"}
CP_NAME = {"MeOH": "Methanol"}
MOL_CANON = {"methanol": "MeOH", "ethylene glycol": "EG", "glycerol": "Gly",
             "MeOH": "MeOH", "EG": "EG", "Gly": "Gly", "DMF": "DMF", "DMSO": "DMSO", "DEC": "DEC"}

# near-Tc unreliability thresholds for the `thermo` proxy (validated in findings/refprop_validation.md)
def thermo_reliable(prop, T, mol):
    if mol == "MeOH":
        return True  # R1 baseline calls CoolProp directly for MeOH -- validated full range
    # NOTE: EG's R1 baseline still calls the `thermo` package directly (there is no pure-EG
    # CoolProp EOS). refprop_output_verification.md validated esolvs.py's OWN transcribed
    # REFPROP values for EG at 450-700K -- it did NOT show that live thermo.Chemical() calls
    # are accurate there. In fact thermo.rhol(EG) demonstrably clips at ~751 kg/m3 for T>=650K
    # (verified while building this plot: thermo gives rhol=750.81 at both 650K and 700K,
    # whereas the true/REFPROP value at 700K is 602.22) -- so EG gets the SAME near-Tc
    # thermo-proxy caveat as the other four molecules, not a blanket pass.
    tratio = T / TC[mol]
    if prop == "rho_v":
        return tratio <= 0.68
    if prop == "rho_l":
        return tratio <= 0.85
    return tratio <= 0.90


def r1_value(mol, prop, T):
    """R1 baseline: CoolProp for MeOH, thermo DIPPR correlation otherwise."""
    try:
        if mol == "MeOH":
            f = "Methanol"
            if prop == "rho_l":
                return PropsSI("D", "T", T, "Q", 0, f)
            if prop == "rho_v":
                return PropsSI("D", "T", T, "Q", 1, f)
            if prop == "Pvap":
                return PropsSI("P", "T", T, "Q", 0, f) / 1000.0
            if prop == "Hvap":
                return (PropsSI("H", "T", T, "Q", 1, f) - PropsSI("H", "T", T, "Q", 0, f)) / 1000.0
            if prop == "gamma":
                return PropsSI("I", "T", T, "Q", 0, f) * 1000.0
        c = Chemical(CAS[mol], T=float(T))
        if prop == "rho_l":
            return c.rhol
        if prop == "rho_v":
            return Chemical(CAS[mol], T=float(T), P=c.Psat).rhog if c.Psat else None
        if prop == "Pvap":
            return c.Psat / 1000.0 if c.Psat else None
        if prop == "Hvap":
            return c.Hvap / 1000.0 if c.Hvap else None
        if prop == "gamma":
            return c.sigma * 1000.0 if c.sigma else None
    except Exception:
        return None
    return None

def r1_critical(mol, prop):
    """Tc/rho_c from thermo/CoolProp for reference (informational; the manuscript's own
    critical-point provenance per molecule is documented in provenance_map.csv)."""
    try:
        c = Chemical(CAS[mol])
        if prop == "Tc":
            return c.Tc, ("thermo Tc" if mol != "MeOH" else "CoolProp Tc")
        if prop == "rho_c":
            return c.rhoc, ("thermo rho_c" if mol != "MeOH" else "CoolProp rho_c")
    except Exception:
        return None, None
    return None, None

def parse_float(v):
    if v in (None, ""):
        return None
    try:
        return float(str(v).split("±")[0].split("+/-")[0].strip())
    except ValueError:
        return None


def density_to_si(v, unit, mol):
    unit = (unit or "").strip()
    if unit in ("kg/m3", "kg/m^3", "kg m-3"):
        return v
    if unit in ("g/cm^3", "g/cm3", "g cm-3", "g/cc", "g/mL"):
        return v * 1000.0
    if unit == "g/L":
        return v
    if unit in ("mol/L", "mol/l"):
        return v * MOL_WT[mol]
    return None


def pvap_to_si(v, unit):
    unit = (unit or "").strip()
    if unit == "kPa":
        return v
    if unit == "bar":
        return v * 100.0
    if unit == "MPa":
        return v * 1000.0
    if unit == "mmHg" or unit == "Torr":
        return v * 0.133322
    if unit == "Pa":
        return v / 1000.0
    return None


def hvap_to_si(v, unit, mol):
    unit = (unit or "").strip()
    if unit == "kJ/kg":
        return v
    if unit == "kJ/mol":
        return v / MOL_WT[mol] * 1000.0
    if unit == "kcal/mol":
        return v * 4.184 / MOL_WT[mol] * 1000.0
    return None


def gamma_to_si(v, unit):
    unit = (unit or "").strip()
    if unit in ("mN/m", "dyn/cm", "10^-3 N/m (mN/m)"):
        return v
    return None


def temp_to_K(T, unit):
    if T is None:
        return None
    if unit in ("degC", "C", "°C"):
        return T + 273.15
    return T


# ============================================================================
# Load all experimental overlay points into: points[mol][prop] = list of dicts
#   {T, value, series, marker, primary(bool)}
# ============================================================================
points = defaultdict(lambda: defaultdict(list))
series_seen = defaultdict(set)  # mol -> set of series labels actually populated (for legend)


def add_point(mol, prop, T, value, series, primary=True, marker="o"):
    if mol not in MOLECULES or T is None or value is None:
        return
    points[mol][prop].append({"T": T, "value": value, "series": series,
                               "primary": primary, "marker": marker})
    series_seen[mol].add((series, marker, primary))


# ---- 1. exp_values_new.csv: primary-source downloaded/digitized measurements ----
PROP_MAP = {"rho_l": "rho_l", "rho_v": "rho_v", "Pvap": "Pvap", "Hvap": "Hvap",
            "gamma": "gamma", "surface_tension": "gamma"}
with open(f"{AD}/exp_values_new.csv") as f:
    for r in csv.DictReader(f):
        mol = MOL_CANON.get(r["molecule"])
        if mol is None or r["is_figure_only"] == "Y":
            continue
        # exp_values_new.csv also stores the MeOH-4P paper's OWN force-field predictions
        # (MD1/MD2/MeOH-4P SI table, tagged "model=..." in notes -- same rows Task 06's
        # build_master_ff_predictions.py pulls out as FF PREDICTIONS, not experimental data).
        # Excluding them here: plotting a force field's own simulated values as "experiment"
        # would be a mislabeled, misleading overlay for a baseline-vs-EXPERIMENT figure.
        if "model=" in (r["notes"] or ""):
            continue
        prop = PROP_MAP.get(r["property"])
        if prop is None:
            continue
        T = parse_float(r["temperature_K"])
        v = parse_float(r["value"])
        if T is None or v is None:
            continue
        if prop in ("rho_l", "rho_v"):
            v = density_to_si(v, r["unit"], mol)
        elif prop == "Pvap":
            v = pvap_to_si(v, r["unit"])
        elif prop == "Hvap":
            v = hvap_to_si(v, r["unit"], mol)
        elif prop == "gamma":
            v = gamma_to_si(v, r["unit"])
        if v is None:
            continue
        cite = r["citation"].split(";")[0].split(".")[0][:28]
        add_point(mol, prop, T, v, f"exp: {cite}", primary=True, marker="o")

# ---- 2. litexp_*.csv: each FF paper's own cited experimental data ----
import glob
LITEXP_PROP = {"rho_l": "rho_l", "rho_v": "rho_v", "Pvap": "Pvap", "Hvap": "Hvap",
               "gamma": "gamma", "surface_tension": "gamma", "heat_of_vaporization": "Hvap",
               "liquid_density": "rho_l"}
for fn in glob.glob(f"{AD}/litexp_*.csv"):
    file_tag = fn.split("litexp_")[-1].replace(".csv", "")
    with open(fn) as f:
        for r in csv.DictReader(f):
            mol = MOL_CANON.get(r["molecule"])
            if mol is None:
                continue
            # litexp_meohB.csv holds BOTH gsv's and meoh4p's own cited experimental data
            # (two different papers sharing one file) -- use the row's own paper_tag column
            # for the series label, not the filename, or gsv/meoh4p citations get wrongly
            # lumped into one legend entry.
            tag = r.get("paper_tag") or file_tag
            # litexp_meohB.csv includes GSV's own cited HIGH-PRESSURE (5000/20000 bar)
            # compressed-liquid density citations alongside the 1-bar saturation-line ones
            # (both genuinely in the paper's Table VIII). Our R1 baseline here is a 1-bar
            # saturation curve, so a high-pressure point isn't a fair comparison point for
            # THIS figure -- exclude, don't silently mismatch apples-to-oranges.
            blob = ((r.get("notes") or "") + " " + (r.get("exp_source_citation") or "")).lower()
            if "high pressure" in blob:
                continue
            prop = LITEXP_PROP.get(r["property"])
            if prop is None:
                continue
            T = temp_to_K(parse_float(r["temperature"]), r["temp_unit"])
            v = parse_float(r["exp_value"])
            if T is None or v is None:
                continue
            unit = r["value_unit"]
            if prop in ("rho_l", "rho_v"):
                v = density_to_si(v, unit, mol)
            elif prop == "Pvap":
                v = pvap_to_si(v, unit)
            elif prop == "Hvap":
                v = hvap_to_si(v, unit, mol)
            elif prop == "gamma":
                v = gamma_to_si(v, unit)
            if v is None:
                continue
            add_point(mol, prop, T, v, f"{tag} paper's cited exp", primary=True, marker="^")

# ---- 3. montana_reference_values.csv: esolvs.py's own reference (the manuscript's ruler) ----
MRV_PROP = {"liq_density": "rho_l", "vap_density": "rho_v", "Pvap": "Pvap",
            "Hvap": "Hvap", "surf_tens": "gamma"}
with open(f"{AD}/montana_reference_values.csv") as f:
    for r in csv.DictReader(f):
        prop = MRV_PROP.get(r["property"])
        if prop is None or r["temperature_K"] == "":
            continue
        mol = r["molecule"]
        T = parse_float(r["temperature_K"])
        v = parse_float(r["value"])
        unit = r["unit"]
        if prop in ("rho_l", "rho_v"):
            v = density_to_si(v, unit, mol)
        elif prop == "Pvap":
            v = pvap_to_si(v, unit)
        elif prop == "Hvap":
            v = hvap_to_si(v, unit, mol)
        elif prop == "gamma":
            v = gamma_to_si(v, unit)
        add_point(mol, prop, T, v, "esolvs.py (manuscript reference)", primary=True, marker="s")

# ---- 4. Montana's raw REFPROP output (MeOH, EG) -- the actual primary source behind both R1 and esolvs ----
def parse_refprop_txt(path, mol):
    with open(path) as f:
        lines = f.readlines()
    for line in lines:
        parts = line.split()
        if len(parts) >= 7 and re.match(r"^\d+$", parts[0]):
            try:
                T, P, rhol, rhov, hvap, sigma = [float(x) for x in parts[1:7]]
            except ValueError:
                continue
            add_point(mol, "rho_l", T, rhol, "Montana's raw REFPROP output", primary=True, marker="D")
            add_point(mol, "rho_v", T, rhov, "Montana's raw REFPROP output", primary=True, marker="D")
            add_point(mol, "Pvap", T, P * 1000.0, "Montana's raw REFPROP output", primary=True, marker="D")
            add_point(mol, "Hvap", T, hvap, "Montana's raw REFPROP output", primary=True, marker="D")
            add_point(mol, "gamma", T, sigma, "Montana's raw REFPROP output", primary=True, marker="D")

parse_refprop_txt(f"{AUDIT}/refprop_output/methanol_refprop_montana.txt", "MeOH")
parse_refprop_txt(f"{AUDIT}/refprop_output/ethyleneglycol_refprop_montana.txt", "EG")

print("Loaded experimental points per molecule:")
for mol in MOLECULES:
    n = sum(len(v) for v in points[mol].values())
    print(f"  {mol}: {n} points across {len(points[mol])} properties, "
          f"{len(series_seen[mol])} distinct series")

# ============================================================================
# Agreement statistics: baseline vs each experimental point
# ============================================================================
CIRCULAR_MOLS = {"MeOH", "EG"}  # esolvs.py IS (transcribed from) REFPROP for these two --
                                  # comparing R1 vs esolvs here is NOT an independent test (F9/refprop_output_verification.md)

stats_rows = []
for mol in MOLECULES:
    for prop in ("rho_l", "rho_v", "Pvap", "Hvap", "gamma"):
        pts = points[mol].get(prop, [])
        if not pts:
            continue
        by_series = defaultdict(list)
        for p in pts:
            by_series[p["series"]].append(p)
        for series, plist in by_series.items():
            devs = []
            for p in plist:
                base = r1_value(mol, prop, p["T"])
                if base is None or base == 0:
                    continue
                pct = abs(p["value"] - base) / abs(base) * 100.0
                devs.append(pct)
            if not devs:
                continue
            n = len(devs)
            n_1pct = sum(1 for d in devs if d <= 1.0)
            n_5pct = sum(1 for d in devs if d <= 5.0)
            circular = (mol in CIRCULAR_MOLS and series == "esolvs.py (manuscript reference)") or \
                       (mol in CIRCULAR_MOLS and series == "Montana's raw REFPROP output")
            stats_rows.append({
                "molecule": mol, "property": prop, "series": series, "n_points": n,
                "frac_within_1pct": round(n_1pct / n, 3), "frac_within_5pct": round(n_5pct / n, 3),
                "max_dev_pct": round(max(devs), 2), "mean_dev_pct": round(sum(devs) / n, 2),
                "circular_vs_R1": "Y" if circular else "N",
            })

with open(f"{AD}/baseline_agreement_stats.csv", "w", newline="") as f:
    fields = ["molecule", "property", "series", "n_points", "frac_within_1pct", "frac_within_5pct",
              "max_dev_pct", "mean_dev_pct", "circular_vs_R1"]
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in stats_rows:
        w.writerow(r)
print(f"\nwrote {len(stats_rows)} rows to {AD}/baseline_agreement_stats.csv")


# ============================================================================
# Plotting
# ============================================================================
def dense_grid(mol, T_min, T_max, step=3.0):
    n = max(2, int((T_max - T_min) / step))
    return np.linspace(T_min, T_max, n)


def split_reliable(mol, prop, Ts, vals):
    """Split into (reliable_T, reliable_v), (unreliable_T, unreliable_v) for solid/dashed plotting."""
    rel_T, rel_v, unrel_T, unrel_v = [], [], [], []
    for T, v in zip(Ts, vals):
        if v is None or (isinstance(v, float) and math.isnan(v)):
            continue
        if thermo_reliable(prop, T, mol):
            rel_T.append(T); rel_v.append(v)
        else:
            unrel_T.append(T); unrel_v.append(v)
    return (rel_T, rel_v), (unrel_T, unrel_v)


MARKER_STYLE = {}  # (series,) -> matplotlib style dict, assigned on first sight for consistency across panels


def style_for(series, marker, primary, idx):
    cmap = plt.get_cmap("tab10")
    if series not in MARKER_STYLE:
        color = cmap(idx % 10)
        MARKER_STYLE[series] = {"marker": marker, "color": color,
                                 "facecolors": color if primary else "none",
                                 "edgecolors": color}
    return MARKER_STYLE[series]


for mol in MOLECULES:
    MARKER_STYLE.clear()
    all_series = sorted(series_seen[mol])
    fig, axes = plt.subplots(2, 2, figsize=(13, 10))
    fig.suptitle(f"{mol}: R1 baseline vs. all gathered experimental data", fontsize=14, fontweight="bold")

    # ---- determine T range for the dense baseline grid ----
    all_T = [p["T"] for prop_pts in points[mol].values() for p in prop_pts]
    T_min_data = min(all_T) if all_T else 250.0
    T_max = 0.99 * TC[mol]
    T_grid = dense_grid(mol, min(T_min_data, 273.15), T_max, step=3.0)

    baseline_label_used = set()

    # ---- Panel 1: T-rho coexistence envelope (rho_l and rho_v on one panel, T on y-axis) ----
    ax = axes[0, 0]
    rho_l_grid = [r1_value(mol, "rho_l", T) for T in T_grid]
    rho_v_grid = [r1_value(mol, "rho_v", T) for T in T_grid]
    for label_suffix, vals in [("liquid branch", rho_l_grid), ("vapor branch", rho_v_grid)]:
        prop_key = "rho_l" if "liquid" in label_suffix else "rho_v"
        (rt, rv), (ut, uv) = split_reliable(mol, prop_key, T_grid, vals)
        base_name = "R1 baseline (CoolProp)" if mol == "MeOH" else "R1 baseline (thermo)"
        lbl = f"{base_name}, {label_suffix}"
        if rt:
            ax.plot(rv, rt, "-", color="black", lw=1.5, label=lbl if lbl not in baseline_label_used else None)
            baseline_label_used.add(lbl)
        if ut:
            lbl2 = f"{base_name} (near-Tc, unvalidated), {label_suffix}"
            ax.plot(uv, ut, "--", color="grey", lw=1.2, label=lbl2 if lbl2 not in baseline_label_used else None)
            baseline_label_used.add(lbl2)
    if RHOC[mol]:
        ax.plot(RHOC[mol], TC[mol], "*", color="red", markersize=16, label="(Tc, rho_c) [esolvs.py]", zorder=5)

    Tc_R1 = r1_critical(mol, "Tc")
    rho_c_R1 = r1_critical(mol, "rho_c")

    if (Tc_R1 and rho_c_R1):
        ax.plot(rho_c_R1[0], Tc_R1[0], "-", marker = "_", color="black",lw=3, zorder = 7)

    idx = 0
    for series, marker, primary in all_series:
        for prop_key in ("rho_l", "rho_v"):
            pts = [p for p in points[mol].get(prop_key, []) if p["series"] == series]
            if not pts:
                continue
            sty = style_for(series, marker, primary, idx)
            ax.scatter([p["value"] for p in pts], [p["T"] for p in pts], s=28, alpha=0.75,
                       label=series if prop_key == "rho_l" else None, **sty)
        idx += 1
    ax.set_xlabel("rho / kg m$^{-3}$"); ax.set_ylabel("T / K")
    ax.set_title("Coexistence envelope (rho_l, rho_v)")
    ax.legend(fontsize=6, loc="best")

    # ---- Panel 2: ln(Pvap) vs 1000/T ----
    ax = axes[0, 1]
    pvap_grid = [r1_value(mol, "Pvap", T) for T in T_grid]
    (rt, rv), (ut, uv) = split_reliable(mol, "Pvap", T_grid, pvap_grid)
    base_name = "R1 baseline (CoolProp)" if mol == "MeOH" else "R1 baseline (thermo)"
    if rt:
        ax.plot([1000.0 / t for t in rt], [math.log(v) for v in rv], "-", color="black", lw=1.5, label=base_name)
    if ut:
        ax.plot([1000.0 / t for t in ut], [math.log(v) for v in uv], "--", color="grey", lw=1.2,
                label=f"{base_name} (near-Tc, unvalidated)")
    idx = 0
    for series, marker, primary in all_series:
        pts = [p for p in points[mol].get("Pvap", []) if p["series"] == series and p["value"] > 0]
        if pts:
            sty = style_for(series, marker, primary, idx)
            ax.scatter([1000.0 / p["T"] for p in pts], [math.log(p["value"]) for p in pts],
                       s=28, alpha=0.75, label=series, **sty)
        idx += 1
    ax.set_xlabel("1000/T / K$^{-1}$"); ax.set_ylabel("ln(P$_{vap}$ / kPa)")
    ax.set_title("Vapor pressure (Clausius-Clapeyron axes)")
    ax.legend(fontsize=6, loc="best")

    # ---- Panel 3: Hvap vs T ----
    ax = axes[1, 0]
    hvap_grid = [r1_value(mol, "Hvap", T) for T in T_grid]
    (rt, rv), (ut, uv) = split_reliable(mol, "Hvap", T_grid, hvap_grid)
    if rt:
        ax.plot(rt, rv, "-", color="black", lw=1.5, label=base_name)
    if ut:
        ax.plot(ut, uv, "--", color="grey", lw=1.2, label=f"{base_name} (near-Tc, unvalidated)")
    idx = 0
    for series, marker, primary in all_series:
        pts = [p for p in points[mol].get("Hvap", []) if p["series"] == series]
        if pts:
            sty = style_for(series, marker, primary, idx)
            ax.scatter([p["T"] for p in pts], [p["value"] for p in pts], s=28, alpha=0.75,
                       label=series, **sty)
        idx += 1
    ax.set_xlabel("T / K"); ax.set_ylabel("Delta H_vap / kJ kg$^{-1}$")
    ax.set_title("Enthalpy of vaporization")
    ax.legend(fontsize=6, loc="best")

    # ---- Panel 4: gamma vs T ----
    ax = axes[1, 1]
    gamma_grid = [r1_value(mol, "gamma", T) for T in T_grid]
    (rt, rv), (ut, uv) = split_reliable(mol, "gamma", T_grid, gamma_grid)
    if rt:
        ax.plot(rt, rv, "-", color="black", lw=1.5, label=base_name)
    if ut:
        ax.plot(ut, uv, "--", color="grey", lw=1.2, label=f"{base_name} (near-Tc, unvalidated)")
    idx = 0
    for series, marker, primary in all_series:
        pts = [p for p in points[mol].get("gamma", []) if p["series"] == series]
        if pts:
            sty = style_for(series, marker, primary, idx)
            ax.scatter([p["T"] for p in pts], [p["value"] for p in pts], s=28, alpha=0.75,
                       label=series, **sty)
        idx += 1
    ax.set_xlabel("T / K"); ax.set_ylabel("gamma / mN m$^{-1}$")
    ax.set_title("Surface tension")
    ax.legend(fontsize=6, loc="best")

    fig.tight_layout(rect=[0, 0, 1, 0.96])
    fig.savefig(f"{FIG}/baseline_vs_exp_{mol}.pdf")
    fig.savefig(f"{FIG}/baseline_vs_exp_{mol}.png", dpi=150)
    plt.close(fig)
    print(f"wrote figures/baseline_vs_exp_{mol}.pdf/.png")

print("\nDone.")

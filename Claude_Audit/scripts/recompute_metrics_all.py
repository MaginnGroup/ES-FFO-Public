"""
Worker Task 06 (WP-C2/A1/A3/D2), Phase 3: recompute MAPD/MAE for every FF x property,
over ALL of that FF's own reported temperatures (fixing M1), on both R1 and R2 (DP-01),
using one unified Hvap methodology for every FF (fixing M2/F7/F4).

Inputs: audit_data/master_ff_predictions.csv (Phase 1), audit_data/reference_values_all.csv
(Phase 2, generic R2 = montana/esolvs), audit_data/litexp_*.csv (paper-specific R2 overrides).

Hvap policy (unified, applied identically to every FF, student and literature):
  - If the FF reports Hvap directly at a temperature, use that value ("direct").
  - Otherwise, if the FF has >=2 of its own Pvap points bracketing that temperature, estimate
    Hvap via the same Clausius-Clapeyron bracket-search formula used throughout this audit
    (utils/prep_ms_data.py:estimate_hvaps, independently reimplemented -- see findings/03) at
    the FF's OWN temperatures where it has rho_l/Pvap data ("CC").
  - Otherwise, skipped (no data to estimate from).
  This is applied identically whether the FF's own Hvap grid is dense (Chen: direct at 6T) or
  a single point (most GSV models: direct at 298K only, CC-extended across their own Pvap grid).

Metric policy: MAPD for rho_l, gamma, Hvap; MAE for rho_v, Pvap (M4 -- MAPD is unstable near
zero for these two, so it is never reported for them).

R2 policy: for each FF, prefer that FF's OWN paper's cited experimental values from
litexp_<paper_tag>.csv where available at/near the needed temperature; otherwise fall back to
reference_values_all.csv's generic R2 (montana_reference_values.csv / esolvs.py).

Output: audit_data/recomputed_metrics.csv
  columns: ff, molecule, property, ruler(R1/R2), metric_type(MAPD/MAE), value, n_points,
           T_range, hvap_method, caveat_flag
"""
import csv
import math
from collections import defaultdict

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
AD = f"{AUDIT}/audit_data"

MOL_WT = {"MeOH": 32.04, "EG": 62.07, "Gly": 92.09, "DMF": 73.09, "DMSO": 78.13, "DEC": 118.13}
R_KJ_PER_MOL_K = 8.31446261815324e-3

PAPER_TAG_TO_LITEXP = {
    "chen": "litexp_meohA.csv", "jorg": "litexp_meohA.csv",
    "gsv": "litexp_meohB.csv", "meoh4p": "litexp_meohB.csv",
    "huang": "litexp_eg.csv", "stubbs": "litexp_eg.csv",
    "jahn": "litexp_jahn.csv",
    "vahid": "litexp_vahid.csv",
    "senapati": "litexp_dmso.csv", "luzar": "litexp_dmso.csv", "borin": "litexp_dmso.csv",
    "gm": "litexp_gm.csv",
    "caleman": "litexp_caleman.csv",
}
LITEXP_PROP_MAP = {"rho_l": "rho_l", "rho_v": "rho_v", "Pvap": "Pvap", "Hvap": "Hvap",
                    "gamma": "gamma", "Tc": "Tc", "rho_c": "rho_c",
                    # aliases used by some litexp files (gm, caleman) -- WP-C3 fix
                    "liquid_density": "rho_l", "vapor_density": "rho_v",
                    "vapor_pressure": "Pvap", "heat_of_vaporization": "Hvap",
                    "surface_tension": "gamma", "critical_temperature": "Tc",
                    "critical_density": "rho_c"}


def parse_float(v):
    if v in (None, ""):
        return None
    try:
        return float(str(v).split("±")[0].strip())
    except ValueError:
        return None


# ---- load master predictions, grouped by (ff, molecule) ----
preds = defaultdict(lambda: defaultdict(list))  # (ff,mol) -> prop -> [(T, value)]
preds_no_cc = defaultdict(lambda: defaultdict(list))  # (ff,mol) -> prop -> [(T, value)]
preds_crit = defaultdict(dict)  # (ff,mol) -> prop -> value  (Tc, rho_c)
ff_paper_tag = {}
with open(f"{AD}/master_ff_predictions.csv") as f:
    for r in csv.DictReader(f):
        key = (r["ff"], r["molecule"])
        ff_paper_tag[r["ff"]] = r["ff"].split(":")[0]
        prop = r["property"]
        if prop == "Pc":
            continue  # not one of the 7 Table 6/7 target properties
        v = float(r["value"])
        if prop in ("Tc", "rho_c"):
            preds_crit[key][prop] = v
        else:
            T = float(r["temperature_K"])
            preds[key][prop].append((T, v))
            if "Hvap_estimates.csv" not in r["source"]:
                preds_no_cc[key][prop].append((T, v))

# ---- load R1/R2 generic reference ----
ref = {}  # (mol,prop,T) -> dict
ref_crit = {}  # (mol,prop) -> dict
with open(f"{AD}/reference_values_all.csv") as f:
    for r in csv.DictReader(f):
        if r["temperature_K"] == "":
            ref_crit[(r["molecule"], r["property"])] = r
        else:
            ref[(r["molecule"], r["property"], round(float(r["temperature_K"]), 2))] = r

# ---- load per-paper litexp (R2 overrides) ----
litexp = defaultdict(lambda: defaultdict(list))  # (paper_tag,mol) -> prop -> [(T,value,unit)]
litexp_crit = defaultdict(dict)
_UNIT_CONV_DENSITY = {"kg/m3": 1.0, "kg/m^3": 1.0, "g/cm3": 1000.0, "g/cm^3": 1000.0, "g cm-3": 1000.0,
                      "g/cc": 1000.0, "g/mL": 1000.0, "g/L": 1.0}
for tag, fn in set(PAPER_TAG_TO_LITEXP.items()):
    path = f"{AD}/{fn}"
    try:
        with open(path) as f:
            rows = list(csv.DictReader(f))
    except FileNotFoundError:
        continue
    for r in rows:
        prop = LITEXP_PROP_MAP.get(r["property"])
        if prop is None:
            continue
        val = parse_float(r["exp_value"])
        if val is None:
            continue
        mol_raw = r["molecule"]
        mol = {"methanol": "MeOH", "ethylene glycol": "EG", "glycerol": "Gly"}.get(mol_raw, mol_raw)
        unit = r["value_unit"]
        if prop in ("rho_l", "rho_v", "rho_c"):
            if unit in ("mol/L", "mol/l"):   # WP-C3 fix: mol/L * g/mol = g/L = kg/m3
                val = val * MOL_WT.get(mol, float("nan"))
            elif unit in ("mol/cm3", "mol/cm^3"):
                val = val * MOL_WT.get(mol, float("nan")) * 1000.0
            else:
                factor = _UNIT_CONV_DENSITY.get(unit)
                if factor is None:
                    continue
                val = val * factor
        elif prop == "Pvap":
            if unit == "kPa":
                pass
            elif unit == "bar":
                val = val * 100.0
            elif unit == "MPa":
                val = val * 1000.0
            elif unit == "Torr" or unit == "torr" or unit == "mmHg":
                val = val * (101.325/760)
            else:
                continue
        elif prop == "Hvap":
            if unit == "kJ/kg":
                pass
            elif unit == "kJ/mol":
                val = val / MOL_WT.get(mol, float("nan")) * 1000.0
            elif unit == "kcal/mol":
                val = val * 4.184 / MOL_WT.get(mol, float("nan")) * 1000.0
            else:
                continue
        elif prop == "gamma":
            if unit not in ("mN/m", "dyn/cm"):
                continue
        T = parse_float(r["temperature"])
        if T is not None and r["temp_unit"] in ("degC", "C"):
            T = T + 273.15
        if prop in ("Tc", "rho_c"):
            if r["paper_tag"] == tag:
                litexp_crit[(tag, mol)][prop] = val
        elif T is not None:
            #If this paper is actually reporting a value for this paper_tag and store it for later lookup.
            if r["paper_tag"] == tag:
                litexp[(tag, mol)][prop].append((T, val))
            # litexp[(tag, mol)][prop].append((T, val))
#Print litexp values for each paper tag and molecule, for debugging.
# for (tag, mol), prop_dict in litexp.items():
#     print(f"litexp for {tag}, {mol}:")
#     print(f"  {prop_dict}")
#     print(f"  {litexp_crit.get((tag, mol), {})}")
#     print(" ")


def r2_lookup(paper_tag, mol, prop, T):
    """Prefer this paper's own litexp value (nearest within 2K); else None (caller falls back to generic)."""
    candidates = litexp.get((paper_tag, mol), {}).get(prop, [])
    if not candidates:
        return None
    near = [(abs(t - T), v) for t, v in candidates if abs(t - T) <= 2]
    if not near:
        return None
    near.sort()
    return near[0][1]


def cc_estimate(pvap_series, T_target, mw, verbose=False):
    """Clausius-Clapeyron bracket-search estimate, reimplemented independently (see findings/03)."""
    temps = sorted(set(t for t, _ in pvap_series))
    pvap_by_T = dict(pvap_series)
    higher = [t for t in temps if t > T_target]
    lower = [t for t in temps if t < T_target]
    if not higher or not lower:
        return None
    T_hi, T_lo = min(higher), max(lower)
    P_hi, P_lo = pvap_by_T[T_hi], pvap_by_T[T_lo]
    if P_hi <= 0 or P_lo <= 0:
        return None
    H_molar = -math.log(P_hi / P_lo) * R_KJ_PER_MOL_K / (1 / T_hi - 1 / T_lo)  # kJ/mol
    if verbose:
        print(T_target, T_lo, T_hi, P_lo, P_hi, mw)
        print(f"H={H_molar / mw * 1000.0:.3f} kJ/kg")
    return H_molar / mw * 1000.0  # kJ/kg


def build_hvap_series(key, mol, pred_vals, verbose=False):
    """Unified Hvap policy: direct where available, else CC from this FF's OWN Pvap curve."""
    direct = dict(pred_vals[key].get("Hvap", []))
    if verbose:
        print(f"  {key}: direct Hvap at {len(direct)} T points")
        #Print T points where direct Hvap is available.
        for T in sorted(direct.keys()):
            print(f"    direct Hvap at {T:.2f}K: {direct[T]:.3f} kJ/kg")
    pvap = pred_vals[key].get("Pvap", [])
    mw = MOL_WT.get(mol)
    out = []  # (T, value, method)
    all_T = sorted(set([t for t, _ in pred_vals[key].get("Hvap", [])] + [t for t, _ in pvap]))
    if verbose:
        print(all_T)
    for T in all_T:
        if T in direct:
            out.append((T, direct[T], "direct"))
        elif len(pvap) >= 2 and mw:
            if verbose:
                est = cc_estimate(pvap, T, mw, verbose=True)
            else:
                est = cc_estimate(pvap, T, mw)
            if est is not None:
                out.append((T, est, "CC"))
    return out


def calc_mapd(pairs):
    """pairs = [(sim, ref)]."""
    if not pairs:
        return None, 0
    errs = [abs(s - r) / abs(r) * 100.0 for s, r in pairs if r != 0]
    if not errs:
        return None, 0
    return sum(errs) / len(errs), len(errs)


def calc_mae(pairs):
    if not pairs:
        return None, 0
    errs = [abs(s - r) for s, r in pairs]
    return sum(errs) / len(errs), len(errs)


rows_out = []
MAPD_PROPS = {"rho_l", "gamma", "Hvap"}
MAE_PROPS = {"rho_v", "Pvap"}

for key in sorted(preds.keys() | preds_crit.keys()):
    ff, mol = key
    paper_tag = ff_paper_tag.get(ff, ff.split(":")[0])

    # ---- T-dependent properties ----
    for prop in ("rho_l", "rho_v", "Pvap", "gamma"):
        series = preds[key].get(prop, [])
        if not series:
            continue
        for ruler in ("R1", "R2"):
            pairs, caveats = [], set()
            for T, sim_v in series:
                if ruler == "R1":
                    rk = ref.get((mol, prop, round(T, 2)))
                    if rk is None or rk["R1_value"] == "":
                        continue
                    ref_v = float(rk["R1_value"])
                    if rk["R1_reliable"] == "False":
                        caveats.add("R1 near-critical/unreliable point included")
                else:
                    ov = r2_lookup(paper_tag, mol, prop, T)
                    if ov is not None:
                        ref_v = ov
                    else:
                        rk = ref.get((mol, prop, round(T, 2)))
                        if rk is None or rk["R2_value"] == "":
                            continue
                        ref_v = float(rk["R2_value"])
                        caveats.add("R2 = generic manuscript reference (no paper-specific value at this T)")
                pairs.append((sim_v, ref_v))
            if not pairs:
                continue
            Ts = [t for t, _ in series]
            if prop in MAPD_PROPS:
                val, n = calc_mapd(pairs)
                metric = "MAPD"
            else:
                val, n = calc_mae(pairs)
                metric = "MAE"
            if val is None:
                continue
            rows_out.append({
                "ff": ff, "molecule": mol, "property": prop, "ruler": ruler, "metric_type": metric,
                "value": round(val, 6), "n_points": n, "T_range": f"{min(Ts):.1f}-{max(Ts):.1f}",
                "hvap_method": "", "caveat_flag": "; ".join(sorted(caveats)),
            })

    # ---- Hvap (unified policy) ----
    hvap_series = build_hvap_series(key, mol, preds)
    hvap_series_no_cc = build_hvap_series(key, mol, preds_no_cc)

    if hvap_series:
        for ruler in ("R1", "R2"):
            if ruler == "R1" and (set(hvap_series_no_cc) != set(hvap_series)):
                print(key)
                series_use = hvap_series_no_cc
            else:
                series_use = hvap_series
            pairs, methods, caveats = [], set(), set()
            for T, sim_v, method in series_use:
                if ruler == "R1":
                    rk = ref.get((mol, "Hvap", round(T, 2)))
                    if rk is None or rk["R1_value"] == "":
                        continue
                    ref_v = float(rk["R1_value"])
                    if rk["R1_reliable"] == "False":
                        caveats.add("R1 near-critical/unreliable point included")
                    # if paper_tag == "meoh4p" and ff == "meoh4p:MD2" and mol == "MeOH":
                    #     print(f"  {T:.2f}K: sim={sim_v:.3f} kJ/kg, ref={ref_v:.3f} kJ/kg (R1)")
                else:
                    ov = r2_lookup(paper_tag, mol, "Hvap", T)
                    if ov is not None:
                        ref_v = ov
                    else:
                        rk = ref.get((mol, "Hvap", round(T, 2)))
                        if rk is None or rk["R2_value"] == "":
                            continue
                        ref_v = float(rk["R2_value"])
                        caveats.add("R2 = generic manuscript reference (no paper-specific value at this T)")
                    # if paper_tag == "meoh4p" and ff == "meoh4p:MD2" and mol == "MeOH":
                    #     print(f"  {T:.2f}K: sim={sim_v:.3f} kJ/kg, ref={ref_v:.3f} kJ/kg (R2)")
                pairs.append((sim_v, ref_v))
                methods.add(method)
            if not pairs:
                continue
            val, n = calc_mapd(pairs)
            if val is None:
                continue
            Ts = [t for t, _, _ in series_use]
            method_str = "+".join(sorted(methods))
            if "CC" in methods:
                caveats.add("some/all points are Clausius-Clapeyron estimates from this FF's own Pvap curve, not directly simulated")
            rows_out.append({
                "ff": ff, "molecule": mol, "property": "Hvap", "ruler": ruler, "metric_type": "MAPD",
                "value": round(val, 6), "n_points": n, "T_range": f"{min(Ts):.1f}-{max(Ts):.1f}",
                "hvap_method": method_str, "caveat_flag": "; ".join(sorted(caveats)),
            })

    # ---- Tc / rho_c (Table 6) ----
    for prop in ("Tc", "rho_c"):
        sim_v = preds_crit.get(key, {}).get(prop)
        if sim_v is None:
            continue
        for ruler in ("R1", "R2"):
            caveats = set()
            if ruler == "R1":
                rk = ref_crit.get((mol, prop))
                if rk is None or rk["R1_value"] == "":
                    continue
                ref_v = float(rk["R1_value"])
            else:
                ov = litexp_crit.get((paper_tag, mol), {}).get(prop)
                if ov is not None:
                    ref_v = ov
                else:
                    rk = ref_crit.get((mol, prop))
                    if rk is None or rk["R2_value"] == "":
                        continue
                    ref_v = float(rk["R2_value"])
                    caveats.add("R2 = generic manuscript reference (no paper-specific value)")
            val = abs(sim_v - ref_v) / abs(ref_v) * 100.0
            prov = ref_crit.get((mol, prop), {}).get("provenance_flag", "")
            if prov and prov != "EOS-AUTH" and "PRIMARY" not in prov:
                caveats.add(f"reference provenance: {prov}")
            rows_out.append({
                "ff": ff, "molecule": mol, "property": prop, "ruler": ruler, "metric_type": "MAPD",
                "value": round(val, 6), "n_points": 1, "T_range": "single-value",
                "hvap_method": "", "caveat_flag": "; ".join(sorted(caveats)),
            })

fields = ["ff", "molecule", "property", "ruler", "metric_type", "value", "n_points", "T_range",
          "hvap_method", "caveat_flag"]
out_path = f"{AD}/recomputed_metrics.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)
print(f"wrote {len(rows_out)} rows to {out_path}")

# ---- sanity check vs student_ff_recompute.csv ----
print("\n=== Sanity check vs student_ff_recompute.csv (per_T_mean) ===")
student_recompute = {}
with open(f"{AD}/student_ff_recompute.csv") as f:
    for r in csv.DictReader(f):
        if r["method"] == "per_T_mean":
            student_recompute[(r["molecule"], r["ff"], r["property"])] = r
mismatches = 0
checked = 0
for r in rows_out:
    if not r["ff"].startswith("student:") or r["ruler"] != "R2":
        continue
    ff_short = r["ff"].split(":", 1)[1]
    key = (r["molecule"], ff_short, r["property"])
    sr = student_recompute.get(key)
    if sr is None or sr["mapd"] in ("", None) and r["metric_type"] == "MAPD":
        continue
    checked += 1
    my_val = r["value"]
    their_val = float(sr["mapd"]) if r["metric_type"] == "MAPD" and sr["mapd"] not in ("", None) else \
                (float(sr["mae"]) if sr["mae"] not in ("", None) else None)
    if their_val is None:
        continue
    # Task 03's Pvap MAE is in bar (its native repo unit); this task's Pvap MAE is in kPa
    # (this task's stated convention) -- apply the known 100x unit factor before comparing.
    if r["property"] == "Pvap" and r["metric_type"] == "MAE":
        their_val = their_val * 100.0
    diff = abs(my_val - their_val)
    if diff > 0.5:  # allow some tolerance since R2 here uses litexp overrides which may differ from esolvs at edges
        mismatches += 1
        print(f"  DIFF {key}: mine={my_val} theirs={their_val} (diff={diff:.3f})")
print(f"checked {checked} student cells against Task 03's per_T_mean recompute; {mismatches} differ by >0.5")

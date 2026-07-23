"""
Worker Task 06 (WP-C2/A1/A3/D2), Phase 2: build the dual reference ruler at every
temperature that appears in Phase 1's master_ff_predictions.csv.

R1 (consistent authoritative, one callable per molecule/property):
    MeOH -> CoolProp (Helmholtz EOS, NIST/REFPROP reference-equation family).
    EG, Gly, DMF, DMSO, DEC -> `thermo` (DIPPR-based correlations), extending the same
    validated proxy approach used in scripts/validate_refprop.py (there restricted to EG;
    here applied to all four non-CoolProp molecules per this task's explicit instruction).
    Near-critical unreliability is flagged using the same T/Tc-ratio thresholds validated
    in findings/refprop_validation.md (rho_v > 0.68 Tc, rho_l > 0.85 Tc, others > 0.90 Tc).
    ME-10 fix: querying CoolProp directly for MeOH Hvap(500K) naturally returns the correct
    physical value (~391 kJ/kg), sidestepping the esolvs.py transcription error entirely --
    no special-case patch needed.

R2 (each paper's own cited experimental data): the GENERIC per-molecule R2 baseline stored
    in this table is `montana_reference_values.csv` (i.e. esolvs.py) -- the manuscript's own
    whole-study experimental ground truth, available at a fine grid for all 6 molecules.
    This is deliberately generic: Phase 3 additionally looks up each literature FF's *own*
    paper's cited values directly from litexp_<tag>.csv when grading that specific FF (a
    single (molecule,property,T) row here cannot hold N different papers' R2 values at once)
    -- see findings/06 for which FFs get a paper-specific R2 override vs. the generic one.

Output: audit_data/reference_values_all.csv
  columns: molecule, property, temperature_K, R1_value, R1_source, R1_reliable(bool),
           R2_value, R2_source, provenance_flag
"""
import csv
import warnings
warnings.filterwarnings("ignore")
from collections import defaultdict

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
AD = f"{AUDIT}/audit_data"

from thermo import Chemical
try:
    from CoolProp.CoolProp import PropsSI
    HAVE_CP = True
except Exception:
    HAVE_CP = False

CAS = {"MeOH": "67-56-1", "EG": "107-21-1", "Gly": "56-81-5", "DMF": "68-12-2",
       "DMSO": "67-68-5", "DEC": "105-58-8"}
TC = {"MeOH": 512.5, "EG": 719.6, "Gly": 850.0, "DMF": 649.6, "DMSO": 718.0, "DEC": 576.0}  # esolvs.py values, for T/Tc ratio only
CP_NAME = {"MeOH": "Methanol"}  # only MeOH in CoolProp among our 6


def coolprop(mol, prop, T):
    if not HAVE_CP or mol not in CP_NAME:
        return None
    f = CP_NAME[mol]
    try:
        if prop == "rho_l":
            return PropsSI("D", "T", T, "Q", 0, f)
        if prop == "rho_v":
            return PropsSI("D", "T", T, "Q", 1, f)
        if prop == "Pvap":
            return PropsSI("P", "T", T, "Q", 0, f) / 1000.0  # Pa -> kPa
        if prop == "Hvap":
            return (PropsSI("H", "T", T, "Q", 1, f) - PropsSI("H", "T", T, "Q", 0, f)) / 1000.0  # J/kg -> kJ/kg
        if prop == "gamma":
            return PropsSI("I", "T", T, "Q", 0, f) * 1000.0  # N/m -> mN/m
    except Exception:
        return None
    return None


def thermo_val(mol, prop, T):
    try:
        c = Chemical(CAS[mol], T=float(T))
        if prop == "rho_l":
            return c.rhol
        if prop == "Pvap":
            return c.Psat / 1000.0 if c.Psat else None  # Pa -> kPa
        if prop == "gamma":
            return c.sigma * 1000.0 if c.sigma else None
        if prop == "Hvap":
            return c.Hvap / 1000.0 if c.Hvap else None
        if prop == "rho_v":
            return Chemical(CAS[mol], T=float(T), P=c.Psat).rhog if c.Psat else None
    except Exception:
        return None
    return None


def r1_value_and_reliability(mol, prop, T):
    """Returns (value, source, reliable(bool))."""
    tratio = T / TC[mol]
    if mol == "MeOH":
        v = coolprop(mol, prop, T)
        # CoolProp is authoritative across the whole range for MeOH (validated to 0.00% in
        # findings/refprop_validation.md); still flag extreme proximity to Tc for transparency.
        reliable = tratio < 0.98
        return v, "CoolProp", reliable
    v = thermo_val(mol, prop, T)
    unreliable = (
        (prop == "rho_v" and tratio > 0.68) or
        (prop == "rho_l" and tratio > 0.85) or
        (prop not in ("rho_v", "rho_l") and tratio > 0.90)
    )
    return v, "thermo", not unreliable


def r1_critical(mol, prop):
    """Tc/rho_c from thermo/CoolProp for reference (informational; the manuscript's own
    critical-point provenance per molecule is documented in provenance_map.csv)."""
    try:
        c = Chemical(CAS[mol])
        if prop == "Tc":
            return c.Tc, ("CoolProp/thermo Tc" if mol != "MeOH" else "CoolProp Tc")
        if prop == "rho_c":
            return c.rhoc, ("CoolProp/thermo rho_c" if mol != "MeOH" else "CoolProp rho_c")
    except Exception:
        return None, None
    return None, None


# ---- load the generic R2 baseline: montana_reference_values.csv (esolvs.py) ----
r2_generic = {}  # (molecule, property, T) -> (value, unit)
MRV_PROP_MAP = {"liq_density": "rho_l", "vap_density": "rho_v", "Pvap": "Pvap",
                "Hvap": "Hvap", "surf_tens": "gamma", "Tc": "Tc", "rhoc": "rho_c"}
with open(f"{AD}/montana_reference_values.csv") as f:
    for r in csv.DictReader(f):
        prop = MRV_PROP_MAP.get(r["property"])
        if prop is None:
            continue
        T = r["temperature_K"]
        v = float(r["value"])
        # montana_reference_values.csv stores Pvap in bar natively (see findings/02) -- convert to
        # kPa here so R2 is expressed in the same units as R1 throughout this table.
        if prop == "Pvap" and r["unit"] == "bar":
            v = v * 100.0
        key = (r["molecule"], prop, round(float(T), 2) if T not in ("", None) else None)
        r2_generic[key] = (v, "kPa" if prop == "Pvap" else r["unit"])

# ---- provenance flags from provenance_map.csv ----
prov_flag = {}
with open(f"{AD}/provenance_map.csv") as f:
    for r in csv.DictReader(f):
        prov_flag[(r["molecule"], r["property"])] = r["source_type"]

# ---- gather every (molecule, property, T) that Phase 1 needs ----
needed = set()
needed_critical = set()  # (molecule, prop) for Tc/rho_c, no T
with open(f"{AD}/master_ff_predictions.csv") as f:
    for r in csv.DictReader(f):
        prop = r["property"]
        if prop in ("rho_l", "rho_v", "Pvap", "Hvap", "gamma") and r["temperature_K"] != "":
            needed.add((r["molecule"], prop, round(float(r["temperature_K"]), 2)))
        elif prop in ("Tc", "rho_c"):
            needed_critical.add((r["molecule"], prop))

rows_out = []
for (mol, prop, T) in sorted(needed):
    r1v, r1src, r1rel = r1_value_and_reliability(mol, prop, T)
    r2key = (mol, prop, T)
    # try exact T match first, else nearest within 2K (0.01K) (esolvs grids are usually exact matches)
    r2v, r2unit = r2_generic.get(r2key, (None, None))
    if r2v is None:
        candidates = [(k, v) for k, v in r2_generic.items() if k[0] == mol and k[1] == prop and k[2] is not None]
        near = [(abs(k[2] - T), v) for k, v in candidates if abs(k[2] - T) <= 2]
        if near:
            near.sort()
            r2v, r2unit = near[0][1]
    rows_out.append({
        "molecule": mol, "property": prop, "temperature_K": T,
        "R1_value": round(r1v, 6) if r1v is not None else "",
        "R1_source": r1src if r1v is not None else "",
        "R1_reliable": r1rel if r1v is not None else "",
        "R2_value": round(r2v, 6) if r2v is not None else "",
        "R2_source": "montana_reference_values.csv (esolvs.py)" if r2v is not None else "",
        "provenance_flag": prov_flag.get((mol, prop), ""),
    })

for (mol, prop) in sorted(needed_critical):
    r1v, r1src = r1_critical(mol, prop)
    r2key_candidates = [(k, v) for k, v in r2_generic.items() if k[0] == mol and k[1] == prop]
    r2v, r2unit = r2key_candidates[0][1] if r2key_candidates else (None, None)
    rows_out.append({
        "molecule": mol, "property": prop, "temperature_K": "",
        "R1_value": round(r1v, 6) if r1v is not None else "",
        "R1_source": r1src if r1v is not None else "",
        "R1_reliable": "True" if r1v is not None else "",
        "R2_value": round(r2v, 6) if r2v is not None else "",
        "R2_source": "montana_reference_values.csv (esolvs.py)" if r2v is not None else "",
        "provenance_flag": prov_flag.get((mol, prop), ""),
    })

fields = ["molecule", "property", "temperature_K", "R1_value", "R1_source", "R1_reliable",
          "R2_value", "R2_source", "provenance_flag"]
out_path = f"{AD}/reference_values_all.csv"
with open(out_path, "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print(f"wrote {len(rows_out)} rows to {out_path}")

n_both = sum(1 for r in rows_out if r["R1_value"] != "" and r["R2_value"] != "")
n_r1_only = sum(1 for r in rows_out if r["R1_value"] != "" and r["R2_value"] == "")
n_r2_only = sum(1 for r in rows_out if r["R1_value"] == "" and r["R2_value"] != "")
n_neither = sum(1 for r in rows_out if r["R1_value"] == "" and r["R2_value"] == "")
n_r1_unreliable = sum(1 for r in rows_out if r["R1_reliable"] is False)
print(f"both R1+R2: {n_both}, R1 only: {n_r1_only}, R2 only: {n_r2_only}, neither: {n_neither}")
print(f"R1 flagged unreliable (near-critical): {n_r1_unreliable}")

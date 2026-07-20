"""
Placeholder validation of the repository's REFPROP-tagged reference values
(esolvs.py, for MeOH and EG) against independent sources, while we wait for
Montana's original REFPROP output files.

Independent references:
  - MeOH: CoolProp (high-accuracy Helmholtz EOS, same reference-equation family
    as NIST/REFPROP) -- an authoritative independent implementation.
  - EG:   thermo (DIPPR-based correlations); CoolProp has no pure ethylene-glycol
    EOS. thermo's simple correlations are unreliable within ~15 % of Tc, so
    near-critical points (flagged) cannot be validated this way and still require
    REFPROP.

Input : audit_data/montana_reference_values.csv (the extracted esolvs values)
Output: audit_data/refprop_validation.csv + printed summary
"""
import warnings; warnings.filterwarnings("ignore")
import numpy as np, pandas as pd, csv, os

from thermo import Chemical
try:
    from CoolProp.CoolProp import PropsSI
    HAVE_CP = True
except Exception:
    HAVE_CP = False

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MRV  = os.path.join(HERE, "audit_data", "montana_reference_values.csv")
OUT  = os.path.join(HERE, "audit_data", "refprop_validation.csv")

TC = {"MeOH": 512.5, "EG": 719.6}
CAS = {"MeOH": "67-56-1", "EG": "107-21-1"}
CP_NAME = {"MeOH": "Methanol"}   # no pure EG in CoolProp

mrv = pd.read_csv(MRV)
def esolvs(mol, prop):
    s = mrv[(mrv.molecule == mol) & (mrv.property == prop)]
    return dict(zip(s.temperature_K, s.value))

def coolprop(mol, prop, T):
    if not HAVE_CP or mol not in CP_NAME: return np.nan
    f = CP_NAME[mol]
    try:
        if prop == "liq_density": return PropsSI("D", "T", T, "Q", 0, f)
        if prop == "vap_density": return PropsSI("D", "T", T, "Q", 1, f)
        if prop == "Pvap":        return PropsSI("P", "T", T, "Q", 0, f) / 1e5      # Pa -> bar
        if prop == "Hvap":        return (PropsSI("H", "T", T, "Q", 1, f) - PropsSI("H", "T", T, "Q", 0, f)) / 1000.0
        if prop == "surf_tens":   return PropsSI("I", "T", T, "Q", 0, f) * 1000.0   # N/m -> mN/m
    except Exception:
        return np.nan
    return np.nan

def thermo_val(mol, prop, T):
    try:
        c = Chemical(CAS[mol], T=float(T))
        if prop == "liq_density": return c.rhol
        if prop == "Pvap":        return c.Psat / 1e5
        if prop == "surf_tens":   return c.sigma * 1000.0 if c.sigma else np.nan
        if prop == "Hvap":        return c.Hvap / 1000.0 if c.Hvap else np.nan
        if prop == "vap_density":
            return Chemical(CAS[mol], T=float(T), P=c.Psat).rhog
    except Exception:
        return np.nan
    return np.nan

rows = []
for mol in ["MeOH", "EG"]:
    for prop in ["liq_density", "vap_density", "Pvap", "Hvap", "surf_tens"]:
        for T, e in sorted(esolvs(mol, prop).items()):
            cp = coolprop(mol, prop, T); th = thermo_val(mol, prop, T)
            # reference of record: CoolProp for MeOH, thermo for EG
            ref = cp if (mol == "MeOH" and cp == cp) else th
            ref_src = "CoolProp" if (mol == "MeOH" and cp == cp) else "thermo"
            dev = (e - ref) / ref * 100 if (ref == ref and ref) else np.nan
            tratio = T / TC[mol]
            unreliable_ref = (ref_src == "thermo") and (
                (prop == "vap_density" and tratio > 0.68) or (prop == "liq_density" and tratio > 0.85)
                or (tratio > 0.90 and prop not in ("Pvap", "surf_tens", "Hvap")))
            if unreliable_ref:
                flag = "REF-UNRELIABLE (near-critical; needs REFPROP)"
            elif abs(dev) < 1.0:
                flag = "VALIDATED"
            elif abs(dev) < 5.0:
                flag = "MINOR-DEV"
            else:
                flag = "ANOMALY (check)"
            rows.append([mol, prop, T, round(tratio, 3), e, cp, th, ref_src,
                         round(dev, 3) if dev == dev else "", flag])

cols = ["molecule", "property", "temperature_K", "T_over_Tc", "esolvs_value",
        "coolprop_value", "thermo_value", "ref_of_record", "pct_dev", "flag"]
with open(OUT, "w", newline="") as f:
    w = csv.writer(f); w.writerow(cols); w.writerows(rows)

df = pd.DataFrame(rows, columns=cols)
print("wrote", OUT, "|", len(df), "rows")
print("\nflag summary:")
print(df["flag"].value_counts().to_string())
print("\nnon-VALIDATED rows:")
print(df[~df.flag.str.startswith("VALIDATED")][["molecule","property","temperature_K","esolvs_value","coolprop_value","thermo_value","pct_dev","flag"]].to_string(index=False))

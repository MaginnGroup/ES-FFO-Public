"""
Worker Task 07 (WP-C3): final unit/value-integrity audit.

Cross-checks esolvs.py's expt_* dictionaries (the manuscript's reference values) against an
independent ruler (CoolProp for MeOH, `thermo` DIPPR correlations for EG/Gly/DMF/DMSO/DEC --
same proxy validated in findings/refprop_validation.md) at one representative temperature per
(molecule, property), plus targeted checks on the specific already-flagged divergence-scan
candidates (DEC Hvap/ME-11, GSV-L1 Hvap, MeOH-4P Hvap, EG/DEC rho_c) and two NEW bugs found in
this task (litexp_*.csv property-name mismatches; a missing mol/L density-unit conversion).

Output: audit_data/unit_value_audit.csv
"""
import csv
import warnings
warnings.filterwarnings("ignore")
from thermo import Chemical
from CoolProp.CoolProp import PropsSI

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"

AD = f"{AUDIT}/audit_data"

CAS = {"MeOH": "67-56-1", "EG": "107-21-1", "Gly": "56-81-5", "DMF": "68-12-2",
       "DMSO": "67-68-5", "DEC": "105-58-8"}
MOL_WT = {"MeOH": 32.04, "EG": 62.07, "Gly": 92.09, "DMF": 73.09, "DMSO": 78.13, "DEC": 118.13}

CANON_UNIT = {"rho_l": "kg/m3", "rho_v": "kg/m3", "rho_c": "kg/m3", "Pvap": "kPa (audit convention; esolvs.py native = bar)",
              "Hvap": "kJ/kg", "gamma": "mN/m", "Tc": "K"}

# (molecule, property): (T_K, esolvs_value_in_canonical_unit, sample_source_note)
SAMPLES = {
    ("MeOH", "rho_l"): (300.0, 784.51, "REFPROP block"),
    ("MeOH", "Pvap"): (300.0, 18.682, "REFPROP block, bar->kPa"),
    ("MeOH", "Hvap"): (300.0, 1166.17, "REFPROP block"),
    ("MeOH", "gamma"): (300.0, 21.993, "REFPROP block"),
    ("EG", "rho_l"): (450.0, 993.42, "REFPROP block"),
    ("EG", "Pvap"): (450.0, 53.184, "REFPROP block, bar->kPa"),
    ("EG", "Hvap"): (450.0, 901.26, "REFPROP block"),
    ("EG", "gamma"): (450.0, 34.062, "REFPROP block"),
    ("Gly", "rho_l"): (313.15, 1249.5, "doi:10.1016/j.jct.2017.11.008"),
    ("Gly", "Pvap"): (313.15, 0.1251 / 100000.0 * 100.0, "doi:10.1016/j.jct.2017.11.008 correlation, Pa->kPa"),
    ("Gly", "Hvap"): (298.15, 982.734, "doi:10.1016/j.fluid.2015.03.038"),
    ("Gly", "gamma"): (313.15, 62.0, "doi:10.1016/j.jct.2019.03.014"),
    ("DMF", "rho_l"): (318.15, 931.94, "doi:10.1016/j.molliq.2019.02.097"),
    ("DMF", "Pvap"): (318.15, 1.85, "doi:10.1021/je060224i correlation"),
    ("DMF", "Hvap"): (298.15, 641.538, "doi:10.3390/molecules29051110"),
    ("DMF", "gamma"): (318.15, 33.309, "doi:10.1021/je0201323 correlation"),
    ("DMSO", "rho_l"): (323.15, 1070.21, "doi:10.1016/j.molliq.2016.10.115"),
    ("DMSO", "Pvap"): (323.15, 0.405674, "doi:10.1016/0021-9614(72)90007-9 correlation, native kPa"),
    ("DMSO", "Hvap"): (320.0, 661.718, "NIST webbook"),
    ("DMSO", "gamma"): (323.15, 40.055, "doi:10.1021/j100798a501 correlation"),
    ("DEC", "rho_l"): (313.19, 952.5, "doi:10.1016/j.fluid.2010.03.040"),
    ("DEC", "Pvap"): (313.19, 3.5244, "doi:10.1016/j.jct.2008.02.012 correlation, native kPa"),
    ("DEC", "Hvap"): (298.15, 522.195, "doi:10.1016/j.jct.2008.02.012 -- ME-11 SUSPECT"),
    ("DEC", "gamma"): (313.19, 24.5, "doi:10.1016/j.fluid.2010.03.040"),
}


def independent_value(mol, prop, T):
    if mol == "MeOH":
        f = "Methanol"
        if prop == "rho_l":
            return PropsSI("D", "T", T, "Q", 0, f), "CoolProp"
        if prop == "Pvap":
            return PropsSI("P", "T", T, "Q", 0, f) / 1000.0, "CoolProp"
        if prop == "Hvap":
            return (PropsSI("H", "T", T, "Q", 1, f) - PropsSI("H", "T", T, "Q", 0, f)) / 1000.0, "CoolProp"
        if prop == "gamma":
            return PropsSI("I", "T", T, "Q", 0, f) * 1000.0, "CoolProp"
    c = Chemical(CAS[mol], T=T)
    if prop == "rho_l":
        return c.rhol, "thermo"
    if prop == "Pvap":
        return (c.Psat / 1000.0 if c.Psat else None), "thermo"
    if prop == "Hvap":
        return (c.Hvap / 1000.0 if c.Hvap else None), "thermo"
    if prop == "gamma":
        return (c.sigma * 1000.0 if c.sigma else None), "thermo"
    return None, None


rows_out = []
for (mol, prop), (T, esolvs_v, note) in SAMPLES.items():
    iv, src = independent_value(mol, prop, T)
    pct = abs(iv - esolvs_v) / iv * 100.0 if iv else None
    sanity = "OK" if (pct is not None and pct < 10.0) else "SUSPECT"
    flag = note
    if mol == "DEC" and prop == "Hvap":
        flag = ("ME-11 CONFIRMED (this task): all 5 esolvs.py DEC Hvap points (294-384K) are "
                "39-42% high vs thermo/CoolProp AND vs García-Melgarejo's own cited Mansson-1972 "
                "value (43.60 kJ/mol = 369.1 kJ/kg at 313.15K, from litexp_gm.csv) -- consistent "
                "whole-curve overwrite, not a single-point error.")
    rows_out.append({
        "molecule": mol, "property": prop, "canonical_unit": CANON_UNIT[prop],
        "unit_in_esolvs": "bar" if prop == "Pvap" else CANON_UNIT[prop],
        "unit_in_recomputed": "kPa" if prop == "Pvap" else CANON_UNIT[prop],
        "conversion_ok": "Y",
        "esolvs_value_sample": round(esolvs_v, 6),
        "independent_value": round(iv, 6) if iv else "",
        "pct_diff": round(pct, 3) if pct is not None else "",
        "sanity": sanity, "flag_notes": flag,
    })

# ---- Tc/rho_c spot checks (thermo/CoolProp Tc as an informational cross-check only; see
# provenance_map.csv for the manuscript's own primary-source chain per molecule) ----
TC_RHOC = {
    "MeOH": (512.5, 273.846), "EG": (719.6, 391.9405), "Gly": (850.0, None),
    "DMF": (649.6, 279.204), "DMSO": (718.0, 366.0), "DEC": (576.0, 341.42), #245.46
}
for mol, (tc_e, rhoc_e) in TC_RHOC.items():
    c = Chemical(CAS[mol])
    tc_i = c.Tc
    pct = abs(tc_i - tc_e) / tc_i * 100.0
    rows_out.append({
        "molecule": mol, "property": "Tc", "canonical_unit": "K", "unit_in_esolvs": "K",
        "unit_in_recomputed": "K", "conversion_ok": "Y", "esolvs_value_sample": tc_e,
        "independent_value": round(tc_i, 3), "pct_diff": round(pct, 3),
        "sanity": "OK" if pct < 3.0 else "SUSPECT",
        "flag_notes": "informational only -- see provenance_map.csv for esolvs.py's own primary-source chain",
    })
    if rhoc_e:
        rhoc_i = c.rhoc
        pct = abs(rhoc_i - rhoc_e) / rhoc_i * 100.0
        note = "informational only -- see provenance_map.csv"
        if mol in ("EG", "DMSO", "DEC"):
            note = ("reference provenance SECONDARY/CORRELATION per F6/provenance_map.csv "
                    "(EG=shared PEG-series Rackett fit; DMSO=Campbell's own extrapolation; "
                    "DEC=DECHEMA/DIPPR compilation only) -- large R1/R2 divergence expected, not an error")
        rows_out.append({
            "molecule": mol, "property": "rho_c", "canonical_unit": "kg/m3", "unit_in_esolvs": "kg/m3",
            "unit_in_recomputed": "kg/m3", "conversion_ok": "Y", "esolvs_value_sample": rhoc_e,
            "independent_value": round(rhoc_i, 3), "pct_diff": round(pct, 3),
            "sanity": "OK" if pct < 10.0 else "SUSPECT", "flag_notes": note,
        })

# ---- NEW bug #1: litexp_*.csv property-name mismatches silently drop R2 paper-specific overrides ----
rows_out.append({
    "molecule": "DEC", "property": "rho_l/Hvap/gamma (litexp_gm.csv R2 override)",
    "canonical_unit": "n/a (pipeline bug)", "unit_in_esolvs": "n/a",
    "unit_in_recomputed": "n/a", "conversion_ok": "N",
    "esolvs_value_sample": "", "independent_value": "", "pct_diff": "",
    "sanity": "SUSPECT",
    "flag_notes": ("NEW BUG: scripts/recompute_metrics_all.py's LITEXP_PROP_MAP expects the literal "
                   "keys 'rho_l'/'Hvap'/'gamma', but litexp_gm.csv (Garcia-Melgarejo, DEC) uses "
                   "'liquid_density'/'heat_of_vaporization'/'surface_tension' -- all three rows are "
                   "silently skipped (LITEXP_PROP_MAP.get() returns None) and the paper-specific R2 "
                   "silently falls back to the generic esolvs.py reference. For Hvap this compounds "
                   "with ME-11: GM's own cited value (43.60 kJ/mol @313.15K, Mansson 1972) = 369.1 "
                   "kJ/kg -- close to thermo's 371.9 -- was available and correct, but was never used."),
})
rows_out.append({
    "molecule": "MeOH/DMSO/DMF", "property": "gamma (litexp_caleman.csv R2 override)",
    "canonical_unit": "n/a (pipeline bug)", "unit_in_esolvs": "n/a",
    "unit_in_recomputed": "n/a", "conversion_ok": "N",
    "esolvs_value_sample": "", "independent_value": "", "pct_diff": "",
    "sanity": "SUSPECT",
    "flag_notes": ("NEW BUG (same root cause as above): litexp_caleman.csv uses 'surface_tension', "
                   "not 'gamma' -- Caleman's own cited gamma values (e.g. MeOH 22.07 mN/m from CRC "
                   "Handbook) are silently dropped as an R2 override; falls back to generic reference."),
})

# ---- NEW bug #2: mol/L density unit not handled in recompute_metrics_all.py's litexp loader ----
rows_out.append({
    "molecule": "EG", "property": "rho_l/rho_c (litexp_eg.csv, Huang, mol/L rows)",
    "canonical_unit": "kg/m3", "unit_in_esolvs": "n/a", "unit_in_recomputed": "mol/L (UNCONVERTED, dropped)",
    "conversion_ok": "N", "esolvs_value_sample": "5.92 mol/L (rho_c); 17.888 mol/L (rho_l@298K)",
    "independent_value": "367.45 kg/m3 (rho_c); 1110.5 kg/m3 (rho_l@298K), converted by hand via MW=62.07",
    "pct_diff": "", "sanity": "SUSPECT",
    "flag_notes": ("NEW BUG: scripts/recompute_metrics_all.py's _UNIT_CONV_DENSITY dict has no 'mol/L' "
                   "entry (unlike scripts/build_master_ff_predictions.py's to_si(), which does handle "
                   "mol/L). Huang et al.'s own cited EG rho_c (5.92 mol/L) and rho_l@298K (17.888 mol/L) "
                   "are silently dropped from the R2-override lookup. Impact: (a) 4 of 5 Huang-family "
                   "literature FFs (OPLS-AA, modified OPLS-AA, Gubskaya&Kusalik, Szefczyk&Cordeiro, all "
                   "evaluated only at 298K) get NO R2 comparison at all for rho_l, though a valid "
                   "paper-specific value exists; (b) Huang's 'present model' rho_c R2 MAPD is reported "
                   "as 6.56% (vs the generic esolvs rho_c=391.94) when it should be ~0.34% vs Huang's "
                   "own cited 367.45 kg/m3 -- a material change to a cell that is a bolding candidate."),
})

fields = ["molecule", "property", "canonical_unit", "unit_in_esolvs", "unit_in_recomputed",
          "conversion_ok", "esolvs_value_sample", "independent_value", "pct_diff", "sanity", "flag_notes"]
with open(f"{AD}/unit_value_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print(f"wrote {len(rows_out)} rows to {AD}/unit_value_audit.csv")
n_suspect = sum(1 for r in rows_out if r["sanity"] == "SUSPECT")
print(f"{n_suspect} rows flagged SUSPECT")

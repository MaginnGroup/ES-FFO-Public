"""
Worker Task 11, Part C -- generate the reference-data provenance table for the report, reconciled
against the LIVE esolvs.py (so it reflects the ME-10/ME-11 corrections and current state, not a
stale snapshot) and the Table-3 citation-key fix (ME-01: Zhang2010 -> Zhang2010ExperimentalKPa).

Output: audit_data/provenance_table.tex (a standalone-compilable/`\\input`-able longtable).
"""
import csv
import re
import sys

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
# REPO = "/Users/adowling/DowlingLab/ES-FFO-Public"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
REPO = "/groups/ed/group_members/Montana_Carlozo/ES-FFO"
# MS = "/Users/adowling/Downloads/ES_FFO-2"
AD = f"{AUDIT}/audit_data"

MOLECULES = ["MeOH", "EG", "Gly", "DMF", "DMSO", "DEC"]
PROP_LABEL = {"rho_l": r"$\rho_l$", "Pvap": r"$P_{vap}$", "Hvap": r"$\Delta H_{vap}$",
              "rho_v": r"$\rho_v$", "gamma": r"$\gamma$", "Tc": r"$T_c$", "rho_c": r"$\rho_c$"}
PROP_UNIT = {"rho_l": "kg/m$^3$", "Pvap": "kPa", "Hvap": "kJ/kg", "rho_v": "kg/m$^3$",
             "gamma": "mN/m", "Tc": "K", "rho_c": "kg/m$^3$"}

# Table-3's own citation key per (molecule, property) -- corrected per ME-01 (Zhang2010 ->
# Zhang2010ExperimentalKPa for DEC Tc/rho_c) since this table should show the CORRECTED state.
TABLE3_KEY = {
    ("MeOH", "rho_l"): "REFPROP", ("EG", "rho_l"): "REFPROP", ("Gly", "rho_l"): "Ghaedi2018DensityCapture",
    ("DMF", "rho_l"): "Alam2019TheProperties", ("DMSO", "rho_l"): "Harifi-Mood2017", ("DEC", "rho_l"): "Zhao2010",
    ("MeOH", "Pvap"): "REFPROP", ("EG", "Pvap"): "REFPROP", ("Gly", "Pvap"): "Verevkin2015",
    ("DMF", "Pvap"): "Cui2006", ("DMSO", "Pvap"): "Jakli1972", ("DEC", "Pvap"): "Kozlova2008",
    ("MeOH", "Hvap"): "REFPROP", ("EG", "Hvap"): "REFPROP", ("Gly", "Hvap"): "Verevkin2015",
    ("DMF", "Hvap"): "Rika2024", ("DMSO", "Hvap"): "Lemmon2023", ("DEC", "Hvap"): "Kozlova2008",
    ("MeOH", "rho_v"): "REFPROP", ("EG", "rho_v"): "REFPROP", ("Gly", "rho_v"): "Peng1976AState",
    ("DMF", "rho_v"): "Peng1976AState", ("DMSO", "rho_v"): "Peng1976AState", ("DEC", "rho_v"): "Zhao2010",
    ("MeOH", "Tc"): "REFPROP", ("EG", "Tc"): "REFPROP", ("Gly", "Tc"): "Nikitin1993MeasurementMethod",
    ("DMF", "Tc"): "Teja1990TheResults", ("DMSO", "Tc"): "Nikitin2018Critical26-Dimethylpiperazine",
    ("DEC", "Tc"): "Zhang2010ExperimentalKPa",  # ME-01 fix (Table 3 currently prints "Zhang2010")
    ("MeOH", "rho_c"): "Benson1948CriticalLiquids", ("EG", "rho_c"): "Tawfik1989TheGlycols",
    ("Gly", "rho_c"): "Peng1976AState", ("DMF", "rho_c"): "Teja1990TheResults",
    ("DMSO", "rho_c"): "Campbell1979TheDensity",
    ("DEC", "rho_c"): "Zhang2010ExperimentalKPa",  # ME-01 fix
    ("MeOH", "gamma"): "REFPROP", ("EG", "gamma"): "REFPROP", ("Gly", "gamma"): "Erfani2019",
    ("DMF", "gamma"): "Kahl2003", ("DMSO", "gamma"): "LawrenceClever1963", ("DEC", "gamma"): "Zhao2010",
}
NON_LITERATURE = {"REFPROP", "Peng1976AState"}


def load_bib():
    text = open(f"/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit/refs.bib", encoding="utf-8", errors="replace").read()
    entries = {}
    for chunk in re.split(r'\n(?=@\w+\{)', text):
        m = re.match(r'@(\w+)\{([^,]+),', chunk)
        if not m:
            continue
        key, body = m.group(2).strip(), chunk[m.end():]
        a = re.search(r'author\s*=\s*\{+(.*?)\}+,?\s*\n', body, re.S)
        y = re.search(r'year\s*=\s*\{+(.*?)\}+,?\s*\n', body, re.S)
        entries[key] = {"author": a.group(1).strip() if a else "", "year": y.group(1).strip() if y else ""}
    return entries


def cite_label(key):
    if key == "REFPROP":
        return "REFPROP"
    if key == "Peng1976AState":
        return "Peng & Robinson 1976"  # tex_escape() at the call site escapes the "&"
    e = bib.get(key)
    if not e:
        return key
    first = e["author"].split(" and ")[0].strip()
    # refs.bib mixes "Last, First" and "First Middle Last" author formats -- handle both.
    surname = first.split(",")[0].strip() if "," in first else first.split()[-1].strip()
    return f"{surname} {e['year']}" if surname else key


def esolvs_summary(mol, prop):
    obj = getattr(esolvs, mol)
    if prop == "Tc":
        return f"{obj.expt_Tc:.2f}", "single value"
    if prop == "rho_c":
        return (f"{obj.expt_rhoc:.2f}" if obj.expt_rhoc is not None else "--"), "single value"
    d = {"rho_l": obj.expt_liq_density, "Pvap": obj.expt_Pvap, "Hvap": obj.expt_Hvap,
         "rho_v": obj.expt_vap_density, "gamma": obj.expt_surf_tens}[prop]
    if not d:
        return "--", ""
    ks = sorted(d.keys())
    n = len(ks)
    trange = f"{ks[0]:g}--{ks[-1]:g}\\,K" if n > 1 else f"{ks[0]:g}\\,K"
    return f"n={n}", trange


bib = load_bib()
sys.path.insert(0, f"{REPO}/utils/molec_class_files")
import esolvs

prov = {}
for r in csv.DictReader(open(f"{AD}/provenance_map.csv")):
    prov[(r["molecule"], r["property"])] = r


def tex_escape(s):
    for a, b in [("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        s = s.replace(a, b)
    return s


PROPS = ["rho_l", "Pvap", "Hvap", "rho_v", "Tc", "rho_c", "gamma"]

lines = []
lines.append(r"% Auto-generated by scripts/generate_provenance_table.py -- DO NOT HAND-EDIT.")
lines.append(r"% Regenerate with: python3 scripts/generate_provenance_table.py")
lines.append(r"% Reconciled against the LIVE esolvs.py (reflects ME-10/ME-11 corrections) and the")
lines.append(r"% corrected Table 3 citation key (ME-01: Zhang2010 -> Zhang2010ExperimentalKPa).")
lines.append(r"\scriptsize")
lines.append(r"\setlength{\tabcolsep}{2pt}")
lines.append(r"\begin{longtable}{@{}l l >{\raggedright\arraybackslash}p{3.1cm} "
             r">{\raggedright\arraybackslash}p{1.9cm} >{\raggedright\arraybackslash}p{2.5cm} "
             r">{\raggedright\arraybackslash}p{3.4cm}@{}}")
lines.append(r"\caption{Provenance of every reference value in \code{esolvs.py} used to grade "
             r"Tables 6 \& 7. ``Value/coverage'' shows the single value for $T_c$/$\rho_c$ or the "
             r"point count and temperature range otherwise. ``Better primary source?'' notes where "
             r"this audit identified an alternative primary measurement (see "
             r"\code{provenance\_map.csv} for detail); a blank means none was sought/needed."
             r"\label{tab:provenance}} \\")
_HEADROW = (r"\textbf{Molecule} & \textbf{Property} & \textbf{Value/coverage} & \textbf{Source} & "
            r"\textbf{Classification} & \textbf{Better primary source?} \\")
# First-page head carries the caption+label; continuation pages repeat only the column header,
# so \label{tab:provenance} is defined exactly once (avoids "multiply defined" in longtable).
lines.append(r"\toprule")
lines.append(_HEADROW)
lines.append(r"\midrule\endfirsthead")
lines.append(r"\multicolumn{6}{@{}l}{\footnotesize\emph{Table~\ref{tab:provenance} continued from previous page}}\\")
lines.append(r"\toprule")
lines.append(_HEADROW)
lines.append(r"\midrule\endhead")

for mol in MOLECULES:
    lines.append(f"\\multirow{{{len(PROPS)}}}{{*}}{{{mol}}}")
    for prop in PROPS:
        key = TABLE3_KEY[(mol, prop)]
        label = cite_label(key)
        val, cov = esolvs_summary(mol, prop)
        coverage = f"{val}" + (f", {cov}" if cov and cov != "single value" else "")
        p = prov.get((mol, prop), {})
        source_type = p.get("source_type", "")
        candidate = p.get("primary_source_candidate", "")
        in_hand = p.get("in_hand", "")
        better = ""
        if candidate:
            better = tex_escape(candidate[:45]) + ("..." if len(candidate) > 45 else "")
            better += f" ({'in hand' if in_hand.startswith('Y') else 'not in hand'})"
        # allow line breaks after "/" so joined tokens (e.g. "estimated/extrapolated") can wrap
        brk = lambda s: s.replace("/", r"/\allowbreak{}")
        lines.append(f"  & {PROP_LABEL[prop]} & {coverage} & {tex_escape(label)} & "
                     f"{brk(tex_escape(source_type))} & {brk(better)} \\\\")
    lines.append(r"\hline")
lines.append(r"\bottomrule")
lines.append(r"\end{longtable}")

with open(f"{AD}/provenance_table.tex", "w") as f:
    f.write("\n".join(lines) + "\n")

print(f"wrote {AD}/provenance_table.tex ({len(MOLECULES)*len(PROPS)} data rows)")

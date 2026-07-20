"""
Worker Task 11 -- audit manuscript Table 3 (reference-data summary) against refs.bib, the live
esolvs.py reference dictionaries, and this audit's own provenance_map.csv.

Table 3 (`\\label{tab:exp_props}`, main_v7.tex lines 564-605) is hand-transcribed below directly
from the source (verified by reading the .tex source, not OCR/parsing -- the table has a irregular
threeparttable/tabular structure not worth writing a fragile LaTeX parser for).

Ground truth:
  - refs.bib: resolved by regex-extracting the @entry{key, ... title = {...}, journal = {...}} block
    for every key used in Table 3's References row and per-cell citations.
  - esolvs.py: imported LIVE (not re-parsed) so dict lengths reflect the actual current state,
    including the ME-10/ME-11 corrections and any other post-audit edits.
  - provenance_map.csv: this audit's existing PRIMARY/SECONDARY/CORRELATION/EOS-AUTH classification.

Output: audit_data/table3_refdata_audit.csv
"""
import csv
import re
import sys

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
# REPO = "/Users/adowling/DowlingLab/ES-FFO-Public"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
REPO = "/groups/ed/group_members/Montana_Carlozo/ES-FFO"
# MS = "/groups/ed/group_members/Montana_Carlozo/ES_FFO-2" #Need to fix this, path to manuscript repo, for refs.bib parsing.
AD = f"{AUDIT}/audit_data"

MOLECULES = ["EG", "Gly", "MeOH", "DMF", "DMSO", "DEC"]  # Table 3's own column order
PROPS = ["rho_l", "Pvap", "Hvap", "rho_v", "Tc", "rho_c", "gamma"]

# ============================================================================
# Table 3, hand-transcribed from main_v7.tex lines 570-597 (columns: EG, Gly, MeOH, DMF, DMSO, DEC)
# ============================================================================
REFERENCES_ROW = {
    "EG": ["Taylor1926", "Carvalho2015", "Tsierkezos1998", "Nikitin2018", "Shcherbina1975",
           "Verevkin2004", "yaws-thermophysical-properties", "Tawfik1989TheGlycols"],
    "Gly": ["Verevkin2015", "Ahmadi2023", "Prieto2016", "Erfani2019", "Alkindi2008",
            "Spittle2022EvolutionSolvents", "surface-tension-data", "Ghaedi2018DensityCapture",
            "Nikitin1993MeasurementMethod"],
    "MeOH": ["Goodwin1987", "Aliev2003", "Safarov2019", "Soukov2008", "Abdulagatov2005", "Stairs1970",
             "Zhang2010ExperimentalKPa", "Kuroki2001", "table-2-7aa-thermophysical", "Machado1983",
             "table-4-8-thermophysical", "yaws-thermophysical-properties", "Benson1948CriticalLiquids"],
    "DMF": ["Mohammad2013", "Komudziska2020", "Cui2006", "Bernal-Garca2008", "Alam2019TheProperties",
            "Stairs1970", "Teja1990TheResults", "Kahl2003", "AlTuwaim2018", "Rika2024", "Boyle1958",
            "yaws-thermophysical-properties"],
    "DMSO": ["Alam2019TheProperties", "Nishimura1972", "Campbell1979TheDensity", "Pruett1985",
             "Jakli1972", "Zhao2015", "LawrenceClever1963", "Harifi-Mood2017", "Geppert-Rybczyska2013",
             "Saleh2001", "Stairs1970", "NIST_DMSO", "Nikitin2018Critical26-Dimethylpiperazine",
             "ThomasDouglas1946", "Casteel1974", "Qian1995", "yaws-thermophysical-properties",
             "Lemmon2023"],
    "DEC": ["Pokorn2017", "Mansson1972", "Marrufo2011", "McKelvey1969", "Zhao2010", "Yang2006",
            "Kozlova2008", "Sun2017", "Zhang2010ExperimentalKPa", "Song2017", "Terrado2005",
            "yaws-thermophysical-properties"],
}

# (property, molecule) -> (total_exp_data, used_count, used_key, starred)
CELLS = {
    ("rho_l", "EG"): (97, 11, "REFPROP", True), ("rho_l", "Gly"): (84, 5, "Ghaedi2018DensityCapture", False),
    ("rho_l", "MeOH"): (315, 7, "REFPROP", True), ("rho_l", "DMF"): (33, 5, "Alam2019TheProperties", False),
    ("rho_l", "DMSO"): (68, 5, "Harifi-Mood2017", False), ("rho_l", "DEC"): (49, 5, "Zhao2010", False),

    ("Pvap", "EG"): (34, 11, "REFPROP", True), ("Pvap", "Gly"): (175, 5, "Verevkin2015", True),
    ("Pvap", "MeOH"): (987, 7, "REFPROP", True), ("Pvap", "DMF"): (83, 5, "Cui2006", True),
    ("Pvap", "DMSO"): (97, 5, "Jakli1972", True), ("Pvap", "DEC"): (132, 5, "Kozlova2008", True),

    ("Hvap", "EG"): (19, 11, "REFPROP", True), ("Hvap", "Gly"): (11, 5, "Verevkin2015", False),
    ("Hvap", "MeOH"): (83, 11, "REFPROP", True), ("Hvap", "DMF"): (1, 7, "Rika2024", False),
    ("Hvap", "DMSO"): (11, 5, "Lemmon2023", False), ("Hvap", "DEC"): (6, 5, "Kozlova2008", False),

    ("rho_v", "EG"): (0, 11, "REFPROP", True), ("rho_v", "Gly"): (0, 5, "Peng1976AState", True),
    ("rho_v", "MeOH"): (75, 7, "REFPROP", True), ("rho_v", "DMF"): (0, 5, "Peng1976AState", True),
    ("rho_v", "DMSO"): (0, 5, "Peng1976AState", True), ("rho_v", "DEC"): (31, 5, "Zhao2010", False),

    ("Tc", "EG"): (2, 1, "REFPROP", True), ("Tc", "Gly"): (3, 1, "Nikitin1993MeasurementMethod", False),
    ("Tc", "MeOH"): (5, 1, "REFPROP", True), ("Tc", "DMF"): (2, 1, "Teja1990TheResults", False),
    ("Tc", "DMSO"): (3, 1, "Nikitin2018Critical26-Dimethylpiperazine", False), ("Tc", "DEC"): (3, 1, "Zhang2010ExperimentalKPa", False),

    ("rho_c", "EG"): (2, 1, "Tawfik1989TheGlycols", True), ("rho_c", "Gly"): (1, 1, "Peng1976AState", True),
    ("rho_c", "MeOH"): (4, 1, "Benson1948CriticalLiquids", False), ("rho_c", "DMF"): (2, 1, "Teja1990TheResults", False),
    ("rho_c", "DMSO"): (2, 1, "Campbell1979TheDensity", False), ("rho_c", "DEC"): (2, 1, "Zhang2010ExperimentalKPa", False),

    ("gamma", "EG"): (11, 11, "REFPROP", True), ("gamma", "Gly"): (12, 5, "Erfani2019", False),
    ("gamma", "MeOH"): (40, 7, "REFPROP", True), ("gamma", "DMF"): (21, 5, "Kahl2003", True),
    ("gamma", "DMSO"): (16, 5, "LawrenceClever1963", True), ("gamma", "DEC"): (31, 5, "Zhao2010", False),
}

# Cells whose source is an EOS/database, not a literature paper -- legitimately absent from the
# molecule's References row.
EOS_KEYS = {"REFPROP", "Peng1976AState"}
# Known citation-key bug already confirmed (ME-01): the per-cell key should be
# Zhang2010ExperimentalKPa (in refs.bib, and already correctly listed in DEC's own References row)
# not the bare "Zhang2010" (a Cu-zeolite catalysis paper, refs.bib key Zhang2010 != Zhang2010ExperimentalKPa).
KNOWN_KEY_FIX = {"Zhang2010": "Zhang2010ExperimentalKPa"}

MRV_PROP_MAP = {"rho_l": "liq_density", "rho_v": "vap_density", "Pvap": "Pvap", "Hvap": "Hvap",
                "gamma": "surf_tens", "Tc": "Tc", "rho_c": "rhoc"}


def load_bib():
    """Split on '@type{' entry boundaries rather than matching each full entry with a single
    non-greedy regex -- some entries' fields contain a '\\n}' pattern inside a nested-brace value
    (e.g. a double-braced title '{{...}}') that can make a naive '(.*?)\\n\\}' match swallow past
    the true entry boundary and skip the next real entry(ies) entirely. Splitting on the reliable
    '@word{' start marker first, then extracting fields within each chunk, avoids that failure
    mode (confirmed: the naive approach silently dropped 3 of 404 entries, including one used by
    Table 3)."""
    text = open(f"../refs.bib", encoding="utf-8", errors="replace").read()
    entries = {}
    chunks = re.split(r'\n(?=@\w+\{)', text)
    for chunk in chunks:
        m = re.match(r'@(\w+)\{([^,]+),', chunk)
        if not m:
            continue
        key, body = m.group(2).strip(), chunk[m.end():]
        t = re.search(r'title\s*=\s*\{+(.*?)\}+,?\s*\n', body, re.S)
        j = re.search(r'journal\s*=\s*\{+(.*?)\}+,?\s*\n', body, re.S)
        entries[key] = {"title": t.group(1).strip() if t else "", "journal": j.group(1).strip() if j else ""}
    return entries


def load_esolvs_counts():
    sys.path.insert(0, f"{REPO}/utils/molec_class_files")
    import esolvs
    counts = {}
    for mol in MOLECULES:
        obj = getattr(esolvs, mol if mol != "Gly" else "Gly")
        counts[(mol, "rho_l")] = len(obj.expt_liq_density)
        counts[(mol, "Pvap")] = len(obj.expt_Pvap)
        counts[(mol, "Hvap")] = len(obj.expt_Hvap)
        counts[(mol, "rho_v")] = len(obj.expt_vap_density) if obj.expt_vap_density else 0
        counts[(mol, "gamma")] = len(obj.expt_surf_tens)
        counts[(mol, "Tc")] = 1
        counts[(mol, "rho_c")] = 1 if obj.expt_rhoc is not None else 0
    return counts


def load_provenance():
    prov = {}
    for r in csv.DictReader(open(f"{AD}/provenance_map.csv")):
        prov[(r["molecule"], r["property"])] = r["source_type"]
    return prov


bib = load_bib()
esolvs_counts = load_esolvs_counts()
prov = load_provenance()

rows_out = []

# ---- per-cell checks ----
for (prop, mol), (total, used, key, starred) in CELLS.items():
    flags = []
    resolved_key = KNOWN_KEY_FIX.get(key, key)
    bib_entry = bib.get(resolved_key)
    cite_ok = bib_entry is not None
    if key in KNOWN_KEY_FIX:
        flags.append(f"CITATION BUG: cites '{key}' (resolves to a DIFFERENT/unrelated paper in refs.bib) "
                      f"-- should be '{KNOWN_KEY_FIX[key]}' (ME-01)")
    elif not cite_ok:
        flags.append(f"CITATION KEY NOT FOUND in refs.bib: '{key}'")

    repo_count = esolvs_counts.get((mol, prop))
    count_flag = ""
    if repo_count is not None and repo_count != used:
        count_flag = f"COUNT MISMATCH: table says {used}, esolvs.py has {repo_count} temperature(s)"
        flags.append(count_flag)
    if used > total and not starred:
        # A starred (EOS/correlation-derived) cell legitimately can use more computed points than
        # the raw "Total Exp. Data" count, since "Total" counts measured data points and the
        # starred value is evaluated from a correlation/EOS at arbitrary temperatures (e.g. all
        # four rho_v EOS-AUTH cells: Total=0 measured points, but N points are still evaluated
        # from Peng-Robinson/REFPROP for grading). This check is only meaningful for a cell that
        # claims to use raw experimental data (no star).
        flags.append(f"INTERNAL INCONSISTENCY: 'Ref. Data in This Work' ({used}) exceeds 'Total Exp. Data' ({total}) for a non-starred (raw-data) cell")

    source_type = prov.get((mol, prop), "")
    star_expected = ("EOS-AUTH" in source_type) or ("CORRELATION" in source_type)
    if star_expected != starred:
        flags.append(f"STAR FLAG {'MISSING' if star_expected else 'SPURIOUS'} "
                      f"(provenance_map source_type='{source_type}')")

    in_refs_row = key in REFERENCES_ROW[mol] or resolved_key in REFERENCES_ROW[mol]
    if key not in EOS_KEYS and not in_refs_row:
        flags.append(f"'{key}' not found in {mol}'s References row (and is not an EOS/database source)")

    rows_out.append({
        "molecule": mol, "property": prop, "table_total_exp_data": total,
        "table_used_count": used, "table_used_key": key, "table_starred": starred,
        "bib_key_resolves": "Y" if cite_ok else "N",
        "bib_title": bib_entry["title"][:90] if bib_entry else "",
        "esolvs_count": repo_count, "provenance_source_type": source_type,
        "star_expected": star_expected,
        "flag": "; ".join(flags) if flags else "OK",
    })

# ---- References-row key resolution (separate rows) ----
for mol, keys in REFERENCES_ROW.items():
    for key in keys:
        entry = bib.get(key)
        rows_out.append({
            "molecule": mol, "property": "REFERENCES_ROW", "table_total_exp_data": "",
            "table_used_count": "", "table_used_key": key, "table_starred": "",
            "bib_key_resolves": "Y" if entry else "N",
            "bib_title": entry["title"][:90] if entry else "",
            "esolvs_count": "", "provenance_source_type": "",
            "star_expected": "",
            "flag": "OK" if entry else "CITATION KEY NOT FOUND in refs.bib",
        })

fields = ["molecule", "property", "table_total_exp_data", "table_used_count", "table_used_key",
          "table_starred", "bib_key_resolves", "bib_title", "esolvs_count",
          "provenance_source_type", "star_expected", "flag"]
with open(f"{AD}/table3_refdata_audit.csv", "w", newline="") as f:
    w = csv.DictWriter(f, fieldnames=fields)
    w.writeheader()
    for r in rows_out:
        w.writerow(r)

print(f"wrote {len(rows_out)} rows to {AD}/table3_refdata_audit.csv")
n_flagged = sum(1 for r in rows_out if r["flag"] != "OK")
print(f"{n_flagged} rows flagged")
for r in rows_out:
    if r["flag"] != "OK":
        print(f"  [{r['molecule']}/{r['property']}] {r['flag']}")

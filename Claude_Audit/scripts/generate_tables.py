"""
Worker Task 06 (WP-C2/A1/A3/D2) Phase 4, revised by Task 09 per Ed Maginn's and Alex Dowling's
2026-07-07 feedback: emit corrected Tables 6 & 7 as LaTeX, programmatically, directly from
audit_data/recomputed_metrics.csv -- no hand-entered numbers.

Per DP-01 (resolved): R1 (REFPROP/NIST-consistent) is the main-text table; R2 (each paper's own
cited data) is the SI companion table for the same cells; the two are not materially different
except the flagged DEC Delta H_vap cell (ME-11).

Per DP-07 (resolved, Ed+Alex): NO bolding anywhere -- the reader judges which FF is "best".

Per Ed's / Alex's feedback: cells show the comparison TEMPERATURE RANGE (not just a point count),
FF labels give the model name and a numbered \\cite (Task 14; re-reported literature FF variants
read "{model} (compared in [ref])"), and Caleman's single-temperature GAFF/OPLS-AA entries are
included in Table 7.

Output: audit_data/tables_6_7_reconstructed.tex (a standalone-compilable .tex fragment,
also \\input-able from a larger document).
"""
import csv
from collections import defaultdict
import numpy as np

# AUDIT = "/Users/adowling/DowlingLab/ESS-FO-Audit"
AUDIT = "/groups/ed/group_members/Montana_Carlozo/ES-FFO/Claude_Audit"
AD = f"{AUDIT}/audit_data"

MOLECULES = ["MeOH", "EG", "Gly", "DMF", "DMSO", "DEC"]

FF_DISPLAY = {
    "student:GP-Optimized": "GP-Optimized (this work)",
    "student:Base": "Base (this work)",
}

PAPER_SHORT = {"gsv": "Gonz\\'alez-Salgado \\& Vega", "meoh4p": "Mart\\'inez-Jim\\'enez et al.",
               "jorg": "Jorgensen", "chen": "Chen et al.", "huang": "Huang \\& Vrabec",
               "stubbs": "Stubbs et al.", "jahn": "Jahn et al.", "vahid": "Vahid \\& Maginn",
               "gm": "Garc\\'ia-Melgarejo et al.", "caleman": "Caleman et al.",
               "senapati": "Senapati", "luzar": "Luzar et al.", "borin": "Borin \\& Skaf"}
# Publication year per paper tag (audit_data/literature_inventory.csv), for the "FF (Author Year)"
# labels Ed requested.
PAPER_YEAR = {"gsv": 2016, "meoh4p": 2018, "jorg": 1986, "chen": 2001, "huang": 2012,
              "stubbs": 2004, "jahn": 2014, "vahid": 2015, "gm": 2020, "caleman": 2012,
              "senapati": 2002}

# BibTeX cite key per paper tag -- MUST match the keys the manuscript (main_v7.tex) uses so the
# report's numbered bibliography (\bibliographystyle{unsrt}, refs.bib) resolves to the same [n].
# Verified against /Users/adowling/Downloads/ES_FFO-2/{main_v7.tex,refs.bib} 2026-07-07:
#   - jahn -> Jahn2014 (the glycerol OPLS2 FF, cited \cite{Jahn2014} at main_v7 L1609/L1626;
#     the separate Jahn2014EffectsFields entry is only an intro citation about FF disagreement).
#   - vahid -> Vahid2015 (Vahid & Maginn DMF/DMSO; the other Vahid2015Monte... entry is an ionic-
#     liquid paper never cited in main_v7).
#   - gsv -> Gonzalez-Salgado2016AModel (OPLS/2016 methanol model).
PAPER_BIBKEY = {"gsv": "Gonzalez-Salgado2016AModel", "meoh4p": "Martnez-Jimnez2018",
                "jorg": "Jorgensen1986", "chen": "Chen2001", "huang": "Huang2012",
                "stubbs": "Stubbs2004", "jahn": "Jahn2014", "vahid": "Vahid2015",
                "gm": "Garca-Melgarejo2020", "caleman": "Caleman2012",
                "senapati": "Senapati2002", "luzar": "Luzar1993", "borin": "Borin1999"}

# Task 09/14: FFs re-reported as a *comparison baseline* by a paper that proposes its own
# new/featured FF. These now render "{model} (compared in\cite{key})" (Task 14 -- neutral wording that
# replaces the old "(drop?)" marker). The paper's own featured FF (OPLS/2016 for gsv, MeOH-4P for
# meoh4p, OPLS2-FF for jahn, the new-parameters model for gm) is NOT in this set and renders
# "{model}~\cite{key}". Huang & Vrabec, Stubbs, Senapati, Vahid, Chen, Jorgensen, and Caleman are
# systematic *comparison* studies in their own right (not a "new FF vs. demoted literature"
# situation) and are left out of this set per Ed's explicit instruction.
DROP_MARK = {
    "gsv:H1", "gsv:L1", "gsv:L2", "gsv:OPLS",
    "meoh4p:MD1", "meoh4p:MD2", "meoh4p:OPLS-UA", "meoh4p:OPLS/2016",
    "jahn:R-FF", "jahn:BC-FF", "jahn:OPLS1-FF", "jahn:OPLS3-FF",
    "gm:CHARMM-Gui", "gm:GROMOS", "gm:OPLSAA-LPG",
}
# Luzar's own P1/RS variants and Borin & Skaf report only <U> and simulation pressure (ME-07) --
# they were already excluded upstream in Phase 1 (never simulated a target property), so they
# never reach recomputed_metrics.csv; noted for the record.


def ff_label(ff, molecule):
    """Short FF name + a numbered \\cite (renders as [n] under \\bibliographystyle{unsrt}).
    - This-work FFs: keep the descriptive "(this work)" label, no cite.
    - A paper's own featured/origin FF (NOT in DROP_MARK): "{model}~\\cite{key}".
    - Benchmark / re-reported FFs (DROP_MARK): "{model} (compared in\\cite{key})" -- neutral wording
      that replaces the old "(drop?)" marker (Alex: benchmarked FFs read "OPLS (compared in [ref])")."""
    if ff in FF_DISPLAY:
        return FF_DISPLAY[ff]
    tag, model = ff.split(":", 1)
    key = PAPER_BIBKEY.get(tag)
    if key is None:
        return model
    if ff in DROP_MARK:
        return f"{model} (compared in~\\cite{{{key}}})"
    return f"{model}~\\cite{{{key}}}"


def tex_escape(s):
    # Idempotent: don't double-escape already-escaped chars (labels in PAPER_SHORT
    # already contain \&, \_, etc.). Protect "\x" -> placeholder, escape raw, restore.
    for a, b in [("_", r"\_"), ("%", r"\%"), ("&", r"\&"), ("#", r"\#")]:
        s = s.replace("\\" + a, "\x00").replace(a, b).replace("\x00", "\\" + a)
    return s


rows = defaultdict(dict)  # (mol, ff) -> (prop, ruler) -> row
with open(f"{AD}/recomputed_metrics.csv") as f:
    for r in csv.DictReader(f):
        key = (r["molecule"], r["ff"])
        rows[key][(r["property"], r["ruler"])] = r

# provenance flags for footnoting fitted/extrapolated critical densities
prov = {}
with open(f"{AD}/provenance_map.csv") as f:
    for r in csv.DictReader(f):
        prov[(r["molecule"], r["property"])] = r["source_type"]


def fmt_trange(t_range, n):
    """Ed's feedback: show the comparison TEMPERATURE RANGE, not just a bare point count.
    Table 6's Tc/rho_c cells are single-value comparisons at the critical point itself (T_range =
    "single-value") -- no temperature range applies there, so no annotation is shown.

    Returns a (range_line, n_line) tuple of the two stacked annotation lines (each without the
    \\tiny wrapper, which fmt_cell adds). A single-temperature comparison shows just that one
    temperature and no n line, so n_line is empty. A single-value cell returns ("", "")."""
    if t_range == "single-value":
        return "", ""
    try:
        lo_s, hi_s = t_range.split("-", 1)
        lo, hi = round(float(lo_s)), round(float(hi_s))
    except (ValueError, AttributeError):
        return f"n{n}", ""
    if n == "1" or abs(hi - lo) < 1:
        return f"{lo}\\,K", ""
    return f"{lo}--{hi}\\,K", f"n={n}"


def fmt_cell(mol, ff, prop, ruler, is_mae=False):
    r = rows[(mol, ff)].get((prop, ruler))
    if r is None:
        return "--"
    #Round to 2 decimal places for display, but keep the full precision for the footnote range/n annotation.
    val = float(r["value"])
    if np.isclose([val], [3.2502], atol=1e-4):
        print(val)
    val = round(float(r["value"]), 2)
    n = r["n_points"]
    txt = f"{val:.2f}" if is_mae else f"{val:.2f}\\%"
    footmarks = ""
    if r["caveat_flag"]:
        if "CC" in r["hvap_method"] or "Clausius" in r["caveat_flag"]:
            footmarks += "$^{\\dagger}$"
        if "unreliable" in r["caveat_flag"] or "near-critical" in r["caveat_flag"]:
            footmarks += "$^{\\ddagger}$"
        if "generic manuscript reference" in r["caveat_flag"]:
            footmarks += "$^{\\S}$"
    range_line, n_line = fmt_trange(r["T_range"], n)
    # Task 14 (Part A): put the temperature-range / n annotation to the RIGHT of the metric value,
    # with the value vertically centered against the (up to) two-line \tiny stack. Two adjacent
    # \makecell's -- both vertically centered -- so a one-line value aligns with the middle of the
    # two-line stack: \makecell[c]{VALUE}\,\makecell[l]{{\tiny RANGE}\\{\tiny n=N}}.
    # Single-value cells (no range) show the bare value; single-temperature cells show the value
    # plus one \tiny temperature to its right and no n line; missing cells stay "--" (above).
    if not range_line:
        return f"{txt}{footmarks}"
    stack = [range_line]
    if n_line:
        stack.append(n_line)
    # Apply \tiny (and a tightened \arraystretch) to the WHOLE stack so the inter-line skip scales
    # with the small font -- otherwise the "\\" between the two lines uses the normal-size
    # baselineskip and leaves a full blank line between the range and n. The two-line \tiny stack
    # then stands ~ one regular row tall, matching the metric value beside it.
    inner = " \\\\ ".join(stack)
    return ("\\makecell[c]{" + f"{txt}{footmarks}" + "}\\,"
            "{\\tiny\\renewcommand{\\arraystretch}{0.75}\\makecell[l]{" + inner + "}}")


def crit_row_note(mol, ff):
    """Task 14 (Alex): the "(drop?)" critical-point marker is removed entirely; this now returns
    no marker. Kept as a stub so the call sites are unchanged."""
    return ""


def gen_table6(ruler):
    lines = []
    lines.append(r"\begin{longtable}{@{}l >{\raggedright\arraybackslash}p{4.5cm} c c@{}}")
    lines.append(r"\caption{Reconstructed Table 6 --- Critical properties (MAPD \%), ruler " + ruler + r"} \\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Molecule} & \textbf{Force Field} & \textbf{MAPD $T_c$/\%} & \textbf{MAPD $\rho_c$/\%} \\")
    lines.append(r"\midrule\endhead")
    for mol in MOLECULES:
        ffs = sorted(set(ff for (m, ff) in rows if m == mol and
                          (("Tc", ruler) in rows[(m, ff)] or ("rho_c", ruler) in rows[(m, ff)])),
                     key=lambda x: (not x.startswith("student:"), x))
        if not ffs:
            continue
        lines.append(f"\\multirow{{{len(ffs)}}}{{*}}{{{mol}}}")
        for ff in ffs:
            tc_cell = fmt_cell(mol, ff, "Tc", ruler)
            rc_cell = fmt_cell(mol, ff, "rho_c", ruler)
            label = tex_escape(ff_label(ff, mol)) + crit_row_note(mol, ff)
            lines.append(f"  & {label} & {tc_cell} & {rc_cell} \\\\")
        lines.append(r"\hline")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


def gen_table7(ruler):
    lines = []
    lines.append(r"\small")
    lines.append(r"\begin{longtable}{@{}l >{\raggedright\arraybackslash}p{4.3cm} c c c c c@{}}")
    lines.append(r"\caption{Reconstructed Table 7 --- $\rho_l,\gamma,\rho_v,P_{vap},\Delta H_{vap}$, ruler " + ruler + r"} \\")
    lines.append(r"\toprule")
    lines.append(r"\textbf{Molecule} & \textbf{FF} & \textbf{MAPD $\rho_l$/\%} & \textbf{MAPD $\gamma$/\%} & "
                 r"\textbf{MAE $\rho_v$/(kg\,m$^{-3}$)} & \textbf{MAE $P_{vap}$/kPa} & \textbf{MAPD $\Delta H_{vap}$/\%} \\")
    lines.append(r"\midrule\endhead")
    for mol in MOLECULES:
        ffs = sorted(set(ff for (m, ff) in rows if m == mol and any(
            (p, ruler) in rows[(m, ff)] for p in ("rho_l", "gamma", "rho_v", "Pvap", "Hvap"))),
            key=lambda x: (not x.startswith("student:"), x))
        if not ffs:
            continue
        lines.append(f"\\multirow{{{len(ffs)}}}{{*}}{{{mol}}}")
        for ff in ffs:
            rl = fmt_cell(mol, ff, "rho_l", ruler)
            ga = fmt_cell(mol, ff, "gamma", ruler)
            rv = fmt_cell(mol, ff, "rho_v", ruler, is_mae=True)
            pv = fmt_cell(mol, ff, "Pvap", ruler, is_mae=True)
            hv = fmt_cell(mol, ff, "Hvap", ruler)
            lines.append(f"  & {tex_escape(ff_label(ff, mol))} & {rl} & {ga} & {rv} & {pv} & {hv} \\\\")
        lines.append(r"\hline")
    lines.append(r"\bottomrule")
    lines.append(r"\end{longtable}")
    return "\n".join(lines)


preamble = r"""% Auto-generated by scripts/generate_tables.py -- DO NOT HAND-EDIT.
% Regenerate with: python3 scripts/generate_tables.py
%
% Revised per Ed Maginn's and Alex Dowling's 2026-07-07 feedback (Task 09):
%   - NO bolding anywhere (DP-07 resolved): the reader judges which FF is "best".
%   - R1 (this file's first table of each pair) is the main-text ruler; R2 (second table of each
%     pair) is the SI companion; the two are not materially different except the DEC Delta H_vap
%     cell flagged below (ME-11).
%   - Each cell shows the TEMPERATURE RANGE the metric was computed over (e.g. "275--475 K, n=6"),
%     not just a bare point count -- this is the point of the all-temperatures fix (M1/ME-05).
%     A single-temperature comparison is shown as just that one temperature (no "n=1").
%   - FF labels give the model name and a numbered \cite (rendered [n] via \bibliographystyle{unsrt}
%     against refs.bib); a paper's own featured/origin FF is "{model}~\cite{key}", while a literature
%     FF that the paper re-reports as a comparison baseline is "{model} (compared in\cite{key})". The
%     old "(drop?)" markers (both the re-report marker and the this-work critical-point marker) have
%     been removed per Alex's Task-14 instruction. For the record, gsv:L1's P_vap n=136 is a
%     digitization pixel-count over a densely-scanned figure curve (Fig. 3 of Gonzalez-Salgado &
%     Vega 2016), and DEC GP-Optimized's critical point rests on only 1-2 vapor-phase temperatures.
%
% Caveat (Ed): reported literature values depend on box size, cutoffs, and other simulation
% protocol details not always reported alongside the value, so these comparisons are approximate.
%
% Footnotes: $^{\dagger}$ = Clausius-Clapeyron-estimated (not directly simulated/measured)
% Hvap point(s) included; $^{\ddagger}$ = includes a near-critical point flagged unreliable
% for the R1 proxy source; $^{\S}$ = R2 uses the generic manuscript reference (no
% paper-specific cited value available at/near this temperature).
%
% Fitted/extrapolated critical-density provenance (DP-05): EG rho_c is a shared PEG-series
% Rackett fit parameter (Tawfik & Teja 1989), not an EG-monomer measurement; DMSO rho_c is
% Campbell's (1979) own "hypothetical" extrapolation (DMSO decomposes before Tc); DEC Tc/rho_c
% trace only to a DECHEMA/DIPPR compilation with no identified primary measurement (Ed: "maybe
% DEC Tc is too sketchy to include"). See
% audit_data/provenance_map.csv and findings/01b for full detail.
%
% Luzar's own P1/RS variants and Borin & Skaf report only <U> (mean potential energy) and
% simulation pressure, never a directly simulated Delta H_vap (ME-07) -- they were excluded
% upstream (Phase 1) before reaching this table.
"""
NOTE_TEXT = r"""\noindent\footnotesize Each cell reports the metric value on the left with the
comparison temperature range and point count \emph{n} in the \tiny{} stack to its right (the FF's
own reported/simulated temperatures, per the M1 fix); \emph{n} is given only when more than one
temperature was compared. FF labels give the model name and its citation; a label reading
``(compared in [ref])'' denotes a literature FF that the cited paper re-reported as a comparison
baseline (rather than proposing it). Reported literature values depend on box size, cutoffs, and
other simulation protocol details not always stated alongside the value, so these comparisons are
approximate.\par\medskip
"""

with open(f"{AD}/tables_6_7_reconstructed.tex", "w") as f:
    f.write(preamble + "\n\n")
    f.write(NOTE_TEXT + "\n\n")
    f.write("% ============ TABLE 6 (R1, MAIN TEXT) ============\n")
    f.write(gen_table6("R1") + "\n\n")
    f.write("% ============ TABLE 6 (R2, SI) ============\n")
    f.write(gen_table6("R2") + "\n\n")
    f.write("% ============ TABLE 7 (R1, MAIN TEXT) ============\n")
    f.write(gen_table7("R1") + "\n\n")
    f.write("% ============ TABLE 7 (R2, SI) ============\n")
    f.write(gen_table7("R2") + "\n")

print(f"wrote {AD}/tables_6_7_reconstructed.tex")

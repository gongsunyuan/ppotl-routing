from __future__ import annotations

import csv
from pathlib import Path

ROOT = Path(__file__).resolve().parent
PAPER = ROOT / "mdpi-revision-advisor-a5-precise"
CSV_PATH = PAPER / "supplementary" / "paired_28_independent.csv"
OUT = ROOT / "wilcoxon-paired-tables.tex"
SCENES = ["LF", "Abilene", "NSFCNET", "Surge"]


def holm(pvalues: list[float]) -> list[float]:
    m = len(pvalues)
    order = sorted(range(m), key=lambda i: pvalues[i])
    adjusted = [0.0] * m
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, (m - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)
    return adjusted


with CSV_PATH.open(newline="", encoding="utf-8") as handle:
    rows = list(csv.DictReader(handle))
assert len(rows) == 28
wilcoxon_holm = holm([float(r["wilcoxon_p"]) for r in rows])
for row, value in zip(rows, wilcoxon_holm):
    row["wilcoxon_holm_p"] = value
with CSV_PATH.open("w", newline="", encoding="utf-8") as handle:
    writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)


def table_for_scene(scene: str) -> str:
    selected = [r for r in rows if r["scenario"] == scene]
    assert len(selected) == 7
    out = [
        "\\begin{table}[H]",
        "\\caption{Independent-test paired comparisons versus A1: " + scene + ". "
        "Both test families are corrected separately over the 28 comparisons using the Holm procedure.}",
        "\\label{tab:stats-" + scene.lower() + "}",
        "\\scriptsize",
        "\\setlength{\\tabcolsep}{2.8pt}",
        "\\begin{tabular}{@{}lccccccc@{}}",
        "\\toprule",
        "\\textbf{Comparison} & \\textbf{$\\Delta$} & \\textbf{95\\% CI} & \\textbf{$d_z$} & "
        "\\textbf{$t$ raw $p$} & \\textbf{$t$ Holm $p$} & "
        "\\textbf{Wilcoxon raw $p$} & \\textbf{Wilcoxon Holm $p$} \\\\",
        "\\midrule",
    ]
    for r in selected:
        out.append(
            f"{r['arm_a']}$-${r['arm_b']} & {float(r['mean_diff']):+.4f} & "
            f"[{float(r['ci95_low']):+.4f},{float(r['ci95_high']):+.4f}] & "
            f"{float(r['cohens_dz']):+.2f} & {float(r['t_p']):.4f} & "
            f"{float(r['holm_p']):.4f} & {float(r['wilcoxon_p']):.4f} & "
            f"{float(r['wilcoxon_holm_p']):.4f} \\\\"
        )
    out.extend(["\\bottomrule", "\\end{tabular}", "\\end{table}", ""])
    return "\n".join(out)


content = "% Complete paired t-test and Wilcoxon signed-rank tables (28 comparisons).\n\n"
content += "\n".join(table_for_scene(scene) for scene in SCENES)
OUT.write_text(content, encoding="utf-8")
(PAPER / "wilcoxon-paired-tables.tex").write_text(content, encoding="utf-8")

for name in ["mdpi-article.tex", "mdpi-article-clean.tex"]:
    path = PAPER / name
    text = path.read_text(encoding="utf-8-sig")
    section = text.index("\\label{app:stats28}")
    start = text.index("\\begin{table}[H]", section)
    end = text.index("\\label{app:factor}", start)
    end = text.rfind("\\section", start, end)
    if name == "mdpi-article.tex":
        replacement = "{\\color{blue}\n\\input{wilcoxon-paired-tables}\n}\n\n"
    else:
        replacement = "\\input{wilcoxon-paired-tables}\n\n"
    text = text[:start] + replacement + text[end:]
    old = (
        "Holm correction is applied jointly across this family using the paired t-test "
        "p-values; Wilcoxon results are retained in the supplementary CSV."
    )
    new = (
        "Paired t-tests and Wilcoxon signed-rank tests are reported with raw and "
        "Holm-adjusted p-values, each corrected jointly across the 28 comparisons "
        "within its own test family."
    )
    if old not in text:
        raise ValueError(f"Appendix description not found in {name}")
    text = text.replace(old, new, 1)
    old_stats = (
        "The original 28 comparisons (seven arms versus A1 in four scenarios) form "
        "the Holm-corrected family at $\\alpha=0.05$; component contrasts and "
        "comparisons with references are treated as exploratory."
    )
    new_stats = (
        "The original 28 comparisons (seven arms versus A1 in four scenarios) are "
        "corrected separately within each test family (paired t and Wilcoxon) at "
        "$\\alpha=0.05$; component contrasts and comparisons with references are "
        "treated as exploratory."
    )
    if old_stats not in text:
        raise ValueError(f"Statistical description not found in {name}")
    text = text.replace(old_stats, new_stats, 1)
    path.write_text(text, encoding="utf-8")

print(OUT)
print("Wilcoxon Holm range", min(wilcoxon_holm), max(wilcoxon_holm))

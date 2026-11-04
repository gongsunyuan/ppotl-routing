from __future__ import annotations

import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from statistics import mean, stdev

from scipy import stats

ROOT = Path(__file__).resolve().parent
RAW = ROOT / "independent-results" / "raw"
OUT = ROOT / "analysis"
OUT.mkdir(exist_ok=True)
SCENES = [
    ("LF", "Claranet_failure.gml", 500),
    ("Abilene", "Abilene.gml", 500),
    ("NSFCNET", "NFSCNET.gml", 500),
    ("Surge", "Claranet.gml", 1500),
]
ARMS = ["A1", "A1b", "A2", "A3", "A3b", "A4", "A5", "A6"]
BASELINES = ["OSPF", "ECMP", "LB", "RR"]


def read_rows():
    rows = []
    for path in RAW.glob("*/summary.csv"):
        with path.open(newline="", encoding="utf-8") as handle:
            for row in csv.DictReader(handle):
                row["rate"] = int(row["rate"])
                row["train_seed"] = int(row["train_seed"])
                for key in (
                    "original_best_score",
                    "independent_test_score",
                    "independent_rd_mean",
                    "independent_rp_mean",
                ):
                    row[key] = float(row[key])
                row["original_final_last_score"] = (
                    float(row["original_final_last_score"])
                    if row["original_final_last_score"]
                    else None
                )
                rows.append(row)
    return rows


ROWS = read_rows()


def records(arm, topo, rate):
    return sorted(
        [r for r in ROWS if r["arm"] == arm and r["topo"] == topo and r["rate"] == rate],
        key=lambda r: r["train_seed"],
    )


def values(arm, topo, rate, key="independent_test_score"):
    return [r[key] for r in records(arm, topo, rate)]


def paired(a, b, topo, rate):
    va = values(a, topo, rate)
    vb = values(b, topo, rate)
    diffs = [x - y for x, y in zip(va, vb)]
    avg = mean(diffs)
    sd = stdev(diffs)
    half = stats.t.ppf(0.975, len(diffs) - 1) * sd / math.sqrt(len(diffs))
    return {
        "arm_a": a,
        "arm_b": b,
        "topo": topo,
        "rate": rate,
        "n_seeds": len(diffs),
        "mean_diff": avg,
        "ci95_low": avg - half,
        "ci95_high": avg + half,
        "cohens_dz": avg / sd if sd else 0.0,
        "t_p": float(stats.ttest_1samp(diffs, 0).pvalue),
        "wilcoxon_p": float(stats.wilcoxon(diffs).pvalue) if any(diffs) else 1.0,
        "diffs": diffs,
    }


def holm(rows):
    ordered = sorted(range(len(rows)), key=lambda i: rows[i]["t_p"])
    running = 0.0
    for rank, idx in enumerate(ordered):
        running = max(running, (len(rows) - rank) * rows[idx]["t_p"])
        rows[idx]["holm_p"] = min(1.0, running)


def write_csv(name, rows, fields=None):
    if not rows:
        return
    fields = fields or [k for k in rows[0] if k != "diffs"]
    with (OUT / name).open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


summary = []
for scene, topo, rate in SCENES:
    for arm in [*ARMS, "A0", *BASELINES]:
        rr = records(arm, topo, rate)
        summary.append(
            {
                "scenario": scene,
                "arm": arm,
                "n": len(rr),
                "independent_mean": mean(r["independent_test_score"] for r in rr),
                "original_best_mean": mean(r["original_best_score"] for r in rr),
                "original_final_last_mean": (
                    mean(r["original_final_last_score"] for r in rr)
                    if rr[0]["original_final_last_score"] is not None
                    else ""
                ),
                "independent_rd_mean": mean(r["independent_rd_mean"] for r in rr),
                "independent_rp_mean": mean(r["independent_rp_mean"] for r in rr),
            }
        )
write_csv("score_summary.csv", summary)

stats28 = []
for scene, topo, rate in SCENES:
    for arm in ["A1b", "A2", "A3", "A3b", "A4", "A5", "A6"]:
        row = paired(arm, "A1", topo, rate)
        row["scenario"] = scene
        stats28.append(row)
holm(stats28)
write_csv("paired_28_independent.csv", stats28)

factors = {
    "PBRS": [("A4", "A1"), ("A5", "A2"), ("A6", "A3")],
    "Pretraining": [("A2", "A1"), ("A5", "A4")],
    "Freezing joint configuration": [("A3", "A2"), ("A6", "A5")],
    "Total budget": [("A6", "A1b")],
    "Adaptation over zero-shot": [("A6", "A0")],
}
factor_rows = []
for factor, pairs in factors.items():
    for scene, topo, rate in SCENES:
        for a, b in pairs:
            row = paired(a, b, topo, rate)
            row.update({"factor": factor, "scenario": scene})
            factor_rows.append(row)
write_csv("factor_contrasts_independent.csv", factor_rows)

a6_ospf = []
for scene, topo, rate in SCENES:
    baseline = values("OSPF", topo, rate)[0]
    diffs = [v - baseline for v in values("A6", topo, rate)]
    avg = mean(diffs)
    sd = stdev(diffs)
    half = stats.t.ppf(0.975, 9) * sd / math.sqrt(10)
    a6_ospf.append(
        {
            "scenario": scene,
            "mean_diff": avg,
            "ci95_low": avg - half,
            "ci95_high": avg + half,
            "cohens_dz": avg / sd,
            "t_p": float(stats.ttest_1samp(diffs, 0).pvalue),
        }
    )
write_csv("a6_vs_ospf_independent.csv", a6_ospf)

for name, rows in [
    ("paired_28_independent.csv", stats28),
    ("factor_contrasts_independent.csv", factor_rows),
]:
    print("\n", name)
    for r in rows:
        print(
            r.get("factor", "28-family"), r["scenario"], f"{r['arm_a']}-{r['arm_b']}",
            f"{r['mean_diff']:+.6f} [{r['ci95_low']:+.6f},{r['ci95_high']:+.6f}]",
            f"dz={r['cohens_dz']:+.3f} p={r['t_p']:.4g}",
            f"holm={r.get('holm_p', '')}",
        )
print("\nA6 vs OSPF", json.dumps(a6_ospf, indent=2))
print("Holm significant", sum(r["holm_p"] < 0.05 for r in stats28))

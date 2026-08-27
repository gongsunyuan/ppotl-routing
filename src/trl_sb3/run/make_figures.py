"""make_figures（M4，计划 §6 / checkbox #12）：runs 目录 → 论文图 2 幅 + 统计。

IO 编排：collect_runs（aggregate）→ compute_table 指标表 → 归一化（figures，
tex:475 口径：场景内跨全部 run 的 r_mean_mean 最大值做分母）→ 图 1 适应曲线
（figures.build_adaptation_figure）+ 图 2 渐近柱（build_asymptote_figure）→
PNG(150dpi)+PDF；统计 stats.json/csv：每对（学习臂 vs A1）逐种子配对差 +
t/Wilcoxon p + cohens_d，跨全部场景对 Holm 校正（族主指标 t_p）。θ/W/k 读
预注册 YAML（config/metrics_prereg.yaml）。figure_manifest.json 记录生成参数
（runs 路径、归一化分母、prereg 摘要、时间戳）。

CLI：`python -m trl_sb3.run make_figures --runs runs_smoke --out figures_smoke`。
"""

from __future__ import annotations

import argparse
import csv
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from matplotlib import pyplot as plt

from trl_sb3.common.config import load_config, resolve_path
from trl_sb3.run.aggregate import (
    NON_SCORING_ARMS,
    collect_runs,
    compute_table,
    holm_family,
    paired_arm_diffs,
)
from trl_sb3.run.figures import (
    _scene_of,
    build_adaptation_figure,
    build_asymptote_figure,
    normalize_records,
)

_BASELINE_ARM = "A1"


def compute_stats(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """统计族（纯函数）：逐场景每学习臂 vs A1 逐种子配对差（final r_mean_mean）
    + paired t/Wilcoxon/cohens_d（aggregate.paired_arm_diffs），跨全部对 Holm
    校正（holm_p 追加入 dict）。共享种子 <2 的对跳过（全量 10 种子不触发）。"""
    pairs: list[dict[str, Any]] = []
    scenes = sorted({_scene_of(r) for r in records if r["curve"] and r["arm"] not in NON_SCORING_ARMS})
    for topo, rate in scenes:
        arms = sorted(
            {
                r["arm"]
                for r in records
                if _scene_of(r) == (topo, rate) and r["curve"]
                and r["arm"] not in NON_SCORING_ARMS and r["arm"] != _BASELINE_ARM
            }
        )
        for arm in arms:
            try:
                pairs.append(paired_arm_diffs(records, arm, _BASELINE_ARM, topo, rate))
            except ValueError:
                continue
    return [dict(pair, holm_p=p) for pair, p in zip(pairs, holm_family(pairs), strict=True)]


def _save(fig, out_dir: Path, stem: str) -> list[Path]:
    """Figure → PNG(150dpi) + 同名 PDF；写完关闭图。"""
    written: list[Path] = []
    for suffix, kwargs in ((".png", {"dpi": 150}), (".pdf", {})):
        path = out_dir / f"{stem}{suffix}"
        fig.savefig(path, bbox_inches="tight", **kwargs)
        written.append(path)
    plt.close(fig)
    return written


def _write_stats(stats: list[dict[str, Any]], out_dir: Path, prereg: dict[str, Any]) -> None:
    """统计 → stats.json（全量嵌套）+ stats.csv（扁平行）。"""
    (out_dir / "stats.json").write_text(
        json.dumps(
            {
                "alpha": prereg.get("alpha"),
                "primary_metric": prereg.get("primary_metric"),
                "baseline": _BASELINE_ARM,
                "correction": "Holm-Bonferroni over t_p family (all scenes × arms)",
                "pairs": stats,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    with (out_dir / "stats.csv").open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ["arm_a", "arm_b", "topo", "rate", "n_seeds", "seeds",
             "mean_diff", "cohens_d", "t_p", "wilcoxon_p", "holm_p"]
        )
        for pair in stats:
            writer.writerow(
                [pair["arm_a"], pair["arm_b"], pair["topo"], pair["rate"], len(pair["seeds"]),
                 ";".join(map(str, pair["seeds"])), pair["mean_diff"], pair["cohens_d"],
                 pair["t_p"], pair["wilcoxon_p"], pair["holm_p"]]
            )


def make_figures(
    runs_dir: str | Path, out_dir: str | Path = "figures", prereg_path: str | Path = "config/metrics_prereg.yaml"
) -> dict[str, Any]:
    """IO 编排：collect_runs → 表/归一化/统计 → 2 图 + stats + figure_manifest。

    相对路径锚定 experiments/ 根（resolve_path）。返回写出的 manifest dict
    （分母、prereg 摘要、文件清单、时间戳——run_id 确定性约束只涉 run 目录，
    图产物不带该约束）。
    """
    runs_path, out_path = resolve_path(runs_dir), resolve_path(out_dir)
    prereg = load_config(resolve_path(prereg_path))
    records = collect_runs(runs_path)
    if not records:
        raise ValueError(f"runs 目录无 DONE 产物：{runs_path}")
    table = compute_table(records, prereg)
    normalized, denominators = normalize_records(records)
    out_path.mkdir(parents=True, exist_ok=True)
    figures = [
        *_save(build_adaptation_figure(normalized, table, int(prereg["window_episodes"])), out_path, "adaptation_curves"),
        *_save(build_asymptote_figure(table, denominators), out_path, "asymptote_bars"),
    ]
    stats = compute_stats(records)
    _write_stats(stats, out_path, prereg)
    manifest = {
        "created_at": datetime.now(UTC).isoformat(),
        "runs": str(runs_path),
        "out": str(out_path),
        "normalization": "per-(topo,rate) max of r_mean_mean across all runs (MDPI tex:475)",
        "denominators": {f"{topo}@{rate}": value for (topo, rate), value in sorted(denominators.items())},
        "prereg": {key: prereg.get(key) for key in ("theta", "window_episodes", "asymptote_k", "alpha", "primary_metric")},
        "figures": [path.name for path in figures],
        "stats_files": ["stats.json", "stats.csv"],
        "n_runs": len(records),
    }
    (out_path / "figure_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(
        f"[make_figures] runs={manifest['runs']} n_runs={manifest['n_runs']} "
        f"denominators={manifest['denominators']} -> {manifest['out']}"
    )
    return manifest


def main(argv: list[str] | None = None) -> int:
    """CLI：`python -m trl_sb3.run make_figures --runs runs_smoke --out figures_smoke`。"""
    parser = argparse.ArgumentParser(
        prog="python -m trl_sb3.run make_figures", description="runs 目录 → 论文图 + 统计（M4）"
    )
    parser.add_argument("--runs", default="runs", help="runs 目录（相对路径锚定 experiments/ 根）")
    parser.add_argument("--out", default="figures", help="输出目录（缺省 figures/）")
    parser.add_argument("--prereg", default="config/metrics_prereg.yaml", help="指标预注册 YAML 路径")
    args = parser.parse_args(argv)
    make_figures(args.runs, args.out, args.prereg)
    return 0

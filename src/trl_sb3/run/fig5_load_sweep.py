"""论文 Fig5 替换图：源域工作点 vs 3× 突增的分组柱状对比。

数据源：runs/ 基线行（OSPF/ECMP/LB/RR，零参数评估）与学习行 eval.json
final。仅呈现方法实际训练过的两个负载档（500 = 源域预训练、1500 = 突增
适应）；250-2000 全档基线测绘行保留在 runs/ 并打印到日志供背景引用，
但不上图。y 轴为原始 r_mean_mean（跨速率可比）。

用法：
    python -m trl_sb3.run.fig5_load_sweep --runs runs --out-dir figures_main
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle

RATES_SWEEP = (250, 500, 1000, 1500, 2000)  # runs 数据完整性校验与日志背景
RATES_PLOTTED = (500, 1500)
BASELINE_ARMS = ("OSPF", "ECMP", "LB", "RR")
TOPO = "Claranet.gml"
METHOD_BARS: dict[int, tuple[tuple[str, str], ...]] = {
    500: (("pretrain", "PPO-TL pretrain (peak)"),),
    1500: (
        ("A1", "A1 scratch"),
        ("A2", "A2 pretrain+FT"),
        ("A5", "A5 pretrain+PBRS (PPO-TL)"),
    ),
}
GROUP_LABELS = {500: "500 (source)", 1500: r"1500 ($3\times$ surge)"}


def _records(runs: Path) -> list[dict[str, Any]]:
    """runs/ → Claranet 全部 DONE 行（manifest 字段 + eval final）。"""
    out: list[dict[str, Any]] = []
    for run in sorted(runs.iterdir()):
        manifest_path, eval_path = run / "manifest.json", run / "eval.json"
        if not (
            manifest_path.is_file() and eval_path.is_file() and (run / "DONE").is_file()
        ):
            continue
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        if manifest.get("topo") != TOPO:
            continue
        ev = json.loads(eval_path.read_text(encoding="utf-8"))
        out.append({**manifest, "final": ev["final"]})
    return out


def _baseline_by_rate(records: list[dict[str, Any]]) -> dict[str, dict[int, float]]:
    """逐基线臂 速率→final 值（扫档缺行 → ValueError，宁可失败）。"""
    series: dict[str, dict[int, float]] = {}
    for arm in BASELINE_ARMS:
        by_rate = {
            r["rate"]: r["final"]["r_mean_mean"] for r in records if r["arm"] == arm
        }
        missing = [rate for rate in RATES_SWEEP if rate not in by_rate]
        if missing:
            raise ValueError(f"基线 {arm} 缺速率档 {missing}（负载扫行不全）")
        series[arm] = by_rate
    return series


def _method_point(
    records: list[dict[str, Any]], arm: str, rate: int
) -> tuple[float, float]:
    """方法柱：(种子 final 均值, SD)。pretrain 行 = 源域峰值检查点评估。"""
    values = [
        r["final"]["r_mean_mean"]
        for r in records
        if r["arm"] == arm and r["rate"] == rate
    ]
    if len(values) < 2:
        raise ValueError(f"{arm}@{rate} 种子数 {len(values)} < 2（无误差棒可言）")
    arr = np.array(values, dtype=np.float64)
    return float(arr.mean()), float(arr.std(ddof=1))


def build_figure(runs: Path, out_dir: Path) -> list[Path]:
    """双工作点分组柱状图：基线（确定性，无误差棒）+ 方法臂（SD 误差棒）。"""
    records = _records(runs)
    baselines = _baseline_by_rate(records)
    points = {
        (arm, rate): _method_point(records, arm, rate)
        for rate, bars in METHOD_BARS.items()
        for arm, _ in bars
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    baseline_color: dict[str, str] = dict(zip(BASELINE_ARMS, colors[:4], strict=True))
    method_color: dict[str, str] = {
        arm: colors[4 + i]
        for i, arm in enumerate(arm for bars in METHOD_BARS.values() for arm, _ in bars)
    }
    bar_width, group_span = 0.09, 0.72
    for group_index, rate in enumerate(RATES_PLOTTED):
        entries: list[tuple[str, float, float | None]] = [
            (arm, baselines[arm][rate], None) for arm in BASELINE_ARMS
        ]
        entries += [
            (arm, points[(arm, rate)][0], points[(arm, rate)][1])
            for arm, _ in METHOD_BARS[rate]
        ]
        n = len(entries)
        xs = np.linspace(group_index - group_span / 2, group_index + group_span / 2, n)
        for x, (arm, mean, sd) in zip(xs, entries, strict=True):
            color = baseline_color.get(arm, method_color.get(arm))
            assert color is not None  # 未登记的臂不上图
            ax.bar(
                x,
                mean,
                width=bar_width,
                color=color,
                yerr=sd,
                error_kw={"elinewidth": 1.4, "capsize": 4},
            )
    handles = [
        Rectangle((0, 0), 1, 1, color=baseline_color[arm]) for arm in BASELINE_ARMS
    ] + [Rectangle((0, 0), 1, 1, color=method_color[arm]) for arm in method_color]
    labels = list(BASELINE_ARMS) + [
        label for bars in METHOD_BARS.values() for _, label in bars
    ]
    ax.legend(handles, labels, frameon=False, fontsize=8, ncol=2)
    ax.set_xticks(range(len(RATES_PLOTTED)))
    ax.set_xticklabels([GROUP_LABELS[rate] for rate in RATES_PLOTTED])
    ax.set_xlim(-0.55, len(RATES_PLOTTED) - 0.45)
    ax.set_xlabel("Average traffic rate (packets/s)")
    ax.set_ylabel("Greedy eval reward (raw r_mean)")
    ax.set_title("Source operating point vs. traffic surge on Claranet")
    ax.grid(alpha=0.3, axis="y")
    out_paths: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig5_load_sweep.{ext}"
        fig.savefig(path, dpi=200)
        out_paths.append(path)
    plt.close(fig)
    for rate in RATES_SWEEP:
        print(
            f"[fig5] sweep@{rate}: "
            + " ".join(f"{arm}={baselines[arm][rate]:.4f}" for arm in BASELINE_ARMS)
        )
    for (arm, rate), (mean, sd) in points.items():
        print(f"[fig5] method {arm}@{rate}: {mean:.4f} ± {sd:.4f}")
    return out_paths


def main(argv: list[str] | None = None) -> int:
    desc = (__doc__ or "").splitlines()[0] if __doc__ else "fig5 load sweep figure"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument("--runs", type=Path, default=Path("runs"), help="runs 目录")
    parser.add_argument(
        "--out-dir", type=Path, default=Path("figures_main"), help="输出目录"
    )
    args = parser.parse_args(argv)
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in build_figure(args.runs, args.out_dir):
        print(f"[fig5] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

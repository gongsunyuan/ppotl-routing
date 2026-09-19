"""论文 Fig5 替换图：Claranet 基线负载响应 + 方法参考点。

数据源：runs/ 基线行（OSPF/ECMP/LB/RR × 速率 250-2000，预注册
narrative_extras.load_sweep，零参数评估）与学习行 eval.json final。
y 轴为原始 r_mean_mean——跨速率可比（场景内归一分母随速率变化，不适用）。
方法参考点：源域 500 = 预训练峰值（10 种子），1500 = 适应臂终评（10 种子）。

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

RATES = (250, 500, 1000, 1500, 2000)
BASELINE_ARMS = ("OSPF", "ECMP", "LB", "RR")
TOPO = "Claranet.gml"
METHOD_ARMS_AT_1500 = ("A2", "A6")
FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def _records(runs: Path) -> list[dict[str, Any]]:
    """runs/ → Claranet 全部 DONE 行（manifest 字段 + eval final/best）。"""
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
        out.append({**manifest, "final": ev["final"], "best": ev.get("best")})
    return out


def _baseline_series(
    records: list[dict[str, Any]],
) -> dict[str, FloatArray]:
    """逐基线臂 5 档速率的 final 值（缺任一档 → ValueError，宁可失败）。"""
    series: dict[str, FloatArray] = {}
    for arm in BASELINE_ARMS:
        by_rate = {
            r["rate"]: r["final"]["r_mean_mean"] for r in records if r["arm"] == arm
        }
        missing = [rate for rate in RATES if rate not in by_rate]
        if missing:
            raise ValueError(f"基线 {arm} 缺速率档 {missing}（负载扫行不全）")
        series[arm] = np.array([by_rate[rate] for rate in RATES], dtype=np.float64)
    return series


def _method_point(
    records: list[dict[str, Any]], arm: str, rate: int
) -> tuple[float, float]:
    """方法参考点：(10 种子 final 均值, SD)。pretrain 行 = 源域峰值检查点评估。"""
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
    """单面板负载响应图：4 基线折线 + 方法参考点（误差棒 = 种子 SD），PNG+PDF。"""
    records = _records(runs)
    series = _baseline_series(records)
    points = {
        ("pretrain", 500): _method_point(records, "pretrain", 500),
        **{
            (arm, 1500): _method_point(records, arm, 1500)
            for arm in METHOD_ARMS_AT_1500
        },
    }
    fig, ax = plt.subplots(figsize=(8.2, 5.2), constrained_layout=True)
    colors = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    for arm, color in zip(BASELINE_ARMS, colors[:4], strict=True):
        ax.plot(RATES, series[arm], marker="o", color=color, label=arm)
    method_specs = (
        (("pretrain", 500), "*", "PPO-TL source peak (pretrain@500)"),
        (("A2", 1500), "D", "A2 pretrain+finetune@1500"),
        (("A6", 1500), "s", "A6 full method@1500"),
    )
    for (key, marker, label), color in zip(method_specs, colors[4:7], strict=True):
        mean, sd = points[key]
        rate = key[1]
        ax.errorbar(
            [rate],
            [mean],
            yerr=[sd],
            marker=marker,
            markersize=11,
            color=color,
            linewidth=0,
            elinewidth=1.4,
            capsize=4,
            label=f"{label} (n=10)",
        )
    ax.set_xticks(RATES)
    ax.set_xlabel("Average traffic rate (packets/s)")
    ax.set_ylabel("Greedy eval reward (raw r_mean)")
    ax.set_title("Baseline load response on Claranet with method reference points")
    ax.grid(alpha=0.3)
    ax.legend(frameon=False, fontsize=9)
    out_paths: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig5_load_sweep.{ext}"
        fig.savefig(path, dpi=200)
        out_paths.append(path)
    plt.close(fig)
    for arm, values in series.items():
        print(f"[fig5] {arm}: " + " ".join(f"{v:.4f}" for v in values))
    for key, (mean, sd) in points.items():
        print(f"[fig5] method {key[0]}@{key[1]}: {mean:.4f} ± {sd:.4f}")
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

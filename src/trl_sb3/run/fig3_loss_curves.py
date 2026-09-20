"""论文 Fig3 替换图：源域训练收敛对照（我们方法 vs 朴素 PPO，3 种子均值±CI）。

数据源：每 run 目录的 losses.csv（每 PPO 更新一批，DEVIATIONS #18）与
eval.json 的周期评估曲线。两侧同域同预算（Claranet@500 × 15000 回合）——
我们方法侧 = 预训练 run（PPO+PBRS），对照侧 = A1@源域（从零、无 PBRS）。
loss 曲线经滚动均值平滑（窗口可调，默认 200 更新≈200 回合）。

用法：
    python -m trl_sb3.run.fig3_loss_curves \
        --ours runs/pretrain-a runs/pretrain-b runs/pretrain-c \
        --baseline runs/A1-x runs/A1-y runs/A1-z \
        --out-dir figures_main
"""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

SMOOTH_WINDOW = 200  # 更新数（= 回合数，n_steps=50 → 每 rollout 一次更新）
SCENE = "Claranet@500"
FloatArray = np.ndarray[tuple[int, ...], np.dtype[np.float64]]


def _load_losses(run: Path) -> dict[str, FloatArray]:
    with (run / "losses.csv").open(encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    out: dict[str, FloatArray] = {}
    for key in ("update", "policy_gradient_loss", "value_loss", "entropy_loss"):
        out[key] = np.array([float(r[key]) for r in rows], dtype=np.float64)
    return out


def _load_reward(run: Path) -> tuple[FloatArray, FloatArray]:
    ev = json.loads((run / "eval.json").read_text(encoding="utf-8"))
    episodes = np.array([p["episode"] for p in ev["curve"]], dtype=np.float64)
    values = np.array([p["r_mean_mean"] for p in ev["curve"]], dtype=np.float64)
    return episodes, values


def _smooth(values: FloatArray, window: int) -> FloatArray:
    if window <= 1 or values.size == 0:
        return values
    kernel = np.ones(min(window, values.size)) / min(window, values.size)
    smoothed = np.convolve(values, kernel, mode="valid")
    return smoothed.astype(np.float64, copy=False)


def _band(
    ax: Any, x: FloatArray, runs: list[FloatArray], color: str, label: str
) -> None:
    """3 种子均值±t95%CI 带（不同 run 长度取最短公共长度对齐）。"""
    n = min(r.size for r in runs)
    mat = np.vstack([r[:n] for r in runs])
    x = x[:n]
    mean = mat.mean(axis=0)
    sem = mat.std(axis=0, ddof=1) / np.sqrt(mat.shape[0])
    ci = 4.303 * sem  # t(2, 0.975)，n=3
    ax.plot(x, mean, color=color, linewidth=1.6, label=label)
    ax.fill_between(x, mean - ci, mean + ci, color=color, alpha=0.18, linewidth=0)


def build_figure(ours: list[Path], baseline: list[Path], out_dir: Path) -> list[Path]:
    """三面板收敛图（actor loss / critic loss / 奖励），PNG+PDF 双格式出图。"""
    ours_losses = [_load_losses(r) for r in ours]
    base_losses = [_load_losses(r) for r in baseline]
    fig, axes = plt.subplots(1, 3, figsize=(16.5, 4.6), constrained_layout=True)
    for ax, (title, key) in zip(
        axes[:2],
        (
            ("|Actor loss| (policy gradient)", "policy_gradient_loss"),
            ("|Critic loss| (value)", "value_loss"),
        ),
        strict=True,
    ):
        for losses, color, label in (
            (
                ours_losses,
                plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
                "PPO-TL (ours, +PBRS)",
            ),
            (
                base_losses,
                plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
                "PPO (scratch, no PBRS)",
            ),
        ):
            # 绝对值先行再平滑：策略梯度损失约八成为负，直接上对数轴会整段不可见
            smoothed = [
                _smooth(np.abs(losses_i[key]), SMOOTH_WINDOW) for losses_i in losses
            ]
            xs = [
                losses_i["update"][: s.size]
                for losses_i, s in zip(losses, smoothed, strict=True)
            ]
            n = min(s.size for s in smoothed)
            _band(ax, xs[0], [s[:n] for s in smoothed], color, label)
        ax.set_title(title)
        ax.set_xlabel("Episode")
        ax.set_yscale("log")
        ax.grid(alpha=0.3)
    ax = axes[2]
    for runs, color, label in (
        (
            ours,
            plt.rcParams["axes.prop_cycle"].by_key()["color"][0],
            "PPO-TL (ours, +PBRS)",
        ),
        (
            baseline,
            plt.rcParams["axes.prop_cycle"].by_key()["color"][1],
            "PPO (scratch, no PBRS)",
        ),
    ):
        curves = [_load_reward(r) for r in runs]
        n = min(e.size for e, _ in curves)
        _band(ax, curves[0][0], [v[:n] for _, v in curves], color, label)
    ax.set_title("Reward (greedy eval, every 100 episodes)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("r_mean")
    ax.grid(alpha=0.3)
    for ax in axes:
        ax.legend(frameon=False, fontsize=9)
    fig.suptitle(
        f"Training convergence at source domain ({SCENE}, 15000 episodes, mean ± 95% CI, n={len(ours)} seeds)"
    )
    out_paths: list[Path] = []
    for ext in ("png", "pdf"):
        path = out_dir / f"fig3_convergence.{ext}"
        fig.savefig(path, dpi=200)
        out_paths.append(path)
    plt.close(fig)
    return out_paths


def main(argv: list[str] | None = None) -> int:
    desc = (__doc__ or "").splitlines()[0] if __doc__ else "fig3 convergence figure"
    parser = argparse.ArgumentParser(description=desc)
    parser.add_argument(
        "--ours",
        nargs="+",
        required=True,
        type=Path,
        help="我们方法侧 run 目录（预训练，≥2 个种子）",
    )
    parser.add_argument(
        "--baseline",
        nargs="+",
        required=True,
        type=Path,
        help="对照侧 run 目录（A1@源域，≥2 个种子）",
    )
    parser.add_argument("--out-dir", type=Path, default=Path("figures_main"))
    args = parser.parse_args(argv)
    if len(args.ours) < 2 or len(args.baseline) < 2:
        parser.error("两侧各需至少 2 个种子（CI 带）")
    args.out_dir.mkdir(parents=True, exist_ok=True)
    for path in build_figure(args.ours, args.baseline, args.out_dir):
        print(f"[fig3] wrote {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

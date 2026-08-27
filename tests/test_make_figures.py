"""M4 make_figures 契约测试：归一化纯函数（tex:475 口径：场景内跨全部 run 取
最大值做分母）、NSFCNET 论文拼写标签、Holm 校正族单调性、端到端出图（真
runs_smoke 数据优先，缺则合成 runs 目录 fixture——8 臂 + OSPF/ECMP/LB/RR/A0）。

锁定：normalize_records 值域 [0,1] 且逐场景 max=1、不改输入；TOPO_LABELS 仅
含论文拼写映射；compute_stats 的 holm_p 按原 p 升序单调不减且 ≥ 原 p；产物
契约 2 PNG + 2 PDF + stats.json/csv + figure_manifest.json 且图文件 >10KB。
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from trl_sb3.run.figures import TOPO_LABELS, normalize_records, topo_label
from trl_sb3.run.make_figures import compute_stats, make_figures

PKG_ROOT = Path(__file__).resolve().parents[1]
SMOKE_RUNS = PKG_ROOT / "runs_smoke"
_PREREG = {"theta": None, "window_episodes": 500, "asymptote_k": 5, "alpha": 0.05, "primary_metric": "tau"}


def _record(
    arm: str, seed: int, curve: list[tuple[int, float]], final: float, *, topo: str = "Abilene.gml", rate: int = 500
) -> dict[str, Any]:
    return {
        "run_id": f"{arm}-{seed}-{topo}",
        "arm": arm,
        "topo": topo,
        "rate": rate,
        "seed": seed,
        "episodes": 600,
        "factors": None,
        "source_run_id": None,
        "curve": [{"episode": e, "r_mean_mean": v} for e, v in curve],
        "final": {"r_mean_mean": final},
    }


def _curve(start: float, end: float) -> list[tuple[int, float]]:
    return [(100 * i, start + (end - start) * (i - 1) / 5) for i in range(1, 7)]


def test_normalize_records_bounds_and_per_scene_max() -> None:
    """Given 两场景记录（Abilene max=0.4 跨 run；NFSCNET 自成 max=0.8）；When
    normalize_records；Then 值域 [0,1]、逐场景最大值=1、输入不被原地修改。"""
    records = [
        _record("A1", 0, [(5, 0.2), (10, 0.4)], 0.4),
        _record("A2", 0, [(5, 0.1), (10, 0.2)], 0.2),
        _record("A1", 1, [(5, 0.5), (10, 0.8)], 0.8, topo="NFSCNET.gml"),
    ]
    normalized, denominators = normalize_records(records)
    assert denominators == {("Abilene.gml", 500): pytest.approx(0.4), ("NFSCNET.gml", 500): pytest.approx(0.8)}
    scenes = {("Abilene.gml", 500), ("NFSCNET.gml", 500)}
    for scene in scenes:
        values = [
            p["r_mean_mean"] for r in normalized if (r["topo"], r["rate"]) == scene for p in r["curve"]
        ] + [r["final"]["r_mean_mean"] for r in normalized if (r["topo"], r["rate"]) == scene]
        assert all(0.0 <= v <= 1.0 for v in values)
        assert max(values) == pytest.approx(1.0)
    assert records[0]["curve"][0]["r_mean_mean"] == 0.2  # 纯函数：输入不被改


def test_topo_label_uses_paper_spelling() -> None:
    """Given NFSCNET.gml 与其余拓扑文件名；When topo_label；Then 论文拼写 NSFCNET、
    其余去 .gml 后缀、映射表仅含论文差异项。"""
    assert topo_label("NFSCNET.gml") == "NSFCNET"
    assert topo_label("Abilene.gml") == "Abilene"
    assert topo_label("CERNET_failure.gml") == "CERNET_failure"
    assert TOPO_LABELS == {"NFSCNET.gml": "NSFCNET"}


def test_compute_stats_holm_monotone_nondecreasing() -> None:
    """Given 单场景 3 学习臂（A1/A1b/A2 各 2 种子、final 两两有差）；When
    compute_stats；Then 族=A1b/A2 vs A1 逐种子配对对、holm_p 按原 p 升序单调
    不减且逐对 ≥ 原 t_p。"""
    records = [
        _record("OSPF", 0, [], 0.10),
        _record("A1", 0, _curve(0.1, 0.60), 0.60),
        _record("A1", 1, _curve(0.1, 0.66), 0.66),
        _record("A1b", 0, _curve(0.1, 0.62), 0.62),
        _record("A1b", 1, _curve(0.1, 0.64), 0.64),
        _record("A2", 0, _curve(0.1, 0.30), 0.30),
        _record("A2", 1, _curve(0.1, 0.35), 0.35),
    ]
    stats = compute_stats(records)
    assert {pair["arm_a"] for pair in stats} == {"A1b", "A2"}  # A1 vs A1 不入族
    assert all(pair["seeds"] == [0, 1] and len(pair["diffs"]) == 2 for pair in stats)
    assert all(pair["holm_p"] >= pair["t_p"] - 1e-12 for pair in stats)
    ordered = [pair["holm_p"] for pair in sorted(stats, key=lambda p: p["t_p"])]
    assert all(later >= earlier for earlier, later in zip(ordered, ordered[1:]))


def _synthetic_runs(runs: Path) -> None:
    """合成 runs 目录 fixture（runs_smoke 缺席时的替身）：8 臂 × 2 种子曲线 +
    OSPF/ECMP/LB/RR 各 1 行 + A0 × 2（同一 Abilene@500 场景）。"""
    def write(run_id: str, arm: str, seed: int, curve: list[tuple[int, float]], final: float) -> None:
        directory = runs / run_id
        directory.mkdir(parents=True, exist_ok=True)
        (directory / "manifest.json").write_text(
            json.dumps({"run_id": run_id, "arm": arm, "topo": "Abilene.gml", "rate": 500, "seed": seed,
                        "episodes": 600, "factors": {}, "source_run_id": None}),
            encoding="utf-8",
        )
        (directory / "eval.json").write_text(
            json.dumps({"curve": [{"episode": e, "r_mean_mean": v} for e, v in curve],
                        "final": {"r_mean_mean": final}}),
            encoding="utf-8",
        )
        (directory / "DONE").write_text("", encoding="utf-8")

    levels = {"A1": 0.60, "A1b": 0.58, "A2": 0.55, "A3": 0.50, "A3b": 0.45, "A4": 0.40, "A5": 0.35, "A6": 0.30}
    for arm, level in levels.items():
        for seed in (0, 1):
            write(f"{arm}-s{seed}", arm, seed, _curve(0.1, level + 0.03 * seed), level + 0.03 * seed)
    for arm, final in (("OSPF", 0.10), ("ECMP", 0.05), ("LB", 0.08), ("RR", 0.07)):
        write(f"{arm}-1", arm, 0, [], final)
    for seed in (0, 1):
        write(f"A0-s{seed}", "A0", seed, [], 0.20 + 0.01 * seed)


@pytest.fixture(name="runs_source")
def _runs_source(tmp_path: Path) -> Path:
    """真冒烟数据优先；缺席 → 合成 runs 目录（同产物契约）。"""
    if SMOKE_RUNS.is_dir():
        return SMOKE_RUNS
    runs = tmp_path / "runs"
    runs.mkdir()
    _synthetic_runs(runs)
    return runs


def test_make_figures_end_to_end_artifacts(runs_source: Path, tmp_path: Path) -> None:
    """Given 冒烟 runs 目录（或合成替身）；When make_figures；Then figures 目录
    出现 2 PNG + 2 PDF + stats.json/csv + figure_manifest.json，图文件 >10KB，
    manifest 记录分母与 prereg 摘要，stats 族非空。"""
    out = tmp_path / "figures"
    manifest = make_figures(runs_source, out, prereg_path=PKG_ROOT / "config" / "metrics_prereg.yaml")
    expected = [
        "adaptation_curves.png", "adaptation_curves.pdf",
        "asymptote_bars.png", "asymptote_bars.pdf",
        "stats.json", "stats.csv", "figure_manifest.json",
    ]
    assert all((out / name).exists() for name in expected)
    for name in ("adaptation_curves.png", "adaptation_curves.pdf", "asymptote_bars.png", "asymptote_bars.pdf"):
        assert (out / name).stat().st_size > 10_000, name
    assert manifest["figures"] == expected[:4]
    assert manifest["denominators"] and manifest["prereg"]["window_episodes"] == 500
    stats = json.loads((out / "stats.json").read_text(encoding="utf-8"))
    assert stats["pairs"] and stats["baseline"] == "A1"
    assert "holm_p" in stats["pairs"][0]
    csv_lines = (out / "stats.csv").read_text(encoding="utf-8").splitlines()
    assert csv_lines[0].startswith("arm_a,arm_b") and len(csv_lines) == 1 + len(stats["pairs"])

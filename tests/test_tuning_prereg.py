"""tuning_prereg.yaml 预注册契约测试(计划 §2 末段/§3.1-§3.4)。

Phase-1 调参预注册是论文第三根预注册支柱:本测试锁定其机器可查结构 —
节完整性、种子流分离、选择集拓扑、ε/σ̄/Score 公式原文、预注册状态 —
防事后择优。改规则须先改 prereg 并升版本,不许让测试迁就执行。

另钉住 metrics_prereg.yaml 的 narrative_extras 注册节(计划 §5 主表余量,
EXPLORATORY)——三项叙事增项须在主表余量数据生成前注册且版本留痕。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
PREREG_PATH = REPO_ROOT / "config" / "tuning_prereg.yaml"
METRICS_PREREG_PATH = REPO_ROOT / "config" / "metrics_prereg.yaml"

REQUIRED_SECTIONS = [
    "meta",
    "red_lines",
    "selection_set",
    "seeds",
    "incumbent",
    "stage_0_anchor",
    "proxy_validity",
    "stage_a",
    "stage_b",
    "adapt_guard",
    "winner_rule",
    "holdout_mechanical_gate",
    "stopping_rule",
    "execution_concurrency",
    "search_space_to_overlay_mapping",
    "deliverables",
]


def _load() -> dict[str, Any]:
    with PREREG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def _load_metrics() -> dict[str, Any]:
    with METRICS_PREREG_PATH.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def test_prereg_parses_with_all_required_sections() -> None:
    """Given 已提交的预注册文件;When safe_load;Then 16 个关键节全部存在。"""
    doc = _load()
    for section in REQUIRED_SECTIONS:
        assert section in doc, f"缺失 prereg 节: {section}"
    for key in (
        "dev_eval_separation",
        "single_global_config",
        "freeze_before_prereg",
        "artifact_isolation",
    ):
        assert key in doc["red_lines"]


def test_prereg_status_marks_registration_before_phase1() -> None:
    """Given meta 节;When 读取;Then 状态为先于 Phase-1 执行的预注册。"""
    meta = _load()["meta"]
    assert meta["status"] == "pre_registered_before_phase1_execution"
    assert meta["date"] == "2026-08-30"
    assert meta["version"] == 5
    change_log = meta["change_log"]
    assert [entry["version"] for entry in change_log] == [5, 4, 3, 2, 1]
    # v5(2026-08-31 hidden 轴 overlay 编码裁决):Block3 overlay 前缀补输入维 287
    assert (
        "裁决" in change_log[0]["change"] and "[287,256,128]" in change_log[0]["change"]
    )
    # v4(2026-08-31 执行期机械勘误):plateau 曲线出处 eval.json + Stage-A 计量 a0_r_mean_mean
    assert (
        "勘误" in change_log[1]["change"]
        and "a0_r_mean_mean" in change_log[1]["change"]
    )
    # v3(2026-08-31 执行前裁决):锚点补齐 3 种子闭合 9 配对点歧义
    assert (
        "裁决" in change_log[2]["change"] and "[100,101,102]" in change_log[2]["change"]
    )
    assert "D4" in change_log[3]["change"] and "[A6]" in change_log[3]["change"]
    assert meta["plan"] == ".omo/plans/experiment-execution-plan.md"


def test_prereg_seed_flows_disjoint() -> None:
    """Given 种子节;When 读取;Then 调参流 {100,101,102} 与试点/主表流分离。"""
    seeds = _load()["seeds"]
    assert seeds["tune"] == [100, 101, 102]
    assert seeds["pilot"] == [0, 1, 2]
    assert seeds["main"] == list(range(10))
    assert set(seeds["tune"]).isdisjoint(seeds["pilot"])
    assert set(seeds["tune"]).isdisjoint(seeds["main"])


def test_prereg_selection_targets_three_topologies_at_500() -> None:
    """Given 选择集;When 读取;Then 恰为 3 个拓扑轴目标 @500 + 2 个 holdout 诊断目标。"""
    sel = _load()["selection_set"]
    assert sel["targets"] == [
        {"topo": "Claranet.gml", "avgrate": 500},
        {"topo": "Gridnet.gml", "avgrate": 500},
        {"topo": "NSF.gml", "avgrate": 500},
    ]
    assert sel["holdout_diagnostics"] == [
        {"topo": "Claranet_failure.gml", "avgrate": 500},
        {"topo": "Claranet.gml", "avgrate": 1000},
    ]
    for row in [*sel["targets"], *sel["holdout_diagnostics"]]:
        assert (REPO_ROOT / "topologies" / row["topo"]).exists()


def test_prereg_epsilon_sigma_bar_score_formulas_verbatim() -> None:
    """Given 公式承载节;When 读取;Then ε/σ̄/Score 定义按计划原文逐字在场。"""
    doc = _load()
    eps = doc["winner_rule"]["epsilon"]
    assert eps["formula"] == "ε = max(0.005, 0.47σ̄)"
    assert eps["floor"] == 0.005
    assert eps["sigma_multiplier"] == 0.47
    assert doc["stage_0_anchor"]["outputs"]["sigma_bar"]["definition"] == (
        "per-target sample SD (ddof=1, n=3) of zeroshot final r_mean across tune seeds, "
        "averaged over the 3 selection targets"
    )
    assert (
        doc["winner_rule"]["score"]["formula"]
        == "Score(C) = mean_{t,s} ZS − 0.5·(mean_t − min_t mean_s ZS)"
    )


def test_prereg_stage_ladder_numbers_match_plan() -> None:
    """Given 阶梯节;When 读取;Then 试验数/种子/回合数/网格与计划数字一致。"""
    doc = _load()
    anchor = doc["stage_0_anchor"]["runs"]
    assert anchor == [
        {"episodes": 2000, "seeds": [100, 101, 102]},
        {"episodes": 8000, "seeds": [100, 101, 102]},
    ]
    stage_a = doc["stage_a"]
    assert (stage_a["trials"], stage_a["seed"], stage_a["episodes"]) == (10, 100, 2000)
    block1 = stage_a["blocks"]["block1"]
    assert block1["lr"] == [3.0e-6, 6.0e-6, 2.0e-5]
    assert block1["opt"] == ["single", "dual"]
    assert block1["trials"] == 6
    assert stage_a["blocks"]["block2"]["ent"] == [0.01, 0.1]
    assert stage_a["blocks"]["block3"]["hidden"] == [[256, 128], [64, 64]]
    # v5 裁决:hidden 轴 overlay 编码 = [287] + 隐层规格(搜索语义不变)
    assert stage_a["blocks"]["block3"]["overlay_encoding"]
    assert "[287, 256, 128]" in stage_a["blocks"]["block3"]["overlay_encoding"]
    assert stage_a["advancement"]["delta_formula"] == "δ = 0.5σ̄"
    # v4 勘误:Stage-A 排名/晋级效应计量钉死 a0_r_mean_mean(Score 保留给 winner_rule)
    assert stage_a["advancement"]["stage_a_metric"]
    assert "a0_r_mean_mean" in stage_a["advancement"]["stage_a_metric"]
    assert len(stage_a["known_limitations"]) == 4
    stage_b = doc["stage_b"]
    assert stage_b["candidates"] == 2
    assert stage_b["episodes"] == 8000
    assert stage_b["seeds"] == [100, 101, 102]
    assert stage_b["zeroshot"]["episodes"] == 10
    assert "复用" in stage_b["incumbent_reuse"]
    pv = doc["proxy_validity"]
    assert pv["paired_points"]["count"] == 9
    assert pv["paired_points"]["threshold"] == 0.6
    # v3 裁决留痕:歧义旗标须有非空 resolution;plateau 聚合口径须有非空钉死文本
    assert pv["paired_points"]["resolution"]
    assert pv["incumbent_plateau"]["plateau_sigma_coefficient"] == 0.1
    # v4 勘误:plateau 曲线出处 = 锚点 pretrain run 的 eval.json curve(非 metrics.csv)
    agg = pv["incumbent_plateau"]["mechanical_aggregation"]
    assert agg and "eval.json" in agg and "metrics.csv" not in agg
    assert pv["conservative_branch"]["min_effect_epsilon_multiplier"] == 2


def test_prereg_adapt_guard_scope() -> None:
    """Given adapt 守卫;When 读取;Then D4 后仅 A6 × winner/incumbent × 3 种子 × 1000ep。"""
    guard = _load()["adapt_guard"]
    assert guard["arms"] == ["A6"]
    assert guard["configs"] == ["winner", "incumbent"]
    assert guard["seeds"] == [100, 101, 102]
    assert guard["adapt_episodes"] == 1000
    assert guard["targets"] == [
        {"topo": "Claranet.gml", "avgrate": 500, "role": "常载"},
        {"topo": "Claranet_failure.gml", "avgrate": 500, "role": "压力轴"},
    ]


def test_prereg_overlay_mapping_and_concurrency_placeholder() -> None:
    """Given 缝映射节;When 读取;Then 轴→overlay 键与缝 1/3 契约一致,并发度留空占位。"""
    doc = _load()
    assert doc["search_space_to_overlay_mapping"]["axes"] == {
        "lr": "ppo.learning_rate",
        "opt": "ppo.lr_critic_ratio",
        "ent": "ppo.ent_coef",
        "hidden": "net.layers",
    }
    assert doc["search_space_to_overlay_mapping"]["tag_convention"] == (
        "<cfg>_a2000 / <cfg>_b8000"
    )
    # 2026-08-30 §2.6 画像定案回填:null → 48(cpu 路线稳态饱和点,证据 profiling/report.md);
    # 纯执行参数,占位契约的预期回填而非规则升版
    assert doc["execution_concurrency"]["value"] == 48
    assert set(doc["deliverables"]) == {"a", "b", "c", "d"}
    assert "trigger" in doc["holdout_mechanical_gate"]
    assert "rule" in doc["stopping_rule"]


def test_metrics_prereg_narrative_extras_registered_and_version_bumped() -> None:
    """Given metrics_prereg.yaml;When safe_load;Then meta.version 升到 4(v4=源域切换
    立项 D6/D7/D8 裁决落档——保护对象改 4 新目标场景、load_sweep 主拓扑随源切
    Claranet;v3=GATE-2 裁决 theta=null 即终局;v2=叙事增项注册留痕;v1=初始)且
    narrative_extras 三项注册齐全:负载扫 rates/主拓扑解释、ospf_agreement 开关
    引用与 determinism 测试、Δ-vs-OSPF 与泛化差距面板定义,统一 EXPLORATORY 标签
    (计划 §5 主表余量);theta 锁死 null(逐场景 theta_rule 推导即终局口径)。"""
    doc = _load_metrics()
    meta = doc["meta"]
    assert meta["version"] == 4
    assert [entry["version"] for entry in meta["change_log"]] == [4, 3, 2, 1]
    assert meta["status"] == "pre_registered_before_main_table_generation"
    # v4 留痕须落源域切换裁决:保护对象重定义 + D6/D7/D8
    assert "源域切换" in meta["change_log"][0]["change"]
    assert "D6" in meta["change_log"][0]["change"]
    assert "Claranet_failure" in meta["change_log"][0]["change"]
    assert doc["theta"] is None  # v3 裁决:null 即终局,此后禁回填全局 float
    # v3 留痕须落 GATE-2 裁决:theta=null 终局 + 种子数缺省
    assert "GATE-2" in meta["change_log"][1]["change"]
    assert "终局" in meta["change_log"][1]["change"]
    assert "n=10" in meta["change_log"][1]["change"]
    # v2 留痕须点名三项增项
    assert "叙事增项" in meta["change_log"][2]["change"]
    assert "EXPLORATORY" in meta["change_log"][2]["change"]

    extras = doc["narrative_extras"]
    assert extras["label"] == "exploratory"
    sweep = extras["load_sweep"]
    assert sweep["rates"] == [250, 500, 1000, 1500, 2000]
    assert sweep["topology"] == "Claranet.gml"  # v4 随源切换更新（v2 注册时为 CERNET.gml）
    assert "v4" in sweep["topology_note"]
    assert sweep["rows"]["arms"] == ["OSPF", "ECMP", "LB", "RR"]

    agreement = extras["ospf_agreement"]
    assert agreement["flag"] == "paper_defaults.eval.ospf_agreement_log"
    assert (
        "ospf_action" in agreement["definition"]
        and "ospf_agreement" in agreement["definition"]
    )
    assert "test_ospf_agreement_bitmatch" in agreement["determinism_proof"]

    assert "r_mean_mean" in extras["delta_vs_ospf"]["definition"]
    assert "OSPF" in extras["delta_vs_ospf"]["panel"]
    assert "A0" in extras["generalization_gap"]["definition"]
    assert "A6" in extras["generalization_gap"]["panel"]

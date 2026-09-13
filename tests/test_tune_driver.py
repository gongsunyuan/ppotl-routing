"""tune 驱动器契约测试（计划 §2 工程缝 4，防混配置表静默腐蚀——§7 单一最大风险）。

覆盖：trial schema 校验与 tag 派生（阶段+集数编码，防 Stage-B 被 Stage-A 的 DONE
静默跳过）、网格合成（overlay paths 强制重定向 runs_tune/<tag>/，隔离红线 4）、
碰撞守卫（同 tag + 不同 overlay/grid/commit/config_hash → 拒绝；全同 → 试验级
幂等跳过；有目录无 meta → 保守拒绝）、dry-run（不 spawn 子进程、不落盘）、
summary.csv 聚合（合成 run 树 → 行内容），以及一条 1 回合真子进程 e2e。
"""

from __future__ import annotations

import csv
import json
import re
import subprocess
import sys
from pathlib import Path
from typing import Any

import pytest
import yaml

import trl_sb3.run.tune as tune_module
from trl_sb3.common.config import load_config, merge
from trl_sb3.common.logging_utils import write_manifest
from trl_sb3.run.sweep import _merged_config, plan_runs, task_run_id
from trl_sb3.run.tune import (
    Trial,
    TrialCollisionError,
    TrialSpecError,
    check_collision,
    load_trials,
    run_tune,
    synthesize_grid,
    trial_meta,
)

SCENARIO = {"topo": "Abilene.gml", "avgrate": 500}


def _base_cfg() -> dict[str, Any]:
    """default.yaml + tmp paths（plan/合成用，绝不触真实 runs/）。"""
    return load_config()


def _trial_dict(**overrides: Any) -> dict[str, Any]:
    trial: dict[str, Any] = {
        "name": "lr2e-5_dual",
        "stage": "a",
        "episodes": 2000,
        "seeds": [100],
        "grid": {"scenarios": [SCENARIO], "seeds": [100], "zeroshot": True},
        "config_overlay": {"ppo": {"learning_rate": 2.0e-5, "lr_critic_ratio": 0.1}},
    }
    trial.update(overrides)
    return trial


def _trial(**overrides: Any) -> Trial:
    return load_trials([_trial_dict(**overrides)])[0]


# ---- trial schema 与 tag 派生 ----


def test_load_trials_accepts_yaml_file_and_inline_list(tmp_path: Path) -> None:
    """Given YAML 文件（trials: 节）与内联 list；When load_trials；Then 等价解析。"""
    path = tmp_path / "trials.yaml"
    path.write_text(yaml.safe_dump({"trials": [_trial_dict()]}), encoding="utf-8")
    from_file = load_trials(path)
    inline = load_trials([_trial_dict()])
    assert from_file == inline
    assert len(from_file) == 1 and from_file[0].name == "lr2e-5_dual"


def test_tag_derived_encodes_stage_and_episodes() -> None:
    """Given stage/episodes；When tag 派生；Then 严格 f"{name}_{stage}{episodes}"
    （a2000/b8000/02000）——Stage-B 绝不与 Stage-A 同 tag，DONE 跳过不可能跨阶段。"""
    assert _trial().tag == "lr2e-5_dual_a2000"
    assert _trial(stage="b", episodes=8000).tag == "lr2e-5_dual_b8000"
    assert _trial(stage=0).tag == "lr2e-5_dual_02000"


def test_manual_tag_key_refused() -> None:
    """Given trial 手写 tag 键；When load_trials；Then 拒绝（tag 必须派生，不收人为值）。"""
    with pytest.raises(TrialSpecError, match="tag"):
        load_trials([_trial_dict(tag="hand_written")])


def test_unknown_trial_key_refused() -> None:
    """Given 未知 trial 键；When load_trials；Then ValueError（严格 schema，防拼写
    静默漂移——如 config_overlay 误写 config_overlays）。"""
    with pytest.raises(TrialSpecError, match="未知键"):
        load_trials([_trial_dict(config_overlays={"ppo": {}})])


@pytest.mark.parametrize("bad_stage", ["c", 1, "A", "0", True])
def test_invalid_stage_refused(bad_stage: Any) -> None:
    """Given stage ∉ {a, b, 0}；When load_trials；Then 拒绝。"""
    with pytest.raises(TrialSpecError, match="stage"):
        load_trials([_trial_dict(stage=bad_stage)])


@pytest.mark.parametrize("bad_name", ["bad name!", "../evil", "-lead", ""])
def test_invalid_name_refused(bad_name: str) -> None:
    """Given name 含路径/空格等不安全字符；When load_trials；Then 拒绝（tag 即目录名）。"""
    with pytest.raises(TrialSpecError, match="name"):
        load_trials([_trial_dict(name=bad_name)])


def test_bad_episodes_or_seeds_refused() -> None:
    """Given episodes≤0 / 空 seeds / 非法 overlay 类型；When load_trials；Then 拒绝。"""
    with pytest.raises(TrialSpecError, match="episodes"):
        load_trials([_trial_dict(episodes=0)])
    with pytest.raises(TrialSpecError, match="seeds"):
        load_trials([_trial_dict(seeds=[])])
    with pytest.raises(TrialSpecError, match="config_overlay"):
        load_trials([_trial_dict(config_overlay=["ppo"])])


def test_duplicate_tags_in_one_file_refused() -> None:
    """Given 两 trial 派生同 tag；When load_trials；Then 拒绝（共享 trial_root = 碰撞）。"""
    with pytest.raises(TrialSpecError, match="重复 tag"):
        load_trials(
            [_trial_dict(), _trial_dict(config_overlay={"ppo": {"ent_coef": 0.01}})]
        )


# ---- 网格合成（隔离红线 4）----


def test_synthesized_grid_redirects_paths_and_forces_isolation(tmp_path: Path) -> None:
    """Given trial + overlay（含恶意 paths 改写企图）；When synthesize_grid；Then
    config_overlay.paths 强制为 trial_root/runs|ckpts（隔离不可协商），trial 级
    seeds/episodes 压过 grid.pretrain，其余 overlay 原样保留。"""
    trial = _trial(
        config_overlay={
            "ppo": {"learning_rate": 2.0e-5, "lr_critic_ratio": 0.1},
            "paths": {"runs_dir": "/evil/runs"},
        }
    )
    root = tmp_path / "runs_tune" / trial.tag
    synth = synthesize_grid(trial, root)
    assert synth["config_overlay"]["paths"] == {
        "runs_dir": str(root / "runs"),
        "ckpts_dir": str(root / "ckpts"),
    }
    assert synth["config_overlay"]["ppo"] == {
        "learning_rate": 2.0e-5,
        "lr_critic_ratio": 0.1,
    }
    assert synth["pretrain"] == {"seeds": [100], "episodes": 2000}
    assert synth["scenarios"] == [SCENARIO] and synth["zeroshot"] is True


def test_synthesis_lands_in_merged_config(tmp_path: Path) -> None:
    """Given 合成网格；When _merged_config（sweep 真实入口）；Then paths 重定向与
    ppo 叠加均到达 merged config——子进程 sweep 将写进 trial_root。"""
    trial = _trial()
    root = tmp_path / "runs_tune" / trial.tag
    cfg = _merged_config(synthesize_grid(trial, root), _base_cfg())
    assert cfg["paths"]["runs_dir"] == str(root / "runs")
    assert cfg["paths"]["ckpts_dir"] == str(root / "ckpts")
    assert (
        cfg["ppo"]["learning_rate"] == 2.0e-5 and cfg["ppo"]["lr_critic_ratio"] == 0.1
    )
    assert cfg["pretrain"]["episodes"] == 2000


# ---- 碰撞守卫 ----


def _write_meta(trial_root: Path, meta: dict[str, Any]) -> None:
    write_manifest(trial_root / "trial_meta.json", meta)


def _current_meta(_tmp_path: Path) -> dict[str, Any]:
    return trial_meta(_trial(), _base_cfg())


def test_guard_refuses_overlay_hash_mismatch(tmp_path: Path) -> None:
    """Given 已存在 trial_meta 且 overlay_hash 不同；When check_collision；Then
    TrialCollisionError 同时点名两个值（防静默腐蚀的核心守卫）。"""
    root = tmp_path / "runs_tune" / _trial().tag
    root.mkdir(parents=True)
    meta = _current_meta(tmp_path)
    meta["overlay_hash"] = "f" * 64
    _write_meta(root, meta)
    current = _current_meta(tmp_path)
    with pytest.raises(
        TrialCollisionError, match=r"overlay_hash.*[0-9a-f]{64}.*[0-9a-f]{64}"
    ):
        check_collision(root, current)


def test_guard_refuses_git_commit_mismatch(tmp_path: Path) -> None:
    """Given 已存在 trial_meta 且 git_commit 不同（代码漂移）；When check_collision；
    Then 拒绝并点名两 commit。"""
    root = tmp_path / "runs_tune" / _trial().tag
    root.mkdir(parents=True)
    meta = _current_meta(tmp_path)
    cur = meta["git_commit"] or "0" * 40
    flipped = (
        "0" if cur[-1] != "0" else "1"
    )  # 末位确定性翻转，保证与原值不同（与 HEAD 实际末位无关）
    meta["git_commit"] = cur[:39] + flipped
    _write_meta(root, meta)
    with pytest.raises(TrialCollisionError, match="git_commit"):
        check_collision(root, _current_meta(tmp_path))


def test_guard_refuses_seed_change_via_grid_hash(tmp_path: Path) -> None:
    """Given 同 tag 同 overlay 但 seeds 变化；When check_collision；Then 拒绝
    （seeds 编码进 grid_hash，静默换种子不可能）。"""
    root = tmp_path / "runs_tune" / _trial().tag
    root.mkdir(parents=True)
    _write_meta(root, trial_meta(_trial(), _base_cfg()))
    varied = trial_meta(_trial(seeds=[101]), _base_cfg())
    with pytest.raises(TrialCollisionError, match="grid_hash"):
        check_collision(root, varied)


def test_guard_refuses_config_protocol_drift(tmp_path: Path) -> None:
    """Given 同 tag 同 overlay 同 commit 但 default.yaml 协议节漂移；When
    check_collision；Then 拒绝（created_config_hash 守卫 config 漂移）。"""
    root = tmp_path / "runs_tune" / _trial().tag
    root.mkdir(parents=True)
    _write_meta(root, trial_meta(_trial(), _base_cfg()))
    drifted_cfg = merge(
        _base_cfg(), {"eval": {"eval_episodes": 5}}
    )  # D1 后缺省 10，漂移到 5
    with pytest.raises(TrialCollisionError, match="created_config_hash"):
        check_collision(root, trial_meta(_trial(), drifted_cfg))


def test_guard_skips_on_identity(tmp_path: Path) -> None:
    """Given 已存在 trial_meta 与当前完全一致；When check_collision；Then "skip"
    （试验级 DONE 幂等语义）。"""
    root = tmp_path / "runs_tune" / _trial().tag
    root.mkdir(parents=True)
    _write_meta(root, trial_meta(_trial(), _base_cfg()))
    assert check_collision(root, trial_meta(_trial(), _base_cfg())) == "skip"


def test_guard_fresh_when_no_dir(tmp_path: Path) -> None:
    """Given trial_root 不存在；When check_collision；Then "fresh"。"""
    assert (
        check_collision(tmp_path / "runs_tune" / _trial().tag, _current_meta(tmp_path))
        == "fresh"
    )


def test_guard_refuses_existing_dir_without_meta(tmp_path: Path) -> None:
    """Given trial_root 存在但无 trial_meta.json（来历不明）；When check_collision；
    Then 保守拒绝（宁可误杀不可静默腐蚀）。"""
    root = tmp_path / "runs_tune" / _trial().tag
    root.mkdir(parents=True)
    with pytest.raises(TrialCollisionError, match="trial_meta"):
        check_collision(root, _current_meta(tmp_path))


def test_guard_refuses_corrupt_meta(tmp_path: Path) -> None:
    """Given trial_meta.json 非法 JSON；When check_collision；Then 保守拒绝。"""
    root = tmp_path / "runs_tune" / _trial().tag
    root.mkdir(parents=True)
    (root / "trial_meta.json").write_text("{not json", encoding="utf-8")
    with pytest.raises(TrialCollisionError):
        check_collision(root, _current_meta(tmp_path))


# ---- 执行 + summary.csv（mock 子进程；合成 run 树）----


def _fake_sweep_side_effect(base: dict[str, Any]) -> Any:
    """模拟子进程 sweep：按命令行里的合成网格 plan_runs 并落 DONE run 树
    （A0 行带 manifest/eval）——全程无 torch 训练。"""

    def _execute(command: list[str]) -> int:
        grid = load_config(command[command.index("--grid") + 1])
        cfg = _merged_config(grid, base)
        for task in plan_runs(grid, base):
            directory = Path(cfg["paths"]["runs_dir"]) / task_run_id(task)
            directory.mkdir(parents=True, exist_ok=True)
            if task.arm == "A0":
                write_manifest(
                    directory / "manifest.json",
                    {"arm": "A0", "topo": task.topo, "rate": task.rate},
                )
                write_manifest(
                    directory / "eval.json",
                    {"curve": [], "final": {"r_mean_mean": 0.625}},
                )
            (directory / "DONE").write_text("", encoding="utf-8")
        return 0

    return _execute


def test_run_tune_executes_summarizes_and_idempotent_skip(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given 新 trial + 模拟子进程（落 2 个 DONE：pretrain + A0）；When run_tune；
    Then 先写 trial_meta、status ok、counts (2,2,0,0)、summary.csv 一行含 dev 指标
    （topo@rate=r_mean）；再次 run_tune → 试验级 skip、无第二次子进程。"""
    calls: list[list[str]] = []
    fake = _fake_sweep_side_effect(_base_cfg())

    def _execute(command: list[str]) -> int:
        calls.append(list(command))
        return fake(command)

    monkeypatch.setattr(tune_module, "_execute_trial", _execute)
    tune_root = tmp_path / "runs_tune"
    results = run_tune([_trial_dict()], device="cpu", tune_root=tune_root)
    assert len(results) == 1 and results[0].status == "ok"
    assert (
        results[0].planned,
        results[0].done,
        results[0].skipped,
        results[0].failed,
    ) == (2, 2, 0, 0)
    assert results[0].dev_rows == (("Abilene.gml", 500, 0.625),)
    trial_root = tune_root / results[0].tag
    assert (trial_root / "trial_meta.json").is_file()
    summary_path = tune_root / "summary.csv"
    assert summary_path.is_file()
    with summary_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert len(rows) == 1
    row = rows[0]
    assert (
        row["tag"] == "lr2e-5_dual_a2000"
        and row["status"] == "ok"
        and row["exit_code"] == "0"
    )
    assert (row["planned"], row["done"], row["skipped"], row["failed"]) == (
        "2",
        "2",
        "0",
        "0",
    )
    assert row["n_a0"] == "1" and row["a0_r_mean_mean"] == "0.625000"
    assert row["dev_detail"] == "Abilene.gml@500=0.625000"
    assert "sweep" in capsys.readouterr().out
    # 幂等重跑：identity → 试验级 skip，不再 spawn 子进程
    rerun = run_tune([_trial_dict()], device="cpu", tune_root=tune_root)
    assert rerun[0].status == "skipped" and len(calls) == 1
    assert not any(part.startswith("--dry") for part in calls[0])


def test_run_tune_records_subprocess_failure(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given 子进程退出码 1（sweep 内部 failed 但 tune 不中断）；When run_tune；
    Then status failed、exit_code 1、counts 全计划未完成。"""
    monkeypatch.setattr(tune_module, "_execute_trial", lambda command: 1)
    results = run_tune([_trial_dict()], device="cpu", tune_root=tmp_path / "runs_tune")
    assert results[0].status == "failed" and results[0].exit_code == 1
    assert (results[0].planned, results[0].done, results[0].failed) == (2, 0, 0)


def test_dry_run_spawns_no_subprocess_and_creates_nothing(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Given dry-run；When run_tune；Then 不 spawn 子进程（spawn 即测试失败）、
    不创建 tune_root 任何目录、不写 summary、stdout 打印计划命令（freeze-gate 式验证）。"""
    monkeypatch.setattr(
        tune_module,
        "_execute_trial",
        lambda command: (_ for _ in ()).throw(
            AssertionError("dry-run 不得 spawn 子进程")
        ),
    )
    tune_root = tmp_path / "runs_tune"
    results = run_tune([_trial_dict()], device="cpu", dry_run=True, tune_root=tune_root)
    assert [r.status for r in results] == ["fresh"]
    assert not tune_root.exists()
    out = capsys.readouterr().out
    assert "sweep" in out and "--device cpu" in out


# ---- e2e：1 回合真子进程（<60s 预算；tmp 隔离的 runs_tune 根）----


def test_end_to_end_one_episode_trial_by_subprocess(tmp_path: Path) -> None:
    """Given 1 回合/1 种子 Abilene trial（YAML 文件）；When run_tune 真子进程；
    Then trial_root/runs 下 pretrain+A0 双 DONE、manifest 溯源三件套在场且
    created_config_hash 与 manifest.config_hash 全等、summary.csv 行含 dev 指标。"""
    trials_yaml = tmp_path / "trials.yaml"
    trials_yaml.write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {
                        "name": "smoke",
                        "stage": "a",
                        "episodes": 1,
                        "seeds": [0],
                        "grid": {
                            "scenarios": [SCENARIO],
                            "seeds": [0],
                            "zeroshot": True,
                        },
                        "config_overlay": {"ppo": {"learning_rate": 1.0e-4}},
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tune_root = tmp_path / "runs_tune"
    results = run_tune(trials_yaml, device="cpu", tune_root=tune_root)
    assert results[0].status == "ok" and results[0].exit_code == 0
    assert (results[0].planned, results[0].done, results[0].failed) == (2, 2, 0)
    trial_root = tune_root / "smoke_a1"
    done_dirs = [d for d in (trial_root / "runs").iterdir() if (d / "DONE").exists()]
    assert len(done_dirs) == 2
    meta = json.loads((trial_root / "trial_meta.json").read_text(encoding="utf-8"))
    for run_path in done_dirs:
        manifest = json.loads((run_path / "manifest.json").read_text(encoding="utf-8"))
        assert re.fullmatch(r"[0-9a-f]{64}", manifest["config_hash"])
        assert manifest["git_commit"] is None or re.fullmatch(
            r"[0-9a-f]{40}", manifest["git_commit"]
        )
        assert manifest["config_hash"] == meta["created_config_hash"]
    assert results[0].dev_rows and results[0].dev_rows[0][:2] == ("Abilene.gml", 500)
    with (tune_root / "summary.csv").open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    assert rows[0]["tag"] == "smoke_a1" and rows[0]["n_a0"] == "1"


def test_tune_cli_dry_run_registered(tmp_path: Path) -> None:
    """Given CLI 注册；When python -m trl_sb3.run tune --dry-run；Then 子命令存在、
    打印计划命令、退出 0、不落任何 tune 产物。"""
    trials_yaml = tmp_path / "trials.yaml"
    trials_yaml.write_text(
        yaml.safe_dump(
            {
                "trials": [
                    {
                        "name": "smoke",
                        "stage": "a",
                        "episodes": 1,
                        "seeds": [0],
                        "grid": {
                            "scenarios": [SCENARIO],
                            "seeds": [0],
                            "zeroshot": True,
                        },
                    }
                ]
            }
        ),
        encoding="utf-8",
    )
    tune_root = tmp_path / "runs_tune"
    proc = subprocess.run(
        [
            sys.executable,
            "-m",
            "trl_sb3.run",
            "tune",
            "--trials",
            str(trials_yaml),
            "--device",
            "cpu",
            "--dry-run",
            "--tune-root",
            str(tune_root),
        ],
        capture_output=True,
        text=True,
        timeout=300,
        check=False,
    )
    assert proc.returncode == 0, proc.stderr
    assert "sweep" in proc.stdout
    assert not tune_root.exists()

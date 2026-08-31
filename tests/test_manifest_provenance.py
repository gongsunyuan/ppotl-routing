"""manifest 溯源字段测试（计划 §2 工程缝 4）：config_hash / git_commit / git_dirty。

config_hash 规范化契约是硬合同（跨机器可复现）：仅取协议承载节
{ppo, net, freeze, eval}，sha256 over canonical JSON（sort_keys、
separators=(",",":")、ensure_ascii=False、utf-8）。预算字段（pretrain.episodes）
不入哈希——回合数是预算不是协议；eval 协议（eval_episodes 等）入哈希。
git 字段经 subprocess 读取，仓库外优雅降级为 (null, null)，绝不因 git 崩掉
一次 run。旧 manifest 无这些字段——消费端必须容忍（additive-only）。
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
from pathlib import Path
from typing import Any

from trl_sb3.common.config import load_config, merge
from trl_sb3.common.run_artifacts import build_manifest, config_hash, git_state


def _cfg(**overlay: Any) -> dict[str, Any]:
    """default.yaml + 测试覆盖（绝不动真实文件）。"""
    return merge(load_config(), overlay)


def _expected_hash(sections: dict[str, Any]) -> str:
    """独立复算文档化规范：canonical JSON → sha256 全 hex。"""
    canonical = json.dumps(sections, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


# ---- config_hash：稳定性与敏感性 ----


def test_config_hash_stable_across_calls() -> None:
    """Given 同一 merged config；When 两次调用；Then 哈希相等且 64-hex。"""
    cfg = _cfg()
    first, second = config_hash(cfg), config_hash(cfg)
    assert first == second
    assert re.fullmatch(r"[0-9a-f]{64}", first)


def test_config_hash_matches_documented_canonicalization() -> None:
    """Given 文档化规范；When 调用；Then 严格等于独立复算——实现未混入额外成分。"""
    cfg = _cfg()
    sections = {key: cfg[key] for key in ("ppo", "net", "freeze", "eval")}
    assert config_hash(cfg) == _expected_hash(sections)


def test_config_hash_sensitive_to_eval_protocol() -> None:
    """Given eval.eval_episodes 10→5（D1 后缺省 10，漂移到任意异值）；When 哈希；Then 必变（eval 协议入哈希）。"""
    assert config_hash(_cfg()) != config_hash(_cfg(eval={"eval_episodes": 5}))


def test_config_hash_sensitive_to_ppo_and_net_and_freeze() -> None:
    """Given 任一协议节键值变化；When 哈希；Then 必变。"""
    base = config_hash(_cfg())
    for overlay in (
        {"ppo": {"learning_rate": 2.0e-5}},
        {"ppo": {"lr_critic_ratio": 0.1}},
        {"net": {"layers": [64, 64]}},
        {"freeze": {"enabled": False}},
    ):
        assert config_hash(_cfg(**overlay)) != base


def test_config_hash_insensitive_to_budget_and_paths() -> None:
    """Given pretrain.episodes / adapt.episodes / paths 变化；When 哈希；Then 不变
    （预算与路径不是协议；run_id 确定性红线不受影响）。"""
    base = config_hash(_cfg())
    assert config_hash(_cfg(pretrain={"episodes": 3})) == base
    assert config_hash(_cfg(adapt={"episodes": 7})) == base
    assert config_hash(_cfg(paths={"runs_dir": "/tmp/elsewhere/runs"})) == base


def test_config_hash_omits_absent_sections() -> None:
    """Given 缺 net 节的部分 config；When 哈希；Then 只哈希在场节（key-set 演化容忍）。"""
    partial = {"ppo": _cfg()["ppo"], "eval": _cfg()["eval"]}
    assert config_hash(partial) == _expected_hash({"ppo": partial["ppo"], "eval": partial["eval"]})


# ---- git_state：仓库内取值 / 仓库外降级 ----


def test_git_state_in_repo_returns_commit_and_bool_dirty() -> None:
    """Given 本仓库根；When git_state；Then commit 40-hex、dirty 为 bool（真值不定）。"""
    commit, dirty = git_state()
    assert commit is not None and re.fullmatch(r"[0-9a-f]{40}", commit)
    assert isinstance(dirty, bool)


def test_git_state_degrades_to_none_outside_repo(tmp_path: Path) -> None:
    """Given 仓库外目录；When git_state；Then (None, None) 且不抛（run 不因 git 崩）。"""
    assert git_state(tmp_path) == (None, None)


def test_git_state_tolerates_missing_git_binary(monkeypatch: Any, tmp_path: Path) -> None:
    """Given git 二进制不可用（OSError）；When git_state；Then (None, None) 不抛。"""

    def _no_git(*_args: Any, **_kwargs: Any) -> Any:
        raise FileNotFoundError("no git binary")

    monkeypatch.setattr(subprocess, "run", _no_git)
    assert git_state(tmp_path) == (None, None)


# ---- build_manifest：新字段落地（additive，旧消费端不受影响） ----


def test_build_manifest_carries_provenance_fields() -> None:
    """Given 完整参数；When build_manifest；Then 含 config_hash（64-hex、与
    config_hash() 一致）、git_commit/git_dirty（本仓库内 40-hex / bool）。"""
    cfg = _cfg()
    manifest = build_manifest(
        run_id="A6-deadbeef00", arm="A6", topo="CERNET.gml", rate=500, seed=3,
        factors={"pbrs": True, "freeze": True}, source_run_id=None, episodes=1000,
        device="cpu", config=cfg,
    )
    assert manifest["config_hash"] == config_hash(cfg)
    assert re.fullmatch(r"[0-9a-f]{64}", manifest["config_hash"])
    assert manifest["git_commit"] is None or re.fullmatch(r"[0-9a-f]{40}", manifest["git_commit"])
    assert manifest["git_dirty"] is None or isinstance(manifest["git_dirty"], bool)


def test_build_manifest_legacy_fields_untouched() -> None:
    """Given 旧字段；When build_manifest；Then 原键原样保留（additive-only，
    runs/ 里 4 条基线 manifest 的消费端契约不破）。"""
    manifest = build_manifest(
        run_id="r", arm="A1", topo="Abilene.gml", rate=500, seed=0,
        factors={}, source_run_id=None, episodes=1, device="cpu", config=_cfg(),
    )
    for key in ("run_id", "arm", "topo", "rate", "seed", "factors", "source_run_id",
                "episodes", "device", "config", "requirements_lock", "git_hash",
                "git_hash_note", "versions", "created_at"):
        assert key in manifest

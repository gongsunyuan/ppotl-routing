"""make_run_id 确定性契约与其余 logging_utils 原语的行为锁定测试。

run_id 确定性是断点续跑、DONE 跳过与结果聚合的根基（根 AGENTS.md 约束 2）。
"""

from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
from typing import Any

import pytest

from trl_sb3.common.logging_utils import (
    MetricsCSVWriter,
    make_run_id,
    run_dir,
    write_manifest,
)


def _expected_hash(factors: dict[str, str]) -> str:
    """独立复算文档化规范：k=v 按 key 字典序、'|' 连接、sha1 取前 10 hex。"""
    canonical = "|".join(f"{k}={v}" for k, v in sorted(factors.items()))
    return hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]


# ---- make_run_id：确定性 ----


def test_make_run_id_same_factors_same_id() -> None:
    """Given 因素完全一致；When 两次调用；Then id 相等。"""
    a = make_run_id(arm="A6", topo="CERNET.gml", rate=500, seed=3, pbrs=True, freeze=True, pretrain="run_ab")
    b = make_run_id(arm="A6", topo="CERNET.gml", rate=500, seed=3, pbrs=True, freeze=True, pretrain="run_ab")
    assert a == b


def test_make_run_id_key_order_independent() -> None:
    """Given 因素相同但 kwargs 顺序不同；When 调用；Then id 相等（字典序归一）。"""
    a = make_run_id(arm="A2", topo="Abilene.gml", rate=500, seed=0)
    b = make_run_id(seed=0, rate=500, topo="Abilene.gml", arm="A2")
    assert a == b


def test_make_run_id_value_change_changes_id() -> None:
    """Given 任一因素值变化；When 调用；Then id 必变。"""
    base: dict[str, Any] = {
        "arm": "A6", "topo": "CERNET.gml", "rate": 500, "seed": 3,
        "pbrs": True, "freeze": True, "pretrain": "run_ab",
    }
    id_base = make_run_id(**base)
    changes: list[tuple[str, Any]] = [
        ("arm", "A5"),
        ("topo", "NFSCNET.gml"),
        ("rate", 1500),
        ("seed", 4),
        ("pbrs", False),
        ("freeze", False),
        ("pretrain", "run_cd"),
    ]
    for key, new_value in changes:
        varied = {**base, key: new_value}
        assert make_run_id(**varied) != id_base


def test_make_run_id_matches_documented_spec_no_timestamp() -> None:
    """Given 文档化规范；When 调用；Then id 严格等于 slug-规范哈希——实现未混入时间/随机成分。"""
    factors = {"arm": "A6", "rate": "500", "seed": "3", "topo": "CERNET.gml"}
    run_id = make_run_id(arm="A6", rate=500, seed=3, topo="CERNET.gml")
    # 与独立复算的规范哈希全等：若实现混入时间戳/随机数，此处必失败。
    assert run_id == f"A6-{_expected_hash(factors)}"
    # 形态约束：slug-[0-9a-f]{10}
    slug, _, hash10 = run_id.rpartition("-")
    assert len(slug) > 0
    assert len(hash10) == 10 and all(c in "0123456789abcdef" for c in hash10)


def test_make_run_id_distinct_across_family() -> None:
    """Given 一族因素组合；When 逐一取 id；Then 两两互异（哈希碰撞检查）。"""
    ids = [
        make_run_id(arm=arm, topo=topo, rate=rate, seed=seed, pbrs=pbrs, freeze=freeze, pretrain=None)
        for arm in ("A1", "A6")
        for topo in ("CERNET.gml", "Abilene.gml")
        for rate in (500, 1500)
        for seed in (0, 9)
        for pbrs in (True, False)
        for freeze in (True, False)
    ]
    assert len(ids) == len(set(ids)) == 64


def test_make_run_id_slug_rules() -> None:
    """Given slug 缺省/显式传入；When 调用；Then 取字典序首因素值 / 用传入 slug。"""
    auto = make_run_id(seed=3, arm="A6")  # 字典序首键 arm
    assert auto.startswith("A6-")
    explicit = make_run_id(slug="pretrain-cernet", arm="A0", topo="CERNET.gml")
    assert explicit.startswith("pretrain-cernet-")


def test_make_run_id_requires_factors() -> None:
    with pytest.raises(ValueError):
        make_run_id()


# ---- run_dir / write_manifest / MetricsCSVWriter ----


def test_run_dir_idempotent(tmp_path: Path) -> None:
    """Given 同一 root+run_id；When 两次创建；Then 幂等且返回同一路径。"""
    first = run_dir(tmp_path, "A6-abc1230123")
    second = run_dir(tmp_path, "A6-abc1230123")
    assert first == second
    assert first.is_dir()


def test_write_manifest_deterministic_bytes(tmp_path: Path) -> None:
    """Given 相同 payload；When 写两个文件；Then 字节级一致且键排序。"""
    payload = {"arm": "A6", "topo": "CERNET.gml", "factors": {"seed": 3, "rate": 500}}
    p1 = tmp_path / "m1" / "manifest.json"
    p2 = tmp_path / "m2" / "manifest.json"
    write_manifest(p1, payload)
    write_manifest(p2, payload)
    assert p1.read_bytes() == p2.read_bytes()
    text = p1.read_text(encoding="utf-8")
    assert json.loads(text) == payload
    assert text.index('"arm"') < text.index('"topo"')  # sort_keys=True
    assert text.index('"rate"') < text.index('"seed"')  # 嵌套同样排序


def test_metrics_csv_writer_header_and_rows(tmp_path: Path) -> None:
    """Given 新建后追加；When 写两行（含一次重开）；Then 表头只写一次、行序保持、可回读。"""
    path = tmp_path / "nested" / "metrics.csv"
    columns = ["arm", "episode", "rd", "rp", "th", "r_mean"]
    with MetricsCSVWriter(path, columns) as w:
        w.write_row({"arm": "A6", "episode": 0, "rd": 0.5, "rp": 0.9, "th": 1.0, "r_mean": 0.42})
    with MetricsCSVWriter(path, columns) as w:  # 追加打开，不得重复表头
        w.write_row({"arm": "A6", "episode": 1, "rd": 0.6, "rp": 0.91, "th": 1.1, "r_mean": 0.43})
    rows = list(csv.DictReader(path.open("r", encoding="utf-8", newline="")))
    assert len(rows) == 2
    assert list(rows[0].keys()) == columns
    assert rows[0]["episode"] == "0"
    assert rows[1]["r_mean"] == "0.43"


def test_metrics_csv_writer_rejects_extra_keys(tmp_path: Path) -> None:
    path = tmp_path / "metrics.csv"
    with MetricsCSVWriter(path, ["arm", "episode"]) as w, pytest.raises(ValueError):
        w.write_row({"arm": "A6", "episode": 0, "unexpected": 1})

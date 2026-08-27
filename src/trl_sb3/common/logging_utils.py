"""确定性 run_id 与产物写入原语（无时间戳、无随机数）。

run_id 确定性是断点续跑、DONE 跳过与结果聚合的根基（根 AGENTS.md 约束 2）。
产物契约（metrics.csv / eval.json / manifest.json / DONE|FAILED）在 M2-3 由 runner 落地，
本模块只提供原子能力。

make_run_id 因素键名规范（后续 runner 一律沿用，勿自造新键；"slug" 保留不作用因素名）：
    arm      消融臂（A1/A1b/A2/A3/A3b/A4/A5/A6/A0）
    topo     拓扑文件名（如 CERNET.gml / CERNET_failure.gml）
    rate     avgrate（统一用 int，如 500/1500；勿混用 500.0）
    seed     配对种子（0..9）
    pbrs     PBRS 开关（bool）
    freeze   冻结开关（bool）
    pretrain 源预训练 run_id（无预训练臂传 None）
"""

from __future__ import annotations

import csv
import hashlib
import json
import re
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

_SLUG_SANITIZE_RE = re.compile(r"[^A-Za-z0-9_.-]+")


def _canon(value: Any) -> str:
    """因素值的规范字符串形式（确定性：bool→true/false，None→none，其余 str()）。"""
    if value is None:
        return "none"
    if isinstance(value, bool):
        return "true" if value else "false"
    return str(value)


def make_run_id(slug: str | None = None, **factors: Any) -> str:
    """构造确定性 run_id：f"{slug}-{sha1[:10]}"，无时间戳成分。

    规范：因素按 key 字典序拼 "k=v"、'|' 连接成规范串，sha1 取前 10 hex。
    同因素同 id；任一因素值变化 id 必变。slug 缺省时取字典序首因素的值（净化为
    文件名安全字符）；显式传入时不参与哈希。
    """
    if not factors:
        raise ValueError("make_run_id requires at least one factor")
    canonical = "|".join(f"{key}={_canon(value)}" for key, value in sorted(factors.items()))
    hash10 = hashlib.sha1(canonical.encode("utf-8")).hexdigest()[:10]
    if slug is None:
        slug = _canon(factors[min(factors)])
    return f"{_SLUG_SANITIZE_RE.sub('_', slug)}-{hash10}"


def run_dir(root: str | Path, run_id: str) -> Path:
    """幂等创建 root/run_id 目录并返回其路径。"""
    directory = Path(root) / run_id
    directory.mkdir(parents=True, exist_ok=True)
    return directory


def write_manifest(path: str | Path, payload: Mapping[str, Any]) -> None:
    """按产物契约写 manifest：utf-8、sort_keys、indent=2（字节级可复现）。"""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(dict(payload), handle, sort_keys=True, ensure_ascii=False, indent=2)


class MetricsCSVWriter:
    """按固定列清单追加写入的 CSV 记录器（utf-8，每行 flush，长表格式）。

    列契约在构造时锁定：新文件先写表头，追加打开不重复表头；多余键报错，
    缺失键写空单元格。用 with 语句或显式 close() 关闭句柄。
    """

    def __init__(self, path: str | Path, columns: Sequence[str]) -> None:
        if not columns:
            raise ValueError("columns must be non-empty")
        self._columns = list(columns)
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._file = self._path.open("a", encoding="utf-8", newline="")
        self._writer = csv.DictWriter(
            self._file, fieldnames=self._columns, restval="", extrasaction="raise"
        )
        if self._path.stat().st_size == 0:
            self._writer.writeheader()
            self._file.flush()

    @property
    def columns(self) -> list[str]:
        return list(self._columns)

    def write_row(self, row: Mapping[str, Any]) -> None:
        """追加一行并立即 flush（崩溃时已写行不丢失）。"""
        self._writer.writerow(dict(row))
        self._file.flush()

    def close(self) -> None:
        self._file.close()

    def __enter__(self) -> MetricsCSVWriter:
        return self

    def __exit__(self, exc_type: Any, exc: Any, traceback: Any) -> None:
        self.close()

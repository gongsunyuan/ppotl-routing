"""配置加载与路径工具：论文超参唯一出处为 config/default.yaml。

路径一律从 `__file__` 上推包根（<repo>/experiments），与 cwd 无关；
论文超参不在代码中硬编码（根 AGENTS.md 约束 1）。
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml

# 本文件位于 experiments/src/trl_sb3/common/ → parents[3] 即 experiments/。
PKG_ROOT: Path = Path(__file__).resolve().parents[3]
DEFAULT_CONFIG_PATH: Path = PKG_ROOT / "config" / "default.yaml"


def load_config(path: str | Path | None = None) -> dict[str, Any]:
    """加载 YAML 配置，返回映射。

    默认读 <experiments>/config/default.yaml（由 __file__ 上推，cwd 无关）。
    若文档根含 `paper_defaults` 键则解包返回该节（调用方直接取 cfg["ppo"] 等）；
    否则原样返回根映射（供覆盖片段 / 网格文件复用同一入口）。
    """
    config_path = Path(path).resolve() if path is not None else DEFAULT_CONFIG_PATH
    with config_path.open("r", encoding="utf-8") as handle:
        doc = yaml.safe_load(handle)
    if not isinstance(doc, dict):
        raise TypeError(f"config root must be a mapping, got {type(doc).__name__}: {config_path}")
    if "paper_defaults" in doc:
        inner = doc["paper_defaults"]
        if not isinstance(inner, dict):
            raise TypeError(f"'paper_defaults' must be a mapping: {config_path}")
        return inner
    return doc


def merge(base: dict[str, Any], override: dict[str, Any]) -> dict[str, Any]:
    """深合并：返回新 dict，override 递归覆盖 base，嵌套 dict 逐层合入。"""
    merged: dict[str, Any] = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def resolve_path(path: str | Path, *, relative_to: Path | None = None) -> Path:
    """解析路径：绝对路径原样返回；相对路径锚定包根（或指定基准目录）。"""
    candidate = Path(path)
    if candidate.is_absolute():
        return candidate
    anchor = relative_to if relative_to is not None else PKG_ROOT
    return anchor / candidate

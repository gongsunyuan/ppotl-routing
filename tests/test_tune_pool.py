"""tune 受控并发池契约测试（计划 §2 缝 6a——§2.6 GPU 利用率画像定案落地）。

覆盖：池有界且真并发、`concurrency=1` = 串行特征化、失败隔离（单试验非零退出
不拖垮兄弟）、summary.csv 行序 = trial 定义序（与完成序无关）、worker 线程配比
注入（决策 c）、concurrency 合法性校验、CLI `--concurrency` 注册，以及一条
3 试验真子进程池 demo（真实 subprocess、隔离 trial 根、summary 正确）。
"""

from __future__ import annotations

import csv
import os
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

import pytest
import yaml

import trl_sb3.run.tune as tune_module
from trl_sb3.run.tune import (
    DEFAULT_CONCURRENCY,
    DEFAULT_WORKER_THREADS,
    TrialSpecError,
    run_tune,
)

SCENARIO = {"topo": "Abilene.gml", "avgrate": 500}


def _trial_dicts(count: int) -> list[dict[str, Any]]:
    """count 个互异 tag 的最小 trial（无臂行 → 聚合便宜,不触真实 runs/）。"""
    return [
        {
            "name": f"pool{i}",
            "stage": "a",
            "episodes": 1,
            "seeds": [0],
            "grid": {"scenarios": [SCENARIO]},
        }
        for i in range(count)
    ]


class _Recorder:
    """假 _execute_trial:记录活跃并发数/执行序,可选 sleep 与定向失败。"""

    def __init__(
        self, *, sleep_s: float = 0.25, fail_tags: set[str] | None = None
    ) -> None:
        self.sleep_s: float = sleep_s
        self.fail_tags: set[str] = fail_tags or set()
        self.lock: threading.Lock = threading.Lock()
        self.active: int = 0
        self.max_active: int = 0
        self.order: list[str] = []  # 完成序(tag)
        self.count: int = 0

    def __call__(self, command: list[str]) -> int:
        tag = Path(command[command.index("--grid") + 1]).stem.removeprefix("grid_")
        with self.lock:
            self.active += 1
            self.max_active = max(self.max_active, self.active)
        time.sleep(self.sleep_s)
        with self.lock:
            self.active -= 1
            self.count += 1
            self.order.append(tag)
        return 1 if tag in self.fail_tags else 0


def test_pool_bounded_and_truly_concurrent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given 6 fresh trial + concurrency=3；When run_tune；Then 池有界（峰值并发恰 3）
    且真并发（墙钟 < 串行下界 6×sleep）。"""
    recorder = _Recorder(sleep_s=0.3)
    monkeypatch.setattr(tune_module, "_execute_trial", recorder)
    wall_start = time.perf_counter()
    results = run_tune(_trial_dicts(6), tune_root=tmp_path / "rt", concurrency=3)
    wall = time.perf_counter() - wall_start
    assert [r.status for r in results] == ["ok"] * 6
    assert recorder.max_active == 3  # 有界:不多不少
    assert recorder.count == 6
    assert wall < 6 * 0.3  # 串行下界 1.8s;3 宽池 ≈0.6s
    assert wall > 0.3  # 防退化成"全部一瞬"(假执行没跑)


def test_concurrency_one_is_serial_characterization(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given concurrency=1；When run_tune；Then 峰值并发恰 1（今日串行行为特征化）,
    结果/summary 语义与串行一致。"""
    recorder = _Recorder(sleep_s=0.05)
    monkeypatch.setattr(tune_module, "_execute_trial", recorder)
    results = run_tune(_trial_dicts(4), tune_root=tmp_path / "rt", concurrency=1)
    assert recorder.max_active == 1
    assert [r.status for r in results] == ["ok"] * 4
    assert recorder.order == [f"pool{i}_a1" for i in range(4)]  # 定义序逐个执行
    summary = list(
        csv.DictReader((tmp_path / "rt" / "summary.csv").open(encoding="utf-8"))
    )
    assert [row["tag"] for row in summary] == [f"pool{i}_a1" for i in range(4)]


def test_pool_failure_isolation(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given 池中 1 个试验退出码 1；When run_tune；Then 其余兄弟全部执行且 ok,
    失败行 status=failed/exit_code=1（子进程崩溃不拖垮池）。"""
    recorder = _Recorder(sleep_s=0.1, fail_tags={"pool2_a1"})
    monkeypatch.setattr(tune_module, "_execute_trial", recorder)
    results = run_tune(_trial_dicts(4), tune_root=tmp_path / "rt", concurrency=4)
    assert recorder.count == 4  # 兄弟无一被拖垮
    assert [(r.tag, r.status, r.exit_code) for r in results] == [
        ("pool0_a1", "ok", 0),
        ("pool1_a1", "ok", 0),
        ("pool2_a1", "failed", 1),
        ("pool3_a1", "ok", 0),
    ]


def test_summary_rows_in_definition_order_despite_completion_order(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given 首试验最慢(完成序 ≠ 定义序)；When run_tune；Then summary.csv 行序与
    results 序都恒为 trial 定义序（池只影响执行,不影响行序语义）。"""
    delays = {"pool0_a1": 0.4, "pool1_a1": 0.02, "pool2_a1": 0.02}
    lock = threading.Lock()
    completion: list[str] = []

    def fake(command: list[str]) -> int:
        tag = Path(command[command.index("--grid") + 1]).stem.removeprefix("grid_")
        time.sleep(delays[tag])
        with lock:
            completion.append(tag)
        return 0

    monkeypatch.setattr(tune_module, "_execute_trial", fake)
    results = run_tune(_trial_dicts(3), tune_root=tmp_path / "rt", concurrency=3)
    assert completion == ["pool1_a1", "pool2_a1", "pool0_a1"]  # 完成序确实乱序
    assert [r.tag for r in results] == [f"pool{i}_a1" for i in range(3)]
    summary = list(
        csv.DictReader((tmp_path / "rt" / "summary.csv").open(encoding="utf-8"))
    )
    assert [row["tag"] for row in summary] == [f"pool{i}_a1" for i in range(3)]


def test_worker_env_forces_thread_allocation(monkeypatch: pytest.MonkeyPatch) -> None:
    """Given 决策 c 配比；When _worker_env/_execute_trial；Then 子进程环境强制
    OMP/MKL = 实测定值（继承其余变量），并真实传给 subprocess.run。"""
    env = tune_module._worker_env()
    assert env["OMP_NUM_THREADS"] == str(DEFAULT_WORKER_THREADS)
    assert env["MKL_NUM_THREADS"] == str(DEFAULT_WORKER_THREADS)
    marker = f"PPOTL_MARKER_{os.getpid()}"
    monkeypatch.setenv(marker, "1")
    assert tune_module._worker_env()[marker] == "1"  # 其余环境变量继承
    seen: dict[str, Any] = {}

    def fake_run(command: Any, **kwargs: Any) -> Any:
        seen.update(kwargs)
        seen["command"] = list(command)
        return subprocess.CompletedProcess(command, 7)

    # 直接 patch 全局 subprocess 模块（tune_module.subprocess 即同一对象；绕道
    # 他模块属性取 re-import 名字会被 reportPrivateLocalImportUsage 拦下）。
    monkeypatch.setattr(subprocess, "run", fake_run)
    assert tune_module._execute_trial(["echo", "hi"]) == 7
    assert seen["env"]["OMP_NUM_THREADS"] == str(DEFAULT_WORKER_THREADS)
    assert seen["check"] is False


def test_concurrency_validated_before_any_execution(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Given concurrency<1；When run_tune；Then 立即 ValueError,零子进程。"""

    def boom(_command: list[str]) -> int:
        raise AssertionError("校验失败前不得执行任何子进程")

    monkeypatch.setattr(tune_module, "_execute_trial", boom)
    with pytest.raises(ValueError, match="concurrency"):
        run_tune(_trial_dicts(2), tune_root=tmp_path / "rt", concurrency=0)


def test_defaults_match_measured_saturation() -> None:
    """Given §2.6 画像定案；When 导入缺省；Then 并发缺省为实测饱和点(正整数)。"""
    assert isinstance(DEFAULT_CONCURRENCY, int) and DEFAULT_CONCURRENCY >= 1
    assert isinstance(DEFAULT_WORKER_THREADS, int) and DEFAULT_WORKER_THREADS >= 1


def test_tune_cli_concurrency_flag_registered(tmp_path: Path) -> None:
    """Given CLI；When tune --dry-run --concurrency 2；Then 退出 0 且打印
    concurrency=2（缺省不传时打印实测饱和点）。"""
    trials_yaml = tmp_path / "trials.yaml"
    trials_yaml.write_text(
        yaml.safe_dump({"trials": [_trial_dicts(1)[0]]}), encoding="utf-8"
    )
    for extra, expected in (
        (["--concurrency", "2"], "concurrency=2"),
        ([], f"concurrency={DEFAULT_CONCURRENCY}"),
    ):
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
                str(tmp_path / "rt"),
                *extra,
            ],
            capture_output=True,
            text=True,
            timeout=300,
            check=False,
        )
        assert proc.returncode == 0, proc.stderr
        assert expected in proc.stdout
        assert not (tmp_path / "rt").exists()


# —— 真 demo:3 个 1 回合试验 × 真子进程 × 池宽 3(隔离 trial 根,summary 正确) ——


def test_real_subprocess_pool_demo(tmp_path: Path) -> None:
    """Given 3 个 1 回合 trial(真 sweep 子进程) + concurrency=3；When run_tune；
    Then 3 试验全 ok、各 trial 根隔离落 DONE pretrain run、trial_meta 齐备、
    summary.csv 3 行定义序;墙钟呈重叠(3 并发 < 0.9×3 串行)。"""
    trials = [
        {
            "name": f"demo{i}",
            "stage": "a",
            "episodes": 1,
            "seeds": [0],
            "grid": {"scenarios": [SCENARIO]},
        }
        for i in range(3)
    ]
    tune_root = tmp_path / "runs_tune"
    wall_start = time.perf_counter()
    results = run_tune(trials, tune_root=tune_root, concurrency=3)
    wall = time.perf_counter() - wall_start
    assert [r.status for r in results] == ["ok"] * 3
    assert all(r.exit_code == 0 for r in results)
    tags = [f"demo{i}_a1" for i in range(3)]
    for tag in tags:  # 隔离根:每 trial 自己的 runs/ckpts + 指纹
        trial_root = tune_root / tag
        assert (trial_root / "trial_meta.json").is_file()
        done_runs = [
            d for d in (trial_root / "runs").iterdir() if (d / "DONE").exists()
        ]
        assert len(done_runs) == 1 and done_runs[0].name.startswith("pretrain-")
    summary = list(csv.DictReader((tune_root / "summary.csv").open(encoding="utf-8")))
    assert [row["tag"] for row in summary] == tags
    assert [row["status"] for row in summary] == ["ok"] * 3
    # 重叠证据:3 真子进程并发墙钟须明显小于 3 倍单试验串行墙钟(参照 run 计时)
    assert wall < 3 * 0.9 * _single_trial_wall(tmp_path)


def _single_trial_wall(tmp_path: Path) -> float:
    """同规格单试验 1 个的串行墙钟(计时参照;独立根防 DONE 跳过)。"""
    wall_start = time.perf_counter()
    result = run_tune(
        [
            {
                "name": "wallref",
                "stage": "a",
                "episodes": 1,
                "seeds": [0],
                "grid": {"scenarios": [SCENARIO]},
            }
        ],
        tune_root=tmp_path / "wall_ref",
    )
    assert result[0].status == "ok"
    return time.perf_counter() - wall_start


def test_trial_spec_error_still_importable() -> None:
    """Given 模块导出面;When import;Then 契约异常类型仍在(回归钉)。"""
    assert issubclass(TrialSpecError, ValueError)

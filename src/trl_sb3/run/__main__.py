"""`python -m trl_sb3.run` CLI 入口（M2-4）：sweep 子命令分派。

spawn 守卫：`if __name__ == "__main__"` 仅此一处；sweep 内顺序执行、无
multiprocessing/subprocess（Windows spawn 安全）。
"""

from __future__ import annotations

import argparse

from trl_sb3.run.sweep import run_sweep


def _main() -> int:
    parser = argparse.ArgumentParser(prog="python -m trl_sb3.run", description="TRL routing ablation 实验编排")
    subparsers = parser.add_subparsers(dest="command", required=True)
    sweep_parser = subparsers.add_parser("sweep", help="按网格编排：预训练 → 消融臂/评估行")
    sweep_parser.add_argument("--grid", default="config/grid_main.yaml", help="网格 YAML 路径（相对路径锚定 experiments/ 根）")
    sweep_parser.add_argument("--dry-run", action="store_true", help="只打印计划 run 清单与状态，不执行")
    sweep_parser.add_argument("--filter", default=None, help="子串过滤：命中 arm 或 topo 名的任务才执行")
    sweep_parser.add_argument("--no-resume", action="store_true", help="不做 sweep 级 DONE 跳过（runner 内部仍幂等跳过）")
    sweep_parser.add_argument("--device", default="cpu", help="训练设备（cpu/cuda）")
    sweep_parser.add_argument("--strict", action="store_true", help="lineage 缺源时抛错终止（默认记 SKIP-LINEAGE 继续）")
    figures_parser = subparsers.add_parser(
        "make_figures", help="runs 目录 → 论文图（适应曲线/渐近柱）+ 统计（配对 t/Wilcoxon + Holm）"
    )
    figures_parser.add_argument("--runs", default="runs", help="runs 目录（相对路径锚定 experiments/ 根）")
    figures_parser.add_argument("--out", default="figures", help="输出目录（PNG 150dpi + PDF + stats + manifest）")
    figures_parser.add_argument("--prereg", default="config/metrics_prereg.yaml", help="指标预注册 YAML 路径")
    args = parser.parse_args()
    if args.command == "sweep":
        run_sweep(
            args.grid,
            dry_run=args.dry_run,
            resume=not args.no_resume,
            filters=args.filter,
            device=args.device,
            strict=args.strict,
        )
    elif args.command == "make_figures":
        from trl_sb3.run.make_figures import make_figures  # 惰性导入：sweep 启动不载 matplotlib

        make_figures(args.runs, args.out, args.prereg)
    return 0


if __name__ == "__main__":
    raise SystemExit(_main())

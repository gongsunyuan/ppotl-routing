# AGENTS.md

PPO-TL 路由实验代码仓库（CPU 逐-env 后端 + GPU 向量化后端）。
对应论文：Efficient Transfer Learning-Enhanced PPO for Dynamic Network Routing。

## 运行（必须用 venv 的 python）

全局 python 是 CPU-only torch；GPU 只在 `.venv`（torch 2.13.0+cu126）中可用：

```powershell
$py = .venv\Scripts\python.exe
& $py -m pytest tests -q                                  # 全部 32 个测试（CPU env 测试 + GPU 测试）
& $py -m pytest tests/test_gpu.py::test_mm1k_torch_matches_numpy -q   # 单个测试
& $py run\run_gpu.py --mode smoke --force                 # GPU 后端冒烟（首选入口）
& $py run\run_gpu.py --mode full                          # 训练机全量（自动检测卡、断点续跑）
& $py run\run_all.py --mode smoke                         # CPU 逐-env 后端（旧路径，仍可用）
& $py run\make_figures.py --mode smoke                    # 从 logs/{mode} 聚合出图+统计
& $py run\run_gpu.py --mode smoke --dry-run               # 只打印任务规划
```

有用 flags：`--scenarios S1,S2,S3,S4,A,H`、`--seeds 0,1`、`--topo cernet`、
`--gpus 0,1 --procs-per-gpu 3`、`--no-ensemble`、`--filter <run_id 子串>`。

环境搭建：`python -m venv .venv` + 安装 CUDA torch（见 README.md）；
CPU-only torch 也可跑（run_gpu.py 自动回退到向量化 CPU 后端）。

## 修改代码的关键约束

1. **双后端必须同步改**：CPU 逐-env（`src/train.py` + `src/env/network_env.py` + `src/agents/ppo.py|dqn.py|grlps.py`）
   与 GPU 向量化（`src/train_gpu.py` + `src/env/vec_env.py` + `src/agents/*_gpu.py|batched_nets.py`）。
   改环境语义、方法行为、超参时两边都要动，否则结果不可比。
2. **run_id 必须保持确定性**（`src/logging_utils.py: make_run_id`，无时间戳）——
   断点续跑、DONE 跳过、ensemble 分组都依赖它。
3. **新增对比方法需改 5 处**：`src/jobs.py`（ALL_METHODS/PPO_METHODS/method_flags/pretrain_key）、
   两个 trainer 的 agent 构造分支、`run/make_figures.py`（METHOD_LABELS/COLORS/METHOD_ORDER）。
4. **论文超参只在 `config/default.yaml: paper_defaults`**，代码不得硬编码。
5. **测试守护语义不变量，改 env/agents 后必须跑 pytest**：
   M/M/1/K torch↔numpy 数值等价、PBRS 伸缩恒等式、冻结层零梯度、ensemble 种子梯度解耦。
6. 产物契约：每个 run 目录写 `metrics.csv`/`eval.json`/`manifest.json`/`DONE|FAILED`，
   `make_figures.py` 按此聚合——两后端产物格式必须一致。
7. 预训练 ckpt 跨任务共享，并发写入有 `.lock` 目录锁（`run/run_gpu.py: ensure_pretrain`），勿绕过。

## 已知坑

- 归一化常数 `d_max/lam_max`：无预训练的方法需 warm-up 估计（`train*.py: warmup_norms`），
  忘开 `train_norm` 会得到 1e8 级指标（修过两次的 bug）。
- GPU ensemble：S=1 与 S>1 走不同 act 路径（`act` vs `act_ensemble`，后者返回 tuple）。
- 与论文的偏差（解析式排队、GRL-PS 简化复现等）记录在 `README.md`，改语义须同步更新。
- Windows PowerShell：bash 工具不支持 heredoc，改文件用 Edit 工具；控制台中文乱码是显示问题。
- GRL-PS 谱嵌入在小拓扑（节点数 ≤ dim+1）走稠密 eigh 路径，勿改回稀疏 eigsh。

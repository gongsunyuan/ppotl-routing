# RUNBOOK — 好机器全量执行手册（TRL routing ablation，M3 试点补全 → M4 主表）

> 本机（笔记本）只做代码 + 冒烟验证出图；全量试点与主表在更强机器上按本手册执行。
> 工作目录一律 `experiments/`，Python 一律 `.venv\Scripts\python.exe`（Windows PowerShell）。
> 唯一事实来源：`.omo/plans/trl-routing-sb3-ablation.md`；已知坑见根 `AGENTS.md` 与 notepad `issues.md`。

## 1. 环境搭建

1. 整目录复制 `experiments/` 到目标机器（含 `src/ tests/ config/ topologies/ requirements.lock`；不需要 `runs_smoke/ ckpts_smoke/ .venv/`）。
2. 装 [uv](https://docs.astral.sh/uv/)（0.12.5+），然后**严格按 `requirements.lock` 头注释的序列**安装（torch 的 cu126 wheel 不在 PyPI，必须带 index-url）：

   ```powershell
   cd <experiments 所在目录>\experiments
   uv venv --python 3.13 .venv     # 必须显式 3.13：缺省会拉 3.14，cu126 wheel 未必存在
   uv pip install "torch==2.13.0+cu126" --index-url https://download.pytorch.org/whl/cu126 --python .venv\Scripts\python.exe
   uv pip install "stable-baselines3==2.7.0" "gymnasium==1.2.0" "networkx==3.4.2" `
     "pyyaml==6.0.2" "numpy==2.2.6" "pytest==8.3.5" "matplotlib==3.10.9" "scipy==1.15.3" `
     --python .venv\Scripts\python.exe
   uv pip check --python .venv\Scripts\python.exe
   uv pip install -e . --no-deps --python .venv\Scripts\python.exe   # editable 装本包，绝不解析依赖
   ```

3. 验证：`& .venv\Scripts\python.exe -m pytest tests -q` 应 **151+ 全绿**（含 M4 新增 make_figures 测试）。任何红 → 停下排查环境，勿继续。

## 2. 前置检查

- **关睡眠**（管理员 PowerShell，长跑必须）：`powercfg /change standby-timeout-ac 0`
- **GPU 验证**：`& .venv\Scripts\python.exe -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"` → `True` + 显卡名。
- 确认磁盘余量：`runs/`（metrics/eval/manifest 文本，量级 MB×374）+ `ckpts/`（每训练 run 一个 policy zip，量级数十 MB×330）。

## 3. 全量试点（M3 补全，`config/grid_pilot.yaml` 已定案预算）

```powershell
& .venv\Scripts\python.exe -m trl_sb3.run sweep --grid config/grid_pilot.yaml --device cuda --dry-run   # 先核对 34 任务规划
& .venv\Scripts\python.exe -m trl_sb3.run sweep --grid config/grid_pilot.yaml --device cuda
```

- 预算（已定案，写死在 grid_pilot.yaml）：预训练 **8000 ep × 3 种子**、适应 **1000 ep**、A1b **9000 ep × 3 种子**；lineage 守卫会**自动先跑预训练**再放行 A2/A3/A5/A6/A0，无需手工排序。
- 计时参考（RTX 4080 Laptop 实测口径）：预训练 ≈**1.84 h/seed**、A1b ≈**0.81 h/seed**、适应臂 ≈0.09 h/run → 试点单 seed 全臂 ≈3.3 h，**3 seed ≈10 h 量级**（顺序执行；更强 GPU 按比例缩短）。
- 中断可**裸重跑同一条命令续**（DONE 跳过幂等；runner 已修中断 append 污染）。
- 跑完出试点图 + 功效材料：

  ```powershell
  & .venv\Scripts\python.exe -m trl_sb3.run make_figures --runs runs --out figures_pilot
  ```

## 4. 功效校验（定终版种子数 + 回填 θ）

1. 读 `figures_pilot/stats.csv`：每对的逐种子配对差在 `stats.json` 的 `pairs[].diffs`，取目标对比（如 A2 vs A1）的差值 SD 记 σ。
2. 期望检出效应 δ 下所需配对种子数：**n = (1.96+0.84)²·σ²/δ² = 7.84·σ²/δ²**（α=0.05 单侧 80% 功效近似；配对设计，σ 为配对差 SD）。反向读法：n=3 时可检出 δ ≈ 2.8·σ/√3 ≈ 1.62σ。据此定主表终版种子数（计划缺省 10，若 10 仍不足 → 回 orchestrator 重新裁决效应量口径，勿静默加种子）。
3. **回填 θ**：`config/metrics_prereg.yaml` 的 `theta: null` 语义 = 尚未预注册、聚合按 `theta_rule` 推导口径跑。试点后把定案 float 写入 `theta:`（此后**不得再改**——防 p-hacking 是预注册的全部意义）。终版种子数与 θ 定案同步记入 notepad decisions.md。

## 5. 主表（M4 全量，`config/grid_main.yaml`）

```powershell
& .venv\Scripts\python.exe -m trl_sb3.run sweep --grid config/grid_main.yaml --device cuda --dry-run  # 先核对：374 任务
& .venv\Scripts\python.exe -m trl_sb3.run sweep --grid config/grid_main.yaml --device cuda
& .venv\Scripts\python.exe -m trl_sb3.run make_figures --runs runs --out figures_main
```

- 374 任务 = 10 预训练 + 8 臂 × 4 场景 × 10 种子 + 40 A0 + 4 OSPF（+ 启发式行按 grid 键）。
- **墙钟量级估算（顺序执行、CUDA）**：预训练 10×1.84h ≈ 18h；适应 7 臂 × 4 场景 × 10 seed（CERNET 系场景约为 Abilene 计时的 ~3 倍）≈ 45–50 h；A1b 9000 ep ≈ 40–46 h；评估行分钟级 → **总计 ≈ 105–120 h（约 4–5 天连续）**。**以试点实测计时为准**（第 3 节命令的 per-arm 实测直接外推）。可 `--filter` 按场景切多进程并行（sweep 自身单进程；并行时各进程用不同 `--filter` 子串，run 目录幂等不冲突）。
- 产物唯一位置：`runs/`（每 run 目录 metrics.csv/eval.json/manifest.json/DONE|FAILED）与 `ckpts/`（run_id.zip）。聚合端（collect_runs）只认 DONE。

## 6. 注意事项 / 裁决悬项

- **幂等续跑**：run 目录含 DONE 即跳过；中断直接重跑同命令。FAILED 目录会重试（is_done 只认 DONE）。
- **双 lr 裁决材料** = pretrain run 的 `metrics.csv` 损失曲线（直读，不走指标表）。若 A6 试点曲线明显差，按计划 Q4 补双 lr 子类（改 grid 加臂行，不动 runner）。
- **评估回合数统一**：悬而未决（issues.md M2.5 条；`metrics_prereg.yaml` 的 `final_eval_episodes: null`）。定案后改该字段并同步基线行 eval episodes，再跑主表。
- 控制台中文乱码是显示问题，中文路径可用；删大目录给足 timeout，失败 fallback `cmd /c "rd /s /q <path>"`。
- 图内文字全英文（无中文字体依赖）；场景标签 NFSCNET.gml 显示为论文拼写 **NSFCNET**（同一网络）。

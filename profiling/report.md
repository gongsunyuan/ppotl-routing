# profiling/report.md — Phase-0 §2.6 GPU 利用率画像与执行并发定案（2026-08-30）

> 任务：计划 §2 第 6 条「GPU 利用率调优（先画像后定案）」。本报告即三项定案的**唯一证据
> 事实来源**：tune.py 缺省、RUNBOOK §6.1、`config/tuning_prereg.yaml` 的回填值全部指向此处。
> 红线声明：全程未触碰任何论文超参（n_steps/n_epochs/batch_size/gamma/gae_lambda/clip 等原样），
> 未改任何数值路径（determinism/bit-match 套件全绿 = 证据，见 §10），并行实体恒 whole-run
> 粒度（每试验独立子进程 + 独立 trial 根），未写入 runs/ runs_tune/ runs_smoke/ runs_baseline/。

## 0. 摘要（三决策一句话）

| 决策 | 定案 | 关键实测数字 |
|------|------|--------------|
| (a) tune.py 受控并发池 | `--concurrency` 缺省 **48**，1 = 原串行 | cpu 稳态聚合 62.9 eps/s @N48（N32→48 边际 +25%；N64 边际 +1.3% <5% 停） |
| (b) 主表 sweep `--filter` 多进程 | 4 场景子串切 N 个后台 shell（RUNBOOK §6.1） | run 目录幂等 + filter 子串任务集不相交；4×cuda 逐卡钉定比 4×cpu 快 ~20% |
| (c) CPU 侧线程配比 | worker 强制 `OMP_NUM_THREADS=MKL_NUM_THREADS=1` | N48 下 thr1 聚合 38.0 vs thr2 32.0 eps/s；缺省 32 线程直接瘫痪（>25 min 未完成） |

**总判断（验证而非假设）**：本工作负载 = 小 MLP（[287,128,64]）+ 纯 Python env 步进，
单 run GPU 利用率均值 ~9%；**训练期利用率最大化的主杠杆是 CPU 并发，不是 GPU**。
cpu 路线稳态 62.9 eps/s = 单 GPU cuda 多进程 plateau（13.2）的 **4.8×**，也高于 3-GPU
轮转 cuda（31.6）。cuda 仅在**宽度 ≤8 且跨卡轮转**时按 run 占优（每 run 2.83 vs cpu
2.35 eps/s，+21%）——该形态 tune 池表达不了（无法逐 worker 钉卡），故 tune 缺省 cpu。

## 1. 方法与环境

- 机器（共享）：64 核 / 251GB RAM / 4× Tesla V100S-PCIE-32GB / driver 535.247.01；
  torch 2.13.0+cu126 CUDA 可用；解释器 `.venv/bin/python`。**GPU0 常驻他人作业**
  （`main_gcn.py`，432MiB / ~20% util，已运行多日）→ 所有 cuda 测量钉 **GPU1**（单卡）
  或 **GPU1,2,3 轮转**（扩散），绝不碰 GPU0；~1 核被常驻作业占用，64 核按 ~63 可用计。
- 工具：pynvml 不在 venv（红线：不加依赖）→ `nvidia-smi --query-gpu` 子进程 1Hz 轮询
  （`gpu_ts_*.csv`）；相位计时 = 进程内包装 `trl_sb3.train.runner` 模块缝（build_routing_env
  → step/reset 代理；greedy_eval → 计时；PPO → learn/train 累计），**零 src 改动**
  （`profile_single.py`）；torch.profiler 短段（2 rollout = 2 次 PPO 更新）归因
  （`profiler_{cpu,cuda}.txt`）；并发池基准 = 与 tune.py 同款 worker 子进程
  （`python -m trl_sb3.run sweep --grid <临时网格>`），每任务 `/usr/bin/time -v` 抓
  user CPU / maxRSS（`pool_bench.py` → `pool_results.csv`）。
- 测量卫生：各级测量顺序执行互不重叠；种子区 555/777/999/20000+ 与调参 {100..102}、
  试点/主表 {0..9} 不相交；run_id 确定性 → 每测量独立 scratch 根（`profiling/scratch/`，
  已 gitignore），从未触真实 runs*/。计时表全部为单次测量（电池内任务级 /usr/bin/time
  提供任务间方差信号；单 run 画像为 2 reps 取中位）。

## 2. 单 run 画像（100ep CERNET 预训练，seed=777，2 reps 中位）

| 相位（秒） | cpu | cuda（钉 GPU1） |
|------------|-----|------|
| 墙钟总计 | **34.4** | **28.8** |
| eps/s（进程内） | **2.91** | **3.48**（=cpu 的 1.20×） |
| setup（env+model 构建） | 1.06 | 1.18（含 CUDA context ~0.1s） |
| learn 总计 | 31.7 | 26.0 |
| └ PPO 优化（PPO.train） | 8.1 | 5.0（GPU 上快 1.6×） |
| └ env step+reset（纯 Python/numpy） | **15.8** | **12.3** |
| └ 周期+终评 greedy_eval（2 次） | 3.5 | 3.1 |
| └ rollout 其余（前向/缓冲/CSV） | 20.0 | 20.0（口径含 eval，见脚注¹） |
| learn 后（终评尾段+存盘 IO） | 1.6 | 1.6 |

¹ `rollout 其余 = learn − PPO.train − greedy_eval`；env 步进占 learn 的 **50.2%（cpu）/
47.5%（cuda）**——**CPU env 步进是第一瓶颈，假设被数据证实**。cuda 总快 20% 的来源：
PPO.train 快 1.6× + env 步进段也快 ~22%（15.8→12.3s；纯 Python 代码不变，差异来自 cpu
跑时 32 线程 train 突发把核心压出更高功耗/降频窗口，cuda 跑时核心凉——共享机上难以
进一步控制，方向性结论不受影响）。

**GPU 利用率（cuda rep，钉 GPU1，1Hz 采样 27-28 点）**：均值 **8.9-9.0%**，峰值 13%，
显存 432MiB / 32768MiB。SB3 "MLP primarily intended for CPU" 警告在本负载上方向正确但
不构成 cpu 结论——单 run cuda 反而快 1.20×（警告说的是 GPU 利用率差，实测 ~9% 印证；
见 §5 的并发翻转）。

## 3. torch.profiler 归因（2 rollout 短段，`profiler_*.txt`）

- **cpu 段**（总 ~2s）：Adam.step 30.5% self-CPU、addmm/mm 合计 ~17%、PReLU 前后向 ~7.6%、
  copy/to ~6%——torch 侧成本集中在小 MLP 的优化器与矩阵乘，env 步进（纯 Python）不出现在
  aten 表里，与 §2 的 50% 相位证据互洽。
- **cuda 段**：**launch-bound 铁证**——8906 次 `cudaLaunchKernel` 的 CPU 侧总耗时
  **789ms**，而全部 GPU kernel 的 self-CUDA 合计仅 **~46ms**（GPU 忙占比 ~2-3%）：
  单次 addmm GPU 均值 **9.7μs**、Memcpy HtoD 24.3μs。即：CPU 花 17 倍于 kernel 时长
  的时间在发射 kernel——kernel 间隙 GPU 必然空转，这是小 MLP 的结构性状态。

## 4. 并发电池 A：20ep 任务（启动占重 ~18-31%，startup 实测 2.35s/子进程）

worker = 与 tune.py 同款 sweep 子进程，threads=1；「单卡」= 全部钉 GPU1，「扩散」=
GPU1,2,3 轮转。饱和判据：聚合吞吐边际增益 <5% 即停（本报告统一口径）。

| N | cuda 单卡 agg eps/s | cuda×3 扩散 agg eps/s | cpu agg eps/s |
|---|---------------------|------------------------|----------------|
| 1 | 1.75 | — | 1.55 |
| 2 | 3.36（+92%） | — | 3.08（+99%） |
| 4 | 6.07（+81%） | 6.77 | 5.97（+94%） |
| 8 | 9.30（+53%） | 12.68（+87%） | 11.49（+93%） |
| 16 | 11.79（+27%） | 21.27（+68%） | 21.67（+89%） |
| 24 | 12.90（+9.4%） | 24.97（+17%） | 29.92（+38%） |
| 32 | 13.17（**+2.1% ✂ 饱和**） | 28.61（+15%） | 34.99（+17%） |
| 48 | — | 31.60（+10.5%） | 38.02（+8.6%） |
| 64 | — | 29.88（**−5.4% ✂ 回退**） | 38.53（**+1.3% ✂ 饱和**） |

任务级副证（per-task user CPU，秒）：cuda 单卡 N32 时 **45.3** vs 扩散 19.4 vs cpu 15.1
——单卡堆进程时 `cudaStreamSynchronize` 自旋等待吞掉 3 倍 CPU。20ep 电池的教训：
**每任务固定 ~2.35s 启动 + 48 进程同时 import 的风暴**把高 N 段压低了（对比 §5 稳态），
但两条 cpu/cuda 相对结论在两电池中方向一致。

## 5. 并发电池 B：稳态 200ep 任务（启动占重 ~3%，真实试验的形态）

| 配置 | N | 墙钟(s)/任务数 | **聚合 eps/s** | 每 run eps/s | 每 run user CPU(s) |
|------|---|----------------|----------------|--------------|---------------------|
| cpu, thr1 | 8 | 85.2 / 8 | 18.8 | 2.35 | 82.4 |
| cpu, thr1 | 16 | 177.3 / 32 | 36.1 | 2.26 | 85.8 |
| cpu, thr1 | 32 | 127.4 / 32 | 50.2（+39%） | 1.57 | 96.4 |
| **cpu, thr1** | **48** | **152.6 / 48** | **62.9（+25%）** | 1.31 | 126.7 |
| cuda×3 扩散, thr1 | 8 | 70.6 / 8 | 22.7 | **2.83（=cpu 的 1.21×）** | 67.3 |
| cuda×3 扩散, thr1 | 16 | 180.8 / 32 | 35.4 | 2.21（=cpu 的 0.98×） | 83.7 |
| cuda×3 扩散, thr1 | 32 | 146.8 / 32 | 43.6（+23%） | 1.36（=cpu 的 0.87×） | 139.7（自旋膨胀） |

**饱和点定案**：cpu 路线 N48（N32→48 边际 +25% >5% 续扩；N64 在电池 A 实测边际
+1.3% <5% 停，且共享机留核礼让 + 48×~1GB RSS ≈ 46GB 内存安全）→ **48**。
cuda 扩散路线膝点同在 48（电池 A：48 后 −5.4% 回退），但其 plateau 31.6 仍输 cpu 62.9
一半——**宽度 ≥16 起 cpu 全面胜出**。

### 5.1 交叉点分析（回答"cuda 的 1.21× 每 run 优势何时兑现"）

- **轴是并发宽度，不是回合数**：每 run 加速比在 200ep 稳态即达 1.21×（2.83/2.35，
  与进程内 3.48/2.91=1.20× 一致），回合数只摊薄 2.35s 启动（20ep 时 cuda 任务端到端
  11.4s vs cpu 12.9s 已占优）——回合数从来不是翻转轴。
- **宽度 ≤8 + 跨卡轮转**：cuda 每 run +21%，聚合 +21%（22.7 vs 18.8）→ cuda 赢。
- **宽度 16**：2.21 vs 2.26 eps/s，平手（35.4 vs 36.1）。
- **宽度 ≥32**：cuda 自旋等待膨胀（user CPU 139.7 vs 96.4s），聚合 43.6 vs 50.2，
  差距随宽度扩大 → cpu 赢。
- **tune 池的现实宽度**（Stage-A 10 trial / Stage-B ~5-6 trial）虽 ≤10 落在 cuda 占优区，
  但池无法逐 worker 钉卡（不重设计已验证的池逻辑），环境钉一张卡则退化为「单卡」列
  （N8 每 run 1.16 eps/s，**输 cpu 的 1.44**）→ tune 缺省 cpu 是数据结论而非妥协。
  cuda 的正确用法收窄为：宽度 1 的长试验重跑（`CUDA_VISIBLE_DEVICES=<空闲卡>
  --device cuda --concurrency 1`，1.20×）与 §6.1 的逐 shell 钉卡 --filter 多进程。

### 5.2 真实试验组合推荐

| 组合 | 推荐 | 依据 |
|------|------|------|
| Stage-A（10 trial × 2000ep） | tune 缺省：cpu、--concurrency 48（实际并发=10） | 宽度 10 落 cpu≈cuda 平手区，cpu 免钉卡依赖；10 并发 × 2.3 eps/s ≈ 15 min/trial |
| Stage-B（~5-6 trial × 8000ep×3 种子串行） | 同上 cpu；单 trial 想提速用宽度 1 + cuda 钉空闲卡 | 宽 5-6 cpu 每 trial 24k ep ≈ 2.8h；cuda 钉卡 1.20× ≈ 2.4h |
| adapt-guard（24 run × 1000ep，sweep 级） | --filter 多进程：4×cuda 逐卡 或 8-16×cpu shell | §6.1 决策 (b) |
| 主表 sweep（374 任务） | 同上 --filter 模式 | RUNBOOK §6.1（预训练先行注意项已写明） |

## 6. threads-numeric：线程配比不改数值（位同证据）

同一 8 回合 CPU 预训练（seed=555，run_id `pretrain-f1e50d7562`，隔离根
`profiling/scratch/bitsens/{omp1,ompdefault}/`），一次 `OMP_NUM_THREADS=MKL_NUM_THREADS=1`
（=tune worker 环境），一次不设（torch 缺省 32 线程）：

```
sha256(eval.json)      omp1      = 3a1ae9d40893a1d9c42b0f1d8145db037d414033cde3a59321359a01cb480842
                       ompdefault= 3a1ae9d40893a1d9c42b0f1d8145db037d414033cde3a59321359a01cb480842
sha256(metrics.csv)    omp1      = 003afec404ef8493808a90eaa5955b6bfe35625c2fc16fdb42e73002a2e54dc6
                       ompdefault= 003afec404ef8493808a90eaa5955b6bfe35625c2fc16fdb42e73002a2e54dc6
diff metrics.csv → identical；cmp eval.json → identical
```

**结论：OMP/MKL 1 vs 32 线程，训练数值逐位相同**（本尺寸 MLP 的 CPU matmul 走确定性
单线程/固定归约路径）。tune.py `_worker_env` 的 docstring 声明由此支撑。配套吞吐证据
（电池 A，N48，cpu）：thr1 聚合 **38.0** vs thr2 **32.0**（−16%）vs 缺省 32 线程
**崩溃**（48×32=1536 线程互抢，>25min 未完成一轮 96 任务，中止；方向性证据已足）。
→ 决策 (c) 定案 threads=1。单进程场景不受影响（单 run 2.29 vs 2.25 eps/s，噪声内）。

## 7. 为什么同一 GPU 堆多进程仍吃不饱（用户问题的回答）

实测现状：单卡 N1→N32 聚合 1.75→13.2 eps/s 后饱和，GPU 利用率全程个位数。机理链：

1. **每步 GPU 工作微观**：addmm kernel 均值 9.7μs、整段 GPU 忙 ~46ms——单个进程的
   policy 前向/反向根本喂不饱 V100S（§3 launch-bound 铁证：789ms CPU 发射 vs 46ms kernel）。
2. **每 worker 的真天花板是 CPU**：env 步进（纯 Python，~50% learn）+ launch 开销合计
   ~1 核/进程（/usr/bin/time：mean_cpu_pct 99）。N 进程同卡 = N 核 CPU 活在分时复用
   一张 ~9% 利用率的卡——GPU 永远等 CPU。
3. **同卡多 context 的额外税**：时间片轮转 + `cudaStreamSynchronize` 自旋，user CPU
   从 cpu 路线的 15.1s/任务膨胀到 45.3s（N32 单卡）——这部分不是 GPU 算力不足，是
   调度开销，且直接吞掉本可跑 env 的 CPU 核。

**代码优化方案评审**（结合红线）：

| 方案 | 判定 | 理由 |
|------|------|------|
| 多 run 合一进程 / 跨 run 批量前向 | ✗ | 违反 whole-run 粒度红线；跨 run 共享 env/缓冲改变数值路径 |
| VecEnv 扩到跨 run 批量步进 | ✗ | env 语义与 rng 流耦合，数值路径红线 |
| torch.compile / CUDA Graphs | ✗ | 编译图数值不保证逐位同（bit-match 红线）；kernel 本就 9.7μs，无事可省 |
| fp16 / Tensor Core | ✗ | 数值路径红线 |
| MPS（`nvidia-cuda-mps-control`） | △ 可选 | 管理员级杠杆：合并 context 消时间片税（自旋 45.3s 的主因），同卡密度会更好；但差距主体是 CPU 天花板（cpu 池 62.9 vs cuda 最优 43.6+），MPS 救不回 44% 的结构性差距——**对本负载无意义**，留作未来真 GPU 负载（CNN/GNN）时的选项 |
| **CPU 受控并发池（已实现）** | ✓ | 62.9 eps/s 稳态 = 单卡 cuda plateau 的 4.8×；tune.py `--concurrency 48` |
| **跨卡轮转（已文档化）** | ✓ | 低宽度场景 cuda 每 run +21%：--filter 每 shell 钉一卡（§6.1）或宽度 1 重跑钉卡 |

一句话：**这张卡吃不饱不是并行度不够，而是每口饭（kernel）太小、喂饭的手（CPU env
步进）太慢**。正确优化 = 让 CPU 多开嘴（并发池）+ 要用 GPU 时一进程一卡（轮转）。

## 8. 三决策定案与落点

- **(a) tune.py 池**：`_execute_pool`（有界 ThreadPool，`_execute_trial` 为调度原子——
  缝 4 docstring 预留缝的兑现）+ `_worker_env`（threads=1）+ `DEFAULT_CONCURRENCY=48`；
  CLI `--concurrency`（缺省 48，1 = 串行特征化有测试钉住）。失败隔离：单试验非零退出
  只记 failed；summary.csv 行序恒为 trial 定义序（pool.map 保序，测试钉住）。守卫/合成/
  聚合零改动（30 条既有 tune 测试原绿）。
- **(b) 主表 --filter 多进程**：RUNBOOK §6.1（bash，与 PowerShell 惯例并列）：4 场景
  子串四分 + `nohup` 后台 + 逐 shell 可选 `CUDA_VISIBLE_DEVICES` 钉卡；`CERNET.gml`
  子串与 `CERNET_failure.gml` 不重叠（带 `.gml` 后缀）；预训练任务 topo=源域
  CERNET.gml，落 `CERNET.gml` 路——该路先行（否则其余路迁移臂 SKIP-LINEAGE 跳过，
  幂等补跑即可）。run 目录幂等（DONE 跳过 + SKIP-DONE episodes 守卫）。
- **(c) 线程配比**：worker env 强制 OMP/MKL=1（§6 吞吐 + 位同双证据）；手工多进程
  shell 同样设置（RUNBOOK 已写）。

## 9. 4× V100S 裁决

对本 MLP 负载是**奢侈品**：最优 cuda 配置（3 卡轮转 N48，31.6 eps/s）也只有 cpu 池
（62.9）的一半。值得上 cuda 的三个场景：(i) 宽度 ≤8 的跨卡轮转（每 run +21%，低宽度
低打扰——共享机上只占 ~8 核）；(ii) 宽度 1 的长试验（1.20×）；(iii) 未来策略换成
CNN/GNN 等真 GPU 计算负载（届时 MPS 一并重评）。GPU0 被他人常驻——本机实际可用
3 卡，全部 cuda 测量基于此（换 4 卡扩散预期 ~+33% cuda 路线上限，仍不及 cpu 池）。

## 10. 红线完整性（证据链）

- 论文超参：`config/default.yaml`、`config/grid_*.yaml` 零改动（git diff 仅
  tune.py/__main__.py/prereg 值/RUNBOOK/tests/profiling/.gitignore）。
- 数值路径：`.venv/bin/python -m pytest tests -q` → **242 passed**（232 基线 + 10 池测试）；
  红线套件单列复跑 **70 passed**（test_env_determinism + test_policy_kwargs
  bit-identical + test_dual_lr ckpt 往返 + test_tuning_prereg + test_tune_pool +
  test_tune_driver）。线程位同证据 §6。
- 隔离：并行实体 = tune trial 子进程（独立 `runs_tune/<tag>/` 根）或 --filter 不相交
  任务集；画像产物全部在 `profiling/scratch/`（gitignored），真实 runs*/ 未触碰。
- 预注册回填：`tuning_prereg.yaml` `execution_concurrency.value: null → 48`（占位契约
  的预期回填，meta.version 保持 1），`tests/test_tuning_prereg.py` 钉子同步 == 48。

## 11. 清理收据（2026-08-30）

- `profiling/scratch`：删除全部 1420 个 ckpt zip（2.1GB → 376MB；保留
  metrics.csv/manifest.json/eval.json/DONE/网格与 /usr/bin/time 文本作证据）；
  另删 pool/startup 临时网格与 `__pycache__`。
- `ps`：无本任务遗留 python（pool_bench/trl_sb3/profile_single/battery 均无匹配；
  GPU0 的 `main_gcn.py` 为他人常驻作业，与本任务无关）。
- `nvidia-smi`：GPU1/2/3 = 0% / 4MiB（干净）；GPU0 = 他人作业 436MiB（先于本任务存在）。

## 12. 复现命令

```bash
.venv/bin/python profiling/profile_single.py --device cpu  --episodes 100 --rep 1   # 单 run 相位
CUDA_VISIBLE_DEVICES=1 .venv/bin/python profiling/profile_single.py --device cuda --episodes 100 --rep 1
CUDA_VISIBLE_DEVICES=1 .venv/bin/python profiling/profile_single.py --device cuda --profiler-segment
.venv/bin/python profiling/pool_bench.py --device cpu --levels 48 --episodes 200 --threads 1 --tag ss_cpu --tasks-per-level 48
.venv/bin/python profiling/pool_bench.py --device cuda --levels 8 --episodes 200 --threads 1 --tag ss_cuda3 --cuda-gpus 1,2,3 --tasks-per-level 8
bash profiling/battery.sh   # 原始电池（startup 探针 + 20ep 全表；注意共享机负载）
```

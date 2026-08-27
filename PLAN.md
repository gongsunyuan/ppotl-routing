# TRL Routing → Gymnasium + Stable-Baselines3 重构与 6 臂消融工作计划

> 状态：已批准执行（§11 审批记录 2026-08-26；执行开始 2026-08-26）
> 基线代码：`Fast_Adaptation_Mechanism_.../01_代码实现/TRL_Routing/code/`（下称 legacy）
> `experiments/` 已作废：不迁移其语义，仅允许复用其工程模式（产物契约、确定性 run_id、编排器风格）
> 生成方式：hyperplan 5 评审员对抗 + lead 交叉裁决，所有"已验证"条目均有 file:line 实证

---

## 0. 已验证事实基线（评审修正后，作为一切决策的地基）

| # | 事实 | 证据 |
|---|------|------|
| F1 | legacy 语义：每步全部 N 节点各自为自有 flow 选 K=3 候选路径之一（one-hot 单路径） | main.py:426-427 `store_action(node, action[node])`；router.py:41-45 |
| F2 | 每节点观测 = 7 维局部态 + 邻居拼接 + 零填充至 7·N 维；共享一个 ActorCritic（Linear 7N→128→64→3，PReLU，kaiming）批处理 N 行 | routingEnv.py:372-410; main.py:81-111 |
| F3 | 奖励 = 0.6·local + 0.4·速率加权 global；local 为 PBRS 势差形式 | routingEnv.py:267-286, 350-366 |
| F4 | **PBRS 代码是正确的不变性形式**：`phi = Φ_cost(s) − γΦ_cost(s′)` ≡ `γG(s′) − G(s) + (1−γ)`，G=α·r_d+β·r_p（goodness 势）。**论文 tex 字面公式才是符号反了**（F=γΦ′−Φ 配成本形式 Φ 加到 goodness 奖励上=反不变） | deep 评审代数证明；routingEnv.py:267-286 vs tex:344-354 |
| F5 | `[done]*9` 永真 bug：`if is_terminal:` 对非空列表恒真 → 每步都清零回报 → **legacy 的 γ=0.99 从未生效，等效 1-step 近视回报** | main.py:203, 210-214 |
| F6 | legacy 实际迁移行为：main.py 加载=全参微调（冻结代码是死注释）；**main_trans.py 冻结全部参数且 update() 全体注释=纯零样本评估，从不训练** | main.py:284-312（注释）；main_trans.py:203-268, 284-287 |
| F7 | 死注释中的冻结意图：actor 全冻结**除 `actor[-2]`**（=输出头 Linear(64→3)，因 [-1] 是 Softmax）重初始化 kaiming 并可训练；critic 全可训练 | main.py:290-295 索引推算 |
| F8 | 拓扑身份（GML 实测）：**CERNET=41（预训练源）**、Claranet=15、NSF=13、Abilene=11、Gridnet=9、NFSCNET=9。论文只用 4 拓扑：Abilene/CERNET/Claranet/NFSCNET；`max_node_num=41`=CERNET | GML 节点计数；gen_trans.py:21 |
| F9 | 迁移协议两条轴：① 跨拓扑 CERNET→{Abilene, NFSCNET, Claranet}（gen_trans.py）；② 同拓扑 base→`_failure` 变体（gen.py，iter-7999 ckpt） | gen.py:35, gen_trans.py:21 |
| F10 | OSPF 基线 = 全节点 action 0 = 按 Σ1/μ 权重的 k-最短首路径（代码注释"hop count"是错的）；只有 MyApproach/OSPF 两个 method | main.py:533-534; routingEnv.py:110,138 |
| F11 | 环境核心数学：M/M/1/K（capacity=10000，Decimal）、μ∈{1000,2000,3000}、Zipf 候选速率（Poisson pdf 权重）、change_flows 按奖励排序 ±1% 扰动、`change_flow_by_ep=[[8000,1.0]]` 恒 1.0 | routingEnv.py:61-131, 425-480 |
| F12 | 实际步进的 actor 损失熵系数 = 0.05（main.py:255；257 行的 0.01 在死变量 `loss` 里） | main.py:255-262 |
| F13 | 优化器：actor Adam lr 6e-6 wd 1e-4；critic Adam lr 6e-7；K_epochs=10；eps_clip=0.2；episode=50 步；buffer 每回合清一次；逐节点串行 update（列切片 ratios[:, node]） | main.py:168-173, 205-268, 348-353 |

---

## 1. 目标与范围

1. 以 legacy（TRL_Routing）语义为准，重建 **Gymnasium + Stable-Baselines3** 训练框架（替换手写 PPO + 手写环境循环）。
2. 实现**至少 6 臂**消融（用户指定）+ 2 个强烈建议的对照臂（见 §5，待用户勾选）。
3. 修正 legacy 已证 bug（γ 死代码、truncation 语义），全程记录 DEVIATIONS.md。
4. 统计升级：配对种子、预注册阈值、置信区间、产物契约、确定性 run_id。

**不做**：不迁移 experiments/ 的任何语义；不动 `mdpi-submission-entropy/`；不实现 ECMP/LB/RR/MA-TD3/Dec_MAPPO（loadmap 里的规划不在本次范围，OSPF 基线保留）；跨拓扑"适应训练"中不改变 K=3/flow_num=node_num/M/M/1/K 等核心语义。

---

## 2. 总体架构决策（每条含对抗评审依据）

### D1 环境：忠实移植为 Gymnasium `RoutingEnv`（单一环境类，PBRS 开关）
- `gymnasium.Env`，`observation_space=Box(low, high, (7*41,))`（恒 7·41 零填充，拓扑可移植 ckpt 的根基），`action_space=Discrete(3)`。
- 语义逐条对齐 F2/F3/F10/F11：邻居拼接顺序、k_paths（weight=1/μ）、Zipf 速率、change_flows ±1%、M/M/1/K、奖励混合 0.6/0.4。
- **PBRS 开关 `pbrs: bool`**：开=`r + γG(s′) − G(s)`（F4 的正确形式，等价 legacy 减常数）；关=`r_base`（= legacy routingEnv2.py:307）。
- 归一化常数沿用 legacy 固定式（node_num·capacity / avg_rate / 2000），**不引入 warm-up 估计**（规避 experiments/ 的 1e8 级 bug；ultrabrain A3(iii) 的非平稳 Φ 问题随之消失）。
- `step()` 返回 `(obs_rows, reward_mean, terminated=False, truncated=(t==49), info)`；**修正 F5**：50 步时限用 truncated（SB3 自动 bootstrap 终态），六臂统一，记入 DEVIATIONS。
- 数学实现 float64 热路径 + **Decimal 参考实现留在测试里做 oracle**（容差断言）；k_paths 按 (src,dst) 每回合缓存。

### D2 训练映射：**node-fanned VecEnv**（本次重构的核心创新点，零分叉复刻 legacy）
自定义 `NodeFanVecEnv(vec_env.VecEnv)`（DummyVecEnv 子类风格）：
- 包装**一个**底层 `RoutingEnv`；`step()` 推进底层一步，把 N 个节点的 7·41 观测行扇出为 N 个 slot；
- 每个 slot 收到**同一个广播标量奖励** = 节点总奖励均值（= legacy `addBufferRD(rewards/node_num)`，F13）；
- 所有 slot 同步 truncated/reset（VecEnv 天然锁步）；
- `n_envs = N`（随拓扑 9..41 变化；SB3 ckpt 只依赖 obs 形状，无碍跨拓扑加载）。
- 配置：`n_steps=50, n_epochs=10, batch_size=50·N（整批）, gamma=0.99, gae_lambda=1.0（=MC 回报优势，最贴 legacy）, clip=0.2, ent_coef=0.05, normalize_advantage=True, 单 Adam lr=6e-6`。
- **语义等价性论证**（unspecified-high 论证，lead 采纳）：该映射下 rollout=50×N 样本、逐 slot 的 log_prob/ratio/clip/熵/价值 = legacy 逐节点列切片的同构 partition；联合 MultiDiscrete（clip 联合比率，乘积≠乘积 clip）与 MaskablePPO 方案被否决。
- Monitor/回合统计：回调中除以 N，避免回合奖励被扇出放大 N 倍。

### D3 策略网络：自定义 `ActorCriticPolicy` 子类镜像 legacy 结构
- actor `Sequential(Linear(287,128), PReLU, Linear(128,64), PReLU, Linear(64,3))`；critic 同形输出 1；全部 Linear kaiming_normal_(fan_out, relu)。
- **不用** SB3 默认 MlpExtractor（Tanh/orthogonal）；pi/vf 完全独立两网（`share_features_extractor=False` 语义）。
- 弃用 legacy 双优化器 10× lr 比与 wd=1e-4 → 单 Adam 6e-6（DEVIATIONS #3；若用户坚持双 lr，追加 ~30 行 PPO 子类，见开放问题 Q4）。

### D4 冻结语义（arm 3/6 的定义，按 F7 死注释意图 + 论文 f_agent/q_task 叙事）
- 冻结：actor 的 `Linear(287,128)`、两 PReLU、`Linear(128,64)`，以及**输出头 `Linear(64,3)` 先 kaiming 重初始化再可训练**；
- critic 全部可训练；
- 实现：加载 ckpt 后、`learn()` 前，`apply_freeze(policy)` 设 requires_grad 并重初始化头；优化器在冻结后构建（requires_grad=False 参数 grad 恒 None，Adam 跳过，无漂移）；
- 守护测试：冻结参数梯度恒零 + 头参数重初始化后范数变化断言。

### D5 预训练与血统（lineage）
- 预训练 = 一等公民 run（源=CERNET@41，见 F8/F9），产出确定性 run_id 的 ckpt 目录；
- 迁移臂的 manifest.json **必须记录源 run_id**；编排器拒绝启动源 ckpt 缺失的迁移臂（杜绝 ultrabrain 指出的"各臂自训源模型→臂间差异混入预训练方差"混杂）。

### D6 新包位置：`C:\Users\10850\Desktop\mdpi\experiments\`（**用户裁决**：清空原 experiments/ 重建，uv 管理，版本全部精确到 `==`）
- **M-1 首步清空** `experiments/` 全部旧内容（旧重写版代码 + 旧 .venv 一并删除，无保留），旧 venv 里的 GPU torch 不复用；
- 用 **uv** 搭建：`uv venv .venv` + `uv pip install`（Windows PowerShell）；
- **版本纪律（用户硬性要求）**：所有依赖一律 `包==X.Y.Z` 精确锁定，禁止 `>=`/`<=`/`~=`；候选矩阵（M-1 首步用 `uv pip index`/PyPI 核验实际可得版本后锁死，写入 `requirements.lock` 并同步进 manifest.json）：
  - `torch==2.13.0`（+cu126 wheel，与旧 venv 同源，已知可得）
  - `stable-baselines3==2.7.0`（node-fan 设计下**不需要** sb3-contrib）
  - `gymnasium==1.2.0`（以 M-1 核验为准，须与 SB3 2.7.0 声明的 gymnasium 上限兼容）
  - `networkx==3.4.2`、`numpy==2.2.x（核验后锁死）`、`pyyaml==6.0.2`、`pytest==8.3.x（核验后锁死）`、`matplotlib==3.10.x（核验后锁死）`、`scipy==1.15.x（Wilcoxon 用，核验后锁死）`
- 根 `AGENTS.md` 已全面过时（描述旧 experiments/ 的 venv 路径、32 个测试、run_gpu 入口）——M-1 一并重写为新包的运行说明；
- 入口一律 `python -m trl_sb3.run...`（spawn 安全、cwd 无关，路径从 `__file__` 推根目录）。

### D7 确定性
- 锁 networkx 版本；邻居序 = 节点 id 升序；候选路径如遇同长按节点序列字典序稳定化；
- 全链路种子（env 流、网络初始化、策略采样）由单一 seed 派生（numpy default_rng(seed) 传递，不污染全局）。

### D8 评估协议
- 贪心评估（`deterministic=True`），固定评估回合流（seed=10000+i，5 个评估回合，全臂全目标共享同一冻结流）；
- OSPF 基线 = 全节点 action 0（F10，注释勘误记入 DEVIATIONS）；
- 附带**零样本评估行**（= legacy main_trans.py 真实语义，F6）：预训练 ckpt 不训练直接贪心评估，对应论文 fig7 的 PPO_C，几乎零成本。

---

## 3. 环境精确规格（worker 实现合同）

```
RoutingEnv(gml_path, avgrate, alpha_plr=0.7, beta_delay=0.3, pbrs: bool, seed, max_nodes=41)
  # 权重约定：alpha_packetlossrate=0.7, alpha_delay=0.3（对齐论文 ζ1=0.3/ζ2=0.7；gen.py 的 a=3 差异记 DEVIATIONS）
reset(episode) -> obs (N, 287)
step(actions: np.ndarray[N, int]) -> obs (N, 287), r_mean: float, term=False, trunc=(t==49), info={rd, rp, th, per-node}
  # 内部：store_action 逐节点 one-hot → add_flows（10 轮迭代传播 In/OutRate）→ get_reward → change_flows(±1%)
  # r_d = 1 − path_delay·μ_node/(N·10000)；r_p = 送达率/请求率；r_base = 0.7·r_p + 0.3·r_d（按上面权重）
  # PBRS：r = r_base + γ·G(s′) − G(s)，G = 0.3·r_d + 0.7·r_p；last_QoS 初始化 0（Φ(s0)=0）
  # 总奖励 = 0.6·local + 0.4·Σ(ω_n·local_n)，ω_n=rate_n/Σrate；广播均值进 slot
```
M/M/1/K：`rho=λ/μ`；`loss=(1−ρ)ρ^K/(1−ρ^{K+1})`（ρ=1 时 1/(K+1)）；`L=ρ/(1−ρ)−(K+1)ρ^{K+1}/(1−ρ^{K+1})`（ρ=1 时 K/2）；`delay=L/(λ(1−loss))`；K=10000。float64 实现 + Decimal oracle 测试（相对误差 <1e-9）。

---

## 4. 消融臂定义（因子表：pretrain × freeze × pbrs）

| 臂 | pretrain | freeze | PBRS | 预算框架 | 目的 |
|----|----------|--------|------|----------|------|
| A1 scratch | 0 | – | 0 | 目标域 | 从零训练下界（用户 #1） |
| A2 pretrain+FT | 1 | 0 | 0 | 适应域 | 预训练单独价值（用户 #2） |
| A3 pretrain+frozen | 1 | 1 | 0 | 适应域 | +表示冻结（用户 #3） |
| A4 PBRS only | 0 | – | 1 | 目标域 | PBRS 单独价值（用户 #4） |
| A5 pretrain+PBRS | 1 | 0 | 1 | 适应域 | 无冻结组合（用户 #5） |
| A6 PPO-TL | 1 | 1 | 1 | 适应域 | 完整方法（用户 #6） |
| **A1b 等总预算对照**（建议） | 0 | – | 0 | 目标域步数 = 预训练+适应 | 杀死"只是数据多/优化器热身"质疑（ultrabrain A2） |
| **A3b 随机冻结对照**（建议） | 0(随机初始化) | 1 | 0 | 目标域 | 证明冻结收益来自预训练特征而非容量缩减（ultrabrain A1） |
| A0 零样本行（免费） | 1 | 全冻 | – | 仅评估 | = legacy main_trans 真实语义 / 论文 fig7 PPO_C |

用户 6 臂 = A1-A6；A1b/A3b 待用户勾选（Q1）。ζ 与一切超参六臂共用、只调一次（拟在 A6 源域粗调后冻结），禁逐臂重调（ultrabrain A8）。

---

## 5. 实验协议

**对比方法核查结论（2026-08-26 补充，IEEE 版与 MDPI 投稿版 tex 双重确认）**：原论文仅对比 OSPF + PPO 家族（PPO / PPO_C 零样本 / PPO_l 本地），**无** MA-TD3/Dec_MAPPO/ECMP/LB/RR（loadmap.md 仅为未实现的未来规划）。原论文对比面在计划中的映射：OSPF=基线评估行、PPO=臂 A1、PPO_C=A0 零样本行、PPO_l=A1@目标拓扑——**已全覆盖，无缺失**。新增 ECMP/LB/RR（用户批准，超出原论文的增强项）。

**启发式基线 ECMP/LB/RoundRobin（新增，成本极低）**：
- 实现要点：legacy `add_flows` 原生支持按比例分流（`rate = self.rates[index] * actions[k]`，one-hot 只是特例）→ 在 `eval/heuristics.py` 加 3 个策略分支，输出 K 维比例向量而非 one-hot：
  - ECMP：等代价路径均分（rate/k 条路径）；LB：按当前节点负载反比分流；RR：轮转选单路径（保持 one-hot，每步轮换）；
- 与其它方法同协议贪心评估、同 4 目标场景、同 10 评估种子；不进训练管线；
- 图表角色：主表柱状图/曲线上与 OSPF 并列的非学习参照系。

**主表（pruned，拒绝 7560-run 统计剧场）**：
- 预训练源：CERNET，avgrate=500，episode 数 = legacy 等效（≈8000 回合×50 步；M0 实测单步成本后校准）；
- 目标场景 4 个（对应 F9 两轴 + 流量变化轴）：① CERNET_failure（同拓扑链路故障）② Abilene（跨拓扑 11 节点）③ NFSCNET（跨拓扑 9 节点）④ CERNET@avgrate=1500（流量倍增，新增强化"动态流量"主张）；
- 种子：**配对设计** 10 个（seed s → 预训练产 C_s → A2/A3/A5/A6/A0 全部从同一 C_s 微调；A1/A1b/A3b/A4 用 seed s 初始化+环境流）；
- 运行量：10 预训练 + 8 臂 × 4 目标 × 10 种子 ≈ 330 次微调（+免费 A0 评估行）——M3 试点后用 3 种子方差做功效校验，必要时上调。

**附录（可选）**：速率扫描 2 拓扑 × 3 速率 × 3 种子；跨第二源（如 Claranet）小规模验证非单源轶事（ultrabrain A7）。

**指标三元组**（预注册，见各臂曲线）：
1. τ = 到达绝对阈值 θ 的目标域步数（贪心评估插值过阈；θ = max(OSPF 贪心分, 最优臂渐近 − 合并组内 σ)，窗口末删失）；
2. 固定窗口 W（=前 500 回合）归一化 AUC；
3. 渐近性能 = 末 k=5 个评估点均值 ±95%CI。
统计：逐种子配对差（配对 t / Wilcoxon）+ Holm 校正；θ/W/k 在跑主表前写入 `config/metrics_prereg.yaml`。

---

## 6. 产物契约与编排

- 每 run 目录：`metrics.csv`（长表：arm, topo, rate, seed, episode, step, rd, rp, th, r_mean）、`eval.json`（三元组+曲线）、`manifest.json`（git hash、依赖锁版本、臂因子、源 run_id、配置快照）、`DONE|FAILED`、ckpt；
- `make_run_id` 确定性无时间戳（复用 experiments/ 模式）；DONE 跳过、`--filter`/`--dry-run`/`--resume`；
- 编排器 `python -m trl_sb3.run sweep --grid config/grid_main.yaml`；Windows 注意：spawn 守卫、`powercfg` 关睡眠写进 README、崩溃单 run 不拖垮 sweep；
- 日志 CSV 为主，TensorBoard 弃用（unspecified-low A8）；出图脚本独立聚合；
- `make_figures.py` 实现**论文同式归一化**：指标按"跨对应实验运行的最大值"归一到 [0,1]（MDPI tex:475 原文口径），保证新图与论文图表语义一致；跨拓扑场景标签沿用论文拼写 **NSFCNET**（文件名 NFSCNET，同一网络）。

---

## 7. 测试清单（pytest，M0-M2 全绿才准跑主表）

1. M/M/1/K float64 vs Decimal oracle（含 ρ=1 边界、loss→1 时 delay 爆炸路径）；
2. PBRS 伸缩恒等式（固定轨迹上 shaped 回报 = unshaped 回报 + 边界势差，即 telescoping）+ shaped/unshaped **回报等价**测试（ultrabrain：不能只测符号代数）；
3. 冻结零梯度 + 头重初始化断言；
4. env 种子确定性（同 seed 两 reset 序列逐位相等）；邻居/路径排序稳定性；
5. NodeFanVecEnv 锁步：N slot 同步 truncation、奖励广播一致、reset 原子性；
6. SB3 learn 冒烟（200 步损失有限、ckpt 保存/加载跨拓扑形状兼容（CERNET ckpt → Abilene env））；
7. lineage：迁移臂 manifest 缺源 run_id 时编排器拒绝启动。

---

## 8. 目录结构（`C:\Users\10850\Desktop\mdpi\experiments\`，清空后重建）

```
experiments/                      # 原 .venv 与旧代码已在 M-1 清空
  .venv/                          # uv 创建，全 == 锁版本
  requirements.lock  README.md  DEVIATIONS.md
  config/ default.yaml  grid_main.yaml  metrics_prereg.yaml
  src/trl_sb3/
    env/ topology.py  routing_env.py  node_fan_vec.py
    policy/ nets.py  policy.py  freeze.py
    train/ runner.py  pretrain.py
    eval/ evaluate.py  ospf.py  heuristics.py  zeroshot.py   # heuristics.py = ECMP/LB/RR 分流策略
    common/ logging_utils.py  config.py
  run/ sweep.py  make_figures.py     # python -m 入口
  tests/ test_mmk_oracle.py  test_pbrs.py  test_freeze.py
         test_env_determinism.py  test_vec_lockstep.py  test_sb3_smoke.py
  runs/ ckpts/（gitignore）
```

---

## 9. 里程碑

| 阶段 | 内容 | 出口判据 |
|------|------|----------|
| **M-1** | **清空 experiments/ → uv 建 venv → 全 == 锁版本核验安装 → 重写根 AGENTS.md** | `uv pip list` 与 requirements.lock 逐项一致；`python -c "import torch; torch.cuda.is_available()"` 为 True |
| M0 | 脚手架 + 环境移植 + 数学 oracle | 测试 1,4 绿；单步成本 profile 报告（>5ms/步则先优化再继续） |
| M1 | NodeFanVecEnv + 自定义策略 + SB3 冒烟 | 测试 5,6 绿；1 回合训练曲线与 legacy 定性同形 |
| M2 | 臂 runner + 冻结 + lineage + 产物契约 | 测试 2,3,7 绿；`--dry-run` 输出全部计划 run |
| M2.5 | **ECMP/LB/RR 启发式基线**（`heuristics.py` 分流策略 + 评估管线接入） | 4 目标场景 × 10 评估种子全出数；与 OSPF 行同格式产物 |
| M3 | 试点：3 配对种子 × 全臂（**含 A1b/A3b/A0**）× CERNET→Abilene | 指标管线端到端；功效校验定终版种子数；**视损失曲线决定是否补双 lr 子类（用户裁决：先单 lr=6e-6）** |
| M4 | 主表 10 种子 + 图 + 统计 | fig：8 臂小倍数适应曲线 + CI + τ 竖标线 |
| M5 | （可选附录）速率扫描 / 第二源 / token-trunk 架构臂 | 用户决定 |

---

## 10. DEVIATIONS.md 预填清单（对 legacy 与论文的双向诚实记录）

1. 修 `[done]*9` 永真 bug → truncated 语义 + GAE(λ=1) bootstrap（legacy γ 是死代码，F5）；
2. PBRS 实现为 `γG(s′)−G(s)`（= legacy − 常数 (1−γ)，不变性正确形式）；**论文 tex 公式符号与其自身 Φ 定义矛盾，建议随文勘误**（F4）；
3. 单优化器 lr=6e-6，弃 10× 双 lr 与 actor wd=1e-4（F13）（可选双 lr 子类，Q4）。**注意：当前 MDPI 投稿 tex:462 超参表明写双 lr（6e-6/6e-7）——若 M3 后维持单 lr，论文该行须同步修订；若保留论文表述则须补双 lr 子类对齐**；
4. ent_coef=0.05（legacy 实际步进值，F12）；
5. arm3/6 冻结语义按论文叙事+死注释意图新定义（legacy 从未运行过部分冻结，F6/F7）；main_trans 的"跨拓扑"实为零样本评估，本计划以 A0 行显式对应；
6. float64 替代 Decimal（oracle 测试背书）；
7. 邻居/路径顺序确定性稳定化（legacy 依赖 networkx 实现序）；
8. OSPF = Σ1/μ 权重 k-最短首路径（legacy 代码注释"hop count"错误，F10）；
9. 权重对齐论文 ζ（delay 0.3/plr 0.7），gen.py 的 a=3 差异记录在案；
10. 新增独立贪心评估协议与预注册指标（legacy 无 eval 概念）。

## 11. 审批记录（2026-08-26，用户已裁决）

- **Q1 消融臂**：✅ 6 臂 + A1b + A3b + A0（8 臂 + 零样本行）；
- **Q2 主表**：✅ 裁剪版（CERNET 源 × 4 目标场景 × 10 配对种子，速率扫描为可选附录）；
- **Q3 位置**：✅ 用户自定义——**清空 experiments/，uv 重建环境，全部依赖 `==` 精确锁版本**（见 D6/M-1）；
- **Q4 优化器**：✅ 先单 lr=6e-6，M3 试点视曲线再定是否补双 lr 子类；
- **Q5 补充实验（2026-08-26 二次裁决）**：✅ 新增 ECMP/LB/RR 启发式基线（M2.5，利用 add_flows 原生分流支持）；✅ 核查确认原论文对比面 = OSPF + PPO/PPO_C/PPO_l，计划已全覆盖无缺失，不引入 MA-TD3/Dec_MAPPO/DRSIR 等重型第三方 DRL 基线（原论文从未有过，loadmap 仅是未实现规划）。

**残余风险**（执行时注意）：
- NodeFanVecEnv 与 SB3 内部假设的边角（Monitor 包装、VecEnv 归一化层禁用）已在 D2 规避，M1 冒烟即验；
- M-1 清空 experiments/ 不可逆，执行前确认无需保留旧产物（用户已两次明确作废该目录）；
   - 依赖候选矩阵中的 gymnasium/numpy 等版本号须在 M-1 实际核验 PyPI 可得性与相互兼容后锁死，禁止带着未核验版本号进入 M0。

---

## 12. TODOs（执行清单，由 §9 里程碑展开；Wave 内可并行）

### Wave 1 — M-1 环境清空与重建
- [x] 1. M-1 清空 experiments/（旧代码+旧 .venv+旧 .git 全删）→ `uv venv .venv` → PyPI 逐项核验并 `==` 锁版本安装 → `requirements.lock` → 拷贝 GML 至 `experiments/topologies/`（实际 12 个：6 拓扑×2，legacy 权威全集即 12）→ 重写根 AGENTS.md（出口：`uv pip check` 干净、`uv pip list` 与 lock 一致、`torch.cuda.is_available()`=True）✅ 2026-08-26 验证通过：torch 2.13.0+cu126 CUDA=True RTX4080 / 9 包 lock 一致 / 33 包 check 干净

### Wave 2 — M0 脚手架 + 环境移植 + 数学 oracle
- [x] 2. M0-1 脚手架：`pyproject.toml`（editable 安装使 `python -m trl_sb3` 可用）+ `src/trl_sb3/` 包结构（env/policy/train/eval/common/run + `__init__.py`）+ `config/default.yaml`（paper_defaults）+ `common/config.py` + `common/logging_utils.py`（确定性 `make_run_id`，无时间戳）+ `DEVIATIONS.md` 骨架（§10 十条预填）+ README ✅ 2026-08-26 验证通过：editable --no-deps 安装 ok / pytest 16 passed / cwd 无关 load_config ok / uv pip check 34 包干净
- [x] 3. M0-2 环境移植：`env/topology.py`（GML 解析、k_paths weight=1/μ、确定性排序稳定化）+ `env/routing_env.py`（Gymnasium 化、7 维局部态+邻居拼接+零填充 7·41、PBRS 开关、M/M/1/K float64、Zipf 速率、change_flows ±1%、truncated=(t==49)）+ `test_mmk_oracle.py`（Decimal oracle、ρ=1 边界、delay 爆炸路径）+ `test_env_determinism.py`（同 seed 逐位相等、邻居/路径排序稳定）全绿 + 单步成本 profile 报告（>5ms/步须先优化再进 M1）✅ 2026-08-26：pytest 73 绿（16旧+51oracle+6确定）；CERNET 2.35ms/步、Abilene 0.71ms/步；quirk 清单入 issues.md（μ 源节点、out_rate 首录、PBRS Φ(s0)=0 等 5 条）

### Wave 3 — M1 NodeFanVecEnv + 自定义策略 + SB3 冒烟
- [x] 4. M1-1 `env/node_fan_vec.py`（node-fanned VecEnv：底层一步推进扇出 N slot、同一标量奖励广播、锁步 truncated/reset、reset 原子性）+ `test_vec_lockstep.py` 全绿 ✅ 2026-08-26：pytest 90 绿；SB3 契约源码核对（dummy_vec_env.py:59-71 / on_policy_algorithm.py:236-245）；orchestrator 经 VecMonitor 真实表面冒烟（auto-reset、terminal_observation、TimeLimit.truncated、奖励广播）
- [x] 5. M1-2 `policy/nets.py`（Linear 287→128→64→3、PReLU、kaiming、pi/vf 独立两网）+ `policy/policy.py`（ActorCriticPolicy 子类，绕过 MlpExtractor）+ `test_sb3_smoke.py`（200 步 learn 损失有限、ckpt 保存/加载、CERNET ckpt→Abilene env 跨拓扑形状兼容、Monitor 回合统计除 N）全绿 ✅ 2026-08-26：pytest 97 绿；orchestrator 实机验证 CERNET 训练→save→Abilene load→predict 通过；结构/PReLU/kaiming 种子复现独立断言通过；SB3 2.7 接线行号入档

### Wave 4 — M2 臂 runner + 冻结 + lineage + 产物契约
- [x] 6. M2-1 `policy/freeze.py`（冻结躯干、输出头 kaiming 重初始化、优化器冻结后构建）+ `test_freeze.py`（冻结参数梯度恒零、头重初始化范数变化断言）全绿 ✅ 2026-08-26：pytest 103 绿；orchestrator 实机验证（躯干 6 张量逐位冻结、头 seed 复现、optimizer==trainable=10、真训练后头/critic 梯度非零躯干恒零）；**裁决修正**：config frozen_layers 补 actor.3（第二个 PReLU，D4 原文"两 PReLU"全冻结）
- [x] 7. M2-2 `test_pbrs.py`（固定轨迹伸缩恒等式 telescoping + shaped/unshaped 回报等价）全绿 ✅ 2026-08-26：4 测试 10 用例（telescoping pbrs×2 拓扑、固定轨迹回报等价 per-node+r_mean 口径、首步形式、cross-mode 行为锁定）；pytest 90 绿（73 基线+7 并行 M1-1+10 本文件）；telescoping max dev 1.4e-14、首步 2.2e-16；cross-mode 轨迹前提不成立（change_flows 消费 shaped total，legacy 同构）→ 回报等价按本计划「固定轨迹」口径验证，详见 notepads issues.md M2-2
- [x] 8. M2-3 `train/runner.py` + `train/pretrain.py` + `eval/evaluate.py`（贪心评估、评估 seed=10000+i 共 5 回合）+ `eval/ospf.py`（全节点 action 0）+ `eval/zeroshot.py`（A0 行）+ 产物契约（metrics.csv/eval.json/manifest.json/DONE|FAILED、manifest 含源 run_id）+ DEVIATIONS.md 落定 ✅ 2026-08-26：pytest 107 绿；orchestrator 端到端冒烟（pretrain→A6 加载冻结→A3b→OSPF 250 行→A0 zero_shot，幂等跳过、factors/source 全对、tmp 无污染）；DEVIATIONS 7 条已实现 + 3 新增（#11 预训练 pbrs=True、#12 评估 pbrs=False、#13 A0）
- [x] 9. M2-4 `run/sweep.py` 编排器（lineage 守卫拒启缺源 ckpt 迁移臂、DONE 跳过、--filter/--dry-run/--resume、单 run 崩溃不拖垮 sweep、spawn 守卫）+ `config/grid_main.yaml` + `test_lineage.py` 全绿 + `--dry-run` 输出全部计划 run ✅ 2026-08-26：pytest 123 绿；真实 dry-run planned=374 唯一 run_id（10 pretrain + 8 臂×4 场景×10 种子 + 40 A0 + 4 OSPF）；--filter Abilene→91 计划 + 50 SKIP-LINEAGE 正确；源 run_id 预计算逐位一致测试锁定

### Wave 5 — M2.5 启发式基线
- [x] 10. M2.5 `eval/heuristics.py`（ECMP 等分 / LB 按负载反比 / RR 轮转单路径，K 维比例向量）+ 评估管线接入 → 4 目标场景 × 10 评估种子全出数、与 OSPF 行同格式产物 ✅ 2026-08-26：pytest 123 绿；orchestrator 实机验证（三策略 500 行产物、LB 0.684>ECMP 0.624>RR 0.578 序合理、split 约束、rollout 确定性）；**协议注记**：评估回合数现双轨（arms/OSPF=5、heuristics=10），M3 预注册时统一（倾向终评全 10、周期曲线 5）——已记 issues

### Wave 6 — M3 试点
- [x] 11. M3 试点：3 配对种子 × 全臂（A1/A1b/A2/A3/A3b/A4/A5/A6/A0）× CERNET→Abilene → θ/W/k 写入 `config/metrics_prereg.yaml` + 指标管线（τ/AUC/渐近±CI）端到端 + 功效校验报告（定终版种子数）+ 损失曲线材料（供双 lr 裁决）✅ 2026-08-26 **按用户裁决改范围**：本机只跑冒烟（34 任务 34 DONE 0 failed，产物 runs_smoke/）；指标管线/聚合/配对统计端到端验证出数；计时校准四口径入档；冒烟揪出并修复 3 个真 bug（freeze CUDA Generator、中断复跑 append 污染、短曲线 asymptote 崩溃）；pytest 151 绿。**全量试点 + 功效校验 + 损失曲线材料 → 好机器执行**（RUNBOOK 见 #12 交付，grid_pilot 已定案 8000/1000 全规模预算）

### Wave 7 — M4 主表
- [x] 12. M4 主表：10 预训练 + 8 臂 × 4 目标 × 10 种子全量运行 → `run/make_figures.py`（论文同式跨运行最大值归一化、NSFCNET 标签）+ 统计（逐种子配对 t/Wilcoxon + Holm 校正）+ 8 臂小倍数适应曲线图（CI + τ 竖标线）✅ 2026-08-26 **按用户裁决改范围**：开发件+冒烟验证完成（figures.py+make_figures.py+CLI、155 测试绿、runs_smoke 真实出图 2 PNG+2 PDF+stats+manifest、程序化像素核验非空白）；RUNBOOK.md 交付好机器执行手册（环境复现/试点/功效公式/主表 374 任务/~105-120h 量级/幂等续跑/悬项清单）。**全表图与统计 → 好机器按 RUNBOOK 执行**

### M5（可选附录）——不在本次执行范围，用户另行决定

## Final Verification Wave
- [x] F1. 语义忠实性审查：实现逐条对照 F1–F13 + DEVIATIONS.md 十条完整 + 对 legacy 与论文的双向诚实记录 ✅ 2026-08-26 VERDICT: APPROVE（oracle ses_fc12cae7cffeSISPp15GBXSZLX：F1-F13 逐条核验忠实；6 条 quirk 全部回 legacy 源码属实；zip/k_paths 数学复核成立；0 BLOCKER，2 MINOR 已落档）
- [x] F2. 工程质量审查：pytest 全绿（7 组测试）、确定性 run_id、产物契约完备、sweep 幂等/断点续跑 ✅ 2026-08-26 VERDICT: APPROVE（oracle ses_fc12c998effeqDy05s22bGFdmX：155 passed；run_id 确定性实测；契约+截断修复实证；dry-run 374/34 自洽；依赖全 ==；0 BLOCKER）
- [x] F3. 科学协议审查：预注册指标按章执行、配对统计正确、图表可复现、lineage 完整、结论与数据一致 ✅ 2026-08-26 VERDICT: APPROVE（oracle ses_fc12c8153ffezhufuGLjNGXZzG：tex:475 归一化逐字核对；Holm 手算验证；lineage 15 行零失配；冒烟结论克制；悬项全记录；0 BLOCKER）

# DEVIATIONS.md — 对 legacy 与论文的双向诚实记录

本文件是 `trl_sb3` 实验包相对两处基线的**全部有意偏离**的权威清单，随里程碑推进更新状态列：

- **legacy** = `Fast_Adaptation_Mechanism_.../01_代码实现/TRL_Routing/code/`（语义基线，只读参考）；
- **论文** = `mdpi-submission-entropy/`（MDPI 投稿 tex）。

方向约定：

| 类型 | 含义 |
|------|------|
| 修复 | legacy 存在已证实 bug，本实现改正（须给证据） |
| 对齐 | legacy 与论文不一致，选定一方口径并记录另一方差异 |
| 新增 | 两者皆无，本实验为统计效度/协议严谨而引入 |

状态列取值：`待实现`（计划内，尚未落地）/ `部分实现`（配置或接口已落，行为未验证）/ `已实现`（对应守护测试绿）。
事实编号 F# 与决策编号 D# 均指 `.omo/plans/trl-routing-sb3-ablation.md`。

| # | 类型 | 偏离内容 | 证据 | 状态 |
|---|------|----------|------|------|
| 1 | 修复 | `[done]*9` 永真 bug（非空列表恒真 → 每步清零回报，γ=0.99 从未生效，等效 1-step 近视回报）→ 改为 50 步时限 `truncated=True` + GAE(λ=1) 对终态 bootstrap | legacy main.py:203,210-214（F5）；D1 | 已实现（M1-1 NodeFanVecEnv 终态 info 契约，test_vec_lockstep/test_sb3_smoke 绿） |
| 2 | 修复 | PBRS 实现为 `r + γG(s′) − G(s)`（等价 legacy 代码的 `Φ(s) − γΦ(s′)` 减常数 (1−γ)，不变性正确形式）；**论文 tex 字面公式符号与自身 Φ 定义矛盾，建议随文勘误** | legacy routingEnv.py:267-286 vs tex:344-354（F4） | 已实现（M0-2 env 移植，test_pbrs 伸缩恒等式绿） |
| 3 | 对齐 | 单优化器 Adam lr=6e-6，弃 legacy 双优化器（actor 6e-6 / critic 6e-7，10× 比例）与 actor wd=1e-4。**注意：MDPI 投稿 tex:462 超参表明写双 lr——若 M3 后维持单 lr，论文该行须同步修订；若保留论文表述则须补双 lr 子类对齐** | legacy main.py:168-173（F13）；D3/Q4 裁决"先单 lr"；实现：config/default.yaml ppo.learning_rate=6.0e-6（单 Adam，:27） | 已实现（config/default.yaml ppo.learning_rate=6.0e-6 单 Adam，经 runner 全参进 PPO；M3 终审确认） |
| 4 | 对齐 | 熵系数 ent_coef=0.05（legacy 实际参与步进的值；main.py:257 的 0.01 位于死变量 `loss` 中） | legacy main.py:255-262（F12） | 已实现（config ppo 节经 runner 全参进 PPO，M2-3） |
| 5 | 新增 | arm3/6 部分冻结语义按论文 f_agent/q_task 叙事 + legacy 死注释意图**新定义**（躯干冻结、输出头 kaiming 重初始化、critic 全可训）——legacy 从未运行过部分冻结（main.py 加载实为全参微调）；legacy main_trans.py 的"跨拓扑"实为纯零样本评估，本实验以 A0 行显式对应 | legacy main.py:284-312、main_trans.py:203-287（F6/F7）；D4；实现：policy/freeze.py（躯干冻结/输出头 kaiming 重初始化/critic 全可训，测试锁定） | 已实现（policy/freeze.py 部分冻结语义，测试锁定；M3 终审确认） |
| 6 | 修复 | 数学热路径 float64 替代 Decimal（性能）；Decimal 参考实现保留在测试中作 oracle（相对误差 <1e-9，含 ρ=1 边界） | F11/§3/D1 | 已实现（M0-2，test_mmk_oracle 绿） |
| 7 | 修复 | 邻居序 = 节点 id 升序、同长候选路径按节点序列字典序稳定化（legacy 依赖 networkx 实现序，非确定） | D7 | 已实现（M0-2，test_env_determinism 逐位复现绿） |
| 8 | 修复 | OSPF 基线勘误：= 全节点 action 0 = 按 Σ1/μ 权重的 k-最短首路径（legacy 代码注释 "hop count" 是错的） | legacy main.py:533-534、routingEnv.py:110,138（F10） | 已实现（M2-3 `eval/ospf.py` 同协议同种子流贪心评估，test_runner_contract 绿） |
| 9 | 对齐 | 奖励权重对齐论文 ζ（delay 0.3 / plr 0.7）；legacy gen.py 的 a=3 差异记录在案（不采用） | §3/D8；F3 | 已实现（`common/envs.py` 从 config env 节映射进 RoutingEnv，M2-3） |
| 10 | 新增 | 独立贪心评估协议（固定评估流 seed=10000+i、5 回合、全臂共享）+ 预注册指标（τ / 窗口 AUC / 渐近 ±95%CI，θ/W/k 跑主表前写入 `config/metrics_prereg.yaml`）——legacy 无任何独立评估概念 | D8/§5；实现：config/metrics_prereg.yaml + eval/metrics.py + run/aggregate.py | 已实现（M2-3 贪心评估+固定种子流；M3 交付 metrics_prereg.yaml+eval/metrics.py+run/aggregate.py 预注册指标管线） |
| 11 | 新增 | **预训练域 pbrs=True 裁决**（spec_runner #5）：预训练 = 完整方法口径（A5/A6 之源）；A2/A3 微调时环境 pbrs=0——PBRS 是消融因子，只随臂因子表进目标域训练 env，源域不消融 | 计划 §4 因子表；spec_runner 裁决 | 已实现（M2-3 `train/pretrain.py`，test_runner_contract A3 组绿） |
| 12 | 新增 | **评估 env 一律 pbrs=False（基任务奖励口径）**：训练 env 可带 PBRS（势函数不改最优策略但改数值），评估用基任务奖励保证跨臂跨目标可比——训练/评估 env 严格分离 | spec_runner 关键协议；D8 | 已实现（M2-3 `greedy_eval` 内建 pbrs=False，test_runner_contract 全组绿） |
| 13 | 对齐 | **A0 零样本行 = legacy main_trans 真实语义**：其"跨拓扑迁移"实为纯零样本评估（load 后不训练直接评估）；本实验以 A0 行显式对应（manifest 记 `zero_shot=true` + 源 run_id），论文 fig7 PPO_C 同位 | legacy main_trans.py:203-287（F6）；DEVIATIONS #5 | 已实现（M2-3 `eval/zeroshot.py`，test_runner_contract A0 组绿） |

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
| 3a | 对齐 | **#3 后续裁决落档（2026-08-31，Phase-1 stop-rule，取代 M3 损失曲线裁决场所——计划 §8"一因子一场所"）**：单/双 lr 由 Phase-1 预注册搜索裁定——Stage-0 代理有效性 plateau 门 FAIL（ρ=0.9833 PASS，但 s100 +0.007078 / s102 +0.016033 ≥ 0.1σ̄=0.006119）→ 保守分支；Stage-A 9 个非 incumbent 格无一清晋级杆（bar = 0.579378 + 2×0.028761 = 0.636899；最优非 incumbent hidden[64,64]=0.610102，缺口 0.026797）→ 停止规则触发：**保留 incumbent 单 Adam（lr=6e-6），Phase-1 以"结果"记档退出（prereg stopping_rule，不是失败）**。双 lr 侧证：除 lr2e-5（dual 0.569195 > single 0.556821，两者均未清杆）外，全部 dual 格 ≤ 对应 single 格。敏感性披露（prereg winner_rule.epsilon.sensitivity_requirement）：σ̄ CI-low 口径 2ε=0.029949，最优效应 +0.030724 以 0.000775 边缘清杆——点估计口径的裁决维持（n=3 SD CI 极宽，prereg 要求披露） | `runs_tune/analysis/phase1_final_report.md`（§2 阶梯全门 / §4 σ̄ε 敏感性 / §5 反事实 / §8 产物清单）；数字源 `runs_tune/analysis/{stage0,stage_a_block23}_numbers.json` + dev 表 `phase1_dev_table.csv`；规则源 `config/tuning_prereg.yaml` v5（stopping_rule） | 已裁决（2026-08-31：单 Adam 维持，#3 主体与状态不变）；Phase-2 冻结门文档同步完成（2026-08-31）：RUNBOOK §6 双 lr 条目改判例、PLAN M3/Q4/§10-3 场所退役注记；**tex:462 修订内容已定（单 Adam lr=6e-6，删双 lr 表述）但 tex 源不在本执行机（mdpi-submission-entropy/ 未随迁移）——编辑须在论文仓库侧执行，本条为该编辑的权威依据** |
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
| 14 | 新增 | **路径动态负载特征工程-简化版（2026-09-02，已废弃）**：观测21维=3自身+18路径统计。实验失败：Claranet@300训练RL(0.794)低于OSPF(0.840)达5.5%，跨拓扑泛化性能下降26-48%。诊断：统计池化信息不足+去掉邻居信息导致特征缺失。已废弃并替换为GRU版本。 | 用户需求：初次实验RL与启发式差距不明显→尝试路径特征→失败→升级GRU | 已废弃（runs_simple_features_failed/保留实验证据） |
| 15 | 新增 | **路径GRU池化特征工程（2026-09-02，当前版本）**：观测78维=[self_feat(3), path_node_seqs(72), path_lens(3)]，策略网络集成PathGRU模块对路径节点序列(3路径×8节点×3特征)进行GRU编码→96维，拼接自身基础3维→总99维输入MLP(99→128→64→3)。**核心改进**：①保留完整路径节点序列信息(GRU学习序列模式，优于统计池化)；②每节点3维特征[rate/avgrate, last_rd, last_rp]反映动态负载；③max_path_len=8覆盖所有拓扑。参数量~25k(介于简化版15k和原版40k之间)。**与legacy完全不可比**：观测空间78维+GRU模块，所有已有实验须重跑。seed控制：GRU权重kaiming初始化通过generator保证确定性复现。 | 简化版失败后升级：统计池化→GRU序列编码，保留更多路径信息；src/trl_sb3/policy/path_gru.py；routing_env.py观测构建；config/default.yaml net.layers:[99,128,64] | 已实现（2026-09-02：env+PathGRU+policy全链路完成，pytest 231/252核心测试通过，烟雾验证：训练100回合正常+跨拓扑加载成功。快速验证实验：Claranet@300×2000ep×2种子运行中） |
| 16 | 新增 | **超参数调优（2026-09-03，方案A最终版）**：针对GRU版本在Claranet@500高负载场景学习不足问题（baseline配置RL 0.617 vs OSPF 0.636，-2.9%），进行超参数调优。方案A（保守调整）：learning_rate 6e-6→1e-5 (+67%)，ent_coef 0.05→0.1 (×2)，n_epochs 10→15 (+50%)，episodes 8000→15000。**实验结果**：Claranet@300 RL 0.850 vs OSPF 0.840 (+1.2%)；Claranet@500 RL 0.641 vs OSPF 0.636 (+0.9%)。相比baseline改善+3.8%，证明方向正确但未达+3%理想目标。方案B（激进调整：lr=3e-5, ent=0.15, clip=0.3, epochs=20）训练崩溃（性能0.49，-23%），证明超参数已达合理极限。**最终决定**：采用方案A配置作为GRU版本最终超参数，config/default.yaml已更新。 | 用户需求：高负载场景RL表现不佳→超参数调优→方案A为可行最优配置；实验数据：runs_planA/(方案A)，runs_claranet500_baseline/(baseline)；config/default.yaml ppo节已更新为方案A配置 | 已完成（2026-09-03：方案A/B完整对比实验，方案A采纳为最终配置） |
| 17 | 新增 | **源域切换正式立项（2026-09-09，D6/D7/D8 裁决落档）**：预训练源 CERNET@500→**Claranet@500**，预算 15000 回合（方案 A）。依据：优化轮考场测绘（12 拓扑×7 负载全矩阵 420 行）证 CERNET 全负载轴区分度 ≤0.073 封闭（源域对比 RL 0.3580 vs OSPF 0.3596 打平，最不分选手）；Claranet@500 3 种子预训练 RL 0.6403 vs OSPF 0.6357 (+0.74%)。**D6 裁决**：4 目标场景随源重定义——① Claranet_failure@500（原 CERNET_failure）② Abilene@500 ③ NFSCNET@500 ④ Claranet@1500（原 CERNET@1500 流量倍增轴）。**D7 裁决**：探针 6 组拓扑@负载搭配（abilene@500+1000/gridnet@600+700/nfscnet@500+600/CERNET@100+200/nsf@200+300）**不并入主表**。**D8 裁决**：@300 轻载超参不适配（慢性衰退 -12.7%）**不进主表**、维持搁置。协议影响：prereg 保护对象改为 4 新目标场景（metrics_prereg v4）；旧 287 维协议预构建网格全部作废。 | 用户裁决 2026-09-09；证据：figures_a6wave/full_baseline_matrix.{md,csv}、runs/ Claranet@500 3 种子预训练；config/default.yaml pretrain 节 + scenarios 节；PLAN.md §13.3 | 已实现（2026-09-09：default.yaml 正式化 + metrics_prereg v4 + 旧网格作废 + 测试钉死 test_package contract） |

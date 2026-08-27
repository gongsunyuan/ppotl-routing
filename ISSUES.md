# issues

## M-1（2026-08-26）

- GML 数量偏差：计划写"拷贝 14 个 GML"，实际 legacy `TRL_Routing/code` 权威拓扑集只有 **12 个**（6 拓扑 × 正常/_failure）。全树其余 GML（network1/、Dec_MAPPO/、ECMP/ 等目录）均为这 12 个的重复副本，`topologies/CERNET.gml` 也是 CERNET.gml 的重复；事件目录里有 `NSFCNET.gml`（2289 字节）与 code 目录的 `NFSCNET.gml` 同大小，系同一文件的命名变体。无第 13/14 个唯一拓扑，按任务"以实际拷贝数为准"拷贝 12 个。
- stable-baselines3 2.7.0 在 PyPI 存在，无偏差；gymnasium 1.2.0 满足 SB3 2.7.0 上限（<1.3.0），无偏差。



## RoutingEnv 移植差异（vs legacy）
- PBRS 按 plan §3 取 Φ(s0)=0（G_prev 由 last_rd/last_rp 派生、reset 归零），legacy φ 跨回合残留 → 每回合首步奖励与 legacy 不同
- last_QoS（last_rd/last_rp）reset 归零 vs legacy 跨回合残留上一回合值
- out_rate 缺 key 时默认 0.0（legacy dict 缺失返回 None，潜在崩溃点）
- μ 源节点 quirk：路径时延 mmk_delay 的 μ 取 flow 源节点（非路径节点 p），与 legacy routingEnv.py:288-353 一致，属故意保留
- st_th 别名差异：legacy step() 返回的 st_th 因别名实为 change_flows 后的原始 rates（未归一化）；本实现取 get_reward 时点的 rates/th_max_rate（先算后存）

## M2-2 PBRS 测试（2026-08-26）
- **cross-mode 轨迹前提不成立（legacy-faithful，非移植 bug）**：计划 §7-2 初稿操作化为"pbrs=True/False 同 seed 同动作 ⇒ 同轨迹"，实测不成立——`change_flows` 消费含 PBRS 项的 `total`（round 后 argsort 定 ±1% 速率），shaped 排序 ≠ base 排序 → 速率/obs 自第 2 步起分歧（seed=7、动作 rng 12345：Abilene/CERNET 均在 obs[1] 分歧；legacy routingEnv.py:366→473 同构反馈）。后果：两实例口径的 `Return_shaped − Return_unshaped == γ^50·G_50` 失败——Abilene max|lhs−rhs|=8.21e-01、CERNET=1.06e+00（G 序列本身已不同源）。处置：按计划正文"**固定轨迹上** shaped 回报 = unshaped 回报 + 边界势差"口径在单一 pbrs=True 轨迹验证（`tests/test_pbrs.py::test_shaped_unshaped_return_equivalence`），行为差异由 `test_cross_mode_first_step_shared_but_dynamics_diverge` 锁定。附注：PBRS 理论的不变性前提（动力学与奖励信号无关）在本 legacy 设计中本就不满足，消融臂 A4/A5 的 on/off 对比不受影响（各臂独立训练）。
- **恒等式本身两拓扑两模式数值成立**（固定轨迹，容差 rtol=1e-9/atol=1e-12）：telescoping max|lhs−rhs| = 1.07e-14（Abilene）/1.42e-14（CERNET）；首步形式 local_1==G_1·(1+γ) max dev = 2.22e-16；pbrs=False 的 local==G 逐位精确（0.0）。
- **total/r_mean 口径不可测恒等式（by design，跳过）**：total 带 Python round(,2)，每节点每步最大 0.005 舍入误差，折扣累计界 ≈ 0.005·Σ_{t=1..50}γ^{t-1} ≈ 0.198，远超容差；env 的 r_mean=mean(rounded total) 同理。故 r_mean 口径断言用 reward_local 的节点均值（未取整）替代。

## M2.5 启发式基线（2026-08-26）——rollout_split_episode vs RoutingEnv.step 语义出入
- **`_t += 1`（step():190）省略**：_t 仅被 step() 自身的 truncated 判定（:189）读取，split 驱动不产生 truncated，语义惰性；reset 会在下回合归零。
- **one-hot 构造（step():181-183）被比例 split 取代**：即 M0-2 固化的 `_add_flows(split)` 分流缝本义（docstring 明示"行和无需为 1"），非出入。
- **`_build_state()`（:188）调用但返回值不消费**：split 策略读 env 内部态而非 obs；保留调用以完整镜像 step 序列（_build_state 纯读无副作用）。
- **LB 回合 2+ 首步携带上回合尾负载**：reset() 不清 `_in_rate`（构造时为空 list，_add_flows 才写入），单 env 连跑 10 回合时回合 2+ 首步的 LB split 用上回合末步入率——在线"最近可得负载估计"语义；仅回合 1 首步（fresh env）无负载退化等分。ECMP/RR 不读 _in_rate，不受影响。
- **评估回合数双轨**：启发式=10（config `eval.heuristic_episodes`，§5 十评估种子协议），OSPF/A0=5（`eval.eval_episodes`）——跨方法比较时按 manifest.episodes 区分，聚合端（M3）勿假设 5。
## M2.5/M2-4（2026-08-26）评估回合数双轨——M3 预注册时统一
- 现状：arms 周期/终评=5 回合（config eval.eval_episodes，D8）；heuristics=10（config eval.heuristic_episodes，计划 §5 字面）；OSPF=5 且 sweep 中每场景 1 行（确定性策略无 seed 因素，幂等）
- 裁决建议（M3 metrics_prereg 时定案）：主表终评统一 10 回合（seed 10000..10009 全行共享），arms 周期曲线保留 5（省时）；OSPF/ECMP/LB/RR 行 episodes 提到 10

## M3（2026-08-26）中断复跑 append 污染（移交前必修）
- 现象：无 DONE 的中断 run 目录重启后 MetricsCSVWriter 以 append 模式打开 → metrics.csv 重复行（本次 3 个中断 pretrain 实证：~74000 行后 kill）
- 影响：好机器上 8000 回合长跑中断续跑会产出重复 episode 行，聚合端无感知
- 修法建议：runner 在非 DONE 复跑时先截断 metrics.csv（或重建 run 目录）；聚合端 collect_runs 加 (arm,episode,step) 去重防御
- 状态：待修（M4 开发批次一并委托）
- 状态更新（2026-08-26）：已修（runner truncate-on-restart + 测试锁定）——_train_and_write 复跑前 unlink 旧 metrics.csv，	est_interrupted_rerun_truncates_stale_metrics 锁定（999 行垃圾被截断）；聚合端去重防御（collect_runs）仍留 M4。

## M3 冒烟观察（2026-08-26，供 M4 验证时复核）
- 2 点曲线下 compute_table 的 asymptote 输出 None（k_eff=2 应可算 CI）——全量表 10 点走 strict 路径不受影响；M4 出图时若渐近列空值异常，先查此处降级逻辑
- 冒烟 τ 全删失属预期（10 回合达不到全量 θ）；A6 vs A1 配对 t_p=0.872 是 n=3 冒烟规模的无意义值，仅证明统计管线出数
## M3 终审 MINOR（2026-08-26）：rate_noise_clip 不随 change_flow scale 缩放
- legacy routingEnv.py:203-204 的 clip 界为 20·scale_factor；本实现固定 rate_noise_clip=20。当前 scale 恒 1.0（F11：change_flow_by_ep=[[8000,1.0]]）下两者等价、无实际影响；若未来启用非 1 scale 配置须同步改 RoutingEnv.reset 的 clip 界（一行：lo/hi 乘 scale）。出处：F1 终审报告。
- F3 MINOR：A3/A6 冒烟终评逐位同分——冻结+lr 6e-6+极少步下 pbrs 因子未改变 argmax；全量跑完后复核 A6 vs A3 差异可见性。出处：F3 终审报告。
- F3 MINOR：collect_runs 聚合端 (arm,episode,step) 去重防御留待 M4（runner 端截断已修，见上"M3 中断复跑"条）。出处：F3 终审报告。

# learnings

## M-1 环境搭建（2026-08-26）

最终版本矩阵（venv: uv venv --python 3.13，实际 CPython 3.13.11，取自 miniconda 全局解释器）：

| 直接依赖 | 锁定版本 | 备注 |
|---|---|---|
| torch | 2.13.0+cu126 | 必须带 `--index-url https://download.pytorch.org/whl/cu126` |
| stable-baselines3 | 2.7.0 | PyPI 存在，要求 gymnasium>=0.29.1,<1.3.0 |
| gymnasium | 1.2.0 | 满足 SB3 2.7.0 上限（1.2.3 亦可，按计划候选取 1.2.0） |
| networkx | 3.4.2 | torch 装时会先拉 3.6.1，显式 ==3.4.2 覆盖即可 |
| pyyaml | 6.0.2 | |
| numpy | 2.2.6 | 2.2.x 最新 patch |
| pytest | 8.3.5 | 8.3.x 最新 patch（8.3.2 缺号，PyPI 无此版） |
| matplotlib | 3.10.9 | 3.10.x 最新 patch（3.10.2 缺号） |
| scipy | 1.15.3 | 1.15.x 最新 patch |

传递依赖（uv resolver 裁决，未锁）：cloudpickle 3.1.2、pandas 3.0.5、pillow 12.3.0、sympy 1.14.0、fsspec 2026.7.0、filelock 3.32.3 等，`uv pip check` 全兼容。

安装坑：
- `uv venv .venv` 默认拉 CPython 3.14.7，cu126 wheel 对 cp314 未必存在；必须 `uv venv --python 3.13` 显式钉 3.13。
- torch cu126 wheel 2.4GiB，下载+安装约 8.5 分钟（uv 有缓存，重装秒级）。
- GML 清单（12 个，拷贝至 experiments/topologies/，Length 逐一核对一致）：
  Abilene.gml 3697 / Abilene_failure.gml 3589 / CERNET.gml 14181 / CERNET_failure.gml 13438 /
  Claranet.gml 3720 / Claranet_failure.gml 3607 / Gridnet.gml 4573 / Gridnet_failure.gml 4335 /
  NFSCNET.gml 2289 / NFSCNET_failure.gml 2220 / NSF.gml 3562 / NSF_failure.gml 3481
- GPU：NVIDIA GeForce RTX 4080 Laptop GPU，torch.cuda.is_available()=True。

## M0-1 脚手架（2026-08-26）

- **pyproject 依赖方案**：主方案可行，未用退路——`dependencies` 完整镜像 requirements.lock 的 9 条 `==`（含 `torch==2.13.0+cu126` 局部版本号），editable 安装一律 `--no-deps`，构建走 hatchling（隔离构建环境，不污染 venv）。uv `--no-deps` 跳过依赖解析，局部版本 pin 不触发解析失败；`uv pip check` 对 editable dist 的 Requires-Dist 校验通过（+cu126 == +cu126 精确匹配）。重装/升级包后需重跑一次 `uv pip install -e . --no-deps`。
- **hatchling editable 的 `__file__` 语义**：editable 安装用 .pth 直指 src 树，`trl_sb3.common.config.__file__` 即真实源文件 → `parents[3]` 稳定推得 experiments/ 根，`load_config()` 默认路径与 cwd 无关（已在 temp 目录验证）。
- **run_id 规范**：`make_run_id(slug=None, **factors)`，因素 k=v 按 key 字典序、`|` 连接、sha1 前 10 hex，返回 `{slug}-{hash10}`；slug 缺省取字典序首因素值；bool→true/false、None→none 规范化。因素键名：arm/topo/rate/seed/pbrs/freeze/pretrain（rate 统一 int，勿 500.0）。测试用"独立复算规范哈希全等"证明无时间戳成分，比对日期段正则更强且不误伤 hash。
- **default.yaml 根键 `paper_defaults`**：`load_config()` 检测到该键即解包返回内层，调用方直接 `cfg["ppo"]`；无该键则原样返回（覆盖片段/网格文件可复用同入口）。
- YAML 陷阱：pyyaml 的 float resolver 要求指数形式必须带小数点——`6.0e-6` 是 float，`6e-6` 会被当字符串。



## RoutingEnv 移植（M 环境落地）
- 7 维观测 = [round(dst/N,2), round(rates/avg_rate,2), len(p1)/max_hop, len(p2)/max_hop, len(p3)/max_hop(缺补0), last_rd, last_rp]；每节点 = own 7 维 + 邻居升序 7 维 + 零填至 7*41=287（legacy routingEnv.py:372-423，round 用 Python 银行家舍入）
- zip_dist 定案：Poisson 权重 e^-1·1^(i+1)/(i+1)! 归一化 → [0.6, 0.3, 0.1]；candidate_rates = [avgrate, avgrate+100, avgrate-100]（legacy routingEnv.py:112-132，第三轮核对）
- MMK 稳定式：rho>1 以 r=rho^-(K+1) 换元，loss=(1/rho-1)/(r-1)、L=rho/(1-rho)-(K+1)/(r-1)，防 rho^(K+1) 上溢与 inf/inf=NaN；rho==1 → loss=1/(K+1)、L=K/2；lam==0 → (0,0,0)（legacy routingEnv.py:425-450；Decimal prec=60 oracle 17 rho × 3 mu 全过 rel 1e-9）
- profile（100 步，pbrs=on，seed=0，avgrate=500）：CERNET 2.35 ms/步、Abilene 0.71 ms/步（均低于 5ms/步阈值）

## M1-1 NodeFanVecEnv（D2 核心创新，2026-08-26）
- **终态 info 契约（D1 修 F5，M1-2/M2-3 必须知道）**：truncated 步每 slot 的 info 含
  `terminal_observation`（该 slot 的 (287,) 末观测行）+ `TimeLimit.truncated=True`；SB3 PPO 在
  `on_policy_algorithm.py:236-245` 当且仅当两键齐备时做 `rewards[idx] += gamma*V(terminal_obs)`
  终态 GAE bootstrap。非截断步也恒有 `TimeLimit.truncated=False` 键（dummy_vec_env.py:66 同构）。
- **广播语义（F13）**：rewards = np.full(N, r_mean)（float64），= legacy addBufferRD(rewards/node_num)；
  锁步 done 向量全 True/全 False；auto-reset 时 reward 用本步、obs 为新回合首观测。
- **rng 流对齐坑（对照测试必看）**：`RoutingEnv.reset(seed=s)` 会**重播种** rng（routing_env.py:134-135），
  裸 `reset()` 走构造器续流——对照裸 env 必须同样 `reset(seed=s)` 才逐位相等；step 不耗 rng，
  故"vec 自动 reset obs == 裸 env 第二次 reset()"成立（reset 状态不依赖上一回合，仅 rng 流位置）。
- **VecEnv 基类坑**：`VecEnv.__init__` 内部调 `get_attr("render_mode")`（base_vec_env.py:75-79）→
  必须先绑底层 env 再 super().__init__；gymnasium Env 类属性 render_mode=None 使其静默通过。
- **get_attr 公共别名**：n/mu/pbrs → 底层 `_n`/`_mu`/`_pbrs`（底层私有命名是 M0 既成事实）；
  Monitor（gym Wrapper）不能包 VecEnv，冒烟/训练用 VecMonitor。
- SB3 seed() 存 `[seed+idx]`，单底层 env 只有 `_seeds[0]` 被消费；`reset(seed=...)` 参数优先于 `_seeds[0]`。

## M1-2 SB3 自定义 policy 接线（D3，2026-08-26）
- **唯二覆写点**：`_build_mlp_extractor`（policies.py:570-583）换 `SeparatePiVfNets`；`_build`（:585-634）
  覆写为 action_net/value_net=Identity（头在两网内，logits 直达 CategoricalDistribution）+ 直接建
  optimizer（跳过 ortho_init 块 :610-631）。`__init__` :535 末行 `self._build(lr_schedule)` 动态分派。
- **消费面零覆写**：forward :636-658 / evaluate_actions :719-741 / get_distribution :743-752 /
  predict_values :754-763 只调 `mlp_extractor.forward_actor/forward_critic`（:650-651/:735-736/:751/:762）
  → 自定义 extractor 只需这四入口 + `latent_dim_pi/vf` 属性（本实现 _build 不再读它，纯兼容保留）。
- **`get_action_dim(Discrete(K))` 返回 1**（动作向量维，preprocessing.py 口径）——Categorical 头维必须
  用 `action_space.n`。实测踩坑：头错建成 Linear(64,1)，断言层形状才暴露。
- **share_features_extractor=False 语义**：只让 extract_features（:660-682）走 pi/vf 双 FlattenExtractor
  （无参数）；pi/vf 独立语义实质落在 mlp_extractor 两网上。ortho_init=False 必须同时设（否则 :612 起
  正交块仍会 init features_extractor/mlp_extractor）。
- **float64 obs 全链路安全**：preprocess_obs（preprocessing.py）对非图像 Box 恒 `.float()` 铸型，
  learn（rollout buffer float64 → obs_as_tensor）与 predict（obs_to_tensor）两路都经 extract_features。
- **kaiming 复现性**：`nn.Linear` 构造即耗全局 rng（默认 kaiming_uniform weight + U(-1/√fan_in) bias）；
  generator 控制必须显式重放两者——kaiming_normal_(fan_out, relu, generator=g) + uniform_(bias,
  -1/√fan_in, 1/√fan_in, generator=g)（复刻 reset_parameters）。PReLU 单参数 0.25 不耗 rng。
  build_nets 先 actor 后 critic 定序消耗同一 Generator。
- **PPO.save/load**：policy 类经 cloudpickle 按引用序列化（importable 即可）；load 走
  `_get_constructor_parameters` kwargs 重建 → 覆写后的 `_build`，再 load_state_dict。ckpt 只依赖
  obs 形状 287 → CERNET ckpt 在 Abilene 上 predict 直接可用（F2/D1 实测锁定）。
- **VecMonitor**：episode_returns float32 累加，`episode["r"]` 不取整（仅 t round 6）→ 断言容差
  rel≈1e-4；`{"r","l","t"}` 键在截断步每 slot 的 info["episode"]。
- **freeze 锚点（M2-1）**：state_dict 前缀 `mlp_extractor.actor.*` / `mlp_extractor.critic.*`
  （config freeze.frozen_layers 的 actor.0/1/2 = mlp_extractor.actor.0/1/2，躯干三模块）。

## M2-1 freeze API（D4，2026-08-26）
- **arm3/6 调用契约（M2-3 runner 必读）**：`apply_freeze(model.policy, seed=run_seed)` →
  `rebuild_optimizer(model)`，两步缺一不可（冻结只动 requires_grad，优化器重建才剔除冻结参数）。
  优化器挂在 `model.policy.optimizer`（SB3 BaseAlgorithm 无 model.optimizer 属性，train 全程读 policy 侧）。
- **冻结语义实测**：SB3 2.7.0 PPO.train 走 `loss.backward()`（ppo/ppo.py:274-278，非 autograd.grad），
  requires_grad=False 参数不建图、`.grad` 恒 None；train 末 `zero_grad()` set_to_none=True →
  learn 后所有 grad 均 None，"在学"断言必须用参数值位移，不能查残留 grad。
- **头重初始化口径**：复用 nets._init_linear（kaiming fan_out/relu weight + U(-1/√fan_in) bias，
  同 Generator 控制）——`apply_freeze(seed=s)` 后头与初始权重无关、同 s 逐位复现。
- **冻结清单口径（已裁决修正）**：初版 config 只冻 actor.0/1/2（单 PReLU），与任务正文冲突时
  曾以 config 为准；后裁决以计划 D4 原文为准修正。终版：D4 冻结全集=actor.0-3（两 PReLU 含内），
  config 已修正为四锚点；若再调冻结范围只改 config 清单，代码零改动。

## M2-3 runner/评估管线（2026-08-26）——#9 sweep / #10 出图必读的 API 契约
- **训练入口**：`run_arm(arm, *, topo, rate, seed, source_run_id=None, episodes=None, out_root=None, device="cpu", config=None) -> Path`（train/runner.py）。臂→因子查 `ARM_FACTORS`（A1/A1b/A2/A3/A3b/A4/A5/A6 → pretrain/freeze/pbrs；A1b 与 A1 同因子，预算差靠 episodes，不入 run_id 因素）。`run_pretrain(seed, ...)`：arm="pretrain"、topo/rate 读 config pretrain 节、**pbrs=True**（裁决，DEVIATIONS #11）。未知臂 ValueError。所有 run_* 带 `config` kwarg 注入（grid 覆盖与测试都靠它）。
- **评估入口**：`greedy_eval(policy_fn, topo, avgrate, *, eval_seeds=None, n_episodes=None, config=None) -> dict`（eval/evaluate.py）——eval env 恒 **pbrs=False**、构造种子=首评估种子、每回合 reset(seed=10000+i)；返回 {"episodes":[{seed,steps,聚合4键}], r_mean_mean, rd_mean, rp_mean, th_mean}（`EVAL_AGG_KEYS` 四元组）。`run_eval_only(arm, topo, rate, *, policy_fn, seed=0, ...)`：eval-only 产物契约（OSPF/A0 共用；run_id 的 pretrain 因素从 extra_manifest["pretrain"] 取）。`ospf_policy(obs)=全 action 0`；`run_zeroshot(source_run_id, ...)`: PPO.load 不训练直接评估，extra_manifest={"zero_shot":true,"pretrain":...,"device":...}。
- **产物字段清单（聚合端按此扫描，#9/#10 继承）**：每 run 目录（`config paths.runs_dir`，缺省 experiments/runs，resolve_path 锚包根）：
  - `metrics.csv` 长表，列 `arm,topo,rate,seed,episode,step,rd,rp,th,r_mean`（rd/rp/th=per-node 均值标量；训练 run 每 env 步一行，eval-only run 每评估步行、episode=评估回合序号）
  - `eval.json` = {"curve":[{episode, *EVAL_AGG_KEYS}...], "final":{*EVAL_AGG_KEYS}}（eval-only 的 curve=[]）
  - `manifest.json`：run_id/arm/topo/rate/seed/factors{pretrain,freeze,pbrs}/source_run_id/episodes/device/config 全文快照/requirements_lock 全文/git_hash=None+git_hash_note/versions{torch,sb3,numpy,networkx}/created_at(ISO,UTC)；extra 顶层合入（zero_shot/device 等）
  - `DONE`（空文件，**最后写**，is_done 唯一跳过依据）/ `FAILED`（traceback 全文，与 DONE 互斥）；ckpt=`paths.ckpts_dir/<run_id>.zip`（全部训练臂都存，含 pretrain）
- **微调臂 ckpt 加载坑（钉死）**：`PPO.load(ckpts_dir/<source>.zip, env=vec, device=...)` 后**必须 `model.batch_size = model.n_steps * N_target`**——ckpt 恢复源拓扑整批（CERNET 2050），目标 N=11 的 rollout 是 550，不重算则 PPO.train mini-batch 切分错。load 不会重播种（torch 全局 rng 不消费），微调采样随机性不跨进程复现（如需复现需另加 manual_seed，M3 再议）。
- **_RunLogger 回合边界**：VecMonitor 透传 NodeFanVecEnv 的 `TimeLimit.truncated` 键（每步恒在，截断步 True）；episode 计数 +1 后 `% eval_interval==0` 触发周期评估（干净 env greedy_eval，不动训练 env rng 流）。`ppo_policy(model)` 内部读**默认** config 的 eval.deterministic（无注入参数，测试勿改该键期望它生效）。
- **新 config 键**（default.yaml）：`eval.eval_interval`（周期评估间隔，M3 可调）、`adapt.episodes`（微调回合缺省）、`paths.runs_dir/ckpts_dir`（产物根，resolve_path 锚 experiments/）。
- **坑**：PowerShell 管道改含中文源码（Get-Content/Set-Content）必把 UTF-8 注释读成 GBK 乱码——改文件一律走 Python（read_text/write_text）或编辑工具。

## M2.5 启发式基线（2026-08-26）——M3/M4 聚合必读
- **API**（eval/heuristics.py）：`SplitFn = Callable[[RoutingEnv, int], np.ndarray]`（读 env 内部态 + 回合内步序 → (N,K) 比例矩阵）；三策略 `ecmp_split/lb_split/rr_split(env, step_idx)`（ECMP=1/n_paths 均分；LB=1/(1+Σ路径节点入率) 归一化、首步无 _in_rate 退化为 ECMP；RR=one-hot 位置 step_idx mod n_paths，无内部计数器→回合内恒从 0 起轮转）。`HEURISTIC_SPLIT_FNS={"ECMP","LB","RR"}`。
- **分流评估协议**（M3/M4 聚合继承）：`rollout_split_episode(env, split_fn, episode_steps)`——调用方先 `env.reset(seed)`（同 greedy_eval 回合循环），内部逐行同构 step()（routing_env.py:184-188：_add_flows→_get_reward→th 先算后存→_change_flows→_build_state），返回单回合 dict（steps+聚合 4 键，seed 由调用方合入）。`heuristic_eval(arm, topo, rate, *, episodes=None, config)`：episodes 缺省读 **config `eval.heuristic_episodes`=10**（§5 十评估种子协议；OSPF/A0 仍是 `eval.eval_episodes`=5——聚合端按 manifest.episodes 区分）。`run_heuristic_eval(name, topo, rate, out_root=None, config)`：与 run_ospf_eval 同格式产物（arm=ECMP/LB/RR、seed=0、pbrs=False、metrics 每评估步行、eval.json curve=[]、manifest、DONE 幂等）。
- **关键事实**：reset() **不清** `_in_rate`——单 env 连跑多回合时，回合 2+ 首步 LB 携带上回合尾负载（在线"最近可得负载估计"语义，记 issues）；ECMP/RR 不读 _in_rate，与 step() 版评估逐位同源。run_eval_only 硬编码 greedy_eval（one-hot step 路径），比例分流无法复用，产物落盘在 heuristics.py 内镜像实现（只 import 原语，不改 evaluate.py）。

## M2-4 sweep 编排器（2026-08-26）——M3/M4 必读的编排契约
- **CLI**：`.venv\Scripts\python.exe -m trl_sb3.run sweep --grid config/grid_main.yaml [--dry-run] [--filter 子串] [--no-resume] [--device cpu] [--strict]`（相对 grid 路径锚定 experiments/ 根，cwd 无关）。分派在 `run/__main__.py`；sweep 顺序执行、无 multiprocessing（spawn 守卫）。
- **API**：`plan_runs(grid: dict, base_cfg: dict) -> list[RunTask]`（纯函数不查盘；顺序=全部预训练→臂×场景×种子→A0→OSPF；RunTask={arm,topo,rate,seed,source_run_id,episodes,config}）；`run_sweep(grid_path, *, dry_run, resume, filters, device, strict, base_config=None) -> SweepSummary{planned,done,skipped_done,skipped_lineage,failed}`。`base_config` 同 runner 的 config kwarg 注入模式（测试/覆盖用）。
- **run_id 预计算**：`task_run_id(task)` 与底层 runner 落盘目录名逐位一致（pretrain 口径：arm=pretrain/pbrs=true/freeze=false/pretrain=none；A0/OSPF 走 run_eval_only 口径 pbrs=false）。测试 `test_planned_run_ids_match_produced_run_dirs` 以"实跑后目录名集合==预计算集合"锁定——改 make_run_id/runner 因素前必跑该测试。
- **lineage 守卫（D5）**：迁移臂（A2/A3/A5/A6）与 A0 分派前查 `ckpts/<source>.zip` 存在且源 run 目录 DONE；缺 → SKIP-LINEAGE 继续（--strict 抛 RuntimeError）。dry-run 例外：计划内预训练视为将来可用源（--filter 滤掉预训练时迁移臂如实报 SKIP-LINEAGE，除非源已在盘上）。
- **grid 格式**（config/grid_main.yaml，自包含）：`pretrain{seeds,episodes}/arms/scenarios[{topo,avgrate}]/seeds/adapt_episodes/a1b_episodes/zeroshot/ospf`。预算字段经 `_merged_config` 并入 config 快照（manifest 会带上）；`a1b_episodes` 缺省=预训练+适应。未知臂 plan 阶段 ValueError。
- **OSPF 行数口径**：run_ospf_eval 恒 seed=0、run_id 无 seed 因素 → 每场景只计划 1 行（多种子会同 id 幂等跳过）。主网格总量 374 = 10 预训练 + 8 臂×4 场景×10 种子 + 40 A0 + 4 OSPF。
- **--filter 是子串**：命中 arm 或 topo 名（预训练行 topo=CERNET.gml，filter "Abilene" 会滤掉预训练）。--no-resume 只关 sweep 级 DONE 跳过，runner 内部仍幂等。

## M3 指标管线/预注册骨架/计时校准（2026-08-26）——orchestrator 试点必读
- **metrics API**（eval/metrics.py，纯函数无 IO，128 纯 LOC）：`time_to_threshold(episodes, values, theta, window) -> (tau, censored)`——相邻点跨 θ 线性插值求交点回合，首点已过阈=首点回合，τ>W 或未过阈 → (W, True) 删失；`window_auc`——取 e≤W 点 + W 边界插值/末值 LOCF、首点前不外推，梯形积分÷W；`asymptote(values, k) -> (mean, CI 半宽)`——末 k 点 t 分布（k≥2 契约，k=1 ValueError）；`mean_t_ci`（n≥2 均值±t 半宽，种子间汇总复用）；`pooled_within_sd(groups)`（√(Σ组内平方和/(N−G))）；`paired_tests(x,y) -> {t_p, wilcoxon_p, mean_diff, cohens_d}`（配对 dz；全零差契约化 p=1/d=0）；`holm(pvalues)`——**自实现 Holm-Bonferroni**（scipy 只有 BH/BY 的 false_discovery_control，无 Holm；升序乘 (m−rank) + 运行 max + 截 1）。
- **aggregate API**（run/aggregate.py，无 CLI，172 纯 LOC）：`collect_runs(runs_dir)`——只扫含 DONE 的 run 目录，manifest+eval.json 契约字段 → 记录 {run_id,arm,topo,rate,seed,episodes,factors,source_run_id,curve,final}；`derive_theta(records, prereg)`——prereg.theta 定值优先，否则 max(OSPF final, 最优臂渐近均值 − pooled σ) 逐 (topo,rate) 场景；**pretrain 行与 eval-only 基线不参与 θ**；缺 OSPF/学习臂 ValueError；`compute_table(records, prereg)`——逐 (arm,topo,rate) 行：学习臂 tau{by_seed,n_censored,mean,ci95}/window_auc/asymptote{by_seed:[mean,半宽]}，eval-only 行仅 final_mean，**pretrain 行整行排除**（损失曲线材料直读其 eval.json）；`paired_arm_diffs(records,a,b,topo,rate)`——final r_mean_mean 逐种子配对（共享种子<2 ValueError）；`holm_family(配对结果列表)`——t_p 族 Holm。
- **prereg 骨架** config/metrics_prereg.yaml：theta:null（试点后按 theta_rule 回填定值）/theta_rule/window_episodes:500/asymptote_k:5/alpha:0.05/primary_metric:tau/**final_eval_episodes:null**（issues 双轨统一字段，定案后基线行 eval episodes 对齐）。
- **grid_pilot.yaml**：3 预训练种子 + 8 臂×Abilene@500×3 种子 + A0×3 + OSPF×1 + heuristic×3 = **34 任务**（dry-run 验证）；episodes=null → 沿用 default.yaml 缺省参与规划，orchestrator 定稿后改；`a1b_budget_rule: pretrain + adapt` 留档键（sweep 不读，a1b_episodes 缺省即此规则）。
- **sweep 新网格键 `heuristic: [ECMP,LB,RR]`**：每场景每策略 1 行（seed=0、确定性分流无 seed 因素，episodes=eval.heuristic_episodes）；未知名 plan 阶段 ValueError；task_run_id 里启发式与 A0/OSPF 合并 eval-only 口径分支（pbrs/freeze=false）；**无该键行为不变**（test_lineage 原 6 测全绿 + 新增 1 测锁定 7 任务全链）。
- **计时校准**（RTX 4080 Laptop，episodes=5 含 1 周期评估+1 终评，CUDA 已 context 预热；SB3 对 MLP policy 上 GPU 发 UserWarning 属预期，实测仍提速）：

  | 口径 | s/episode | 外推 8000 预训练 | 1000 适应 | 9000 A1b |
  |---|---|---|---|---|
  | CERNET pretrain **CPU** | 1.623 | **3.61 h** | — | — |
  | CERNET pretrain **CUDA** | 0.830 | **1.84 h** | — | — |
  | Abilene A1 **CPU** | 0.528 | — | 0.15 h | 1.32 h |
  | Abilene A1 **CUDA** | 0.325 | — | 0.09 h | 0.81 h |

  口径注意：5 回合短测含模型构建/首次调用/2 次评估的固定开销 → s/ep 偏保守（高估）；正式跑 eval_interval=100 时 8000 回合含 80 次周期评估（每次≈5 评估回合）≈5% 额外。试点单种子全臂粗估（CUDA，预算 8000/1000/9000）：≈1.84 + 7×0.09 + 0.81 + 评估行分钟级 ≈ 3.3 h；CPU ≈ 3.61+7×0.15+1.32 ≈ 6 h。**预算外推线：单 seed 全臂（CUDA）≈3.3 h，3 seed 试点 ≈10 h 量级**（orchestrator 据此定试点规模/预算裁剪）。
- **asymptote strict 参数语义（M4 make_figures 继承）**：`asymptote(values, k, *, strict=True)`——strict=True 点数<k 抛 ValueError（预注册防静默降级）；strict=False 降级 k_eff=min(k, 点数)（短曲线可算、CI 自然变宽），k_eff<2 恒 ValueError。聚合端 derive_theta/compute_table 已传 strict=False；曲线 <2 点的记录不进渐近（compute_table 该种子渐近键缺、全缺则整键 None；derive_theta 学习臂全不可估 → θ 退化 max(OSPF)，pooled σ 自由度不足（N−G<1，如单臂单种子）→ σ=0）。τ/AUC 不受影响。

## M4 make_figures（2026-08-26）——好机器出图/统计必读
- **两模块分层**（250 纯 LOC 上限拆分）：`run/figures.py`（191 纯 LOC）= 归一化 + 图形纯构建；`run/make_figures.py`（116 纯 LOC）= 统计族 + IO 编排 + CLI。matplotlib.use("Agg") 在 figures.py 模块头（import 即生效）；__main__.py 分派惰性 import make_figures（sweep 启动不载 matplotlib）。
- **figures API**：`TOPO_LABELS={"NFSCNET.gml": "NSFCNET"}`（论文拼写）+ `topo_label(topo)`（其余去 .gml）；`normalization_denominators(records) -> {(topo,rate): max}`——tex:475 口径：场景内跨全部 run 的曲线点+final r_mean_mean 最大值（分母 ≤0 抛 ValueError）；`normalize_records(records) -> (归一副本, 分母表)`（纯函数不改输入）；`build_adaptation_figure(records_norm, table, window) -> Figure`（2×4 逐臂，episode 点种子均值±95%CI 带、τ 竖虚线取 table 行 episode 原量纲、全删失画 W 边界+子图标 "τ censored"、OSPF/ECMP/LB/RR 灰点线 hline；多场景=同子图多色线；场景色 zip prop_cycle strict=False）；`build_asymptote_figure(table, denominators)`（逐场景子图 ≤2 列，学习臂柱=asymptote.mean/den+CI 半宽误差棒，OSPF/ECMP/LB/RR/A0 灰柱=final_mean/den，asymptote None 学习臂跳过+图底注 "curves <2 points"）。constrained_layout + fig.legend(loc="outside lower center") + savefig(bbox_inches="tight")。
- **make_figures API**：`compute_stats(records)`——逐场景学习臂(≠A1, curve 非空) vs A1 配对对 + holm_family（共享种子<2 的对静默跳过）；`make_figures(runs_dir, out_dir="figures", prereg_path=...) -> manifest dict`（相对路径 resolve_path 锚 experiments 根；θ/W/k 全从 prereg 读；runs 无 DONE → ValueError）；`main(argv)` argparse --runs/--out/--prereg。
- **figures 产物契约**（默认 figures/，冒烟实测 figures_smoke/）：`adaptation_curves.png(150dpi ~96KB)+.pdf`、`asymptote_bars.png+.pdf`、`stats.json`（alpha/primary_metric/baseline/correction 头 + pairs[] 含 seeds/diffs/t_p/wilcoxon_p/mean_diff/cohens_d/holm_p）、`stats.csv`（扁平行同列）、`figure_manifest.json`（created_at/runs/out/normalization 说明/denominators{topo@rate}/prereg 摘要/文件清单/n_runs）。冒烟 34 run → 7 配对对（A1b..A6 vs A1）全 holm_p=1.0（n=3 规模无意义，与 issues 记录一致）；分母含 pretrain 场景 CERNET.gml@500（无害，图不消费）。
- **坑**：τ 未删失时插值交点 ≤ 末曲线点 → vline 恒在 xlim 内；全删失 vline 在 W=500 而冒烟曲线只到 ep10 → 子图 xlim 扩到 500（诚实但冒烟图左挤，全量 1000 ep 无此问题）；A3/A6 冒烟 final 逐位相同（同因子的巧合数据，统计对各自独立）。

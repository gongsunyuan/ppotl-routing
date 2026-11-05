# Adaptive Intelligent Routing Based on Transfer Reinforcement Learning in Dynamic Networks

> **Revision note (Entropy manuscript, entropy-4625886, major revision):** the revised manuscript adds a disjoint independent test stream (seeds 20000–20009). Where independent tests reverse the selection-stream pattern, the Summary of Findings below states both; the manuscript's Sections 5.2–5.6 take precedence.

## What this repository contains

The experimental framework (Gymnasium + Stable-Baselines3), configurations, run artifacts, and analysis scripts for a controlled study of transfer reinforcement learning in dynamic network routing. The study separates source-domain pretraining, representation freezing, and potential-based reward shaping (PBRS) and tests whether each component provides an independent benefit. The proposed configuration, PPO-TL, combines pretraining with PBRS and keeps every actor and critic parameter trainable; freezing is evaluated separately as an ablation.

The environment, method, and statistics are specified in the manuscript (Sections 3–5); intentional differences from the legacy implementation are recorded in [DEVIATIONS.md](https://github.com/gongsunyuan/ppotl-routing/blob/main/DEVIATIONS.md).

## Protocol in brief

- Ten paired seeds; every transfer arm loads the source checkpoint produced by its own seed.
- Evaluation uses greedy actions and fixed streams: selection seeds 10000–10009, independent test seeds 20000–20009. PBRS is disabled during evaluation.
- Best-checkpoint and end-of-training scores are both retained. The best-checkpoint rule was fixed before the v5 rerun, after an earlier terminal-score run revealed late-training decay, and applied uniformly to all arms.
- Independent tests, terminal references, confidence intervals, and effect sizes were added for the journal revision and are reported separately from the original adaptation analysis.

## Scenarios and arms

| Scenario | Topology | Average rate | Purpose |
|---|---|---|---|
| Link failure | Claranet\_failure | 500 | Links removed from the source topology |
| Cross-topology | Abilene | 500 | 11-node topology |
| Cross-topology | NSFCNET | 500 | 9-node topology |
| Increased traffic | Claranet | 1500 | Source traffic tripled |

Arms: A0 zero-shot, A1 scratch, A1b equal budget, A2 pretrain + fine-tune, A3 pretrain + freeze, A3b random freeze, A4 PBRS only, A5 pretrain + PBRS (PPO-TL), A6 all three. A0 is evaluated without target training and is not counted among the target-trained learning arms.

Baselines: OSPF is the static shortest-path candidate (no dynamic reconvergence); ECMP splits equally over available candidates without equal-cost testing; LB splits inversely to load; RR rotates round-robin.

## Summary of findings

Selection stream (seeds 10000–10009): all eight target-trained arms exceeded OSPF under link failure; PBRS had nine of twelve positive contrasts; freezing had all eight negative; pretraining was scenario-dependent.

Independent test stream (seeds 20000–20009): the proposed A5 configuration did not exceed OSPF in any of the four scenarios (mean differences negative throughout). PBRS contrasts split six positive and six negative; the joint freezing configuration has three positive and five negative; both pretraining contrasts are negative in all four scenarios. None of the 28 arm-versus-scratch comparisons survived Holm correction.

Best-checkpoint reporting reduced the effect of late-training decline without changing the underlying training trajectories. Interpretation limits: ten paired seeds; checkpoint selection favors learning arms over deterministic baselines; light-load experiments were excluded after the selected hyperparameters proved unsuitable for that regime.

## Artifacts

- [src/trl\_sb3/](https://github.com/gongsunyuan/ppotl-routing/blob/main/src/trl_sb3) — environment, policies, training and evaluation code
- [config/](https://github.com/gongsunyuan/ppotl-routing/blob/main/config) — experiment defaults, scenario grids, the fixed evaluation protocol
- [topologies/](https://github.com/gongsunyuan/ppotl-routing/blob/main/topologies) — network topology files
- [runs/](https://github.com/gongsunyuan/ppotl-routing/blob/main/runs) — v5 run artifacts (per-run metrics, evaluations, manifests, checkpoints)
- [runs\_final\_ckpt\_v1/](https://github.com/gongsunyuan/ppotl-routing/blob/main/runs_final_ckpt_v1), `ckpts_final_ckpt_v1/`, [figures\_main\_v4/](https://github.com/gongsunyuan/ppotl-routing/blob/main/figures_main_v4) — previous last-checkpoint protocol, kept for comparison
- [DEVIATIONS.md](https://github.com/gongsunyuan/ppotl-routing/blob/main/DEVIATIONS.md) — all intentional differences from the legacy implementation and earlier protocols
- [RUNBOOK.md](https://github.com/gongsunyuan/ppotl-routing/blob/main/RUNBOOK.md) — execution and artifact-management procedures

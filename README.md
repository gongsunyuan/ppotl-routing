# Adaptive Intelligent Routing Based on Transfer Reinforcement Learning in Dynamic Networks

## Experiment Overview

This repository contains the experimental framework, configurations, and analysis artifacts for a controlled study of transfer reinforcement learning (TRL) in dynamic network routing. The study examines a method that combines source-domain pretraining, representation freezing, and potential-based reward shaping (PBRS), and asks whether each component provides an independent benefit.

The original routing environment was rebuilt with Gymnasium and Stable-Baselines3. The experimental framework adds deterministic run identities, paired random seeds, fixed evaluation streams, preregistered metrics, and explicit records of deviations from the legacy implementation.

The study addresses four questions:

- Does source-domain pretraining improve target-domain performance?
- Does freezing pretrained decision layers help during adaptation?
- Does PBRS improve learning or final routing performance?
- Can the observed effects be explained by the total training budget rather than transfer learning?

## Routing Task and Method

The routing task models a network in which every node repeatedly selects one of three candidate paths for its own traffic. The objective is to reduce packet loss and delay while traffic conditions change. Links use finite-capacity M/M/1/K queueing models, and flow rates are perturbed during each episode to represent dynamic demand.

The learning agent uses Proximal Policy Optimization (PPO). A gated recurrent unit (GRU) encodes the node sequence of each candidate path, and the resulting path representations are combined with local node state before being passed to independent actor and critic networks. A node-fanout mapping presents the observations of all routers as synchronized learning samples with a shared network-level reward.

This GRU path representation is an intentional change from the legacy 287-dimensional observation design. The change and its experimental consequences are documented in [`DEVIATIONS.md`](DEVIATIONS.md).

## Experimental Design

### Source and Target Scenarios

The source policy is pretrained on the Claranet topology at an average traffic rate of 500 for 15,000 episodes. Target-domain adaptation uses 1,000 episodes. The equal-budget control trains from scratch for 16,000 episodes so that its total budget matches pretraining plus adaptation.

Ten paired seeds are used. All transfer configurations associated with a given seed load the source checkpoint produced by that same seed, preventing source-model variation from being mixed with the component comparison.

| Target scenario | Topology | Average rate | Purpose |
| --- | --- | ---: | --- |
| Link failure | Claranet_failure | 500 | Test robustness when links are removed from the source topology |
| Cross-topology transfer | Abilene | 500 | Test generalization to an 11-node topology |
| Cross-topology transfer | NFSCNET | 500 | Test generalization to a smaller 9-node topology |
| Increased traffic | Claranet | 1500 | Test adaptation when the source-topology traffic is tripled |

### Learning Configurations

The three main factors are pretraining, decision-layer freezing, and PBRS. Additional controls isolate zero-shot transfer, random freezing, and total training budget.

| Configuration | Description |
| --- | --- |
| A0: Zero-shot | Evaluate the pretrained source policy without target-domain training |
| A1: Scratch | Train in the target domain without pretraining or PBRS |
| A1b: Equal budget | Train from scratch with the combined pretraining and adaptation budget |
| A2: Pretraining + fine-tuning | Adapt the pretrained policy without freezing or PBRS |
| A3: Pretraining + freezing | Adapt with the decision trunk frozen and without PBRS |
| A3b: Random freezing | Apply the same freezing pattern to a randomly initialized policy |
| A4: PBRS only | Train from scratch with PBRS |
| A5: Pretraining + PBRS | Fine-tune the pretrained policy with PBRS and without freezing |
| A6: Complete method | Combine pretraining, decision-layer freezing, and PBRS |

Freezing applies only to the multilayer-perceptron decision trunk. The GRU feature extractor remains trainable, the output head is reinitialized and trained, and the critic remains trainable. The freezing results therefore concern the decision layers rather than the complete representation.

### Non-Learning Baselines

Four deterministic routing baselines provide non-learning reference points:

- **OSPF** selects the first path from candidate paths ranked by the sum of inverse link service rates.
- **ECMP** divides traffic equally among equal-cost candidate paths.
- **LB** divides traffic inversely according to current load.
- **RR** rotates among candidate paths in round-robin order.

## Evaluation and Statistical Analysis

Training and evaluation are separated. Evaluation uses greedy actions, ten fixed evaluation episodes, and the same evaluation streams across configurations and scenarios. PBRS is used only where specified during training; evaluation rewards do not include the shaping term.

The final v5 protocol evaluates each learning configuration every 100 training episodes and retains a checkpoint whenever performance reaches a new maximum. The primary final result is obtained from this best checkpoint, while the last-checkpoint result is also retained for comparison with the earlier v4 protocol. The best-checkpoint rule was preregistered before the v5 rerun and applied uniformly to all learning configurations.

The preregistered measures are time to cross a scenario-specific performance threshold, area under the learning curve over the first 500 episodes, and asymptotic performance over the final five evaluation points. Component comparisons use paired t-tests and Wilcoxon signed-rank tests, with Holm correction for multiple comparisons.

## Summary of Findings

The completed v5 experiment supports the following observations:

- All eight target-trained learning configurations exceeded OSPF in the link-failure scenario. Selected configurations also matched or exceeded OSPF in the other scenarios, but the results do not establish general superiority over OSPF.
- PBRS was the most consistently positive component. Nine of twelve component comparisons were strictly positive, and the remaining three were within 0.0003 of zero.
- Decision-layer freezing was not beneficial. All eight relevant paired comparisons were negative.
- Pretraining had scenario-dependent effects. It provided a positive signal on Abilene but was generally unfavorable in the increased-traffic scenario.
- None of the 28 paired statistical comparisons remained significant after Holm correction. The experiment therefore does not support claims of statistically significant improvement at the available sample size.
- Best-checkpoint evaluation reduced the effect of late-training decline on reported final performance without changing the underlying training trajectories.

These findings must be interpreted together with their limitations. Best-checkpoint selection gives learning configurations a selection advantage that deterministic baselines do not receive. Ten paired seeds provide limited statistical power, and the GRU representation means that the final method is not identical to the legacy method. Light-load experiments were excluded from the main comparison after the selected hyperparameters proved unsuitable for that regime.

## Experimental Artifacts and Documentation

- [`src/trl_sb3/`](src/trl_sb3/) contains the environment, policies, training pipeline, evaluation methods, and aggregation code.
- [`config/`](config/) contains the experiment defaults, scenario grids, and preregistered evaluation protocol.
- [`topologies/`](topologies/) contains the network topology files used by the experiments.
- [`runs/`](runs/) contains the current v5 run artifacts, including per-run metrics, evaluations, manifests, checkpoints, and completion records where applicable.
- [`figures_main/`](figures_main/) contains the current figures and statistical summaries.
- [`runs_final_ckpt_v1/`](runs_final_ckpt_v1/), `ckpts_final_ckpt_v1/`, and [`figures_main_v4/`](figures_main_v4/) preserve the previous last-checkpoint protocol for comparison.
- [`REPORT.md`](REPORT.md) provides the complete experimental report, numerical results, statistical analysis, and discussion.
- [`DEVIATIONS.md`](DEVIATIONS.md) records intentional differences from the legacy implementation and earlier experimental protocols.
- [`RUNBOOK.md`](RUNBOOK.md) documents the execution and artifact-management procedures.

# Adaptive Intelligent Routing Based on Transfer Reinforcement Learning in Dynamic Networks

> **Revision note:** This repository accompanies the Entropy manuscript (manuscript ID entropy-4625886, major revision). The revised manuscript adds a disjoint independent test stream (seeds 20000–20009). Where the independent tests reverse the selection-stream pattern, the Summary of Findings below states both outcomes, and the revised manuscript's Sections 5.2–5.6 take precedence.

## Experiment Overview

This repository contains the experimental framework, configurations, and analysis artifacts for a controlled study of transfer reinforcement learning (TRL) in dynamic network routing. The study examines transfer mechanisms for PPO-based routing, namely source-domain pretraining, representation freezing, and potential-based reward shaping (PBRS), and asks whether each component provides an independent benefit. The proposed configuration, PPO-TL, combines source-domain pretraining with PBRS and keeps every actor and critic parameter trainable; freezing is evaluated separately as an ablation.

The original routing environment was rebuilt with Gymnasium and Stable-Baselines3. The experimental framework adds deterministic run identities, paired random seeds, fixed evaluation streams, a fixed adaptation-analysis plan with a documented protocol history, and explicit records of deviations from the legacy implementation.

The study addresses four questions:

- Does source-domain pretraining improve target-domain performance?
- Does freezing pretrained decision layers help during adaptation?
- Does PBRS improve learning or final routing performance?
- Can the observed effects be explained by the total training budget rather than transfer learning?

## Routing Task and Method

The routing task models a network in which every node repeatedly selects one of three candidate paths for its own traffic. The objective is to reduce packet loss and delay while traffic conditions change. Routers use finite-capacity M/M/1/K queues; link bandwidth is not imposed as an explicit hard constraint, and propagation, transmission, and processing delays are not simulated. Flow rates are perturbed during each episode to represent dynamic demand.

The learning agent uses Proximal Policy Optimization (PPO). A gated recurrent unit (GRU) encodes the node sequence of each candidate path, and the resulting path representations are combined with local node state before being passed to independent actor and critic networks. A node-fanout mapping presents the observations of all routers as synchronized learning samples with a shared network-level reward.

This GRU path representation is an intentional change from the legacy 287-dimensional observation design. The change and its experimental consequences are documented in [DEVIATIONS.md](DEVIATIONS.md).

## Experimental Design

### Source and Target Scenarios

The source policy is pretrained on the Claranet topology at an average traffic rate of 500 for 15,000 episodes. Target-domain adaptation uses 1,000 episodes. The equal-budget control trains from scratch for 16,000 episodes so that its total budget matches pretraining plus adaptation.

Ten paired seeds are used. All transfer configurations associated with a given seed load the source checkpoint produced by that same seed, preventing source-model variation from being mixed with the component comparison.

| Target scenario | Topology | Average rate | Purpose |
|---|---|---:|---|
| Link failure | Claranet\_failure | 500 | Test robustness when links are removed from the source topology |
| Cross-topology transfer | Abilene | 500 | Test generalization to an 11-node topology |
| Cross-topology transfer | NSFCNET | 500 | Test generalization to a smaller 9-node topology |
| Increased traffic | Claranet | 1500 | Test adaptation when the source-topology traffic is tripled |

### Learning Configurations

The three main factors are pretraining, decision-layer freezing, and PBRS. Additional controls isolate zero-shot transfer, random freezing, and total training budget.

| Configuration | Description |
|---|---|
| A0: Zero-shot | Evaluate the pretrained source policy without target-domain training |
| A1: Scratch | Train in the target domain without pretraining or PBRS |
| A1b: Equal budget | Train from scratch with the combined pretraining and adaptation budget |
| A2: Pretraining + fine-tuning | Adapt the pretrained policy without freezing or PBRS |
| A3: Pretraining + freezing | Adapt with the decision trunk frozen and without PBRS |
| A3b: Random freezing | Apply the same freezing pattern to a randomly initialized policy |
| A4: PBRS only | Train from scratch with PBRS |
| A5: Pretraining + PBRS | Fine-tune the pretrained policy with PBRS and without freezing |
| A6: Full three-factor combination | Combine pretraining, decision-layer freezing, and PBRS |

A0 is evaluated without target training and is not counted among the target-trained learning configurations.

Freezing applies only to the multilayer-perceptron decision trunk. The GRU feature extractor remains trainable, the output head is reinitialized and trained, and the critic remains trainable. The freezing results therefore concern the decision layers rather than the complete representation.

### Non-Learning Baselines

Four deterministic routing baselines provide non-learning reference points:

- **OSPF** selects the first path from candidate paths ranked by the sum of inverse link service rates.
- **ECMP** divides traffic equally among the available candidate paths, without equal-cost testing.
- **LB** divides traffic inversely according to current load.
- **RR** rotates among candidate paths in round-robin order.

## Evaluation and Statistical Analysis

Training and evaluation are separated. Evaluation uses greedy actions, ten fixed evaluation episodes, and the same evaluation streams across configurations and scenarios. PBRS is used only where specified during training; evaluation rewards do not include the shaping term.

The final v5 protocol evaluates each learning configuration every 100 training episodes and retains a checkpoint whenever performance reaches a new maximum. The primary final result is obtained from this best checkpoint, while the last-checkpoint result is also retained for comparison with the earlier v4 protocol. The best-checkpoint rule was fixed before the v5 rerun, after an earlier terminal-score run revealed late-training decay, and it was applied uniformly to all learning configurations.

The fixed adaptation measures are time to cross a scenario-specific performance threshold, area under the learning curve over the first 500 episodes, and asymptotic performance over the final five evaluation points. Component comparisons use paired t-tests and Wilcoxon signed-rank tests, with Holm correction for multiple comparisons. Independent final tests (a disjoint test stream of seeds 20000–20009), terminal references, confidence intervals, and effect sizes were added for the journal revision and are reported separately from the original adaptation analysis.

## Summary of Findings

The findings below separate the original selection-stream results from the independent-test results added during the journal revision. Checkpoint selection on the selection stream gives learning configurations a selection advantage that the independent stream does not.

On the fixed selection stream (seeds 10000–10009):

- All eight target-trained learning configurations exceeded OSPF in the link-failure scenario. Selected configurations also matched or exceeded OSPF in the other scenarios, but the results do not establish general superiority over OSPF.
- PBRS was the most consistently positive component: nine of twelve selection-stream component comparisons were strictly positive, and the remaining three were within 0.0003 of zero.
- Decision-layer freezing was not beneficial on the selection stream: all eight relevant paired comparisons were negative.
- Pretraining had scenario-dependent effects. It provided a positive signal on Abilene but was generally unfavorable in the increased-traffic scenario.

On the independent test stream (seeds 20000–20009; added during the journal revision):

- The proposed A5 configuration did not exceed OSPF in any of the four scenarios; its mean differences from OSPF are negative throughout.
- PBRS contrasts split six positive and six negative, instead of the nine positive contrasts on the selection stream.
- The joint freezing configuration has three positive and five negative independent-test contrasts, instead of eight negative selection-stream contrasts.
- Both pretraining contrasts are negative in all four independent-test scenarios.
- None of the 28 paired arm-versus-scratch comparisons survived Holm correction. The experiment therefore does not support claims of statistically significant improvement at the available sample size.

On both streams, best-checkpoint evaluation reduced the effect of late-training decline on reported final performance without changing the underlying training trajectories.

These findings must be interpreted together with their limitations. Best-checkpoint selection gives learning configurations a selection advantage that deterministic baselines do not receive. Ten paired seeds provide limited statistical power, and the GRU representation means that the final method is not identical to the legacy method. Light-load experiments were excluded from the main comparison after the selected hyperparameters proved unsuitable for that regime.

## Supplementary Analysis

The [`supplementary/`](supplementary/) directory contains the independent-test analysis added for the journal revision:

- [`independent_test_per_seed.csv`](supplementary/independent_test_per_seed.csv): per-seed evaluation results for the independent test stream.
- [`paired_28_independent.csv`](supplementary/paired_28_independent.csv): all 28 arm-versus-scratch comparisons with confidence intervals, effect sizes, paired t-test and Wilcoxon results, and Holm-adjusted p-values.
- [`a5_vs_ospf_independent.csv`](supplementary/a5_vs_ospf_independent.csv): A5-versus-OSPF differences by scenario.
- [`a5_budget_zero_independent.csv`](supplementary/a5_budget_zero_independent.csv): A5-versus-A1b and A5-versus-A0 comparisons.
- [`factor_contrasts_a5_independent.csv`](supplementary/factor_contrasts_a5_independent.csv): component and factor contrasts.
- [`score_summary.csv`](supplementary/score_summary.csv) and [`selection_stream_metrics.json`](supplementary/selection_stream_metrics.json): aggregated scores and selection-stream metrics.
- [`run_independent_test.py`](supplementary/run_independent_test.py), [`analyze_independent_test.py`](supplementary/analyze_independent_test.py), [`analyze_a5_supplement.py`](supplementary/analyze_a5_supplement.py), and [`add_wilcoxon_tables.py`](supplementary/add_wilcoxon_tables.py): scripts used to produce and verify the tables.

## Experimental Artifacts and Documentation

- [`src/trl_sb3/`](src/trl_sb3/) contains the environment, policies, training pipeline, evaluation methods, and aggregation code.
- [`config/`](config/) contains the experiment defaults, scenario grids, and fixed evaluation protocol.
- [`topologies/`](topologies/) contains the network topology files used by the experiments.
- [`tests/`](tests/) contains the test suite.
- [`supplementary/`](supplementary/) contains the independent-test data, statistical tables, and analysis scripts.
- [`figures_a6wave/`](figures_a6wave/), [`figures_pilot/`](figures_pilot/), [`figures_smoke/`](figures_smoke/), [`runs_baseline/`](runs_baseline/), [`runs_smoke/`](runs_smoke/), [`runs_smoke2/`](runs_smoke2/), and [`profiling/`](profiling/) contain archived pilot, smoke-test, baseline, and profiling artifacts.
- [`DEVIATIONS.md`](DEVIATIONS.md) records intentional differences from the legacy implementation and earlier experimental protocols.
- [`RUNBOOK.md`](RUNBOOK.md) documents the execution and artifact-management procedures.
- [`PLAN.md`](PLAN.md), [`LEARNINGS.md`](LEARNINGS.md), and [`ISSUES.md`](ISSUES.md) document the experimental plan, lessons learned, and known issues.

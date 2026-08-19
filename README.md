# PPO-TL Routing Experiments

Code for regenerating all experiments of the paper "Efficient Transfer Learning-Enhanced PPO
for Dynamic Network Routing via Selective Freezing and Potential-Based Reward Shaping".

## Environment setup (isolated .venv, CUDA torch)

```powershell
cd experiments
python -m venv .venv
.venv\Scripts\python -m pip install torch --index-url https://download.pytorch.org/whl/cu126
.venv\Scripts\python -m pip install numpy scipy pandas matplotlib pyyaml pytest
# verify GPU
.venv\Scripts\python -c "import torch; print(torch.cuda.is_available())"
```

All commands below assume `.venv\Scripts\python` (or activate `.venv\Scripts\Activate.ps1`).
A CPU-only torch also works — the GPU orchestrator falls back to the vectorized CPU backend.

## Layout

```
config/default.yaml   all hyperparameters (paper values are defaults) and smoke/full schedules
src/topo/             four topologies (Abilene 11/28, CERNET 41/116, Claranet 15/36, NSFCNET 9/20,
                      directed edge counts). Abilene follows Topology Zoo exactly; the other three
                      are structural encodings that match the published node/edge counts.
src/env/              analytical M/M/1/K queueing environment (paper Eqs. 4-9), state/action/reward,
                      traffic modes, link failures, topology switching
src/agents/           PPO (with selective freezing + PBRS), DQN, GRL-PS (simplified: spectral graph
                      embedding pretraining + DQN + transfer)
src/baselines/        OSPF
run/run_all.py        orchestrator, resumable, --mode smoke|full
run/make_figures.py   aggregation, figures (PDF+PNG) and statistical tests
logs/{mode}/          one dir per run: train.log, metrics.csv, manifest.json, eval.json, ckpt
results/{mode}/       figures + summary_grouped.csv + stats_tests.csv
tests/                unit tests (queueing formulas vs brute force, PBRS telescoping, freezing)
```

## Methods

| key | meaning |
|---|---|
| ospf | shortest-path baseline |
| ppo_vanilla | PPO trained from scratch on target |
| ppo_naive | pretrain + full fine-tuning (no freezing, no shaping) |
| dqn | DQN routing from scratch |
| grlps | simplified GRL-PS: spectral embedding + DQN, transferred |
| ppotl | ours: pretrain + backbone freezing + PBRS |
| ppotl_freeze_only / ppotl_pbrs_only | ablations |
| ppo_local | locally trained PPO reference (S4 only) |

Source domain is always the same topology at 750 pkt/s constant traffic (paper setting).
Source pretraining is plain PPO (no freezing/shaping) shared by all transfer variants.

## Quick start (smoke, ~minutes, CPU)

```
cd experiments
python -m pytest tests -q
python run/run_all.py --mode smoke
python run/make_figures.py --mode smoke
```

## GPU multi-GPU backend (run_gpu.py)

The GPU backend re-implements training on a fully vectorized stack and is the
recommended path for real runs:

- `src/env/vec_env.py` — N environment copies as one batched tensor set on the device;
  analytical M/M/1/K in float64; candidate-path tables precomputed once (Yen k-shortest,
  module-cached) and converted to index tensors; `scatter_add` aggregation.
- `src/agents/batched_nets.py` + `ppo_gpu.py` — S seeds stacked into `[S, ...]` weight
  tensors (L3 ensemble): one broadcast-matmul forward trains all seeds; a single Adam
  over stacked parameters is exactly S independent Adams (elementwise updates). Measured
  ~288x more env-episodes per second than the per-env CPU backend on the same machine.
- `run/run_gpu.py` — auto-detects GPUs (name/free VRAM), picks `n_vec_envs`
  (128/256/512), groups PPO-family jobs into ensemble units, shards units across
  (GPU x proc) workers by cost via `torch.multiprocessing.spawn`, pretrains are shared
  and protected by a filesystem lock, and everything resumes from DONE markers.

```
python run/run_gpu.py --mode full                     # all GPUs, ensemble on
python run/run_gpu.py --mode full --gpus 0,1 --procs-per-gpu 2
python run/run_gpu.py --mode full --no-ensemble       # per-seed units
python run/run_gpu.py --mode full --n-vec-envs 512
python run/make_figures.py --mode full
```

Monitoring utilization on the training machine:

```
nvidia-smi dmon -s u -c 0        # continuous SM/mem utilization
```

Notes:
- A CPU-only torch build falls back to `device=cpu` with the *vectorized* backend
  (useful for logic checks on laptops); install a CUDA torch build to use real GPUs.
- Small MLPs cannot reach 100% SM utilization by construction; throughput is maximized
  by vectorized envs + S-seed ensembles + `--procs-per-gpu` oversubscription.
- GPU semantics deviations (documented, all verified by tests):
  standard PPO minibatch advantage (computed once pre-update) instead of the CPU
  backend's per-sample sequential updates; ensemble runs share the topology/failure
  realization across the S seeds' env copies (traffic RNG stays per copy); under link
  failures the candidate table is approximated by shortest-path fallback.
- Checkpoints carry both stacked and CPU-named per-seed state dicts; run dirs and
  DONE markers are shared with the CPU backend, so `make_figures.py` aggregates either.

## Full run (training machine)

```
pip install -r requirements.txt
python run/run_all.py --mode full            # resumable: rerun skips DONE dirs
python run/make_figures.py --mode full
```

Sharding across machines/seeds:

```
python run/run_all.py --mode full --seeds 0,1,2,3,4
python run/run_all.py --mode full --seeds 5,6,7,8,9 --scenarios S1,S2
python run/run_all.py --mode full --topo cernet
```

Parallel shards on one machine (one process per core), PowerShell:

```powershell
0..9 | ForEach-Object { Start-Process -NoNewWindow python "run\run_all.py --mode full --seeds $_" }
```

or cmd:

```bat
for /L %s in (0,1,9) do start /min python run/run_all.py --mode full --seeds %s
```

Runs are resumable and idempotent (deterministic run ids + DONE markers), so shards never
collide. Estimated cost (single core, measured from smoke scaling): ~2400 runs, roughly
60-90 CPU-hours total; with 10 parallel seed-shards this is 6-9 h wall clock. Pretrain
checkpoints are shared across scenarios and reused automatically.

Copy back `logs/full/` (only `metrics.csv`, `eval.json`, `manifest.json` matter for figures;
delete `ckpt*.pt` if too large) and run `make_figures.py --mode full`.

## Experiment matrix (full mode)

- S1 constant traffic: 4 topologies x 4 rates x 6 methods x 10 seeds
- S2 varying traffic: 4 topologies x 6 methods x 10 seeds, eval per rate
- S3 link failures: ratios 0.1/0.2/0.3 with mid-episode failure events
- S4 transfer: pretrain on CERNET, zero-shot + adapt on other three
- A  ablation: freeze-only / PBRS-only / naive / vanilla / full
- H  sensitivity: k in {2,4,8,16}, freeze fraction {0,.25,.5,.75,1}, zeta1, alpha1

Statistics: mean +/- 95% CI over seeds, Welch t-test and Wilcoxon with Holm correction
(results/stats_tests.csv, reference = ppotl).

## Notes / deviations from the paper

- Queueing metrics are computed analytically from the M/M/1/K steady state (paper formulas),
  not by packet-level simulation.
- Background flows (20) are routed by shortest path; the agent flow is routed by the policy.
- The hop-count feature is normalized per source-destination pair by the longest candidate
  path (paper's global normalizer is ambiguous for k > 1 candidates).
- GRL-PS is a simplified reimplementation (spectral embedding instead of DeepWalk-style
  pretraining); this is stated as a limitation, consistent with the paper's argument that
  exact GRL-PS is not directly comparable.
- Paper learning rates (actor 6e-6, critic 6e-7) are slow; if convergence looks insufficient
  in full runs, override `paper_defaults.actor_lr/critic_lr` in config and rerun.

# AGENTS.md — trl_sb3 experiments package (TRL routing ablation)

This directory is self-contained (no files outside it are required). Full
pilot / main-table runs happen on the GPU machine per `RUNBOOK.md`.

## Reading order

1. `RUNBOOK.md` — execution manual: env setup, pilot sweep, power analysis, main table.
2. `PLAN.md` — F1-F13 verified legacy semantics + design decisions; the authority
   when cross-checking implementation behavior.
3. `LEARNINGS.md` — API contracts and pitfalls (version matrix, SB3/Gymnasium quirks).
4. `ISSUES.md` — known quirks and open items.

## Hard rules (non-negotiable)

1. Paper hyperparameters live ONLY in `config/default.yaml` (`paper_defaults` section);
   never hardcode them in code.
2. `run_id` must stay deterministic (no timestamps): changing a factor changes identity.
   Resume / DONE-skip / aggregation all depend on it.
3. Artifact contract per run dir: `metrics.csv` / `eval.json` / `manifest.json` /
   `DONE|FAILED` — never break it; the aggregation side scans by this contract.
4. All dependencies are pinned `==` in `requirements.lock`: change the lock first, then
   install. torch is `2.13.0+cu126` (local version) — must install with
   `--index-url https://download.pytorch.org/whl/cu126` (wheel is not on PyPI).
5. Always use basedpyright to validate the correction of code and use ruff to format after edit 
6. Always use tmux to dispatch the running process

## Environments

- Windows workstation: control plane, editing, tests, and Git operations.
- Remote GPU server: Ubuntu 20.04, bash, SSH alias `806-net-2`.
- Remote project root: `/mnt/data/gsy/ppotl-routing`.
- Remote interpreter: `/mnt/data/gsy/ppotl-routing/.venv/bin/python` (Python 3.13).
- Use the remote project `.venv`; do not use the system `python`.

## Remote GPU server

Training, sweeps, and full evaluations run on the remote server, not on the
Windows workstation. The SSH alias `806-net-2` is defined in the operator's
local `~/.ssh/config`. Keep the host, user, port, and key details there: never
commit them, a private key, a token, or an `.env` file to this repository.

Connect and run a one-off command:

```powershell
ssh 806-net-2
ssh 806-net-2 "cd /mnt/data/gsy/ppotl-routing && .venv/bin/python -m pytest tests -q"
```

Long jobs must run in `tmux` (hard rule 6). Do not start them in a transient SSH
shell:

```powershell
ssh 806-net-2 "cd /mnt/data/gsy/ppotl-routing && mkdir -p logs && tmux new-session -d -s ppotl-run '.venv/bin/python -m trl_sb3.run sweep --grid config/grid_main.yaml --device cuda > logs/ppotl-run.log 2>&1'"
```

Inspect, attach, or stop the session:

```powershell
ssh 806-net-2 "tmux ls; tail -n 40 /mnt/data/gsy/ppotl-routing/logs/ppotl-run.log"
ssh -t 806-net-2 "tmux attach -t ppotl-run"   # detach with Ctrl-b, then d
ssh 806-net-2 "tmux kill-session -t ppotl-run"
```

Before launching, check the machine and choose a free GPU. The server has four
V100S GPUs; `CUDA_VISIBLE_DEVICES` is set per process:

```powershell
ssh 806-net-2 "nvidia-smi; df -h /mnt/data"
```

Use `--device cpu` for CPU-bound sweeps when appropriate; see `RUNBOOK.md`
§6.1 for concurrency guidance.

### Transferring files

Use `scp` from Windows. The installed Windows `rsync` is not reliable with this
server, and it treats a `C:` path as a remote host. Transfer only the files
needed, not the whole `runs/` tree.

```powershell
# local -> remote
scp .\AGENTS.md 806-net-2:/mnt/data/gsy/ppotl-routing/AGENTS.md
scp -r .\supplementary 806-net-2:/mnt/data/gsy/ppotl-routing/

# remote -> local
scp 806-net-2:/mnt/data/gsy/ppotl-routing/revision_independent_test/results/lf.log .\lf.log
scp -r 806-net-2:/mnt/data/gsy/ppotl-routing/revision_independent_test/results .\remote-results\
```

For a large directory, archive it remotely first:

```powershell
ssh 806-net-2 "cd /mnt/data/gsy/ppotl-routing && tar -czf /tmp/ppotl-results.tgz revision_independent_test/results"
scp 806-net-2:/tmp/ppotl-results.tgz .\ppotl-results.tgz
```

Result locations on the server: `runs/`, `ckpts/`,
`revision_independent_test/results/`, and logs under `logs/`.

### Code and artifact sync

Edit and test on the Windows checkout, commit and push to the project's GitHub
repository, then update the server checkout with `git pull --ff-only` only when
the server's tracked tree is clean. Do not maintain two divergent copies of a
tracked source file. Large run artifacts stay untracked and move via `scp`.

## Common commands

Run these on the GPU server from `/mnt/data/gsy/ppotl-routing`:

```bash
.venv/bin/python -m pytest tests -q
.venv/bin/python -m trl_sb3.run sweep --grid config/grid_main.yaml --device cuda --dry-run
.venv/bin/python -m trl_sb3.run sweep --grid config/grid_main.yaml --device cuda
.venv/bin/python -m trl_sb3.run make_figures --runs runs --out figures_main
```

Pilot / smoke grids (`grid_pilot.yaml`, `grid_smoke.yaml`) and the full sequence:
see `RUNBOOK.md`.

## Testing

After changing environment semantics or method behavior, run
`.venv/bin/python -m pytest tests -q` on the remote project root — 155+ green
is the baseline; investigate any red before proceeding.

## Known pitfalls

- Disable sleep before long runs (admin PowerShell): `powercfg /change standby-timeout-ac 0`.
- Console CJK garbling is a display issue only; CJK paths resolve fine.
- CLOSED (2026-09-01, Phase-3 GATE-2): `theta` in `config/metrics_prereg.yaml` stays
  `null` as the FINAL preregistered form (per-scenario derivation via `theta_rule`);
  do NOT backfill a global float. Eval-episode dual-track CLOSED 2026-08-30 (D1=YES,
  unified at 10, OSPF row regenerated).

## Protocol state (2026-09-09)

Method = GRU path-feature version (78-dim obs, DEVIATIONS #15) + Plan-A hyperparams
(#16); source domain = Claranet@500 × 15000 episodes (#17, D6/D7/D8 adjudicated);
prereg = metrics_prereg v4 (protection set: Claranet_failure@500 / Abilene@500 /
NFSCNET@500 / Claranet@1500). Old 287-dim-protocol prebuilt grids are deleted —
regenerate grids from `config/default.yaml` only.

## Baseline rows (pre-completed 2026-08-30)

OSPF/ECMP/LB/RR for the pilot scenario (Abilene@500) are DONE in `runs/` via
`config/grid_baseline.yaml`; sweeps DONE-skip these 4 rows — do NOT delete
`runs/`. Rows are deterministic (seed=0, no budget/seed factor; finals
bit-match the `runs_smoke/` evidence). Full record + numbers: `RUNBOOK.md` §3.
On a fresh machine, either re-run the baseline grid (~20 s, identical output)
or let the pilot sweep recreate the rows.

## Layout (one-liner)

`src/trl_sb3/{env,policy,train,eval,common,run}` · `config/` (6 yamls) ·
`topologies/` (12 GML) · `tests/` · `runs*/ ckpts*/ figures*/` artifacts.

## Comunicate with User

You should always speek in Simplified CN

Your text output is what the user reads; they usually can't see your thinking or the raw tool results. Write it for a teammate who stepped away and is catching up, not for a log file: they don't know the codenames or shorthand you created along the way, and they didn't watch your process unfold. Before your first tool call, say in a sentence what you're about to do; while working, give brief updates when you find something load-bearing or change direction.

Text you write between tool calls may not be shown to the user. Everything the user needs from this turn — answers, summaries, findings, conclusions, deliverables — must be in the final text message of your turn, with no tool calls after it. Keep text between tool calls to brief status notes. If something important appeared only mid-turn or in your thinking, restate it in that final message.

Lead with the outcome. Your first sentence after finishing should answer "what happened" or "what did you find" — the thing the user would ask for if they said "just give me the TLDR." Supporting detail and reasoning come after, for readers who want them.

Being readable and being concise are different things, and readable matters more. If the user has to reread your summary or ask you to explain, any time saved by brevity is gone. The way to keep output short is to be selective about what you include (drop details that don't change what the reader would do next), not to compress the writing into fragments, abbreviations, arrow chains like `A → B → fails`, or jargon. What you do include, write in complete sentences with the technical terms spelled out. Don't make the reader cross-reference labels or numbering you invented earlier; say what you mean in place.

Match the response to the question: a simple question gets a direct answer in prose, not headers and sections. Use tables only for short enumerable facts, with explanations in the surrounding prose rather than the cells. Calibrate to the user — a bit tighter for an expert, more explanatory for someone newer.

Write code that reads like the surrounding code: match its comment density, naming, and idiom.
Only write a code comment to state a constraint the code itself can't show — never to say where it came from, what the next line does, or why your change is correct; that's you talking to the reviewer, not the next reader, and it's noise the moment the PR merges.

For actions that are hard to reverse or outward-facing, confirm first unless durably authorized or explicitly told to proceed without asking; approval in one context doesn't extend to the next. Sending content to an external service publishes it; it may be cached or indexed even if later deleted. Before deleting or overwriting, look at the target — if what you find contradicts how it was described, or you didn't create it, surface that instead of proceeding. Report outcomes faithfully: if tests fail, say so with the output; if a step was skipped, say that; when something is done and verified, state it plainly without hedging.


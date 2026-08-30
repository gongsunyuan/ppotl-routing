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

1. Python is ALWAYS `.venv\Scripts\python.exe` — the global python is CPU-only torch.
2. Paper hyperparameters live ONLY in `config/default.yaml` (`paper_defaults` section);
   never hardcode them in code.
3. `run_id` must stay deterministic (no timestamps): changing a factor changes identity.
   Resume / DONE-skip / aggregation all depend on it.
4. Artifact contract per run dir: `metrics.csv` / `eval.json` / `manifest.json` /
   `DONE|FAILED` — never break it; the aggregation side scans by this contract.
5. All dependencies are pinned `==` in `requirements.lock`: change the lock first, then
   install. torch is `2.13.0+cu126` (local version) — must install with
   `--index-url https://download.pytorch.org/whl/cu126` (wheel is not on PyPI).

## Common commands

```powershell
& .venv\Scripts\python.exe -m pytest tests -q                                              # tests
& .venv\Scripts\python.exe -m trl_sb3.run sweep --grid config/grid_main.yaml --device cuda --dry-run
& .venv\Scripts\python.exe -m trl_sb3.run sweep --grid config/grid_main.yaml --device cuda  # main table
& .venv\Scripts\python.exe -m trl_sb3.run make_figures --runs runs --out figures_main
```

Pilot / smoke grids (`grid_pilot.yaml`, `grid_smoke.yaml`) and the full sequence:
see `RUNBOOK.md`.

## Testing

After changing environment semantics or method behavior, run
`& .venv\Scripts\python.exe -m pytest tests -q` — 155+ green is the baseline;
investigate any red before proceeding.

## Known pitfalls

- Disable sleep before long runs (admin PowerShell): `powercfg /change standby-timeout-ac 0`.
- Console CJK garbling is a display issue only; CJK paths resolve fine.
- OPEN ITEMS (see `ISSUES.md`): eval-episode dual-track (5 vs 10) and the not-yet
  backfilled `theta` in `config/metrics_prereg.yaml` — both must be adjudicated
  before building the main table.

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

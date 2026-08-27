# experiments — TRL routing ablation package

Clean-room rebuild of the TRL routing experiments on Stable-Baselines3.
Old code, old `.venv` and old git history were purged on purpose; this tree is
the single source of truth from M-0 onward.

## Environment

All commands must use this interpreter (global python is CPU-only torch):

```powershell
$py = .venv\Scripts\python.exe
```

Direct dependency versions are pinned in `requirements.lock` (`==` only).
Reproduce the venv with the command sequence in that file's header comment.

## Package

`trl_sb3` is a src-layout package installed editable into the venv:

```powershell
uv pip install -e . --python .venv\Scripts\python.exe --no-deps
```

`pyproject.toml` mirrors the 9 `==` pins from `requirements.lock` (torch is
`2.13.0+cu126`, whose wheel lives at `https://download.pytorch.org/whl/cu126`,
not PyPI). Always install with `--no-deps`: dependency management belongs to
`requirements.lock`, not the build backend.

Paper hyperparameters live only in `config/default.yaml` (`paper_defaults`
section) — never hardcoded. `trl_sb3.common.config.load_config()` resolves that
file from `__file__`, so it works from any cwd. Run identities are deterministic
(no timestamps): see the factor-key convention in
`trl_sb3/common/logging_utils.py`.

Every intentional deviation from the legacy code and the paper is recorded in
`DEVIATIONS.md` with evidence and status.

## Layout

- `src/trl_sb3/` — package, all modules landed: `env/` `policy/` `train/`
  `eval/` `common/` `run/` (`common/` holds config + logging primitives).
- `config/` — 5 yamls: `default.yaml` (paper_defaults), `grid_main.yaml`,
  `grid_pilot.yaml`, `grid_smoke.yaml`, `metrics_prereg.yaml`.
- `topologies/` — 12 GML topology files (6 networks x normal/_failure variants):
  Abilene, CERNET, Claranet, Gridnet, NFSCNET, NSF. Copied verbatim from the
  legacy `TRL_Routing/code` directory.
- `tests/` — pytest suite (`python -m pytest tests -q`).
- `requirements.lock` — exact direct-dependency pins (torch is +cu126).
- `runs_smoke/` `figures_smoke/` — 2026-08-26 smoke-run evidence (34/34 DONE);
  do not delete, use as reference.
- `runs/` `ckpts/` — run artifacts (gitignored, created by the runner).

## Entrypoints

```powershell
& $py -m trl_sb3                                   # package self-check
& $py -m trl_sb3.run sweep --grid config/grid_main.yaml
& $py -m trl_sb3.run sweep --grid config/grid_main.yaml --dry-run
& $py -m trl_sb3.run make_figures --runs runs --out figures
& $py -m pytest tests -q
```

## AI context

- `PLAN.md` — F1-F13 verified legacy semantics + design decisions (authority
  for cross-checking implementation behavior).
- `LEARNINGS.md` — API contracts and pitfalls (version matrix, SB3/Gymnasium quirks).
- `ISSUES.md` — known quirks and open items (eval-episode dual-track, theta backfill).
- `RUNBOOK.md` — execution manual for the GPU machine (env setup, pilot,
  power analysis, main table).

## Windows notes

- Disable sleep before long training runs (admin PowerShell):
  `powercfg /change standby-timeout-ac 0`
- Chinese characters garbling in the console is a display issue only; paths
  with CJK names still resolve correctly.

# Codex task (follow-up): FINISH the migration and remove the old scaffolding

Your previous pass created the new structure (`sparsepano/`, `pipelines/`, `configs/`, `tests/`,
`pyproject.toml`, `MIGRATION.md`) but **left the entire old structure in place**, so the repo now has
BOTH and is messier than before. Your own `MIGRATION.md` lists 8 phases; you completed ~1–4. This task
is: **complete phases 5–8 and delete the now-redundant old files.** You now HAVE delete permission — use
it, but only under the strict ordering below so the validated numbers never break.

## The core problem to fix
The new code is still a thin bridge over the old code:
- `sparsepano/datasets/zind.py` does `from the former src package import zind` (depends on `former-former-src/`).
- `pipelines/run.py` does `from the former experiments package import exp28_gtfree_connectivity` (depends on `experiments/`).

So `former-former-src/` and `experiments/` are still load-bearing. **You must move the real implementations into the
package and repoint the bridges BEFORE deleting anything.** Migrate, don't wrap.

## Strict ordering (per module — never delete before the replacement is proven)
For each old module/experiment:
1. **Move the real implementation** into its topical home in `sparsepano/…` (or a `pipelines/…` runner),
   per the mapping in `MIGRATION.md`. The code lives in the new file — not a wrapper that imports the old one.
2. **Repoint all importers** (pipelines, tests, benchmarks, other modules) to the new location.
   Remove every `from the former src package import …` and `from the former experiments package import …`.
3. **Run the regression + smoke tests** (`tests/`): connectivity assign-AP must still be ≈0.913 (GT doors)
   / ≈0.842 (detected). If a number moves, STOP and leave `TODO(codex):` — do not delete.
4. **Only then delete** the old file. Git history preserves it; superseded one-offs go to `legacy/` instead
   of deletion (see mapping).

Do this in small commits (one topical group per commit): geometry, doors, pose, gs, viz, then the
connectivity pipeline (fold `exp28`/`exp12`/`exp23` logic into `pipelines/connectivity.py` +
`sparsepano/metrics/connectivity.py` so `run.py` no longer imports `experiments`), then pose/layout, then
dataset builders.

## Explicit deletions (now authorised)
Once the above proves the pipeline runs WITHOUT the old imports:
- **Delete `former-former-src/`** entirely (all implementations migrated into `sparsepano/…`; git history keeps it).
- **Delete `experiments/`** from the top level: migrated ones → `pipelines/`; superseded one-offs
  (exp01–08, 11, 13, 15–17, 19–22, 24–26 per `MIGRATION.md`) → `legacy/experiments/` with a one-line note
  each in `legacy/README.md`. After this, NO `experiments/` folder remains at top level.
- **Move `overlap_probe/` → `benchmarks/overlap_probe/`** unchanged; fix only external import paths.
- **Delete the deprecated doc stubs** (they contain only a `[Deprecated — consolidated]` redirect):
  `MeetingSummary.md`, `ProposalDraft.md`, and `ContextMDs/{NextStage,PaperV2Plan,OpenQuestions,PaperNotes,Direction_2026-07}.md`.
  (These were flagged "user-edited, do not touch" in your MIGRATION.md — that was stale; they are now
  confirmed deprecated stubs, safe to delete.)
- **Move the 4 canonical docs → `docs/`**: `ContextMDs/{ProjectOverview,ResearchLog,Roadmap,RelatedWork}.md`.
  Also move `CLUSTER.md`, `CODEX_REFACTOR_PROMPT.md`, `CODEX_CLEANUP_PROMPT.md`, and `MIGRATION.md` into
  `docs/`. Then remove the now-empty `ContextMDs/`.
- **Delete root `config.py`** once everything imports `sparsepano.config` (repoint first, then delete).
- **Move weights → `weights/`** (`best.pt`, `best_hardneg.pt`, `door_encoder.pt`) and gitignore that dir
  (do not delete the files). Update any code/paths that referenced them at root.
- **Gitignore + leave in place** (do not delete data): `data_doorpairs/`, `data_floors/`, `runs/`,
  `results/`, `logs/`, `torchhub.tgz`, `__pycache__/`, `.venv-gs/`, editor dirs. Ensure `.gitignore`
  covers them so they stop cluttering `git status`.

## Keep at root (do NOT move/delete)
`AGENTS.md`, `README.md` (rewrite/update), `TODO.md`, `NEW_CHAT_PROMPT.md`, `pyproject.toml`,
`requirements.txt`, `.gitignore`, `.gitattributes`. (`TODO.md` and `NEW_CHAT_PROMPT.md` were recently
user-edited and are current — you may update paths inside them if docs moved, but keep their content.)

## Target end-state top level (this is the acceptance check)
```
sparsepano/  pipelines/  benchmarks/  configs/  scripts/  docs/  tests/  legacy/  weights/
AGENTS.md  README.md  TODO.md  NEW_CHAT_PROMPT.md  pyproject.toml  requirements.txt  .gitignore  .gitattributes
(gitignored, present but not tracked: results/ runs/ data_*/ logs/ __pycache__/ *.tgz .venv-gs/ editor dirs)
```
NO `former-former-src/`, NO `experiments/`, NO top-level `overlap_probe/`, NO `config.py` at root, NO `ContextMDs/`,
NO `[Deprecated]` stub files, NO loose weights at root.

## Done criteria
1. `pip install -e .` works; `python -m pipelines.run --dataset zind --stage connectivity --doors gt`
   reproduces ≈0.913 and `--doors detected` ≈0.842, with **zero imports from `src` or `experiments`**
   (grep the tree to confirm).
2. `pytest` (fast selftests + the slow regression, documented) passes.
3. `git status` is clean of generated clutter; the top level matches the target above.
4. Update `MIGRATION.md` to mark phases 5–8 done and record every deletion/move; update `README.md`
   quickstart to the new layout.

# Reproducibility

The starter CLI can run entirely on synthetic fixtures after installation; its package has no runtime dependencies. The full unit suite now includes M2 raster/vector fixtures and needs `requirements/m2-pipeline.lock`. The build backend is pinned in `pyproject.toml`. The M0–M2 local environment uses Python 3.12. CI runs the full M2 suite with the lock on Python 3.12 and checks the standard-library core on Python 3.11; NumPy 2.5.3 and Rasterio 1.5.2 in the M2 lock require Python 3.12 or newer. A training/serving lock will be selected only when those stages are feasible; no pretrained weights are included.

Create the scaffold environment with Python 3.12, then run `python -m venv .venv` and `.\.venv\Scripts\python.exe -m pip install -e .` on Windows. The status and synthetic demo commands need no scientific packages. For the full tests and local M2 source pipeline, install `.\.venv\Scripts\python.exe -m pip install -r requirements/m2-pipeline.lock`. Run `.\.venv\Scripts\python.exe scripts/build_m2_pilot.py` only after acquiring the ignored pilot files. The M2 lock is not a training environment.

## Local data and run conventions

- Keep downloaded source bytes immutable under `data/raw/`, with acquisition metadata and checksums. Do not put raw data in Git.
- Put canonical derivatives under `data/processed/`; use `runs/<UTC timestamp>_<short experiment name>/` for run configuration, logs, status, metrics and hashes. Keep large model files under `artifacts/` or `models/`, all ignored by Git.
- Record source IDs, acquisition times, split/config/code revision, environment versions, seed, model revision and weight hash, output references, exclusions and run status. A failed or interrupted run retains its ID and last valid checkpoint.
- Publish only reviewed aggregate results and attributed media under `docs/results/` and `docs/assets/`. Keep full reports under ignored roots. Use relative logical paths in portable manifests rather than personal machine paths.
- The M1 pilot was capped at 250 MiB and actually transferred 26,314,140 bytes across the retained source inventory. Recheck free space and object sizes before any larger transfer; full archives and pretrained weights require a separate capacity decision.

Every real experiment needs immutable source manifests, group-split hashes, preprocessing/config revision, code revision, environment lock, seeds, exact model revision and weights hash, input/output hashes and a complete run status. Keep raw data and large artefacts outside Git, with documented logical paths and regeneration commands.

M2 provides the first complete pilot example: per-run outputs in `data/processed/m2_runs/<run-id>/`, diagnostics in `reports/m2/<run-id>/`, and status in `runs/<run-id>/`. The build prints the ID. `scripts/verify_m2_run.py <run-id>` checks the completed manifest, each used source hash and every derived output hash. Repeating an unchanged build should leave the manifest hash identical. For visual QA, open `reports/m2/<run-id>/qa_Spain_7370579.png` and `qa_Bolivia_103757.png`; compare the labels, sensor nodata and Spain trace/road overlay with tile metadata and vector audit. Gray label pixels and black sensor pixels are unknown or nodata, while interpreted road damage is not closure evidence.

Distinguish a cached-result rebuild from fresh model training. A clean clone must run the synthetic smoke mode, while real-data results may need licensed downloads, credentials, time and GPU resources. Report these requirements explicitly. Do not promise byte-identical neural training across hardware; define tolerances and record deterministic settings.

Until M8, the public repository is work in progress and has no real-data performance claims. CI initially verifies only the actual lightweight code. Extend checks as raster/model/API functionality is implemented.

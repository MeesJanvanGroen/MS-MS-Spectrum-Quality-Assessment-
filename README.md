# MS/MS Chemical Plausibility Quality Assessment - Delivery Package

This folder contains the curated files for sharing the final project workflow without the exploratory notebooks.

Folder structure:

- `docs/` contains short documentation files describing the workflow, data, and software environment.
- `workflow_scripts/` contains the main Python scripts used for matched-subset model building and final CNL scoring.
- `models/` contains the saved CatBoost model files.
- `metrics/` contains the parameter files and compact validation/test metric tables.
- `data_description/` contains the starting data descriptions, the CNL and m/z grid definitions, and the matched spectrum ID files used for the subset comparison.

Recommended order for reading:

1. Read `docs/WORKFLOW_OVERVIEW.md`.
2. Read `docs/DATA_DESCRIPTION.md`.
3. Read `docs/SOFTWARE_ENVIRONMENT.md`.
4. Use the files in `workflow_scripts/`, `models/`, `metrics/`, and `data_description/` as needed for reproduction or inspection.

Included workflow scope:

- final CNL CatBoost model
- matched CNL versus m/z comparison
- saved model files
- compact parameter and metric outputs
- starting data definitions for the final feature spaces and matched subsets

Not included:

- exploratory notebooks
- large raw data files
- intermediate temporary files

These were intentionally excluded to keep the handoff folder compact and focused on the final workflow.

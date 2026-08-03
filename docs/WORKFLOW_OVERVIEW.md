# Workflow Overview

The main project workflow consisted of:

1. preprocessing harmonized MS/MS spectra in the TIC-based positive-ion workflow
2. converting fragment-ion spectra into cumulative neutral loss (CNL) space
3. constructing queried-bin classification rows for masked peak versus absent-bin prediction
4. training a CatBoost classifier to predict whether a queried bin corresponded to a true masked peak
5. aggregating queried-bin predictions into spectrum-level scores for chemical plausibility assessment
6. benchmarking the CNL-based workflow against an external handcrafted-feature Random Forest model
7. comparing a matched CNL representation with a direct fragment m/z representation on the same subset of spectra

Files most relevant for the final workflow:

- `workflow_scripts/catboost_classification_tic.py`
  Purpose: main CNL CatBoost classification workflow script
- `workflow_scripts/score_full_cnl_test.py`
  Purpose: score the saved final CNL model on the full test split
- `workflow_scripts/Build_matched_CNL_mz_subset_by_key.py`
  Purpose: build the matched subset used for direct CNL versus m/z comparison
- `workflow_scripts/catboost_classification_tic_matched_subset.py`
  Purpose: train and evaluate the matched-subset CNL CatBoost model
- `workflow_scripts/catboost_classification_mz_matched_subset.py`
  Purpose: train and evaluate the matched-subset m/z CatBoost model

Saved models:

- `models/TIC_catboost_classification_model.cbm`
  Final full-data CNL CatBoost model
- `models/TIC_catboost_classification_model_matched_80k_10k_10k.cbm`
  Matched-subset CNL CatBoost comparison model
- `models/TIC_catboost_classification_mz_model_matched_80k_10k_10k.cbm`
  Matched-subset m/z CatBoost comparison model

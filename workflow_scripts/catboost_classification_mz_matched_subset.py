from pathlib import Path
from datetime import datetime
import json

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)
from sklearn.model_selection import StratifiedKFold

try:
    from catboost import CatBoostClassifier
except ImportError as exc:
    print("CatBoost is not installed in this Python environment.")
    print("Install it first, for example with: pip install catboost")
    raise SystemExit(1) from exc


WINDOWS_BASE_DIR = Path(r"C:\Users\MeesJ\OneDrive\Documenten\PM\Stage 6 maanden\Data")
LINUX_BASE_DIR = Path("/home/mgroen2/my_model_project/Data")
BASE_DIR = WINDOWS_BASE_DIR if WINDOWS_BASE_DIR.exists() else LINUX_BASE_DIR
PREPARED_DIR = BASE_DIR / "PreparedClassification"
MODEL_DIR = BASE_DIR / "Model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


DATASET_TAG = "matched_80k_10k_10k"
TRAIN_LONG_FILENAME = f"TIC_mz_train_classification_long_df_{DATASET_TAG}_oxygen.pkl"
VAL_LONG_FILENAME = f"TIC_mz_val_classification_long_df_{DATASET_TAG}_oxygen.pkl"
TEST_LONG_FILENAME = f"TIC_mz_test_classification_long_df_{DATASET_TAG}_oxygen.pkl"
GRID_FILENAME = "TIC_mz_classification_grid.csv"

K_FOLDS = 3
MODEL_RANDOM_SEED = 42

CATBOOST_BASE_PARAMS = {
    "iterations": 512,
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": MODEL_RANDOM_SEED,
    "verbose": False,
    "thread_count": -1,
}

HYPERPARAMETER_GRID = [
    {"depth": 8, "learning_rate": 0.1, "l2_leaf_reg": 6},
    {"depth": 8, "learning_rate": 0.2, "l2_leaf_reg": 6},
    {"depth": 10, "learning_rate": 0.1, "l2_leaf_reg": 6},
    {"depth": 10, "learning_rate": 0.1, "l2_leaf_reg": 10},
]


print("Matched-subset absolute m/z CatBoost training configuration:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"TRAIN_LONG_FILENAME: {TRAIN_LONG_FILENAME}")
print(f"VAL_LONG_FILENAME: {VAL_LONG_FILENAME}")
print(f"TEST_LONG_FILENAME: {TEST_LONG_FILENAME}")
print(f"K_FOLDS: {K_FOLDS}")
for key, value in CATBOOST_BASE_PARAMS.items():
    print(f"{key}: {value}")
print("Hyperparameter grid:")
for i, params in enumerate(HYPERPARAMETER_GRID, start=1):
    print(f"  {i}: {params}")
print()


def build_feature_target_matrices(long_df):
    presence_matrix = np.vstack(long_df["BINARY_PRESENCE_VECTOR"].to_numpy()).astype(np.float32)
    mask_matrix = np.vstack(long_df["MASKING_VECTOR"].to_numpy()).astype(np.float32)
    precursor_array = long_df["PRECURSOR_ION_NUMERIC"].to_numpy(dtype=np.float32).reshape(-1, 1)
    queried_index_array = long_df["QUERIED_BIN_INDEX"].to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.hstack([presence_matrix, mask_matrix, precursor_array, queried_index_array]).astype(np.float32)
    y = long_df["TARGET_LABEL"].to_numpy(dtype=np.uint8)
    return X, y


def evaluate_classifier(model, X, y, label):
    y_pred = model.predict(X)
    if isinstance(y_pred, pd.DataFrame):
        y_pred = y_pred.to_numpy()
    y_pred = np.asarray(y_pred).reshape(-1).astype(np.uint8)
    return {
        "DATASET": label,
        "N_QUERIED_VALUES": int(len(y)),
        "ACCURACY": float(accuracy_score(y, y_pred)),
        "BALANCED_ACCURACY": float(balanced_accuracy_score(y, y_pred)),
        "PRECISION": float(precision_score(y, y_pred, zero_division=0)),
        "RECALL": float(recall_score(y, y_pred, zero_division=0)),
        "F1": float(f1_score(y, y_pred, zero_division=0)),
        "POSITIVE_RATE_TRUE": float(np.mean(y)),
        "POSITIVE_RATE_PRED": float(np.mean(y_pred)),
    }


def full_catboost_params(grid_params):
    params = CATBOOST_BASE_PARAMS.copy()
    params.update(grid_params)
    return params


print("Loading matched-subset absolute m/z classification data...")
train_long_df = pd.read_pickle(PREPARED_DIR / TRAIN_LONG_FILENAME)
val_long_df = pd.read_pickle(PREPARED_DIR / VAL_LONG_FILENAME)
test_long_df = pd.read_pickle(PREPARED_DIR / TEST_LONG_FILENAME)
grid_df = pd.read_csv(PREPARED_DIR / GRID_FILENAME)

print(f"Shared model m/z grid size: {len(grid_df)}")
print(f"Train queried-bin rows: {len(train_long_df)}")
print(f"Val queried-bin rows: {len(val_long_df)}")
print(f"Test queried-bin rows: {len(test_long_df)}")
print(f"Train spectra: {train_long_df['SPECTRUM_INDEX'].nunique()}")
print(f"Val spectra: {val_long_df['SPECTRUM_INDEX'].nunique()}")
print(f"Test spectra: {test_long_df['SPECTRUM_INDEX'].nunique()}")

tuning_df = pd.concat([train_long_df, val_long_df], ignore_index=True)

print("Building dense tuning and test matrices...")
X_tune, y_tune = build_feature_target_matrices(tuning_df)
X_test, y_test = build_feature_target_matrices(test_long_df)

print(f"X_tune shape: {X_tune.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Positive label fraction (tune): {y_tune.mean()}")
print(f"Positive label fraction (test): {y_test.mean()}")

print("Tuning CatBoost hyperparameters with stratified K-fold on train+val...")
cv = StratifiedKFold(n_splits=K_FOLDS, shuffle=True, random_state=MODEL_RANDOM_SEED)
cv_rows = []

for param_index, grid_params in enumerate(HYPERPARAMETER_GRID, start=1):
    params = full_catboost_params(grid_params)
    print(f"Hyperparameter set {param_index}/{len(HYPERPARAMETER_GRID)}: {grid_params}")

    for fold_index, (fold_train_idx, fold_val_idx) in enumerate(cv.split(X_tune, y_tune), start=1):
        fold_model = CatBoostClassifier(**params)
        fold_model.fit(
            X_tune[fold_train_idx],
            y_tune[fold_train_idx],
            eval_set=(X_tune[fold_val_idx], y_tune[fold_val_idx]),
            use_best_model=True,
            verbose=False,
        )
        fold_metrics = evaluate_classifier(
            fold_model,
            X_tune[fold_val_idx],
            y_tune[fold_val_idx],
            f"cv_param_{param_index}_fold_{fold_index}",
        )
        cv_row = {
            "PARAM_INDEX": param_index,
            "FOLD": fold_index,
            "BEST_ITERATION": int(fold_model.get_best_iteration() or params["iterations"]),
            **grid_params,
            **fold_metrics,
        }
        cv_rows.append(cv_row)
        print(
            f"  fold {fold_index}: F1={fold_metrics['F1']:.4f}, "
            f"balanced_accuracy={fold_metrics['BALANCED_ACCURACY']:.4f}, "
            f"best_iteration={cv_row['BEST_ITERATION']}"
        )

cv_results_df = pd.DataFrame(cv_rows)
cv_summary_df = (
    cv_results_df
    .groupby("PARAM_INDEX", as_index=False)
    .agg({
        "F1": ["mean", "std"],
        "BALANCED_ACCURACY": ["mean", "std"],
        "ACCURACY": ["mean", "std"],
        "BEST_ITERATION": "mean",
    })
)
cv_summary_df.columns = [
    "_".join(col).strip("_").upper() if isinstance(col, tuple) else col
    for col in cv_summary_df.columns
]

param_table_df = pd.DataFrame([
    {"PARAM_INDEX": i, **params}
    for i, params in enumerate(HYPERPARAMETER_GRID, start=1)
])
cv_summary_df = cv_summary_df.merge(param_table_df, on="PARAM_INDEX", how="left")
cv_summary_df = cv_summary_df.sort_values(
    ["F1_MEAN", "BALANCED_ACCURACY_MEAN"],
    ascending=False,
).reset_index(drop=True)

best_param_index = int(cv_summary_df.loc[0, "PARAM_INDEX"])
best_grid_params = HYPERPARAMETER_GRID[best_param_index - 1]
best_params = full_catboost_params(best_grid_params)

print("Cross-validation summary:")
print(cv_summary_df.to_string(index=False))
print(f"Selected hyperparameters: {best_grid_params}")

print("Training final matched-subset absolute m/z CatBoost classifier on train+val...")
catboost_model = CatBoostClassifier(**best_params)
catboost_model.fit(X_tune, y_tune, verbose=False)
print("Matched-subset absolute m/z CatBoost training finished.")

print("Scoring final classifier on train+val and held-out test data...")
tune_metrics = evaluate_classifier(catboost_model, X_tune, y_tune, "train_val_classification")
test_metrics = evaluate_classifier(catboost_model, X_test, y_test, "test_classification")
metrics_df = pd.DataFrame([tune_metrics, test_metrics])
print(metrics_df.to_string(index=False))

model_path = MODEL_DIR / f"TIC_catboost_classification_mz_model_{DATASET_TAG}.cbm"
metrics_path = MODEL_DIR / f"TIC_catboost_classification_mz_metrics_{DATASET_TAG}.csv"
experiment_log_path = MODEL_DIR / f"TIC_catboost_classification_mz_experiment_log_{DATASET_TAG}.csv"
params_path = MODEL_DIR / f"TIC_catboost_classification_mz_params_{DATASET_TAG}.json"
cv_results_path = MODEL_DIR / f"TIC_catboost_classification_mz_cv_results_{DATASET_TAG}.csv"
cv_summary_path = MODEL_DIR / f"TIC_catboost_classification_mz_cv_summary_{DATASET_TAG}.csv"

catboost_model.save_model(str(model_path))
metrics_df.to_csv(metrics_path, index=False)
cv_results_df.to_csv(cv_results_path, index=False)
cv_summary_df.to_csv(cv_summary_path, index=False)

with open(params_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "DATASET_TAG": DATASET_TAG,
            "TRAIN_LONG_FILENAME": TRAIN_LONG_FILENAME,
            "VAL_LONG_FILENAME": VAL_LONG_FILENAME,
            "TEST_LONG_FILENAME": TEST_LONG_FILENAME,
            "GRID_FILENAME": GRID_FILENAME,
            "K_FOLDS": K_FOLDS,
            "CATBOOST_BASE_PARAMS": CATBOOST_BASE_PARAMS,
            "HYPERPARAMETER_GRID": HYPERPARAMETER_GRID,
            "SELECTED_PARAMS": best_params,
        },
        f,
        indent=2,
    )

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
experiment_row = {
    "RUN_TIMESTAMP": run_timestamp,
    "DATASET_TAG": DATASET_TAG,
    "K_FOLDS": K_FOLDS,
    "TRAIN_ROWS_USED_FOR_TUNING": int(len(train_long_df)),
    "VAL_ROWS_USED_FOR_TUNING": int(len(val_long_df)),
    "TEST_ROWS_USED": int(len(test_long_df)),
    "TRAIN_SPECTRA_USED_FOR_TUNING": int(train_long_df["SPECTRUM_INDEX"].nunique()),
    "VAL_SPECTRA_USED_FOR_TUNING": int(val_long_df["SPECTRUM_INDEX"].nunique()),
    "TEST_SPECTRA_USED": int(test_long_df["SPECTRUM_INDEX"].nunique()),
    "N_FEATURES": int(X_tune.shape[1]),
    "POSITIVE_RATE_TUNE": float(y_tune.mean()),
    "POSITIVE_RATE_TEST": float(y_test.mean()),
    "SELECTED_PARAM_INDEX": best_param_index,
    "CV_F1_MEAN": float(cv_summary_df.loc[0, "F1_MEAN"]),
    "CV_BALANCED_ACCURACY_MEAN": float(cv_summary_df.loc[0, "BALANCED_ACCURACY_MEAN"]),
}

for key, value in best_params.items():
    experiment_row[f"SELECTED_HP_{key.upper()}"] = value

for metric_key, metric_value in tune_metrics.items():
    if metric_key != "DATASET":
        experiment_row[f"TRAIN_VAL_{metric_key}"] = metric_value

for metric_key, metric_value in test_metrics.items():
    if metric_key != "DATASET":
        experiment_row[f"TEST_{metric_key}"] = metric_value

experiment_df = pd.DataFrame([experiment_row])
if experiment_log_path.exists():
    existing_log_df = pd.read_csv(experiment_log_path)
    experiment_df = pd.concat([existing_log_df, experiment_df], ignore_index=True)

experiment_df.to_csv(experiment_log_path, index=False)

print(f"Saved model: {model_path}")
print(f"Saved metrics: {metrics_path}")
print(f"Saved CV results: {cv_results_path}")
print(f"Saved CV summary: {cv_summary_path}")
print(f"Saved params: {params_path}")
print(f"Saved experiment log: {experiment_log_path}")

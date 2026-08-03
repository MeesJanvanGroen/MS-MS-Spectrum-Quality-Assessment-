from pathlib import Path
from datetime import datetime
import json
import sys

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
)

try:
    from catboost import CatBoostClassifier
except ImportError as exc:
    print("CatBoost is not installed in this Python environment.")
    print("Install it first, for example with: pip install catboost")
    raise SystemExit(1) from exc


# Paths
BASE_DIR = Path(r"C:\Users\MeesJ\OneDrive\Documenten\PM\Stage 6 maanden\Data")
PREPARED_DIR = BASE_DIR / "PreparedClassification"
MODEL_DIR = BASE_DIR / "Model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


# Train setup
TRAIN_ROWS = 100000
VAL_ROWS = 50000
MODEL_RANDOM_SEED = 42


CATBOOST_PARAMS = {
    "iterations": 512,
    "depth": 12,
    "learning_rate": 0.2,
    "l2_leaf_reg": 10,
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": MODEL_RANDOM_SEED,
    "verbose": 100,
    "thread_count": -1,
}


print("CatBoost classification training configuration:")
print(f"TRAIN_ROWS: {TRAIN_ROWS}")
print(f"VAL_ROWS: {VAL_ROWS}")
for key, value in CATBOOST_PARAMS.items():
    print(f"{key}: {value}")
print()


def build_feature_target_matrices(long_df):
    presence_matrix = np.vstack(long_df["BINARY_PRESENCE_VECTOR"].to_numpy()).astype(np.float32)
    mask_matrix = np.vstack(long_df["MASKING_VECTOR"].to_numpy()).astype(np.float32)
    precursor_array = long_df["PRECURSOR_ION_NUMERIC"].to_numpy(dtype=np.float32).reshape(-1, 1)
    queried_index_array = long_df["QUERIED_BIN_INDEX"].to_numpy(dtype=np.float32).reshape(-1, 1)
    X = np.hstack([presence_matrix, mask_matrix, precursor_array, queried_index_array]).astype(np.float32)
    y = long_df["TARGET_LABEL"].to_numpy(dtype=np.uint8)
    return X, mask_matrix, y


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


print("Loading prepared long-format classification data...")
train_long_df = pd.read_pickle(PREPARED_DIR / "TIC_train_classification_long_df.pkl")
val_long_df = pd.read_pickle(PREPARED_DIR / "TIC_val_classification_long_df.pkl")

print(f"Train queried-bin rows available: {len(train_long_df)}")
print(f"Val queried-bin rows available: {len(val_long_df)}")

if TRAIN_ROWS is not None and len(train_long_df) > TRAIN_ROWS:
    rng = np.random.default_rng(MODEL_RANDOM_SEED)
    chosen_idx = np.sort(rng.choice(len(train_long_df), size=TRAIN_ROWS, replace=False))
    train_fit_df = train_long_df.iloc[chosen_idx].reset_index(drop=True)
else:
    train_fit_df = train_long_df.reset_index(drop=True)

if VAL_ROWS is not None and len(val_long_df) > VAL_ROWS:
    rng = np.random.default_rng(MODEL_RANDOM_SEED + 10)
    chosen_idx = np.sort(rng.choice(len(val_long_df), size=VAL_ROWS, replace=False))
    val_fit_df = val_long_df.iloc[chosen_idx].reset_index(drop=True)
else:
    val_fit_df = val_long_df.reset_index(drop=True)

print(f"Train queried-bin rows used: {len(train_fit_df)}")
print(f"Val queried-bin rows used: {len(val_fit_df)}")

print("Building dense classification matrices...")
X_train, mask_train, y_train = build_feature_target_matrices(train_fit_df)
X_val, mask_val, y_val = build_feature_target_matrices(val_fit_df)

print(f"X_train shape: {X_train.shape}")
print(f"mask_train shape: {mask_train.shape}")
print(f"y_train shape: {y_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"mask_val shape: {mask_val.shape}")
print(f"y_val shape: {y_val.shape}")
print(f"Positive label fraction (train): {y_train.mean()}")
print(f"Positive label fraction (val): {y_val.mean()}")

print("Training CatBoost classifier...")
catboost_model = CatBoostClassifier(**CATBOOST_PARAMS)
catboost_model.fit(X_train, y_train)
print("CatBoost classification training finished.")

print("Scoring classifier on train and validation data...")
train_metrics = evaluate_classifier(catboost_model, X_train, y_train, "train_classification")
val_metrics = evaluate_classifier(catboost_model, X_val, y_val, "val_classification")
metrics_df = pd.DataFrame([train_metrics, val_metrics])
print(metrics_df.to_string(index=False))

model_path = MODEL_DIR / "TIC_catboost_classification_model.cbm"
metrics_path = MODEL_DIR / "TIC_catboost_classification_metrics.csv"
experiment_log_path = MODEL_DIR / "TIC_catboost_classification_experiment_log.csv"
params_path = MODEL_DIR / "TIC_catboost_classification_params.json"

catboost_model.save_model(str(model_path))
metrics_df.to_csv(metrics_path, index=False)

with open(params_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "TRAIN_ROWS": TRAIN_ROWS,
            "VAL_ROWS": VAL_ROWS,
            "CATBOOST_PARAMS": CATBOOST_PARAMS,
        },
        f,
        indent=2,
    )

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
experiment_row = {
    "RUN_TIMESTAMP": run_timestamp,
    "TRAIN_ROWS_REQUESTED": TRAIN_ROWS,
    "VAL_ROWS_REQUESTED": VAL_ROWS,
    "TRAIN_ROWS_USED": int(len(train_fit_df)),
    "VAL_ROWS_USED": int(len(val_fit_df)),
    "N_FEATURES": int(X_train.shape[1]),
    "N_TRAIN_AVAILABLE": int(len(train_long_df)),
    "N_VAL_AVAILABLE": int(len(val_long_df)),
    "POSITIVE_RATE_TRAIN": float(y_train.mean()),
    "POSITIVE_RATE_VAL": float(y_val.mean()),
    "BEST_ITERATION": catboost_model.get_best_iteration(),
}

for key, value in CATBOOST_PARAMS.items():
    experiment_row[f"HP_{key.upper()}"] = value

for metric_key, metric_value in train_metrics.items():
    if metric_key != "DATASET":
        experiment_row[f"TRAIN_{metric_key}"] = metric_value

for metric_key, metric_value in val_metrics.items():
    if metric_key != "DATASET":
        experiment_row[f"VAL_{metric_key}"] = metric_value

experiment_df = pd.DataFrame([experiment_row])
if experiment_log_path.exists():
    existing_log_df = pd.read_csv(experiment_log_path)
    experiment_df = pd.concat([existing_log_df, experiment_df], ignore_index=True)

experiment_df.to_csv(experiment_log_path, index=False)

print(f"Saved model: {model_path}")
print(f"Saved metrics: {metrics_path}")
print(f"Saved params: {params_path}")
print(f"Saved experiment log: {experiment_log_path}")

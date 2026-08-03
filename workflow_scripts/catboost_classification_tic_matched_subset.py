from pathlib import Path
from datetime import datetime
import json
import os

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


WINDOWS_BASE_DIR = Path(r"C:\Users\MeesJ\OneDrive\Documenten\PM\Stage 6 maanden\Data")
LINUX_BASE_DIR = Path("/home/mgroen2/my_model_project/Data")
if os.name == "nt":
    BASE_DIR = WINDOWS_BASE_DIR
else:
    BASE_DIR = LINUX_BASE_DIR if LINUX_BASE_DIR.exists() else WINDOWS_BASE_DIR
PREPARED_DIR = BASE_DIR / "PreparedClassification"
MODEL_DIR = BASE_DIR / "Model"
MODEL_DIR.mkdir(parents=True, exist_ok=True)


DATASET_TAG = "matched_80k_10k_10k"
TRAIN_LONG_FILENAME = f"TIC_train_classification_long_df_{DATASET_TAG}_oxygen.pkl"
VAL_LONG_FILENAME = f"TIC_val_classification_long_df_{DATASET_TAG}_oxygen.pkl"
TEST_LONG_FILENAME = f"TIC_test_classification_long_df_{DATASET_TAG}_oxygen.pkl"

MODEL_RANDOM_SEED = 42

CATBOOST_PARAMS = {
    "iterations": 512,
    "depth": 12,
    "learning_rate": 0.15,
    "l2_leaf_reg": 14.0,
    "loss_function": "Logloss",
    "eval_metric": "F1",
    "random_seed": MODEL_RANDOM_SEED,
    "verbose": False,
    "thread_count": -1,
}


print("Matched-subset CNL CatBoost training configuration:")
print(f"BASE_DIR: {BASE_DIR}")
print(f"TRAIN_LONG_FILENAME: {TRAIN_LONG_FILENAME}")
print(f"VAL_LONG_FILENAME: {VAL_LONG_FILENAME}")
print(f"TEST_LONG_FILENAME: {TEST_LONG_FILENAME}")
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
    return X, y


def evaluate_split_consistency(split_name, long_df):
    presence_lengths = long_df["BINARY_PRESENCE_VECTOR"].map(len).unique().tolist()
    mask_lengths = long_df["MASKING_VECTOR"].map(len).unique().tolist()
    index_to_cnl_df = (
        long_df.groupby("QUERIED_BIN_INDEX")["QUERIED_CNL_BIN"]
        .agg(lambda x: tuple(sorted(pd.unique(np.round(x.astype(float), 6)))))
        .reset_index()
        .rename(columns={"QUERIED_CNL_BIN": "CNL_VALUES"})
    )
    conflicting_count = int((index_to_cnl_df["CNL_VALUES"].map(len) > 1).sum())
    summary = {
        "split": split_name,
        "n_rows": int(len(long_df)),
        "n_spectra": int(long_df["SPECTRUM_INDEX"].nunique()),
        "positive_rate": float(long_df["TARGET_LABEL"].mean()),
        "queried_bin_index_min": int(long_df["QUERIED_BIN_INDEX"].min()),
        "queried_bin_index_max": int(long_df["QUERIED_BIN_INDEX"].max()),
        "queried_cnl_min": float(long_df["QUERIED_CNL_BIN"].min()),
        "queried_cnl_max": float(long_df["QUERIED_CNL_BIN"].max()),
        "presence_lengths": presence_lengths,
        "mask_lengths": mask_lengths,
        "n_index_to_cnl_conflicts": conflicting_count,
        "index_to_cnl_df": index_to_cnl_df,
    }
    return summary


def compare_index_to_cnl_maps(reference_df, other_df, reference_name, other_name):
    merged = reference_df.merge(
        other_df,
        on="QUERIED_BIN_INDEX",
        how="inner",
        suffixes=(f"_{reference_name}", f"_{other_name}"),
    )
    ref_col = f"CNL_VALUES_{reference_name}"
    other_col = f"CNL_VALUES_{other_name}"
    conflicts = merged[merged[ref_col] != merged[other_col]].copy()
    return conflicts


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


print("Loading matched-subset CNL long-format classification data...")
train_long_df = pd.read_pickle(PREPARED_DIR / TRAIN_LONG_FILENAME)
val_long_df = pd.read_pickle(PREPARED_DIR / VAL_LONG_FILENAME)
test_long_df = pd.read_pickle(PREPARED_DIR / TEST_LONG_FILENAME)

train_summary = evaluate_split_consistency("train", train_long_df)
val_summary = evaluate_split_consistency("val", val_long_df)
test_summary = evaluate_split_consistency("test", test_long_df)

print("Basic split summary:")
print(
    pd.DataFrame([
        {k: v for k, v in train_summary.items() if k not in {"presence_lengths", "mask_lengths", "index_to_cnl_df"}},
        {k: v for k, v in val_summary.items() if k not in {"presence_lengths", "mask_lengths", "index_to_cnl_df"}},
        {k: v for k, v in test_summary.items() if k not in {"presence_lengths", "mask_lengths", "index_to_cnl_df"}},
    ]).to_string(index=False)
)
print()

for split_summary in [train_summary, val_summary, test_summary]:
    print(
        f"{split_summary['split']}: "
        f"presence lens={split_summary['presence_lengths']} | "
        f"mask lens={split_summary['mask_lengths']} | "
        f"index->CNL conflicts={split_summary['n_index_to_cnl_conflicts']}"
    )
print()

train_val_conflicts = compare_index_to_cnl_maps(
    train_summary["index_to_cnl_df"], val_summary["index_to_cnl_df"], "train", "val"
)
train_test_conflicts = compare_index_to_cnl_maps(
    train_summary["index_to_cnl_df"], test_summary["index_to_cnl_df"], "train", "test"
)
val_test_conflicts = compare_index_to_cnl_maps(
    val_summary["index_to_cnl_df"], test_summary["index_to_cnl_df"], "val", "test"
)

print(
    "Cross-split queried_bin_index -> queried_cnl conflicts "
    f"(only on shared indices): train/val={len(train_val_conflicts)}, "
    f"train/test={len(train_test_conflicts)}, val/test={len(val_test_conflicts)}"
)
if len(train_val_conflicts) > 0 or len(train_test_conflicts) > 0 or len(val_test_conflicts) > 0:
    print("True shared-index conflicts were found. Stopping before training.")
    raise SystemExit(1)
print("Shared queried_bin_index mappings are consistent across splits.")
print()

tune_df = pd.concat([train_long_df, val_long_df], ignore_index=True)

print("Building dense matrices...")
X_train, y_train = build_feature_target_matrices(train_long_df)
X_val, y_val = build_feature_target_matrices(val_long_df)
X_tune, y_tune = build_feature_target_matrices(tune_df)
X_test, y_test = build_feature_target_matrices(test_long_df)

if not (X_train.shape[1] == X_val.shape[1] == X_tune.shape[1] == X_test.shape[1]):
    print("Feature-width mismatch detected across splits. Stopping before training.")
    print(f"X_train shape: {X_train.shape}")
    print(f"X_val shape: {X_val.shape}")
    print(f"X_tune shape: {X_tune.shape}")
    print(f"X_test shape: {X_test.shape}")
    raise SystemExit(1)

print(f"X_train shape: {X_train.shape}")
print(f"X_val shape: {X_val.shape}")
print(f"X_tune shape: {X_tune.shape}")
print(f"X_test shape: {X_test.shape}")
print(f"Positive label fraction (train): {y_train.mean()}")
print(f"Positive label fraction (val): {y_val.mean()}")
print(f"Positive label fraction (train+val): {y_tune.mean()}")
print(f"Positive label fraction (test): {y_test.mean()}")

print("Training final matched-subset CNL CatBoost classifier on train+val...")
catboost_model = CatBoostClassifier(**CATBOOST_PARAMS)
catboost_model.fit(X_tune, y_tune)
print("Matched-subset CNL CatBoost training finished.")

print("Scoring final classifier on train+val and held-out test data...")
train_metrics = evaluate_classifier(catboost_model, X_train, y_train, "train_classification")
val_metrics = evaluate_classifier(catboost_model, X_val, y_val, "val_classification")
tune_metrics = evaluate_classifier(catboost_model, X_tune, y_tune, "train_val_classification")
test_metrics = evaluate_classifier(catboost_model, X_test, y_test, "test_classification")
metrics_df = pd.DataFrame([train_metrics, val_metrics, tune_metrics, test_metrics])
print(metrics_df.to_string(index=False))

model_path = MODEL_DIR / f"TIC_catboost_classification_model_{DATASET_TAG}.cbm"
metrics_path = MODEL_DIR / f"TIC_catboost_classification_metrics_{DATASET_TAG}.csv"
experiment_log_path = MODEL_DIR / f"TIC_catboost_classification_experiment_log_{DATASET_TAG}.csv"
params_path = MODEL_DIR / f"TIC_catboost_classification_params_{DATASET_TAG}.json"

catboost_model.save_model(str(model_path))
metrics_df.to_csv(metrics_path, index=False)

with open(params_path, "w", encoding="utf-8") as f:
    json.dump(
        {
            "DATASET_TAG": DATASET_TAG,
            "TRAIN_LONG_FILENAME": TRAIN_LONG_FILENAME,
            "VAL_LONG_FILENAME": VAL_LONG_FILENAME,
            "TEST_LONG_FILENAME": TEST_LONG_FILENAME,
            "CATBOOST_PARAMS": CATBOOST_PARAMS,
        },
        f,
        indent=2,
    )

run_timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
experiment_row = {
    "RUN_TIMESTAMP": run_timestamp,
    "DATASET_TAG": DATASET_TAG,
    "TRAIN_ROWS_USED_FOR_TUNING": int(len(train_long_df)),
    "VAL_ROWS_USED_FOR_TUNING": int(len(val_long_df)),
    "TEST_ROWS_USED": int(len(test_long_df)),
    "TRAIN_SPECTRA_USED_FOR_TUNING": int(train_long_df["SPECTRUM_INDEX"].nunique()),
    "VAL_SPECTRA_USED_FOR_TUNING": int(val_long_df["SPECTRUM_INDEX"].nunique()),
    "TEST_SPECTRA_USED": int(test_long_df["SPECTRUM_INDEX"].nunique()),
    "N_FEATURES": int(X_tune.shape[1]),
    "POSITIVE_RATE_TUNE": float(y_tune.mean()),
    "POSITIVE_RATE_TEST": float(y_test.mean()),
}

for key, value in CATBOOST_PARAMS.items():
    experiment_row[f"HP_{key.upper()}"] = value

for metric_key, metric_value in tune_metrics.items():
    if metric_key != "DATASET":
        experiment_row[f"TRAIN_VAL_{metric_key}"] = metric_value

for metric_key, metric_value in val_metrics.items():
    if metric_key != "DATASET":
        experiment_row[f"VAL_{metric_key}"] = metric_value

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
print(f"Saved params: {params_path}")
print(f"Saved experiment log: {experiment_log_path}")

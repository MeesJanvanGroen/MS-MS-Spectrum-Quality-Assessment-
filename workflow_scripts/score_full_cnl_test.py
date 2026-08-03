from pathlib import Path
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

TEST_LONG_PATH = PREPARED_DIR / "TIC_test_classification_long_df.pkl"
MODEL_PATH = MODEL_DIR / "TIC_catboost_classification_model.cbm"
PARAMS_PATH = MODEL_DIR / "TIC_catboost_classification_params_final_round3_downloaded.json"
OUTPUT_PATH = MODEL_DIR / "TIC_catboost_classification_full_test_metrics.csv"


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
        "SPLIT": label,
        "N_QUERIED_VALUES": int(len(y)),
        "ACCURACY": float(accuracy_score(y, y_pred)),
        "BALANCED_ACCURACY": float(balanced_accuracy_score(y, y_pred)),
        "PRECISION": float(precision_score(y, y_pred, zero_division=0)),
        "RECALL": float(recall_score(y, y_pred, zero_division=0)),
        "F1": float(f1_score(y, y_pred, zero_division=0)),
        "POSITIVE_RATE_TRUE": float(np.mean(y)),
        "POSITIVE_RATE_PRED": float(np.mean(y_pred)),
    }


print("Scoring saved full CNL CatBoost model on full test set...")
print(f"BASE_DIR: {BASE_DIR}")
print(f"TEST_LONG_PATH: {TEST_LONG_PATH}")
print(f"MODEL_PATH: {MODEL_PATH}")
print()

if not TEST_LONG_PATH.exists():
    raise FileNotFoundError(f"Missing test file: {TEST_LONG_PATH}")
if not MODEL_PATH.exists():
    raise FileNotFoundError(f"Missing model file: {MODEL_PATH}")

if PARAMS_PATH.exists():
    with open(PARAMS_PATH, "r", encoding="utf-8") as f:
        saved_params = json.load(f)
    print("Found downloaded final-round parameter file:")
    print(json.dumps(saved_params, indent=2))
    print()

test_long_df = pd.read_pickle(TEST_LONG_PATH)
print(f"Full test queried-bin rows available: {len(test_long_df)}")
print(f"Full test spectra available: {test_long_df['SPECTRUM_INDEX'].nunique()}")

X_test, y_test = build_feature_target_matrices(test_long_df)
print(f"X_test shape: {X_test.shape}")
print(f"Positive label fraction (test): {y_test.mean()}")
print()

model = CatBoostClassifier()
model.load_model(str(MODEL_PATH))

test_metrics = evaluate_classifier(model, X_test, y_test, "test_full")
test_metrics_df = pd.DataFrame([test_metrics])

print("Full-model test metrics:")
print(test_metrics_df.to_string(index=False))

test_metrics_df.to_csv(OUTPUT_PATH, index=False)
print()
print(f"Saved full-model test metrics: {OUTPUT_PATH}")

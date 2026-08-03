from pathlib import Path
import os

import pandas as pd


WINDOWS_BASE_DIR = Path(r"C:\Users\MeesJ\OneDrive\Documenten\PM\Stage 6 maanden\Data")
LINUX_BASE_DIR = Path("/home/mgroen2/my_model_project/Data")
BASE_DIR = WINDOWS_BASE_DIR if os.name == "nt" else (LINUX_BASE_DIR if LINUX_BASE_DIR.exists() else WINDOWS_BASE_DIR)
PREPARED_DIR = BASE_DIR / "PreparedClassification"

OUTPUT_TAG = "matched_80k_10k_10k"
TARGET_ROWS = {"train": 80000, "val": 10000, "test": 10000}
RANDOM_SEED = 42

# These columns should describe the same underlying spectrum in both workflows.
MATCH_BASE_COLS = [
    "INCHIKEY",
    "PRECURSOR_ION_NUMERIC",
    "N_PEAKS",
    "N_MASKED_POSITIVE",
    "N_MASKED_NEGATIVE",
]


def estimated_rows_per_spectrum(class_df):
    return (class_df["N_MASKED_POSITIVE"].astype(int) + class_df["N_MASKED_NEGATIVE"].astype(int)).rename("ESTIMATED_LONG_ROWS")


def choose_spectra_for_target(class_df, target_rows, seed):
    work_df = class_df[["SPECTRUM_INDEX"]].copy()
    work_df["ESTIMATED_LONG_ROWS"] = estimated_rows_per_spectrum(class_df).to_numpy()
    work_df = work_df.sample(frac=1.0, random_state=seed).reset_index(drop=True)

    chosen_ids = []
    running_total = 0
    for row in work_df.itertuples(index=False):
        next_total = running_total + int(row.ESTIMATED_LONG_ROWS)
        current_gap = abs(target_rows - running_total)
        next_gap = abs(target_rows - next_total)

        if len(chosen_ids) == 0 or next_gap <= current_gap or running_total < target_rows:
            chosen_ids.append(int(row.SPECTRUM_INDEX))
            running_total = next_total
        if running_total >= target_rows and next_gap > current_gap:
            break

    return chosen_ids, running_total


def make_oxygen_safe(df):
    df = df.copy()
    string_cols = df.select_dtypes(include=["string"]).columns
    for col in string_cols:
        df[col] = df[col].astype(object)
    df.columns = pd.Index([str(col) for col in df.columns], dtype=object)
    if getattr(df.index, "dtype", None) == "string":
        df.index = pd.Index([str(idx) for idx in df.index], dtype=object)
    return df


def save_with_regular_and_oxygen_versions(df, regular_path, oxygen_path):
    df.to_pickle(regular_path)
    make_oxygen_safe(df).to_pickle(oxygen_path)


def add_match_key(class_df):
    work_df = class_df.copy()
    work_df["DUP_RANK"] = work_df.groupby(MATCH_BASE_COLS).cumcount()
    work_df["MATCH_KEY"] = (
        work_df["INCHIKEY"].astype(str)
        + "|"
        + work_df["PRECURSOR_ION_NUMERIC"].round(6).astype(str)
        + "|"
        + work_df["N_PEAKS"].astype(int).astype(str)
        + "|"
        + work_df["N_MASKED_POSITIVE"].astype(int).astype(str)
        + "|"
        + work_df["N_MASKED_NEGATIVE"].astype(int).astype(str)
        + "|"
        + work_df["DUP_RANK"].astype(int).astype(str)
    )
    return work_df


def summarize_long_grid(split_name, df):
    return {
        "SPLIT": split_name,
        "N_ROWS": int(len(df)),
        "N_SPECTRA": int(df["SPECTRUM_INDEX"].nunique()),
        "PRESENCE_VECTOR_LENGTH": int(df["BINARY_PRESENCE_VECTOR"].map(len).iloc[0]),
        "MASK_VECTOR_LENGTH": int(df["MASKING_VECTOR"].map(len).iloc[0]),
        "QUERIED_BIN_INDEX_MAX": int(df["QUERIED_BIN_INDEX"].max()),
        "QUERIED_CNL_MAX": float(df["QUERIED_CNL_BIN"].max()),
    }


print(f"BASE_DIR: {BASE_DIR}")
print(f"PREPARED_DIR: {PREPARED_DIR}")
print(f"OUTPUT_TAG: {OUTPUT_TAG}")

cnl_class = {
    "train": pd.read_pickle(PREPARED_DIR / "TIC_train_classification_df.pkl"),
    "val": pd.read_pickle(PREPARED_DIR / "TIC_val_classification_df.pkl"),
    "test": pd.read_pickle(PREPARED_DIR / "TIC_test_classification_df.pkl"),
}
cnl_long = {
    "train": pd.read_pickle(PREPARED_DIR / "TIC_train_classification_long_df.pkl"),
    "val": pd.read_pickle(PREPARED_DIR / "TIC_val_classification_long_df.pkl"),
    "test": pd.read_pickle(PREPARED_DIR / "TIC_test_classification_long_df.pkl"),
}
mz_class = {
    "train": pd.read_pickle(PREPARED_DIR / "TIC_mz_train_classification_df.pkl"),
    "val": pd.read_pickle(PREPARED_DIR / "TIC_mz_val_classification_df.pkl"),
    "test": pd.read_pickle(PREPARED_DIR / "TIC_mz_test_classification_df.pkl"),
}
mz_long = {
    "train": pd.read_pickle(PREPARED_DIR / "TIC_mz_train_classification_long_df.pkl"),
    "val": pd.read_pickle(PREPARED_DIR / "TIC_mz_val_classification_long_df.pkl"),
    "test": pd.read_pickle(PREPARED_DIR / "TIC_mz_test_classification_long_df.pkl"),
}

print("\nOriginal CNL grid check:")
cnl_grid_summary_df = pd.DataFrame([summarize_long_grid(split, cnl_long[split]) for split in ["train", "val", "test"]])
print(cnl_grid_summary_df.to_string(index=False))

selected_ids_cnl = {}
selected_ids_mz = {}
selection_rows = []

for offset, split_name in enumerate(["train", "val", "test"]):
    cnl_class_keyed = add_match_key(cnl_class[split_name])
    mz_class_keyed = add_match_key(mz_class[split_name])

    shared_df = cnl_class_keyed.merge(
        mz_class_keyed,
        on=["MATCH_KEY"] + MATCH_BASE_COLS,
        how="inner",
        suffixes=("_CNL", "_MZ"),
    )

    if len(shared_df) == 0:
        raise ValueError(f"No shared spectra found for split '{split_name}' using metadata-based keys.")

    cnl_shared_for_sampling = shared_df[
        ["SPECTRUM_INDEX_CNL", "N_MASKED_POSITIVE", "N_MASKED_NEGATIVE"]
    ].rename(columns={"SPECTRUM_INDEX_CNL": "SPECTRUM_INDEX"})

    chosen_cnl_ids, estimated_total = choose_spectra_for_target(
        cnl_shared_for_sampling,
        TARGET_ROWS[split_name],
        seed=RANDOM_SEED + offset,
    )

    chosen_pairs_df = shared_df[shared_df["SPECTRUM_INDEX_CNL"].isin(chosen_cnl_ids)].copy()
    selected_ids_cnl[split_name] = set(chosen_pairs_df["SPECTRUM_INDEX_CNL"].astype(int).tolist())
    selected_ids_mz[split_name] = set(chosen_pairs_df["SPECTRUM_INDEX_MZ"].astype(int).tolist())

    selection_rows.append({
        "SPLIT": split_name,
        "SHARED_MATCHED_SPECTRA": int(len(shared_df)),
        "SELECTED_SPECTRA": int(len(selected_ids_cnl[split_name])),
        "ESTIMATED_CNL_LONG_ROWS": int(estimated_total),
    })

selection_df = pd.DataFrame(selection_rows)
print("\nSelection summary:")
print(selection_df.to_string(index=False))

cnl_class_subset = {}
cnl_long_subset = {}
mz_class_subset = {}
mz_long_subset = {}
actual_rows = []

for split_name in ["train", "val", "test"]:
    cnl_keep = selected_ids_cnl[split_name]
    mz_keep = selected_ids_mz[split_name]

    cnl_class_subset[split_name] = cnl_class[split_name][cnl_class[split_name]["SPECTRUM_INDEX"].isin(cnl_keep)].copy()
    cnl_long_subset[split_name] = cnl_long[split_name][cnl_long[split_name]["SPECTRUM_INDEX"].isin(cnl_keep)].copy()
    mz_class_subset[split_name] = mz_class[split_name][mz_class[split_name]["SPECTRUM_INDEX"].isin(mz_keep)].copy()
    mz_long_subset[split_name] = mz_long[split_name][mz_long[split_name]["SPECTRUM_INDEX"].isin(mz_keep)].copy()

    actual_rows.append({
        "SPLIT": split_name,
        "CNL_SPECTRA": int(cnl_class_subset[split_name]["SPECTRUM_INDEX"].nunique()),
        "CNL_LONG_ROWS": int(len(cnl_long_subset[split_name])),
        "MZ_SPECTRA": int(mz_class_subset[split_name]["SPECTRUM_INDEX"].nunique()),
        "MZ_LONG_ROWS": int(len(mz_long_subset[split_name])),
    })

actual_rows_df = pd.DataFrame(actual_rows)
print("\nActual matched subset sizes:")
print(actual_rows_df.to_string(index=False))

print("\nMatched CNL grid check:")
matched_cnl_grid_summary_df = pd.DataFrame([summarize_long_grid(split, cnl_long_subset[split]) for split in ["train", "val", "test"]])
print(matched_cnl_grid_summary_df.to_string(index=False))

for split_name in ["train", "val", "test"]:
    pd.DataFrame({"SPECTRUM_INDEX_CNL": sorted(selected_ids_cnl[split_name]), "SPECTRUM_INDEX_MZ": sorted(selected_ids_mz[split_name])}).to_csv(
        PREPARED_DIR / f"matched_spectrum_ids_{split_name}_{OUTPUT_TAG}.csv",
        index=False,
    )

    save_with_regular_and_oxygen_versions(
        cnl_class_subset[split_name],
        PREPARED_DIR / f"TIC_{split_name}_classification_df_{OUTPUT_TAG}.pkl",
        PREPARED_DIR / f"TIC_{split_name}_classification_df_{OUTPUT_TAG}_oxygen.pkl",
    )
    save_with_regular_and_oxygen_versions(
        cnl_long_subset[split_name],
        PREPARED_DIR / f"TIC_{split_name}_classification_long_df_{OUTPUT_TAG}.pkl",
        PREPARED_DIR / f"TIC_{split_name}_classification_long_df_{OUTPUT_TAG}_oxygen.pkl",
    )

    save_with_regular_and_oxygen_versions(
        mz_class_subset[split_name],
        PREPARED_DIR / f"TIC_mz_{split_name}_classification_df_{OUTPUT_TAG}.pkl",
        PREPARED_DIR / f"TIC_mz_{split_name}_classification_df_{OUTPUT_TAG}_oxygen.pkl",
    )
    save_with_regular_and_oxygen_versions(
        mz_long_subset[split_name],
        PREPARED_DIR / f"TIC_mz_{split_name}_classification_long_df_{OUTPUT_TAG}.pkl",
        PREPARED_DIR / f"TIC_mz_{split_name}_classification_long_df_{OUTPUT_TAG}_oxygen.pkl",
    )

print("\nSaved rebuilt metadata-matched subset files.")
print("Next steps:")
print("1. Upload the new *_matched_80k_10k_10k_oxygen.pkl files to Oxygen.")
print("2. Rerun the matched CNL and matched m/z training scripts.")

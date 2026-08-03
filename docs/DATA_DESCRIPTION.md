# Data Description

This folder does not include the full raw or prepared training data tables because these files are large. Instead, it includes the files needed to understand the starting data used for the final workflows, the final feature spaces, and the matched comparison subset.

Included files:

- `data_description/TIC_classification_grid.csv`
  Final retained CNL grid derived from the TIC-preprocessed starting spectra and used in the TIC-based CNL workflow.
- `data_description/TIC_mz_classification_grid.csv`
  Fragment m/z grid derived from the same starting spectra and used in the direct m/z comparison workflow.
- `data_description/matched_spectrum_ids_train_matched_80k_10k_10k.csv`
  Spectrum IDs from the starting dataset included in the matched training subset.
- `data_description/matched_spectrum_ids_val_matched_80k_10k_10k.csv`
  Spectrum IDs from the starting dataset included in the matched validation subset.
- `data_description/matched_spectrum_ids_test_matched_80k_10k_10k.csv`
  Spectrum IDs from the starting dataset included in the matched test subset.

Matched comparison subset:

- training queried-bin rows: 80,000
- validation queried-bin rows: 10,000
- test queried-bin rows: 10,000

The matched subset was constructed to ensure that the CNL-based and m/z-based comparison models were trained and evaluated on the same underlying spectra.

Starting data note:

- the full project started from harmonized raw MS/MS spectra compiled from multiple libraries
- after preprocessing, those spectra were converted either to a CNL grid or to a fragment m/z grid, depending on the workflow
- the files in this folder describe those starting feature spaces and the spectrum IDs that were carried into the matched comparison workflow

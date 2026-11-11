# Eligibility and the carbon cost of representational repair in reuse-centric federated learning

Code, protocol and result tables for the study of device-ownership eligibility in reuse-centric federated learning, with two clinical tasks: conduction disturbance on the electrocardiogram (PTB-XL development; CODE-15% and Chapman-Shaoxing plus Ningbo external) and malignancy on smartphone dermatology images (PAD-UFES-20; Fitzpatrick17k check).

## Layout

| Directory | Content |
|---|---|
| `protocol/` | analysis plan (Task A), Task B protocol, participation model, data sources |
| `data_acquisition/` | download, completeness repair and integrity verification of the public data sets |
| `preprocessing/` | waveform and image caches (`preprocess_ptbxl.py`, `preprocess_code15.py`, `preprocess_chapman.py`, `preprocess_derm.py`) |
| `participation/` | HINTS 7 models (`fit_hints.py`, `propensity.py`, functional-form check `age_form_check.py`) and the participation arms (`arms.py`) |
| `modelling/` | compact networks, training jobs for Task A and Task B, evaluation metrics |
| `experiments/` | job suites and workers (`run.py`, `run_derm.py`), worker launcher, tuning (`tune.py`, `tune_derm.py`, and `tune_independent.py` with `indep_confirm.py` for the hyperparameter selection restricted to folds 1 to 6) |
| `analysis/` | pooled out-of-fold analysis (optionally restricted to test folds), one-threshold age-disparity analysis, external scoring, decision curves, Fitzpatrick17k check, variance components |
| `carbon/` | recruitment statistics and remediation efficiency |
| `reporting/` | collection of every reported number from `results/` into a ledger; figure scripts and SVG figures |
| `results/` | analysis output tables (CSV, JSON) |
| `tests/` | unit tests (no data required) |
| `run_pipeline.sh` | the full sequence, data to figures |

## Requirements

Python 3; packages in `requirements.txt`. Training uses CUDA (two GPUs in the reported runs; one job holds its cohort on the GPU, about 2 GB). Preprocessing and analysis run on CPU. Figures additionally need `matplotlib`.

## Reproduction

`run_pipeline.sh` lists every step in order with the commands and arguments; set `RAW`, `CACHE`, `RES`, `AN`, `LOGS` and `HINTS_SAV` and run it, or run the steps individually. PAD-UFES-20 (Mendeley Data) and the HINTS 7 file are downloaded manually; see `protocol/data_sources.md`.

Training suites (7,596 Task A jobs and 6,300 Task B jobs, plus 400 retraining jobs per task if the hyperparameter selection restricted to folds 1 to 6 selects different values) are resumable: rerunning a worker command skips finished jobs. Seeds are fixed in the suite definitions, so the analysis of a rerun reproduces the committed tables up to GPU non-determinism in training.

Checks that need no data:

    python3 modelling/models.py && python3 modelling/evaluate.py
    python3 tests/test_participation.py && python3 tests/test_trainer.py && python3 tests/test_trainer_derm.py && python3 tests/test_round7.py

The reported numbers are produced from `results/` by `reporting/build_ledger.py`.

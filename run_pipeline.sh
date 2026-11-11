#!/bin/bash
# Full reconstruction of the study, in dependency order. Each step can be run on its own.
# Set the variables below; steps marked [GPU] need CUDA devices, the rest run on CPU.
set -e
RAW=${RAW:?raw data root}           # downloads
CACHE=${CACHE:?cache dir}           # preprocessed arrays
RES=${RES:?results dir}             # trained-job outputs
AN=${AN:?analysis dir}              # analysis outputs
LOGS=${LOGS:?log dir}
HINTS_SAV=${HINTS_SAV:?path to hints7_public.sav}
W0=${W0:-4}; W1=${W1:-4}            # workers per GPU
R=$(cd "$(dirname "$0")" && pwd)

# 1. Data (see protocol/data_sources.md; PAD-UFES-20 and HINTS 7 are downloaded manually)
bash $R/data_acquisition/download_ptbxl.sh $RAW
python3 $R/data_acquisition/repair_ptbxl.py $RAW/ptbxl/physionet.org/files/ptb-xl/1.0.3 --fetch
bash $R/data_acquisition/download_code15.sh $RAW
bash $R/data_acquisition/download_chapman.sh $RAW
CHAP=$RAW/chapman_shaoxing/physionet.org/files/ecg-arrhythmia/1.0.0
python3 $R/data_acquisition/extract_chapman_headers.py $CHAP $RAW/chapman_headers.csv
python3 $R/data_acquisition/scrape_fitzpatrick17k.py --csv $RAW/fitzpatrick17k/fitzpatrick17k.csv --out-dir $RAW/fitzpatrick17k/images
python3 $R/data_acquisition/quarantine_fitzpatrick.py $RAW/fitzpatrick17k/images $RAW/fitzpatrick17k/images_quarantine

# 2. Preprocessing
python3 $R/preprocessing/preprocess_ptbxl.py --root $RAW/ptbxl/physionet.org/files/ptb-xl/1.0.3 --out-dir $CACHE
python3 $R/preprocessing/preprocess_code15.py --raw $RAW/code15 --out-dir $CACHE
python3 $R/preprocessing/preprocess_chapman.py --root $CHAP --headers $RAW/chapman_headers.csv --out-dir $CACHE
python3 $R/preprocessing/preprocess_derm.py pad  --root $RAW/pad_mendeley --metadata $RAW/pad_mendeley/metadata.csv --out-dir $CACHE
python3 $R/preprocessing/preprocess_derm.py fitz --root $RAW/fitzpatrick17k --out-dir $CACHE

# 3. Participation models (reproduces the constants in participation/propensity.py)
python3 $R/participation/fit_hints.py --sav $HINTS_SAV
python3 $R/participation/age_form_check.py --sav $HINTS_SAV --out $R/results/participation/age_form_check.csv
python3 $R/participation/hints_descriptives.py --sav $HINTS_SAV --out $R/results/participation/hints_descriptives.json

# 4. Unit tests (no data needed)
for t in tests/test_participation.py tests/test_trainer.py tests/test_trainer_derm.py tests/test_round7.py; do python3 $R/$t; done

# 5. Tuning [GPU]; selected values: DEFAULT_TRAIN (trainer.py), DEFAULT_TRAIN_DERM (trainer_derm.py)
python3 $R/experiments/tune.py      --cache-dir $CACHE
python3 $R/experiments/tune_derm.py --cache-dir $CACHE
# 5b. Hyperparameter selection restricted to folds 1-6 [GPU]; one learning rate per process may be run in parallel with --lr; if the selection
#     differs from the values above, retrain the key arms for test folds 7-10 with experiments/indep_confirm.py (suites indep_conf, derm_indep_conf)
for lr in 3e-4 1e-3 3e-3 1e-2; do python3 $R/experiments/tune_independent.py --task ecg --cache-dir $CACHE --out-dir $RES/tuning --lr $lr; done
for lr in 1e-3 3e-3 1e-2; do python3 $R/experiments/tune_independent.py --task derm --cache-dir $CACHE --out-dir $RES/tuning --lr $lr; done
python3 $R/experiments/tune_independent.py --task ecg  --out-dir $RES/tuning --summarise
python3 $R/experiments/tune_independent.py --task derm --out-dir $RES/tuning --summarise

# 6. Training suites [GPU]; WAIT=1 makes launch.sh return when its workers finish; rerunning resumes
for s in primary frontier volume_control val_common arch_wide sens_none sens_consent sens_phone sens_shallow sens_steep sens_stream sens_band band_vc varcomp; do
  WAIT=1 bash $R/experiments/launch.sh ecg $s $W0 $W1 $CACHE $RES/ecg $LOGS
done
for s in derm_primary derm_frontier derm_volume derm_val_common derm_arch_wide derm_sens_none derm_sens_consent derm_sens_shallow derm_sens_steep derm_sens_stream derm_sens_band; do
  WAIT=1 bash $R/experiments/launch.sh derm $s $W0 $W1 $CACHE $RES/derm $LOGS
done

# 7. Analysis
python3 $R/analysis/ecg_analysis.py --results $RES/ecg --cache-dir $CACHE --out-dir $AN/ecg_analysis \
    --suites primary frontier volume_control val_common band_vc arch_wide sens_none sens_consent sens_phone sens_shallow sens_steep sens_stream sens_band
python3 $R/analysis/ecg_analysis.py --task derm --results $RES/derm --cache-dir $CACHE --out-dir $AN/derm_analysis \
    --suites derm_primary derm_frontier derm_volume derm_val_common derm_arch_wide derm_sens_none derm_sens_consent derm_sens_shallow derm_sens_steep derm_sens_stream derm_sens_band
python3 $R/analysis/varcomp_analysis.py --results $RES/ecg/varcomp --out-dir $AN/varcomp
python3 $R/analysis/decision_curves.py --task ecg  --results $RES/ecg  --cache-dir $CACHE --out-dir $AN/dca --boot 1000
python3 $R/analysis/decision_curves.py --task derm --results $RES/derm --cache-dir $CACHE --out-dir $AN/dca --boot 1000
python3 $R/analysis/score_external.py --results $RES/ecg/primary --cache-dir $CACHE --out-dir $AN/ecg_analysis --gpu 0   # [GPU]
python3 $R/analysis/external_analysis.py --scores $AN/ecg_analysis --cache-dir $CACHE --out-dir $AN/ecg_analysis
python3 $R/analysis/score_fitz.py --results $RES/derm/derm_primary --cache-dir $CACHE --out-dir $AN/derm_analysis --gpu 0 # [GPU]
python3 $R/analysis/fitz_analysis.py --scores $AN/derm_analysis --cache-dir $CACHE --out-dir $AN/derm_analysis

# 8. Carbon accounting
python3 $R/carbon/recruit_stats.py --cache-dir $CACHE --out-dir $AN/ecg_analysis
python3 $R/carbon/recruit_stats.py --cache-dir $CACHE --out-dir $AN/ecg_analysis --seeds 10 --age-bound band
python3 $R/carbon/eta.py --analysis-dir $AN/ecg_analysis
python3 $R/carbon/eta.py --analysis-dir $AN/ecg_analysis --suite val_common
python3 $R/carbon/eta.py --analysis-dir $AN/ecg_analysis --suite sens_band --recruit-file recruit_stats_band.csv
python3 $R/carbon/eta.py --analysis-dir $AN/ecg_analysis --suite band_vc --recruit-file recruit_stats_band.csv
python3 $R/analysis/seed_patient_bootstrap.py --results $RES/ecg --cache-dir $CACHE --out-dir $AN/ecg_analysis \
    --suites primary frontier volume_control val_common --recruit-jobs $AN/ecg_analysis/recruit_stats_jobs.csv
python3 $R/analysis/seed_patient_bootstrap.py --results $RES/ecg --cache-dir $CACHE --out-dir $AN/ecg_analysis \
    --suites sens_band band_vc --recruit-jobs $AN/ecg_analysis/recruit_stats_band_jobs.csv
# evaluation restricted to the test folds that took no part in the selection on folds 1-6, and the one-threshold age-disparity analysis
python3 $R/analysis/ecg_analysis.py --results $RES/ecg --cache-dir $CACHE --out-dir $AN/ecg_analysis/indep --folds 7 8 9 10 \
    --suites primary frontier volume_control val_common
python3 $R/analysis/ecg_analysis.py --task derm --results $RES/derm --cache-dir $CACHE --out-dir $AN/derm_analysis/indep --folds 7 8 9 10 \
    --suites derm_primary derm_frontier derm_volume derm_val_common
python3 $R/analysis/common_threshold.py --results $RES/ecg --cache-dir $CACHE --out-dir $AN/ecg_analysis \
    --suites primary frontier volume_control val_common sens_band band_vc arch_wide
python3 $R/analysis/common_threshold.py --task derm --results $RES/derm --cache-dir $CACHE --out-dir $AN/derm_analysis \
    --suites derm_primary derm_frontier derm_volume derm_val_common derm_sens_band derm_arch_wide
python3 $R/analysis/seed_patient_bootstrap.py --task derm --results $RES/derm --cache-dir $CACHE --out-dir $AN/derm_analysis \
    --suites derm_primary derm_val_common

# 9. Reporting inputs (reads the committed results/ tables)
python3 $R/reporting/build_ledger.py
python3 $R/reporting/make_figures.py

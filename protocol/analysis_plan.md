# Analysis plan: Task A (ECG conduction disturbance)


## 1. Estimands

Eligibility to contribute data in reuse-centric learning depends on owning the device that produces the signal; for the electrocardiogram this is a wearable. The plan estimates (i) the sensitivity cost of this rule for patients aged 65 and over, (ii) the part of the cost that is due to composition and the part due to lost data, and (iii) the carbon cost of each repair.

Primary estimand: difference in sensitivity at 95% specificity among patients aged 65 and over between the availability arm and the uniform arm, from pooled out-of-fold predictions over all ten PTB-XL folds, label = conduction disturbance (CD), averaged over seeds.

## 2. Design decisions

| Item | Decision |
|---|---|
| Eligibility device | wearable (the device that produces the ECG) |
| Engagement (A) | `tilt` (primary): engagement tilts per-round selection among eligible patients. `none` (device use only) and `consent` (persistent consent) are sensitivity suites |
| Repair target (B) | `pool` (primary): the repair arm re-levels the age-stratum composition of the enrolled pool. `stream` is a sensitivity suite |
| Strata for re-levelling | under 65, 65 to 74, 75 and over; population shares of the training set |
| Arms | uniform, size_matched, availability, repr_aware (boost 1), post_hoc; volume control recruit_random |
| Unit | the patient; eligibility is drawn once per (eligibility seed, test fold, set) and persists |
| Folds | each of the ten PTB-XL folds as test fold; validation = next fold cyclically; training = the remaining eight |
| Evaluation | the full held-out test fold, never filtered by eligibility |
| Seeds | 20 (initialisation and eligibility seed equal within a job); variance split by suite `varcomp` |
| Model | CompactECGNet, 164,289 parameters |

Arms 2 to 5 share the eligibility vector and the recruitment order within a seed (common random numbers), so contrasts are paired.

## 3. Hypotheses

H1 (equivalence): availability vs uniform, sensitivity at 95% specificity in the 65+ stratum. The arms are equivalent if the 90% confidence interval of the difference lies within plus or minus 4 percentage points (two one-sided tests). The margin of 4 points is fixed in this plan and does not derive from a clinical reference standard.

Secondary family (Benjamini-Hochberg): size_matched vs uniform (lost data), availability vs size_matched (composition), repr_aware vs availability, post_hoc vs availability; for sensitivity in 65+ and 75+, overall AUC, and calibration (slope, intercept, expected calibration error with ten bins). Absolute performance is reported beside every gap.

## 4. Inference

Metrics are computed on pooled out-of-fold predictions per (arm, seed). Patient-level bootstrap (2,000 resamples of whole patients) with resample indices shared across arms, so differences are paired; the interval of a contrast combines the patient bootstrap with the across-seed spread (mean over seeds, bootstrap within seed). Exact sign-flip test across seeds. Decision-curve analysis at thresholds 0.05 to 0.5 for the primary arms. Subgroup sizes and positives accompany every estimate.

## 5. Suites

| Suite | Jobs | Content |
|---|---|---|
| primary | 1,000 | 5 arms x 20 seeds x 10 folds |
| frontier | 600 | repr_aware at boost 0.25, 0.5, 0.75 |
| volume_control | 200 | recruit_random: availability pool plus as many random recruits as repr_aware |
| val_common | 1,600 | 20 seeds x 10 folds x arms size_matched, availability, repr_aware, post_hoc, recruit_random and repr_aware at boost 0.25, 0.5, 0.75, with checkpoint selection on the complete validation fold (unit loss weights); uniform shared with primary |
| arch_wide | 500 | 5 arms x 10 seeds x 10 folds with CompactECGNet at twice the channel width (base_ch 64, 643,969 parameters) |
| band_vc | 800 | categorical-age participation model with checkpoint selection on the complete validation fold: arms size_matched, availability, repr_aware, post_hoc, recruit_random and repr_aware at boost 0.25, 0.5, 0.75; 10 seeds x 10 folds (uniform shared with primary) |
| sens_none, sens_consent, sens_phone, sens_shallow, sens_steep, sens_stream, sens_band | 400 each | 10 seeds x 10 folds x arms size_matched, availability, repr_aware, post_hoc (uniform shared with primary) |
| varcomp | 96 | 4 x 4 initialisation x eligibility seeds, folds 2, 5, 9, arms availability and repr_aware |

`sens_none` and `sens_consent` vary the engagement reading; `sens_phone` uses smartphone eligibility; `sens_shallow` and `sens_steep` use the age coefficient at the confidence limit nearest to and furthest from zero; `sens_stream` changes the repair target. All suites are reported. `sens_band` replaces linear age by six age bands (18-34, 35-49, 50-64, 65-74, 75-84, 85-90) in the ownership and engagement models (`BAND_MODELS` in `participation/propensity.py`); its uniform arm is shared with primary because the uniform pool does not depend on the participation model.

The suite `arch_wide` repeats the five primary arms with a wider network to test whether the participation contrasts depend on model capacity. `analysis/seed_patient_bootstrap.py` adds intervals that resample training seeds together with patients (the same patient resamples as the main analysis) for the main suites, and a joint interval of remediation efficiency that includes the seed-to-seed variation of R_a; the coefficient uncertainty of the HINTS models is not resampled.

## 6. Carbon accounting

Per job: participations (rounds x K), steps, samples processed, recruits, wall time. Remediation efficiency (Eq. 5) is the extra campaign carbon of an arm over the availability arm per percentage point of recovered 65+ sensitivity (`carbon/eta.py`); the frontier comes from the `frontier` suite. Intervals come from the paired bootstrap; where the interval of the denominator includes zero the ratio is reported as unbounded.

## 7. External cohorts and Task B

CODE-15% (formal tests) and Chapman plus Ningbo (descriptive; record-level bootstrap because the headers carry no patient identifiers; the two hospitals reported separately) are scored with the PTB-XL models, inference only. Task B is specified in `taskB_protocol.md`.

## 8. Training protocol

Budget in epoch-equivalents of the full training set, K = 512 participants per round, batch 128, AdamW, cosine schedule, bf16 autocast, no augmentation. Every arm runs the same number of gradient steps. The checkpoint with the lowest validation loss is kept; the validation fold passes through the same eligibility rule and is evaluated every 25 rounds.

Tuning (`experiments/tune.py`): learning rate {3e-4, 1e-3, 3e-3} x epoch-equivalents {20, 40, 80} on the uniform arm, validation loss only, seeds 900 to 902 (outside the analysis seeds), test folds 1, 5, 9.

## 9. Training rule, tuning result and suites added to the plan

Early stopping is off: every job runs its whole step budget and keeps the checkpoint with the lowest validation loss, which the trainer saves as `<hash>.pt` so that the external cohorts are scored with the same models. `tune.py` follows the same rule.

Mean best validation loss on the uniform arm in the tuning grid:

| learning rate | 20 epoch-equiv | 40 | 80 |
|---|---|---|---|
| 3e-4 | 0.2768 | 0.2776 | 0.2769 |
| 1e-3 | 0.2765 | 0.2769 | 0.2763 |
| 3e-3 | 0.2732 | 0.2750 | 0.2751 |

The largest difference between settings is 0.0044. Selected: learning rate 3e-3, 20 epoch-equivalents (about 2,700 gradient steps per job); this setting is `DEFAULT_TRAIN` in `modelling/trainer.py`.

The suite `volume_control` (arm recruit_random: the availability pool plus as many randomly chosen recruits as the repair arm) separates the effect of added training volume from the effect of age targeting.

The suite `val_common` repeats the primary suite with one change: the validation fold used for checkpoint selection and early stopping is not filtered by the arm's eligibility rule, so every arm selects its checkpoint on the same patients. Contrasts against the primary suite show how much of the arm-specific effects is carried by checkpoint selection on an arm-specific validation set.

## 10. Analysis conventions

Decision-curve thresholds in the analysis are 0.02 to 0.30 (Task A) and 0.10 to 0.70 (Task B). The Benjamini-Hochberg family consists of the 11 sensitivity contrasts of the primary suite (size_matched vs uniform, availability vs size_matched, repr_aware vs availability and post_hoc vs availability at 65+ and 75+; availability vs uniform at 75+; repr_aware vs uniform at 65+ and 75+); area under the curve and calibration are reported without adjustment, and supplementary suites are adjusted within the contrasts they contain. Intervals are the 2.5th and 97.5th percentiles of the seed-averaged metric over patient resamples shared by all arms and seeds; the across-seed standard deviation is reported separately.

Age-disparity quantities are reported outside the multiplicity family (`agegap/<suite>_arms.csv`, `agegap/<suite>_contrasts.csv`): sensitivity at 95% specificity below 65 years, the age gap (65+ minus under 65, each with its subgroup threshold), their contrasts between arms with the shared patient bootstrap and the sign-flip test, and the direct contrast of full repair against the 0.75 boost level in the frontier suite.

## 11. Hyperparameter selection restricted to folds 1 to 6, categorical age with common validation, one-threshold analysis

Hyperparameter selection restricted to folds 1 to 6. The grids of section 8 used test folds 1, 5, 9 with validation folds 2, 6, 10, so every record of the development cohort took part in some tuning job. `experiments/tune_independent.py` repeats the selection on folds 1 to 6 only (the cohort is restricted to those folds; tuning jobs use test folds 1, 3, 5, validation folds 2, 4, 6 and the remaining folds for training; uniform arm, validation loss, seeds 900 to 902), with a grid extended by one step on each side of the selected values (ECG: learning rate {3e-4, 1e-3, 3e-3, 1e-2} x epoch-equivalents {10, 20, 40, 80}; dermatology: {1e-3, 3e-3, 1e-2} x {10, 20, 40, 80}). Folds 7 to 10 are not read. `ecg_analysis.py --folds 7 8 9 10` evaluates the main suites on the test folds 7 to 10 only, so those estimates do not depend on records used for selection. If the selection on folds 1 to 6 differs from the values in `DEFAULT_TRAIN` / `DEFAULT_TRAIN_DERM`, `experiments/indep_confirm.py` retrains the arms uniform, size_matched, availability, repr_aware and recruit_random (20 seeds, test folds 7 to 10, 400 jobs per task) with the selected values (suites `indep_conf`, `derm_indep_conf`).

Categorical age with common validation. The suite `band_vc` combines the participation model of `sens_band` with the validation rule of `val_common`. Recruitment statistics for the categorical-age model (all five repair labels) come from `carbon/recruit_stats.py --age-bound band`, and `carbon/eta.py --suite band_vc --recruit-file recruit_stats_band.csv` gives the remediation efficiency.

One-threshold analysis. The age gap of section 1 is the difference of sensitivities at subgroup-specific 95% specificity. `analysis/common_threshold.py` adds a deployment-oriented estimand: one threshold per model, the score at which specificity over all test patients is 95%; sensitivity and specificity are then read within patients aged 65 and over and under 65, and the gap is their difference. The threshold is re-estimated on every patient resample; contrasts use the pairs and resamples of the main analysis (`commonthr/<suite>_arms.csv`, `_contrasts.csv`).

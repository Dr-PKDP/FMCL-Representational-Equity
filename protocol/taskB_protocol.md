# Protocol: Task B (dermatology, PAD-UFES-20)

Items not listed follow `analysis_plan.md`.

## 1. Data and label

PAD-UFES-20: 2,298 smartphone images, 1,373 patients, 10 patient-grouped stratified folds (seed 20261107; no patient in two folds). Primary label `has_malignant` = basal cell carcinoma, squamous cell carcinoma or melanoma (1,089 images). The label that also counts actinic keratosis (`has_premalignant`) is stored and not analysed. Subgroups: age 65+ and 75+ (a patient attribute).

Fitzpatrick17k (3,887 verified images, 702 malignant) is a secondary check with Fitzpatrick skin type as subgroup, scored with the PAD-trained models.

## 2. Eligibility

Smartphone ownership (propensity model `phone`), engagement tilt, repair target = pool, as in the primary Task A configuration except for the device. Unknown sex (804 images, 35%) enters the eligibility model as 0.5; a missing age is replaced by the cohort median for the eligibility draw only. Arms: uniform, size_matched, availability, repr_aware, post_hoc; volume control recruit_random.

## 3. Training

CompactDermNet (302,425 parameters), 128 x 128 images, K = 128 participants per round, batch 64, AdamW, cosine schedule, bf16, identical step budget across arms, best-validation checkpoint, no early stopping. Training images are flipped horizontally, vertically and transposed at random (all arms alike).

Tuning on the uniform arm by validation loss (seeds 900 to 902, test folds 1, 5, 9; learning rate {1e-3, 3e-3} x epoch-equivalents {20, 40, 80}); mean best validation loss:

| learning rate | 20 | 40 | 80 |
|---|---|---|---|
| 1e-3 | 0.5582 | 0.5613 | 0.5652 |
| 3e-3 | 0.5591 | 0.5527 | 0.5572 |

Selected: learning rate 3e-3, 40 epoch-equivalents (`DEFAULT_TRAIN_DERM` in `modelling/trainer_derm.py`).

## 4. Suites

| Suite | Jobs | Content |
|---|---|---|
| derm_primary | 1,000 | 5 arms x 20 seeds x 10 folds |
| derm_frontier | 600 | repr_aware at boost 0.25, 0.5, 0.75 |
| derm_volume | 200 | recruit_random |
| derm_sens_none, _consent, _shallow, _steep, _stream | 400 each | 10 seeds x 4 arms |

Total 3,800.

## 5. Analysis

Same estimands, pooled out-of-fold predictions, paired patient-level bootstrap (1,373 patients) and contrasts as Task A. Task B is descriptive: the equivalence flag printed by the analysis script is not interpreted, and precision is stated by the observed interval widths.

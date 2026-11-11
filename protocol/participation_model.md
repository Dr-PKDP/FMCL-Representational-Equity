# Participation model (HINTS 7)

Source: Health Information National Trends Survey, cycle 7 (2024), public-use file `hints7_public.sav` (National Cancer Institute; open access). About 7,278 respondents.

## Estimation (`participation/fit_hints.py`)

Weighted logistic regression (statsmodels GLM, variance weights `PERSON_FINWT0`) on age (capped at 90, as the PTB-XL top code) and sex (`BirthSex`: 1 = female, 2 = male). Negative values are survey missing codes and are dropped. Standard errors by delete-one jackknife over the replicate weights `PERSON_FINWT1..50`: v = (R - 1)/R x sum_r (b_r - b)^2, R = 50.

Outcomes: `UseDevice_SmPhone` (smartphone), `UseDevice_SmWatch` (smartwatch or wearable), `SharedHealthDeviceInfo2` (shared data from a health device with a clinician in the past year; 1 = yes, 2 = no; other codes excluded).

## Fitted models, logit P = b0 + b_age x age + b_female x female

| Model | b0 | b_age (SE) | 95% CI for b_age | b_female (SE) |
|---|---|---|---|---|
| phone | 7.2520 | -0.07774 (0.01627) | -0.1096, -0.0459 | +0.1730 (0.2059) |
| watch | 0.8638 | -0.03055 (0.00273) | -0.0359, -0.0252 | +0.2749 (0.0974) |
| shared (engagement) | -2.0813 | +0.01652 (0.00282) | +0.0110, +0.0221 | +0.0131 (0.1105) |

On the complete-case sample with valid income and education (6,240 respondents), adding household income range and education level as linear terms changes b_age of the phone model from -0.0756 to -0.0717 (5.1%), and of the watch model from -0.0294 to -0.0326 (`participation/hints_descriptives.py`, `results/participation/hints_descriptives.json`). The coefficients are the constants `MODELS` in `participation/propensity.py`. The confidence limits of b_age define the shallow and steep sensitivity bounds (`age_bound` in `propensity.prob`).

## Weighted descriptives by age band (under 65 / 65-74 / 75-89 / 90+)

| Quantity | Under 65 | 65-74 | 75-89 | 90+ |
|---|---|---|---|---|
| Smartphone use | 97.6% | 90.1% | 68.5% | 41.2% |
| Wearable use | 44.8% | 20.2% | 13.0% | 10.7% |
| Shared device data with a clinician | 19.9% | 28.9% | 35.4% | 28.5% (n = 41) |

## Use in the experiments

Eligibility is a persistent per-patient draw from the device-use model of the task's signal device (wearable for ECG, smartphone for dermatology). Engagement enters in one of three readings: `tilt` (primary; per-round selection weight among eligible patients), `none` (device use only) and `consent` (a patient is eligible only if a single uniform draw falls below the product of the device-use and engagement probabilities; persistent).

Implied composition on PTB-XL ages (21,799 records; 65+ share 43.5%, 75+ share 21.1%; mean of 200 seeded draws):

| Device | Engagement | Pool | Pool 65+ | Stream 65+ | Full-repair recruits |
|---|---|---|---|---|---|
| wearable | none | 31% | 30.7% | 30.7% | 1,546 |
| wearable | tilt | 31% | 30.7% | 38.5% | 467 |
| wearable | consent | 7% | 38.7% | 38.7% | 140 |
| smartphone | none | 89% | 39.2% | 39.2% | 1,483 |
| smartphone | tilt | 89% | 39.2% | 47.0% | 602 |
| smartphone | consent | 22% | 47.1% | 47.1% | 510 |

"Pool" is the set of patients who can ever participate; "stream" is the composition of what is sampled per round.

## Scope

The transfer from a US general-population survey of device use and past-year data sharing to a field-participation propensity is an assumption; the three engagement readings and the two coefficient bounds are the sensitivity analyses. Engagement is past-year sharing of health-device data with a clinician (`SharedHealthDeviceInfo2`), not willingness to contribute to model training, so its absolute level (17% to 36%) is not a consent probability.

"""
propensity.py

Participation model fitted on HINTS 7 (2024): weighted logistic regression on age (capped
at 90) and sex, point estimates from PERSON_FINWT0, standard errors by the delete-one
jackknife with the 50 replicate weights, v = (R-1)/R * sum((b_r - b)^2).

Three fitted models, all  logit P = intercept + b_age * age + b_female * female :
    phone    use of a smartphone
    watch    use of a smartwatch or wearable
    shared   shared data from a health device with a clinician in the past year
             (engagement among respondents who answered yes/no)

Eligibility (does the patient own the device that produces the task's signal?) uses an
ownership model. Engagement is a per-round selection tilt among eligible patients only.
Which device defines eligibility is a task-level choice, set in CONFIG, not here:
ECG -> wearable, dermatology -> smartphone (the device that generates the signal).
"""
import numpy as np

# name: (intercept, age coef, female coef, 95% CI on age coef)
MODELS = {
    "phone":  (7.2520, -0.07774, 0.1730, (-0.1096, -0.0459)),
    "watch":  (0.8638, -0.03055, 0.2749, (-0.0359, -0.0252)),
    "shared": (-2.0813, 0.01652, 0.0131, (0.0110, 0.0221)),
}
AGE_CAP = 90

# Functional-form sensitivity (age_bound="band"): six age bands instead of linear age, same sex term, fitted by
# fit_hints.py.  name: (log-odds of the bands 18-34, 35-49, 50-64, 65-74, 75-84, 85-90 for a man, female coef).
# Ages below 18 take the first band; ages are capped at 90.
BAND_EDGES = (35, 50, 65, 75, 85)
BAND_MODELS = {
    "phone":  ((3.8173, 4.1441, 3.2872, 2.1347, 0.8542, -0.0842), 0.1592),
    "watch":  ((-0.1064, -0.2693, -0.6376, -1.4992, -1.9478, -2.3689), 0.2580),
    "shared": ((-1.6142, -1.3596, -1.2592, -0.9246, -0.6210, -0.6251), 0.0200),
}


def _sigmoid(z):
    return 1.0 / (1.0 + np.exp(-z))


def prob(model, age, female, age_bound=None):
    """P(outcome | age, sex). age_bound: 'shallow' / 'steep' replaces the age coefficient by the CI endpoint nearest
    to / furthest from zero; 'band' uses the categorical-age model BAND_MODELS."""
    if age_bound == "band":
        lo, b_f = BAND_MODELS[model]
        a = np.minimum(np.asarray(age, dtype=float), AGE_CAP)
        return _sigmoid(np.asarray(lo)[np.digitize(a, BAND_EDGES)] + b_f * np.asarray(female, dtype=float))
    b0, b_age, b_f, ci = MODELS[model]
    if age_bound == "shallow":
        b_age = min(ci, key=abs)
    elif age_bound == "steep":
        b_age = max(ci, key=abs)
    age = np.minimum(np.asarray(age, dtype=float), AGE_CAP)
    return _sigmoid(b0 + b_age * age + b_f * np.asarray(female, dtype=float))

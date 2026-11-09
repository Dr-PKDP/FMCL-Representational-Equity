"""
preprocess_chapman.py

Builds the replication cache for Task A from the PhysioNet combined release "ecg-arrhythmia" 1.0.0
(Chapman-Shaoxing + Ningbo, 45,152 records, 500 Hz, 10 s, 12 leads).

Output (in --out-dir):
    chapman_100hz_signals.npy   float32, (N, 12, 1000), same lead order and normalisation as the PTB-XL cache
    chapman_meta.csv            one row per header-table row
    chapman_preprocess_report.json

Decisions:
  * Signals are read from the .mat file ('val', 12 x 5000, ADC counts). Per-lead z-scoring removes the gain, so no
    gain is needed. Lead names are read from each .hea and the channels are put into the PTB-XL order by name.
  * Resampling 500 to 100 Hz with an anti-aliasing polyphase filter (exactly 1,000 samples).
  * Label: conduction-disturbance (CD) = SNOMED 270492004 (1st-degree AV block), 59118001 (RBBB) or 164909002 (LBBB).
  * Age 0 and missing age are invalid: age_valid = 0, excluded from age strata (the records stay in overall analyses).
    sex: 1 = female, 0 = male, -1 = unknown.
  * No patient identifiers exist: `patient_id` equals the record id, so any bootstrap is record-level.
  * Site: JS numbers up to 10646 are attributed to Shaoxing, above to Ningbo. This is INFERRED from numbering and
    counts, stored as `site_inferred`.
  * A record whose signal cannot be read, has the wrong shape or non-finite values is unusable: loaded = 0, counted.
    Strict: the run aborts if more than 1% are unusable or if a header-table record has no file.

Usage:
    python3 preprocess_chapman.py --root <ecg-arrhythmia/1.0.0> --headers <chapman_headers.csv> --out-dir <cache>
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.io import loadmat
from scipy.signal import resample_poly

LEADS = ["I", "II", "III", "AVR", "AVL", "AVF", "V1", "V2", "V3", "V4", "V5", "V6"]
CD_CODES = {"270492004", "59118001", "164909002"}
SHAOXING_MAX = 10646


def _norm_name(s):
    return s.strip().upper().replace("AVR", "AVR")


def read_one(args):
    i, mat_path = args
    try:
        hea = mat_path[:-4] + ".hea"
        names = []
        for l in open(hea).read().splitlines()[1:13]:
            t = l.split()
            if t: names.append(_norm_name(t[-1]))
        if sorted(names) != sorted(LEADS):
            return i, None, f"lead names {names}"
        x = np.asarray(loadmat(mat_path)["val"], dtype=np.float64)
        if x.shape != (12, 5000):
            return i, None, f"shape {x.shape}"
        x = x[[names.index(n) for n in LEADS]]
        y = resample_poly(x, 1, 5, axis=1)
        if y.shape != (12, 1000) or not np.isfinite(y).all():
            return i, None, "non-finite or wrong length after resampling"
        mu, sd = y.mean(axis=1, keepdims=True), y.std(axis=1, keepdims=True)
        flat = sd[:, 0] < 1e-9
        y = (y - mu) / np.where(sd < 1e-9, 1.0, sd); y[flat] = 0.0
        return i, y.astype(np.float32), int(flat.sum())
    except Exception as e:
        return i, None, f"{type(e).__name__}: {e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True); ap.add_argument("--headers", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=8); ap.add_argument("--limit", type=int, default=None)
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True); t0 = time.time()

    h = pd.read_csv(a.headers)
    if a.limit: h = h.iloc[:a.limit].copy()
    N = len(h)
    where = {}
    for r, _, files in os.walk(os.path.join(a.root, "WFDBRecords")):
        for f in files:
            if f.endswith(".mat"): where[f[:-4]] = os.path.join(r, f)
    missing = [r for r in h.record if r not in where]
    if missing:
        print(f"ABORT: {len(missing)} header-table records have no .mat file, first: {missing[:5]}"); sys.exit(1)
    print(f"{N} records; reading with {a.workers} workers", flush=True)

    sig_path = os.path.join(a.out_dir, "chapman_100hz_signals.npy")
    arr = np.lib.format.open_memmap(sig_path, mode="w+", dtype=np.float32, shape=(N, 12, 1000))
    done = np.zeros(N, dtype=bool); nflat = np.zeros(N, dtype=int); errs = []
    with Pool(a.workers) as pool:
        for k, (i, y, info) in enumerate(pool.imap_unordered(read_one, [(i, where[r]) for i, r in enumerate(h.record)], chunksize=64)):
            if y is None: errs.append((h.record.iloc[i], info)); continue
            arr[i] = y; done[i] = True; nflat[i] = info
            if (k + 1) % 10000 == 0: print(f"  {k+1}/{N} ({time.time()-t0:.0f}s)", flush=True)
    arr.flush()
    if len(errs) > 0.01 * N:
        print(f"ABORT: {len(errs)} unusable records (limit 1%); first: {errs[:5]}"); sys.exit(1)

    num = h.record.str[2:].astype(int)
    codes = h.dx.fillna("").apply(lambda s: set(s.split(";")))
    age = h.age.astype(float); age_valid = (age > 0) & age.notna()
    sex = h.sex.map({"Male": 0, "Female": 1}).fillna(-1).astype(int)
    meta = pd.DataFrame({
        "row_idx": np.arange(N), "record": h.record.values, "patient_id": h.record.values,
        "age_raw": age.values, "age_valid": age_valid.astype(int).values, "age_capped": np.minimum(age, 90).values,
        "sex": sex.values, "site_inferred": np.where(num <= SHAOXING_MAX, "Shaoxing", "Ningbo"),
        "has_cd": codes.apply(lambda s: int(len(s & CD_CODES) > 0)).values,
        "av1": codes.apply(lambda s: int("270492004" in s)).values, "rbbb": codes.apply(lambda s: int("59118001" in s)).values,
        "lbbb": codes.apply(lambda s: int("164909002" in s)).values,
        "elderly65": ((age >= 65) & age_valid).astype(int).values, "elderly75": ((age >= 75) & age_valid).astype(int).values,
        "n_flat_leads": nflat, "loaded": done.astype(int)})
    meta.to_csv(os.path.join(a.out_dir, "chapman_meta.csv"), index=False)

    v = meta[(meta.age_valid == 1) & (meta.loaded == 1)]
    band = pd.cut(v.age_raw, [0, 64.999, 74.999, 200], labels=["<65", "65-74", "75+"])
    prev = lambda d, b: {str(k): round(float(x), 4) for k, x in d.groupby(b, observed=True).has_cd.mean().items()}
    report = {"n_records": int(N), "n_loaded": int(done.sum()), "n_unusable": len(errs), "unusable_examples": errs[:5],
              "cd_positive": int(meta.has_cd.sum()), "cd_by_code": {k: int(meta[k].sum()) for k in ("av1", "rbbb", "lbbb")},
              "age_valid_records": int(meta.age_valid.sum()), "sex_counts": {int(k): int(c) for k, c in meta.sex.value_counts().items()},
              "site_counts": meta.site_inferred.value_counts().to_dict(),
              "cd_prevalence_by_age_all": prev(v, band),
              "cd_prevalence_by_age_Shaoxing": prev(v[v.site_inferred == "Shaoxing"], band[v.site_inferred == "Shaoxing"]),
              "cd_prevalence_by_age_Ningbo": prev(v[v.site_inferred == "Ningbo"], band[v.site_inferred == "Ningbo"]),
              "expected_record01": {"cd_positive": 1957, "av1/rbbb/lbbb": [1140, 649, 240], "age_valid_n": 44850,
                                    "all": [0.022, 0.054, 0.092], "Shaoxing": [0.041, 0.094, 0.181], "Ningbo": [0.016, 0.041, 0.066],
                                    "sites": {"Shaoxing": 10247, "Ningbo": 34905}},
              "records_with_flat_lead": int((nflat > 0).sum()), "seconds": round(time.time() - t0, 1)}
    json.dump(report, open(os.path.join(a.out_dir, "chapman_preprocess_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    chk = np.load(sig_path, mmap_mode="r"); s = chk[np.where(done)[0][:: max(1, int(done.sum()) // 200)]]
    assert chk.shape == (N, 12, 1000) and np.isfinite(s).all(); print("cache self-check passed")


if __name__ == "__main__":
    main()

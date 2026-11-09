"""
preprocess_ptbxl.py

Builds the training cache for Task A from the raw PTB-XL release.

Output (in --out-dir):
    ptbxl_100hz_signals.npy   float32, shape (N, 12, 1000): 12 leads, 10 s at 100 Hz
    ptbxl_meta.csv            one row per signal row, in the same order
    ptbxl_preprocess_report.json

What is decided here, and why:
  * 100 Hz records (records100), which is the rate the compact model takes (1000 samples).
  * Per-record, per-lead z-scoring. It uses no statistics from any other record, so no
    information can pass between training and test folds. Flat leads stay at zero.
  * Label: conduction-disturbance (CD) diagnostic superclass, via scp_statements.csv,
    exactly as in data_ptbxl.py. The other four superclasses are kept as columns.
  * Age is kept raw (300 = PTB-XL's code for ages above 89) and as age_capped (300 -> 90).
    The elderly flags count age 300 as elderly, so these patients stay in the oldest stratum.
  * strat_fold is copied unchanged. PTB-XL's folds are patient-stratified (checked on the
    real table: no patient appears in more than one fold).
  * Strict by default: any unreadable or wrongly shaped record aborts the run.

Usage:
    python3 preprocess_ptbxl.py --root <ptb-xl/1.0.3 dir> \
        --out-dir <cache>
"""
import argparse, ast, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd

N_LEADS, T_SAMPLES, FS = 12, 1000, 100
SUPERCLASSES = ["NORM", "MI", "STTC", "CD", "HYP"]


def read_one(args):
    """Read one record; return (row_idx, signal(12,T) float32 or None, n_nan, n_flat, sig_name, error)."""
    import wfdb
    row_idx, path = args
    try:
        sig, fields = wfdb.rdsamp(path)
    except Exception as e:                      # missing or unreadable file
        return row_idx, None, 0, 0, None, f"{type(e).__name__}: {e}"
    if sig.shape != (T_SAMPLES, N_LEADS) or int(fields["fs"]) != FS:
        return row_idx, None, 0, 0, None, f"unexpected shape/fs: {sig.shape} @ {fields['fs']} Hz"
    sig = sig.astype(np.float64)               # statistics in float64: float32 rounding makes a
    n_nan = int(np.isnan(sig).sum())           # constant lead look non-constant (std ~7e-6)
    if n_nan:
        sig = np.nan_to_num(sig, nan=0.0)
    mu = sig.mean(axis=0, keepdims=True)
    sd = sig.std(axis=0, keepdims=True)
    flat = sd[0] < 1e-6
    sig = (sig - mu) / np.where(sd < 1e-6, 1.0, sd)
    sig[:, flat] = 0.0
    return row_idx, np.ascontiguousarray(sig.T.astype(np.float32)), n_nan, int(flat.sum()), list(fields["sig_name"]), None


def build_meta(root, limit=None):
    db = pd.read_csv(os.path.join(root, "ptbxl_database.csv"))
    scp = pd.read_csv(os.path.join(root, "scp_statements.csv"), index_col=0)
    diag = scp[scp.diagnostic == 1]
    if limit:
        db = db.iloc[:limit].copy()

    def supers(s):
        try:
            codes = ast.literal_eval(s)
        except Exception:
            return set()
        out = set()
        for c in codes:
            if c in diag.index and pd.notna(diag.loc[c, "diagnostic_class"]):
                out.add(diag.loc[c, "diagnostic_class"])
        return out

    sc = db.scp_codes.apply(supers)
    meta = pd.DataFrame({
        "row_idx": np.arange(len(db)),
        "ecg_id": db.ecg_id.values,
        "patient_id": db.patient_id.astype(np.int64).values,
        "age_raw": db.age.values,
        "age_capped": db.age.apply(lambda a: 90 if a == 300 else min(a, 90)).values,
        "sex": db.sex.values,                    # PTB-XL: 0 = male, 1 = female
        "strat_fold": db.strat_fold.values,
        "site": db.site.values,
        "device": db.device.values,
        "filename_lr": db.filename_lr.values,
    })
    for k in SUPERCLASSES:
        meta[k.lower()] = sc.apply(lambda s: k in s).astype(int).values
    meta["has_cd"] = meta["cd"]
    meta["elderly65"] = db.age.apply(lambda a: int(a == 300 or a >= 65)).values
    meta["elderly75"] = db.age.apply(lambda a: int(a == 300 or a >= 75)).values
    return meta


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", required=True, help="PTB-XL 1.0.3 directory (contains ptbxl_database.csv, records100/)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=max(1, min(32, (os.cpu_count() or 2) - 2)))
    ap.add_argument("--limit", type=int, default=None, help="process only the first N records (testing)")
    ap.add_argument("--allow-missing", action="store_true", help="drop unreadable records instead of aborting")
    a = ap.parse_args()

    os.makedirs(a.out_dir, exist_ok=True)
    t0 = time.time()
    meta = build_meta(a.root, a.limit)
    N = len(meta)
    print(f"{N} records in the database table; reading with {a.workers} workers ...", flush=True)

    sig_path = os.path.join(a.out_dir, "ptbxl_100hz_signals.npy")
    arr = np.lib.format.open_memmap(sig_path, mode="w+", dtype=np.float32, shape=(N, N_LEADS, T_SAMPLES))
    jobs = [(int(i), os.path.join(a.root, fn)) for i, fn in zip(meta.row_idx, meta.filename_lr)]

    n_nan_tot = n_flat_tot = 0
    nan_per = np.zeros(N, dtype=int); flat_per = np.zeros(N, dtype=int)
    errors, names, done = [], set(), np.zeros(N, dtype=bool)
    with Pool(a.workers) as pool:
        for k, (i, sig, n_nan, n_flat, sig_name, err) in enumerate(pool.imap_unordered(read_one, jobs, chunksize=64)):
            if err:
                errors.append((int(i), meta.filename_lr[i], err)); continue
            arr[i] = sig; done[i] = True
            nan_per[i], flat_per[i] = n_nan, n_flat
            names.add(tuple(sig_name))
            if (k + 1) % 2000 == 0:
                print(f"  {k+1}/{N}  ({time.time()-t0:.0f}s)", flush=True)
    arr.flush()

    if errors:
        print(f"\n{len(errors)} records failed; first 5:")
        for e in errors[:5]: print("  ", e)
        if not a.allow_missing:
            print("Aborting (strict mode). Fix the files, or rerun with --allow-missing to drop them.")
            sys.exit(1)
    if len(names) > 1:
        print("ERROR: lead order differs between records:", names); sys.exit(1)

    meta["n_nan"], meta["n_flat_leads"], meta["loaded"] = nan_per, flat_per, done.astype(int)
    meta.to_csv(os.path.join(a.out_dir, "ptbxl_meta.csv"), index=False)

    report = {
        "n_records": int(N), "n_loaded": int(done.sum()), "n_failed": len(errors),
        "n_patients": int(meta.patient_id.nunique()),
        "lead_order": list(next(iter(names))) if names else None,
        "records_with_nan": int((nan_per > 0).sum()), "records_with_flat_lead": int((flat_per > 0).sum()),
        "superclass_record_counts": {k: int(meta[k.lower()].sum()) for k in SUPERCLASSES},
        "cd_positive_records": int(meta.has_cd.sum()),
        "elderly65_records": int(meta.elderly65.sum()), "elderly75_records": int(meta.elderly75.sum()),
        "age_300_records": int((meta.age_raw == 300).sum()),
        "normalisation": "per-record per-lead z-score; flat leads set to 0",
        "shape": [int(N), N_LEADS, T_SAMPLES], "fs_hz": FS,
        "seconds": round(time.time() - t0, 1),
    }
    json.dump(report, open(os.path.join(a.out_dir, "ptbxl_preprocess_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))

    # self-check on the written cache
    chk = np.load(sig_path, mmap_mode="r")
    sample = chk[np.where(done)[0][:: max(1, int(done.sum()) // 200)]]
    assert chk.shape == (N, N_LEADS, T_SAMPLES) and np.isfinite(sample).all()
    print("cache self-check passed")


if __name__ == "__main__":
    main()

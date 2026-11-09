"""
preprocess_code15.py

Builds the external-validation cache for Task A from the CODE-15% release (Zenodo 4916206).

Output (in --out-dir):
    code15_100hz_signals.npy   float32, (N, 12, 1000): 12 leads, 10 s at 100 Hz, same lead order as the PTB-XL cache
    code15_meta.csv            one row per signal row, in exams.csv order
    code15_preprocess_report.json

What is decided here:
  * The archives hold HDF5 files with `tracings` (n, 4096, 12) and `exam_id`. The dataset description gives 400 Hz
    and zero padding on both sides. Measured on archive 0 the real spans are NOT the 2,800 / 4,000 samples of the
    description: about 2,934 samples (7.3 s, centred, 581 zeros each side) and the full 4,096 (10.2 s), with a
    few other spans. So the span (first to last non-zero sample) is measured per trace and used as is.
  * Traces with a span under 2,000 samples (5 s) or all zero are unusable: loaded = 0, counted in the report,
    and dropped by the cohort loader. Traces in the archives that are absent from exams.csv are ignored and counted.
  * Resampling 400 to 100 Hz with an anti-aliasing polyphase filter on the span; at most the first 10 s (1,000
    samples) are kept; a shorter trace is z-scored on its span and zero-padded at the end. `span_samples` is saved
    so results can be stratified by trace length.
  * Per-record, per-lead z-score in float64 (as for PTB-XL); flat leads set to 0.
  * Lead order DI, DII, DIII, AVR, AVL, AVF, V1 to V6 equals the PTB-XL order.
  * Label: conduction-disturbance analogue = 1dAVb or RBBB or LBBB. sex: 1 = female (as PTB-XL).
  * Strict: the run aborts if an exam listed in exams.csv appears in no archive, or if more than 1% of exams are unusable.

Usage:
    python3 preprocess_code15.py --raw <code15 dir> --out-dir <cache> [--workers 6] [--limit-parts 1]
"""
import argparse, glob, json, os, shutil, sys, tempfile, time, zipfile
from multiprocessing import Pool

import numpy as np
import pandas as pd
from scipy.signal import resample_poly

T_OUT, N_LEADS, DOWN, MIN_SPAN = 1000, 12, 4, 2000


def span_of(x):
    nz = np.flatnonzero(np.abs(x).sum(axis=1) > 0)
    return (0, 0) if len(nz) == 0 else (int(nz[0]), int(nz[-1]) + 1)


def process_trace(x):
    """x: (4096, 12) at 400 Hz. Returns (signal (12, 1000) float32, span_samples, n_flat), or raises ValueError."""
    a, b = span_of(x)
    if b - a < MIN_SPAN:
        raise ValueError("all zero" if b == a else f"span {b - a}")
    y = resample_poly(x[a:b].astype(np.float64), 1, DOWN, axis=0)
    y = y[:T_OUT]
    mu, sd = y.mean(axis=0, keepdims=True), y.std(axis=0, keepdims=True)
    flat = sd[0] < 1e-9
    y = (y - mu) / np.where(sd < 1e-9, 1.0, sd)
    y[:, flat] = 0.0
    out = np.zeros((T_OUT, N_LEADS)); out[:len(y)] = y
    return np.ascontiguousarray(out.T.astype(np.float32)), b - a, int(flat.sum())


def do_part(args):
    zpath, sig_path, row_of, tmp_root = args
    import h5py
    tmp = tempfile.mkdtemp(dir=tmp_root)
    try:
        with zipfile.ZipFile(zpath) as z:
            names = [n for n in z.namelist() if n.endswith((".hdf5", ".h5"))]
            assert len(names) == 1, f"{zpath}: expected one HDF5 file, found {names}"
            z.extract(names[0], tmp)
        h5 = os.path.join(tmp, names[0])
        arr = np.load(sig_path, mmap_mode="r+")
        lens, flats, seen, bad, unmatched, ids_all = [], [], [], [], 0, []
        with h5py.File(h5, "r") as f:
            assert f["tracings"].shape[1:] == (4096, 12), f["tracings"].shape
            ids = f["exam_id"][:].astype(np.int64)
            for i0 in range(0, len(ids), 512):
                block = f["tracings"][i0:i0 + 512]
                for k in range(block.shape[0]):
                    eid = int(ids[i0 + k])
                    r = row_of.get(eid)
                    if r is None:
                        unmatched += 1; continue                       # trace without metadata: ignored, counted
                    ids_all.append(eid)
                    try:
                        sig, L, nf = process_trace(block[k])
                    except ValueError as e:
                        bad.append((eid, str(e))); continue             # unusable trace: loaded = 0
                    arr[r] = sig; lens.append((eid, L)); flats.append(nf); seen.append(eid)
        arr.flush()
        return os.path.basename(zpath), seen, lens, flats, bad, unmatched, ids_all
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", required=True, help="directory with exams.csv and exams_part*.zip")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--limit-parts", type=int, default=None, help="process only the first N archives (test)")
    ap.add_argument("--tmp", default=None, help="scratch directory for unpacked HDF5 (default: out-dir/tmp)")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)
    tmp_root = a.tmp or os.path.join(a.out_dir, "tmp"); os.makedirs(tmp_root, exist_ok=True)
    t0 = time.time()

    ex = pd.read_csv(os.path.join(a.raw, "exams.csv"))
    zips = sorted(glob.glob(os.path.join(a.raw, "exams_part*.zip")), key=lambda p: int(os.path.basename(p)[10:-4]))
    if a.limit_parts:
        zips = zips[:a.limit_parts]
        ex = ex[ex.trace_file.isin([os.path.basename(z).replace(".zip", ".hdf5") for z in zips])].reset_index(drop=True)
    N = len(ex)
    row_of = {int(e): i for i, e in enumerate(ex.exam_id.values)}
    print(f"{N} exams in metadata, {len(zips)} archives, {a.workers} workers", flush=True)

    sig_path = os.path.join(a.out_dir, "code15_100hz_signals.npy")
    arr = np.lib.format.open_memmap(sig_path, mode="w+", dtype=np.float32, shape=(N, N_LEADS, T_OUT)); del arr

    done = np.zeros(N, dtype=bool); present = np.zeros(N, dtype=bool); vlen = np.zeros(N, dtype=int); nflat = np.zeros(N, dtype=int)
    bad_all, unmatched_all = [], 0
    with Pool(a.workers) as pool:
        for k, (name, seen, lens, flats, bad, unmatched, ids_all) in enumerate(pool.imap_unordered(do_part, [(z, sig_path, row_of, tmp_root) for z in zips])):
            for e, L in lens: vlen[row_of[e]] = L
            for e, nf in zip(seen, flats): nflat[row_of[e]] = nf; done[row_of[e]] = True
            bad_all += bad; unmatched_all += unmatched
            for e in ids_all: present[row_of[e]] = True
            print(f"  {name}: {len(seen)} ok, {len(bad)} unusable, {unmatched} not in exams.csv  ({k+1}/{len(zips)}, {time.time()-t0:.0f}s)", flush=True)
    shutil.rmtree(tmp_root, ignore_errors=True)

    vc = pd.Series(vlen[done]).value_counts()
    print("span distribution (samples at 400 Hz), top 6:\n", vc.head(6).to_string())
    n_missing = int((~present).sum()); n_unusable = int((present & ~done).sum())
    reasons = pd.Series([r.split(" ")[0] for _, r in bad_all]).value_counts().to_dict()
    print(f"exams in no archive: {n_missing}; unusable traces: {n_unusable} {reasons}; traces without metadata (ignored): {unmatched_all}")
    if n_missing and not a.limit_parts:
        print("ABORT: exams listed in exams.csv were not found in any archive"); sys.exit(1)
    if n_unusable > 0.01 * N:
        print(f"ABORT: {100*n_unusable/N:.2f}% of exams are unusable (limit 1%)"); sys.exit(1)

    cd = (ex["1dAVb"].astype(bool) | ex["RBBB"].astype(bool) | ex["LBBB"].astype(bool)).astype(int)
    age = ex.age.astype(float)
    meta = pd.DataFrame({
        "row_idx": np.arange(N), "exam_id": ex.exam_id.values, "patient_id": ex.patient_id.values, "age_raw": age.values,
        "age_capped": np.minimum(age, 90).values, "sex": (~ex.is_male.astype(bool)).astype(int).values,
        "trace_file": ex.trace_file.values, "has_cd": cd.values,
        "av1": ex["1dAVb"].astype(int).values, "rbbb": ex["RBBB"].astype(int).values, "lbbb": ex["LBBB"].astype(int).values,
        "elderly65": (age >= 65).astype(int).values, "elderly75": (age >= 75).astype(int).values,
        "span_samples_400hz": vlen, "n_flat_leads": nflat, "loaded": done.astype(int)})
    meta.to_csv(os.path.join(a.out_dir, "code15_meta.csv"), index=False)

    band = pd.cut(age, [-1, 64.999, 74.999, 200], labels=["<65", "65-74", "75+"])
    prev = {str(k): round(float(v), 4) for k, v in meta.groupby(band, observed=True).has_cd.mean().items()}
    report = {"n_exams": int(N), "n_loaded": int(done.sum()), "n_patients": int(meta.patient_id.nunique()),
              "cd_positive": int(cd.sum()), "cd_prevalence_by_age": prev, "age_missing": int(age.isna().sum()),
              "span_counts": {int(k): int(v) for k, v in vc.head(10).items()}, "unusable_traces": n_unusable, "unusable_reasons": reasons,
              "traces_without_metadata": int(unmatched_all), "exams_in_no_archive": n_missing,
              "records_with_flat_lead": int((nflat > 0).sum()), "shape": [int(N), N_LEADS, T_OUT],
              "expected_prevalence_record01": {"<65": 0.029, "65-74": 0.090, "75+": 0.136}, "seconds": round(time.time() - t0, 1)}
    json.dump(report, open(os.path.join(a.out_dir, "code15_preprocess_report.json"), "w"), indent=2)
    print(json.dumps(report, indent=2))
    chk = np.load(sig_path, mmap_mode="r"); s = chk[:: max(1, N // 200)]
    assert chk.shape == (N, N_LEADS, T_OUT) and np.isfinite(s).all(); print("cache self-check passed")


if __name__ == "__main__":
    main()

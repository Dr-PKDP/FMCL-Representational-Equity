"""
preprocess_derm.py

Builds the Task B caches: PAD-UFES-20 (development, smartphone images) and Fitzpatrick17k (secondary check).

    python3 preprocess_derm.py pad  --root <pad_mendeley dir> --metadata <metadata.csv> --out-dir <cache>
    python3 preprocess_derm.py fitz --root <fitzpatrick17k dir> --out-dir <cache>

Output: pad_ufes20_128.npy / pad_meta.csv and fitz17k_128.npy / fitz_meta.csv, float32 (N, 3, 128, 128).

Decisions:
  * Every image is decoded and checked; PAD-UFES-20 aborts if any of the 2,298 images cannot be decoded.
  * Letterbox: the longest side is scaled to 128 and the image is centred on a square canvas, so no lesion is cropped.
    Each image is standardised per channel on its own pixels (no statistics cross images or folds); padding is 0.
  * PAD-UFES-20 label (primary): malignant = BCC, SCC or MEL (`has_malignant`). The supplementary label
    `has_premalignant` also counts ACK (actinic keratosis). The other diagnoses are NEV and SEK.
  * Folds (`fold`, 1 to 10): StratifiedGroupKFold, groups = patient_id, stratified on `has_malignant`, seed 20261107,
    so no patient appears in two folds.
  * Sex: 1 = female, 0 = male, -1 = unknown (35% of PAD-UFES-20 records have no gender or skin type; same records).
  * Fitzpatrick17k: only images that are present and decode are used (the verified set); label from
    `three_partition_label` (malignant / benign / non-neoplastic); skin type from `fitzpatrick_scale` (-1 = unknown).
"""
import argparse, json, os, sys, time
from multiprocessing import Pool

import numpy as np
import pandas as pd
from PIL import Image

S = 128


def load_image(path):
    """Return float32 (3, S, S), standardised per channel on the image's own pixels, padding = 0; or raise."""
    im = Image.open(path); im.load()
    im = im.convert("RGB")
    w, h = im.size
    if w < 16 or h < 16:
        raise ValueError(f"image too small {w}x{h}")
    sc = S / max(w, h)
    nw, nh = max(1, round(w * sc)), max(1, round(h * sc))
    x = np.asarray(im.resize((nw, nh), Image.LANCZOS), dtype=np.float64)       # (nh, nw, 3)
    mu, sd = x.reshape(-1, 3).mean(0), x.reshape(-1, 3).std(0)
    x = (x - mu) / np.where(sd < 1e-6, 1.0, sd)
    out = np.zeros((S, S, 3)); y0, x0 = (S - nh) // 2, (S - nw) // 2
    out[y0:y0 + nh, x0:x0 + nw] = x
    return np.ascontiguousarray(out.transpose(2, 0, 1).astype(np.float32))


def _work(a):
    i, p = a
    try:
        return i, load_image(p), None
    except Exception as e:
        return i, None, f"{type(e).__name__}: {e}"


def run(paths, out_npy, workers):
    N = len(paths)
    arr = np.lib.format.open_memmap(out_npy, mode="w+", dtype=np.float32, shape=(N, 3, S, S))
    ok = np.zeros(N, dtype=bool); errs = []
    with Pool(workers) as pool:
        for k, (i, x, err) in enumerate(pool.imap_unordered(_work, list(enumerate(paths)), chunksize=8)):
            if x is None: errs.append((os.path.basename(paths[i]), err)); continue
            arr[i] = x; ok[i] = True
            if (k + 1) % 500 == 0: print(f"  {k+1}/{N}", flush=True)
    arr.flush()
    return ok, errs


def index_files(root, exts):
    d = {}
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(exts): d.setdefault(f, os.path.join(r, f))
    return d


def do_pad(a):
    t0 = time.time()
    m = pd.read_csv(a.metadata)
    files = index_files(a.root, (".png",))
    missing = [x for x in m.img_id if x not in files]
    if missing:
        print(f"ABORT: {len(missing)} images in metadata not found, first: {missing[:5]}"); sys.exit(1)
    ok, errs = run([files[x] for x in m.img_id], os.path.join(a.out_dir, "pad_ufes20_128.npy"), a.workers)
    if errs:
        print(f"ABORT: {len(errs)} images could not be decoded: {errs[:5]}"); sys.exit(1)

    from sklearn.model_selection import StratifiedGroupKFold
    y = m.diagnostic.isin(["BCC", "SCC", "MEL"]).astype(int).values
    fold = np.zeros(len(m), dtype=int)
    sgkf = StratifiedGroupKFold(n_splits=10, shuffle=True, random_state=20261107)
    for f, (_, te) in enumerate(sgkf.split(np.zeros(len(m)), y, groups=m.patient_id.values), start=1):
        fold[te] = f
    meta = pd.DataFrame({
        "row_idx": np.arange(len(m)), "img_id": m.img_id.values, "patient_id": m.patient_id.values, "lesion_id": m.lesion_id.values,
        "age_raw": m.age.values, "age_capped": np.minimum(m.age, 90).values,
        "sex": m.gender.map({"MALE": 0, "FEMALE": 1}).fillna(-1).astype(int).values,
        "fitzpatrick": m.fitspatrick.fillna(-1).astype(int).values, "diagnostic": m.diagnostic.values,
        "has_malignant": y, "has_premalignant": m.diagnostic.isin(["BCC", "SCC", "MEL", "ACK"]).astype(int).values,
        "elderly65": (m.age >= 65).astype(int).values, "elderly75": (m.age >= 75).astype(int).values,
        "strat_fold": fold, "loaded": ok.astype(int)})
    meta.to_csv(os.path.join(a.out_dir, "pad_meta.csv"), index=False)
    g = meta.groupby("strat_fold")
    leaked = int((meta.groupby("patient_id").strat_fold.nunique() > 1).sum())
    rep = {"n_images": len(m), "n_loaded": int(ok.sum()), "n_patients": int(m.patient_id.nunique()),
           "diagnostic_counts": m.diagnostic.value_counts().to_dict(), "malignant": int(y.sum()),
           "patients_in_more_than_one_fold": leaked,
           "fold_sizes": g.size().to_dict(), "fold_malignant": g.has_malignant.sum().to_dict(),
           "n_65plus": int(meta.elderly65.sum()), "malignant_65plus": int(meta[meta.elderly65 == 1].has_malignant.sum()),
           "n_75plus": int(meta.elderly75.sum()), "malignant_75plus": int(meta[meta.elderly75 == 1].has_malignant.sum()),
           "unknown_sex": int((meta.sex == -1).sum()), "unknown_fitzpatrick": int((meta.fitzpatrick == -1).sum()),
           "sex_values_in_file": m.gender.value_counts(dropna=False).to_dict(), "seconds": round(time.time() - t0, 1)}
    assert leaked == 0
    json.dump(rep, open(os.path.join(a.out_dir, "pad_preprocess_report.json"), "w"), indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))


def do_fitz(a):
    t0 = time.time()
    c = pd.read_csv(os.path.join(a.root, "fitzpatrick17k.csv"))
    files = {os.path.splitext(f)[0]: p for f, p in index_files(os.path.join(a.root, "images"), (".jpg", ".jpeg", ".png")).items()}
    c = c[c.md5hash.isin(files)].reset_index(drop=True)
    print(f"{len(c)} csv rows have an image file in images/ (the quarantine folder is not read)", flush=True)
    ok, errs = run([files[h] for h in c.md5hash], os.path.join(a.out_dir, "fitz17k_128.npy"), a.workers)
    meta = pd.DataFrame({"row_idx": np.arange(len(c)), "md5hash": c.md5hash.values, "fitzpatrick": c.fitzpatrick_scale.values,
                         "label": c.label.values, "three_partition": c.three_partition_label.values,
                         "has_malignant": (c.three_partition_label == "malignant").astype(int).values, "loaded": ok.astype(int)})
    meta.to_csv(os.path.join(a.out_dir, "fitz_meta.csv"), index=False)
    v = meta[meta.loaded == 1]
    rep = {"n_images": len(c), "n_loaded": int(ok.sum()), "undecodable": errs[:5], "n_undecodable": len(errs),
           "malignant": int(v.has_malignant.sum()), "by_skin_type": v.groupby("fitzpatrick").has_malignant.agg(["size", "sum"]).to_dict("index"),
           "three_partition_counts": v.three_partition.value_counts().to_dict(), "seconds": round(time.time() - t0, 1)}
    json.dump(rep, open(os.path.join(a.out_dir, "fitz_preprocess_report.json"), "w"), indent=2, default=str)
    print(json.dumps(rep, indent=2, default=str))


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("which", choices=["pad", "fitz"]); ap.add_argument("--root", required=True)
    ap.add_argument("--metadata"); ap.add_argument("--out-dir", required=True); ap.add_argument("--workers", type=int, default=8)
    a = ap.parse_args(); os.makedirs(a.out_dir, exist_ok=True)
    (do_pad if a.which == "pad" else do_fitz)(a)

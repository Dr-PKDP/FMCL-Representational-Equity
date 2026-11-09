"""
score_external.py: score CODE-15% and Chapman plus Ningbo with every trained primary-suite model.

    python3 score_external.py --results <results/ecg/primary> --cache-dir <cache> --out-dir <analysis dir> --gpu 1

The models (<hash>.pt + <hash>.json in --results) are loaded once; each cohort is streamed from its memory-mapped
cache in chunks, and every chunk is scored by every model, so each cohort is read from disk once.
Outputs (out-dir):
    ext_models.csv               model_idx, label, init_seed, elig_seed, fold, stem
    ext_code15.npy               float32 (n_models, n_code15_rows)  predicted probability, rows as in code15_meta.csv
    ext_chapman.npy              float32 (n_models, n_chapman_rows) rows as in chapman_meta.csv
Rows with loaded == 0 are scored too (all zeros in, meaningless out); the analysis drops them.
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "modelling"))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--chunk", type=int, default=4096)
    ap.add_argument("--cohorts", nargs="+", default=["code15", "chapman"])
    a = ap.parse_args()
    os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from models import CompactECGNet
    from trainer import PatientSet  # noqa: F401  (import check only)
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from ecg_analysis import label_of
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.allow_tf32 = True

    rows, models = [], []
    for jp in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        pt = jp[:-5] + ".pt"
        if not os.path.exists(pt): continue
        j = json.load(open(jp))["job"]
        m = CompactECGNet(); m.load_state_dict(torch.load(pt, map_location="cpu")); m.to(dev).eval()
        rows.append(dict(model_idx=len(rows), label=label_of(j), init_seed=j["init_seed"], elig_seed=j["elig_seed"], fold=j["fold"], stem=os.path.basename(jp)[:-5]))
        models.append(m)
    if not models: raise SystemExit("no .pt files found: the results must come from the trainer that saves weights")
    os.makedirs(a.out_dir, exist_ok=True)
    pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "ext_models.csv"), index=False)
    print(f"{len(models)} models on {dev}", flush=True)

    for name in a.cohorts:
        X = np.load(os.path.join(a.cache_dir, f"{name}_100hz_signals.npy"), mmap_mode="r")
        N = X.shape[0]; out = np.lib.format.open_memmap(os.path.join(a.out_dir, f"ext_{name}.npy"), mode="w+", dtype=np.float32, shape=(len(models), N))
        t0 = time.time()
        for s in range(0, N, a.chunk):
            x = torch.from_numpy(np.ascontiguousarray(X[s:s + a.chunk])).to(dev)
            with torch.no_grad():
                res = torch.stack([torch.sigmoid(m(x).squeeze(1).float()) for m in models]).cpu().numpy()
            out[:, s:s + x.shape[0]] = res
            if (s // a.chunk) % 10 == 0: print(f"  {name} {s}/{N} ({time.time()-t0:.0f}s)", flush=True)
        out.flush(); del out
        print(f"{name} done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

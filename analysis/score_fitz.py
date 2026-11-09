"""
score_fitz.py: score Fitzpatrick17k (verified images) with every PAD-trained model of a Task B suite.

    python3 score_fitz.py --results <results/derm/derm_primary> --cache-dir <cache> --out-dir <analysis dir> --gpu 1

Writes fitz_models.csv (model_idx, label, init_seed, elig_seed, fold, stem) and fitz_scores.npy (float32, n_models x n_images,
rows as in fitz_meta.csv).
"""
import argparse, glob, json, os, sys, time
import numpy as np
import pandas as pd

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, "..", "modelling")); sys.path.insert(0, HERE)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", required=True); ap.add_argument("--cache-dir", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--gpu", type=int, default=0); ap.add_argument("--chunk", type=int, default=512)
    a = ap.parse_args(); os.environ["CUDA_VISIBLE_DEVICES"] = str(a.gpu)
    import torch
    from models import CompactDermNet
    from ecg_analysis import label_of
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows, models = [], []
    for jp in sorted(glob.glob(os.path.join(a.results, "*.json"))):
        pt = jp[:-5] + ".pt"
        if not os.path.exists(pt): continue
        j = json.load(open(jp))["job"]
        m = CompactDermNet(); m.load_state_dict(torch.load(pt, map_location="cpu")); models.append(m.to(dev).eval())
        rows.append(dict(model_idx=len(rows), label=label_of(j), init_seed=j["init_seed"], elig_seed=j["elig_seed"], fold=j["fold"], stem=os.path.basename(jp)[:-5]))
    if not models: raise SystemExit("no .pt files found: run the suite with the trainer that saves weights")
    os.makedirs(a.out_dir, exist_ok=True); pd.DataFrame(rows).to_csv(os.path.join(a.out_dir, "fitz_models.csv"), index=False)
    X = np.load(os.path.join(a.cache_dir, "fitz17k_128.npy"), mmap_mode="r"); N = X.shape[0]
    out = np.zeros((len(models), N), np.float32); t0 = time.time()
    print(f"{len(models)} models on {dev}; {N} images", flush=True)
    for s in range(0, N, a.chunk):
        x = torch.from_numpy(np.array(X[s:s + a.chunk])).to(dev)
        with torch.no_grad():
            out[:, s:s + x.shape[0]] = torch.stack([torch.sigmoid(m(x).squeeze(1).float()) for m in models]).cpu().numpy()
        if (s // a.chunk) % 2 == 0: print(f"  {s}/{N} ({time.time()-t0:.0f}s)", flush=True)
    np.save(os.path.join(a.out_dir, "fitz_scores.npy"), out); print(f"done in {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()

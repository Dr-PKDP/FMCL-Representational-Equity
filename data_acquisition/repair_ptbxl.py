"""
repair_ptbxl.py: find and re-fetch PTB-XL files that are missing, empty or the wrong size.
Expected sizes (PTB-XL 10 s, 12 leads, 16-bit): 100 Hz .dat = 24,000 bytes; 500 Hz .dat = 120,000 bytes.
Usage: python3 repair_ptbxl.py <ptbxl_1.0.3_dir> [--fetch]
"""
import csv, os, sys, urllib.request
root = sys.argv[1]; do_fetch = "--fetch" in sys.argv
SIZE = {"records100": 24000, "records500": 120000}
BASE = "https://physionet.org/files/ptb-xl/1.0.3/"
bad = []
for r in csv.DictReader(open(os.path.join(root, "ptbxl_database.csv"))):
    for col in ("filename_lr", "filename_hr"):
        for ext in (".hea", ".dat"):
            p = r[col] + ext; fp = os.path.join(root, p)
            ok = os.path.exists(fp) and os.path.getsize(fp) > 0
            if ok and ext == ".dat":
                ok = os.path.getsize(fp) == SIZE[p.split("/")[0]]
            if not ok:
                bad.append(p)
print(len(bad), "missing, empty or wrongly sized files")
if do_fetch:
    for p in bad:
        fp = os.path.join(root, p); os.makedirs(os.path.dirname(fp), exist_ok=True)
        if os.path.exists(fp): os.remove(fp)
        try:
            urllib.request.urlretrieve(BASE + p, fp); print("got", p)
        except Exception as e:
            print("FAILED", p, e); os.path.exists(fp) and os.remove(fp)

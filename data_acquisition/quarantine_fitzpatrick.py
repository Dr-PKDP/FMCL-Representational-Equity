"""quarantine_fitzpatrick.py: move every image whose md5 differs from its file name out of the image folder.
Usage: python3 quarantine_fitzpatrick.py <images_dir> <quarantine_dir>"""
import hashlib, os, shutil, sys
D, Q = sys.argv[1], sys.argv[2]; os.makedirs(Q, exist_ok=True); kept = moved = 0
for f in os.listdir(D):
    if f.endswith(".csv"): continue
    p = os.path.join(D, f)
    if hashlib.md5(open(p, "rb").read()).hexdigest() == os.path.splitext(f)[0]: kept += 1
    else: shutil.move(p, os.path.join(Q, f)); moved += 1
print("kept (verified):", kept, "| moved to quarantine:", moved)

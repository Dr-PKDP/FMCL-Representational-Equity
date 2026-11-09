"""
verify_pad_ufes20.py: counts, and byte-level comparison of a mirror against the Mendeley original.
Reference archive: Mendeley Data 10.17632/zr7vgbcyr2.1 (zr7vgbcyr2-1.zip, 3,593,344,681 bytes; images/imgs_part_1..3.zip and metadata.csv).
Usage: python3 verify_pad_ufes20.py <dir_A> [<dir_B>]   (with two dirs: names + md5 comparison)
Expected: 2298 png files; with two dirs: 2298 byte-identical.
"""
import hashlib, os, sys
def index(root):
    d = {}
    for r, _, fs in os.walk(root):
        for f in fs:
            if f.lower().endswith(".png"): d.setdefault(f, []).append(os.path.join(r, f))
    return d
def md5(p):
    h = hashlib.md5()
    with open(p, "rb") as fh:
        for c in iter(lambda: fh.read(1 << 20), b""): h.update(c)
    return h.hexdigest()
A = index(sys.argv[1]); print(len(A), "png names in", sys.argv[1])
if len(sys.argv) > 2:
    B = index(sys.argv[2]); print(len(B), "png names in", sys.argv[2])
    print("only in A:", len(set(A) - set(B)), "| only in B:", len(set(B) - set(A)))
    same = [n for n in sorted(set(A) & set(B)) if md5(A[n][0]) == md5(B[n][0])]
    print("byte-identical:", len(same), "| same name, different bytes:", len(set(A) & set(B)) - len(same))

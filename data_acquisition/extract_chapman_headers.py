"""
extract_chapman_headers.py: age, sex, SNOMED diagnosis codes, rate and length from every .hea header.
Usage: python3 extract_chapman_headers.py <ecg-arrhythmia_1.0.0_dir> <out.csv>
Expected last line: "45152 headers; 0 without a .mat". JS01052 has a malformed first header line.
"""
import csv, os, sys
root, out = sys.argv[1], sys.argv[2]
rows = []
for r, _, files in os.walk(os.path.join(root, "WFDBRecords")):
    for f in files:
        if not f.endswith(".hea"): continue
        rid = f[:-4]; age = sex = dx = fs = ns = ""
        lines = open(os.path.join(r, f)).read().splitlines()
        first = lines[0].split()
        if len(first) >= 4: fs, ns = first[2], first[3]
        for l in lines:
            if l.startswith("#Age:"): age = l.split(":", 1)[1].strip()
            elif l.startswith("#Sex:"): sex = l.split(":", 1)[1].strip()
            elif l.startswith("#Dx:"): dx = l.split(":", 1)[1].strip().replace(",", ";")
        rows.append((rid, age, sex, dx, fs, ns, int(os.path.exists(os.path.join(r, rid + ".mat")))))
rows.sort()
with open(out, "w", newline="") as o:
    w = csv.writer(o); w.writerow(["record", "age", "sex", "dx", "fs", "n_samples", "has_mat"]); w.writerows(rows)
print(len(rows), "headers;", sum(x[6] == 0 for x in rows), "without a .mat")

#!/bin/bash
# CODE-15% (Zenodo record 4916206, open): 18 zip parts of ~2.5 GB each (~45 GB total).
# Resumable; a part is accepted only when its size equals the remote Content-Length, and every
# archive is CRC-tested at the end. Usage: bash download_code15.sh <raw_root>
RAW="${1:?usage: download_code15.sh <raw_root>}"
mkdir -p "$RAW/code15" && cd "$RAW/code15"
for i in $(seq 0 17); do
  f="exams_part${i}.zip"; url="https://zenodo.org/records/4916206/files/${f}?download=1"
  remote=$(curl -sIL "$url" | tr -d '\r' | awk 'tolower($1)=="content-length:"{l=$2} END{print l}')
  have=$( [ -f "$f" ] && stat -c%s "$f" || echo 0 )
  if [ -n "$remote" ] && [ "$have" = "$remote" ]; then echo "$f complete ($have bytes)"; continue; fi
  echo "$f: have $have of ${remote:-unknown} bytes, resuming"; wget -c -O "$f" "$url"
done
python3 - << 'PY'
import zipfile, glob
fs = sorted(glob.glob('exams_part*.zip'), key=lambda x: int(x.split('part')[1].split('.')[0]))
print(len(fs), "archives (expect 18)")
for f in fs:
    try: print(f, "OK" if zipfile.ZipFile(f).testzip() is None else "CORRUPT", flush=True)
    except Exception as e: print(f, "FAILED:", e, flush=True)
PY

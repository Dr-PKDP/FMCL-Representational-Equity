#!/bin/bash
# PTB-XL v1.0.3 (PhysioNet, open). Usage: bash download_ptbxl.sh <raw_root>
# wget -r can skip files after transient errors; follow with
#   python3 repair_ptbxl.py <ptbxl_dir> --fetch
# which checks every record listed in the database table.
RAW="${1:?usage: download_ptbxl.sh <raw_root>}"
mkdir -p "$RAW/ptbxl"
wget -r -N -c -np -P "$RAW/ptbxl" https://physionet.org/files/ptb-xl/1.0.3/
echo ".hea files: $(find "$RAW/ptbxl" -name '*.hea' | wc -l)  (expect 43598 = 21799 records x 2 rates)"

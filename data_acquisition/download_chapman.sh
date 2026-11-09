#!/bin/bash
# Chapman-Shaoxing + Ningbo (PhysioNet ecg-arrhythmia 1.0.0, open). 45,152 records (.hea + .mat).
# This is the combined release (Chapman-Shaoxing and Ningbo), not the 10,646-record Chapman-only release.
# wget -r can skip files after transient errors; the checksum step below finds and re-fetches them.
# Usage: bash download_chapman.sh <raw_root>
RAW="${1:?usage: download_chapman.sh <raw_root>}"
OUT="$RAW/chapman_shaoxing"; mkdir -p "$OUT"
wget -r -N -c -np -P "$OUT" https://physionet.org/files/ecg-arrhythmia/1.0.0/
cd "$OUT/physionet.org/files/ecg-arrhythmia/1.0.0"
echo ".hea files: $(find . -name '*.hea' | wc -l)  (expect 45152)"
F="$RAW/chapman_checksum_failures.txt"
sha256sum -c SHA256SUMS.txt 2>/dev/null | grep -v ': OK$' > "$F"
echo "checksum failures: $(wc -l < "$F")  (files listed there are missing; refetching)"
cut -d: -f1 "$F" | while read f; do
  rm -f "$f"; mkdir -p "$(dirname "$f")"
  if wget -q -O "$f" "https://physionet.org/files/ecg-arrhythmia/1.0.0/$f"; then echo "got $f"; else echo "FAILED $f"; rm -f "$f"; fi
done
grep -F -f <(cut -d: -f1 "$F") SHA256SUMS.txt | sha256sum -c -

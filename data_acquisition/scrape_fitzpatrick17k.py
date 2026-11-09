"""
scrape_fitzpatrick17k.py. Downloads Fitzpatrick17k images from the source atlases and verifies each file
against the CSV's md5hash, the md5 of the image bytes. A file is kept only if md5(bytes) == md5hash, because
some hosts answer missing images with an HTML page and HTTP 200.
Status values: ok | failed: <reason> | failed: no_valid_url | failed: md5_mismatch. 404/410 are not retried.
Usage: python3 scrape_fitzpatrick17k.py --csv fitzpatrick17k.csv --out-dir images [--limit N]
"""
import argparse, csv, hashlib, os, socket, time, urllib.error, urllib.request
RATE, TIMEOUT, RETRIES = 0.3, 15, 2

def download_one(url, md5, out_path):
    if not url or not url.strip().lower().startswith(("http://", "https://")):
        return "failed: no_valid_url"
    if os.path.exists(out_path):
        return "already_exists" if hashlib.md5(open(out_path, "rb").read()).hexdigest() == md5 else _drop(out_path)
    for attempt in range(RETRIES + 1):
        try:
            req = urllib.request.Request(url.strip(), headers={"User-Agent": "Mozilla/5.0 (research download)"})
            with urllib.request.urlopen(req, timeout=TIMEOUT) as r:
                data = r.read()
            if hashlib.md5(data).hexdigest() != md5:
                return "failed: md5_mismatch"
            open(out_path, "wb").write(data)
            return "ok"
        except (urllib.error.HTTPError, urllib.error.URLError, socket.timeout, ConnectionError, ValueError, OSError) as e:
            if isinstance(e, urllib.error.HTTPError) and e.code in (404, 410):
                return f"failed: HTTPError: {e}"
            if attempt == RETRIES:
                return f"failed: {type(e).__name__}: {e}"
            time.sleep(1)
    return "failed: unknown"

def _drop(p):
    os.remove(p); return "failed: stale_file_removed"

def scrape(csv_path, out_dir, limit=None):
    os.makedirs(out_dir, exist_ok=True); log_path = os.path.join(out_dir, "download_log.csv")
    done = set()
    if os.path.exists(log_path):
        for r in csv.DictReader(open(log_path, newline="")):
            if r["status"] in ("ok", "already_exists"): done.add(r["md5hash"])
    rows = list(csv.DictReader(open(csv_path, newline="", encoding="utf-8")))[:limit]
    new = not os.path.exists(log_path); lf = open(log_path, "a", newline="")
    w = csv.DictWriter(lf, fieldnames=["md5hash", "status", "url"]); new and w.writeheader()
    ok = fail = 0
    for i, r in enumerate(rows):
        if r["md5hash"] in done: continue
        url = (r.get("url") or "").strip(); ext = os.path.splitext(url)[1] or ".jpg"
        st = download_one(url, r["md5hash"], os.path.join(out_dir, r["md5hash"] + ext))
        w.writerow({"md5hash": r["md5hash"], "status": st, "url": url}); lf.flush()
        ok += st in ("ok", "already_exists"); fail += st not in ("ok", "already_exists")
        if (i + 1) % 200 == 0: print(f"  [{i+1}/{len(rows)}] ok={ok} fail={fail}")
        if url: time.sleep(RATE)
    print(f"done. ok={ok} fail={fail} (log: {log_path})")

if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--csv", required=True); ap.add_argument("--out-dir", required=True)
    ap.add_argument("--limit", type=int); a = ap.parse_args(); scrape(a.csv, a.out_dir, a.limit)

#!/usr/bin/env bash
set -euo pipefail

# Build repos/models/LLM/data.txt from Project Gutenberg (public domain)
# Produces a reasonably large plain-text corpus for toy LLM training.

ROOT="$(cd "$(dirname "$0")" && pwd)"
RAW="$ROOT/raw"
OUT="$ROOT/data.txt"
TMP="$ROOT/.tmp"
mkdir -p "$RAW" "$TMP"

echo "[1/5] Downloading starter Gutenberg books..."
BOOK_URLS=(
  "https://www.gutenberg.org/cache/epub/1342/pg1342.txt"
  "https://www.gutenberg.org/cache/epub/11/pg11.txt"
  "https://www.gutenberg.org/cache/epub/84/pg84.txt"
  "https://www.gutenberg.org/cache/epub/1661/pg1661.txt"
  "https://www.gutenberg.org/cache/epub/98/pg98.txt"
)
: > "$RAW/gutenberg_seed.txt"
for u in "${BOOK_URLS[@]}"; do
  echo "  - $u"
  curl -fsSL "$u" >> "$RAW/gutenberg_seed.txt" || true
  printf "\n\n" >> "$RAW/gutenberg_seed.txt"
done

echo "[2/5] Expanding corpus from Gutenberg Top 100..."
python3 - "$RAW/gutenberg_top.txt" <<'PY'
import re, sys, urllib.request
out = sys.argv[1]

def get(url, timeout=30):
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
    with urllib.request.urlopen(req, timeout=timeout) as r:
        return r.status, r.read().decode("utf-8", errors="ignore")

status, html = get("https://www.gutenberg.org/browse/scores/top")
if status != 200:
    raise SystemExit(f"Failed top list: {status}")

ids = []
for m in re.finditer(r'/ebooks/(\d+)', html):
    i = m.group(1)
    if i not in ids:
        ids.append(i)
ids = ids[:120]

with open(out, "w", encoding="utf-8") as f:
    for i in ids:
        url = f"https://www.gutenberg.org/cache/epub/{i}/pg{i}.txt"
        try:
            status, txt = get(url)
            if status == 200 and len(txt) > 2000:
                f.write(txt + "\n\n")
                print(f"ok {i}")
            else:
                print(f"skip {i} ({status})")
        except Exception:
            print(f"skip {i} (error)")
PY

echo "[3/5] Cleaning text..."
python3 - "$RAW/gutenberg_seed.txt" "$RAW/gutenberg_top.txt" "$OUT" <<'PY'
import re, sys
seed_path, top_path, out_path = sys.argv[1:]

def clean(text: str) -> str:
    text = text.replace("\r", "\n")
    text = re.sub(r"(?is).*?\*\*\*\s*START OF (?:THE|THIS) PROJECT GUTENBERG.*?\*\*\*", "", text)
    text = re.sub(r"(?is)\*\*\*\s*END OF (?:THE|THIS) PROJECT GUTENBERG.*", "", text)
    text = re.sub(r"[_*~`]+", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

buf = []
for p in (seed_path, top_path):
    with open(p, "r", encoding="utf-8", errors="ignore") as f:
        buf.append(clean(f.read()))
text = "\n\n".join(x for x in buf if x).strip() + "\n"
with open(out_path, "w", encoding="utf-8") as f:
    f.write(text)
print(f"Wrote {out_path}")
print(f"Characters: {len(text):,}")
print(f"Approx tokens: {int(len(text)*0.25):,}")
PY

echo "[4/5] Final normalize..."
awk 'NF{print}' "$OUT" > "$TMP/data.clean" && mv "$TMP/data.clean" "$OUT"

echo "[5/5] Done"
wc -c "$OUT"
echo "Output: $OUT"

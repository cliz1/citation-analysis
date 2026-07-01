# sample_for_spotcheck.py
# Draws a random sample of citations across all four conferences for manual
# accuracy spot-checking. Pools csv/<Conference>_citations_matched.csv and
# writes source_paper, source_conference, raw_reference, venue_raw,
# venue_source for the sampled rows.
import argparse
import csv
import glob
import random
from pathlib import Path

_parser = argparse.ArgumentParser(description="Sample citations for manual spot-checking")
_parser.add_argument("--n", dest="n", type=int, default=200,
                      help="Number of citations to sample (default: 200)")
_parser.add_argument("--seed", dest="seed", type=int, default=None,
                      help="Random seed for reproducible sampling (default: unseeded, a fresh random sample each run)")
_parser.add_argument("--out", dest="out", default=None,
                      help="Output CSV path (default: validation_spot_check_<n>.csv, auto-numbered to avoid overwriting prior rounds)")
args = _parser.parse_args()

if args.seed is not None:
    random.seed(args.seed)

rows = []
for f in sorted(glob.glob("csv/*_citations_matched.csv")):
    conference = Path(f).name.replace("_citations_matched.csv", "")
    with open(f, newline="") as fh:
        reader = csv.DictReader((line.replace("\x00", "") for line in fh), escapechar="\\")
        for r in reader:
            rows.append({
                "source_paper": r["source_paper"],
                "source_conference": conference,
                "raw_reference": r["raw_reference"],
                "venue_raw": r["venue_raw"],
                "venue_matched": r["venue_matched"],
                "venue_source": r["venue_source"],
            })

if args.n > len(rows):
    raise ValueError(f"Requested sample of {args.n} exceeds pool size of {len(rows)}")

sample = random.sample(rows, args.n)

if args.out:
    out_path = Path(args.out)
else:
    i = 1
    while (out_path := Path(f"validation_spot_check_{i}.csv")).exists():
        i += 1

with open(out_path, "w", newline="") as out:
    writer = csv.DictWriter(out, fieldnames=["source_paper", "source_conference", "raw_reference", "venue_raw", "venue_matched", "venue_source"])
    writer.writeheader()
    writer.writerows(sample)

print(f"Pool size: {len(rows)}")
print(f"Wrote {len(sample)} sampled rows to {out_path}")

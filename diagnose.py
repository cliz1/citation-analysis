# diagnose.py
# Quantifies citation losses at each pipeline stage per conference.
import csv
import io
import re
from collections import Counter
from pathlib import Path

RAW_CSVS = {
    "EuroCrypt": "csv/EuroCrypt_citations_raw.csv",
    "Crypto":    "csv/Crypto_citations_raw.csv",
    "Oakland":   "csv/Oakland_citations_raw.csv",
    "USENIX":    "csv/USENIX_citations_raw.csv",
}

MATCHED_CSVS = {
    "EuroCrypt": "csv/EuroCrypt_citations_matched.csv",
    "Crypto":    "csv/Crypto_citations_matched.csv",
    "Oakland":   "csv/Oakland_citations_matched.csv",
    "USENIX":    "csv/USENIX_citations_matched.csv",
}

SPECIAL_VENUES = {"ePrint", "arXiv", "GitHub", "web", "web_forum"}

# Heuristic: "J. Lastname" patterns that the journal regex misidentifies as journal names
AUTHOR_INITIAL_RE = re.compile(r"^J\.\s+[A-Z][a-z]{2,}$")
# Two-column artifact: hyphen at end of a word mid-string
HYPHEN_ARTIFACT_RE = re.compile(r"\w-\s+\w")


def read_csv(path: str) -> list[dict]:
    """Read a CSV, stripping NUL bytes that appear in some files."""
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text), escapechar="\\"))


def section(title: str):
    print(f"\n{'='*60}")
    print(f"  {title}")
    print(f"{'='*60}")


def bar(label: str, n: int, total: int, width: int = 30):
    pct = 100 * n / total if total else 0
    filled = int(pct / 100 * width)
    print(f"  {label:<35} {n:>5}  ({pct:5.1f}%)  {'█'*filled}")


# ----------------------------------------------------------------
# Stage 1: raw CSV analysis
# ----------------------------------------------------------------
section("STAGE 1 — Venue extraction (citation_export.py output)")

raw_summaries = {}
for conf, path in RAW_CSVS.items():
    if not Path(path).exists():
        print(f"\n  {conf}: file not found — {path}")
        continue

    rows = read_csv(path)
    total = len(rows)

    no_venue        = [r for r in rows if not r["venue_raw"]]
    web_catch       = [r for r in rows if r["venue_raw"] == "web"]
    special         = [r for r in rows if r["venue_raw"] in SPECIAL_VENUES - {"web"}]
    author_fp       = [r for r in rows if AUTHOR_INITIAL_RE.match(r["venue_raw"])]
    hyphen_refs     = [r for r in rows if HYPHEN_ARTIFACT_RE.search(r["raw_reference"])]
    hyphen_no_venue = [r for r in rows if HYPHEN_ARTIFACT_RE.search(r["raw_reference"]) and not r["venue_raw"]]
    extracted       = [r for r in rows if r["venue_raw"] and r["venue_raw"] not in ("web",)]

    print(f"\n  [{conf}]  total citations: {total}")
    bar("no venue extracted",           len(no_venue),    total)
    bar("venue extracted (incl. web)",  len(extracted),   total)
    bar("  └─ ePrint / arXiv / GitHub", len(special),     total)
    bar("  └─ 'web' catch-all",         len(web_catch),   total)
    bar("  └─ author-initial false pos (J. Name)", len(author_fp), total)
    bar("refs with hyphen artifacts",   len(hyphen_refs), total)
    bar("  └─ hyphen artifacts + no venue", len(hyphen_no_venue), total)

    raw_summaries[conf] = {
        "total": total, "no_venue": len(no_venue),
        "web_catch": len(web_catch), "author_fp": len(author_fp),
        "hyphen_no_venue": len(hyphen_no_venue),
    }

# ----------------------------------------------------------------
# Stage 2: matched CSV analysis
# ----------------------------------------------------------------
section("STAGE 2 — Venue matching (venue_match.py output)")

for conf, path in MATCHED_CSVS.items():
    if not Path(path).exists():
        print(f"\n  {conf}: file not found — {path}")
        continue

    rows = read_csv(path)
    total = len(rows)

    had_venue     = [r for r in rows if r["venue_raw"]]
    matched       = [r for r in rows if r["venue_matched"]]
    matched_special = [r for r in rows if r["venue_matched"] in SPECIAL_VENUES]
    matched_academic = [r for r in rows if r["venue_matched"] and r["venue_matched"] not in SPECIAL_VENUES]

    # Citations that had a venue_raw but still failed matching
    venue_but_no_match = [r for r in rows if r["venue_raw"] and not r["venue_matched"]]

    print(f"\n  [{conf}]  total citations: {total}")
    bar("had venue_raw",                     len(had_venue),          total)
    bar("matched to canonical venue",        len(matched),            total)
    bar("  └─ academic venue",               len(matched_academic),   total)
    bar("  └─ special (ePrint/arXiv/web…)",  len(matched_special),    total)
    bar("had venue_raw but lost at matching",len(venue_but_no_match), total)
    bar("no venue at all (raw was empty)",   total - len(had_venue),  total)

    # Top unmatched venue_raw values
    if venue_but_no_match:
        unmatched_counts = Counter(r["venue_raw"] for r in venue_but_no_match)
        print(f"\n  Top unmatched venue_raw values for {conf}:")
        for v, n in unmatched_counts.most_common(15):
            tag = ""
            if AUTHOR_INITIAL_RE.match(v):
                tag = " ← author initial FP"
            elif v in ("NSDI", "SOSP", "OSDI", "ICML", "EuroSys", "S&P"):
                tag = " ← missing from ABBREV_MAP"
            print(f"    {n:4d}  {v!r}{tag}")

# ----------------------------------------------------------------
# Stage 3: "web" false-positive audit
# ----------------------------------------------------------------
section("STAGE 3 — 'web' catch-all audit (potential false positives)")

for conf, path in RAW_CSVS.items():
    if not Path(path).exists():
        continue
    rows = read_csv(path)
    web_rows = [r for r in rows if r["venue_raw"] == "web"]
    if not web_rows:
        print(f"\n  {conf}: no 'web' rows")
        continue
    # Sample the raw_reference to see what URLs are being caught
    url_re = re.compile(r"https?://[^\s,\"]+")
    url_counts: Counter = Counter()
    for r in web_rows:
        for url in url_re.findall(r["raw_reference"]):
            domain = re.sub(r"https?://(?:www\.)?([^/]+).*", r"\1", url)
            url_counts[domain] += 1
    print(f"\n  [{conf}]  {len(web_rows)} 'web' citations — top domains:")
    for domain, n in url_counts.most_common(12):
        print(f"    {n:4d}  {domain}")

# ----------------------------------------------------------------
# Summary funnel
# ----------------------------------------------------------------
section("SUMMARY — Citation funnel per conference")

print(f"\n  {'Conf':<12} {'Parsed':>7} {'NoVenue':>8} {'WebFP':>7} {'AuthFP':>7} {'HyphenMiss':>11}")
print(f"  {'-'*12} {'-'*7} {'-'*8} {'-'*7} {'-'*7} {'-'*11}")
for conf, s in raw_summaries.items():
    print(
        f"  {conf:<12} {s['total']:>7} {s['no_venue']:>7} ({100*s['no_venue']/s['total']:.0f}%)"
        f"  {s['web_catch']:>5} ({100*s['web_catch']/s['total']:.0f}%)"
        f"  {s['author_fp']:>5} ({100*s['author_fp']/s['total']:.0f}%)"
        f"  {s['hyphen_no_venue']:>5} ({100*s['hyphen_no_venue']/s['total']:.0f}%)"
    )

print()
print("  Columns: Parsed=total citations extracted from PDFs")
print("           NoVenue=missed by regex+DBLP")
print("           WebFP=caught by 'web' catch-all (may be misclassified)")
print("           AuthFP='J. Name' mistaken for journal abbreviation")
print("           HyphenMiss=two-column artifacts with no venue extracted")

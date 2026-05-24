"""
Validates citation extraction against Semantic Scholar.

Two modes:
  count  — compare our per-paper citation count vs S2's referenceCount (fast)
  refs   — fetch S2's full reference list and diff against our extracted refs (slow, deep)

Usage:
  python validate_extraction.py --mode count --key YOUR_S2_KEY
  python validate_extraction.py --mode refs  --key YOUR_S2_KEY --paper "GoFetch: Breaking..."
  python validate_extraction.py --mode count  # no key = anonymous tier (very slow, ~5 req/min)

Free API key: https://www.semanticscholar.org/product/api#api-key-form
"""

import argparse, json, ssl, time, urllib.parse, urllib.request
import certifi, pandas as pd
from pathlib import Path

# ---------------------------------------------------------------------------
# S2 helpers
# ---------------------------------------------------------------------------

_S2_BASE = "https://api.semanticscholar.org/graph/v1"
_CTX = ssl.create_default_context(cafile=certifi.where())


def _get(url: str, key: str) -> dict:
    headers = {"User-Agent": "citation-analysis-research/1.0"}
    if key:
        headers["x-api-key"] = key
    req = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(req, timeout=15, context=_CTX) as r:
        return json.loads(r.read())


def _search_paper(title: str, key: str) -> dict | None:
    """Return the top S2 hit for a paper title, or None."""
    query = urllib.parse.quote(title[:120])
    url = f"{_S2_BASE}/paper/search?query={query}&fields=paperId,title,referenceCount&limit=1"
    try:
        data = _get(url, key)
        hits = data.get("data", [])
        return hits[0] if hits else None
    except Exception as e:
        print(f"    S2 search error: {e}")
        return None


def _fetch_references(paper_id: str, key: str) -> list[dict]:
    """Fetch all references for a paper (handles S2 pagination, max 1000)."""
    refs, offset = [], 0
    fields = "title,venue,year,externalIds"
    while True:
        url = f"{_S2_BASE}/paper/{paper_id}/references?fields={fields}&limit=100&offset={offset}"
        try:
            data = _get(url, key)
        except Exception as e:
            print(f"    S2 refs error at offset {offset}: {e}")
            break
        batch = data.get("data", [])
        refs.extend(batch)
        if len(batch) < 100:
            break
        offset += 100
        time.sleep(1.1)
    return refs


# ---------------------------------------------------------------------------
# Mode: count
# ---------------------------------------------------------------------------

def run_count(key: str, sample: int | None, out: Path):
    rows = []
    for conf in ["Crypto", "EuroCrypt", "Oakland", "USENIX"]:
        csv_path = Path(f"csv/{conf}_citations_raw.csv")
        if not csv_path.exists():
            print(f"Missing {csv_path}, skipping.")
            continue
        df = pd.read_csv(csv_path)
        counts = df.groupby("source_paper").size()
        if sample:
            counts = counts.sample(min(sample, len(counts)), random_state=42)
        for title, our_count in counts.items():
            print(f"  [{conf}] querying: {title[:60]}...")
            hit = _search_paper(title, key)
            s2_count = hit["referenceCount"] if hit else None
            diff = (our_count - s2_count) if s2_count is not None else None
            pct  = (our_count / s2_count * 100) if s2_count else None
            rows.append({
                "venue":     conf,
                "title":     title,
                "our_count": our_count,
                "s2_count":  s2_count,
                "diff":      diff,
                "pct_of_s2": round(pct, 1) if pct else None,
                "s2_title":  hit["title"] if hit else None,
            })
            # rate: 1 req/s with key, be conservative without
            time.sleep(1.1 if key else 12)

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)
    print(f"\nSaved {len(df_out)} rows to {out}")

    # summary
    valid = df_out.dropna(subset=["s2_count"])
    exact = (valid["diff"] == 0).sum()
    within2 = (valid["diff"].abs() <= 2).sum()
    over = (valid["diff"] > 2).sum()
    under = (valid["diff"] < -2).sum()
    print(f"\nSummary ({len(valid)} papers with S2 match):")
    print(f"  Exact match:    {exact}/{len(valid)}")
    print(f"  Within ±2:      {within2}/{len(valid)}")
    print(f"  Over by >2:     {over}  (potential FPs)")
    print(f"  Under by >2:    {under}  (potential missed citations)")
    print(f"  No S2 match:    {len(df_out) - len(valid)}")
    if len(valid):
        print(f"  Median pct:     {valid['pct_of_s2'].median():.1f}% of S2 count")


# ---------------------------------------------------------------------------
# Mode: refs (deep diff for a single paper)
# ---------------------------------------------------------------------------

def run_refs(key: str, paper_title: str, conf: str | None, out: Path):
    print(f"Searching S2 for: {paper_title[:80]}...")
    hit = _search_paper(paper_title, key)
    if not hit:
        print("No S2 match found.")
        return

    paper_id = hit["paperId"]
    s2_total = hit["referenceCount"]
    print(f"Found: {hit['title']} (S2 id={paper_id}, referenceCount={s2_total})")
    print("Fetching S2 reference list...")
    s2_refs = _fetch_references(paper_id, key)
    print(f"  Retrieved {len(s2_refs)} references from S2")

    # Load our extracted refs for this paper
    our_refs = []
    for c in ([conf] if conf else ["Crypto", "EuroCrypt", "Oakland", "USENIX"]):
        csv_path = Path(f"csv/{c}_citations_raw.csv")
        if not csv_path.exists():
            continue
        df = pd.read_csv(csv_path)
        match = df[df["source_paper"].str.contains(paper_title[:40], case=False, na=False)]
        if not match.empty:
            our_refs = match.to_dict("records")
            print(f"  Our extracted refs: {len(our_refs)} (from {c} CSV)")
            break

    # Build comparison table
    rows = []
    for entry in s2_refs:
        cited = entry.get("citedPaper", {})
        title = cited.get("title", "")
        venue = cited.get("venue", "")
        year  = cited.get("year", "")
        rows.append({
            "s2_title":   title,
            "s2_venue":   venue,
            "s2_year":    year,
            "in_our_csv": any(
                title[:30].lower() in r.get("raw_reference", "").lower()
                for r in our_refs
            ) if title else None,
        })

    df_out = pd.DataFrame(rows)
    df_out.to_csv(out, index=False)
    print(f"\nSaved {len(df_out)}-row reference diff to {out}")
    missing = df_out[df_out["in_our_csv"] == False]
    print(f"References in S2 not found in our CSV: {len(missing)}")
    if not missing.empty:
        print(missing[["s2_title", "s2_venue", "s2_year"]].to_string(index=False))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Validate citation extraction against Semantic Scholar")
    parser.add_argument("--mode",   choices=["count", "refs"], default="count")
    parser.add_argument("--key",    default="", help="S2 API key (free at semanticscholar.org/product/api)")
    parser.add_argument("--sample", type=int, default=None, help="Papers per venue to sample (count mode)")
    parser.add_argument("--paper",  default="", help="Exact paper title (refs mode)")
    parser.add_argument("--conf",   default=None, help="Conference to search (refs mode, optional)")
    parser.add_argument("--out",    default=None, help="Output CSV path")
    args = parser.parse_args()

    if args.mode == "count":
        out = Path(args.out or "validation_count.csv")
        run_count(key=args.key, sample=args.sample, out=out)
    else:
        if not args.paper:
            parser.error("--paper required for refs mode")
        out = Path(args.out or "validation_refs.csv")
        run_refs(key=args.key, paper_title=args.paper, conf=args.conf, out=out)

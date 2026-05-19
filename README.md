# Citation Analysis Pipeline

## Overview

This project extracts and analyzes venue-level citation distributions from academic security and cryptography papers. Papers are sourced from four conferences: **EuroCrypt**, **Crypto**, **Oakland (IEEE S&P)**, and **USENIX Security**.

The pipeline runs in three stages:

```
citation_export.py  →  venue_match.py  →  venue_charts.py / venue_by_awareness.py
```

---

## Stage 1: Citation Extraction — `citation_export.py`

Reads paper metadata from a Google Sheet (one sheet per conference), locates the corresponding PDFs in a local Zotero storage directory, extracts the References section from each PDF, and attempts to assign a venue label to each citation.

**Venue assignment methods (in priority order):**

1. **Regex** (`extract_venue()`) — pattern-matches known venue strings, journal abbreviations, ePrint identifiers, and DOI-style URLs directly in the reference text.
2. **DBLP lookup** (`query_dblp_for_venue()`) — if regex finds nothing, extracts a likely title from the reference string and queries the DBLP API. Results are cached per-conference in `json/<Conference>_dblp_cache.json`.
3. **None** — if both methods fail, venue is recorded as empty.

**Output:** `csv/<Conference>_citations_raw.csv` — one row per citation, with columns including the raw reference text, extracted venue string, and `venue_source` (regex / dblp / none).

**Configuration** (top of script):

```python
ZOTERO_STORAGE = Path("/Users/.../Zotero/storage/")
SPREADSHEET_ID = "..."
TEST_RANGE = "USENIX"  # one conference per run
```

### Known extraction losses

Overall extraction rate: EuroCrypt **90.2%**, Crypto **87.5%**, Oakland **83.5%**, USENIX **84.5%**

Pipeline leaks at this stage fall into two categories:

| Loss type | Description |
|---|---|
| **No venue** | Neither regex nor DBLP found a venue — citation counted but unlabeled |
| **Hyphen artifact** | Two-column PDF layout inserts soft hyphens at line breaks, fragmenting venue names; `dehyphenate()` mitigates this but does not fully eliminate it |

Quantified losses by conference (from `diagnose.py`):

| Conference | No venue | Hyphen + no venue |
|---|---|---|
| EuroCrypt | 10% | 3% |
| Crypto | 12% | 5% |
| Oakland | 17% | 4% |
| USENIX | 15% | 10% |

USENIX's hyphen artifact rate (10%) remains the highest due to its two-column PDF format. EuroCrypt has the lowest losses overall — LNCS single-column format is the most parser-friendly.

**Known parser-level issues (citation boundary errors):**

- *Missing final citation before appendix:* if a post-bibliography appendix immediately follows the last reference, the parser may absorb the appendix into the final citation and discard it. This accounts for at most one missed citation per affected paper.
- *Proof text parsed as a citation:* proof lines beginning with `"1."` can be mistaken for numbered citations. Keyword filters catch most cases; the remainder almost always fall into the `other` catch-all bucket.
- *Papers missing from Zotero:* roughly 5 papers per conference are coded in the spreadsheet but have no matching PDF in Zotero, causing them to be silently skipped.

---

## Stage 2: Venue Matching — `venue_match.py`

Takes `csv/<Conference>_citations_raw.csv` and normalizes the raw venue strings into canonical full names. Uses a two-step process:

1. **Abbreviation map** (`ABBREV_MAP`) — direct lookup of known short-form venue strings (e.g., `"CCS"` → `"ACM Conference on Computer and Communications Security"`).
2. **Fuzzy match** — for strings not in the map, runs fuzzy string matching against a list of known venue names.

**Output:** `csv/<Conference>_citations_matched.csv` — same rows as the raw file with an added `matched_venue` column.

### Known matching losses

Overall matching rate: EuroCrypt **80.2%**, Crypto **76.7%**, Oakland **65.7%**, USENIX **67.7%**

Citations that remain unmatched after this stage are primarily:
- Venues not yet in `ABBREV_MAP` (known gaps: NSDI, SOSP, OSDI, EuroSys, ICML — these appear in Oakland's top-15 unmatched list)
- Garbled or highly fragmented venue strings from the hyphen artifact problem upstream

---

## Stage 3: Visualization

### `venue_charts.py`

Produces a per-conference bar chart of the top 15 cited venues. Reads all four `csv/*_citations_matched.csv` files and outputs a multi-page PDF (`venue_charts.pdf`).

### `venue_by_awareness.py`

Breaks down venue citation share by application awareness level (from the Google Sheet). Produces `venue_by_awareness.pdf`.

### `web_venue_breakdown.py`

Analyzes the `"web"` catch-all bucket in more detail.

---

## Output Files

The pipeline produces two CSV files per conference, stored in `csv/`.

### `csv/<Conference>_citations_raw.csv`

One row per citation extracted from the conference corpus. Produced by `citation_export.py`.

| Column | Description |
|---|---|
| `source_paper` | Title of the paper that contains this citation, as it appears in the Google Sheet |
| `app_awareness` | Application awareness score of the source paper |
| `venue_raw` | Raw venue string extracted from the reference - the abbreviated or partially cleaned venue name as it appeared in the PDF (e.g., `"FOCS"`, `"J. ACM 45"`, `"Algorithmica"`). Empty string if extraction failed. |
| `venue_source` | How the venue was determined: `"regex"` (pattern matched directly in the reference text), `"dblp"` (looked up via the DBLP API using an extracted title), or `"none"` (both methods failed) |
| `raw_reference` | Full reference string as extracted from the PDF, including author list, title, and publication details. This is the raw fitz output after dehyphenation — expect ligature artifacts (e.g., `ﬁ` instead of `fi`) and occasional line-break noise. |

**Note on `venue_source = "none"`:** The `venue_raw` column will be an empty string for these rows. The citation is still present in the data; it simply could not be assigned a venue. These rows are the primary source of loss when computing venue distributions.

**Note on `venue_source = "dblp"`:** The DBLP API is queried using a title extracted from the reference string, which is heuristic and occasionally (although rarely) wrong. The returned venue is the DBLP-assigned publication venue for the best-matching paper.

---

### `csv/<Conference>_citations_matched.csv`

Extends the raw file with two additional columns after venue normalization by `venue_match.py`. All rows from the raw file are preserved.

| Column | Description |
|---|---|
| `source_paper` | *(same as raw)* |
| `app_awareness` | *(same as raw)* |
| `venue_raw` | *(same as raw)* |
| `venue_source` | *(same as raw)* |
| `raw_reference` | *(same as raw)* |
| `venue_matched` | Canonical full venue name after normalization (e.g., `"IEEE Annual Symposium on Foundations of Computer Science"`). Empty if `venue_raw` was empty or could not be matched. |
| `match_score` | Fuzzy match confidence score (0–100). Rows resolved via `ABBREV_MAP` direct lookup receive a score of 100. Lower scores indicate fuzzy matches and may warrant spot-checking. |

For analysis, `venue_matched` is the primary field to aggregate on. `venue_raw` is useful for debugging unmatched or low-confidence rows.

---

### `json/<Conference>_dblp_cache.json`

A key-value store mapping extracted reference titles to the DBLP-returned venue string. Produced and read by `citation_export.py` to avoid redundant API calls across runs.

Keyed by the title string passed to DBLP; values are the raw venue string returned. If you re-run `citation_export.py` on a conference that already has a cache file, only titles not already in the cache will trigger new API requests. This keeps runs fast and results reproducible — without the cache, DBLP results can shift as their database updates.

If you want to force a fresh DBLP lookup for a conference, delete its cache file before running.

---

## Dependencies

```
pip install pymupdf pandas matplotlib fuzzywuzzy google-api-python-client google-auth google-auth-oauthlib
```

- `credentials.json` — Google OAuth credentials for Sheets access
- `token.pickle` — cached OAuth token (auto-generated on first run)

---

## Running the Pipeline

```bash
# Step 1: extract citations for one conference at a time
#   Edit TEST_RANGE in citation_export.py, then:
python citation_export.py

# Step 2: normalize venue strings
python venue_match.py csv/<Conference>_citations_raw.csv

# Step 3: generate charts
python venue_charts.py
python venue_by_awareness.py
```

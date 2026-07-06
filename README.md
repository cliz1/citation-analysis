# Citation Analysis Pipeline

## Overview

This project extracts and analyzes citation distributions from academic security and cryptography papers. Papers are sourced from four conferences: **EuroCrypt**, **Crypto**, **Oakland (IEEE S&P)**, and **USENIX Security**.

The pipeline runs in four stages across four scripts:

```
citation_export.py  →  venue_export.py  →  venue_match.py  →  venue_charts.py / venue_by_awareness.py
    stage 1               stage 2             stage 3                    stage 4
 (extraction)          (venue labels)       (normalization)           (visualization)
```

---

## Stage 1: Citation Extraction — `citation_export.py`

Reads paper metadata from a Google Sheet (one tab per conference), locates the corresponding PDFs in a local Zotero storage directory, and produces a list of raw reference strings from each PDF.

**Steps:**

1. **Corpus loading** — Fetches paper titles and app-awareness scores from the Google Sheet; fuzzy-matches each title to a PDF in `ZOTERO_STORAGE`.
2. **Text extraction** (`extract_text_from_pdf`) — Reads all pages of a matched PDF via `fitz`.
3. **Reference section isolation** (`extract_references_section`) — Finds the "References" / "Bibliography" heading and truncates at any trailing appendix or acknowledgements section.
4. **Reference parsing** (`parse_references`) — Splits the section text into individual citation strings by detecting numeric (`[1]`, `1.`) and alpha (`[ABC+23]`, `ABC+23.`) citation markers.

**Output:** 1. `csv/<Conference>_citations_raw.csv` — one row per extracted citation, with `source_paper`, `app_awareness`, and `raw_reference`. 
2. `text/<Conference>/<title>.txt` - full text for each paper in the corpus. No venue information at this stage.

**Configuration:** All pipeline-wide config (paths, spreadsheet ID, tuning constants) lives in `config.py` at the repo root, and every value there can be overridden with an identically-named environment variable without touching code — e.g. `ZOTERO_STORAGE`, `SPREADSHEET_ID`, `CSV_DIR`, `DBLP_QUERY_DELAY_SECONDS`, `FUZZY_MATCH_CUTOFF`.

---

## Stage 2: Venue Extraction — `venue_export.py`

Reads `csv/<Conference>_citations_raw.csv` and assigns a venue label to each citation. Three passes are attempted in order; the first to succeed wins.

**FP filter** (`is_likely_real_citation()`): Before venue extraction, each reference is checked for hallmarks of a real citation (a publication year, URL, or known structural cue). References that fail are flagged as suspected parser artifacts and written to a separate audit CSV — they do not reach venue extraction or DBLP.

**Pass 1 — Regex patterns** (`extract_venue()`): Pattern-matches known venue strings, journal abbreviations, ePrint identifiers, publisher URL domains, and structural cues (`In:`, `eds.`, ordinal prefixes) directly in the reference text. Handles the large majority of citations.

**Pass 2 — DBLP title lookup** (`query_dblp_for_venue()`): If regex finds nothing, extracts a likely title from the reference string and queries the [DBLP API](https://dblp.org/faq/How+to+use+the+dblp+search+API.html). Results are cached per-conference in `json/<Conference>_dblp_cache.json` to avoid redundant calls across runs.

DBLP failures aren't all equal: request errors (timeouts, non-429 HTTP errors, dropped connections) aren't cached and are simply retried next run. A 200-with-no-match is weaker than it looks — DBLP can return that under load with no error at all — so misses need `DBLP_MISS_CONFIRM_THRESHOLD` (default 3) consecutive misses across runs before they're trusted and skipped; a hit at any point overwrites it. See the [cache file format](#jsonconference_dblp_cachejson) below for details.

**Pass 3 — Standards and grey literature** (`match_standards()`, `match_grey_lit()`): Post-DBLP pattern match for reference types DBLP cannot resolve: RFCs, NIST publications, FIPS standards, ISO/IEC documents, technical reports, theses, and books. Only runs on DBLP misses.

**If all three passes fail**, `venue_raw` is recorded as an empty string and `venue_source` is set to `"none"`.

**Output:** `csv/<Conference>_citations_venues.csv` — one row per real citation with venue labels added; `csv/<Conference>_suspected_fps.csv` — suspected parser artifacts for manual audit.

**Overall extraction rate (venue assigned):** EuroCrypt **95.1%**, Crypto **96.3%**, Oakland **93.2%**, USENIX **94.0%**. See [Known Limitations](#known-limitations) below for the full breakdown of unresolved citations and known regex tradeoffs.

---

## Stage 3: Venue Matching — `venue_match.py`

Takes `csv/<Conference>_citations_venues.csv` and normalizes the raw venue strings into canonical full names. Uses a two-step process:

1. **Abbreviation map** (`ABBREV_MAP`) — direct lookup of known short-form venue strings (e.g., `"CCS"` → `"ACM Conference on Computer and Communications Security"`).
2. **Fuzzy match** — for strings not in the map, runs fuzzy string matching against a list of known venue names.

**Output:** `csv/<Conference>_citations_matched.csv` — same rows as the venues file with added `venue_matched` and `match_score` columns.

**Overall matching rate:** EuroCrypt **83.0%**, Crypto **81.8%**, USENIX **74.7%**, Oakland **72.1%**. See [Known Limitations](#known-limitations) for what's driving the remainder.

---

## Stage 4: Visualization

### `venue_charts.py`

Produces a per-conference bar chart of the top 15 cited venues. Reads all four `csv/*_citations_matched.csv` files and outputs a multi-page PDF (`venue_charts.pdf`).

### `venue_by_awareness.py`

Breaks down venue citation share by application awareness level (from the Google Sheet). Produces `venue_by_awareness.pdf`.

### `web_venue_breakdown.py`

A work-in-progress approach that uses keyword matching to analyze the `"web"` citation bucket in more detail.

---

## Known Limitations

Everything below is a characterized, known gap in the pipeline.

**Stage 1 (extraction):** hyphen artifacts from two-column PDF layouts fragment venue names; `dehyphenate()` mitigates but doesn't fully eliminate this. USENIX is worst-affected (two-column), EuroCrypt least (single-column LNCS). A small number of PDFs also have no detectable "References" heading, or have their final reference clipped by a directly-adjacent appendix section.

**Stage 1, citation-count accuracy:** spot-checked against a hand-verified true count across 80 papers (20 per conference; see `Grobid vs Our Pipeline - validation_spot_check.csv.csv`). Citation count is exact on 75.0% of papers and within ±1 on 87.5%, mean error 1.4% of true count. On the 40-paper subset also run through [GROBID](https://github.com/kermitt2/grobid) for comparison, our error (1.05%) is roughly a third of GROBID's (2.98%) — exact match 77.5% vs. 45.0%, within ±1 92.5% vs. 77.5%.

**Stage 2 (venue assignment), structural gaps not pursued (would need architecture changes):**
- **Back-references** (`In: Wiener [53], https://doi.org/...`) — points to another entry in the same bibliography; ~3 entries.
- **DOI-only references** — bare DOI, no venue text for the title heuristic to use; ~6 entries.
- **Editor-preamble citations** (`In: Kaliski Jr. (ed.) CRYPTO '97`) — editor name sits where the venue acronym is expected.
- **Niche abbreviated journals** (`Period. Math. Hungar.`, `Phys. Rev. A`, `Quantum Inf. Comput.`) — would need a large hand-built lookup table; ~12 entries.
- **Generic-noun misfires** (`"...Test in Europe. ACM"` → `venue_raw = "Europe"`) — ~4 entries.

**Stage 2, citations with no venue to find** (not pattern gaps): physics/math/CS-theory journals outside what we track, books and textbooks, cross-references to other papers in the same proceedings, metadata-free preprints, GitHub/blog/lecture-note citations, and proof-body or appendix text that bled into the reference section during PDF extraction.

**Stage 2, DBLP resolution rate:** of citations that reach Pass 2 (regex found nothing), **54.7–59.7%** get a venue directly from DBLP — the rest fall through to Pass 3 (standards/grey-lit) or end unresolved as `"none"`. This is a resolution rate, not a correctness check: a DBLP hit is trusted as-is, with no independent verification that it matched the right paper. The title-extraction heuristic that builds the DBLP query is necessarily imprecise for citations with no clean title delimiter, which accounts for a real share of the misses.

**Stage 3 (matching)**, unmatched remainder by cause: upstream regex artifacts like `"Springer"` mis-extracted as a venue or truncated fragments (`"Annual Symposium on"`) — top unmatched string in 3 of 4 conferences, and not actually a Stage 3 gap; Pass 3 grey-lit/standards labels with no canonical form (`Tech. Rep.`, `PhD Thesis`, `Whitepaper`); and genuine `ABBREV_MAP` gaps (`VLDB`, `SIGMOD`, `NSDI`, `IACR PKC`).

---

## Output Files

The pipeline produces three CSV files per conference, stored in `csv/`.

### `csv/<Conference>_citations_raw.csv`

One row per citation extracted from the conference corpus. Produced by `citation_export.py`.

| Column | Description |
|---|---|
| `source_paper` | Title of the paper that contains this citation, as it appears in the Google Sheet |
| `app_awareness` | Application awareness score of the source paper |
| `raw_reference` | Full reference string as extracted from the PDF, after dehyphenation and watermark stripping |

---

### `csv/<Conference>_citations_venues.csv`

One row per real citation (suspected parser artifacts excluded) with venue labels assigned. Produced by `venue_export.py`.

| Column | Description |
|---|---|
| `source_paper` | Title of the paper that contains this citation |
| `app_awareness` | Application awareness score of the source paper |
| `venue_raw` | Raw venue string extracted from the reference — the abbreviated or partially cleaned venue name as it appeared in the PDF (e.g., `"FOCS"`, `"J. ACM 45"`, `"Algorithmica"`). Empty string if extraction failed. |
| `venue_source` | How the venue was determined: `"regex"` (pattern matched directly in the reference text), `"dblp"` (DBLP API lookup), `"standards"` (RFC/NIST/FIPS/ISO post-DBLP match), `"grey_lit"` (book/thesis/tech-report post-DBLP match), or `"none"` (all methods failed) |
| `raw_reference` | Full reference string as extracted from the PDF, after dehyphenation and watermark stripping. Expect ligature artifacts (e.g., `ﬁ` instead of `fi`) and occasional line-break noise. |
| `suspected_fp` | `False` for all rows in this file (suspected artifacts are written to `_suspected_fps.csv` instead) |

**Note on `venue_source = "none"`:** The `venue_raw` column will be an empty string for these rows. The citation is still present in the data; it simply could not be assigned a venue. These rows are the primary source of loss when computing venue distributions.

**Note on `venue_source = "dblp"`:** The DBLP API is queried using a title extracted from the reference string, which is heuristic and occasionally (although rarely) wrong. The returned venue is the DBLP-assigned publication venue for the best-matching paper.

---

### `csv/<Conference>_citations_matched.csv`

Extends the venues file with two additional columns after venue normalization by `venue_match.py`. All rows from the venues file are preserved.

| Column | Description |
|---|---|
| `source_paper` | *(same as venues)* |
| `app_awareness` | *(same as venues)* |
| `venue_raw` | *(same as venues)* |
| `venue_source` | *(same as venues)* |
| `raw_reference` | *(same as venues)* |
| `suspected_fp` | *(same as venues)* |
| `venue_matched` | Canonical full venue name after normalization (e.g., `"IEEE Annual Symposium on Foundations of Computer Science"`). Empty if `venue_raw` was empty or could not be matched. |
| `match_score` | Fuzzy match confidence score (0–100). Rows resolved via `ABBREV_MAP` direct lookup receive a score of 100. Lower scores indicate fuzzy matches and may warrant spot-checking. |

For analysis, `venue_matched` is the primary field to aggregate on. `venue_raw` is useful for debugging unmatched or low-confidence rows.

---

### `json/<Conference>_dblp_cache.json`

Maps extracted title variants to their DBLP lookup result. Two value shapes:
- **Hit** — the full DBLP `info` dict (venue read from its `venue` field).
- **Miss** — `{"__miss__": true, "count": N}`, re-queried live each run until `N` reaches `DBLP_MISS_CONFIRM_THRESHOLD`.

Delete the cache file to force a fresh lookup for a conference.

---

### `text/<Conference>/<title>.txt`

Full extracted text of each matched PDF, one file per paper. Produced by `citation_export.py` so later stages (or re-parsing) don't require re-reading the PDFs.

### `logs/<Conference>_dblp_misses.txt`

Raw `raw_reference` strings for citations that failed every pass (regex, DBLP, standards, grey-lit) — the `venue_source = "none"` rows. Produced by `venue_export.py`; useful as an audit list when tuning Pass 1/3 patterns.

The other `logs/*_run.txt` files (`_citation_run`, `_venue_run`, `_venue_match_run`) are raw stdout captured via `tee` when a stage is run — handy for pulling stats from a specific run (e.g. the DBLP resolution rate above), but they're overwritten on every re-run and have no fixed schema, so don't treat them as stable output. The old flat `<Conference>_run.txt` files predate the citation/venue/match split and are stale — not worth keeping.

---

## Dependencies

```
pip install -r requirements.txt
```

- `credentials.json` — Google OAuth credentials for Sheets access
- `token.pickle` — cached OAuth token (auto-generated on first run)

---

## Running the Pipeline

```bash
# Stage 1: extract raw citations from PDFs (fast, no network after Sheets auth)
python citation_export.py --conference Crypto   # choices: Crypto, EuroCrypt, Oakland, USENIX

# Stage 2: assign venue labels (hits DBLP — ~25-37 min per conference)
python venue_export.py --conference Crypto

# Stage 3: normalize venue strings
python venue_match.py csv/Crypto_citations_venues.csv

# Stage 4: generate charts
python venue_charts.py
python venue_by_awareness.py

# Or run stages 1–2 for all conferences with automatic DBLP cooldowns:
./run_all_conferences.sh

TODO: new full pipeline script updates (stage 3 and flags)
```


---

## Testing

---

## Notes for future work

prototypes for author affiliation, web breakdown 

things that would be important to know if running the pipeline on a different or expanded corpus

---

## AI Use Disclosure

Claude Code (Anthropic) was used as the primary tool for implementation and debugging throughout this pipeline. The author retains responsibility for validation, documentation, and design decisions. 


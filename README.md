# Citation Analysis Pipeline

## Overview

This project extracts and analyzes venue-level citation distributions from academic security and cryptography papers. Papers are sourced from four conferences: **EuroCrypt**, **Crypto**, **Oakland (IEEE S&P)**, and **USENIX Security**.

The pipeline runs in four stages. Stages 1 and 2 share a script; stages 3 and 4 are separate:

```
citation_export.py                         →  venue_match.py  →  venue_charts.py / venue_by_awareness.py
 ├─ stage 1: citation extraction                 stage 3              stage 4
 └─ stage 2: venue extraction
```

---

## Stage 1: Citation Extraction — `citation_export.py`

Reads paper metadata from a Google Sheet (one tab per conference), locates the corresponding PDFs in a local Zotero storage directory, and produces a list of raw reference strings from each PDF.

**Steps:**

1. **Corpus loading** — Fetches paper titles and app-awareness scores from the Google Sheet; fuzzy-matches each title to a PDF in `ZOTERO_STORAGE`.
2. **Text extraction** (`extract_text_from_pdf`) — Reads all pages of a matched PDF via `fitz`.
3. **Reference section isolation** (`extract_references_section`) — Finds the last "References" / "Bibliography" heading and truncates at any trailing appendix or acknowledgements section.
4. **Reference parsing** (`parse_references`) — Splits the section text into individual citation strings by detecting numeric (`[1]`, `1.`) and alpha (`[ABC+23]`, `ABC+23.`) citation markers.
5. **Artifact filtering** (`is_likely_real_citation`) — Discards parser artifacts (table rows, proof steps, DOI fragments) that lack a publication year or URL.

**Output of this stage:** a list of raw reference strings, one per citation, ready for venue extraction.

**Configuration** (top of script):

```python
ZOTERO_STORAGE = Path("/Users/.../Zotero/storage/")
SPREADSHEET_ID = "..."
```

### Known extraction losses

Pipeline leaks at this stage:

| Loss type | Description |
|---|---|
| **No references section** | PDF text extraction succeeded but no "References" heading was found |
| **Truncated section** | A post-bibliography appendix immediately follows the last reference; the section is cut short and the final citation may be lost |
| **Hyphen artifacts** | Two-column PDF layout inserts soft hyphens at line breaks, fragmenting venue names; `dehyphenate()` mitigates this but does not fully eliminate it |

USENIX has the highest hyphen artifact rate due to its two-column PDF format. EuroCrypt's single-column LNCS format is the most parser-friendly.

---

## Stage 2: Venue Extraction — `citation_export.py`

Takes the raw reference strings from Stage 1 and assigns a venue label to each one. Three passes are attempted in order; the first to succeed wins.

**Pass 1 — Regex patterns** (`extract_venue()`): Pattern-matches known venue strings, journal abbreviations, ePrint identifiers, publisher URL domains, and structural cues (`In:`, `eds.`, ordinal prefixes) directly in the reference text. No network calls. Handles the large majority of citations.

**Pass 2 — DBLP title lookup** (`query_dblp_for_venue()`): If regex finds nothing, extracts a likely title from the reference string and queries the [DBLP API](https://dblp.org/faq/How+to+use+the+dblp+search+API.html). Results are cached per-conference in `json/<Conference>_dblp_cache.json` to avoid redundant calls across runs.

**Pass 3 — Standards and grey literature** (`match_standards()`, `match_grey_lit()`): Post-DBLP pattern match for reference types DBLP cannot resolve: RFCs, NIST publications, FIPS standards, ISO/IEC documents, technical reports, theses, and books. Only runs on DBLP misses.

**If all three passes fail**, `venue_raw` is recorded as an empty string and `venue_source` is set to `"none"`.

**Final output:** `csv/<Conference>_citations_raw.csv` — one row per citation.

### Known extraction losses

Overall extraction rate (venue assigned): EuroCrypt **95.1%**, Crypto **96.3%**, Oakland **93.2%**, USENIX **94.0%**

The remaining ~4–7% are unresolved citations where all three passes failed. The breakdown of why falls into three categories.

#### Structural limitations (harder to fix, not pursued)

These would require changes to the pipeline architecture rather than new patterns:

- **Back-references** (`In: Wiener [53], https://doi.org/...`): The citation points to another entry in the same bibliography. Resolving it would require a second lookup pass over the already-parsed reference list. ~3 entries.
- **DOI-only references**: A bare DOI with no venue text before the URL. The title-extraction heuristic fires but produces nothing usable for DBLP. ~6 entries.
- **Editor-preamble citations**: `In: Kaliski Jr. (ed.) CRYPTO '97` — the editor's name appears immediately after `In:`, so the venue name is never at the expected position. The `In:` pattern family requires the venue acronym to follow the colon directly. Would require parsing past the `(ed.)` prefix for old-style LNCS citations.
- **Niche abbreviated journals** (`Period. Math. Hungar.`, `Phys. Rev. A`, `Quantum Inf. Comput.`): Would require maintaining a large lookup table of abbreviated journal names. ~12 entries across all conferences.

#### Citations without a Venue

These are not pattern gaps — they are reference types for which no venue exists or can be inferred from the text:

- Physics, math, and CS theory journals cited in cryptography papers (`Phys. Rev. A/X/Lett.`, `Theor. Comput. Sci.`, `Mathematische Annalen`, etc.) — these are real citations but the venue isn't one we track
- Books and textbooks (Cambridge UP, MIT Press, Elsevier, Springer monographs)
- Cross-references to other papers in the same proceedings (`In Takagi and Peyrin [35]`, `In Canetti and Garay [10]`)
- Preprints with no venue metadata (no year, no URL, no conference)
- GitHub repositories, blog posts, and lecture notes cited as references
- Proof-body or appendix text that bled into the reference section during PDF extraction (not a citation at all)

---

## Stage 3: Venue Matching — `venue_match.py`

Takes `csv/<Conference>_citations_raw.csv` and normalizes the raw venue strings into canonical full names. Uses a two-step process:

1. **Abbreviation map** (`ABBREV_MAP`) — direct lookup of known short-form venue strings (e.g., `"CCS"` → `"ACM Conference on Computer and Communications Security"`).
2. **Fuzzy match** — for strings not in the map, runs fuzzy string matching against a list of known venue names.

**Output:** `csv/<Conference>_citations_matched.csv` — same rows as the raw file with an added `venue_matched` column.

### Known matching losses

Overall matching rate: EuroCrypt **80.2%**, Crypto **76.7%**, Oakland **65.7%**, USENIX **67.7%**

Citations that remain unmatched after this stage are primarily:
- Venues not yet in `ABBREV_MAP` (known gaps: NSDI, SOSP, OSDI, EuroSys, ICML — these appear in Oakland's top-15 unmatched list)
- Garbled or highly fragmented venue strings from hyphen artifacts upstream

---

## Stage 4: Visualization

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
| `venue_raw` | Raw venue string extracted from the reference — the abbreviated or partially cleaned venue name as it appeared in the PDF (e.g., `"FOCS"`, `"J. ACM 45"`, `"Algorithmica"`). Empty string if extraction failed. |
| `venue_source` | How the venue was determined: `"regex"` (pattern matched directly in the reference text), `"dblp"` (DBLP API lookup), `"standards"` (RFC/NIST/FIPS/ISO post-DBLP match), `"grey_lit"` (book/thesis/tech-report post-DBLP match), or `"none"` (all methods failed) |
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
# Step 1 & 2: extract citations and assign venue labels (one conference at a time)
python citation_export.py --conference Crypto   # choices: Crypto, EuroCrypt, Oakland, USENIX

# Step 3: normalize venue strings
python venue_match.py csv/<Conference>_citations_raw.csv

# Step 4: generate charts
python venue_charts.py
python venue_by_awareness.py
```

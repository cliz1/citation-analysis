# citation_export.py
# Nathaniel Clizbe (github.com/cliz1), January 2026
from pathlib import Path
import fitz
import re
import unicodedata
from collections import defaultdict
from googleapiclient.discovery import build
from google_auth_oauthlib.flow import InstalledAppFlow
from google.auth.transport.requests import Request
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import time
import urllib.parse
import urllib.request
import json
import ssl
import certifi
import urllib.error
import argparse

# -----------------------------
# Config
# -----------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Paths
ZOTERO_STORAGE = Path("/Users/nathanielclizbe/Zotero/storage/")

SPREADSHEET_ID = "1I2eZyK7PIhXEMwy30w8BgEcuRrLQQw4wK6GlxfAsuWE"

_parser = argparse.ArgumentParser(description="Export and classify paper citations")
_parser.add_argument("--conference", dest="conference", default="Crypto",
                     choices=["Crypto", "EuroCrypt", "Oakland", "USENIX"],
                     help="Google Sheets tab (conference) to process")
CONFERENCE = _parser.parse_args().conference

DBLP_CACHE_FILE = Path(f"json/{CONFERENCE}_dblp_cache.json")
DBLP_MISSES_FILE = Path(f"logs/{CONFERENCE}_dblp_misses.txt")
DBLP_MISSES_FILE.parent.mkdir(exist_ok=True)
dblp_cache: dict[str, dict] = {}
if DBLP_CACHE_FILE.exists():
    with open(DBLP_CACHE_FILE) as _f:
        dblp_cache = json.load(_f)
print(f"Loaded {len(dblp_cache)} cached DBLP hits from {DBLP_CACHE_FILE}")


# -----------------------------
# Google Sheets Helper
# -----------------------------

def get_sheets_service():
    creds = None

    # Load cached token if it exists
    if Path("token.pickle").exists():
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

    # Authenticate if needed
    if not creds or not creds.valid:
        if creds and creds.expired and creds.refresh_token:
            creds.refresh(Request())
        else:
            flow = InstalledAppFlow.from_client_secrets_file(
                "credentials.json", SCOPES
            )
            creds = flow.run_local_server(port=0)

        with open("token.pickle", "wb") as token:
            pickle.dump(creds, token)

    return build("sheets", "v4", credentials=creds)

# -----------------------------
# Title cleaning helper
# -----------------------------

def clean_title(raw: str) -> str:
    # Strip HTML sub/superscript blocks including content (e.g. k<sub>1, đots</sub> → k)
    raw = re.sub(r'<(?:sub|sup)[^>]*>.*?</(?:sub|sup)>', '', raw, flags=re.I | re.S)
    # Strip remaining HTML tags
    raw = re.sub(r'<[^>]+>', '', raw)

    # Extract content from LaTeX text commands (\texttt{Anemoi} → Anemoi, \text{X} → X)
    raw = re.sub(r'\\text\w*\{([^}]+)\}', r'\1', raw)
    # Remove remaining LaTeX math delimiters and commands
    raw = re.sub(r'\$+', '', raw)
    raw = re.sub(r'\\[a-zA-Z]+', '', raw)
    # Strip residual "tt" prefix from \texttt{} import artifacts (e.g. ttAnemoi → Anemoi)
    raw = re.sub(r'\btt([A-Z])', r'\1', raw)

    # Remove author prefix before colon (e.g. "Aldo Gunsing: Title")
    raw = re.sub(r"^.*?:\s*", "", raw)

    # Remove year in parentheses
    raw = re.sub(r"\(\d{4}\)", "", raw)

    return raw.strip()


# ------------------------------------------------------------------------------
# Collect titles from google sheets AND record app. awareness for later use
# ------------------------------------------------------------------------------

paper_titles = []
app_awareness = {}
awareness_values = ["1", "2", "3", "4"]

service = get_sheets_service()
sheet = service.spreadsheets()

result = sheet.values().get(
    spreadsheetId=SPREADSHEET_ID,
    range=CONFERENCE
).execute()

rows = result.get("values", [])

if not rows:
    print("No data found in sheet.")
else:
    for row in rows:
        if len(row) > 2:
            if row[2] == "1" or row[2] == "2": # Crypto or Analysis
                cleaned_title = clean_title(row[0].strip())
                paper_titles.append(cleaned_title)
                if row[4] in awareness_values: # for coders who didn't record app. awareness correctly >:(
                    app_awareness[cleaned_title] = row[4]
        

print(f"Loaded {len(paper_titles)} paper titles from google sheets.")

# -----------------------------------------------------------
# Find pdfs in Zotero that match the titles in our collection
# -----------------------------------------------------------

# Find all PDFs recursively
pdf_files = list(ZOTERO_STORAGE.rglob("*.pdf"))
print(f"Found {len(pdf_files)} PDFs in Zotero storage")

def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))

    text = text.lower()
    text = re.sub(r"-", " ", text)        # hyphens → word separators
    text = re.sub(r"\s+", " ", text)
    text = re.sub(r"[^a-z0-9 ]", "", text)

    return text.strip()

STOPWORDS = {
    "the","a","an","and","or","of","for","to","in","on",
    "with","without","via","using","from","by","at",
    "scheme","protocol","system","method","analysis","attack",
    "paper","model","efficient"
}

def pdf_matches_title(pdf_path: Path, titles: list[str]):
    filename = normalize(pdf_path.stem)
    filename_tokens = {t for t in filename.split() if t not in STOPWORDS}

    for title in titles:
        title_norm = normalize(title)
        title_tokens = {t for t in title_norm.split() if t not in STOPWORDS}

        if len(title_tokens) < 2:
            continue

        overlap = filename_tokens & title_tokens

        # Require a long distinctive word
        long_words = [w for w in title_tokens if len(w) >= 7]
        if long_words and not any(w in filename_tokens for w in long_words):
            continue

        # Require >=70% title coverage
        if len(overlap) / len(title_tokens) >= 0.7:
            return title

    return None

matches = defaultdict(list)

for pdf in pdf_files:
    matched_title = pdf_matches_title(pdf, paper_titles)
    if matched_title:
        matches[matched_title].append(pdf)

print(f"{sum(len(v) for v in matches.values())} PDFs match titles in papers_list.txt")

# matches maps each title -> list of all PDFs that matched it (Zotero can store
# duplicates across storage subdirs). Collapse to one PDF per title by picking
# the largest file — a proxy for the most complete version (e.g. publisher copy
# over preprint). Result: deduped_pdfs is dict[title_str -> single best Path].
deduped_pdfs = {}

for title, pdf_list in matches.items():
    best_pdf = max(pdf_list, key=lambda p: p.stat().st_size)
    deduped_pdfs[title] = best_pdf

# --- PDF coverage audit ---
unmatched_titles = [t for t in paper_titles if t not in deduped_pdfs]
print(f"\nPDF coverage: {len(deduped_pdfs)}/{len(paper_titles)} papers matched to a PDF")
if unmatched_titles:
    print(f"  {len(unmatched_titles)} papers with NO PDF found (likely removed from Zotero):")
    for t in unmatched_titles:
        print(f"    - {t}")

# Estimate how many unmatched titles are lost specifically due to the long-word
# filter vs. simply having no PDF in Zotero. Re-run matching without that guard
# and see how many additional titles get picked up.
def _pdf_matches_title_relaxed(pdf_path: Path, titles: list[str]):
    filename = normalize(pdf_path.stem)
    filename_tokens = {t for t in filename.split() if t not in STOPWORDS}
    for title in titles:
        title_norm = normalize(title)
        title_tokens = {t for t in title_norm.split() if t not in STOPWORDS}
        if len(title_tokens) < 2:
            continue
        overlap = filename_tokens & title_tokens
        if len(overlap) / len(title_tokens) >= 0.7:
            return title
    return None

if unmatched_titles:
    relaxed_matches: set[str] = set()
    for pdf in pdf_files:
        hit = _pdf_matches_title_relaxed(pdf, unmatched_titles)
        if hit:
            relaxed_matches.add(hit)
    longword_losses = len(relaxed_matches)
    truly_missing = len(unmatched_titles) - longword_losses
    print(f"\n  Long-word filter diagnostic ({len(unmatched_titles)} unmatched titles):")
    print(f"    {longword_losses} would match if long-word guard were removed (filter is costing us these)")
    print(f"    {truly_missing} have no PDF in Zotero at all (no match even with relaxed filter)")

    # 70% coverage threshold diagnostic: for each unmatched title, find the best
    # overlap ratio any PDF achieves (long-word filter removed so we isolate the
    # threshold). Titles with best overlap 50-69% are genuine threshold casualties
    # — a PDF exists but was just short of 70% due to Zotero filename truncation.
    title_best_overlap: dict[str, float] = {t: 0.0 for t in unmatched_titles}
    for pdf in pdf_files:
        fname_tokens = {t for t in normalize(pdf.stem).split() if t not in STOPWORDS}
        for title in unmatched_titles:
            title_tokens = {t for t in normalize(title).split() if t not in STOPWORDS}
            if len(title_tokens) < 2:
                continue
            ratio = len(fname_tokens & title_tokens) / len(title_tokens)
            if ratio > title_best_overlap[title]:
                title_best_overlap[title] = ratio

    threshold_casualties = [(t, r) for t, r in title_best_overlap.items() if 0.5 <= r < 0.7]
    print(f"\n  70% coverage threshold diagnostic ({len(unmatched_titles)} unmatched titles):")
    print(f"    {len(threshold_casualties)} are close misses (best overlap 50–69%) — lost to truncated Zotero filename:")
    for t, r in sorted(threshold_casualties, key=lambda x: -x[1]):
        print(f"      {r:.0%}  {t}")
    no_signal = sum(1 for r in title_best_overlap.values() if r < 0.5)
    print(f"    {no_signal} have best overlap <50% — no plausible PDF in Zotero")

print()


# ---------------------------------------------
# Functions for Parsing and Extracting References
# ---------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    """
    Input:  Path to a PDF file.
    Output: Full plain text of the PDF, all pages concatenated with newlines.
            Returns "" and prints a warning if the file cannot be opened.
    """
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_references_section(text: str) -> tuple[str, bool]:
    """
    Returns (references_text, truncated_early).
    references_text is "" if no references heading was found.
    truncated_early is True if a post-bibliography section marker (appendix /
    acknowledgements / supplementary) was found and the text was cut there.
    """
    # Match "References" as a full word at start of line or after newline
    matches = list(re.finditer(r"(?:^|\n)\s*(references|bibliography|references and notes)\b", text, re.I))
    if not matches:
        return "", False
    m = matches[-1]  # use the last match — bibliography is always at end of paper

    references_text = text[m.end():]
    truncated_early = False

    # stop at common section markers after references
    for marker in ["appendix", "acknowledgements", "supplementary"]:
        idx = re.search(rf"\n\s*{marker}\b", references_text, re.I)
        if idx:
            references_text = references_text[:idx.start()]
            truncated_early = True

    return references_text.strip(), truncated_early

def dehyphenate(text: str) -> str:
    """
    Input:  A raw reference string that may contain line-break hyphenation artifacts
            from two-column PDF extraction (e.g. "dif- ferential" or "commit-\nment").
    Output: The same string with line-break hyphens collapsed — the whitespace after
            the hyphen is removed so the two halves rejoin as one token
            (e.g. "dif- ferential" → "dif-ferential"). Without this, "dif" and
            "ferential" would be treated as separate words by downstream regex.
            Genuine compound-word hyphens are unaffected because they are not
            followed by whitespace.
    """
    return re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)



def parse_references(text: str, debug=False):
    """
    Input:  Raw references-section text (from extract_references_section), plus an
            optional debug flag that prints skipped blocks when True.
    Output: (refs, stats) where refs is a list of raw citation strings and stats is
            a dict of fall-through counts:
              dropped_body_text  — blocks rejected by looks_like_body_text()
              dropped_too_short  — blocks rejected for being < 35 chars
              stray_lines        — lines before the first citation marker (ignored)
            Handles three citation marker styles:
              - numeric:           "1. ..."  or  "[1] ..."
              - bracketed alpha:   "[ANWW13] ..."  /  "[BCG+19a] ..."
              - bare alpha (LNCS): "ABC+23."  alone on its own line
    """

    # Remove leading "References" header
    text = re.sub(r"^\s*references\s*", "", text, flags=re.IGNORECASE)

    # Remove common page-range patterns like "pp. 33–53."
    text = re.sub(r"\bpp\.?\s*\d{1,4}\s*[-–]\s*\d{1,4}\.?", "", text)

    # Insert newline before reference markers to help splitting
    marker_lookahead = r"(?=(\d{1,3}\.\s|\[\d{1,3}\]\s|\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*))"
    text = re.sub(r"\s" + marker_lookahead, "\n", text)

    # Split into lines and strip whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    refs = []
    stats = {"dropped_too_short": 0, "stray_lines": 0}
    current = None
    # Bare alpha key: e.g. "ABC+23." or "Abe01." — short key ending with 2-digit year, alone on line.
    # Require trailing $ so it only fires when the key is the entire stripped line.
    # \[\d{1,2}\]\s* uses \s* (not \s) to also match numeric bracket keys that sit alone on a line
    # with no trailing space (e.g. Bulletproofs++ PDF style: "[1]" on its own line).
    marker_re = re.compile(
        r"^(\d{1,2}\.\s"
        r"|\[\d{1,3}\]\s*"
        r"|\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*"
        r"|[A-Za-z][A-Za-z0-9+\-]{0,13}\d{2}\.\s*$"  # bare alpha: ABC+23.
        r")"
    )

    # IEEE bibliography continuation markers — not citation keys, always part of preceding ref
    ieee_continuation_re = re.compile(r'^\[(Online|Accessed)[^\]]*\]', re.I)

    for ln in lines:
        if marker_re.match(ln) and not ieee_continuation_re.match(ln):
            # start a new reference
            if current:
                if len(current) <= 35:
                    stats["dropped_too_short"] += 1
                    if debug:
                        print("Skipping too-short block:", current[:80].replace("\n", " "))
                else:
                    refs.append(current.strip())
            current = ln
        else:
            if current is not None:
                current += " " + ln
            else:
                stats["stray_lines"] += 1

    # append the final buffered reference if any
    if current:
        if len(current) <= 35:
            stats["dropped_too_short"] += 1
        else:
            refs.append(current.strip())

    return refs, stats


# -------------------------------------
# DBLP helper
# -------------------------------------

_DBLP_VENUE_MAP = {
    "corr": "arXiv",
    "arxiv": "arXiv",
    "iacr cryptol. eprint arch.": "ePrint",
    "iacr cryptology eprint archive": "ePrint",
    "iacr eprint": "ePrint",
    "iacr trans. cryptogr. hardw. embed. syst.": "TCHES",
    "j. cryptology": "Journal of Cryptology",
    "journal of cryptology": "Journal of Cryptology",
}


class _DblpRateLimited(Exception):
    pass


def _fetch_dblp_venue(title: str) -> dict | None:
    """Single DBLP title lookup; returns full info dict on hit, None on miss.
    Raises _DblpRateLimited on HTTP 429 so the caller can bail instead of retrying."""
    try:
        query = urllib.parse.quote(title.replace('-', ' '))
        url = f"https://dblp.org/search/publ/api?q={query}&format=json&h=1"
        req = urllib.request.Request(url, headers={"User-Agent": "citation-analysis-research/1.0"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=10, context=context) as response:
            data = json.loads(response.read())
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return None
        info = hits[0].get("info", {})
        return info if info.get("venue") else None
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise _DblpRateLimited()
        return None
    except Exception:
        return None


def query_dblp_for_venue(raw_reference: str) -> str:
    title = ""

    # LNCS / author-year: "...: Title. In ..."
    m = re.search(r":\s+([A-Z][^:\.]{10,120})\.\s+In\b", raw_reference)
    if m:
        title = m.group(1).strip()

    # Numeric style: "Authors. Title. In VENUE" — last sentence before ". In"
    if not title:
        m = re.search(r'\.\s+([A-Z][^\.]{15,120})\.\s+(?:In\b|IACR|arXiv|CoRR)', raw_reference)
        if m:
            title = m.group(1).strip()

    # Quoted title
    if not title:
        m = re.search(r'"([^"]{10,120})"', raw_reference)
        if m:
            title = m.group(1).strip()

    # Fallback: strip citation marker, skip past author list via ": " separator.
    # LNCS format always puts the title after "Authors: Title", so colon-space
    # is more reliable than period-space for finding where the title starts.
    if not title:
        body = re.sub(r'^\[?\d{1,3}\]?\.?\s*|^\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*', '', raw_reference)
        colon_pos = body.find(': ')
        title = body[colon_pos + 2:] if colon_pos >= 0 else body

    # Strip any trailing ". In VENUE" / ". arXiv" / ". IACR" that leaked into the title
    title = re.sub(r'\.\s+(?:In\b|arXiv|IACR|CoRR).*$', '', title, flags=re.I).strip()

    # Strip Chicago/Biblatex back-reference annotations: "(cit. on pp. 3, 4)"
    title = re.sub(r'\s*\(cit\.\s+on\s+pp?\.\s*[\d,\s]+\)', '', title).strip()

    # Normalize Unicode ligatures from PDF font encoding before sending to DBLP.
    # e.g. "Eﬃcient" → "Efficient", "diﬀerential" → "differential"
    title = title.translate(str.maketrans({'ﬀ':'ff','ﬁ':'fi','ﬂ':'fl','ﬃ':'ffi','ﬄ':'ffl','ﬅ':'st','ﬆ':'st'}))

    # Skip obviously hopeless queries
    if re.fullmatch(r'(?i)private\s+communication\.?', title.strip()):
        return ""

    # LNCS continuation lines parsed as citations: "20. LNCS, vol. 11999, Springer..."
    if re.match(r'^LNCS,\s+vol\.', title.strip(), re.I):
        return ""

    if not title or len(title) < 12:
        return ""

    # Try progressively shorter prefixes: 10 → 7 → 5 words.
    # Shorter queries strip garbled suffix tokens while keeping the
    # distinctive title head that DBLP needs for a confident match.
    words = title.split()
    seen: set[str] = set()
    variants: list[str] = []
    for n in (10, 7, 5):
        candidate = " ".join(words[:n]).strip()
        if len(candidate) >= 12 and candidate not in seen:
            seen.add(candidate)
            variants.append(candidate)

    for i, variant in enumerate(variants):
        # Check cache before hitting DBLP
        if variant in dblp_cache:
            raw_venue = dblp_cache[variant].get("venue", "")
            if isinstance(raw_venue, list):
                raw_venue = raw_venue[0] if raw_venue else ""
            venue = _DBLP_VENUE_MAP.get(raw_venue.lower(), raw_venue)
            print(f"    DBLP cache hit ({len(variant.split())}w): '{variant[:70]}' → {venue}")
            return venue

        print(f"    DBLP query ({len(variant.split())}w): '{variant[:70]}'")
        try:
            info = _fetch_dblp_venue(variant)
        except _DblpRateLimited:
            print("    DBLP rate-limited (429) — skipping remaining variants")
            return ""
        if info:
            raw_venue = info.get("venue", "")
            if isinstance(raw_venue, list):
                raw_venue = raw_venue[0] if raw_venue else ""
            dblp_cache[variant] = info  # cache full info dict; misses retried next run
            return _DBLP_VENUE_MAP.get(raw_venue.lower(), raw_venue)
        if i < len(variants) - 1:
            time.sleep(1.0)  # match inter-ref cadence; genuine miss, not rate-limit

    return ""

# ---------------------------------
# Venue extraction helper
# ---------------------------------

def extract_venue(reference: str) -> str:
    # Collapse soft-wrapped URLs: "https: //foo.com" → "https://foo.com"
    reference = re.sub(r"(https?:)\s+(//)", r"\1\2", reference)

    # LNCS style: "In: Editors (eds.) VENUE YEAR. LNCS"
    m = re.search(r"In:\s+.{0,80}?\(eds?\.\)\s+([A-Z][A-Za-z0-9 &\-–—]+\d{4})", reference)
    if m:
        return m.group(1).strip()

    # LNCS abbreviated: "In: VENUE YEAR. LNCS, vol."
    m = re.search(r"In:\s+([A-Z][A-Za-z0-9 &\-–—]+\d{4})\.", reference)
    if m:
        return m.group(1).strip()

    # LNCS short without volume: "In: CRYPTO (2023)" / "In: ASIACRYPT (2023)" / "In: NSPW (1993)"
    # Character class restricted to uppercase so long mixed-case titles don't fire.
    m = re.search(r"In:\s+([A-Z][A-Z0-9&\- ]{1,25})\s+\(\d{4}\)", reference)
    if m:
        return m.group(1).strip()

    # Journal style: checked before academic URL block so DOI-bearing journal refs
    # (e.g. "J. Cryptol. 32(2)... https://doi.org/...") are caught here rather than
    # intercepted by the URL handler and forwarded to DBLP unnecessarily.
    m = re.search(r"\b((?:J\.|Journal|Trans\.|IEEE Trans\.|ACM Trans\.)\s+[A-Za-z][A-Za-z0-9 \.]{2,40})", reference)
    if m:
        candidate = re.sub(r'\s+\d[\d\.]*$', '', m.group(1).strip())  # strip trailing volume number
        # Bypass guard for known J.-prefixed journal abbreviations that the guard misidentifies
        # as author initials (e.g. "J. Cryptol." matches the guard's end-of-string branch).
        if re.match(r'^J\.\s+(?:Cryptol|ACM\b|Comput|Math|Number|Symb\.)', candidate, re.I):
            return candidate
        if not re.match(r'^J\.\s+(?:[A-Z]\.\s+)*[A-Z][a-z]{2,}(?:[\.,](?:\s+[A-Z]|$)|\s+[a-z]|$)', candidate):
            return candidate

    # (eds.) abbreviation + ordinal: "Boneh, D. (eds.) 45th ACM STOC, ACM Press"
    # Placed before URL handler so DOI-bearing refs are also caught.
    m = re.search(r"\(eds?\.\),?\s+\d+(?:st|nd|rd|th)\s+([A-Z][A-Za-z0-9&\- ]{1,25})[,\.]", reference)
    if m:
        return m.group(1).strip()

    # (eds.) abbreviation + all-caps acronym/venue: "(ed.) SODA," / "(eds.) ACM CCS 2019"
    m = re.search(r"\(eds?\.\),?\s+([A-Z][A-Z0-9&\- ]{1,20})(?:[,\.]|\s+\d{4}|\s+\()", reference)
    if m:
        return m.group(1).strip()

    # "In: VENUE, pp." — acronym with page range before year: "In: ITCS, pp. 7:1–7:29"
    m = re.search(r"In:\s+([A-Z][A-Z0-9&\- ]{1,20}),\s*pp\.", reference)
    if m:
        return m.group(1).strip()

    # IEEE numeric style (no colon after "In"):
    # "In ACM CCS, pages 1068–1079, 2014" — venue before ", pages \d"
    m = re.search(r"\bIn\s+([A-Z][A-Za-z0-9 &'\-]{2,30}),\s*pages\s+\d", reference)
    if m:
        return m.group(1).strip()

    # "In STOC. ACM, 1990" — venue before ". Publisher"
    m = re.search(r"\bIn\s+([A-Z][A-Z0-9]{2,10})\.\s+(?:ACM|IEEE|Springer)[,\s]", reference)
    if m:
        return m.group(1).strip()

    # "In CRYPTO (2), pages" — venue with volume number in parens
    m = re.search(r"\bIn\s+([A-Z][A-Z0-9]{2,10})\s+\(\d{1,2}\)[,\s]", reference)
    if m:
        return m.group(1).strip()

    # Academic publisher URLs — try to extract venue from pre-URL text, else "" → DBLP.
    # doi\.\s*org handles soft-wrapped "doi. org" artifacts from two-column PDF extraction.
    _ACADEMIC_URL_RE = re.compile(
        r"https?://(?:dx\.)?doi\.\s*org"                          # DOI (incl. soft-wrapped)
        r"|https?://(?:dl|doi)\.acm\.org"                         # ACM DL
        r"|https?://ieeexplore\.ieee\.org"                        # IEEE Xplore
        r"|https?://(?:link\.)?springer\.com"                     # Springer / LNCS
        r"|https?://drops\.dagstuhl\.de"                          # LIPIcs (Dagstuhl)
        r"|https?://(?:www\.)?usenix\.org"                        # USENIX proceedings
        r"|https?://(?:www\.)?ndss-symposium\.org"                # NDSS
        r"|https?://(?:tches|tosc|iacr)\.iacr\.org"              # IACR journal sites
        r"|https?://epubs\.siam\.org"                             # SIAM journals
        r"|https?://(?:www\.)?sciencedirect\.com"                 # Elsevier
        r"|https?://(?:onlinelibrary\.)?wiley\.com"               # Wiley
        r"|https?://eccc\.weizmann\.ac\.il"                       # ECCC
        r"|https?://hal\.(?:science|inria\.fr|archives-ouvertes\.fr)"  # HAL preprints
        r"|https?://api\.semanticscholar\.org"                    # Semantic Scholar
        r"|https?://(?:www\.)?eudml\.org",                        # EuDML (math journals)
        re.I
    )
    _url_m = _ACADEMIC_URL_RE.search(reference)
    if _url_m:
        pre_url = reference[:_url_m.start()]
        m = re.search(r"([A-Z][A-Za-z0-9 &\-–—]+\d{4})\.", pre_url)
        if m:
            return m.group(1).strip()
        return ""

    # Alpha-key style: venue appears after editors list — "In EDITORS, editors, CRYPTO 2019"
    # Handles apostrophe-year too: "editors, ASIACRYPT'99"
    # Two leading uppercase letters required to exclude book titles like "Some Title 2019".
    m = re.search(r"\bed-?itor(?:s)?,\s+(?:\d+(?:st|nd|rd|th)\s+)?([A-Z][A-Z][A-Za-z0-9 &'\-–]{1,30}?(?:\s+\d{2,4}\b|'\d{2}))", reference)
    if m:
        return m.group(1).strip()

    # Ordinal conference where year is not adjacent: "editor, 30th SODA, pages"
    m = re.search(r"\bed-?itor(?:s)?,\s+\d+(?:st|nd|rd|th)\s+([A-Z][A-Za-z0-9&\- ]{1,20})\b", reference)
    if m:
        return m.group(1).strip()

    # Ordinal venue without editor prefix: "In: 61st FOCS, IEEE" / "In 19th ACM STOC, ACM Press"
    m = re.search(r"In:?\s+\d+(?:st|nd|rd|th)\s+([A-Z][A-Za-z0-9&\- ]{1,25})[,\.\s]", reference)
    if m:
        return m.group(1).strip()

    # IEEE year-first style: "In 2018 IEEE Symposium on Security and Privacy, pages"
    m = re.search(r"\bIn\s+\d{4}\s+([A-Z][A-Za-z0-9 &'\-]{3,50})(?:,|\s+pages|\s+vol|\.\s)", reference)
    if m:
        return m.group(1).strip()

    # Slash-joined venue: "In Approx/Random, volume"
    m = re.search(r"\bIn\s+([A-Z][A-Za-z0-9]+(?:/[A-Za-z0-9]+)+)[,\s]", reference)
    if m:
        return m.group(1).strip()

    # IEEE/ACM short style: "in S&P 2021," / "In CCS 2018," / "In ACM PODC, 2019"
    m = re.search(r"\bin\s+([A-Z][A-Za-z0-9 &'\-]{2,40}),?\s*\d{4}", reference, re.I)
    if m:
        return m.group(1).strip()

    # Standalone venue without adjacent year: "NeurIPS, 2018." / "In SOSP, page"
    m = re.search(r"\b(NeurIPS|SOSP|OSDI)(?:[,\s]+\d{4})?", reference)
    if m:
        return m.group(0).strip().rstrip(',').strip()

    # "In Proceedings of ..." or "In Proc. ..." (with or without colon after In)
    # Pro-?ceedings also handles PDF hyphenation artifact "Pro-ceedings".
    m = re.search(r"\bIn:?\s+(?:Pro-?ceedings\s+of\s+|Proc\.?\s+)([^,\.]+)", reference, re.I)
    if m:
        return m.group(1).strip()

    # Short acronym style: "In: ITC (2023)" or "In: 24th ACM STOC"
    m = re.search(r"In:\s+(?:\d+(?:st|nd|rd|th)\s+)?([A-Z][A-Z0-9&\- ]{1,30})[,\s]+(?:ACM|IEEE|Springer)?\s*(?:Press\b|,)?\s*\w*\s*\d{4}", reference)
    if m:
        return m.group(1).strip()

    # Abbreviated journal names ending in volume/issue numbers.
    # Each word must start uppercase so the pattern can't span into a paper title.
    # [\s,]+ handles "Commun. ACM, 24(2)" where comma separates journal from volume.
    m = re.search(r"([A-Z][A-Za-z\.]{1,12}(?:\s+[A-Z][A-Za-z\.]{1,12}){0,4})[\s,]+\d+\(\d+\)", reference)
    if m:
        return m.group(1).strip()

    if re.search(r"https?:\s*//\s*github\.com", reference, re.I):
        return "GitHub"


    # ePrint / arXiv — ia.cr is the IACR ePrint URL shortener (ia.cr/YYYY/NNN)
    if re.search(r"eprint\.iacr\.org|https?://ia\.cr/|Cryptol(?:ogy)?\.?\s*ePrint", reference, re.I):
        return "ePrint"
    if re.search(r"arxiv\.org|arXiv", reference, re.I):
        return "arXiv"
    
    # Web/blog/forum references with no venue
    if re.search(r"https?://", reference):
        if re.search(r"github\.\s*(?:com|io)|gitlab\.\s*com", reference, re.I):
            return "GitHub"
        if re.search(r"ethresear\.ch|vitalik\.ca|bitcointalk", reference, re.I):
            return "web_forum"
        return "web"

    return ""

# ---------------------------------
# Standards / grey-literature matcher
# ---------------------------------

def match_standards(ref: str) -> str:
    """Post-DBLP fallback: returns a label like 'RFC 8446' or '' if no pattern fires.
    Only called on DBLP misses, so there is no risk of shadowing a real venue match."""
    m = re.search(r'\bRFC\s*(\d{3,4})\b', ref, re.I)
    if m:
        return f"RFC {m.group(1)}"
    m = re.search(r'\bFIPS\s+(?:PUB\s+)?(\d[\d\-]*)', ref, re.I)
    if m:
        return f"FIPS {m.group(1)}"
    m = re.search(r'\bNIST\s+(?:SP|Special\s+Publication)\s+([\d\-A-Za-z]+)', ref, re.I)
    if m:
        return f"NIST SP {m.group(1)}"
    m = re.search(r'\bISO/IEC\s+(\d[\d\-]*)', ref, re.I)
    if m:
        return f"ISO/IEC {m.group(1)}"
    m = re.search(r'\bANSI\s+(X[\d\.]+)', ref, re.I)
    if m:
        return f"ANSI {m.group(1)}"
    m = re.search(r'\bU\.?S\.?\s+Patent\s+([\d,]+)', ref, re.I)
    if m:
        return f"U.S. Patent {m.group(1)}"
    return ""


def match_grey_lit(ref: str) -> str:
    """Post-DBLP fallback for books and technical reports.
    Returns a specific label (e.g. 'Cambridge University Press', 'Tech. Rep. TR-21-05')
    or '' if no pattern fires. source='grey_lit' groups both types in analysis."""
    # Whitepapers / blog posts explicitly labelled as such
    if re.search(r'\bwhite\s*paper\b', ref, re.I):
        return "Whitepaper"

    # PhD theses
    m = re.search(r'\bPh\.?D\.?\s+[Tt]hesis\b', ref)
    if m:
        return "PhD Thesis"

    # Competition submissions (crypto-specific: CAESAR, NIST LWC, etc.)
    m = re.search(r'\bSubmission\s+to\s+(?:the\s+)?([A-Z][A-Za-z0-9 \-]+[Cc]ompetition)', ref)
    if m:
        return m.group(1).strip()

    # Technical reports — distinctive enough to check without guards
    m = re.search(r'\bTech(?:nical)?\.?\s+Rep(?:ort)?\.?(?:\s+([\w\-/]+))?', ref, re.I)
    if m:
        num = (m.group(1) or "").strip()
        return f"Tech. Rep. {num}".strip() if num else "Tech. Rep."

    # Books — publisher name present, no "In:" (which signals a conference paper)
    if not re.search(r'\bIn:', ref):
        m = re.search(
            r'\b(Cambridge University Press|MIT Press|Oxford University Press'
            r'|CRC Press|Springer(?:\s+(?:Berlin|Verlag|Heidelberg))?'
            r'|Wiley|O\'Reilly|Addison-Wesley|Prentice.?Hall)\b',
            ref, re.I
        )
        if m:
            return m.group(1)
    return ""


# ---------------------------------
# Real-citation heuristic
# ---------------------------------

def is_likely_real_citation(raw_ref: str) -> bool:
    """
    Returns False if raw_ref looks like a parser artifact rather than a real citation.
    A real citation almost always contains a 4-digit publication year or a URL.
    Known artifact types that fail both tests:
      - Table rows (e.g. "[82] FA,FE,C10 Trim FedSGD ...")
      - Proof steps (e.g. "1. The general case follows from ...")
      - DOI fragments (e.g. "05. doi: 10.1007/978-...")
    Rows that return False skip venue extraction and DBLP entirely and are written
    to the FP audit CSV. Confirmed against 893 flagged rows across 4 conferences:
    0 true positives.
    """
    return bool(
        re.search(r'\b(?:19|20)\d{2}\b', raw_ref)
        or re.search(r'https?:\s*//', raw_ref)  # also catches soft-wrapped "https: //"
    )


# TODO: disentangle citation extraction from venue matching — these should be
# separate passes so each can be unit-tested independently.

# ---------------------------------
# Main Processing Loop — venue extraction
# ---------------------------------

# Fall-through counters — printed in the summary at the end of the run.
n_pdf_errors       = 0  # fitz raised an exception opening the PDF
n_no_refs_heading  = 0  # no "References"/"Bibliography" heading found in text
n_truncated_early  = 0  # references section cut short by appendix/acknowledgements marker
n_empty_refs_text  = 0  # heading found but extracted text was empty after truncation
# TODO: body-text filter removed — more blocks now reach DBLP as bogus queries.
# May need to revisit the 1.5s rate-limit delay or add a per-run query cap if
# DBLP starts returning 429s more frequently.
parse_totals = {"dropped_too_short": 0, "stray_lines": 0}
n_dblp_hits      = 0
n_dblp_misses    = 0
n_standards_hits = 0
n_grey_lit_hits  = 0
dblp_miss_refs: list[str] = []

citation_rows = []

for title, pdf_path in deduped_pdfs.items():
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        n_pdf_errors += 1
        continue

    references_text, truncated = extract_references_section(pdf_text)
    if truncated:
        n_truncated_early += 1
    if not references_text and not truncated:
        n_no_refs_heading += 1
        print(f"No References section found: {pdf_path.name}")
        continue
    if not references_text:
        n_empty_refs_text += 1
        print(f"Empty references after truncation: {pdf_path.name}")
        continue

    parsed_refs, parse_stats = parse_references(references_text)
    for k in parse_totals:
        parse_totals[k] += parse_stats[k]
    awareness = app_awareness.get(title)

    for ref in parsed_refs:
        ref_clean = dehyphenate(ref)

        if not is_likely_real_citation(ref_clean):
            citation_rows.append({
                "source_paper": title,
                "app_awareness": awareness,
                "venue_raw": "",
                "venue_source": "none",
                "raw_reference": ref,
                "suspected_fp": True,
            })
            continue

        venue = extract_venue(ref_clean)
        if venue:
            source = "regex"
        else:
            print(f"  DBLP query ref: {ref_clean[:80]}")
            venue = query_dblp_for_venue(ref_clean)
            if venue:
                source = "dblp"
                n_dblp_hits += 1
            else:
                standards_label = match_standards(ref_clean)
                if standards_label:
                    venue = standards_label
                    source = "standards"
                    n_standards_hits += 1
                else:
                    grey_lit_label = match_grey_lit(ref_clean)
                    if grey_lit_label:
                        venue = grey_lit_label
                        source = "grey_lit"
                        n_grey_lit_hits += 1
                    else:
                        source = "none"
                        n_dblp_misses += 1
                        dblp_miss_refs.append(ref_clean)
            print(f"  DBLP result: {venue or 'none'}")
            time.sleep(1.5)

        citation_rows.append({
            "source_paper": title,
            "app_awareness": awareness,
            "venue_raw": venue,
            "venue_source": source,
            "raw_reference": ref,
            "suspected_fp": False,
        })

df_citations = pd.DataFrame(citation_rows)
is_fp = df_citations["suspected_fp"].astype(str).str.lower().isin(["true", "1"])
df_real = df_citations[~is_fp]
df_fps  = df_citations[is_fp]

_csv_opts = dict(index=False, escapechar="\\", quoting=1)
df_real.to_csv(f"csv/{CONFERENCE}_citations_raw.csv", **_csv_opts)
df_fps.to_csv(f"csv/{CONFERENCE}_suspected_fps.csv", **_csv_opts)

print(f"Saved {len(df_real)} citation rows to csv/{CONFERENCE}_citations_raw.csv")
print(f"Saved {len(df_fps)} suspected FPs to csv/{CONFERENCE}_suspected_fps.csv (gitignored — manual audit)")

with open(DBLP_CACHE_FILE, 'w') as _f:
    json.dump(dblp_cache, _f, indent=2)
print(f"Saved {len(dblp_cache)} DBLP hits to {DBLP_CACHE_FILE}")

with open(DBLP_MISSES_FILE, 'w') as _f:
    _f.write(f"# DBLP misses for {CONFERENCE} — {n_dblp_misses} failed queries\n\n")
    for r in dblp_miss_refs:
        _f.write(r + "\n---\n")
print(f"Saved {n_dblp_misses} DBLP misses to {DBLP_MISSES_FILE}")

# Fall-through summary
print(f"\nPDF extraction fall-throughs ({len(deduped_pdfs)} papers attempted):")
print(f"  fitz errors (could not open PDF):          {n_pdf_errors}")
print(f"  no references heading found:               {n_no_refs_heading}")
print(f"  references section truncated by marker:    {n_truncated_early}  (still processed)")
print(f"  empty text after truncation:               {n_empty_refs_text}")
print(f"\nReference parser fall-throughs (across all papers):")
print(f"  blocks dropped — too short (<35 chars):    {parse_totals['dropped_too_short']}")
print(f"  stray lines before first marker (ignored): {parse_totals['stray_lines']}")

n_dblp_total = n_dblp_hits + n_standards_hits + n_grey_lit_hits + n_dblp_misses
dblp_rate = 100 * n_dblp_hits / n_dblp_total if n_dblp_total else 0
print(f"\nDBLP query results ({n_dblp_total} live queries this run):")
print(f"  hits:      {n_dblp_hits}  ({dblp_rate:.1f}% success rate)")
print(f"  standards: {n_standards_hits}  (RFC/NIST/FIPS/ISO — post-DBLP pattern match)")
print(f"  grey lit:  {n_grey_lit_hits}  (books/tech reports — post-DBLP pattern match)")
print(f"  misses:    {n_dblp_misses}  — see {DBLP_MISSES_FILE}")

extracted = df_real[df_real["venue_raw"] != ""].shape[0]
total = len(df_real)
fp_count = len(df_fps)
papers_processed = len(deduped_pdfs)
expected = papers_processed * 60
print(f"Papers in corpus: {len(paper_titles)} in sheet, {papers_processed} PDFs found and processed")
print(f"Expected citations (rough): {papers_processed} papers × 60 = {expected}")
print(f"Citations extracted:        {total}  ({100*total/expected:.1f}% of expected, {fp_count} suspected FPs excluded)")
print(f"Venue extracted: {extracted}/{total} ({100*extracted/total:.1f}%)")

 # Check venue extraction rate by app_awareness level
print("\nExtraction rate by awareness level:")
df_real["extracted"] = df_real["venue_raw"] != ""
print(df_real.groupby("app_awareness")["extracted"].mean().round(3))
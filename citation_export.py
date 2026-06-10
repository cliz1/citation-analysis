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
import argparse

# -----------------------------
# Config
# -----------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Paths
ZOTERO_STORAGE = Path("/Users/nathanielclizbe/Zotero/storage/")

SPREADSHEET_ID = "1I2eZyK7PIhXEMwy30w8BgEcuRrLQQw4wK6GlxfAsuWE"

_parser = argparse.ArgumentParser(description="Extract citations from conference PDFs")
_parser.add_argument("--conference", dest="conference", default="Crypto",
                     choices=["Crypto", "EuroCrypt", "Oakland", "USENIX"],
                     help="Google Sheets tab (conference) to process")
CONFERENCE = _parser.parse_args().conference


# -----------------------------
# Corpus loading: Google Sheets
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

# -----------------------------
# Corpus loading: PDF matching
# -----------------------------

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


# ==============================================================================
# Citation extraction
# PDF text → list of raw reference strings
# ==============================================================================

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


# ==============================================================================
# Main pipeline
# For each paper: extract citations from PDF → write to raw CSV
# ==============================================================================

n_pdf_errors       = 0  # fitz raised an exception opening the PDF
n_no_refs_heading  = 0  # no "References"/"Bibliography" heading found in text
n_truncated_early  = 0  # references section cut short by appendix/acknowledgements marker
n_empty_refs_text  = 0  # heading found but extracted text was empty after truncation
parse_totals = {"dropped_too_short": 0, "stray_lines": 0}

TEXT_DIR = Path(f"text/{CONFERENCE}")
TEXT_DIR.mkdir(parents=True, exist_ok=True)

def _sanitize_filename(title: str, max_len: int = 100) -> str:
    s = re.sub(r'[^\w\s\-]', '', title)
    s = re.sub(r'\s+', '_', s.strip())
    return s[:max_len].rstrip('_')

citation_rows = []

for title, pdf_path in deduped_pdfs.items():
    pdf_text = extract_text_from_pdf(pdf_path)
    if not pdf_text:
        n_pdf_errors += 1
        continue

    (TEXT_DIR / f"{_sanitize_filename(title)}.txt").write_text(pdf_text, encoding="utf-8")

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
        ref_clean = re.sub(r'\s*Authorized licensed use limited to:.*$', '', ref_clean, flags=re.I | re.DOTALL).strip()

        citation_rows.append({
            "source_paper": title,
            "app_awareness": awareness,
            "raw_reference": ref_clean,
        })

df_citations = pd.DataFrame(citation_rows)
_csv_opts = dict(index=False, escapechar="\\", quoting=1)
df_citations.to_csv(f"csv/{CONFERENCE}_citations_raw.csv", **_csv_opts)
print(f"Saved {len(df_citations)} citation rows to csv/{CONFERENCE}_citations_raw.csv")
print(f"Saved {len(deduped_pdfs) - n_pdf_errors} full-text files to {TEXT_DIR}/")

# Fall-through summary
print(f"\nPDF extraction fall-throughs ({len(deduped_pdfs)} papers attempted):")
print(f"  fitz errors (could not open PDF):          {n_pdf_errors}")
print(f"  no references heading found:               {n_no_refs_heading}")
print(f"  references section truncated by marker:    {n_truncated_early}  (still processed)")
print(f"  empty text after truncation:               {n_empty_refs_text}")
print(f"\nReference parser fall-throughs (across all papers):")
print(f"  blocks dropped — too short (<35 chars):    {parse_totals['dropped_too_short']}")
print(f"  stray lines before first marker (ignored): {parse_totals['stray_lines']}")

papers_processed = len(deduped_pdfs)
total = len(df_citations)
expected = papers_processed * 60
print(f"\nPapers in corpus: {len(paper_titles)} in sheet, {papers_processed} PDFs found and processed")
print(f"Expected citations (rough): {papers_processed} papers × 60 = {expected}")
print(f"Citations extracted:        {total}  ({100*total/expected:.1f}% of expected)")

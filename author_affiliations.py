# author_affiliations.py
# Nathaniel Clizbe (github.com/cliz1), Feb 2026
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

# -----------------------------
# Config
# -----------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Paths
ZOTERO_STORAGE = Path("/Users/nathanielclizbe/Zotero/storage/") # replace with path to local Zotero storage

SAMPLE_SIZE = 400


SPREADSHEET_ID = "1I2eZyK7PIhXEMwy30w8BgEcuRrLQQw4wK6GlxfAsuWE" # find this in the sheets URL should it ever change
TEST_RANGE = "USENIX" # One conference (sheet) per run


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
    # Remove author prefix before colon
    raw = re.sub(r"^.*?:\s*", "", raw)

    # Remove year in parentheses
    raw = re.sub(r"\(\d{4}\)", "", raw)

    # Remove LaTeX math
    raw = re.sub(r"\$.*?\$", "", raw)

    return raw.strip()


# ------------------------------------------------------------------------------
# Collect titles from google sheets AND record app. awareness for later use
# ------------------------------------------------------------------------------

count = 0
paper_titles = []
app_awareness = {}
awareness_values = ["1", "2", "3", "4"]

service = get_sheets_service()
sheet = service.spreadsheets()

result = sheet.values().get(
    spreadsheetId=SPREADSHEET_ID,
    range=TEST_RANGE
).execute()

rows = result.get("values", [])

if not rows:
    print("No data found in sheet.")
else:
    for row in rows:
        if count == SAMPLE_SIZE:
            break
        if len(row) > 2:
            if row[2] == "1" or row[2] == "2": # Crypto or Analysis
                count +=1
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

        if len(title_tokens) < 4:
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

deduped_pdfs = {}

for title, pdf_list in matches.items():
    best_pdf = max(pdf_list, key=lambda p: p.stat().st_size)
    deduped_pdfs[title] = best_pdf


# -----------------------------------------------------------
# Helper for author extraction
# -----------------------------------------------------------

def extract_author_section(text: str, max_chars: int = 5000) -> str:
    """
    Returns text from the beginning of the paper up to the Abstract section.
    Only inspects the first `max_chars` to avoid scanning the entire document.
    """

    # Only look at beginning of document
    text = text[:max_chars]

    # Stop at Abstract (common heading)
    m = re.search(r"(?:^|\n)\s*(abstract)\b", text, re.I)
    if m:
        text = text[:m.start()]

    return text.strip()

def looks_like_affiliation(text: str) -> bool:
    """
    Returns True if the line likely contains an institutional affiliation.
    """

    if not text:
        return False

    keywords = [
        "university", "institute", "department", "school",
        "college", "laboratory", "lab", "research",
        "corp", "inc", "ltd", "llc", "company",
        "gmbh", "faculty", "centre", "center"
    ]

    return bool(re.search(r"\b(" + "|".join(keywords) + r")\b", text, re.I))

def parse_affiliations(text: str, title: str = "", debug=False):
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    affiliations = []
    title_norm = normalize(title) if title else ""
    title_tokens = set(title_norm.split())

    # NEW: detect author-name lines (has superscript markers or (B) suffix)
    author_line_pattern = re.compile(r"\(\s*[Bb]\s*\)|^\d+$|\d+[,\d]*\s*\(")

    prev_affiliation = None

    for ln in lines:
        original_ln = ln

        ln = re.sub(r"^\d+\s*", "", ln)
        ln = re.sub(r"\s\d+$", "", ln)
        ln = re.sub(r"\S+@\S+", "", ln).strip(" ,.")

        if not ln:
            continue

        if len(ln) > 200:
            continue

        # skip author name lines with (B) or numeric superscripts
        if author_line_pattern.search(original_ln):
            if debug:
                print("Skipping author name line:", original_ln)
            continue

        # skip very short geographic-only fragments like "USA", "Germany"
        if re.match(r"^[A-Z]{2,3}$", ln) or (len(ln.split()) == 1 and ln[0].isupper()):
            if debug:
                print("Skipping lone token:", ln)
            continue

        if title_tokens:
            ln_norm = normalize(ln)
            # normalize hyphens before tokenizing to catch hyphenated title fragments
            ln_norm_dehyphen = ln_norm.replace("-", " ")
            title_norm_dehyphen = title_norm.replace("-", " ")
            title_tokens_dehyphen = set(title_norm_dehyphen.split())
            ln_tokens = set(ln_norm_dehyphen.split())

            if len(ln_tokens) >= 3:
                overlap = ln_tokens & title_tokens_dehyphen
                if len(overlap) / len(ln_tokens) > 0.4:
                    if debug:
                        print("Skipping title-like line:", original_ln)
                    continue

        academic_patterns = [
            r"\buniversity\b", r"\buniv\b",
            r"\bcollege\b", r"\binstitute\b",
            r"\bschool\b", r"\bfaculty\b",
            r"\bdepartment\b", r"\bcentre\b",
            r"\bcenter\b", r"\bcnrs\b",
            r"\birisa\b", r"\bens\b",
            r"\buc\s?[A-Z]"
        ]

        industry_patterns = [
            r"\bcorp\b", r"\binc\b", r"\bltd\b",
            r"\bllc\b", r"\bcompany\b",
            r"\bgmbh\b", r"\blabs?\b",
            r"\bresearch\b"
        ]

        location_pattern = r"^[A-Z][A-Za-z\- ]+,\s*[A-Z][A-Za-z\- ]+$"

        is_academic = any(re.search(p, ln, re.I) for p in academic_patterns)
        is_industry = any(re.search(p, ln, re.I) for p in industry_patterns)
        is_location = re.match(location_pattern, ln)

        # capital_phrase no longer alone qualifies — must also match academic/industry/location
        capital_phrase = (
            1 <= len(ln.split()) <= 4
            and all(word[0].isupper() for word in ln.split() if word[0].isalpha())
        )

        if is_academic or is_industry or is_location:
            affiliations.append(ln)
        elif debug:
            print("Skipping non-affiliation:", original_ln)

    return affiliations

# -----------------------------------------------------------
#  Categorization helper
# -----------------------------------------------------------

def categorize_affiliation(aff: str) -> str:
    aff_lower = aff.lower()

    academic_patterns = [
        r"\buniversity\b", r"\buniv\b", r"\bcollege\b",
        r"\binstitute of technology\b", r"\binstitute\b",
        r"\bschool of\b", r"\bfaculty\b", r"\bdepartment\b",
        r"\bacademy\b", r"\bpolytechnic\b"
    ]

    government_patterns = [
        r"\bnational laboratory\b", r"\bnational lab\b",
        r"\bgovernment\b", r"\bministry\b", r"\bagency\b",
        r"\bnist\b", r"\bnsa\b", r"\bdarpa\b", r"\bdod\b",
        r"\bcnrs\b", r"\binria\b", r"\bcas\b",
        r"\bnational research\b", r"\bfederal\b",

        # European agencies
        r"\banssi\b",       # France
        r"\bbsi\b",         # Germany
        r"\bncsc\b",        # UK / Netherlands
        r"\bgchq\b",        # UK
        r"\baivd\b",        # Netherlands
        r"\bbnd\b",         # Germany
        r"\bfsi\b",         # various

        # Research funding bodies often attached to government
        r"\bfraunhofer\b",  # Germany
        r"\bcei\b",
        r"\bcea\b",         # France atomic energy
        r"\bdstl\b",        # UK defence science

        # Asia-Pacific
        r"\bnict\b",        # Japan
        r"\bnist\b",
        r"\basd\b",         # Australia signals
        r"\bkisa\b",        # Korea
    ]

    industry_patterns = [
        r"\bcorp\b", r"\binc\b", r"\bltd\b", r"\bllc\b",
        r"\bgmbh\b", r"\bcompany\b", r"\bco\.\b",
        r"\blabs?\b", r"\bresearch lab\b",
        r"\bntt\b", r"\bibm\b", r"\bgoogle\b", r"\bmicrosoft\b",
        r"\bamazon\b", r"\bmeta\b", r"\bapple\b", r"\bintel\b",
        r"\bpqshield\b", r"\bsilence labs\b", r"\bdeel\b"
    ]

    is_academic = any(re.search(p, aff_lower) for p in academic_patterns)
    is_government = any(re.search(p, aff_lower) for p in government_patterns)
    is_industry = any(re.search(p, aff_lower) for p in industry_patterns)

    # Avoid mislabeling academic institutions that mention "research"
    if is_academic and is_industry:
        return "Academic"
    if is_academic:
        return "Academic"
    if is_government:
        return "Government"
    if is_industry:
        return "Industry"

    return "Unknown"

# -----------------------------------------------------------
# Extract and display author affiliations for matched papers
# -----------------------------------------------------------

def extract_text_from_pdf_first_page(pdf_path: Path) -> str:
    """
    Extract text only from the first page (where affiliations almost always appear).
    """
    try:
        doc = fitz.open(pdf_path)
        if len(doc) > 0:
            return doc[0].get_text("text")
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return ""


results = {}

# ---------------------------------
# Build results and display
# ---------------------------------

rows = []

for title, pdf_path in deduped_pdfs.items():
    print("=" * 80)
    print(f"TITLE: {title}")
    print(f"PDF: {pdf_path.name}")

    text = extract_text_from_pdf_first_page(pdf_path)

    if not text:
        print("No text extracted.")
        continue

    author_section = extract_author_section(text)
    affiliations = parse_affiliations(author_section, title)

    # Deduplicate
    affiliations = list(dict.fromkeys(affiliations))

    # Categorize
    categories = [categorize_affiliation(aff) for aff in affiliations]
    category_counts = {"Academic": 0, "Government": 0, "Industry": 0, "Unknown": 0}
    for cat in categories:
        category_counts[cat] += 1

    if SAMPLE_SIZE <= 20:
        # Text mode
        if not affiliations:
            print("No affiliations detected.")
        else:
            print("\nDetected Affiliations:")
            for aff, cat in zip(affiliations, categories):
                print(f"  - [{cat}] {aff}")
    
    results[title] = affiliations

    # Attach awareness for graph mode
    awareness = app_awareness.get(title)
    if awareness is None:
        continue

    total = sum(category_counts.values())
    if total == 0:
        continue

    normalized = {k: v / total for k, v in category_counts.items()}
    rows.append({
        "title": title,
        "app_awareness": int(awareness),
        **normalized
    })

print("\nDone.")

# ---------------------------------
# Graph mode
# ---------------------------------

if SAMPLE_SIZE > 20 and rows:
    df = pd.DataFrame(rows)

    bucket_cols = ["Academic", "Government", "Industry", "Unknown"]
    grouped = df.groupby("app_awareness")[bucket_cols].mean()

    colors = ["#1f77b4", "#2ca02c", "#ff7f0e", "#d62728"]

    grouped.plot(
        kind="bar",
        stacked=True,
        figsize=(8, 5),
        color=colors
    )

    plt.ylabel("Average Affiliation Share")
    plt.xlabel("Application Awareness Level")
    plt.title(f"Author Affiliation Distribution by Application Awareness Level - {TEST_RANGE}")
    plt.legend(title="Affiliation Type", bbox_to_anchor=(1.05, 1))
    plt.tight_layout()
    plt.show()
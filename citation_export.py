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

# -----------------------------
# Config
# -----------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Paths
ZOTERO_STORAGE = Path("/Users/nathanielclizbe/Zotero/storage/") 

SAMPLE_SIZE = 400


SPREADSHEET_ID = "1I2eZyK7PIhXEMwy30w8BgEcuRrLQQw4wK6GlxfAsuWE" 
TEST_RANGE = "Crypto" # One conference (sheet) per run

CRYPTO_KEYWORDS =  [
        "crypto",
        "eurocrypt",
        "asiacrypt",
        "pkc",
        "tcc",
        "fse",
        "ch es",
        "icalp",  
        "lncs", 
        "ppml",
        "mpc",
        "differential privacy",
         "arxiv", "eprint", "corr", "acns", "sac", "fc",
        "pldi", "focs", "stock", "soda", "siam", "ieee trans.", "journal of cryptology",
        "open whisper systems", "arkworks", "blockchain protocols", "pos proofs", "oracles", "group key exchange",
        "double ratchet", "streamlet", "crypten"
    ]

SECURITY_SYSTEMS_KEYWORDS =  [
        "ccs",
        "oakland",
        "ieee s&p",
        "security and privacy",
        "ndss",
        "usenix security",
        "usenix",
        "sigcomm",
        "www",
        "acsac",
        "EuroS&P",
        "ieee european symposium on security and privacy",
        "computer security", "eurosec", "provsec", "communications security",
        "csf", "micro", "dimva", "host", "s&p", "ieee security & privacy", "iee trans. dependable secure comput.", "network security",
        "anonymous messaging", "ddos", "side channel", "side-channel", "matrix.org",
            "consensus",
        "byzantine fault tolerance",
        "bft",
        "paxos",
        "dag-based",
        "hotstuff",
        "zyzzyva",
        "sync hotstuff",
        "state machine replication",
        "permissionless network",
        "stellar",
        "ethereum node tracker",
        "distributed systems",
        "peer-to-peer",
        "accountable systems",
        "mempool",
        "network measurement",
        "sosp",
        "eurosys",
        "icdcs",
        "podc",
        "acm trans. comput. syst.",
    ]

REAL_WORLD_KEYWORDS = [
    "reuters", "bbc", "guardian", "new york times", "nyt",
    "washington post", "washpost", "wsj", "wall street journal",
    "bloomberg", "financial times", "ft.com", "cnn", "fox news","npr",
    "associated press", "ap news", "politico", "axios", "the verge",
    "wired", "ars technica", "techcrunch", "engadget", "vice", "forbes",
    "nist",
    "white house",
    "executive order",
    "department of",
    "ministry of",
    "u.s. government",
    "us government",
    "congress",
    "house of representatives",
    "senate",
    "federal register",
    "government accountability office",
    "gao",
    "department of justice",
    "doj",
    "department of homeland security",
    "dhs",
    "cisa",
    "nsa",
    "cia",
    "commission",
      "github.com",
    "source code",
    "repo",
    "repository",
        "aws",
    "amazon",
    "google cloud",
    "pricing",
    "prometheus",
    "druid",
        "twitter",
    "medium",
    "blog",
    "substack",
    "github.io",
    "sites.google.com",
    "bitcointalk.org",
        "game theory",
    "cooperative game",
    "non-cooperative",
    "repeated games",
    "supergames",
    "equilibrium",
    "sequential equilibria",
    "reputation",
    "market insurance",
    "self-insurance",
    "self-protection",
    "informal institutions",
    "institutional economics",
    "economic review",
    "american economic review",
        "iot",
    "internet of things",
    "sensor network",
    "wireless sensor",
    "lora",
    "helium network",
    "sigfox",
    "bluetooth",
    "ble",
    "wifi",
    "mobile ad hoc network",
    "manet",
    "starlink",
    "edge computing",
    "federated infrastructure",
    "city-scale sensing",
    "smart parking",
    "wildlife tracking",
    "forest fire detection",
    "energy harvesting sensors",
    "zebranet",
    "signpost platform",
    "esp-idp",
    "nordic semi",
    "location privacy",
    "wireless sensor nodes",
    "lan"

    ]

POLICY_KEYWORDS = [
    "nist",
    "white house",
    "executive order",
    "department of",
    "ministry of",
    "u.s. government",
    "us government",
    "congress",
    "house of representatives",
    "senate",
    "federal register",
    "government accountability office",
    "gao",
    "department of justice",
    "doj",
    "department of homeland security",
    "dhs",
    "cisa",
    "nsa",
    "cia",
    "commission"]

TECH_DOC_KEYWORDS = [
    "rfc ",
    "rfc-",
    "rfc:",
    "internet-draft",
    "ietf draft",
    "ietf",
    "request for comments",
    "w3c",
    "iso/",
    "iec",
    "white paper", "whitepaper", "documentation"]

SOURCE_CODE_KEYWORDS = [
    "github.com",
    "source code",
    "repo",
    "repository"
]

VENDOR_DOC_KEYWORDS = [
    "aws",
    "amazon",
    "google cloud",
    "pricing",
    "prometheus",
    "druid"
]

INDUSTRY_BLOG_KEYWORDS = [
    "twitter",
    "medium",
    "blog",
    "substack",
    "github.io",
    "sites.google.com",
    "bitcointalk.org"
]

MATH_KEYWORDS = ["gaussian", "matrix", "l2 norm", "discrete logarithms", "modular multiplication", "statistical", "probability"]

ECONOMICS_GT_KEYWORDS = [
    "game theory",
    "cooperative game",
    "non-cooperative",
    "repeated games",
    "supergames",
    "equilibrium",
    "sequential equilibria",
    "reputation",
    "market insurance",
    "self-insurance",
    "self-protection",
    "informal institutions",
    "institutional economics",
    "economic review",
    "american economic review",
]

IOT_NETWORK_KEYWORDS = [
    "iot",
    "internet of things",
    "sensor network",
    "wireless sensor",
    "lora",
    "helium network",
    "sigfox",
    "bluetooth",
    "ble",
    "wifi",
    "mobile ad hoc network",
    "manet",
    "starlink",
    "edge computing",
    "federated infrastructure",
    "city-scale sensing",
    "smart parking",
    "wildlife tracking",
    "forest fire detection",
    "energy harvesting sensors",
    "zebranet",
    "signpost platform",
    "esp-idp",
    "nordic semi",
    "location privacy",
    "wireless sensor nodes",
    "lan"
]


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


# ---------------------------------------------
# Functions for Parsing and Extracting References
# ---------------------------------------------
def extract_text_from_pdf(pdf_path: Path) -> str:
    text = ""
    try:
        doc = fitz.open(pdf_path)
        for page in doc:
            text += page.get_text("text") + "\n"
    except Exception as e:
        print(f"Error reading {pdf_path}: {e}")
    return text

def extract_references_section(text: str) -> str:
    """
    Returns everything after the last occurrence of "References" as a standalone heading.
    Using the last match avoids false hits on table-of-contents entries (e.g. "References ... 42").
    """
    # Match "References" as a full word at start of line or after newline
    matches = list(re.finditer(r"(?:^|\n)\s*(references|bibliography|references and notes)\b", text, re.I))
    if not matches:
        return ""  # no references section found
    m = matches[-1]  # use the last match — bibliography is always at end of paper

    # slice from the end of the match
    references_text = text[m.end():]

    # stop at common section markers after references
    for marker in ["appendix", "acknowledgements", "supplementary"]:
        idx = re.search(rf"\n\s*{marker}\b", references_text, re.I)
        if idx:
            references_text = references_text[:idx.start()]

    return references_text.strip()

def dehyphenate(text: str) -> str:
    """Collapse soft-hyphen line-break artifacts from two-column PDF extraction.'.
    """
    return re.sub(r'(\w)-\s+(\w)', r'\1-\2', text)


def looks_like_body_text(text: str) -> bool:
    """
    Returns True if the block looks like body/prose rather than a reference.
    Checks for keywords 'we', 'hence', or 'therefore'.
    """
    if not text:
        return False

    # Return True if any of the keywords appear (case-insensitive)
    return bool(re.search(r"\b(we|hence|therefore|such that|this algorithm|sends|s.t.|←|adversary outputs|given|this is|as well as|indeed|compromised|Theorem|assume|lemma|Corollary|computes)\b", text, re.I))


def parse_references(text: str, debug=False):
    """
    Parses references handling both:
      - numeric markers: 1. ...  or [1] ...
      - bracketed alphanumeric markers: [ANWW13], [BCG+19a], [BCGI18], etc.
    Skips blocks that look like body text.
    """

    # Remove leading "References" header
    text = re.sub(r"^\s*references\s*", "", text, flags=re.IGNORECASE)

    # Remove common page-range patterns like "pp. 33–53."
    text = re.sub(r"\bpp\.?\s*\d{1,4}\s*[-–]\s*\d{1,4}\.?", "", text)

    # Insert newline before reference markers to help splitting
    marker_lookahead = r"(?=(\d{1,2}\.\s|\[\d{1,2}\]\s|\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*))"
    text = re.sub(r"\s" + marker_lookahead, "\n", text)

    # Split into lines and strip whitespace
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    refs = []
    current = None
    # Regex to detect start-of-reference lines
    marker_re = re.compile(r"^(\d{1,2}\.\s|\[\d{1,2}\]\s|\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*)")

    for ln in lines:
        if marker_re.match(ln):
            # start a new reference
            if current:
                # Skip blocks that look like body text or if too short
                if not looks_like_body_text(current) and len(current) > 35:
                    refs.append(current.strip())
                elif debug:
                    print("Skipping body-like candidate:", current[:80].replace("\n", " "))
            current = ln
        else:
            # continuation line: append to the current ref if exists
            if current is not None:
                current += " " + ln
            else:
                # stray line before first marker: ignore
                pass

    # append the final buffered reference if any
    if current and not looks_like_body_text(current):
        refs.append(current.strip())
    elif current and debug:
        print("Skipping final body-like candidate:", current[:80].replace("\n", " "))

    return refs

# -------------------------------------
# Function for Categorizing References - also collect some "other" citations for inspection
# -------------------------------------

others = []
def classify_reference(reference: str):
    c = reference.lower()

    scores = {
        "crypto": 0,
        "security": 0,
        "standards": 0,
        "external": 0,
        "unclassified": 0,
    }

    scores["crypto"] += sum(k in c for k in CRYPTO_KEYWORDS)
    scores["security"] += sum(k in c for k in SECURITY_SYSTEMS_KEYWORDS)
    scores["standards"] += sum(k in c for k in TECH_DOC_KEYWORDS)
    scores["external"] += sum(k in c for k in REAL_WORLD_KEYWORDS)

    # No matches at all -> OTHER
    if all(v == 0 for v in scores.values()):
        #others.append(reference)
        return "unclassified"

    # Pick category with highest score
    best = max(scores, key=scores.get)

    return best


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


def _fetch_dblp_venue(title: str) -> str:
    """Single DBLP title lookup; returns raw venue string or '' on miss.
    Raises _DblpRateLimited on HTTP 429 so the caller can bail instead of retrying."""
    try:
        query = urllib.parse.quote(title)
        url = f"https://dblp.org/search/publ/api?q={query}&format=json&h=1"
        req = urllib.request.Request(url, headers={"User-Agent": "citation-analysis-research/1.0"})
        context = ssl.create_default_context(cafile=certifi.where())
        with urllib.request.urlopen(req, timeout=10, context=context) as response:
            data = json.loads(response.read())
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return ""
        venue = hits[0].get("info", {}).get("venue", "")
        return venue.strip() if venue else ""
    except urllib.error.HTTPError as e:
        if e.code == 429:
            raise _DblpRateLimited()
        return ""
    except Exception:
        return ""


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

    # Fallback: strip citation marker, skip past author list (first ". "),
    # take the remaining text as the title start.
    if not title:
        body = re.sub(r'^\[?\d{1,3}\]?\s*|^\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*', '', raw_reference)
        first_period = body.find('. ')
        title = body[first_period + 2:] if first_period > 5 else body

    # Strip any trailing ". In VENUE" / ". arXiv" / ". IACR" that leaked into the title
    title = re.sub(r'\.\s+(?:In\b|arXiv|IACR|CoRR).*$', '', title, flags=re.I).strip()

    # Skip obviously hopeless queries
    if re.fullmatch(r'(?i)private\s+communication\.?', title.strip()):
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
        print(f"    DBLP query ({len(variant.split())}w): '{variant[:70]}'")
        try:
            raw_venue = _fetch_dblp_venue(variant)
        except _DblpRateLimited:
            print("    DBLP rate-limited (429) — skipping remaining variants")
            return ""
        if raw_venue:
            return _DBLP_VENUE_MAP.get(raw_venue.lower(), raw_venue)
        if i < len(variants) - 1:
            time.sleep(1.0)  # match inter-ref cadence; genuine miss, not rate-limit

    return ""

# ---------------------------------
# Venue extraction helper
# ---------------------------------

def extract_venue(reference: str) -> str:

    # LNCS style: "In: Editors (eds.) VENUE YEAR. LNCS"
    m = re.search(r"In:\s+.{0,80}?\(eds?\.\)\s+([A-Z][A-Za-z0-9 &\-]+\d{4})", reference)
    if m:
        return m.group(1).strip()

    # LNCS abbreviated: "In: VENUE YEAR. LNCS, vol."
    m = re.search(r"In:\s+([A-Z][A-Za-z0-9 &\-]+\d{4})\.", reference)
    if m:
        return m.group(1).strip()

    # LNCS style: "In: Editors (eds.) VENUE YEAR. LNCS"
    m = re.search(r"In:\s+.{0,80}?\(eds?\.\)\s+([A-Z][A-Za-z0-9 &\-–—]+\d{4})", reference)

    # LNCS abbreviated: "In: VENUE YEAR. LNCS, vol."
    m = re.search(r"In:\s+([A-Z][A-Za-z0-9 &\-–—]+\d{4})\.", reference)

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

    # IEEE/ACM short style: "in S&P 2021," / "In CCS 2018," / "In ACM PODC, 2019"
    m = re.search(r"\bin\s+([A-Z][A-Za-z0-9 &'\-]{2,40}),?\s*\d{4}", reference, re.I)
    if m:
        return m.group(1).strip()

    # "In Proceedings of ..." or "In Proc. ..."
    m = re.search(r"\bIn\s+(?:Proceedings\s+of\s+|Proc\.?\s+)([^,\.]+)", reference, re.I)
    if m:
        return m.group(1).strip()

    # Journal style: "J. Cryptology" / "SIAM J. Comput." etc.
    m = re.search(r"\b((?:J\.|Journal|Trans\.|IEEE Trans\.|ACM Trans\.)\s+[A-Za-z][A-Za-z0-9 \.]{2,40})", reference)
    if m:
        candidate = m.group(1).strip()
        # "J. Lastname" / "J. B. Lastname" patterns are author initials, not journals.
        # Real journal abbreviations either contain all-caps words (ACM, SIAM) or
        # words ending in a period (Cryptol., Comput.). A bare capitalized surname has neither.
        if not re.match(r'^J\.\s+(?:[A-Z]\.\s+)*[A-Z][a-z]{2,}(?:[\.,](?:\s+[A-Z]|$)|\s+[a-z]|$)', candidate):
            return candidate
    
    # Short acronym style: "In: ITC (2023)" or "In: 24th ACM STOC"
    m = re.search(r"In:\s+(?:\d+(?:st|nd|rd|th)\s+)?([A-Z][A-Z0-9&\- ]{1,30})[,\s]+(?:ACM|IEEE|Springer)?\s*(?:Press\b|,)?\s*\w*\s*\d{4}", reference)
    if m:
        return m.group(1).strip()

    # Abbreviated journal names ending in volume/issue numbers
    m = re.search(r"([A-Z][A-Za-z\. ]{3,40})\s+\d+\(\d+\)", reference)
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
# Main Processing Loop — venue extraction
# ---------------------------------

citation_rows = []

for title, pdf_path in deduped_pdfs.items():
    pdf_text = extract_text_from_pdf(pdf_path)
    references_text = extract_references_section(pdf_text)
    if not references_text:
        print(f"No References section found: {pdf_path.name}")
        continue

    parsed_refs = parse_references(references_text)
    awareness = app_awareness.get(title)

    for ref in parsed_refs:
        ref_clean = dehyphenate(ref)
        venue = extract_venue(ref_clean)
        if venue:
            source = "regex"
        else:
            print(f"  DBLP query ref: {ref_clean[:80]}")
            venue = query_dblp_for_venue(ref_clean)
            source = "dblp" if venue else "none"
            print(f"  DBLP result: {venue or 'none'}")
            time.sleep(1.5)

        citation_rows.append({
            "source_paper": title,
            "app_awareness": awareness,
            "venue_raw": venue,
            "venue_source": source,
            "raw_reference": ref,
        })

df_citations = pd.DataFrame(citation_rows)
df_citations.to_csv(
    f"{TEST_RANGE}_citations_raw.csv",
    index=False,
    escapechar="\\",
    quoting=1  # QUOTE_ALL — wraps every field in quotes, sidesteps the issue entirely
)
print(f"Saved {len(df_citations)} citation rows to {TEST_RANGE}_citations_raw.csv")

# Quick diagnostic: how many references got a venue extracted?
extracted = df_citations[df_citations["venue_raw"] != ""].shape[0]
total = len(df_citations)
print(f"Venue extracted: {extracted}/{total} ({100*extracted/total:.1f}%)")

# Preview unmatched to tune regex
unmatched_sample = df_citations[df_citations["venue_raw"] == ""]["raw_reference"].head(20)
#print("\nSample of references with no venue extracted:")
#for r in unmatched_sample:
    #print(" ", r[:120])

 # Check venue extraction rate by app_awareness level
print("\nExtraction rate by awareness level:")
df_citations["extracted"] = df_citations["venue_raw"] != ""
print(df_citations.groupby("app_awareness")["extracted"].mean().round(3))
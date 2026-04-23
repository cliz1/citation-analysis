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

# -----------------------------
# Config
# -----------------------------

SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]

# Paths
ZOTERO_STORAGE = Path("/Users/nathanielclizbe/Zotero/storage/") # replace with path to local Zotero storage

SAMPLE_SIZE = 5


SPREADSHEET_ID = "1I2eZyK7PIhXEMwy30w8BgEcuRrLQQw4wK6GlxfAsuWE" # find this in the sheets URL should it ever change
TEST_RANGE = "EuroCrypt" # One conference (sheet) per run

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
    Returns everything after the first occurrence of "References" as a standalone word or heading.
    Ignores words like "dereferences" that contain "references".
    """
    # Match "References" as a full word, optionally with punctuation, at start of line or after newline
    m = re.search(r"(?:^|\n)\s*(references|bibliography|references and notes)\b", text, re.I)
    if not m:
        return ""  # no references section found

    # slice from the end of the match
    references_text = text[m.end():]

    # stop at common section markers after references
    for marker in ["appendix", "acknowledgements", "supplementary"]:
        idx = re.search(rf"\n\s*{marker}\b", references_text, re.I)
        if idx:
            references_text = references_text[:idx.start()]

    return references_text.strip()

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

def query_dblp_for_venue(raw_reference: str) -> str:
    title = ""

    # LNCS style: "Lastname, F.: Title. In: ..."
    m = re.search(r":\s+([A-Z][^:\.]{10,120})\.\s+In:", raw_reference)
    if m:
        title = m.group(1).strip()

    # Quoted title fallback
    if not title:
        m = re.search(r'"([^"]{10,120})"', raw_reference)
        if m:
            title = m.group(1).strip()

    # Period-delimited before In/IACR/arXiv fallback
    if not title:
        m = re.search(r'\.\s+([A-Z][^\.]{10,120})\.\s+(?:In|IACR|arXiv)', raw_reference)
        if m:
            title = m.group(1).strip()

    print(f"    extracted title: '{title[:60] if title else 'NONE'}'")  # add this

    if not title or len(title) < 10:
        title = raw_reference  # fallback

    try:
        query = urllib.parse.quote(title)
        url = f"https://dblp.org/search/publ/api?q={query}&format=json&h=1"
        req = urllib.request.Request(url, headers={"User-Agent": "citation-analysis-research/1.0"})
        with urllib.request.urlopen(req, timeout=10) as response:
            data = json.loads(response.read())
        hits = data.get("result", {}).get("hits", {}).get("hit", [])
        if not hits:
            return ""
        info = hits[0].get("info", {})
        venue = info.get("venue", "")
        return venue.strip() if venue else ""
    except Exception:
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

    # IEEE/ACM short style: "in S&P 2021," or "in CCS 2018,"
    m = re.search(r"\bin\s+([A-Z][A-Za-z0-9 &'\-]{2,40}),?\s*\d{4}", reference)
    if m:
        return m.group(1).strip()

    # "In Proceedings of ..." or "In Proc. ..."
    m = re.search(r"\bIn\s+(?:Proceedings\s+of\s+|Proc\.?\s+)([^,\.]+)", reference, re.I)
    if m:
        return m.group(1).strip()

    # Journal style: "J. Cryptology" / "SIAM J. Comput." etc.
    m = re.search(r"\b((?:J\.|Journal|Trans\.|IEEE Trans\.|ACM Trans\.)\s+[A-Za-z][A-Za-z0-9 \.]{2,40})", reference)
    if m:
        return m.group(1).strip()
    
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
    

    # ePrint / arXiv
    if re.search(r"eprint\.iacr\.org|Cryptol(?:ogy)?\s+ePrint", reference, re.I):
        return "ePrint"
    if re.search(r"arxiv\.org|arXiv", reference, re.I):
        return "arXiv"
    
    # Web/blog/forum references with no venue
    if re.search(r"https?://", reference):
        if re.search(r"github\.com|gitlab\.com", reference, re.I):
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
        venue = extract_venue(ref)
        if not venue:
            print(f"  DBLP query ref: {ref[:80]}")
            venue = query_dblp_for_venue(ref)
            print(f"  DBLP result: {venue or 'none'}")
            time.sleep(0.5)
            
        citation_rows.append({
            "source_paper": title,
            "app_awareness": awareness,
            "venue_raw": venue,
            "venue_source": "regex" if venue else "dblp" if venue else "none",
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
print("\nSample of references with no venue extracted:")
for r in unmatched_sample:
    print(" ", r[:120])

 # Check venue extraction rate by app_awareness level
print("\nExtraction rate by awareness level:")
df_citations["extracted"] = df_citations["venue_raw"] != ""
print(df_citations.groupby("app_awareness")["extracted"].mean().round(3))
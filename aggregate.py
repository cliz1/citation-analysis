# ref_analysis_multi.py

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

ZOTERO_STORAGE = Path("/Users/nathanielclizbe/Zotero/storage/")
SAMPLE_SIZE = 400
SPREADSHEET_ID = "1I2eZyK7PIhXEMwy30w8BgEcuRrLQQw4wK6GlxfAsuWE"

venues = ["USENIX", "Oakland", "Crypto", "EuroCrypt"]

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

    if Path("token.pickle").exists():
        with open("token.pickle", "rb") as token:
            creds = pickle.load(token)

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
# Helpers
# -----------------------------

def clean_title(raw: str) -> str:
    raw = re.sub(r"^.*?:\s*", "", raw)
    raw = re.sub(r"\(\d{4}\)", "", raw)
    raw = re.sub(r"\$.*?\$", "", raw)
    return raw.strip()

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

        long_words = [w for w in title_tokens if len(w) >= 7]
        if long_words and not any(w in filename_tokens for w in long_words):
            continue

        if len(overlap) / len(title_tokens) >= 0.7:
            return title

    return None

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
    m = re.search(r"(?:^|\n)\s*(references|bibliography|references and notes)\b", text, re.I)
    if not m:
        return ""

    references_text = text[m.end():]

    for marker in ["appendix", "acknowledgements", "supplementary"]:
        idx = re.search(rf"\n\s*{marker}\b", references_text, re.I)
        if idx:
            references_text = references_text[:idx.start()]

    return references_text.strip()

def parse_references(text: str):
    text = re.sub(r"^\s*references\s*", "", text, flags=re.IGNORECASE)

    marker_lookahead = r"(?=(\d{1,2}\.\s|\[\d{1,2}\]\s|\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*))"
    text = re.sub(r"\s" + marker_lookahead, "\n", text)

    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]

    refs = []
    current = None
    marker_re = re.compile(r"^(\d{1,2}\.\s|\[\d{1,2}\]\s|\[[A-Za-z][A-Za-z0-9+\-]{1,30}\]\s*)")

    for ln in lines:
        if marker_re.match(ln):
            if current and len(current) > 35:
                refs.append(current.strip())
            current = ln
        else:
            if current:
                current += " " + ln

    if current and len(current) > 35:
        refs.append(current.strip())

    return refs

def classify_reference(reference: str):
    # assumes keyword arrays exist
    c = reference.lower()

    scores = {
        "crypto": sum(k in c for k in CRYPTO_KEYWORDS),
        "security": sum(k in c for k in SECURITY_SYSTEMS_KEYWORDS),
        "standards": sum(k in c for k in TECH_DOC_KEYWORDS),
        "external": sum(k in c for k in REAL_WORLD_KEYWORDS),
        "unclassified": 0,
    }

    if all(v == 0 for v in scores.values()):
        return "unclassified"

    return max(scores, key=scores.get)

# -----------------------------
# Core pipeline per venue
# -----------------------------

def compute_grouped(TEST_RANGE):

    service = get_sheets_service()
    sheet = service.spreadsheets()

    result = sheet.values().get(
        spreadsheetId=SPREADSHEET_ID,
        range=TEST_RANGE
    ).execute()

    rows = result.get("values", [])

    paper_titles = []
    app_awareness = {}

    count = 0

    for row in rows:
        if count == SAMPLE_SIZE:
            break
        if len(row) > 2:
            if row[2] in ["1", "2"]:
                count += 1
                title = clean_title(row[0].strip())
                paper_titles.append(title)
                if row[4] in ["1","2","3","4"]:
                    app_awareness[title] = row[4]

    pdf_files = list(ZOTERO_STORAGE.rglob("*.pdf"))

    matches = defaultdict(list)

    for pdf in pdf_files:
        matched_title = pdf_matches_title(pdf, paper_titles)
        if matched_title:
            matches[matched_title].append(pdf)

    deduped_pdfs = {
        title: max(pdfs, key=lambda p: p.stat().st_size)
        for title, pdfs in matches.items()
    }

    data = {}

    for title, pdf_path in deduped_pdfs.items():
        data[title] = {"crypto":0,"security":0,"standards":0,"external":0,"unclassified":0}

        text = extract_text_from_pdf(pdf_path)
        refs_section = extract_references_section(text)

        if refs_section:
            refs = parse_references(refs_section)
            for ref in refs:
                bucket = classify_reference(ref)
                data[title][bucket] += 1

    rows = []

    for title, buckets in data.items():
        total = sum(buckets.values())
        if total == 0:
            continue

        normalized = {k: v/total for k,v in buckets.items()}
        awareness = app_awareness.get(title)

        if awareness is None:
            continue

        rows.append({
            "title": title,
            "app_awareness": int(awareness),
            **normalized
        })

    df = pd.DataFrame(rows)

    bucket_cols = ["crypto","security","external","standards","unclassified"]

    return df.groupby("app_awareness")[bucket_cols].mean()

# -----------------------------
# Run all venues + plot
# -----------------------------

plt.rcParams.update({
    "font.size": 12,
    "axes.titlesize": 14,
    "axes.labelsize": 12
})
plt.rcParams["pdf.fonttype"] = 42

results = {}

for venue in venues:
    print(f"Processing {venue}...")
    results[venue] = compute_grouped(venue)

fig, axes = plt.subplots(2, 2, figsize=(12, 8), sharey=True)
axes = axes.flatten()

colors = ["#4C72B0","#DD8452","#55A868","#C44E52","#8172B3"]

for i, (venue, grouped) in enumerate(results.items()):
    ax = axes[i]

    grouped.plot(
        kind="bar",
        stacked=True,
        ax=ax,
        color=colors,
        legend=False
    )

    ax.set_title(venue)
    ax.text(-0.15, 1.05, f"({chr(97+i)})", transform=ax.transAxes,
            fontsize=12, fontweight="bold")

# shared labels
fig.text(0.5, 0.04, "Application Awareness Level", ha="center")
fig.text(0.04, 0.5, "Average Citation Share", va="center", rotation="vertical")

# shared legend
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, title="Reference Type", loc="center right")

plt.tight_layout(rect=[0.05, 0.05, 0.85, 1])

plt.savefig("citation_distribution_all_venues.pdf")
plt.savefig("citation_distribution_all_venues.png", dpi=300)

plt.close()
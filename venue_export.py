# venue_export.py
# Nathaniel Clizbe (github.com/cliz1), January 2026
# Reads {CONFERENCE}_citations_raw.csv produced by citation_export.py and runs
# venue extraction (regex → DBLP → standards → grey literature) on each citation.
# Outputs {CONFERENCE}_citations_venues.csv and {CONFERENCE}_suspected_fps.csv.
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request
import json
import ssl
import certifi
import urllib.error
import pandas as pd
import argparse

# -----------------------------
# Config
# -----------------------------
 
_parser = argparse.ArgumentParser(description="Extract venues from raw citation CSV")
_parser.add_argument("--conference", dest="conference", default="Crypto",
                     choices=["Crypto", "EuroCrypt", "Oakland", "USENIX"],
                     help="Conference to process (must match a citations_raw CSV)")
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
# Load Stage 1 CSV
# -----------------------------

_raw_csv = Path(f"csv/{CONFERENCE}_citations_raw.csv")
if not _raw_csv.exists():
    raise FileNotFoundError(f"{_raw_csv} not found — run citation_export.py first")

df_raw = pd.read_csv(_raw_csv, escapechar="\\")
print(f"Loaded {len(df_raw)} citation rows from {_raw_csv}")


# ==============================================================================
# Venue extraction
# raw reference string → venue label; three passes in order:
#   Pass 1: regex patterns (fast, no network)  →  extract_venue()
#   Pass 2: DBLP title lookup (network fallback)  →  query_dblp_for_venue()
#   Pass 3: standards / grey-literature patterns (post-DBLP)  →  match_standards() / match_grey_lit()
# ==============================================================================

# ── FP filter ────────────────────────────────────────────────────────────────

def is_likely_real_citation(raw_ref: str) -> bool:
    """
    Returns False if raw_ref looks like a parser artifact rather than a real citation.
    Known artifact types that fail all checks below:
      - Table rows (e.g. "[82] FA,FE,C10 Trim FedSGD ...")
      - Proof steps (e.g. "1. The general case follows from ...")
      - DOI fragments (e.g. "05. doi: 10.1007/978-...")
    Rows that return False skip venue extraction and DBLP entirely and are written
    to the FP audit CSV. Confirmed against 893 flagged rows across 4 conferences:
    0 true positives.
    """
    return bool(
        re.search(r'\b(?:19|20)\d{2}\b', raw_ref)               # 4-digit year
        or re.search(r'https?:\s*//', raw_ref)                   # https:// URL (also catches soft-wrapped)
        or re.search(r'\bdoi\s*:\s*10\.\d{4}/', raw_ref, re.I)  # bare DOI without https:// prefix
        or re.search(r'\bIn:\s+[A-Z]', raw_ref)                 # "In: VENUE" — LNCS citation marker
        or re.search(r'\bser\.\s+[A-Z]', raw_ref)               # "ser. VENUE'YY" — IEEE citation marker
        or re.search(r'\bAdvances\s+in\s+Cryptology\b', raw_ref, re.I)  # CRYPTO/EUROCRYPT series name
        or re.search(r'\bProceedings\b', raw_ref)                        # "in Proceedings of ..."
    )


# ── Pass 1: regex patterns ────────────────────────────────────────────────────

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

# Full conference name → standard acronym; checked in extract_venue() before the URL block.
_CONF_FULL_NAMES: list[tuple[str, str]] = [
    (r"Annual International Cryptology Conference", "CRYPTO"),
    (r"Annual Cryptology Conference", "CRYPTO"),   # Chicago-style, drops "International"
    (r"Annual International Conference on the Theory and Applications of Cryptographic Techniques", "EUROCRYPT"),
    (r"International Conference on the Theory and Application of Cryptology and Information Security", "ASIACRYPT"),
    (r"IACR International (?:Conference|Workshop) on Public.Key Cryptography", "PKC"),
    (r"ACM (?:SIGSAC )?Conference on Computer and Communications Security", "CCS"),
    (r"(?:International Conference on\s+)?Applied Cryptography and Network Security", "ACNS"),
    (r"International Workshop on Fast Software Encryption", "FSE"),
    (r"International Workshop on Selected Areas in Cryptography", "SAC"),
    (r"Theory of Cryptography Conference", "TCC"),
    (r"Innovations in Theoretical Computer Science(?:\s+Conference)?", "ITCS"),
    (r"(?:International Conference on\s+)?Financial Cryptography and Data Security", "FC"),
    (r"(?:IEEE\s+)?European Symposium on Security and Privacy", "EuroS&P"),
    (r"European Symposium on Research in Computer Security", "ESORICS"),
    (r"Annual Computer Security Applications Conference", "ACSAC"),
    (r"USENIX Security Symposium|USENIX Security", "USENIX Security"),
    (r"Proceedings on Privacy Enhancing Technologies", "PoPETs"),
    (r"IACR Real World Crypto Symposium", "RWC"),
    (r"IEEE International Symposium on Information Theory", "ISIT"),
]


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
    m = re.search(r"\b((?:J\.|Journal|Trans\.|IEEE Trans(?:actions)?\.?|ACM Trans\.)\s+[A-Za-z][A-Za-z0-9 \.]{2,40})", reference)
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

    # "In 2020 IEEE Symposium on Security and Privacy (SP), pages 947" — year-first with
    # parenthetical acronym between venue name and page range
    m = re.search(r"\bIn\s+\d{4}\s+([^(]+?)\s*\([A-Z]+\),\s*pages\s+\d", reference)
    if m:
        return m.group(1).strip()

    # "In STOC. ACM, 1990" / "In NDSS. The Internet Society, 1999" — venue before ". Publisher"
    m = re.search(r"\bIn\s+([A-Z][A-Z0-9]{2,10})\.\s+(?:ACM|IEEE|Springer|USENIX|The\s+Internet\s+Society)[,\s]", reference, re.I)
    if m:
        return m.group(1).strip()

    # "In CRYPTO (2), pages" — venue with volume number in parens
    m = re.search(r"\bIn\s+([A-Z][A-Z0-9]{2,10})\s+\(\d{1,2}\)[,\s]", reference, re.I)
    if m:
        return m.group(1).strip()

    # "in FMCAD, ser. Lecture Notes..." — venue immediately before ", ser."
    m = re.search(r"\bIn\s+([A-Z][A-Z0-9&\- ]{1,20}),\s*ser\.", reference, re.I)
    if m:
        return m.group(1).strip()

    # "ser. SP, 2013" / "ser. Crypto'12, 2012" — terse format with no "In" prefix
    # Include apostrophe so "Crypto'12" is captured whole rather than cut at the apostrophe.
    m = re.search(r"\bser\.\s+([A-Z][A-Za-z0-9']{1,15}),?\s+\d{4}", reference)
    if m:
        return m.group(1).strip()

    # Full conference name aliases (e.g. "Annual International Cryptology Conference" → "CRYPTO").
    for _pat, _label in _CONF_FULL_NAMES:
        if re.search(_pat, reference, re.I):
            return _label

    # "Advances in Cryptology – CRYPTO 2021" / "Topics in Cryptology – CT-RSA 2020"
    m = re.search(r"\b(?:Advances|Topics)\s+in\s+Cryp-?tology\s*[–\-—]\s*([A-Z][A-Z0-9\-]+)\s+(\d{4})", reference, re.I)
    if m:
        return f"{m.group(1)} {m.group(2)}"

    # "In ACRONYM, volume X of LIPIcs" — Dagstuhl LIPIcs series with acronym before volume
    m = re.search(r"\bIn\s+([A-Z][A-Z0-9&\-]{1,15}),\s+volume\s+\d+\s+of\s+LIPIcs\b", reference, re.I)
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
        m = re.search(r"\bIn:?\s+(?:Pro-?ceedings\s+of\s+|Proc\.?\s+)([^,\.]+)", pre_url, re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"\bIn\s+(?:\d{4}|\d+(?:st|nd|rd|th))\s+([A-Z][A-Za-z0-9 &\-']{4,45})[,\.\s(]", pre_url, re.I)
        if m:
            return m.group(1).strip()
        m = re.search(r"\bIn\s+([A-Z][A-Za-z ]+?)\s+-\s+\d+\w+\s+International\s+Conference", pre_url, re.I)
        if m:
            return m.group(1).strip()
        # "IACR TCHES 2023(3), 164–193" — TCHES DOI refs where vol(issue) blocks the YEAR. pattern
        if re.search(r"\bIACR\s+TCHES\b", pre_url, re.I):
            return "TCHES"
        return ""

    # Alpha-key style: venue appears after editors list — "In EDITORS, editors, CRYPTO 2019"
    # Handles apostrophe-year too: "editors, ASIACRYPT'99" / "editors, EUROCRYPT’96"
    # Two leading uppercase letters required to exclude book titles like "Some Title 2019".
    # Character class and year suffix both accept typographic right-quote (U+2019) from PDFs.
    m = re.search(r"\bed-?itor(?:s)?,\s+(?:\d+(?:st|nd|rd|th)\s+)?([A-Z][A-Z][A-Za-z0-9 &''’\-–]{1,30}?(?:\s+\d{2,4}\b|['’]\d{2}))", reference)
    if m:
        return m.group(1).strip()

    # Ordinal conference where year is not adjacent: "editor, 30th SODA, pages"
    m = re.search(r"\bed-?itor(?:s)?,\s+\d+(?:st|nd|rd|th)\s+([A-Z][A-Za-z0-9&\- ]{1,20})\b", reference)
    if m:
        return m.group(1).strip()

    # Mixed-case conference name: "editors, Theory of Cryptography, pages 3" /
    # "editors, ProvSec 2020, volume 12505". Requires ", pages \d" or ", volume \d"
    # suffix as a structural guard against matching paper titles.
    m = re.search(r"\bed-?itor(?:s)?,\s+([A-Z][A-Za-z0-9 \-–—]{3,50}?),\s*(?:volume\s+\d|pages\s+\d)", reference)
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

    # IEEE/ACM short style: "in S&P 2021," / "In CCS 2018," / "In TCC (2005)" / "In ACM PODC, 2019"
    m = re.search(r"\bin\s+([A-Z][A-Za-z0-9 &'\-]{2,40}),?\s*\(?\d{4}", reference, re.I)
    if m:
        return m.group(1).strip()

    # Standalone venue without adjacent year: "NeurIPS, 2018." / "In SOSP, page"
    m = re.search(r"\b(NeurIPS|SOSP|OSDI)(?:[,\s]+\d{4})?", reference)
    if m:
        return m.group(0).strip().rstrip(',').strip()

    # "In Proceedings of ..." or "In Proc. ..." (with or without colon after In)
    # Pro-?ceedings also handles PDF hyphenation artifact "Pro-ceedings".
    # "of" is optional to catch "In Proceedings 32nd Annual Symposium of Foundations..."
    # Ordinal prefix (e.g. "32nd") is optionally consumed so the capture starts at the name.
    m = re.search(r"\bIn:?\s+(?:Pro-?ceedings(?:\s+of)?\s+(?:\d+\w*\s+)?|Proc\.?\s+)([^,\.]+)", reference, re.I)
    if m:
        return m.group(1).strip()

    # Short acronym style: "In: ITC (2023)" or "In: 24th ACM STOC"
    m = re.search(r"In:\s+(?:\d+(?:st|nd|rd|th)\s+)?([A-Z][A-Z0-9&\- ]{1,30})[,\s]+(?:ACM|IEEE|Springer)?\s*(?:Press\b|,)?\s*\w*\s*\(?\d{4}", reference)
    if m:
        return m.group(1).strip()

    # Abbreviated journal names ending in volume/issue numbers.
    # Each word must start uppercase so the pattern can't span into a paper title.
    # [\s,]+ handles "Commun. ACM, 24(2)" where comma separates journal from volume.
    m = re.search(r"([A-Z][A-Za-z\.]{1,12}(?:\s+[A-Z][A-Za-z\.]{1,12}){0,4})[\s,]+\d+\(\d+\)", reference)
    if m:
        return m.group(1).strip()

    # Journal names with "vol. X, no. Y" notation: "Int. J. Inf. Sec., vol. 14, no. 6, 2015"
    # Each word must start uppercase (same guard as above) to prevent spanning paper titles.
    m = re.search(r"([A-Z][A-Za-z\.]{1,12}(?:\s+[A-Z][A-Za-z\.&]{1,12}){0,4}),?\s+vol\.\s*\d", reference)
    if m:
        return m.group(1).strip()

    # Standalone venue names that appear without an "In" prefix.
    # PoPETS (all-caps) and PoPETs (mixed) both appear in the wild.
    m = re.search(r"\b(PoPETs?|PoPETS|PVLDB|VLDB|PETS|TCHES)\b", reference)
    if m:
        return m.group(1).strip()

    # PoPETs abbreviated as "Proc. Priv. Enh(ancing) Technol." in some citation styles
    if re.search(r"\bProc\.\s+Priv\.\s+Enh", reference, re.I):
        return "PoPETs"

    # Black Hat — security conference cited without "In" prefix
    if re.search(r"\bBlack\s*Hat\b", reference, re.I):
        return "Black Hat"

    if re.search(r"https?:\s*//\s*github\.com", reference, re.I):
        return "GitHub"

    # ePrint / arXiv — ia.cr is the IACR ePrint URL shortener (ia.cr/YYYY/NNN)
    if re.search(r"eprint\.iacr\.org|https?://ia\.cr/|Cryptol(?:ogy)?\.?\s*ePrint|IACR\s+(?:Cryptology\s+)?ePrint", reference, re.I):
        return "ePrint"
    if re.search(r"arxiv\.org|arXiv|\bCoRR\b", reference, re.I):
        return "arXiv"

    # Web/blog/forum references with no venue
    if re.search(r"https?://|(?<!\w)www\.[a-zA-Z]", reference):
        if re.search(r"github\.\s*(?:com|io)|gitlab\.\s*com", reference, re.I):
            return "GitHub"
        if re.search(r"ethresear\.ch|vitalik\.ca|bitcointalk", reference, re.I):
            return "web_forum"
        return "web"

    return ""


# ── Pass 2: DBLP title lookup ─────────────────────────────────────────────────

class _DblpRateLimited(Exception):
    pass


def _fetch_dblp_venue(title: str) -> dict | None:
    """Single DBLP title lookup; returns full info dict on hit, None on miss.
    Raises _DblpRateLimited on HTTP 429 so the caller can bail instead of retrying.
    Always sleeps 1.5s after the HTTP call so rate-limit courtesy is enforced
    only when an actual network request is made (not on cache hits)."""
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
    finally:
        time.sleep(1.5)


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

    _MISS = "__miss__"  # sentinel: variant was queried and got nothing from DBLP

    for i, variant in enumerate(variants):
        # Check cache before hitting DBLP
        if variant in dblp_cache:
            if dblp_cache[variant] == _MISS:
                continue  # confirmed miss for this prefix — try next shorter variant
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
            dblp_cache[variant] = info
            return _DBLP_VENUE_MAP.get(raw_venue.lower(), raw_venue)
        dblp_cache[variant] = _MISS

    return ""


# ── Pass 3: standards and grey-literature fallbacks ───────────────────────────

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
    m = re.search(r'\bNIST\s+IR\s+([\d\-A-Za-z\.]+)', ref, re.I)
    if m:
        return f"NIST IR {m.group(1)}"
    return ""


def match_grey_lit(ref: str) -> str:
    """Post-DBLP fallback for books and technical reports.
    Returns a specific label (e.g. 'Cambridge University Press', 'Tech. Rep. TR-21-05')
    or '' if no pattern fires. source='grey_lit' groups both types in analysis."""
    # Legal case citations: "New York v. Ferber, 458 US 747 (1982)"
    if re.search(r'\bv\.\s+[A-Z][a-z]', ref) and re.search(r'\b\d+\s+(?:U\.?S\.?|F\.\s*\d[a-z]*|[A-Z][a-z]+\.)\s+\d+', ref):
        return "Court Case"

    # IETF Internet-Drafts: "draft-ietf-..." or "Internet-Draft"
    if re.search(r'\bdraft-ietf-\S+|\bInternet[- ]Draft\b', ref, re.I):
        return "IETF Draft"

    # Congressional legislation: "U.S.C.", "Public Law", "H.R.", "S. \d+ (Congress)"
    if re.search(r'\bU\.S\.C\.\s*§|\bPublic\s+Law\s+\d|\bH\.R\.\s*\d|\bS\.\s*\d+\s*\(\d+\w+\s+Cong', ref, re.I):
        return "Legislation"

    # Whitepapers / blog posts explicitly labelled as such
    if re.search(r'\bwhite\s*paper\b', ref, re.I):
        return "Whitepaper"

    # PhD theses
    m = re.search(r'\bPh\.?D\.?\s+[Tt]hesis\b', ref)
    if m:
        return "PhD Thesis"

    # Other thesis types (honors, master's, bachelor's, undergraduate)
    if re.search(r'\b(?:honors?|master(?:\'s)?|bachelor(?:\'s)?|undergraduate)\s+thesis\b', ref, re.I):
        return "Thesis"

    # Competition submissions (crypto-specific: CAESAR, NIST LWC, etc.)
    m = re.search(r'\bSubmission\s+to\s+(?:the\s+)?([A-Z][A-Za-z0-9 \-]+[Cc]ompetition)', ref)
    if m:
        return m.group(1).strip()

    # NIST LWC / PQC submissions and workshops: "Submission to NIST Lightweight Cryptography",
    # "Second Round Submission to NIST", "NIST Lightweight Cryptography Workshop"
    if re.search(r'\b(?:(?:(?:First|Second|Final|Round\s+\d+)\s+)?Submission\s+to\s+NIST'
                 r'|NIST\s+Lightweight\s+Cryptography\s+(?:Round|Workshop))', ref, re.I):
        return "NIST Submission"

    # Technical reports — distinctive enough to check without guards
    m = re.search(r'\bTech(?:nical)?\.?\s+Rep(?:ort)?\.?(?:\s+([\w\-/]+))?', ref, re.I)
    if m:
        num = (m.group(1) or "").strip()
        return f"Tech. Rep. {num}".strip() if num else "Tech. Rep."

    # Books — publisher name present, no "In:" (which signals a conference paper)
    if not re.search(r'\bIn:', ref):
        m = re.search(
            r'\b(Cambridge University Press|MIT Press|Oxford University Press'
            r'|CRC Press|Academic Press'
            r'|Springer(?:\s+(?:Berlin|Verlag|Heidelberg))?'
            r'|Wiley|O\'Reilly|Addison-Wesley|Prentice.?Hall)\b',
            ref, re.I
        )
        if m:
            return m.group(1)
    return ""


# ==============================================================================
# Main pipeline
# For each citation row: FP filter → venue extraction → write CSVs
# ==============================================================================

n_dblp_hits      = 0
n_dblp_misses    = 0
n_standards_hits = 0
n_grey_lit_hits  = 0
dblp_miss_refs: list[str] = []

citation_rows = []

for _, row in df_raw.iterrows():
    title = row["source_paper"]
    awareness = row.get("app_awareness")
    ref = str(row["raw_reference"])

    if not is_likely_real_citation(ref):
        citation_rows.append({
            "source_paper": title,
            "app_awareness": awareness,
            "venue_raw": "",
            "venue_source": "none",
            "raw_reference": ref,
            "suspected_fp": True,
        })
        continue

    venue = extract_venue(ref)
    if venue:
        source = "regex"
    else:
        print(f"  DBLP query ref: {ref[:80]}")
        venue = query_dblp_for_venue(ref)
        if venue:
            source = "dblp"
            n_dblp_hits += 1
        else:
            standards_label = match_standards(ref)
            if standards_label:
                venue = standards_label
                source = "standards"
                n_standards_hits += 1
            else:
                grey_lit_label = match_grey_lit(ref)
                if grey_lit_label:
                    venue = grey_lit_label
                    source = "grey_lit"
                    n_grey_lit_hits += 1
                else:
                    source = "none"
                    n_dblp_misses += 1
                    dblp_miss_refs.append(ref)
        print(f"  DBLP result: {venue or 'none'}")

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
df_real.to_csv(f"csv/{CONFERENCE}_citations_venues.csv", **_csv_opts)
df_fps.to_csv(f"csv/{CONFERENCE}_suspected_fps.csv", **_csv_opts)

print(f"Saved {len(df_real)} citation rows to csv/{CONFERENCE}_citations_venues.csv")
print(f"Saved {len(df_fps)} suspected FPs to csv/{CONFERENCE}_suspected_fps.csv (gitignored — manual audit)")

with open(DBLP_CACHE_FILE, 'w') as _f:
    json.dump(dblp_cache, _f, indent=2)
print(f"Saved {len(dblp_cache)} DBLP hits to {DBLP_CACHE_FILE}")

with open(DBLP_MISSES_FILE, 'w') as _f:
    _f.write(f"# DBLP misses for {CONFERENCE} — {n_dblp_misses} failed queries\n\n")
    for r in dblp_miss_refs:
        _f.write(r + "\n---\n")
print(f"Saved {n_dblp_misses} DBLP misses to {DBLP_MISSES_FILE}")

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
print(f"Citations processed: {total}  ({fp_count} suspected FPs excluded)")
print(f"Venue extracted: {extracted}/{total} ({100*extracted/total:.1f}%)" if total else "No citations processed.")

print("\nExtraction rate by awareness level:")
df_real["extracted"] = df_real["venue_raw"] != ""
print(df_real.groupby("app_awareness")["extracted"].mean().round(3))

# config.py
# Centralized, environment-overridable configuration for the citation-analysis
# pipeline (citation_export.py -> venue_export.py -> venue_match.py ->
# venue_charts.py / venue_by_awareness.py). Every value below can be overridden
# without touching code by exporting the same-named environment variable.
import os
from pathlib import Path

CONFERENCES = ["Crypto", "EuroCrypt", "Oakland", "USENIX"]

# -----------------------------
# Google Sheets (Stage 1)
# -----------------------------
SHEETS_SCOPES = ["https://www.googleapis.com/auth/spreadsheets.readonly"]
SPREADSHEET_ID = os.environ.get("SPREADSHEET_ID", "1I2eZyK7PIhXEMwy30w8BgEcuRrLQQw4wK6GlxfAsuWE")
CREDENTIALS_FILE = Path(os.environ.get("CREDENTIALS_FILE", "credentials.json"))
TOKEN_FILE = Path(os.environ.get("TOKEN_FILE", "token.pickle"))

# -----------------------------
# Local paths
# -----------------------------
ZOTERO_STORAGE = Path(os.environ.get("ZOTERO_STORAGE", "/Users/nathanielclizbe/Zotero/storage/"))

CSV_DIR = Path(os.environ.get("CSV_DIR", "csv"))
JSON_DIR = Path(os.environ.get("JSON_DIR", "json"))
LOGS_DIR = Path(os.environ.get("LOGS_DIR", "logs"))
TEXT_DIR = Path(os.environ.get("TEXT_DIR", "text"))
CHARTS_DIR = Path(os.environ.get("CHARTS_DIR", "charts"))

DBLP_LABELS_FILE = Path(os.environ.get("DBLP_LABELS_FILE", "dblp-labels.csv"))

# -----------------------------
# Pipeline tuning
# -----------------------------
DBLP_QUERY_DELAY_SECONDS = float(os.environ.get("DBLP_QUERY_DELAY_SECONDS", "1.5"))
# A DBLP "miss" (200 response, empty/no-venue hits) isn't reliable on a single
# query — DBLP has been observed silently returning empty results under load
# with no error signal at all, distinct from the request-level failures
# (timeouts, 429s, dropped connections) that are never cached in the first
# place. A variant is only treated as a confirmed, cache-skippable miss once
# it has missed this many times across independent runs.
DBLP_MISS_CONFIRM_THRESHOLD = int(os.environ.get("DBLP_MISS_CONFIRM_THRESHOLD", "3"))
FUZZY_MATCH_CUTOFF = int(os.environ.get("FUZZY_MATCH_CUTOFF", "85"))
CHART_TOP_N = int(os.environ.get("CHART_TOP_N", "15"))
AWARENESS_CHART_TOP_N = int(os.environ.get("AWARENESS_CHART_TOP_N", "12"))


# -----------------------------
# Per-conference path helpers
# -----------------------------
def citations_raw_csv(conference: str) -> Path:
    return CSV_DIR / f"{conference}_citations_raw.csv"


def citations_venues_csv(conference: str) -> Path:
    return CSV_DIR / f"{conference}_citations_venues.csv"


def suspected_fps_csv(conference: str) -> Path:
    return CSV_DIR / f"{conference}_suspected_fps.csv"


def citations_matched_csv(conference: str) -> Path:
    return CSV_DIR / f"{conference}_citations_matched.csv"


def dblp_cache_json(conference: str) -> Path:
    return JSON_DIR / f"{conference}_dblp_cache.json"


def dblp_misses_txt(conference: str) -> Path:
    return LOGS_DIR / f"{conference}_dblp_misses.txt"


def paper_text_dir(conference: str) -> Path:
    return TEXT_DIR / conference

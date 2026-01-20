**`ref_analysis.py`** extracts and analyzes references from PDFs stored in a local Zotero library: it filters papers by title, extracts the References section, parses individual citations, and classifies them (crypto, security, news, policy/gov, technical docs, other).
**Requirements:** Python 3.10+, Zotero with local storage enabled, and PyMuPDF (`pip install pymupdf`).
**Usage:** add target paper titles (one per line) to `papers_list.txt`, set `ZOTERO_STORAGE` to your Zotero storage path, then run `python ref_analysis.py` to see per-paper reference category counts printed to stdout.


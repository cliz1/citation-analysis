The **`ref_analysis.py`** script extracts references from PDFs stored in a local Zotero library. It filters papers by title (using `papers_list.txt`), extracts the References section, parses individual citations, and sorts them into the following buckets (crypto, security, news, policy/gov, technical docs, other).

**Requirements:** Zotero with local storage enabled, and PyMuPDF (`pip install pymupdf`).

**Usage:** add target paper titles (one per line) to `papers_list.txt`, set `ZOTERO_STORAGE` to your Zotero storage path. You can find this by right clicking a paper in Zotero and clicking "show in finder." Then run the script - currently, it just prints the data by title to stdout. 


# Reference Analysis Pipeline

**Author:** Nathaniel Clizbe
**Date:** January 2026

## Overview

`ref_analysis.py` is a data pipeline for analyzing citation distributions in academic papers.

The script:

1. Pulls coded paper metadata from Google Sheets
2. Matches those papers to local PDFs stored in Zotero
3. Extracts and parses the References section of each paper
4. Classifies each citation into predefined buckets
5. Computes normalized citation distributions
6. Aggregates by “application awareness” level
7. Produces a stacked bar chart of average citation share

The primary goal is to study how citation composition varies by application awareness level.

---

## High-Level Workflow

### 1. Load Paper Metadata

* Connects to Google Sheets (read-only scope)
* Reads a specific sheet (e.g., `"Crypto"`)
* Filters rows by coded category (e.g., Crypto or Analysis)
* Collects:

  * Cleaned paper titles
  * Application awareness score (1–4)

---

### 2. Match Papers to Zotero PDFs

* Recursively scans a local Zotero storage directory
* Normalizes filenames and titles
* Matches PDFs to titles using:

  * Token overlap threshold (≥70%)
  * Long-word requirement (≥7 characters)
  * Stopword filtering

If multiple PDFs match a title, the largest file is selected.

---

### 3. Extract and Parse References

Using PyMuPDF (`fitz`):

* Extract full PDF text
* Locate the References section
* Stop parsing at:

  * Appendix
  * Acknowledgements
  * Supplementary sections (best effort)

Handles citation formats such as:

* `1. ...`
* `[1] ...`
* `[ANWW13] ...`
* `[BCG+19a] ...`

Includes heuristics to avoid parsing:

* Proof text
* Theorem statements
* Body prose

---

### 4. Classify Citations

Each reference is scored against keyword sets and assigned to the highest-scoring category:

* crypto
* security
* news
* policy_gov
* technical_doc
* source_code
* vendor_doc
* industry_blog
* systems
* IoT_networks
* economics
* other (catch-all)

Classification is keyword-based and additive.

---

### 5. Normalize and Aggregate

For each paper:

* Count references per bucket
* Normalize to fraction of total citations
* Attach application awareness score

Then:

* Group by awareness level
* Compute mean citation share per bucket

---

### 6. Visualization

Produces a stacked bar chart:

* X-axis: Application Awareness Level
* Y-axis: Average Citation Share
* Bars: Stacked by citation category

---

## Configuration

Edit these values at the top of the script:

```python
ZOTERO_STORAGE = Path("/Users/.../Zotero/storage/")
SPREADSHEET_ID = "..."
TEST_RANGE = "Crypto"
SAMPLE_SIZE = 400
```

---

## Dependencies

* PyMuPDF (`fitz`)
* pandas
* matplotlib
* google-api-python-client
* google-auth
* google-auth-oauthlib

Install via pip:

```
pip install pymupdf pandas matplotlib google-api-python-client google-auth google-auth-oauthlib
```

You will also need:

* `credentials.json` for Google API access
* OAuth token will be cached as `token.pickle`

---

## Known Design Tradeoffs

* Keyword classification is heuristic and not semantic
* Reference parsing is regex-driven and imperfect
* PDF text extraction quality varies by formatting
* Matching titles to PDFs is approximate

---

# ERROR DOC

## 1: Missing final reference when followed by an appendix

* summary: parsing complication that accounts for at most 1 missing citation per paper,
  if the paper has a post-bibliography appendix

* details: when a paper has an appendix after its references section,
  the parser will include the appendix as part of the final citation,
  and then weed it out because it looks like body text. This can be
  mitigated, but is difficult to fully resolve, since there is no
  agreed pattern for how these appendices begin.

* example: Leakage-Abuse Attacks Against Structured Encryption for SQL. (USENIX)
  for this paper, the final citation is
  `[49] Zheguang Zhao, Seny Kamara, Tarik Moataz, and Stan Zdonik. Encrypted databases: From theory to systems. In CIDR, 2021.`
  But the parser only sees up to citation 48. This is because after the last citation, the paper includes:

  ```
  "A Column Equality Theory  This section contains theorems 
  and proofs omitted from the main text for brevity. 
  We recall an attack from prior work in Figure 10. 
  The following definition and lemma are useful in proofs below..."
  ```

  the parser doesn't have a good way to detect the end of the references section,
  so this part gets attached to final reference and weeded out because it
  sees clear indicators that this is not a citation (i.e., the word 'lemma').

  my previous attempts at detecting the end of the references section have all resulted
  in detecting the end too early, so right now, this missing citation in this edge case
  is a sacrifice to keep most of the citations in all cases.

---

## 2: Proofs or body text getting parsed as references

* summary: parsing bug accounting for at most 5 non-citations appearing in the data,
  though almost always falling into the "other" catch-all bucket.

* details: proof lines start with numbers like "1.". So do citations in certain conferences.
  I'm looking at you, Crypto. My parser uses keywords like "hence" or "lemma" to weed these
  out but sometimes they slip through. However, due to these being math/proof lines, we can
  almost always assume these will not contain keywords that will put them in buckets other than
  the catch-all.

* example: in *Formal verification of the PQXDH Post-Quantum key agreement protocol for end-to-end secure messaging*, the following is included by the parser as a citation:

  ```
  "1. the exhaustive case disjunction under which the key computed by the responder Alex is secret;"
  ```

  I am currently experimenting with the idea of only grabbing text if it is below the "references"
  marker, but for now this issue is present in the code.

---

## 3: Papers from Google Docs not showing up in results

* summary: class of data entry errors that result in ~5 papers per conference not being included in the data at all

* details: sometimes a coded paper doesn't get saved to Zotero as a PDF, or sometimes it doesn't
  get saved to Zotero at all. From observation, it sometimes seems intentional and sometimes not.

* example:
  *Secure Account Recovery for a Privacy-Preserving Web Service.* This paper is not in Zotero,
  but it was coded. However, I recall this one might have been intentionally left out of the Zotero pool? Unsure.

  *Marco Palazzo et al.: Privacy-Preserving Data Aggregation with Public Verifiability Against Internal Adversaries:*
  also coded but not in Zotero.

---

If you'd like, I can also produce a shorter “publication-facing” version of this README and keep this one as a technical/internal version.

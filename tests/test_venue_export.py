"""
Regression tests for venue_export.py's regex venue extraction (Pass 1) and
standards/grey-lit fallback (Pass 3). Each test pins down a real input that
previously broke before a specific fix this week, so the bug can't silently
come back.

venue_export.py runs argparse parsing and file I/O at the top of the module
(see CONFERENCE = _parser.parse_args().conference), so importing it directly
under pytest would try to parse pytest's own CLI args and load conference
CSVs. Instead this file loads just the function definitions via source
slicing -- the same technique used by hand to verify each fix during
development (diffing extract_venue() output across the full corpus before
and after each change). If venue_export.py is ever refactored to guard its
top-level work behind `if __name__ == "__main__":`, this loader can be
replaced with a plain import.
"""
import re
from pathlib import Path

_VENUE_EXPORT_PATH = Path(__file__).parent.parent / "venue_export.py"


def _load_functions():
    src = _VENUE_EXPORT_PATH.read_text()
    start = src.index("_DBLP_VENUE_MAP = {")
    end = src.index("# ==============================================================================\n# Main pipeline")
    ns = {"re": re}
    exec(compile(src[start:end], str(_VENUE_EXPORT_PATH), "exec"), ns)
    return ns["extract_venue"], ns["match_standards"], ns["match_grey_lit"]


extract_venue, match_standards, match_grey_lit = _load_functions()


# ── Fix: a blanket re.I on the ". Publisher" pattern let mixed-case words
#    like "Proc"/"Proceedings" defeat the all-caps acronym guard ──────────

def test_proc_does_not_match_publisher_pattern():
    ref = ("[AGS11] Carlos Aguilar, Philippe Gaborit, and Julien Schrek. A new "
           "zero-knowledge code based identification scheme with reduced "
           "communication. In Proc. IEEE Inf. Theory Workshop-ITW 2011, pages 648-65")
    assert extract_venue(ref) != "Proc"


def test_allcaps_acronym_before_publisher_still_matches():
    ref = "[78] Rafail Ostrovsky. Efficient computation on oblivious RAMs. In STOC. ACM, 1990."
    assert extract_venue(ref) == "STOC"


def test_mixed_case_venue_before_publisher_still_matches():
    # "EuroSys" is a real venue name (not a generic structural word like
    # "Proc") and relies on case-insensitivity here, so it must still match.
    ref = ('[27] G. Danezis, L. Kokoris-Kogias, A. Sonnino, and A. Spiegelman, '
           '"Narwhal and Tusk: A DAG-based mempool," in EuroSys. ACM, 2022,')
    assert extract_venue(ref) == "EuroSys"


# ── Fix: connector-word guard against journal-name truncation, e.g.
#    "Mathematics of Computation" -> "Computation" ─────────────────────────

def test_journal_name_truncation_is_deferred_not_returned_wrong():
    ref = "[50] Neal Koblitz. Elliptic Curve Cryptosystems. Mathematics of Computation, 48(177):203-209, 1987."
    assert extract_venue(ref) == ""  # deferred to DBLP, not the wrong fragment "Computation"


def test_acm_publisher_name_not_returned_as_journal_venue():
    ref = "[36] Adi Shamir. How to share a secret. Communications of the ACM, 22(11), 1979."
    assert extract_venue(ref) == ""  # "Communications of the ACM" is the journal; "ACM" alone is wrong


def test_in_acronym_vol_style_not_caught_by_connector_guard():
    # "in" is deliberately excluded from the connector denylist: "In FOCS,
    # vol. 82" uses "in" as the citation's own marker, not a truncation signal.
    ref = '[54] A. C.-C. Yao, "Protocols for secure computations," in FOCS, vol. 82, 1982,'
    assert extract_venue(ref) == "FOCS"


def test_in_ndss_vol_style_not_caught_by_connector_guard():
    ref = ('[71] J. Newsome and D. X. Song, "Dynamic taint analysis for automatic '
           'detection," in NDSS, vol. 1, 2005,')
    assert extract_venue(ref) == "NDSS"


# ── Fix: "web" catch-all swallowed standards docs (RFC/NIST/FIPS/ISO) that
#    carry a URL but no recognized venue text. The actual fix reorders calls
#    in venue_export.py's per-row main-loop dispatch, which is inline script
#    code rather than a standalone function, so it isn't unit-testable here.
#    These two tests instead pin the building blocks the fix depends on. ───

def test_rfc_citation_falls_back_to_web_in_pass1():
    ref = ('[45] M. Jones, J. Bradley, and N. Sakimura. JSON web token (JWT). '
           'https://datatracker.ietf.org/doc/html/rfc7519, 2015. RFC 7519.')
    assert extract_venue(ref) == "web"


def test_match_standards_resolves_the_same_rfc_citation():
    ref = ('[45] M. Jones, J. Bradley, and N. Sakimura. JSON web token (JWT). '
           'https://datatracker.ietf.org/doc/html/rfc7519, 2015. RFC 7519.')
    assert match_standards(ref) == "RFC 7519"


# ── Fix: short-surname authors (J. Xu, J. Wu, J. Li, etc.) slipped past the
#    J.-initial guard because [a-z]{2,} required 2+ lowercase chars; 2-letter
#    surnames like Xu/Wu/Li only have one. Guard changed to [a-z]+.
#    "J. Am." legitimately starts journal names (J. Am. Stat. Assoc. etc.)
#    and is now protected via the bypass allowlist. ─────────────────────────

def test_short_surname_author_not_returned_as_venue():
    # "J. Xu" is the last author in the list; Opaque is the paper title.
    ref = ('[43] S. Jarecki, H. Krawczyk, and J. Xu. Opaque: An asymmetric PAKE '
           'protocol secure against pre-computation attacks. In Proceedings of the '
           'International Conference on the Theory and Applications of Cryptographic '
           'Techniques (EUROCRYPT), 2018.')
    assert extract_venue(ref) != "J. Xu. Opaque"


def test_j_am_journal_still_returned_after_short_surname_fix():
    # "J. Am." is a journal abbreviation prefix (J. Am. Stat. Assoc., etc.)
    # and must not be blocked by the tightened surname guard.
    ref = '14. Freedman, D.: A remark. J. Am. Stat. Assoc. 72(359), 1977.'
    result = extract_venue(ref)
    assert result is not None and result.startswith("J. Am.")

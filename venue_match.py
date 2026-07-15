import sys
import csv
import html
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
import config

ABBREV_MAP = {
    "CCS": "Annual ACM Conference on Computer and Communications Security (CCS)",
    "SP": "IEEE Symposium on Security and Privacy (SP)",
    "CRYPTO": "Annual International Cryptology Conference (CRYPTO)",
    "EUROCRYPT": "International Conference on the Theory and Application of Cryptographic Techniques (EUROCRYPT)",
    "ASIACRYPT": "International Conference on the Theory and Application of Cryptology and Information Security (ASIACRYPT)",
    "USENIX Security Symposium": "USENIX Security Symposium",
    "NDSS": "Network and Distributed System Security Symposium (NDSS)",
    "TCC": "Theory of Cryptography Conference (TCC)",
    "PKC": "International Conference on Theory and Practice of Public Key Cryptography (PKC)",
    "ePrint": "ePrint",
    "arXiv": "arXiv",
    "GitHub": "GitHub",
    "web": "web",
    "web_forum": "web_forum",

        # Theory venues
    "FOCS": "IEEE Annual Symposium on Foundations of Computer Science (FOCS)",
    "STOC": "Symposium on the Theory of Computing (STOC)",
    "ICALP": "International Colloquium on Automata, Languages and Programming (ICALP)",
    "PODC": "ACM Symposium on Principles of Distributed Computing",
    "OPODIS": "International Conference on Principles of Distributed Systems",
    "Theory Comput.": "Theory of Computing",

    # Journals
    "J. ACM": "Journal of the ACM",
    "J. Cryptol.": "Journal of Cryptology",
    "J. Cryptology": "Journal of Cryptology",
    "J. Comput.": "SIAM Journal on Computing",
    "SIAM J. Comput.": "SIAM Journal on Computing",
    "J. Comput. Syst. Sci.": "Journal of Computer and System Sciences",
    "J. Theor. Comput. Sci.": "Theoretical Computer Science",
    "J. Math. Cryptol.": "Journal of Mathematical Cryptology",
    "J. Appl. Cryptogr.": "Journal of Applied Cryptography",
    "J. Autom. Reason.": "Journal of Automated Reasoning",
    "IEEE Trans. Inf. Theory": "IEEE Transactions on Information Theory",
    "Rand. Struct. Algor.": "Random Structures and Algorithms",
    "Acta Inf.": "Acta Informatica",

    # Crypto venues missing
    "CT-RSA": "The Cryptographer's Track at RSA Conference (CT-RSA)",
    "PQCrypto": "Post-Quantum Cryptography (PQCrypto)",
    "USENIX Security": "USENIX Security Symposium",

    # Crypto workshop venues
"FSE": "Fast Software Encryption Workshop (FSE)",
"CHES": "Conference on Cryptographic Hardware and Embedded Systems (CHES)",
"SAC": "Selected Areas in Cryptography (SAC)",
"ACNS": "International Conference on Applied Cryptography and Network Security (ACNS)",
"INDOCRYPT": "International Conference on Cryptology in India (INDOCRYPT)",
"AFRICACRYPT": "International Conference on Cryptology in Africa (AFRICACRYPT)",
"LATINCRYPT": "International Conference on Cryptology and Information Security in Latin America (LATINCRYPT)",
"ACISP": "Australasian Conference on Information Security and Privacy (ACISP)",
"ICISC": "International Conference on Information Security and Cryptology (ICISC)",
"CANS": "Cryptology and Network Security (CANS)",
"IWSEC": "International Workshop on Security (IWSEC)",
"SCN": "International Conference on Security and Cryptography for Networks (SCN)",
"WISA": "International Conference on Information Security Applications (WISA)",
"FC": "Financial Cryptography and Data Security (FC)",
"SECRYPT": "International Conference on Security and Cryptography (SECRYPT)",
"ITCS": "Innovations in Theoretical Computer Science (ITCS)",
"STACS": "Symposium on Theoretical Aspects of Computer Science (STACS)",
"CaLC": "Cryptography and Lattices (CaLC)",
"SCC": "International Workshop on Signal Design and Its Applications in Communications (IWSDA)",
"NSS": "International Conference on Network and System Security (NSS)",
"CSCML": "International Conference on Cyber Security Cryptography and Machine Learning (CSCML)",
"WAIFI": "International Workshop on Arithmetic of Finite Fields (WAIFI)",
"TQC": "Theory of Quantum Computation, Communication, and Cryptography (TQC)",
"PODC": "ACM Symposium on Principles of Distributed Computing (PODC)",

# ACM CCS variants
"ACM CCS": "Annual ACM Conference on Computer and Communications Security (CCS)",

# Trans. Symmetric Cryptol. variants
"Trans. Symmetric Cryptol.": "IACR Transactions on Symmetric Cryptology (ToSC)",
"Trans. Symm. Cryptol.": "IACR Transactions on Symmetric Cryptology (ToSC)",
"Trans. Sym": "IACR Transactions on Symmetric Cryptology (ToSC)",
"Trans. Cryptogr. Hardw. Embed. Syst.": "IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)",

# Journals
"Des. Codes Crypt.": "Designs, Codes and Cryptography",
"Des. Codes Cryptogr.": "Designs, Codes and Cryptography",
"J. Symb. Comput.": "Journal of Symbolic Computation",
"J. Symbolic Comput.": "Journal of Symbolic Computation",
"Theoret. Comput. Sci.": "Theoretical Computer Science",
"ACM Trans. Comput. Theory": "ACM Transactions on Computation Theory",
"Proc. Priv. Enhancing Technol.": "Proceedings on Privacy Enhancing Technologies (PoPETs)",
"Phys. Rev. Lett.": "Physical Review Letters",
"Phys. Rev. A": "Physical Review A",
"Duke Math. J.": "Duke Mathematical Journal",
"Ann. Probab.": "Annals of Probability",
"Math. Program.": "Mathematical Programming",
"Math. Notes": "Mathematical Notes",
"J. Algorithms": "Journal of Algorithms",
"J. Pure Appl. Algebra": "Journal of Pure and Applied Algebra",

# Verbose variants that should map to existing entries
"Advances in Cryptology - EUROCRYPT": "International Conference on the Theory and Application of Cryptographic Techniques (EUROCRYPT)",
"Information Security and Cryptology - ICISC": "International Conference on Information Security and Cryptology (ICISC)",

# Venues you're missing
"CCC": "Computational Complexity Conference (CCC)",
"ESA": "European Symposium on Algorithms (ESA)",
"ISSAC": "International Symposium on Symbolic and Algebraic Computation (ISSAC)",
"LATIN": "Latin American Symposium on Theoretical Informatics (LATIN)",
"ESORICS": "European Symposium on Research in Computer Security (ESORICS)",
"ICICS": "International Conference on Information, Communications and Signal Processing (ICICS)",
"NordSec": "Nordic Conference on Secure IT Systems (NordSec)",
"FC": "Financial Cryptography and Data Security (FC)",
"SEC": "IFIP International Information Security Conference (SEC)",
"Inscrypt": "International Conference on Information Security and Cryptology (ICISC)",
"AES 2004": "International Conference on Advanced Encryption Standard (AES)",

# IEEE Trans variants - add to normalize_venue strip logic
"IEEE Trans. Inform. Theory": "IEEE Transactions on Information Theory",
"IEEE Trans. Inf. Theor.": "IEEE Transactions on Information Theory",
"IEEE Trans. Comput.": "IEEE Transactions on Computers",
"IEEE Trans. Emerg. Top. Comput.": "IEEE Transactions on Emerging Topics in Computing",

# Journals
"J. Complex.": "Journal of Complexity",
"J. Algebra": "Journal of Algebra",
"Commun. Algebra": "Communications in Algebra",
"Compos. Math.": "Compositio Mathematica",
"Distrib. Comput.": "Distributed Computing",
"ACM Trans. Program. Lang. Syst.": "ACM Transactions on Programming Languages and Systems",
"Linear Multilinear Algebra": "Linear and Multilinear Algebra",

# Verbose CRYPTO/EUROCRYPT/PKC variants
"Advances in Cryptology - CRYPTO": "Annual International Cryptology Conference (CRYPTO)",
"Public-Key Cryptography - PKC": "International Conference on Theory and Practice of Public Key Cryptography (PKC)",
"Cryptology - EUROCRYPT": "International Conference on the Theory and Application of Cryptographic Techniques (EUROCRYPT)",

    # S&P / SP variants
    "S&P": "IEEE Symposium on Security and Privacy (SP)",
    "IEEE S&P": "IEEE Symposium on Security and Privacy (SP)",
    "EuroS&P": "European Symposium on Security and Privacy (EuroS&P)",

    # Systems venues
    "SOSP": "Symposium on Operating Systems Principles (SOSP)",
    "OSDI": "USENIX Symposium on Operating Systems Design and Implementation (OSDI)",
    "NSDI": "Symposium on Networked Systems Design and Implementation (NSDI)",
    "EuroSys": "European Conference on Computer Systems (EuroSys)",
    "ASPLOS": "International Conference on Architectural Support for Programming Languages and Operating Systems (ASPLOS)",
    "ICDCS": "IEEE International Conference on Distributed Computing Systems (ICDCS)",

    # ML / data venues
    "ICML": "International Conference on Machine Learning and Computing (ICMLC)",
    "NeurIPS": "Conference on Neural Information Processing Systems (NeurIPS)",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining (KDD)",
    "WWW": "The Web Conference (WWW)",
    "RecSys": "ACM Conference on Recommender Systems (RecSys)",

    # Security venues
    "CSF": "IEEE Computer Security Foundations Symposium (CSF)",
    "IEEE CNS": "IEEE Conference on Communications and Network Security (CNS)",
    "Financial Cryptography": "Financial Cryptography and Data Security (FC)",
    "Proceedings of USENIX Security": "USENIX Security Symposium",

    # Theory venues
    "SODA": "ACM-SIAM Symposium on Discrete Algorithms (SODA)",
    "ITC": "Conference on Information-Theoretic Cryptography (ITC)",
    "ISIT": "International Symposium on Information Theory (ISIT)",
    "ACM STOC": "Symposium on the Theory of Computing (STOC)",
    "Electron. Colloquium Comput. Complex.": "Electronic Colloquium on Computational Complexity",

    # Journals
    "IACR Trans. Symmetric Cryptol.": "IACR Transactions on Symmetric Cryptology (ToSC)",
    "TCHES": "IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)",
    "Theor. Comput. Sci.": "Theoretical Computer Science",
    "Comput. Complex.": "Computational Complexity",
    "Commun. ACM": "Communications of the ACM",
    "Inf. Process. Lett.": "Information Processing Letters",
    "IEEE Trans. Inf. Forensics Secur.": "IEEE Transactions on Information Forensics and Security",
    "IEEE Trans. Computers": "IEEE Transactions on Computers",

    # Verbose / split-column variants
    "Public Key Cryptography": "International Conference on Theory and Practice of Public Key Cryptography (PKC)",
    "Proceedings of the ACM CCS": "ACM Conference on Computer and Communications Security (CCS)",
    "EURO-CRYPT": "International Conference on the Theory and Application of Cryptographic Techniques (EUROCRYPT)",
    "ASI-ACRYPT": "International Conference on the Theory and Application of Cryptology and Information Security (ASIACRYPT)",

    # RFCs are IETF documents, not conference papers
    "RFC": "web",

    # Unfixable
    "August": "",  # month name leaking in, discard
    "Adv. Comput. Res.": "",  # obscure, leave unmatched
    "Providing Sound Foundations for Cryptography": "",  # book title, not a venue

    # Privacy venues
    "PoPETs": "Proceedings on Privacy Enhancing Technologies (PoPETs)",
    "PoPETS": "Proceedings on Privacy Enhancing Technologies (PoPETs)",
    "PETS": "Proceedings on Privacy Enhancing Technologies (PoPETs)",

    # Security venues
    "AsiaCCS": "ACM Asia Conference on Computer and Communications Security (AsiaCCS)",
    "ACM AsiaCCS": "ACM Asia Conference on Computer and Communications Security (AsiaCCS)",
    "SIGMOD": "ACM SIGMOD Conference (SIGMOD)",
    "ACSAC": "Annual Computer Security Applications Conference (ACSAC)",
    "WPES@CCS": "Workshop on Privacy in the Electronic Society (WPES)",

    # ML / PL / hardware venues
    "ICLR": "International Conference on Learning Representations (ICLR)",
    "PLDI": "ACM-SIGPLAN Symposium on Programming Language Design and Implementation (PLDI)",
    "CAV": "International Conference on Computer Aided Verification (CAV)",
    "DATE": "Design, Automation and Test in Europe (DATE)",
    "DAC": "Design Automation Conference (DAC)",

    # Distributed computing variants
    "ACM PODC": "ACM Symposium on Principles of Distributed Computing (PODC)",

    # Journal variants
    "IEEE Trans. Dependable Secur. Comput.": "IEEE Transactions on Dependable and Secure Computing",
    "IACR TCHES": "IACR Transactions on Cryptographic Hardware and Embedded Systems (TCHES)",
    "IACR CRYPTO": "Annual International Cryptology Conference (CRYPTO)",
    "IACR Eurocrypt": "International Conference on the Theory and Application of Cryptographic Techniques (EUROCRYPT)",
    "Commun. Assoc. Comput. Mach.": "Communications of the ACM",

    # Hyphen OCR artifact
    "IEEE Sym-posium on Security and Privacy": "IEEE Symposium on Security and Privacy (SP)",

    # Publishers / non-venues — discard
    "Springer": "",
    "Springer Berlin": "",
    "LNCS": "",
    "Tech. Rep.": "",
    "NIST Submission": "",
    "Cambridge University Press": "",
    "Annual Symposium on": "",
    "Annual ACM Symposium on": "",
    "PhD Thesis": "",
    "Ed.": "",
    "ACM": "",
}

def normalize_venue(venue_raw: str) -> str:
    raw = venue_raw.strip()

    # Check map on raw value first (catches short-but-valid keys like "SP")
    if raw in ABBREV_MAP:
        return ABBREV_MAP[raw]

    # Unescape HTML entities (e.g. EuroS&amp;P → EuroS&P)
    normalized = html.unescape(raw)
    if normalized in ABBREV_MAP:
        return ABBREV_MAP[normalized]

    # Strip trailing year with optional letter
    normalized = re.sub(r'\s+\d{4}[a-z]?$', '', normalized)
    # Strip trailing volume/issue numbers
    normalized = re.sub(r'\s+\d+(\(\d+\))?$', '', normalized)
    # Fix hyphenation artifacts (hyphen followed by whitespace, or mid-word hyphen before lowercase)
    normalized = re.sub(r'-\s+', '', normalized)
    normalized = re.sub(r'([A-Za-z])-([a-z])', r'\1\2', normalized)
    # Collapse internal spaces after hyphen removal
    normalized = re.sub(r'\s{2,}', ' ', normalized)
    # Strip "Advances in Cryptology - " prefix (handles both hyphen and em-dash)
    normalized = re.sub(r'^Advances in Cryptology\s*[-–]\s*', '', normalized)
    # Strip "Information Security and Cryptology - " prefix
    normalized = re.sub(r'^Information Security and Cryptology\s*-\s*', '', normalized)
    # Handle truncated entries - too short to be useful
    if len(normalized) < 3:
        return ""
    return ABBREV_MAP.get(normalized, ABBREV_MAP.get(raw, ""))

def main():
    if len(sys.argv) < 2:
        print("Usage: python venue_match.py <input_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace("_citations_venues.csv", "_citations_matched.csv")

    dblp_venues = []
    with open(config.DBLP_LABELS_FILE, 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # skip header
        for row in reader:
            dblp_venues.append(row[0])

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader((line.replace('\x00', '') for line in infile), escapechar='\\')
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header + ["venue_matched", "match_score"])

        count = 0
        for row in reader:
            count+=1
            if count % 100 == 0:
                print(f"Processed {count} rows...", file=sys.stderr)
            venue_raw = row[2]  # venue_raw column

            # Skip empty venues
            if not venue_raw:
                writer.writerow(row + ["", "0"])
                continue

            # Normalize: strip year, look up in abbrev map
            normalized = normalize_venue(venue_raw)

            if normalized:
                writer.writerow(row + [normalized, "100"])
                continue

            # Special cases that won't match DBLP
            if venue_raw in ("ePrint", "arXiv", "GitHub", "web", "web_forum"):
                writer.writerow(row + [venue_raw, "100"])
                continue

            # Fuzzy match
            best_match, score = process.extractOne(
                venue_raw, dblp_venues, scorer=fuzz.token_sort_ratio
            )
            if score >= config.FUZZY_MATCH_CUTOFF:
                writer.writerow(row + [best_match, str(score)])
            else:
                writer.writerow(row + ["", str(score)])

if __name__ == "__main__":
    main()
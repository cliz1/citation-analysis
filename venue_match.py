import sys
import csv
import html
from fuzzywuzzy import fuzz
from fuzzywuzzy import process
import re
 
ABBREV_MAP = {
    "CCS": "ACM Conference on Computer and Communications Security",
    "SP": "IEEE Symposium on Security and Privacy",
    "CRYPTO": "International Cryptology Conference",
    "EUROCRYPT": "Annual International Conference on the Theory and Applications of Cryptographic Techniques",
    "ASIACRYPT": "International Conference on the Theory and Application of Cryptology and Information Security",
    "USENIX Security Symposium": "USENIX Security Symposium",
    "NDSS": "Network and Distributed System Security Symposium",
    "TCC": "Theory of Cryptography Conference",
    "PKC": "International Conference on Practice and Theory of Public Key Cryptography",
    "ePrint": "ePrint",
    "arXiv": "arXiv",
    "GitHub": "GitHub",
    "web": "web",
    "web_forum": "web_forum",

        # Theory venues
    "FOCS": "IEEE Annual Symposium on Foundations of Computer Science",
    "STOC": "ACM Symposium on Theory of Computing",
    "ICALP": "International Colloquium on Automata, Languages and Programming",
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
    "CT-RSA": "RSA Conference, Cryptographers Track",
    "PQCrypto": "Post-Quantum Cryptography",
    "USENIX Security": "USENIX Security Symposium",

    # Crypto workshop venues
"FSE": "Fast Software Encryption",
"CHES": "Cryptographic Hardware and Embedded Systems",
"SAC": "Selected Areas in Cryptography",
"ACNS": "Applied Cryptography and Network Security",
"INDOCRYPT": "International Conference on Cryptology in India",
"AFRICACRYPT": "International Conference on Cryptology in Africa",
"LATINCRYPT": "International Conference on Cryptology and Information Security in Latin America",
"ACISP": "Australasian Conference on Information Security and Privacy",
"ICISC": "International Conference on Information Security and Cryptology",
"CANS": "Cryptology and Network Security",
"IWSEC": "International Workshop on Security",
"SCN": "Security and Cryptography for Networks",
"WISA": "International Workshop on Information Security Applications",
"FC": "Financial Cryptography and Data Security",
"SECRYPT": "International Conference on Security and Cryptography",
"ITCS": "Innovations in Theoretical Computer Science",
"STACS": "Symposium on Theoretical Aspects of Computer Science",
"CaLC": "Cryptography and Lattices Conference",
"SCC": "International Workshop on Signal Design and its Applications in Communications",
"NSS": "Network and System Security",
"CSCML": "International Symposium on Cyber Security Cryptography and Machine Learning",
"WAIFI": "International Workshop on Arithmetic of Finite Fields",
"TQC": "Conference on the Theory of Quantum Computation",
"PODC": "ACM Symposium on Principles of Distributed Computing",

# ACM CCS variants
"ACM CCS": "ACM Conference on Computer and Communications Security",

# Trans. Symmetric Cryptol. variants
"Trans. Symmetric Cryptol.": "IACR Transactions on Symmetric Cryptology",
"Trans. Symm. Cryptol.": "IACR Transactions on Symmetric Cryptology",
"Trans. Sym": "IACR Transactions on Symmetric Cryptology",
"Trans. Cryptogr. Hardw. Embed. Syst.": "IACR Transactions on Cryptographic Hardware and Embedded Systems",

# Journals
"Des. Codes Crypt.": "Designs, Codes and Cryptography",
"Des. Codes Cryptogr.": "Designs, Codes and Cryptography",
"J. Symb. Comput.": "Journal of Symbolic Computation",
"J. Symbolic Comput.": "Journal of Symbolic Computation",
"Theoret. Comput. Sci.": "Theoretical Computer Science",
"ACM Trans. Comput. Theory": "ACM Transactions on Computation Theory",
"Proc. Priv. Enhancing Technol.": "Proceedings on Privacy Enhancing Technologies",
"Phys. Rev. Lett.": "Physical Review Letters",
"Phys. Rev. A": "Physical Review A",
"Duke Math. J.": "Duke Mathematical Journal",
"Ann. Probab.": "Annals of Probability",
"Math. Program.": "Mathematical Programming",
"Math. Notes": "Mathematical Notes",
"J. Algorithms": "Journal of Algorithms",
"J. Pure Appl. Algebra": "Journal of Pure and Applied Algebra",

# Verbose variants that should map to existing entries
"Advances in Cryptology - EUROCRYPT": "Annual International Conference on the Theory and Applications of Cryptographic Techniques",
"Information Security and Cryptology - ICISC": "International Conference on Information Security and Cryptology",

# Venues you're missing
"CCC": "Computational Complexity Conference",
"ESA": "European Symposium on Algorithms",
"ISSAC": "International Symposium on Symbolic and Algebraic Computation",
"LATIN": "Latin American Theoretical Informatics Symposium",
"ESORICS": "European Symposium on Research in Computer Security",
"ICICS": "International Conference on Information and Communications Security",
"NordSec": "Nordic Conference on Secure IT Systems",
"FC": "Financial Cryptography and Data Security",
"SEC": "IFIP International Information Security Conference",
"Inscrypt": "International Conference on Information Security and Cryptology",
"AES 2004": "Advanced Encryption Standard Conference",

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
"Advances in Cryptology - CRYPTO": "International Cryptology Conference",
"Public-Key Cryptography - PKC": "International Conference on Practice and Theory of Public Key Cryptography",
"Cryptology - EUROCRYPT": "Annual International Conference on the Theory and Applications of Cryptographic Techniques",

    # S&P / SP variants
    "S&P": "IEEE Symposium on Security and Privacy",
    "IEEE S&P": "IEEE Symposium on Security and Privacy",
    "EuroS&P": "IEEE European Symposium on Security and Privacy",

    # Systems venues
    "SOSP": "ACM Symposium on Operating Systems Principles",
    "OSDI": "USENIX Symposium on Operating Systems Design and Implementation",
    "NSDI": "USENIX Symposium on Networked Systems Design and Implementation",
    "EuroSys": "European Conference on Computer Systems",
    "ASPLOS": "International Conference on Architectural Support for Programming Languages and Operating Systems",
    "ICDCS": "IEEE International Conference on Distributed Computing Systems",

    # ML / data venues
    "ICML": "International Conference on Machine Learning",
    "NeurIPS": "Annual Conference on Neural Information Processing Systems",
    "KDD": "ACM SIGKDD Conference on Knowledge Discovery and Data Mining",
    "WWW": "The Web Conference",
    "RecSys": "ACM Conference on Recommender Systems",

    # Security venues
    "CSF": "IEEE Computer Security Foundations Symposium",
    "Financial Cryptography": "Financial Cryptography and Data Security",
    "Proceedings of USENIX Security": "USENIX Security Symposium",

    # Theory venues
    "SODA": "ACM-SIAM Symposium on Discrete Algorithms",
    "ITC": "Innovations in Theoretical Cryptography",
    "ISIT": "IEEE International Symposium on Information Theory",
    "ACM STOC": "ACM Symposium on Theory of Computing",
    "Electron. Colloquium Comput. Complex.": "Electronic Colloquium on Computational Complexity",

    # Journals
    "IACR Trans. Symmetric Cryptol.": "IACR Transactions on Symmetric Cryptology",
    "TCHES": "IACR Transactions on Cryptographic Hardware and Embedded Systems",
    "Theor. Comput. Sci.": "Theoretical Computer Science",
    "Comput. Complex.": "Computational Complexity",
    "Commun. ACM": "Communications of the ACM",
    "Inf. Process. Lett.": "Information Processing Letters",
    "IEEE Trans. Inf. Forensics Secur.": "IEEE Transactions on Information Forensics and Security",
    "IEEE Trans. Computers": "IEEE Transactions on Computers",

    # Verbose / split-column variants
    "Public Key Cryptography": "International Conference on Practice and Theory of Public Key Cryptography",
    "Proceedings of the ACM CCS": "ACM Conference on Computer and Communications Security",
    "EURO-CRYPT": "Annual International Conference on the Theory and Applications of Cryptographic Techniques",
    "ASI-ACRYPT": "International Conference on the Theory and Application of Cryptology and Information Security",

    # RFCs are IETF documents, not conference papers
    "RFC": "web",

    # Unfixable
    "August": "",  # month name leaking in, discard
    "Adv. Comput. Res.": "",  # obscure, leave unmatched
    "Providing Sound Foundations for Cryptography": "",  # book title, not a venue

    # Privacy venues
    "PoPETs": "Proceedings on Privacy Enhancing Technologies",
    "PoPETS": "Proceedings on Privacy Enhancing Technologies",
    "PETS": "Proceedings on Privacy Enhancing Technologies",

    # Security venues
    "AsiaCCS": "ACM Asia Conference on Computer and Communications Security",
    "ACSAC": "Annual Computer Security Applications Conference",
    "WPES@CCS": "Workshop on Privacy in the Electronic Society",

    # ML / PL / hardware venues
    "ICLR": "International Conference on Learning Representations",
    "PLDI": "ACM SIGPLAN Conference on Programming Language Design and Implementation",
    "CAV": "International Conference on Computer Aided Verification",
    "DATE": "Design, Automation and Test in Europe",
    "DAC": "Design Automation Conference",

    # Distributed computing variants
    "ACM PODC": "ACM Symposium on Principles of Distributed Computing",

    # Journal variants
    "IEEE Trans. Dependable Secur. Comput.": "IEEE Transactions on Dependable and Secure Computing",
    "IACR TCHES": "IACR Transactions on Cryptographic Hardware and Embedded Systems",
    "IACR CRYPTO": "International Cryptology Conference",
    "IACR Eurocrypt": "Annual International Conference on the Theory and Applications of Cryptographic Techniques",
    "Commun. Assoc. Comput. Mach.": "Communications of the ACM",

    # Hyphen OCR artifact
    "IEEE Sym-posium on Security and Privacy": "IEEE Symposium on Security and Privacy",

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
    # Fix hyphenation artifacts (hyphen followed by whitespace)
    normalized = re.sub(r'-\s+', '', normalized)
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
    with open('dblp-labels.csv', 'r') as infile:
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
            if score >= 85:
                writer.writerow(row + [best_match, str(score)])
            else:
                writer.writerow(row + ["", str(score)])

if __name__ == "__main__":
    main()
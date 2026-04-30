import sys
import csv
from fuzzywuzzy import fuzz
from fuzzywuzzy import process

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
}

def main():
    if len(sys.argv) < 2:
        print("Usage: python venue_match.py <input_csv>")
        sys.exit(1)

    input_file = sys.argv[1]
    output_file = input_file.replace("_raw.csv", "_matched.csv")

    dblp_venues = []
    with open('dblp-labels.csv', 'r') as infile:
        reader = csv.reader(infile)
        next(reader)  # skip header
        for row in reader:
            dblp_venues.append(row[0])

    with open(input_file, 'r') as infile, open(output_file, 'w', newline='') as outfile:
        reader = csv.reader(infile)
        writer = csv.writer(outfile)

        header = next(reader)
        writer.writerow(header + ["venue_matched", "match_score"])

        for row in reader:
            venue_raw = row[2]  # venue_raw column

            # Skip empty venues
            if not venue_raw:
                writer.writerow(row + ["", "0"])
                continue

            if venue_raw in ABBREV_MAP:
                writer.writerow(row + [ABBREV_MAP[venue_raw], "100"])
                continue

            # Exact match
            if venue_raw in dblp_venues:
                writer.writerow(row + [venue_raw, "100"])
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
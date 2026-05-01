import csv
import sys
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

# ----------------------------
# Configuration
# ----------------------------

INPUT_FILES = {
    "EuroCrypt": "EuroCrypt_citations_matched.csv",
    "Oakland":   "Oakland_citations_matched.csv",
    "Crypto":    "Crypto_citations_matched.csv",
}

TOP_N = 15  # how many venues to show per chart

DISPLAY_NAMES = {
    "International Cryptology Conference": "CRYPTO",
    "Annual International Conference on the Theory and Applications of Cryptographic Techniques": "EUROCRYPT",
    "International Conference on the Theory and Application of Cryptology and Information Security": "ASIACRYPT",
    "Theory of Cryptography Conference": "TCC",
    "International Conference on Practice and Theory of Public Key Cryptography": "PKC",
    "ACM Conference on Computer and Communications Security": "CCS",
    "IEEE Symposium on Security and Privacy": "S&P",
    "Network and Distributed System Security Symposium": "NDSS",
    "USENIX Security Symposium": "USENIX Security",
    "ACM Symposium on Theory of Computing": "STOC",
    "IEEE Annual Symposium on Foundations of Computer Science": "FOCS",
    "Fast Software Encryption": "FSE",
    "Cryptographic Hardware and Embedded Systems": "CHES",
    "Selected Areas in Cryptography": "SAC",
    "Applied Cryptography and Network Security": "ACNS",
    "Financial Cryptography and Data Security": "FC",
    "Privacy Enhancing Technologies": "PETS",
    "Proceedings on Privacy Enhancing Technologies": "PoPETs",
    "International Colloquium on Automata, Languages and Programming": "ICALP",
    "Computational Complexity Conference": "CCC",
    "Innovations in Theoretical Computer Science": "ITCS",
    "European Symposium on Research in Computer Security": "ESORICS",
    "IACR Transactions on Symmetric Cryptology": "ToSC",
    "IACR Transactions on Cryptographic Hardware and Embedded Systems": "TCHES",
    "Designs, Codes and Cryptography": "Des. Codes Cryptogr.",
    "Journal of Cryptology": "J. Cryptology",
    "Journal of the ACM": "J. ACM",
    "SIAM Journal on Computing": "SIAM J. Comput.",
    "IEEE Transactions on Information Theory": "IEEE Trans. Inf. Theory",
    "RSA Conference, Cryptographers Track": "CT-RSA",
    "Post-Quantum Cryptography": "PQCrypto",
    "International Conference on Cryptology in India": "INDOCRYPT",
    "International Conference on Cryptology in Africa": "AFRICACRYPT",
    "International Conference on Cryptology and Information Security in Latin America": "LATINCRYPT",
    "International Conference on Information Security and Cryptology": "ICISC",
    "European Symposium on Algorithms": "ESA",
    "ePrint": "ePrint",
    "arXiv": "arXiv",
    "web": "Web",
    "web_forum": "Web Forum",
    "GitHub": "GitHub",
}

# Publication-quality settings
plt.rcParams.update({
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

COLORS = [
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
    "#937860", "#DA8BC3", "#8C8C8C", "#CCB974", "#64B5CD",
    "#4C72B0", "#DD8452", "#55A868", "#C44E52", "#8172B3",
]

# ----------------------------
# Load and count venues
# ----------------------------

def load_venue_counts(filepath):
    counts = Counter()
    total_rows = 0
    matched_rows = 0

    with open(filepath, 'r') as f:
        reader = csv.reader(f, escapechar='\\')
        header = next(reader)

        try:
            vm_idx = header.index('venue_matched')
        except ValueError:
            print(f"ERROR: 'venue_matched' column not found in {filepath}")
            print(f"  Available columns: {header}")
            sys.exit(1)

        for row in reader:
            total_rows += 1
            if len(row) <= vm_idx:
                continue
            venue = row[vm_idx].strip()
            if venue:
                matched_rows += 1
                display = DISPLAY_NAMES.get(venue, venue)
                counts[display] += 1

    print(f"  {filepath}: {matched_rows}/{total_rows} rows matched ({100*matched_rows/total_rows:.1f}%)")
    return counts

# ----------------------------
# Plot
# ----------------------------

def plot_venue_bar(ax, counts, title, top_n=TOP_N):
    top = counts.most_common(top_n)
    if not top:
        ax.text(0.5, 0.5, "No data", ha='center', va='center', transform=ax.transAxes)
        ax.set_title(title)
        return

    labels = [v for v, _ in top]
    values = [c for _, c in top]
    total = sum(counts.values())

    # Convert to percentages
    pcts = [100 * v / total for v in values]

    bars = ax.barh(
        range(len(labels)),
        pcts,
        color=COLORS[:len(labels)],
        edgecolor='white',
        linewidth=0.5,
    )

    ax.set_yticks(range(len(labels)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of matched citations", fontsize=10)
    ax.set_title(title, fontsize=13, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    # Annotate bars with count
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_width() + 0.3,
            bar.get_y() + bar.get_height() / 2,
            f"n={val}",
            va='center', fontsize=7.5, color='#444444'
        )

# ----------------------------
# Main
# ----------------------------

def main():
    output_pdf = "venue_charts.pdf"

    all_counts = {}
    for venue_name, filepath in INPUT_FILES.items():
        print(f"Loading {venue_name}...")
        try:
            all_counts[venue_name] = load_venue_counts(filepath)
        except FileNotFoundError:
            print(f"  WARNING: {filepath} not found, skipping.")

    if not all_counts:
        print("No data loaded. Check that your matched CSV files are in the current directory.")
        sys.exit(1)

    with pdf_backend.PdfPages(output_pdf) as pdf:

        # --- Page 1: One chart per source venue ---
        n = len(all_counts)
        fig, axes = plt.subplots(1, n, figsize=(6 * n, 8))
        if n == 1:
            axes = [axes]

        for ax, (venue_name, counts) in zip(axes, all_counts.items()):
            plot_venue_bar(ax, counts, f"{venue_name} — Top {TOP_N} Cited Venues")

        fig.suptitle(
            "Venue Distribution of Citations by Source Conference",
            fontsize=14, fontweight='bold', y=1.01
        )
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Page 2: Combined counts across all venues ---
        combined = Counter()
        for counts in all_counts.values():
            combined.update(counts)

        fig, ax = plt.subplots(figsize=(9, 7))
        plot_venue_bar(ax, combined, f"All Venues Combined — Top {TOP_N} Cited Venues", top_n=TOP_N)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # --- Page 3: IACR vs non-IACR summary table ---
        IACR_VENUES = {
            "CRYPTO", "EUROCRYPT", "ASIACRYPT", "TCC", "PKC",
            "FSE", "CHES", "ToSC", "TCHES", "CT-RSA", "ePrint"
        }

        fig, ax = plt.subplots(figsize=(8, 4))
        ax.axis('off')

        table_data = [["Source Venue", "IACR Citations", "Non-IACR Citations", "% IACR"]]
        for venue_name, counts in all_counts.items():
            iacr = sum(v for k, v in counts.items() if k in IACR_VENUES)
            non_iacr = sum(v for k, v in counts.items() if k not in IACR_VENUES)
            total = iacr + non_iacr
            pct = f"{100*iacr/total:.1f}%" if total > 0 else "N/A"
            table_data.append([venue_name, str(iacr), str(non_iacr), pct])

        table = ax.table(
            cellText=table_data[1:],
            colLabels=table_data[0],
            loc='center',
            cellLoc='center'
        )
        table.auto_set_font_size(False)
        table.set_fontsize(11)
        table.scale(1.2, 1.8)

        # Style header row
        for j in range(len(table_data[0])):
            table[0, j].set_facecolor('#4C72B0')
            table[0, j].set_text_props(color='white', fontweight='bold')

        ax.set_title("IACR vs. Non-IACR Citation Summary", fontsize=13, fontweight='bold', pad=20)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

    print(f"\nSaved to {output_pdf}")

if __name__ == "__main__":
    main()
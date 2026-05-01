import csv
import sys
from collections import Counter, defaultdict
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

TOP_N = 12

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

IACR_VENUES = {
    "CRYPTO", "EUROCRYPT", "ASIACRYPT", "TCC", "PKC",
    "FSE", "CHES", "ToSC", "TCHES", "CT-RSA", "ePrint"
}

AWARENESS_COLORS = {
    "1": "#4C72B0",
    "2": "#DD8452",
    "3": "#55A868",
    "4": "#C44E52",
}

plt.rcParams.update({
    "font.size": 10,
    "axes.titlesize": 12,
    "axes.labelsize": 10,
    "font.family": "serif",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
})

# ----------------------------
# Load data
# ----------------------------

def load_by_awareness(filepath):
    """Returns dict: awareness_level -> Counter of display venue names"""
    counts = defaultdict(Counter)
    total = 0
    matched = 0

    with open(filepath, 'r') as f:
        reader = csv.reader(f, escapechar='\\')
        header = next(reader)

        try:
            vm_idx = header.index('venue_matched')
            aw_idx = header.index('app_awareness')
        except ValueError as e:
            print(f"ERROR: Missing column in {filepath}: {e}")
            sys.exit(1)

        for row in reader:
            total += 1
            if len(row) <= max(vm_idx, aw_idx):
                continue
            awareness = row[aw_idx].strip()
            venue = row[vm_idx].strip()
            if awareness not in ("1", "2", "3", "4"):
                continue
            if venue:
                matched += 1
                display = DISPLAY_NAMES.get(venue, venue)
                counts[awareness][display] += 1

    print(f"  {filepath}: {matched}/{total} matched rows across awareness levels")
    return counts

# ----------------------------
# Plot helpers
# ----------------------------

def plot_awareness_bars(ax, counts_by_level, title, top_n=TOP_N):
    """One horizontal bar chart per awareness level, stacked vertically in subplots."""
    # This ax is actually a container — we use it just for the title
    # Real plotting happens in the figure-level subplots
    pass

def make_awareness_page(pdf, source_venue, counts_by_level, top_n=TOP_N):
    """One page: 2x2 grid of charts, one per awareness level."""
    levels = ["1", "2", "3", "4"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fig.suptitle(
        f"{source_venue} — Cited Venues by Application Awareness Level",
        fontsize=14, fontweight='bold', y=1.01
    )

    for ax, level in zip(axes, levels):
        counter = counts_by_level.get(level, Counter())
        top = counter.most_common(top_n)

        if not top:
            ax.text(0.5, 0.5, f"No data for level {level}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Level {level}", fontsize=12)
            continue

        labels = [v for v, _ in top]
        values = [c for _, c in top]
        total = sum(counter.values())
        pcts = [100 * v / total for v in values]

        color = AWARENESS_COLORS[level]
        bars = ax.barh(range(len(labels)), pcts,
                       color=color, alpha=0.85,
                       edgecolor='white', linewidth=0.5)

        ax.set_yticks(range(len(labels)))
        ax.set_yticklabels(labels, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of matched citations", fontsize=9)
        ax.set_title(f"Level {level}  (n={total} matched citations)", fontsize=11, fontweight='bold')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.2,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={val}", va='center', fontsize=7, color='#444444')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def make_overlay_page(pdf, source_venue, counts_by_level, top_n=TOP_N):
    """
    One page: top venues for each level overlaid on one chart,
    showing how the top-N venues shift across awareness levels.
    Only includes venues that appear in any level's top N.
    """
    # Collect union of top venues across all levels
    all_top = set()
    for level in ["1", "2", "3", "4"]:
        counter = counts_by_level.get(level, Counter())
        for v, _ in counter.most_common(top_n):
            all_top.add(v)

    venues = sorted(all_top)
    if not venues:
        return

    levels = ["1", "2", "3", "4"]
    x = range(len(venues))
    width = 0.2

    fig, ax = plt.subplots(figsize=(14, 6))

    for i, level in enumerate(levels):
        counter = counts_by_level.get(level, Counter())
        total = sum(counter.values()) or 1
        pcts = [100 * counter.get(v, 0) / total for v in venues]
        offset = (i - 1.5) * width
        ax.bar([xi + offset for xi in x], pcts,
               width=width, label=f"Level {level}",
               color=AWARENESS_COLORS[level], alpha=0.85,
               edgecolor='white', linewidth=0.5)

    ax.set_xticks(list(x))
    ax.set_xticklabels(venues, rotation=45, ha='right', fontsize=8)
    ax.set_ylabel("% of matched citations")
    ax.set_title(f"{source_venue} — Venue Share by Awareness Level (Top {top_n} venues, any level)",
                 fontsize=12, fontweight='bold')
    ax.legend(title="Awareness Level", fontsize=9)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


def make_iacr_summary_page(pdf, all_data):
    """
    Summary page: IACR vs non-IACR citation share by awareness level,
    across all source venues combined.
    """
    combined = defaultdict(Counter)
    for source_venue, counts_by_level in all_data.items():
        for level, counter in counts_by_level.items():
            combined[level].update(counter)

    levels = ["1", "2", "3", "4"]
    iacr_pcts = []
    non_iacr_pcts = []
    totals = []

    for level in levels:
        counter = combined.get(level, Counter())
        iacr = sum(v for k, v in counter.items() if k in IACR_VENUES)
        non_iacr = sum(v for k, v in counter.items() if k not in IACR_VENUES)
        total = iacr + non_iacr
        totals.append(total)
        iacr_pcts.append(100 * iacr / total if total else 0)
        non_iacr_pcts.append(100 * non_iacr / total if total else 0)

    fig, ax = plt.subplots(figsize=(8, 5))
    x = range(len(levels))
    width = 0.35

    ax.bar([xi - width/2 for xi in x], iacr_pcts, width,
           label="IACR venues", color="#4C72B0", alpha=0.85, edgecolor='white')
    ax.bar([xi + width/2 for xi in x], non_iacr_pcts, width,
           label="Non-IACR venues", color="#DD8452", alpha=0.85, edgecolor='white')

    ax.set_xticks(list(x))
    ax.set_xticklabels([f"Level {l}\n(n={t})" for l, t in zip(levels, totals)], fontsize=10)
    ax.set_ylabel("% of matched citations")
    ax.set_title("IACR vs. Non-IACR Citation Share by Awareness Level\n(All Source Venues Combined)",
                 fontsize=12, fontweight='bold')
    ax.legend(fontsize=10)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.yaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)


# ----------------------------
# Main
# ----------------------------

def main():
    output_pdf = "venue_by_awareness.pdf"

    all_data = {}
    for source_venue, filepath in INPUT_FILES.items():
        print(f"Loading {source_venue}...")
        try:
            all_data[source_venue] = load_by_awareness(filepath)
        except FileNotFoundError:
            print(f"  WARNING: {filepath} not found, skipping.")

    if not all_data:
        print("No data loaded.")
        sys.exit(1)

    with pdf_backend.PdfPages(output_pdf) as pdf:
        for source_venue, counts_by_level in all_data.items():
            print(f"Plotting {source_venue}...")
            make_awareness_page(pdf, source_venue, counts_by_level)
            make_overlay_page(pdf, source_venue, counts_by_level)

        print("Plotting combined IACR summary...")
        make_iacr_summary_page(pdf, all_data)

    print(f"\nSaved to {output_pdf}")

if __name__ == "__main__":
    main()
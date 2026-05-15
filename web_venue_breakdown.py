import csv
import io
import sys
from collections import Counter, defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.backends.backend_pdf as pdf_backend

# ----------------------------
# Configuration
# ----------------------------

INPUT_FILES = {
    "EuroCrypt": "EuroCrypt_citations_matched.csv",
    "Oakland":   "Oakland_citations_matched.csv",
    "Crypto":    "Crypto_citations_matched.csv",
    "USENIX":    "USENIX_citations_matched.csv",
}

# Keyword lists matched against lowercased raw_reference.
# Order matters: first match wins.
CATEGORIES = {
    "Standards & Specs": [
        "rfc ", "rfc-", "rfc:", " rfc ",
        "ietf", "internet-draft", "datatracker",
        "request for comments", "rfc-editor",
        "w3c", "iso/", "fips",
        "csrc.nist.gov", "nist.gov",
        "whitepaper", "white paper",
        "docs.zkproof", "specification",
    ],
    "Government & Policy": [
        "white house", "executive order",
        "department of", "ministry of",
        "u.s. government", "us government",
        "congress", "house of representatives", "senate",
        "federal register",
        "government accountability office", " gao",
        " doj", "department of justice",
        " dhs", "department of homeland security",
        " cisa", " nsa", " cia",
        "icrc.org",
        "commission", "regulation",
    ],
    "News & Tech Media": [
        "reuters", "bbc", "guardian", "new york times", "nytimes.com",
        "washington post", "wsj", "wall street journal",
        "bloomberg", "financial times", "ft.com",
        "cnn", "fox news", "npr",
        "associated press", "ap news", "politico", "axios",
        "the verge", "wired.com", "arstechnica.com", "ars technica",
        "techcrunch", "engadget", "vice", "forbes",
    ],
    "Vendor & Company": [
        "aws.amazon", "docs.aws", "amazon.com",
        "cloud.google", "google cloud",
        "azure.microsoft", "microsoft.com",
        "apple.com",
        "signal.org",
        "noiseprotocol.org",
        "zoom.us",
        "bridgefy",
        "sagemath",
    ],
    "Blog & Community": [
        "medium.com", "substack",
        "twitter.com",
        "github.io", "sites.google.com",
        "bitcointalk.org",
        "hackmd.io",
        "web.archive.org", "archive.today",
        "keccak.team",
        "blog.",
        ".blog",
        " blog",
    ],
}

ALL_CATEGORIES = list(CATEGORIES.keys()) + ["Other"]

CATEGORY_COLORS = {
    "Standards & Specs":   "#4C72B0",
    "Government & Policy": "#DD8452",
    "News & Tech Media":   "#55A868",
    "Vendor & Company":    "#C44E52",
    "Blog & Community":    "#8172B3",
    "Other":               "#8C8C8C",
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
# Classification
# ----------------------------

def classify_web_ref(raw_reference: str) -> str:
    c = raw_reference.lower()
    for category, keywords in CATEGORIES.items():
        if any(kw in c for kw in keywords):
            return category
    return "Other"

# ----------------------------
# Data loading
# ----------------------------

def read_csv(path: str) -> list[dict]:
    raw = Path(path).read_bytes().replace(b"\x00", b"")
    text = raw.decode("utf-8", errors="replace")
    return list(csv.DictReader(io.StringIO(text), escapechar="\\"))


def load_web_counts(filepath: str) -> Counter:
    counts = Counter()
    for r in read_csv(filepath):
        if r.get("venue_matched") == "web":
            counts[classify_web_ref(r.get("raw_reference", ""))] += 1
    return counts


def load_web_by_awareness(filepath: str) -> dict:
    counts = defaultdict(Counter)
    for r in read_csv(filepath):
        if r.get("venue_matched") != "web":
            continue
        level = r.get("app_awareness", "").strip()
        if level not in ("1", "2", "3", "4"):
            continue
        counts[level][classify_web_ref(r.get("raw_reference", ""))] += 1
    return counts

# ----------------------------
# Plot helpers
# ----------------------------

def plot_web_bars(ax, counts: Counter, title: str):
    values = [counts.get(cat, 0) for cat in ALL_CATEGORIES]
    total = sum(values) or 1
    pcts = [100 * v / total for v in values]
    colors = [CATEGORY_COLORS[cat] for cat in ALL_CATEGORIES]

    bars = ax.barh(range(len(ALL_CATEGORIES)), pcts,
                   color=colors, edgecolor='white', linewidth=0.5)
    ax.set_yticks(range(len(ALL_CATEGORIES)))
    ax.set_yticklabels(ALL_CATEGORIES, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("% of web citations", fontsize=9)
    ax.set_title(title, fontsize=12, fontweight='bold', pad=8)
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.xaxis.grid(True, linestyle='--', alpha=0.4)
    ax.set_axisbelow(True)

    for bar, val in zip(bars, values):
        ax.text(bar.get_width() + 0.5,
                bar.get_y() + bar.get_height() / 2,
                f"n={val}", va='center', fontsize=7.5, color='#444444')


def make_awareness_page(pdf, source_venue: str, counts_by_level: dict):
    levels = ["1", "2", "3", "4"]
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    axes = axes.flatten()

    fig.suptitle(
        f"{source_venue} — Web Citation Breakdown by Application Awareness Level",
        fontsize=13, fontweight='bold', y=1.01,
    )

    for ax, level in zip(axes, levels):
        counter = counts_by_level.get(level, Counter())
        total = sum(counter.values())

        if total == 0:
            ax.text(0.5, 0.5, f"No web citations for level {level}",
                    ha='center', va='center', transform=ax.transAxes)
            ax.set_title(f"Level {level}", fontsize=11)
            continue

        values = [counter.get(cat, 0) for cat in ALL_CATEGORIES]
        pcts = [100 * v / total for v in values]
        colors = [CATEGORY_COLORS[cat] for cat in ALL_CATEGORIES]

        bars = ax.barh(range(len(ALL_CATEGORIES)), pcts,
                       color=colors, alpha=0.85,
                       edgecolor='white', linewidth=0.5)
        ax.set_yticks(range(len(ALL_CATEGORIES)))
        ax.set_yticklabels(ALL_CATEGORIES, fontsize=8)
        ax.invert_yaxis()
        ax.set_xlabel("% of web citations", fontsize=9)
        ax.set_title(f"Level {level}  (n={total} web citations)",
                     fontsize=11, fontweight='bold',
                     color=AWARENESS_COLORS[level])
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.xaxis.grid(True, linestyle='--', alpha=0.4)
        ax.set_axisbelow(True)

        for bar, val in zip(bars, values):
            ax.text(bar.get_width() + 0.5,
                    bar.get_y() + bar.get_height() / 2,
                    f"n={val}", va='center', fontsize=7, color='#444444')

    plt.tight_layout()
    pdf.savefig(fig, bbox_inches='tight')
    plt.close(fig)

# ----------------------------
# Main
# ----------------------------

def main():
    output_pdf = "web_venue_breakdown.pdf"

    all_counts = {}
    all_by_awareness = {}

    for source_venue, filepath in INPUT_FILES.items():
        print(f"Loading {source_venue}...")
        try:
            all_counts[source_venue] = load_web_counts(filepath)
            all_by_awareness[source_venue] = load_web_by_awareness(filepath)
            total = sum(all_counts[source_venue].values())
            print(f"  {total} web citations")
        except FileNotFoundError:
            print(f"  WARNING: {filepath} not found, skipping.")

    if not all_counts:
        print("No data loaded.")
        sys.exit(1)

    with pdf_backend.PdfPages(output_pdf) as pdf:

        # Page 1: one panel per source conference
        n = len(all_counts)
        fig, axes = plt.subplots(1, n, figsize=(5 * n, 7))
        if n == 1:
            axes = [axes]

        for ax, (source_venue, counts) in zip(axes, all_counts.items()):
            total = sum(counts.values())
            plot_web_bars(ax, counts, f"{source_venue}\n(n={total} web citations)")

        fig.suptitle("Web Citation Breakdown by Source Conference",
                     fontsize=14, fontweight='bold', y=1.02)
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Page 2: combined across all conferences
        combined = Counter()
        for counts in all_counts.values():
            combined.update(counts)

        fig, ax = plt.subplots(figsize=(8, 5))
        plot_web_bars(ax, combined,
                      f"All Conferences Combined (n={sum(combined.values())} web citations)")
        plt.tight_layout()
        pdf.savefig(fig, bbox_inches='tight')
        plt.close(fig)

        # Pages 3–6: per-conference breakdown by awareness level
        for source_venue, counts_by_level in all_by_awareness.items():
            print(f"Plotting awareness breakdown for {source_venue}...")
            make_awareness_page(pdf, source_venue, counts_by_level)

        # Final page: all conferences combined by awareness level
        combined_by_level = defaultdict(Counter)
        for counts_by_level in all_by_awareness.values():
            for level, counter in counts_by_level.items():
                combined_by_level[level].update(counter)
        make_awareness_page(pdf, "All Conferences", combined_by_level)

    print(f"\nSaved to {output_pdf}")


if __name__ == "__main__":
    main()

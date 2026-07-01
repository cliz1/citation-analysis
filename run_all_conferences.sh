#!/usr/bin/env bash
# Run citation_export.py, venue_export.py, then venue_match.py for all four conferences.
# citation_export (PDF extraction) runs back-to-back with no cooldown — no DBLP.
# venue_export hits DBLP at ~1.5s/query (~25-37 min of sustained API use per
# conference); 10 min between venue_export runs lets their rate-limit window clear.
# venue_match.py is purely local (fuzzy match against dblp-labels.csv) — no cooldown.
# The inter-conference cooldown is skipped when live DBLP queries fall below
# LIVE_QUERY_THRESHOLD — most queries should be cached after the first full run.
#
# Usage: ./run_all_conferences.sh [--skip-extraction] [--skip-venue] [--skip-matching]
#   --skip-extraction  Skip citation_export.py; use existing *_citations_raw.csv files.
#   --skip-venue       Skip venue_export.py; use existing *_citations_venues.csv files.
#   --skip-matching    Skip venue_match.py; do not produce *_citations_matched.csv files.

set -euo pipefail

SKIP_EXTRACTION=false
SKIP_VENUE=false
SKIP_MATCHING=false
for arg in "$@"; do
    [[ "$arg" == "--skip-extraction" ]] && SKIP_EXTRACTION=true
    [[ "$arg" == "--skip-venue" ]]      && SKIP_VENUE=true
    [[ "$arg" == "--skip-matching" ]]   && SKIP_MATCHING=true
done

CONFERENCES=("Crypto" "EuroCrypt" "Oakland" "USENIX")
WAIT_SECONDS=600          # 10 minutes — only applied when DBLP traffic is high
LIVE_QUERY_THRESHOLD=200  # skip cooldown if fewer than this many live queries

for i in "${!CONFERENCES[@]}"; do
    conf="${CONFERENCES[$i]}"

    if [[ "$SKIP_EXTRACTION" == false ]]; then
        echo "========================================"
        echo "Starting citation extraction: $conf  ($(date))"
        echo "========================================"
        python3 citation_export.py --conference "$conf" 2>&1 | tee "logs/${conf}_citation_run.txt"
        echo ""
    fi

    if [[ "$SKIP_VENUE" == false ]]; then
        echo "========================================"
        echo "Starting venue extraction: $conf  ($(date))"
        echo "========================================"
        python3 venue_export.py --conference "$conf" 2>&1 | tee "logs/${conf}_venue_run.txt"
        echo ""
    fi

    if [[ "$SKIP_MATCHING" == false ]]; then
        echo "Starting venue matching: $conf  ($(date))"
        echo "========================================"
        python3 venue_match.py "csv/${conf}_citations_venues.csv" 2>&1 | tee "logs/${conf}_venue_match_run.txt"
        echo ""
    fi

    if [[ $i -lt $(( ${#CONFERENCES[@]} - 1 )) ]]; then
        next="${CONFERENCES[$((i+1))]}"
        echo "Run complete: $conf  ($(date))"
        if [[ "$SKIP_VENUE" == false ]]; then
            live_queries=$(grep -c "DBLP query ([0-9]" "logs/${conf}_venue_run.txt") || live_queries=0
            live_queries=${live_queries:-0}
            if [[ "$live_queries" -ge "$LIVE_QUERY_THRESHOLD" ]]; then
                echo "$live_queries live DBLP queries — waiting ${WAIT_SECONDS}s before starting $next ..."
                sleep "$WAIT_SECONDS"
            else
                echo "$live_queries live DBLP queries (mostly cached) — skipping cooldown, starting $next immediately."
            fi
        fi
    fi
done

echo ""
echo "All runs complete  ($(date))"

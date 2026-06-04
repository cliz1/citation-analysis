#!/usr/bin/env bash
# Run citation_export.py then venue_export.py for all four conferences.
# citation_export (PDF extraction) runs back-to-back with no cooldown — no DBLP.
# venue_export hits DBLP at ~1.5s/query (~25-37 min of sustained API use per
# conference); 10 min between venue_export runs lets their rate-limit window clear.

set -euo pipefail

CONFERENCES=("Crypto" "EuroCrypt" "Oakland" "USENIX")
WAIT_SECONDS=600  # 10 minutes

for i in "${!CONFERENCES[@]}"; do
    conf="${CONFERENCES[$i]}"
    echo "========================================"
    echo "Starting citation extraction: $conf  ($(date))"
    echo "========================================"

    python3 citation_export.py --conference "$conf" 2>&1 | tee "logs/${conf}_citation_run.txt"

    echo ""
    echo "Starting venue extraction: $conf  ($(date))"
    echo "========================================"

    python3 venue_export.py --conference "$conf" 2>&1 | tee "logs/${conf}_venue_run.txt"

    if [[ $i -lt $(( ${#CONFERENCES[@]} - 1 )) ]]; then
        next="${CONFERENCES[$((i+1))]}"
        echo ""
        echo "Run complete: $conf  ($(date))"
        echo "Waiting ${WAIT_SECONDS}s before starting $next ..."
        sleep "$WAIT_SECONDS"
    fi
done

echo ""
echo "All runs complete  ($(date))"

#!/usr/bin/env bash
# Run citation_export.py for all four conferences with a 10-minute cooldown between
# runs. DBLP paces queries at 1.5s each (~25-37 min of sustained API use per run);
# 10 min between runs lets their sliding rate-limit window clear.

set -euo pipefail

CONFERENCES=("Crypto" "EuroCrypt" "Oakland" "USENIX")
WAIT_SECONDS=600  # 10 minutes

for i in "${!CONFERENCES[@]}"; do
    conf="${CONFERENCES[$i]}"
    echo "========================================"
    echo "Starting run: $conf  ($(date))"
    echo "========================================"

    python3 citation_export.py --conference "$conf" 2>&1 | tee "logs/${conf}_run.txt"

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

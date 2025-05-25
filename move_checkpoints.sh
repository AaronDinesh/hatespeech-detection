#!/bin/bash

SRC_CHECKPOINTS="MultiModN/pipelines/mmhs/checkpoints"
SRC_RESULTS="MultiModN/pipelines/mmhs/results"
DEST_BASE="/mnt/course-ee-559/collaborative/group038"
INTERVAL=60  # seconds

echo "Watching for new .pt and .csv files every $INTERVAL seconds..."

while true; do
    # --- Move .pt files ---
    find "$SRC_CHECKPOINTS" -type f -name "*.pt" | while read -r filepath; do
        rel_path="${filepath#$SRC_CHECKPOINTS/}"
        dest_path="$DEST_BASE/mmn_checkpoints/checkpoints/$rel_path"

        mkdir -p "$(dirname "$dest_path")"
        if [ -f "$filepath" ]; then
            mv "$filepath" "$dest_path" && echo "$(date '+%Y-%m-%d %H:%M:%S') Moved: $filepath → $dest_path"
        fi
    done

    # --- Move .csv files ---
    find "$SRC_RESULTS" -type f -name "*.csv" -path "*/result*/*" | while read -r filepath; do
        rel_path="${filepath#MultiModN/pipelines/mmhs/results/}"  # keep 'result*/...'
        dest_path="$DEST_BASE/results/$rel_path"

        mkdir -p "$(dirname "$dest_path")"
        if [ -f "$filepath" ]; then
            mv "$filepath" "$dest_path" && echo "$(date '+%Y-%m-%d %H:%M:%S') Moved: $filepath → $dest_path"
        fi
    done	


    sleep "$INTERVAL"
done


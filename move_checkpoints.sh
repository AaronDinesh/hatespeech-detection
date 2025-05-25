#!/bin/bash

SRC_DIR="MultiModN/pipelines/mmhs/checkpoints"
DEST_DIR="/mnt/course-ee-559/collaborative/group038/mmn_checkpoints"
INTERVAL=60  # seconds

echo "Watching for new .pt files in $SRC_DIR..."

while true; do
    find "$SRC_DIR" -type f -name "*.pt" | while read -r filepath; do
        # Compute relative path
        rel_path="${filepath#$SRC_DIR/}"
        dest_path="$DEST_DIR/$rel_path"

        # Create destination directory if needed
        mkdir -p "$(dirname "$dest_path")"

        # Move the file
        if [ -f "$filepath" ]; then
            mv "$filepath" "$dest_path" && echo "Moved: $filepath → $dest_path"
        fi
    done

    sleep "$INTERVAL"
done


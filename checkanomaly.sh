#!/bin/bash
IMAGE_PATH="${1}"  # Use $1
if [ -z "$IMAGE_PATH" ]; then
	echo "Usage: $0 <image_path> [output_dir]"
	exit 1
fi
OUTPUT_DIR="${2:-processed_image}"    # Use $2 or default
source .venv/bin/activate
python apply_filter.py "$IMAGE_PATH" -o "$OUTPUT_DIR"

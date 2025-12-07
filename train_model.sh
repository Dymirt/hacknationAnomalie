#!/bin/bash
IMAGE_PATH="${1:-default/path.bmp}"  # Use $1 or default
OUTPUT_DIR="${2:-processed_image}"    # Use $2 or default
source .venv/bin/activate
python data_normalization.py "$IMAGE_PATH" -o "$OUTPUT_DIR"

python train.py

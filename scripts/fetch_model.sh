#!/bin/sh
# fetch_model.sh - download the original Lexnet files (weights, reference lexnet.c, kernel
# header, README) from Hugging Face into model/. They are Apache-2.0 licensed and are not
# committed to this repository; the reference binary and the differential test need them.
set -e
DIR=$(cd "$(dirname "$0")/.." && pwd)
DEST="$DIR/model"
REPO="alpindale/lexnet"
FILES="lexnet_weights.bin lexnet.c lexnet_kernel.h README.md"

mkdir -p "$DEST"
if command -v hf >/dev/null 2>&1; then
    hf download "$REPO" $FILES --local-dir "$DEST"
else
    for f in $FILES; do
        echo "fetching $f"
        curl -fsSL "https://huggingface.co/$REPO/resolve/main/$f" -o "$DEST/$f"
    done
fi
ls -la "$DEST"

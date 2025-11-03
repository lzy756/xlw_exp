#!/bin/bash
# Optional script to download and prepare DomainNet dataset
# This script is provided for convenience but not required if data already exists

set -e

DATA_DIR="${1:-./data/domainnet}"
DOMAINS=("clipart" "infograph" "painting" "quickdraw" "real" "sketch")

echo "================================================"
echo "DomainNet Dataset Preparation Script"
echo "================================================"
echo "This script helps download DomainNet if needed."
echo "Data directory: $DATA_DIR"
echo ""

# Check if data already exists
if [ -d "$DATA_DIR" ]; then
    echo "Data directory already exists at $DATA_DIR"
    echo -n "Files found: "
    find "$DATA_DIR" -name "*.jpg" -o -name "*.png" | wc -l
    echo ""
    read -p "Data appears to exist. Skip download? (Y/n): " skip
    if [[ "$skip" != "n" && "$skip" != "N" ]]; then
        echo "Skipping download. Exiting."
        exit 0
    fi
fi

echo ""
echo "NOTE: DomainNet is a large dataset (~40GB)."
echo "Download URLs for each domain:"
echo "- clipart: http://csr.bu.edu/ftp/visda/2019/multi-source/clipart.zip"
echo "- infograph: http://csr.bu.edu/ftp/visda/2019/multi-source/infograph.zip"
echo "- painting: http://csr.bu.edu/ftp/visda/2019/multi-source/painting.zip"
echo "- quickdraw: http://csr.bu.edu/ftp/visda/2019/multi-source/quickdraw.zip"
echo "- real: http://csr.bu.edu/ftp/visda/2019/multi-source/real.zip"
echo "- sketch: http://csr.bu.edu/ftp/visda/2019/multi-source/sketch.zip"
echo ""
echo "Please download manually and extract to: $DATA_DIR"
echo ""

echo "Creating directory structure..."
mkdir -p "$DATA_DIR"

echo ""
echo "After downloading and extracting, your directory should look like:"
echo "$DATA_DIR/"
echo "├── clipart/"
echo "│   ├── aircraft_carrier/"
echo "│   ├── airplane/"
echo "│   └── ..."
echo "├── infograph/"
echo "├── painting/"
echo "├── quickdraw/"
echo "├── real/"
echo "└── sketch/"
echo ""

echo "Once data is ready, you'll need to create an index.json file."
echo "You can generate it using a Python script that scans the directories."
echo ""
echo "Done! Please prepare the data and then run the experiment."
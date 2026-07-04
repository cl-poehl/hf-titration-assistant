#!/bin/bash
#
# Downloads the open-access MIMIC-IV and eICU *demo* datasets from PhysioNet into
# data/external/. These are the small, openly licensed demo subsets (ODbL) — NOT
# the full credentialed databases — so no PhysioNet login is required. They are
# intentionally not committed to this repo (see .gitignore).
#
# After running this, build the real-data feature matrix and model with:
#   python src/data_generation/extract_mimic.py
#   python src/data_generation/extract_eicu.py
#   python src/data_generation/combine_datasets.py
#   python src/features/build_features.py --combined
#   python src/models/train.py --real
#
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
EXTERNAL_DIR="${HFTA_DATA_DIR:-$PROJECT_ROOT/data}/external"
mkdir -p "$EXTERNAL_DIR"
cd "$EXTERNAL_DIR"

echo "Downloading MIMIC-IV clinical database demo (v2.2)..."
wget -r -N -c -np -nH --cut-dirs=4 -P mimic-iv-demo \
  https://physionet.org/files/mimic-iv-demo/2.2/

echo "Downloading eICU collaborative research database demo (v2.0.1)..."
wget -r -N -c -np -nH --cut-dirs=4 -P eicu-demo \
  https://physionet.org/files/eicu-crd-demo/2.0.1/

echo "Done. Demo datasets are in $EXTERNAL_DIR"
echo "Project pages:"
echo "  https://physionet.org/content/mimic-iv-demo/2.2/"
echo "  https://physionet.org/content/eicu-crd-demo/2.0.1/"

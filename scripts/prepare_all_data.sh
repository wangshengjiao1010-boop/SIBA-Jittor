#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT=${PROJECT_ROOT:-/root/autodl-tmp/SIBA-Jittor}

bash "$PROJECT_ROOT/scripts/download_public_repos.sh"
bash "$PROJECT_ROOT/scripts/download_test_datasets.sh"
bash "$PROJECT_ROOT/scripts/materialize_official_test_sources.sh"
bash "$PROJECT_ROOT/scripts/run_manifests.sh"
bash "$PROJECT_ROOT/scripts/prepare_test_datasets.sh"

echo "All SIBA training and test data are prepared"

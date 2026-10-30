#!/bin/bash
# Check tracked and unignored files in Git, or the folder contents before git init.
set -euo pipefail
cd "$(dirname "$0")/.."
exec "${PY:-python3}" scripts/check_release.py

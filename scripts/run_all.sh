#!/bin/bash
# Full pipeline. Each step can also be run on its own; see docs/01_개요.md for inputs and outputs of every step.
set -euo pipefail
D="$(dirname "$0")"
for s in 01_baseline 02_mapping 03_census 04_blocks 05_masks 06_generate 07_validate 08_results; do bash "$D/$s.sh"; done

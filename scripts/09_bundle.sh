#!/bin/bash
# 09. Generation bundle (bundle/): models, masks, synthetic census base and census aggregates, so that users can generate without the raw data.
source "$(dirname "$0")/_env.sh"
run bundle $PY -m krpersona.report.bundle

#!/bin/bash
# 01. MatrAIx baseline: 1,000,000 MatrAIx personas with the official sampler; keeps Korean-culture natives and a uniform sample.
source "$(dirname "$0")/_env.sh"
run baseline_gen      $PY -m krpersona.baseline.gen 1000000
run baseline_analyze  $PY -m krpersona.baseline.analyze
run baseline_checks   $PY -m krpersona.baseline.demo_checks

#!/bin/bash
# 03. Census 2% sample -> analysis tables -> CART synthesis (all households; 80% for validation; 50% for the memorization test) -> checks.
source "$(dirname "$0")/_env.sh"
run census_prep      $PY -m krpersona.census.prep
run census_aggregates $PY -m krpersona.census.aggregates
run census_synth_s1  $PY -m krpersona.census.synth 0 1
run census_synth_s5  $PY -m krpersona.census.synth 0 5 0.8
run census_synth_s7  $PY -m krpersona.census.synth 0 7 0.5
run census_check_s1  $PY -m krpersona.census.check syn_s1
for t in syn_s1 syn_s5_f0.8 syn_s7_f0.5; do run census_finalize_$t $PY -m krpersona.census.finalize $t; done
run census_holdout   $PY -m krpersona.census.holdout syn_s7_f0.5 7

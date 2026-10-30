#!/bin/bash
# 04. Block models: on all data (model files without suffix) and on 80% of every source (KR_HOLDOUT=1, suffix _80).
# Order matters: values needs pers_params; health needs pers_params; kgss needs value_params; dev needs labor_income constants.
source "$(dirname "$0")/_env.sh"
for H in 0 1; do
  export KR_HOLDOUT=$H; s=$([ $H = 1 ] && echo _80 || echo "")
  run pers$s          $PY -m krpersona.blocks.pers
  run values$s        $PY -m krpersona.blocks.values
  run health$s        $PY -m krpersona.blocks.health
  run labor_income$s  $PY -m krpersona.blocks.labor_income
  run kgss$s          $PY -m krpersona.blocks.kgss
  run kgss_english$s  $PY -m krpersona.blocks.kgss english_proficiency
  run family$s        $PY -m krpersona.blocks.family
  run dev$s           $PY -m krpersona.blocks.dev
done

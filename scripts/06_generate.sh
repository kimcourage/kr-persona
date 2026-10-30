#!/bin/bash
# 06. Samples: validation (80% models) with and without the value -> attitude links; final (100% models) with and without MatrAIx attributes.
source "$(dirname "$0")/_env.sh"
KR_HOLDOUT=1 run gen_val_on   $PY -m krpersona.generate.run 20000 11 on val_on
KR_HOLDOUT=1 KR_NO_VLINKS=1 run gen_val_novl $PY -m krpersona.generate.run 20000 11 on val_novl
run gen_out_on  $PY -m krpersona.generate.run 20000 1 on out_on
run gen_out_off $PY -m krpersona.generate.run 5000 2 off out_off

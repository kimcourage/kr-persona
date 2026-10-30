#!/bin/bash
# 05. Masks: (a) MatrAIx masks with a Korean target, classified on 60,000 stage-0 personas;
# (b) masks compiled from rule sets 1-3 on a baseline sample (the stored masks used samples/test5, made with KR_HOLDOUT=1, 5,000 personas, seed 3,
#     on 2026-10-29 before any compiled masks existed). A new baseline is made without compiled masks (KR_NO_RULE_MASKS=1) so that it does not depend on them.
source "$(dirname "$0")/_env.sh"
run classify_masks $PY -m krpersona.matraix.classify_masks
if [ ! -f "$ART/samples/test5/personas.parquet" ]; then KR_HOLDOUT=1 KR_NO_RULE_MASKS=1 run gen_test5 $PY -m krpersona.generate.run 5000 3 on test5; fi
run compile_masks $PY -m krpersona.generate.compile_masks "$ART/samples/test5/personas.parquet"

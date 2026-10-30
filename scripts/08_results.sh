#!/bin/bash
# 08. Plain result tables (results/검증_결과.xlsx, results/csv/*.csv) and documentation tables (docs/tables/*.csv).
source "$(dirname "$0")/_env.sh"
run results $PY -m krpersona.report.results
run doc_tables $PY -m krpersona.report.doc_tables
run tables_blocks_a $PY -m krpersona.report.tables_blocks_a
run tables_blocks_b $PY -m krpersona.report.tables_blocks_b

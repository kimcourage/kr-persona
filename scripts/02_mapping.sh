#!/bin/bash
# 02. KGSS block structure from the mapping workbook (mapping/변수대응표.xlsx is maintained by hand and tracked in git).
source "$(dirname "$0")/_env.sh"
run kgss_analysis $PY -m krpersona.mapping.kgss_analysis

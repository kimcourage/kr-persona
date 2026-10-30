# common settings for the step scripts: run from the repository root
set -euo pipefail
cd "$(dirname "${BASH_SOURCE[0]}")/.."
export PYTHONPATH="$PWD${PYTHONPATH:+:$PYTHONPATH}"
PY=${PY:-python3}
ART=$($PY -c "from krpersona import paths; print(paths.art())")
LOG="$ART/logs"; mkdir -p "$LOG"
run() { local name=$1; shift; echo "[$(date '+%H:%M:%S')] $name"; "$@" > "$LOG/$name.log" 2>&1; }

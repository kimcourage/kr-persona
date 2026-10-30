#!/bin/bash
# 07. Validation against held-out real data, consistency rules, renderer checks. Results in artifacts/validation.
source "$(dirname "$0")/_env.sh"
V="$ART/validation"; mkdir -p "$V"; SV="$ART/samples/val_on"; SO="$ART/samples/out_on"; H="$ART/samples/history"
KR_HOLDOUT=1 run validate        $PY -m krpersona.validate.validate "$SV/personas.parquet" "$SV/internal.parquet" "$V/val_results.pkl"
KR_HOLDOUT=1 run pmse_subset     $PY -m krpersona.validate.pmse_subset "$SV/personas.parquet" "$V/pmse_matraix_subset.pkl"
KR_HOLDOUT=1 run value_links     $PY -m krpersona.validate.value_link_check "$SV/internal.parquet" "$ART/samples/val_novl/internal.parquet" "$V/value_link_check.pkl"
KR_HOLDOUT=1 run english_check   $PY -m krpersona.validate.english_check "$SV/personas.parquet" "$SV/internal.parquet" "$V/english_check.pkl"
KR_HOLDOUT=1 run eaps_decomp     $PY -m krpersona.validate.eaps_decomp "$SV/personas.parquet" "$V/eaps_decomp.pkl"
run render_check_val  $PY -m krpersona.validate.render_check "$SV/personas.parquet" "$V/render_check_val.pkl"
run render_check_full $PY -m krpersona.validate.render_check "$SO/personas.parquet" "$V/render_check_full.pkl"
run compare           $PY -m krpersona.validate.compare "$SO/internal.parquet" "$SO/personas.parquet" "$V/compare_out_on.pkl"
run sampler_check     $PY tests/test_sampler.py
run audit             $PY -m krpersona.validate.audit "$SO/personas.parquet" "$V/audit_out_on.pkl"
# Earlier development samples are optional and are not distributed with this repository.
# Preserve their order when present: rule-column classification uses the first sample.
L=()
for entry in "2차 수정 후=$H/personas_round2.parquet" "3차 수정 직후=$H/personas_round3.parquet"; do
  if [ -f "${entry#*=}" ]; then L+=("$entry"); fi
done
L+=("3차 보완 후 검증 표본=$SV/personas.parquet" "3차 보완 후 최종 표본=$SO/personas.parquet")
run language_check $PY -m krpersona.validate.language_check "$V/language_check.pkl" "${L[@]}"

T=()
for entry in "수정 전(4단계 최종 표본)=$H/personas_before_fix.parquet" "1차 수정 후(최종 표본)=$H/personas_round1.parquet" \
             "2차 수정 후(최종 표본)=$H/personas_round2.parquet" "3차 수정 직후(최종 표본)=$H/personas_round3.parquet"; do
  if [ -f "${entry#*=}" ]; then T+=("$entry"); fi
done
T+=("3차 보완 후 검증 표본(80% 모형)=$SV/personas.parquet" "3차 보완 후 최종 표본(100% 모형)=$SO/personas.parquet")
for k in 4 3 2 1; do run rules_set$k $PY -m krpersona.validate.rules_summary $k "$V/rules_set$k.pkl" "${T[@]}"; done
run rules_groups_set4 $PY -m krpersona.validate.rules_groups 4 "$V/rules4_groups.pkl" "${T[@]}"

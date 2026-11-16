#!/usr/bin/env bash
# Sequential: exactly one pytest process at a time.
cd /d/Projekty/intergrax/.claude/worktrees/pdc-x-ia-d4-966471
D=tests/qualification/pdc_x/d4
declare -A T=(
 [M1_forward_stop_order]="$D/test_d4_lifecycle_suite.py"
 [M2_no_runtime_degradation]="$D/test_d4_lifecycle_suite.py"
 [M3_no_cancel]="$D/test_xd05_drain_background_work.py"
 [M4_no_exclusive_lock]="$D/test_d4_lifecycle_suite.py"
 [M5_no_digest_verification]="$D/test_d4_restart_digest.py"
 [M6_no_metrics]="$D/test_d4_lifecycle_metrics.py"
 [M7_never_restart]="$D/test_d4_crash_restart_subprocess.py"
 [M8_public_surface_change]="$D/test_d4_public_api_manifest.py"
 [M9_drain_skipped]="$D/test_xd05_drain_background_work.py"
 [M10_startup_rollback_skipped]="$D/test_d4_lifecycle_suite.py"
 [M11_lease_not_released]="$D/test_d4_lifecycle_suite.py"
 [M12_exit_class_label_constant]="$D/test_d4_lifecycle_metrics.py"
)
OUT=.tmp/session/ia-d4/mut/results.txt
: > $OUT
for m in M1_forward_stop_order M2_no_runtime_degradation M3_no_cancel M4_no_exclusive_lock M5_no_digest_verification M6_no_metrics M7_never_restart M8_public_surface_change M9_drain_skipped M10_startup_rollback_skipped M11_lease_not_released M12_exit_class_label_constant; do
  IAD4_MUT=$m PYTHONPATH=.tmp/session/ia-d4/mut timeout 300 uv run --extra mcp pytest -p mut_plugin ${T[$m]} -q -x --no-header > .tmp/session/ia-d4/mut/$m.log 2>&1
  rc=$?
  last=$(grep -E "passed|failed|error" .tmp/session/ia-d4/mut/$m.log | tail -1)
  first_fail=$(grep -E "^FAILED" .tmp/session/ia-d4/mut/$m.log | head -1 | cut -c1-170)
  echo "$m | file=${T[$m]##*/} | rc=$rc | $last | $first_fail" >> $OUT
done
echo DONE >> $OUT

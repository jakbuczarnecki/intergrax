#!/usr/bin/env bash
# IA-D6 mutation sweep: ONE pytest process at a time, sequential.
cd "$(dirname "$0")/../../../.." || exit 1
D=tests/qualification/pdc_x/d6
OUT=.tmp/session/ia-d6/mut
declare -A FILES
FILES[M1]="$D/test_d6_signing_fail_closed.py"
FILES[M2]="$D/test_d6_poisoning_default.py"
FILES[M3]="$D/test_d6_tool_input_defense_path.py"
FILES[M4]="$D/test_d6_plugin_provenance.py"
FILES[M5]="$D/test_d6_demo_seed_gate.py"
FILES[M6]="$D/test_xd06_credential_revocation.py $D/test_xd06_provider_boundary.py"
FILES[M7]="$D/test_xd06_delegation_widening.py"
FILES[M8]="$D/test_d6_secrets.py"
FILES[M9]="$D/test_d6_audit_trail_durable.py $D/test_d6_secrets.py"
FILES[M10]="$D/test_d6_audit_trail_durable.py"
FILES[M11]="$D/test_d6_security_config_fail_closed.py"
FILES[M12]="$D/test_d6_demo_seed_gate.py $D/test_d6_amendment1_gap_closures.py"
for m in M1 M2 M3 M4 M5 M6 M7 M8 M9 M10 M11 M12; do
  IA_MUT=$m PYTHONPATH="$PWD/$OUT" timeout 300 uv run pytest -p ia_mut -q ${FILES[$m]} > "$OUT/$m.log" 2>&1
  echo "$m exit=$?" >> "$OUT/summary.txt"
done
echo DONE >> "$OUT/summary.txt"

#!/bin/bash
OUT=/d/Projekty/intergrax/.tmp/session/mcp-p7p8-run
PY=/d/Projekty/intergrax/.venv/Scripts/python.exe
FILES="tests/unit/applications/test_mcp_surface_opt_in.py tests/unit/applications/test_mcp_surface_opt_in_runtime.py tests/unit/applications/test_mcp_catalog_e2e_hung_server_qualification.py"
for n in p7-base-dev p7-base-p5; do
  cd /d/Projekty/_wt/intergrax/$n || exit 9
  SHA=$(git rev-parse HEAD); log="$OUT/baseline_${n}.log"
  { echo "command: pytest $FILES -q"; echo "cwd: $(pwd)"; echo "git_head: $SHA"; echo "date_utc: $(date -u +%FT%TZ)"; } > "$log"
  "$PY" -m pytest $FILES -q >> "$log" 2>&1; rc=$?
  { echo "exit_code: $rc"; echo "date_end_utc: $(date -u +%FT%TZ)"; } >> "$log"
  echo "$n $SHA exit=$rc $(grep -E 'passed|failed' "$log" | tail -1)"
done

#!/bin/bash
# usage: run_p7_suites.sh <worktree> <label>   (one pytest process at a time)
set -u
WT="$1"; LABEL="$2"; OUT=/d/Projekty/intergrax/.tmp/session/mcp-p7p8-run
PY=/d/Projekty/intergrax/.venv/Scripts/python.exe
cd "$WT" || exit 9
SHA=$(git rev-parse HEAD)
run() { # name, then pytest args
  local name="$1"; shift
  local log="$OUT/${LABEL}_${name}.log"
  { echo "command: pytest $* -q"; echo "cwd: $WT"; echo "git_head: $SHA"; echo "date_utc: $(date -u +%FT%TZ)"; } > "$log"
  "$PY" -m pytest "$@" -q >> "$log" 2>&1
  local rc=$?
  { echo "exit_code: $rc"; echo "date_end_utc: $(date -u +%FT%TZ)"; } >> "$log"
  echo "$name exit=$rc $(grep -E 'passed|failed|error' "$log" | tail -1)"
}
NEW=$("$PY" -I -c "import sys;sys.path.insert(0,'.');from tests.qualification.mcp_server_x.p7_selection import MCP_SERVER_X_P7_NEW_QUALIFICATION as n;print(' '.join(n))")
REG=$("$PY" -I -c "import sys;sys.path.insert(0,'.');from tests.qualification.mcp_server_x.p7_selection import MCP_SERVER_X_P7_REGRESSION_SELECTION as n;print(' '.join(n))")
run new $NEW
run consistency tests/qualification/mcp_server_x/test_mcp_server_x_p7_matrix_consistency.py --deselect tests/qualification/mcp_server_x/test_mcp_server_x_p7_matrix_consistency.py::test_acc_20_exact_sha_package_record_exists_and_pins_a_test_execution_sha
run regression $REG
echo "DONE sha=$SHA"

# PLAN T-IGX-062-B — ENT-AUDIT-X sandbox `uv` executable resolution

## Problem

`CheckExecutionSandbox` uses argv `uv` without resolution; Windows operators with `uv` outside PATH get `WinError 2`, blocking semantic ENT-A-DUP-01 qualification.

## Change

1. Add `_resolve_uv_executable()`: optional env `INTERGRAX_UV_EXECUTABLE` (fail-closed if set but missing), else `shutil.which("uv")`, else `UV` invoker path when set by `uv run`.
2. Build subprocess argv with resolved executable.
3. Unit tests: mock `which` / env; assert cmd[0] is resolved path; assert clear error when neither is available.

## Write set

```text
intergrax/platform/enterprise_audit/sandbox.py
tests/unit/platform/enterprise_audit/test_sandbox_uv_resolution.py
governance/tickets/T-IGX-062-B.md
governance/tickets/plans/PLAN_T-IGX-062-B.md
```

## Test contract

```text
uv run pytest tests/unit/platform/enterprise_audit/test_sandbox_uv_resolution.py tests/qualification/ent_audit_x/unit/test_sandbox_isolation.py -q --tb=line
uv run pytest tests/qualification/ent_audit_x/acceptance/test_ent_a_ent_a_dup_01.py -q --tb=line
```
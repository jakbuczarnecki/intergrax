# T-IGX-075-D — EC-W3 mapped corrective settlement (EQO-CL-01 / EC-W3-06)

## Scope

Enterprise Campaign EQO pre-G2 mapped corrective execution EC-W3-CORRECTIVE-06: authorized test-only corrections (Packages A–C), five-group mapped regression with cross-application isolation witness, evidence under `eqo_closure/runs/pre_g2_mapped_corrective_06_<sha8>/`. No product code changes.

## Plan digest

Plan digest: n/a (EC-W3-CORRECTIVE-06 EQO-CL-01 correction batch)

## Approved

Approved: operator-delegate @ 2026-11-27T11:20:00+01:00

Delegation: EC-W3-CORRECTIVE-06 operator execution authority (Cursor session INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-06)

## Write set

```text
applications/local_workspace_application/tests/host/test_local_workspace_policy_wiring.py
tests/unit/applications/test_cross_application_policy_isolation.py
tests/unit/runtime/security/test_p0_safety_7_sandbox_isolation_fail_closed.py
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/pre_g2_mapped_corrective_06_*/**
governance/tickets/T-IGX-075-D.md
```

## Test contract

```text
uv run ruff check applications/local_workspace_application/tests/host/test_local_workspace_policy_wiring.py tests/unit/applications/test_cross_application_policy_isolation.py tests/unit/runtime/security/test_p0_safety_7_sandbox_isolation_fail_closed.py
git diff --check
uv run python -m scripts.governance.igx_gov ticket
```

Package D five-group pytest per DES_PREREG; Package B/C regressions per EC-W3-CORRECTIVE-06.

## Out of scope

EDER/FRZ register edits, EQO closure promotion, Harness Policy Catalog product fix, qualification queue execution, intergrax/** and application production code.

## Lifecycle state

in_progress
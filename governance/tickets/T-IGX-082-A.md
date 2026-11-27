# T-IGX-082-A — EC-W3 defect settlement 09 (EQO-CL-01 / EC-W3-09)

## Scope

PRE_G2_TARGETED_DEFECT_SETTLEMENT EC-W3-DEFECT-SETTLEMENT-09: execution environment evidence (Package A), notification_channel reference binding (Package B), GCA smoke continuation fixture (Package C), offline demo governance identity (Package D), XD-12 impact map paths (Package E), evidence integrity and mechanical reconciliation (Packages G–I), publication under `eqo_closure/runs/pre_g2_defect_settlement_09_<sha8>/`.

## Plan digest

Plan digest: n/a (EC-W3-DEFECT-SETTLEMENT-09 operator batch; EQO-W3 Campaign B-1; EQO-CL-01 correction)

## Approved

Approved: operator-delegate @ 2026-11-27T14:40:00+01:00

Delegation: EC-W3-DEFECT-SETTLEMENT-09 operator execution authority (Cursor session INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-09 DEFECT SETTLEMENT)

## Write set

```text
applications/governed_contractor_application/host/environment_profile.py
applications/governed_contractor_application/host/offline_demo.py
intergrax/platform/enterprise_audit/impact_map.py
intergrax/platform/enterprise_audit/data/changed_since_impact_map.v1.json
applications/governed_contractor_application/tests/host/test_governed_contractor_host_smoke.py
tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/pre_g2_defect_settlement_09_*/
governance/tickets/T-IGX-082-A.md
```

## Test contract

Targeted regression per EC-W3-09 Packages A–I; `uv run ruff check` on write set; `git diff --check`; `uv run python -m scripts.governance.igx_gov ticket`.

## Out of scope

COMPAT-X owner reconciliation, ROADMAP-REPLAY-X scoped delta, conftest/lockfile/CI/EDER/FRZ edits, formal G2 certification, EQO_CLOSED promotion, orchestrator.py changes.

## Lifecycle state

in_progress
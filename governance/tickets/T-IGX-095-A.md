# T-IGX-095-A — EC-W3 evidence integrity recovery 12 (EQO-W3 B-1 / EC-W3-12)

## Scope

PRE_G2_EVIDENCE_INTEGRITY_RECOVERY: EC-F15 P2C inbound verifier fixture (test-only), EC-F14 supplementary FC1 negative coverage, semantic evidence runner/reconciliation/validator (EC-W3-12 evidence root), consolidated bounded replay D01–D05 and Q01–Q09, EQO-RV-01 reconciliation, publication under `eqo_closure/runs/pre_g2_evidence_integrity_recovery_12_<sha8>/`.

## Plan digest

Plan digest: n/a (EC-W3-EVIDENCE-INTEGRITY-RECOVERY-12 operator batch; EQO-W3 Campaign B-1; EQO-CL-01 / EQO-RV-01 / EQO-X-06)

## Approved

Approved: operator-delegate @ 2026-11-28T07:20:00+01:00

Delegation: EC-W3-EVIDENCE-INTEGRITY-RECOVERY-12 operator execution authority (Cursor session INTEGRAX | ENTERPRISE CAMPAIGN EQO | W3 B-1 | EC-W3-12 EVIDENCE INTEGRITY RECOVERY)

## Write set

```text
tests/qualification/prod_q/test_prod_q_r1_r7_q_p2_c_http_execution_admission.py
tests/unit/runtime/nexus/test_fc1_nexus_loop_persistence_posture.py
docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/runs/pre_g2_evidence_integrity_recovery_12_*/
governance/tickets/T-IGX-095-A.md
```

## Test contract

EC-W3-12 packages A–G; 28/28 validator adversarial controls; semantic validator exit 0; P2C 7/7; FC1 16/16; FC3 12/12; MCP P4 5/5; `uv run ruff check` on write set; `git diff --check`; `uv run python -m scripts.governance.igx_gov ticket`.

## Out of scope

Production code, intergrax/** product changes, lockfile/CI/hooks, historical evidence directories (EC-W3-08/11), COMPAT-X / REPLAY-X operator decisions, formal G2/EQO_CLOSED.

## Lifecycle state

in_progress
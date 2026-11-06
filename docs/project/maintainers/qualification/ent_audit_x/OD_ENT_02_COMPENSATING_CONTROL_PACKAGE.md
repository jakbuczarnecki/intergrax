# OD-ENT-02 — Compensating control package (Option B)

**Operator disposition:** Option B — compensating controls acceptable **only** after independent audit acceptance; **no equivalence claim** to live branch protection without `gh` proof.

**OPEN-BP-01:** Live GitHub branch protection remains documented open backlog in [`ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md`](./ENT_AUDIT_X_ARCH_FREEZE_HANDOFF.md) until LIVE path or operator ARCH gate.

## Package contents

1. Threat model: [`OD_ENT_02_THREAT_MODEL.md`](./OD_ENT_02_THREAT_MODEL.md)
2. Local fail-closed gates: pre-push hooks, `igx_gov ticket`, mechanical readiness where applicable
3. CI required contexts documented in `.github/workflows/enterprise-audit.yml` (`ent-audit-fast`, `ent-audit-pr-gate`)
4. Independent audit session (IIAC) with manifest + handoff record
5. SHA-before-publish discipline for closure commits
6. Recorded limitations + auditor verdict on compensating scope
7. ECAP / roadmap / FRZ / preflight updates **before** compensating path use (Session A draft)
8. Session D **REJECT** if material gaps (operator rule 10)

## Preflight semantics

When IIAC `IN_FORCE`, set `od_ent_02_mode` in [`ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json`](./ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json):

- `live` — requires [`verify_ent_audit_branch_protection.py`](../../../../scripts/governance/verify_ent_audit_branch_protection.py) exit 0
- `compensating` — requires sub-manifest `od_ent_02_compensating.v1.json` (when published) + [`verify_od_ent_02_compensating_control.py`](../../../../scripts/governance/verify_od_ent_02_compensating_control.py) exit 0 + auditor **ACCEPT**

## Fail-closed

No self-certification of compensating sufficiency by implementer alone. No statement that compensating equals live protection.

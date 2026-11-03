# ENT-AUDIT-X — Protocol 03 independent audit dossier (draft)

**Status:** READY FOR INDEPENDENT B6 MECHANICAL REVIEW — **no PASS declared by agent**

| Field | Value |
| --- | --- |
| Execution SSOT | [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](./ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md) |
| Audited IMPLEMENTATION_SHA | `d4974efc255bf6ff6ca96e00fb084293ea685d66` |
| Plan mechanical validator | `scripts/enterprise_audit/validate_post_implementation_master_plan.py` (exit 0 required before B6 dossier fill) |
| TEST_EXECUTION_SHA | *operator pins at qual replay — not historical `0cee41a8`* |
| EVIDENCE_CANDIDATE_SHA | *unset until L3 green bundle* |
| EVIDENCE_PUBLICATION_SHA | *unset* |
| B6_ACCEPTANCE_PUBLICATION_SHA | *external auditor* |
| ACCEPTED_BASELINE_SHA | *operator OD-ENT-01 after B6* |

## Scope

- Tier-0 package `intergrax/platform/enterprise_audit/`
- CLI `scripts/enterprise_audit/run_enterprise_audit.py`
- Qualification tree `tests/qualification/ent_audit_x/` (164+ tests)
- Workflows `.github/workflows/enterprise-audit*.yml`

## Evidence pointers

- Corrective implementation plan: [ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md](./ENT_AUDIT_X_FINAL_CORRECTIVE_IMPLEMENTATION_AND_B6_READINESS_PLAN.md)
- SHA ladder: [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md)
- Gap register: [ANNEX_F_GAP_REGISTER.md](./ANNEX_F_GAP_REGISTER.md)
- CI runbook: [ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md](./ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md)

## Mechanical preflight

Run:

```bash
uv run python scripts/enterprise_audit/mechanical_readiness_check.py
```

## Operator boundaries (F-07)

- Baseline **promotion** requires operator OD-ENT-01 record (`promote-baseline` CLI).
- Agent/bootstrap may write **candidate** baseline only.
- Independent B6 issues PASS/FAIL; this dossier does not substitute external attestation.

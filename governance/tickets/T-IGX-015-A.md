# T-IGX-015-A — ENT-AUDIT-X formal certification and closure (F0–F7)

## Scope

Formal closure campaign **ENT-AUDIT-X-FORMAL-CERTIFICATION-AND-CLOSURE** on branch `development`:

- Publish [ENT_AUDIT_X_COMPLETE_FORMAL_CERTIFICATION_AND_CLOSURE_MASTER_PLAN.md](../../docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_COMPLETE_FORMAL_CERTIFICATION_AND_CLOSURE_MASTER_PLAN.md) (orchestration SSOT).
- Supersede narrative pointer on [ENT_AUDIT_X_FINAL_COMPLETION_AND_CERTIFICATION_MASTER_PLAN.md](../../docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_FINAL_COMPLETION_AND_CERTIFICATION_MASTER_PLAN.md).
- Operator / auditor **preparation** packets, terminal report, governance verifiers (`verify_ent_audit_branch_protection.py`, `ent_audit_x_formal_closure_preflight.py`).
- Reconcile [ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json](../../docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json) campaign fields (no fabricated B6/OD/FRZ PASS).

**Technical SSOT unchanged:** [ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md](../../docs/project/maintainers/qualification/ent_audit_x/ENT_AUDIT_X_POST_IMPLEMENTATION_FINAL_CLOSURE_MASTER_PLAN.md).

## Plan digest

Plan digest: `0a4dcd0f11af322abbbef931804c62dac656c01b8535eb79dbf4848d722b139d`

Source: Cursor plan `ent-audit-x_formal_closure_97253736.plan.md`

## Approved

Approved: operator @ 2026-11-05T12:00:00Z (formal closure campaign F0–F7)

## Test contract

```powershell
uv run pytest tests/qualification/ent_audit_x/unit/test_b5_publication_verifier.py tests/qualification/ent_audit_x/unit/test_mechanical_readiness.py -q
uv run python scripts/enterprise_audit/mechanical_readiness_check.py
uv run python scripts/qualification/ent_audit_x_formal_closure_preflight.py  # expect FAIL until authority
uv run python scripts/governance/verify_ent_audit_branch_protection.py       # expect FAIL until OD-ENT-02
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Re-opening W-A..W-E implementation, B3 88-test semantic re-run, B4 mechanical wave, FULL/ADV31/nightly without trigger.
- Fabricated Protocol 03 B6 PASS, `auditor_id`, OD signatures, FRZ mass PASS, `FORMAL_CLOSURE_PUBLICATION_SHA` in same commit that creates it (EBH-7).
- ARCH-FREEZE stage implementation; EVOL/ENT B6 human signatures by agent.

## Lifecycle state

quality_gate

## Checkpoint

(pending — paste `igx_gov ticket` stdout before push)

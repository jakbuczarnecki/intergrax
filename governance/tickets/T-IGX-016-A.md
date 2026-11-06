# T-IGX-016-A — IIAC governance implementation (Session A: G0+G1+G2+G3)

## Scope

**INTEGRAX Internal Independent Agent Certification (IIAC)** governance reform on branch `development`:

- G0: `generate_iiac_authority_impact_matrix.py` → `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`
- G1: IIAC protocol, schemas, activation template, migration matrix, OD-ENT-02 compensating package, ECAP/FRZ/roadmap/cross-stage amendments
- G2: Handoff + manifest fields documented; optional example manifest (not a PASS record)
- G3: `verify_iiac_*`, `verify_od_ent_02_compensating_control`, evol/ent preflight IIAC profile gate, negative tests N-01..N-12
- Session handoff instructions (B/C/D), publication contract, Session A terminal report

Supersedes **authority narrative** of [T-IGX-015-A.md](./T-IGX-015-A.md) for external-human B6 requirement; preserves F0–F7 artifact history on 015-A.

## Plan digest

Plan digest: `a2e4e09011bc54b985784125bed85e30da0cfe11b2ee736da130a17b5dc09a6f`

Source: Cursor plan `iiac_governance_reform_b138b2da.plan.md`

## Approved

Approved: operator Session A @ 2026-11-05T13:00:00+01:00 (IIAC-GOVERNANCE-REFORM)

## Test contract

```powershell
uv run pytest tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_iiac_g0_inventory.py -q
uv run python scripts/governance/verify_iiac_g0_inventory_completeness.py
uv run python scripts/governance/verify_iiac_policy_activation.py --expect-not-in-force
uv run python scripts/governance/verify_iiac_audit_manifest.py docs/project/maintainers/qualification/examples/iiac_audit_session_manifest.example.json
uv run python scripts/governance/verify_iiac_role_separation.py docs/project/maintainers/qualification/examples/iiac_audit_session_manifest.example.json
uv run python scripts/governance/verify_od_ent_02_compensating_control.py
uv run python scripts/qualification/evol_x_closure_publication_preflight.py
uv run python scripts/qualification/ent_audit_x_formal_closure_preflight.py
uv run python -m scripts.governance.igx_gov ticket
```

## Out of scope

- Policy activation `IN_FORCE`, Session B PASS, EVOL/ENT IIAC CLOSED claims, FRZ mass PASS
- Retroactive rewrite of historical human Protocol 03 PASS records
- ARCH-FREEZE stage implementation; Docker/vendor/production config changes (C10/C15)

## Lifecycle state

quality_gate

## Checkpoint

`igx_gov ticket`: OK @ F01 corrective push; `GOVERNANCE_IMPLEMENTATION_SHA=2ae1014116d353bc7742985e62f2bccc2bfac798`; `CORRECTIVE_IMPLEMENTATION_SHA=7d6061e48c6f84bbf1b05e34a28d4fd495602ae8`

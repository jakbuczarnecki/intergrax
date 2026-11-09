# ENT-AUDIT-X OD-ENT-02 — branch protection route (COMPENSATING primary)

**Decision ID:** OD-ENT-02  
**Stage:** ENT-AUDIT-X  
**Policy version:** IIAC-2026-11-05.1  
**Record origin:** operator-delegate (designated signing session `ent-sign-w3-2026-11-09`)

## Operator disposition

**Primary route:** `compensating` (per master plan §12 and operator chat 2026-11-08).  
**LIVE** remains a pre-wired alternative if repo admin enables required checks before W5.

This record **does not** claim equivalence between COMPENSATING controls and live GitHub branch protection. **`OPEN-BP-01`** remains registered open debt (owner: repo admin / ARCH-FREEZE; trigger: live protection enabled on `development`).

## Package basis

- `docs/project/maintainers/qualification/ent_audit_x/OD_ENT_02_COMPENSATING_CONTROL_PACKAGE.md` (refreshed 2026-11-09 signing session)
- `docs/project/maintainers/qualification/ent_audit_x/OD_ENT_02_THREAT_MODEL.md`
- Independent sufficiency: **Session D auditor** must accept compensating scope (W3-03 / plan §10 step 9) — **not** ruled by this signing session alone
- `od_ent_02_mode` set to `compensating` in `ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json`

## Controls (summary)

1. Ruleset evidence (deletion + non-fast-forward on `development`; no required ENT checks — documented gap)
2. CI workflow contexts documented (`ent-audit-fast`, `ent-audit-pr-gate`; nightly FULL evidence @ TES `b5c2481d…`, run `42593448899`)
3. Gate A/B closure preflight wiring (`ent_audit_x_formal_closure_preflight.py`)
4. IIAC independent audit PASS_SCOPED @ B6 pin `add6a6ce…`
5. `verify_od_ent_02_compensating_control.py` (real checks; must pass at publication SHA when compensating preflight runs)
6. **OPEN-BP-01** register entry (this record)

## Signature (S-2)

Approved OD-ENT-02: operator-delegate @ 2026-11-09T16:21:00+01:00

Delegation: docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/operator_decisions/ENT_AUDIT_X_O1_SIGNING_DELEGATION_RECORD.md@c7d18b61bb837a2708955da62adbe538b65e82bd

## Assurance

`process_enforced` — operator chat commission + designated session; auditor sufficiency verdict is a **separate** independent session (W4-02).

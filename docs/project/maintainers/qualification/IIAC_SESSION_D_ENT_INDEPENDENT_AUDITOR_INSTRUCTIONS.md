# IIAC Session D — ENT independent auditor instructions

**Precondition:** EVOL IIAC CLOSED manifest @ predecessor SHA; ENT TES/B5 pins from [`ENT_AUDIT_X_SHA_LADDER.md`](./ent_audit_x/ENT_AUDIT_X_SHA_LADDER.md).

**Role:** `independent-quality-auditor` — new chat.

## Scope

ENT-AUDIT-X + OD-ENT-02 LIVE or compensating package review.

## Execute

1. Protocol 03 semantics @ [`b6_independent_handoff/`](./ent_audit_x/b6_independent_handoff/) bundle.
2. OD-ENT-02: LIVE (`verify_ent_audit_branch_protection.py`) **or** compensating package + Session D sufficiency verdict.
3. Normal FREEZE qualification path @ SHA — not skipped.
4. FRZ-AUD-01..10 individual promotion; FRZ-AUD-10 = LIVE or COMPENSATING scoped PASS.
5. CLOSED publication + ARCH-FREEZE handoff with **OPEN-BP-01** if compensating.

## Reject

If compensating controls insufficient (operator rule 10 in master plan).

## Matrix C

Six rows C001,C002,C003,C042,C057,C065 — per [`IIAC_REQUIREMENT_MIGRATION_MATRIX.md`](./IIAC_REQUIREMENT_MIGRATION_MATRIX.md).

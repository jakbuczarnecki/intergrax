# EVOL-X — B6 to E4 execution sequence (W7-01 → W7-06)

**Authority:** Master execution plan · Normative: [EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md](../EVOL_X_FINAL_COMPLETION_MASTER_PLAN_v3.md) Batch G · Handoff prep: [b6_technical_qualification/EVOL_X_E4_W7_HANDOFF_DOSSIER.md](./b6_technical_qualification/EVOL_X_E4_W7_HANDOFF_DOSSIER.md).

**Pinned evidence (do not move without W7-02):**

| Role | SHA |
|------|-----|
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| Qualified CI | `41337077334` (290/290) |

---

## Phase 0 — Preconditions (Batch 1)

1. ECAP maintainer publication → `ECAP-P4-EVOL-SCOPE-AUTO-B6` **IN_FORCE** (Path B) **or** explicit Path A waiver documented in operator register.
2. Signed independent security review on file.
3. Independent key custody + published trust root (`custody_domain=INDEPENDENT_EXTERNAL`).
4. Attestor principal provisioned (read-only @ pinned SHAs).
5. Record OD-EVOL-11 auditor id **or** AUTO-B6 custodian in [EVOL_X_OPERATOR_DECISION_REGISTER_W1.md](./EVOL_X_OPERATOR_DECISION_REGISTER_W1.md).

**Verifier:** [EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json](./EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json) conjunction + [scripts/qualification/evol_x_b6_attestation.py](../../../../scripts/qualification/evol_x_b6_attestation.py).

---

## W7-01 — Formal B6 (`B6_ACCEPTANCE_PUBLICATION_SHA`)

| Step | Actor | Action | PASS criterion |
|------|-------|--------|----------------|
| 1 | Attestor / auditor | Confirm independence vs implementer | Signed attestation or Protocol 03 header |
| 2 | Attestor | Run orchestrator: Mode A @ FINAL, Mode B @ TES | `EVIDENCE_INTEGRITY_OK` |
| 3 | Attestor | Path B: sign production attestation; Path A: complete [EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md](./EVOL_X_B6_INDEPENDENT_AUDIT_PACKAGE.md) | Evaluator `ELIGIBLE` or signed PASS record |
| 4 | Authorized publisher | Commit `EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` (not draft filename) | Descendant of FINAL |
| 5 | Publisher | Update [EVOL_X_SHA_LADDER.md](./EVOL_X_SHA_LADDER.md) `B6_ACCEPTANCE_PUBLICATION_SHA` | Referential integrity |
| 6 | — | Close F-004; G-05, G-06, G-07, G-14 formal PASS | Findings register |

**Forbidden:** Implementer publishes PASS; agent self-appointment (G-02).

---

## W7-02 — Conditional corrective (only if triggered)

See [EVOL_X_CLOSURE_GATES_AND_BLOCKERS.md](./EVOL_X_CLOSURE_GATES_AND_BLOCKERS.md). **Current:** skip.

---

## W7-03 — FRZ + OD-EVOL-12

| Step | Actor | Action |
|------|-------|--------|
| 1 | Independent party | Confirm FRZ-EVOL-07 checklist wording supports qualification-only scoped PASS (Part X) → **Outcome A** or **Outcome B** |
| 2 | Operator | Sign **OD-EVOL-12** using [EVOL_X_OD_EVOL_12_DECISION_PACKAGE.md](./EVOL_X_OD_EVOL_12_DECISION_PACKAGE.md) |
| 3 | Operator | Edit [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md) FRZ-EVOL-01..10 per signed OD |
| 4 | Publisher | Commit checklist + register @ descendant of B6 SHA |

**Outcome B:** FRZ-EVOL-07 stays OPEN; stop promotion; document remediation path.

---

## W7-04 — OD-EVOLX-08

| Step | Actor | Action |
|------|-------|--------|
| 1 | Operator | Review [EVOL_X_OD_EVOLX_08_CLOSURE_PACKAGE.md](./EVOL_X_OD_EVOLX_08_CLOSURE_PACKAGE.md) |
| 2 | Operator | Sign OD-EVOLX-08; publish signature record |
| 3 | Publisher | Update operator register; close F-007 |

---

## W7-05 — Formal closure publication

| Step | Actor | Action |
|------|-------|--------|
| 1 | Operator | Authorize atomic publication pack [EVOL_X_W7_05_ATOMIC_PUBLICATION_PACK.md](./EVOL_X_W7_05_ATOMIC_PUBLICATION_PACK.md) |
| 2 | Publisher | Single commit (or two-phase SHA pin): roadmap §3/§5, ladder, governance `evol_x_closed: true`, closure index |
| 3 | Publisher | Record `FORMAL_CLOSURE_PUBLICATION_SHA` on ladder |
| 4 | — | Parent disposition: `CLOSED — INDEPENDENTLY ACCEPTED (AUTOMATION_FIRST_SCOPED) — PUBLISHED` |

**Pre-flight script:** `scripts/qualification/evol_x_closure_publication_preflight.py` (must exit 0 before apply).

---

## W7-06 — ENT-AUDIT-X handoff

| Step | Actor | Action |
|------|-------|--------|
| 1 | Publisher | Finalize [EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md](./EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md) @ descendant of closure SHA |
| 2 | B6 party | Attest digest chain (D10) |
| 3 | — | **Do not** implement ENT-AUDIT-X code in this wave |

---

## Separation of duties

| Function | Must not also |
|----------|----------------|
| Evidence producer (W6) | Sign B6 or promote FRZ |
| Implementer / Cursor | ECAP maintainer act, security sign-off, production key custody |
| Technical verifier scripts | Write governance `IN_FORCE` without maintainer record |
| Operator OD-EVOLX-08 | Substitute for independent B6 |

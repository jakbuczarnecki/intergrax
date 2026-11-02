# EVOL-X AUTO-B6 — full wave execution report (OD-EVOL-15)

**Disposition:** **`EVOL-X AUTO-B6 — TECHNICAL AND AUTHORITY PREPARATION COMPLETE / EXACT EXTERNAL BLOCKERS`**

**Not claimed:** ECAP `IN_FORCE`, independent security **signature**, production attestation, `B6_ACCEPTANCE_PUBLICATION_SHA`, FRZ promotion, OD-EVOL-12 / OD-EVOLX-08, EVOL-X CLOSED.

## 1. Git lineage

| Role | SHA |
|------|-----|
| START HEAD (wave) | `eef50d532a1f7ee52b17358712108786a543e991` |
| END HEAD | `88971400d1e0c79e69ef5bf7a0eb2a4332844f1f` |
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` (unchanged) |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` (unchanged) |
| Qualified CI | `41337077334` — 290 passed (unchanged; not re-dispatched) |
| B6_ACCEPTANCE_PUBLICATION_SHA | **— (unset)** |
| FORMAL_CLOSURE_PUBLICATION_SHA | **— (unset)** |

## 2. Decision register

| ID | Status |
|----|--------|
| OD-EVOL-13 | OPERATOR APPROVED — scoped M-09A |
| OD-EVOL-14 | OPERATOR APPROVED — direction + implementation only |
| OD-EVOL-15 | OPERATOR APPROVED — full AUTO-B6 execution within operator authority (this wave) |
| OD-EVOL-11 | Appointment approved; **auditor identity NOT NAMED** (G-02) |
| OD-EVOL-12 | **PENDING** |
| OD-EVOLX-08 | **PENDING** |

## 3. ECAP amendment

| Field | Value |
|-------|-------|
| ID | `ECAP-P4-EVOL-SCOPE-AUTO-B6` |
| Version | `2026-11-02.1-pending` |
| Activation | `PENDING_REQUIRED_APPROVALS` |
| Normative effect | **`NOT_IN_FORCE`** |
| Maintainer publication | **Absent** — disposition: **APPROVED DIRECTION / PENDING MAINTAINER PUBLICATION** |

## 4. Maintainer approval provenance

No ECAP maintainer publication act is recorded in-repo. Implementer commits do not activate the amendment.

## 5. Security review

| Item | Status |
|------|--------|
| Signed independent review | **ABSENT / PENDING** |
| Technical evidence package | [EVOL_X_AUTO_B6_SECURITY_REVIEW_TECHNICAL_EVIDENCE_UNSIGNED.md](./EVOL_X_AUTO_B6_SECURITY_REVIEW_TECHNICAL_EVIDENCE_UNSIGNED.md) (**UNSIGNED**) |
| Adversarial unit tests | **29 passed** (`tests/unit/qualification/test_evol_x_b6_attestation_contract.py`) |

## 6. Independent key custody

**Not provisioned.** `credential_class_authorized_for_formal_b6`: **null**. No production private key in this repository.

## 7. Trust root / runner / ledger

| Control | Status |
|---------|--------|
| Distinct attestor identity | Absent |
| External trust root | Absent |
| Append-only replay ledger (independent) | Absent |
| Verifier orchestration | Implemented; session JSON under `.tmp/session/evol-x-auto-b6-full-wave-v2/` (not formal audit ledger) |

Dossier: [../EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md](../EVOL_X_AUTO_B6_INDEPENDENT_ATTESTATION_DEPLOYMENT_DOSSIER.md).

## 8. AUTO-B6 qualification population

| Population | Location | Wave result |
|------------|----------|-------------|
| EVOL-X VL3 (290 nodes) | `tests/qualification/evol_x/` | **Unchanged** @ TES `3c4a641b…`; CI `41337077334` remains authoritative |
| Attestation + adversarial contract | `tests/unit/qualification/test_evol_x_b6_attestation_contract.py` | **29 passed** @ END HEAD |
| Mode B replay | Orchestrator | **Not re-run** (TES/FINAL/290 population unchanged; prior qualified replay + CI) |
| Mode A @ FINAL | Orchestrator | **PASS** after verifier fix (`uv run --directory <repo>` for P03 in nested worktree) |

## 9. Mode A/B and CI

| Check | Result |
|-------|--------|
| Mode A @ `bbf3e33…` | **PASS** (orchestrated; profile `AUTOMATION_FIRST_SCOPED`) |
| Mode B @ `3c4a641b…` | **Skipped this session** (no new TES; prior qualification + CI `41337077334`) |
| Adversarial invalid `--evidence-ref` | **PASS** (fail-closed) |
| Technical verdict (skip Mode B) | **INCOMPLETE** (expected) |
| Formal eligibility | **NOT_ELIGIBLE** |

## 10. Formal attestation evidence

**None.** Attestation verification: **ABSENT**. No `PRODUCTION_INDEPENDENT` signature.

## 11. B6 acceptance SHA

**Unset.** Protocol 03 PASS record not published.

## 12. FRZ-EVOL-01..10

All remain **OPEN** in [PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md](../../PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md). No promotion performed.

## 13. OD-EVOL-12 / OD-EVOLX-08

Both **PENDING**. W7-03 / W7-04 not executed.

## 14. E4 execution

**Not executed.** Prepared handoff remains [EVOL_X_E4_W7_HANDOFF_DOSSIER.md](./EVOL_X_E4_W7_HANDOFF_DOSSIER.md). Gates closed until formal B6.

## 15. Formal closure SHA

**Unset.** EVOL-X remains **OPEN**.

## 16. Exact external blockers and owners

| Blocker | Owner |
|---------|--------|
| ECAP maintainer publication → `IN_FORCE` | ECAP maintainer (not implementer) |
| Signed independent security review | Independent security reviewer |
| Independent key custodian + trust root + replay ledger | Operator-designated custodian / infrastructure owner |
| Name OD-EVOL-11 auditor or AUTO-B6 custodian in register | Operator |
| Production attestation + Protocol 03 PASS + `B6_ACCEPTANCE_PUBLICATION_SHA` | Independent attestor after gates above |
| OD-EVOL-12, FRZ batch W7-03 | Operator |
| OD-EVOLX-08, W7-04 | Operator |
| W7-05 closure + `FORMAL_CLOSURE_PUBLICATION_SHA` | Operator after B6 |
| W7-06 ENT-AUDIT-X handoff | Operator after closure |

## Authority matrix (A1 summary)

Organizational B6 acceptance requires separation of evidence producer, independent acceptance authority, and operator closure signatures under ECAP-P4. Automated technical PASS (Mode A/B) and implementer-run adversarial tests do **not** satisfy independent organizational B6 while amendment is `NOT_IN_FORCE` and custody is absent.

## Implementation note (this wave)

Fixed Mode A orchestration in `evol_x_b6_independent_verification.py`: P03 Mode A now runs via `uv run --directory <orchestrator repo>` with `PYTHONPATH` pointed at the FINAL worktree, avoiding nested-worktree interpreter/git mismatch on Windows.

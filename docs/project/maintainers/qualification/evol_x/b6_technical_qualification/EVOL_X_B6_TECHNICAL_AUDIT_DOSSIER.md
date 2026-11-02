# EVOL-X — B6 technical audit dossier (A5 consolidated)

> **DRAFT / UNSIGNED / NOT FORMAL B6 ACCEPTANCE**  
> Multi-agent Cursor technical review **≠** OD-EVOL-11 independent organizational acceptance.

## Executive disposition

**EVOL-X E3 — B6 TECHNICAL QUALIFICATION PASS / DURABLE EVIDENCE PUBLISHED / FORMAL INDEPENDENT B6 PENDING**

Parent **EVOL-X: OPEN**. No FRZ promotion. No `EVOL-X CLOSED`.

| Field | Value |
|-------|-------|
| Git START HEAD | `f5b23ffacdd5a9ecf3653d7988c338b5b705c2d5` |
| TEST_EXECUTION_SHA | `3c4a641b3e48a13f437e830a0562e2a80b03aabb` |
| EVIDENCE_FINAL_PUBLICATION_SHA | `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` |
| CI run | `41337077334` (success @ TES) |
| VL3 | 290/290 |
| CI JUnit SHA256 | `69c6a9ad9721b950008888dbf0d68852a35cc9e8b9df0ab5e3d895fabed15ea4` |
| Certification profile | `AUTOMATION_FIRST_SCOPED` |
| EVOL-M-09A | MEASURED / PASS |
| EVOL-M-09H | NOT_MEASURED |
| OD-EVOL-13 | OPERATOR APPROVED |

## Agent reports

| Agent | Verdict | Artifact |
|-------|---------|----------|
| A1 Provenance | PASS | [A1_PROVENANCE_AUDIT.md](./A1_PROVENANCE_AUDIT.md) |
| A2 Mode A/B | PASS (G-06 v2) | [A2_MODE_AB_RECONCILIATION.md](./A2_MODE_AB_RECONCILIATION.md) |
| A3 Adversarial | PASS (scoped) | [A3_ADVERSARIAL_CONTRACT_AUDIT.md](./A3_ADVERSARIAL_CONTRACT_AUDIT.md) |
| A4 FRZ/governance | E4 readiness; formal pending | [A4_FRZ_GOVERNANCE_READINESS.md](./A4_FRZ_GOVERNANCE_READINESS.md) |
| A5 Consolidator | See findings | [EVOL_X_B6_FINDINGS_REGISTER.md](./EVOL_X_B6_FINDINGS_REGISTER.md) |

## Mode A / Mode B (summary)

- **Mode A:** `EVIDENCE_INTEGRITY_OK` @ FINAL `bbf3e33b…` ([EVOL_X_B6_MODE_A_PROVENANCE.md](./EVOL_X_B6_MODE_A_PROVENANCE.md)).
- **Mode B v2:** 290 pytest PASS + `EVIDENCE_INTEGRITY_OK` @ TES with `--evidence-ref` FINAL ([EVOL_X_B6_MODE_B_REPLAY_PROVENANCE.md](./EVOL_X_B6_MODE_B_REPLAY_PROVENANCE.md)).
- **Invalid G-06 attempt:** F-001 preserved ([EVOL_X_B6_PROCEDURAL_FINDING_F001.md](./EVOL_X_B6_PROCEDURAL_FINDING_F001.md)).

## Formal blockers

1. G-02 — auditor identity not named  
2. G-07 — no signed B6 record / no `B6_ACCEPTANCE_PUBLICATION_SHA`  
3. G-08 — FRZ-EVOL-07 Outcome A/B not formally decided  
4. G-09 / G-10 — OD-EVOL-12, OD-EVOLX-08 pending  
5. W7-05 — cannot declare EVOL-X CLOSED until above resolved  

## E4

[EVOL_X_E4_W7_HANDOFF_DOSSIER.md](./EVOL_X_E4_W7_HANDOFF_DOSSIER.md)

## Draft formal record (unsigned)

[EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_RECORD_DRAFT.md](./EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_RECORD_DRAFT.md)

## Publication note

Durable evidence root: `docs/project/maintainers/qualification/evol_x/b6_technical_qualification/` (this tree). Session dossier `.tmp/session/evol-x-b6-multi/` remains supplementary.

**Git END HEAD (W7-01 publication):** `004c3a8d75fe0852eb2f403de5112b416bc56d09` — [EVOL_X_W7_01_FORMAL_INDEPENDENT_B6_ACCEPTANCE_PACKAGE.md](../EVOL_X_W7_01_FORMAL_INDEPENDENT_B6_ACCEPTANCE_PACKAGE.md)

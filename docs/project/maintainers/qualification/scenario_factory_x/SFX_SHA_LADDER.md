# SCENARIO-FACTORY-X — SHA ladder

**Source:** master plan R1 §11 ([`SFX-MASTER-PLAN.md`](../../../../../governance/tickets/plans/SFX-MASTER-PLAN.md)) · **Index:** [`SFX_INDEX.md`](./SFX_INDEX.md)

Rules:
- Every rung is a full 40-character SHA on `origin/development` (or, for external rungs, on the owning stage's ladder), recorded by a dedicated pin commit after the event, never by the commit itself.
- Each later rung descends from `BASE_SHA`; ancestry is verified with `git merge-base --is-ancestor` at P13-01.
- Evidence rungs (handoff HO-7, pilots) pin the exact SHA the evidence was produced at.
- `_unset_` means the event has not happened. No rung other than `BASE_SHA` is set at P0.

## Pre-freeze (W0–W2)

| Role | SHA | Notes |
| --- | --- | --- |
| BASE_SHA | `eadb1c2116feb0a2090b234a36fd4182bd6ace8d` | `origin/development` at campaign start; worktree `sfx-p0` cut |
| ROADMAP_AMENDMENT_SHA | _unset_ | Merge of the W0 PR carrying roadmap A1–A10 (T-IGX-037-A) |
| P0_REGISTRATION_SHA | _unset_ | Merge of the W0 PR (T-IGX-037-A); same value as `ROADMAP_AMENDMENT_SHA` when both land in one PR |
| P1_PUBLICATION_SHA | _unset_ | Merge of T-IGX-037-B (research, inventory re-pin) |
| P2_PUBLICATION_SHA | _unset_ | Merge of T-IGX-037-C (ADR-SFX-001..006 Proposed, contract drafts) |
| P3_PUBLICATION_SHA | _unset_ | Merge of T-IGX-037-D (UX, usability round 0) |
| GEN_SEAL_SHA | _unset_ | Commit that publishes the GEN set digest (SFX-P3-02); must precede W3 |
| P4_PUBLICATION_SHA | _unset_ | Merge of T-IGX-037-E (IDE toolkit v0, SFX-P4-A01/A02) |
| PRE_IMPLEMENTATION_READY_SHA | _unset_ | `origin/development` when all of P0–P4-A are published (OD-SFX-15 target) |

## Entry (external pins and gate)

| Role | SHA | Notes |
| --- | --- | --- |
| FREEZE_CERTIFICATION_SHA | _unset_ | ARCH-FREEZE certification (owned by the ARCH-FREEZE ladder); inventory re-pin target |
| INVENTORY_REPIN_SHA | _unset_ | Publication of the capability inventory re-pinned at `FREEZE_CERTIFICATION_SHA` |
| ADR_ACCEPTANCE_SHA | _unset_ | Publication of ADR-SFX-001..006 Accepted (OD-SFX-06) |
| GATE_SFX_ENTRY_SHA | _unset_ | Publication of the GATE-SFX-ENTRY record |
| GATE_SEF_CONSUME_REF | _unset_ | SEF-X ladder reference (GATE-SEF-FULL + scoped B6 T-P12-003) satisfying GATE-SEF-CONSUME (§3.3) |

## Post-freeze waves (W3–W8)

| Role | SHA | Notes |
| --- | --- | --- |
| W3_IMPLEMENTATION_SHA | _unset_ | Wave branch head (SFX-P4-B01/B02, SFX-P5-*) |
| W3_PUBLICATION_SHA | _unset_ | Merge of the W3 PR |
| REBIND_W3_AUDIT_EXECUTION_SHA | _unset_ | IIAC ACT-N11 `audit_execution_sha` for the W3 re-bind |
| REBIND_W3_ACTIVATION_SHA | _unset_ | IIAC ACT-N11 re-bind activation after W3 |
| W4_IMPLEMENTATION_SHA | _unset_ | Wave branch head (SFX-P6..P8) |
| W4_PUBLICATION_SHA | _unset_ | Merge of the W4 PR |
| REBIND_W4_AUDIT_EXECUTION_SHA | _unset_ | ACT-N11 audit execution, W4 |
| REBIND_W4_ACTIVATION_SHA | _unset_ | ACT-N11 re-bind activation, W4 |
| W5_IMPLEMENTATION_SHA | _unset_ | Wave branch head (SFX-P9, SFX-P10) |
| W5_PUBLICATION_SHA | _unset_ | Merge of the W5 PR |
| REBIND_W5_AUDIT_EXECUTION_SHA | _unset_ | ACT-N11 audit execution, W5 |
| REBIND_W5_ACTIVATION_SHA | _unset_ | ACT-N11 re-bind activation, W5 |
| W6_IMPLEMENTATION_SHA | _unset_ | Wave branch head (SFX-P11) |
| W6_PUBLICATION_SHA | _unset_ | Merge of the W6 PR |
| REBIND_W6_AUDIT_EXECUTION_SHA | _unset_ | ACT-N11 audit execution, W6 |
| REBIND_W6_ACTIVATION_SHA | _unset_ | ACT-N11 re-bind activation, W6 |
| W7_IMPLEMENTATION_SHA | _unset_ | Wave branch head (SFX-P12) |
| W7_PUBLICATION_SHA | _unset_ | Merge of the W7 PR |
| REBIND_W7_AUDIT_EXECUTION_SHA | _unset_ | ACT-N11 audit execution, W7 |
| REBIND_W7_ACTIVATION_SHA | _unset_ | ACT-N11 re-bind activation, W7 |
| W8_IMPLEMENTATION_SHA | _unset_ | Wave branch head (SFX-P13) |
| W8_PUBLICATION_SHA | _unset_ | Merge of the W8 PR |
| REBIND_W8_AUDIT_EXECUTION_SHA | _unset_ | ACT-N11 audit execution, W8 |
| REBIND_W8_ACTIVATION_SHA | _unset_ | ACT-N11 re-bind activation, W8 |

## Pilot evidence (W7)

| Role | SHA | Notes |
| --- | --- | --- |
| REF_1_EVIDENCE_SHA | _unset_ | REF-1 bundle, both modes (SFX-P12-01) |
| REF_2_EVIDENCE_SHA | _unset_ | REF-2 bundle, both modes (SFX-P12-02; scope per OD-SFX-08) |
| REF_3_EVIDENCE_SHA | _unset_ | REF-3 bundle, both modes (SFX-P12-03; scope per OD-SFX-08) |
| GEN_REVEAL_SHA | _unset_ | GEN set reveal (SFX-P12-04); content digest must equal the `GEN_SEAL_SHA` seal |
| GEN_RUN_SHA | _unset_ | Factory SHA of the GEN run and metrics G-1..G-5; any later factory change requires a re-run |
| PERFORMANCE_MEASUREMENT_SHA | _unset_ | Performance metrics M-1..M-9 (SFX-P12-06) |

## Closure (W8)

| Role | SHA | Notes |
| --- | --- | --- |
| TEST_EXECUTION_SHA | _unset_ | Audited SHA of the closure qualification replay (no drift to audit) |
| AUDIT_SHA | _unset_ | Independent audit (SFX-P13-02/03) |
| AUDIT_EXECUTION_SHA | _unset_ | IIAC `audit_execution_sha` for the stage certification |
| CERTIFICATION_SHA | _unset_ | IIAC independent product certification (SFX-P13-03) |
| EVIDENCE_PUBLICATION_SHA | _unset_ | Evidence publication (SFX-P13-01) |
| FORMAL_CLOSURE_SHA | _unset_ | Operator closure OD-SFX-11 + roadmap row update (SFX-P13-04) |
| FINAL_REMOTE_HEAD | _unset_ | Head verified after closure publication |

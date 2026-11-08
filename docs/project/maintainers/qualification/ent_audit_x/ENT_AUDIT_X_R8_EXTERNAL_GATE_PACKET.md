# ENT-AUDIT-X — R8 external gate stop (operator packet)

> **Status overlay (2026-11-08, `T-IGX-024-A`):** closure authority for ENT-AUDIT-X is now IIAC (`IIAC-2026-11-05.1`, IN_FORCE, activation `f8c29fb8dfe5e14d9b06ff16659fd9cbe9ad326d`). Statements below that assume a human Protocol 03 auditor, a named external auditor, or "ECAP-P4 NOT IN FORCE" are historical and **superseded**; this text is preserved unchanged. Current status: [ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md](ENT_AUDIT_X_IIAC_CLOSURE_MATRIX.md).

**Agent stop line:** TECHNICAL work through R7 complete; **do not** declare B6 PASS, OD approved, FRZ CLOSED, or ENT stage FORMALLY CLOSED.

## Current pins (W8 corrective — active qualification)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` |
| EVIDENCE_PUBLICATION_SHA | `300859537f12a1de7bf3d75fd685bc3a5c727183` |
| INDEPENDENT_TECHNICAL_REAUDIT_SHA | `0e538269a7188075003b9ff2c7cce2545d8f1ccb` |
| B5 bundle (current) | [qualification_at_89c76eb7…/](./b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/) |

## Historical pins (R6/R7 defect record — not current B6 evidence)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `87203c1b6ce9786dce0523cca29b556106b6bc8d` |
| EVIDENCE_PUBLICATION_SHA | `d18f01f2291328a073dcbdfbe05489faf98013b7` |
| B5 bundle | [qualification_at_87203c1b…/](./b6_independent_handoff/qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d/) |

## Required operator sequence (ECAP PL-08)

1. **OD-ENT-00** — EVOL-X formally CLOSED or documented exception (`BLK-ROADMAP-ENT-DEP`).
2. **Independent Protocol 03 B6** — auditor reviews dossier + B5 @ NEW_TES (`BLK-ENT-B6`). Agent does not sign PASS.
3. **OD-ENT-01** — operator record + `ACCEPTED_BASELINE_SHA` after B6.
4. **OD-ENT-02** — [OD_ENT_02_OPERATOR_PACKET.md](./OD_ENT_02_OPERATOR_PACKET.md) branch protection + required checks @ TES.
5. **OD-ENT-08** + **FRZ-AUD-01..10** promotion — formal ENT-AUDIT-X closure + roadmap §5.

## Evidence paths (session)

- Certifying qual: `.tmp/session/ent-audit-x-r6-certifying/qual_wave.log`
- Pytest R2: `.tmp/session/ent-audit-x-r2/pytest-post-r1fix.log`
- B5 publish: `.tmp/session/ent-audit-x-r7/publish.log`

## Remaining automated blockers @ HEAD (non-external)

- Nightly CI proof @ NEW_TES commit (dispatch run `41686980919` may still be in progress).
- `qualification_current_post_tes` clears after operator accepts NEW_TES as current qual baseline in readiness config (post-R7 doc pin).

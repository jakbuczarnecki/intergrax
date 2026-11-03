# ENT-AUDIT-X — R8 external gate stop (operator packet)

**Agent stop line:** TECHNICAL work through R7 complete; **do not** declare B6 PASS, OD approved, FRZ CLOSED, or ENT stage FORMALLY CLOSED.

## Current pins (post R6/R7)

| Role | SHA |
| --- | --- |
| TEST_EXECUTION_SHA | `87203c1b6ce9786dce0523cca29b556106b6bc8d` |
| EVIDENCE_PUBLICATION_SHA | See latest append in [ENT_AUDIT_X_SHA_LADDER.md](./ENT_AUDIT_X_SHA_LADDER.md) |
| B5 bundle | [b6_independent_handoff/qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d/](./b6_independent_handoff/qualification_at_87203c1b6ce9786dce0523cca29b556106b6bc8d/) |

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

# ENT-AUDIT-X — Operator decision and external authority packet

**Campaign:** `ENT-AUDIT-X-FORMAL-CERTIFICATION-AND-CLOSURE`  
**Audience:** Operator · independent auditor · GitHub admin  
**Agent status:** preparation only — **no signatures applied in this batch**

## Signature methods

| Actor | Method | Artifact |
| --- | --- | --- |
| Operator (OD-*) | Signed git commit on `development` with decision record path cited | `operator_records/` or EVOL/ENT OD markdown packages |
| Independent auditor (Protocol 03 B6) | Non-draft PASS record file + ladder SHA pin commit | `EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` · `ENT_AUDIT_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` |
| GitHub admin (OD-ENT-02) | Branch protection / ruleset + evidence JSON path in OD record | `gh api repos/jakbuczarnecki/intergrax/branches/development/protection` |

**REJECT consequence:** formal disposition unchanged; append blockers; no roadmap `CLOSED — INDEPENDENTLY ACCEPTED` strings.

---

## F1 — EVOL predecessor (blocking ENT formal chain)

### OD-EVOL-11 — auditor appointment

| Item | Detail |
| --- | --- |
| Status | **SIGNED** appointment · **`auditor_id` NOT RECORDED** |
| Record | [EVOL_X_OD_EVOL_11_INDEPENDENT_AUDITOR_APPOINTMENT_RECORD.md](../evol_x/EVOL_X_OD_EVOL_11_INDEPENDENT_AUDITOR_APPOINTMENT_RECORD.md) |
| Operator actions | 1) Set `auditor_id` + display name in register + appointment record. 2) Ensure auditor completes independence declaration. |
| Agent | **FORBIDDEN** to fabricate identity |

### EVOL Protocol 03 B6

| Item | Detail |
| --- | --- |
| Dossier | [EVOL_X_B6_FINAL_AUDIT_DOSSIER_FOR_INDEPENDENT_SIGNATURE.md](../evol_x/EVOL_X_B6_FINAL_AUDIT_DOSSIER_FOR_INDEPENDENT_SIGNATURE.md) |
| Pins | TES `3c4a641b3e48a13f437e830a0562e2a80b03aabb` · FINAL `bbf3e33bc62a05fd205e07f3d2e35a6d1d689ca1` · CI `41337077334` |
| Publication | Pin `B6_ACCEPTANCE_PUBLICATION_SHA` on [EVOL_X_SHA_LADDER.md](../evol_x/EVOL_X_SHA_LADDER.md) |
| Preflight | `uv run python scripts/qualification/evol_x_closure_publication_preflight.py` → exit **0** before W7-05 |

### OD-EVOL-12 · OD-EVOLX-08 · W7-05 / W7-06

| Decision | Package |
| --- | --- |
| OD-EVOL-12 | [EVOL_X_OD_EVOL_12_DECISION_PACKAGE.md](../evol_x/EVOL_X_OD_EVOL_12_DECISION_PACKAGE.md) |
| OD-EVOLX-08 | [EVOL_X_OD_EVOLX_08_CLOSURE_PACKAGE.md](../evol_x/EVOL_X_OD_EVOLX_08_CLOSURE_PACKAGE.md) |
| W7-05 atomic pack | [EVOL_X_W7_05_ATOMIC_PUBLICATION_PACK.md](../evol_x/EVOL_X_W7_05_ATOMIC_PUBLICATION_PACK.md) |
| W7-06 handoff | [EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md](../evol_x/EVOL_X_ENT_AUDIT_X_HANDOFF_RECORD.md) |

---

## F2 — ENT Protocol 03 B6

| Item | Detail |
| --- | --- |
| Draft package | [ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md](./ENT_AUDIT_X_B6_FORMAL_ACCEPTANCE_PACKAGE_DRAFT.md) |
| Dossier | [ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md](./ENT_AUDIT_X_PROTOCOL_03_DOSSIER.md) |
| Handoff tree | `b6_independent_handoff/qualification_at_89c76eb7b53a0a3ce6f75ab395cf4a2802638a91/` |
| Pins | TES `89c76eb7b53a0a3ce6f75ab395cf4a2802638a91` · B5 `300859537f12a1de7bf3d75fd685bc3a5c727183` |
| Auditor replay | `verify_b5_publication_tree` · Mode A/B per dossier · **6** Matrix C rows operator-gated until OD-ENT-01 |
| Publication | Signed `ENT_AUDIT_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md` + `B6_ACCEPTANCE_PUBLICATION_SHA` |

**Prerequisite:** EVOL formal **`CLOSED — INDEPENDENTLY ACCEPTED`** for Mode B parent (OD-ENT-00).

---

## F3 — OD-ENT-02 branch protection

| Item | Detail |
| --- | --- |
| Runbook | [ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md](./ENT_AUDIT_X_CI_ENFORCEMENT_RUNBOOK.md) |
| Operator packet | [OD_ENT_02_OPERATOR_PACKET.md](./OD_ENT_02_OPERATOR_PACKET.md) |
| Required contexts (minimum) | `Enterprise audit (ENT-AUDIT-X) / ent-audit-fast` · `Enterprise audit (ENT-AUDIT-X) / ent-audit-pr-gate` |
| Verifier | `uv run python scripts/governance/verify_ent_audit_branch_protection.py` |
| Campaign F0 API | **404 Branch not protected** (see `.tmp/session/ENT-AUDIT-X-FORMAL-CLOSURE/gh_branch_protection.json`) |
| Negative test | PR with intentional audit failure → merge blocked |

---

## F4 — Operator baseline

| Decision | When | Action |
| --- | --- | --- |
| OD-ENT-00 | After EVOL formal CLOSED | Accept predecessor per [ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md](../ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md) |
| OD-ENT-01 | After ENT B6 ACCEPT | JSON under [operator_records/](./operator_records/) + `promote_accepted_baseline` |
| Matrix C | Post OD-ENT-01 | C001,C002,C003,C042,C057,C065 → QUALIFIED if in scope; close GAP-RC-OD-TRUST |

---

## F5 — FRZ-AUD-01..10

Per [ANNEX_B_FRZ_AUD_CLOSURE_MATRIX.md](./ANNEX_B_FRZ_AUD_CLOSURE_MATRIX.md) after certifying FREEZE @ accepted baseline. **FRZ-AUD-10** requires live OD-ENT-02 proof.

---

## F6 — OD-ENT-08 formal stage closure

Preflight: `uv run python scripts/qualification/ent_audit_x_formal_closure_preflight.py` (exit **0** required).

Atomic publication (operator/maintainer — **not** this agent batch):

- Roadmap ENT row → `CLOSED — INDEPENDENTLY ACCEPTED — PUBLISHED`
- `FORMAL_CLOSURE_PUBLICATION_SHA` on ladder (**EBH-7:** pin in follow-up commit, not same commit that first creates closure docs)
- [ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json](./ENT_AUDIT_X_CLOSURE_EXECUTION_STATUS.json) formal fields

---

## Internal-only (does not satisfy formal gates)

[EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md](../EVOL_ENT_INTERNAL_CERTIFICATION_RECORD_2026_11_04.md) (OD-IC-01)

Ref: governance/tickets/T-IGX-015-A.md

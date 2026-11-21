# EQO wave 12 — parallel non-terminal execution manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/` (+ optional ticket checkpoint line).

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `292ca3348` (PR #704, post wave-6).

## Parallel lanes

| Lane | Branch | Worktree | Deliverable | Commit |
|------|--------|----------|-------------|--------|
| **G-W2** | `eqo/wave-12/w2` | `eqo-w12-w2` | #680 track + W2-02 read @ post wave-6 land | `a4fd55ade` |
| **G-X** | `eqo/wave-12/x` | `eqo-w12-x` | X WBS disposition read memos | `feabdfc70` |
| **G-W9** | `eqo/wave-12/w9` | `eqo-w12-w9` | C01–C25 sweep; W9-01 prep (no `EQO_CLOSED`) | `23bbead01` |
| **G-INT** | `eqo/wave-12/integ` | `eqo-wv-w12-integ` | This manifest, README index, lane merges, PR → `development` | _(integrator tip)_ |

**Out of scope:** pytest EXE, FRZ promotion, product edits, AUD verdict changes, `EQO_CLOSED` / G6 / IIAC IN_FORCE claims.

## #680 (track only)

No new ticket; gap in `EQO_W2_TRAIN_VERIFICATION.md` + `pr680_governance_fast.md`. C16 unchanged.

## Merge ladder

```text
eqo/wave-12/w2  ──┐
eqo/wave-12/x   ──┼──► eqo/wave-12/integ ──► development (PR; docs gates)
eqo/wave-12/w9  ──┘
```

**Docs gates (integrator PR):** `ticket-governance` + `governance-policy-pin` per T-IGX-046-F merge policy; do not block on full platform CI when pre-existing drift is outside write set.

All qualification commits: trailer `Ref: governance/tickets/T-IGX-046-F.md`.
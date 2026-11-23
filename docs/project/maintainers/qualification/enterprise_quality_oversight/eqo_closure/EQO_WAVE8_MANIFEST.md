# EQO wave 8 — parallel non-terminal execution manifest

**Generated:** 2026-11-21  
**Ticket Ref:** `governance/tickets/T-IGX-046-F.md`  
**Write set:** `docs/project/maintainers/qualification/enterprise_quality_oversight/eqo_closure/` (+ optional ticket checkpoint line).

| Pin | SHA |
|-----|-----|
| DOC_PIN | `8cee3c5a913f234e8fb65053b6ab4f34ed78249b` |
| TRAIN_PIN | `28c27cb896d02e0185b9fa02047411fa152f6469` |

**Merge base:** `origin/development` @ `569254840` (PR #698).

## Parallel lanes

| Lane | Branch | Worktree | Deliverable | Commit |
|------|--------|----------|-------------|--------|
| **G-W2** | `eqo/wave-8/w2` | `eqo-w8-w2` | W2-02 replay doc read; #680 on rw-train, not on `development` | `a8fde582c` |
| **G-X** | `eqo/wave-8/x` | `eqo-w8-x` | X WBS read memos English refresh | `fe3a92adc` |
| **G-W9** | `eqo/wave-8/w9` | `eqo-w8-w9` | C01–C25 partial; W9-01 prep (no `EQO_CLOSED`) | `b53793858` |
| **G-INT** | `eqo/wave-8/integ` | `eqo-wv-w8-integ` | This manifest, README index, lane merges, PR → `development` |

**Out of scope:** pytest EXE, FRZ promotion, product edits, AUD verdict changes, `EQO_CLOSED` / G6 / IIAC IN_FORCE claims.

## #680 (track only)

No new ticket; gap documented in `EQO_W2_TRAIN_VERIFICATION.md` + `pr680_governance_fast.md` (English refresh). C16 unchanged.

## Merge ladder

```text
eqo/wave-8/w2  ──┐
eqo/wave-8/x   ──┼──► eqo/wave-8/integ ──► development (PR; docs gates)
eqo/wave-8/w9  ──┘
```

All qualification commits: trailer `Ref: governance/tickets/T-IGX-046-F.md`.
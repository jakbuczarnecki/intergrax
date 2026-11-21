# EQO-W2 — Train ticket read-only verification @ TRAIN_PIN

**Pin:** `28c27cb896d02e0185b9fa02047411fa152f6469` (`origin/arch-freeze/rw-train`, tip merge #679)  
**Method:** `git merge-base --is-ancestor`, `git log --grep`, ticket file presence on pin (no product edits).  
**Executed:** 2026-11-21 UTC.

## Summary

| Task | Ticket / PR | Merge on TRAIN_PIN? | EQO verification status | Notes |
|------|-------------|---------------------|-------------------------|-------|
| W2-01 | T-IGX-043-A / #680 | **No** | `PR_OPEN` | Not ancestor of `28c27cb`; draft `af2/pt-fc1`; governance-fast FAIL (W0-03). |
| W2-02 | T-IGX-043-G / #642 | **Yes** `05699b633cdb` | `MERGED_GATE_PENDING` | Tier-boundary work merged; EQO-W3-WB05/pytest must replay import gates. |
| W2-02 | T-IGX-043-H / #654 | **Yes** `6b94d202fb79` | `MERGED_GATE_PENDING` | Paired with 043-G; same gate replay obligation. |
| W2-03 | T-IGX-043-H scope | n/a | `SCOPE_OK` | Merge #654 on pin; Part G / A4-040 not asserted in this diff pass (held W2-13). |
| W2-04 | T-IGX-043-D / #668 | **Yes** `33a91bfdd` | `MERGED_UNREPLAYED` | Skill pin durability; live/Redis gap = `LIVE_QUALIFICATION_ONLY`. |
| W2-05 | T-IGX-043-I / #669 | **Yes** `c2a88c344` | `MERGED_PARTIAL` | Parts A–D on pin; Part E ticket `T-IGX-043-M` **ABSENT** (W0-02). |
| W2-06 | T-IGX-051-D / #670 | **Yes** `cc5d2d0bd` | `MERGED_UNREPLAYED` | SDK parity limits documented in ticket; closure needs WB09 EXE. |
| W2-07 | T-IGX-043-B / #672 | **Yes** `f262edce6` | `MERGED_UNREPLAYED` | Part 1 fail-closed; part 2 sandbox = B4 / `NOT_STARTED`. |
| W2-07 | T-IGX-043-E / #677 | **Yes** `475e34cde` | `MERGED_PARTIAL` | Part 1 on pin; part 2 out of #677. |
| W2-08 | T-IGX-044-F / #673 | **Yes** `9b7eca0e9` | `MERGED_UNREPLAYED` | Material-governance path validator present; not IIAC IN_FORCE. |
| W2-09 | T-IGX-051-E / #674 | **Yes** `e7dd2f0bc` | `MERGED_PARTIAL` | Drain/CAS merge; F-1 residual (`NOT_STARTED`, W2-12). |
| W2-10 | T-IGX-051-I / #675 | **Yes** `4c49cf877e65` | `MERGED_PARTIAL` | A4-007 caller-refusal tests in merge; residual EDER-N-0003 (W0-04). |
| W2-11 | T-IGX-051-B / #676 | **Yes** `37fd3e984` | `MERGED_PARTIAL` | Parts C–I; A/B and F-5 (R-1..R-8) not on pin. |
| W2-12 | Record-2 F-1,B3,B4,F-4,F-5 | **No** | `NOT_MERGED` | Ticket files for F-4 (`T-IGX-043-N`), F-5, B3, B4 absent on pin (W0-02). |
| W2-13 | F-3, 052-E, Part G, A4-040 | **No** | `WAIT_FOR_OPERATOR` | OD-EQO-04; `T-IGX-052-E` absent on pin. |

## Evidence commands (reproducible)

```text
git merge-base --is-ancestor 05699b633cdb 28c27cb896d02e0185b9fa02047411fa152f6469
git log -1 --format=%H 28c27cb896d02e0185b9fa02047411fa152f6469
git cat-file -e 28c27cb896d02e0185b9fa02047411fa152f6469:governance/tickets/T-IGX-043-G.md
```

## EQO disposition

- **Train integration:** fourteen merge commits (#642–#677, #679) are ancestors of `TRAIN_PIN`; **#680 is not**.
- **Closure:** W2 does not certify findings; it establishes merge presence vs `EQO-W3` replay and `EQO-X` blockers (A4-003/A4-004 gate replay per W2-02).
- **Exact blockers carried forward:** #680 open; record-2 items not merged; Operator band G items W2-13.
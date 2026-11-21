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

## Wave-4 read pass (W2-02..W2-11, docs-only)

**Executed:** 2026-11-21 (lane `eqo/wave-4/w2`) · re-read ticket paths on `TRAIN_PIN` only; no pytest, no train write set.

| Task | Read focus | Wave-4 disposition |
|------|------------|-------------------|
| W2-02 | T-IGX-043-G/H merge SHAs on pin; tier-boundary scope | Unchanged `MERGED_GATE_PENDING`; ties `EQO-X-01` replay queue (AUD session E) |
| W2-03 | #654 scope note; Part G held | `SCOPE_OK`; W2-13 still `WAIT_FOR_OPERATOR` |
| W2-04 | T-IGX-043-D / #668 | `MERGED_UNREPLAYED`; skill durability → WB09/W4-04 REQUALIFY links |
| W2-05 | T-IGX-043-I / #669 Parts A–D | `MERGED_PARTIAL`; Part E absent (W0-02) |
| W2-06 | T-IGX-051-D / #670 | `MERGED_UNREPLAYED`; WB09 SDK slice |
| W2-07 | T-IGX-043-B/E #672/#677 | `MERGED_PARTIAL`; B4 part 2 `NOT_STARTED` |
| W2-08 | T-IGX-044-F / #673 | `MERGED_UNREPLAYED`; WB01 governance slice |
| W2-09 | T-IGX-051-E / #674 | `MERGED_PARTIAL`; F-1 residual W2-12 |
| W2-10 | T-IGX-051-I / #675 | `MERGED_PARTIAL` @ `4c49cf877e65`; A4-007 tests on pin; EDER-N-0003 residual (W0-04) |
| W2-11 | T-IGX-051-B / #676 | `MERGED_PARTIAL` @ `37fd3e984`; Parts C–I on pin; A/B + F-5 R-1..R-8 absent |

Repro (wave-4): `git merge-base --is-ancestor <merge_sha> 28c27cb896d02e0185b9fa02047411fa152f6469` for rows above.

## #680 gap track (no new ticket)

| Item | Status (wave-4) | Status (wave-8) | Status (wave-12) |
|------|-----------------|-----------------|------------------|
| PR #680 / T-IGX-043-A | Still **not** ancestor of `TRAIN_PIN` `28c27cb`; draft branch `af2/pt-fc1` | **Unchanged** @ pin `28c27cb` | Merge `65ef1f196` **is** ancestor of rw-train @ `8e146e21` via `dbaebfa8`; **not** on `development` @ `a0598ba3` |
| `governance-fast` class | KR-08 / W0-03 | Track-only memo | **Track-only** — `pr680_governance_fast.md`; no merge / waiver |
| C16 matrix | `#680 open` — no waiver, no merge claim | **Unchanged** — no merge claim | **Unchanged** |
| New ticket | **None** (operator band D out of EQO write set) | **None** | **None** |

## Wave-8 read pass (W2-02 replay doc, docs-only)

**Executed:** 2026-11-21 · lane `eqo/wave-8/w2` · **Land:** `569254840` (`origin/development`, PR #698). Pins unchanged (`TRAIN_PIN` `28c27cb`, `DOC_PIN` `8cee3c5`).

| Task | Read focus | Wave-8 disposition |
|------|------------|-------------------|
| W2-02 | `T-IGX-043-G.md` / `T-IGX-043-H.md` present on `TRAIN_PIN`; merge SHAs `05699b633cdb`, `6b94d202fb79` | **`MERGED_GATE_PENDING`** — tier-boundary import gates not replayed on pin |
| W2-02 replay queue | `runs/wb05/AUD_SESSION_E.md` (AUD-1 **BLOCKED**); `x/A4-003/A4_GATE_COUNTER_READ.md` | Gate replay still **queued**; ties `EQO-X-01` + session E WB05 — not closed |
| W2-02 evidence | No pytest EXE this wave; read-only ticket paths on pin | No status promotion; blocker class unchanged |

Repro (wave-8 W2-02): `git cat-file -e 28c27cb896d02e0185b9fa02047411fa152f6469:governance/tickets/T-IGX-043-G.md` and ancestor check for merge SHAs above.

**Recorded:** lane `eqo/wave-8/w2` · `Ref: governance/tickets/T-IGX-046-F.md`

## Wave-12 read pass (#680 track + W2-02, docs-only)

**Executed:** 2026-11-21 · lane `eqo/wave-12/w2` · **Land:** `292ca3348` (`origin/development`, PR #704). Pins unchanged (`TRAIN_PIN` `28c27cb`, `DOC_PIN` `8cee3c5`).

| Task | Read focus | Wave-12 disposition |
|------|------------|---------------------|
| #680 | `git merge-base --is-ancestor 65ef1f196 8e146e21cd8c63a47d621edc05d5f0036ba0c9ee` (neg vs `development` @ `a0598ba3`) | On rw-train @ `8e146e21`; not on `development`; C16 `NOT_VERIFIED` — green replay missing |
| W2-02 | Post wave-6: `EQO_WAVE6_MANIFEST.md` WB09 EXE exit `0` @ pin | **WB09 SDK slice replayed**; W2-02 tier-boundary import gates (`043-G`/`043-H`) remain **`MERGED_GATE_PENDING`** |
| W2-02 queue | `runs/wb05/AUD_SESSION_E.md` **BLOCKED**; `x/A4-003/A4_GATE_COUNTER_READ.md` | No promotion; gate replay not closed |
| W2-02 evidence | Docs-only; no pytest in this lane | No PASS / closure claim |

Repro (wave-12 #680): `git merge-base --is-ancestor $(git rev-parse af2/pt-fc1) 28c27cb896d02e0185b9fa02047411fa152f6469` → exit 1.

**Recorded:** lane `eqo/wave-12/w2` · `Ref: governance/tickets/T-IGX-046-F.md`

## EQO disposition

- **Train integration @ rw-train `8e146e21`:** #642–#680 (incl. merge `65ef1f196` via `dbaebfa8`) are ancestors; **#680 is not** on `development` @ `a0598ba3`.
- **Closure:** W2 does not certify findings; it establishes merge presence vs `EQO-W3` replay and `EQO-X` blockers (A4-003/A4-004 gate replay per W2-02).
- **Exact blockers carried forward:** #680 open; record-2 items not merged; Operator band G items W2-13.
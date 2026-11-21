# EQO orchestration handoff (F1 bookkeeping)

**Generated:** 2026-11-21T13:00:00+00:00  
**Branch:** `eqo/w1-f1-view`  
**Ticket:** `governance/tickets/T-IGX-046-F.md` (F1 view slice)

## Verified audit facts (2026-11-21)

| Ref | SHA | Role |
|-----|-----|------|
| `origin/development` | `a0598ba3e4f4f903eb4da3db8ea012d5cbf71225` | F1 projection evidence SHA |
| `origin/arch-freeze/rw-train` | `8e146e21cd8c63a47d621edc05d5f0036ba0c9ee` | Train pin |
| PR #680 merge | `65ef1f196` | On train via parent `dbaebfa8`; **not** on `development` |

`IIAC_POLICY_ACTIVATION_RECORD.json` remains `IN_FORCE` globally. That is **not** EQO certification and **not** FRZ promotion.

**Integration:** not done. **Push** and **PR** are operator actions only. **FCS**, **PT-Q**, and **M3** are not executed by EQO.
## Operator merge rule (2026-11-21)

Feature branches are cut from `origin/development` (pin `a0598ba3e4f4f903eb4da3db8ea012d5cbf71225`) and integrate **only** to `development` (operator push/PR). No feature-to-feature merge; no merge target `arch-freeze/rw-train` or another `eqo/*` branch. Unique commits that had been made on train-based `eqo/*` branches are replayed onto **development-based** branches (cherry-pick of feature commits only); the train line (`8e146e21`) itself is **not** merged into `development` (FCS). Train branches with zero unique commits stay replay evidence only. Recorded in `EQO_ORCHESTRATION_STATE.json` → `operator_merge_rule` and `development_branch_replay`.

### Development-based replay (2026-11-21)

| Source branch | Unique commits | Dev branch | HEAD | Cherry-pick |
|---------------|----------------|------------|------|-------------|
| `eqo/w2-a4-027` | `9c909d96b`, `06604af53` | `eqo/dev-a4-027` | `537a06296` | **REPLAYED_BEHAVIOR** — pytest 1 passed — `governance/tickets/T-IGX-052-A.md`, `governance/tickets/T-IGX-055-B.md`, `…/EQO-PKG-A4-027/IMPLEMENTATION_EVIDENCE.md` |
| `eqo/w2-a4034` | `34a0af435` | `eqo/dev-a4-034` | `d1da3b3c1` | CLEAN |
| `eqo/w2-a6006` | `c5b9648b4` | `eqo/dev-a6-006` | `b02278643` | CLEAN |

Worktrees: `eqo-dev-a4027`, `eqo-dev-a4034`, `eqo-dev-a6006` under `D:\Projekty\_wt\intergrax`. Verified: `8e146e21` is not an ancestor of any new dev branch.

## 1029 obligation paths (section 2 first-match)

Checksum: `a11375e09b251be364151547d9dc3450402f6662dad5a7f3fda896459e95e27a`

```
OWNER_DECISION_REQUIRED=313;UNRESOLVED_EXACT_BLOCKER=310;DOCUMENTATION_CORRECTION_REQUIRED=167;FORMAL_REQUALIFICATION_REQUIRED=165;EXISTING_IMPLEMENTATION_REQUIRES_REPLAY=40;NEW_IMPLEMENTATION_REQUIRED=33;EXTERNAL_ARCH_FREEZE_DEPENDENCY=1;total=1029
```

| Path | Count |
|------|------:|
| `OWNER_DECISION_REQUIRED` | 313 |
| `UNRESOLVED_EXACT_BLOCKER` | 310 |
| `DOCUMENTATION_CORRECTION_REQUIRED` | 167 |
| `FORMAL_REQUALIFICATION_REQUIRED` | 165 |
| `EXISTING_IMPLEMENTATION_REQUIRES_REPLAY` | 40 |
| `NEW_IMPLEMENTATION_REQUIRED` | 33 |
| `EXTERNAL_ARCH_FREEZE_DEPENDENCY` | 1 |
| `EXISTING_VERIFIED_EVIDENCE` | 0 |
| `FORMAL_ACCEPTED_LIMITATION` | 0 |
| `DUPLICATE_OR_INDEX_ONLY` | 0 |

## FORMAL_REQUALIFICATION_REQUIRED currency (165 rows)

Artifact: `EQO_REQUAL_CURRENCY.json` (register-sourced SHA fields; view blanket `evidence_sha` not copied).

| Currency | Count |
|----------|------:|
| `CURRENT` (matches `development` or train pin) | 0 |
| `STALE` (missing or other SHA) | 165 |

No register row `lifecycle_state` set to `CLOSED` in this slice.

## Package and replay facts (verified, not pushed unless noted)

### WAVE-01 — EQO-PKG-A4-006

| Field | Value |
|-------|-------|
| Branch | `eqo/w1-a4-006` |
| Commit | `960205aaaf7adf8c8bfd7075e9bfd60031b76417` |
| pytest | 3 passed |
| Technical verdict | `PASS_WITH_SCOPED_LIMITATIONS` |
| Pushed | no |
| Trailer | `Co-authored-by Cursor` present — **strip before any push** |

### WAVE-02 — EQO-PKG-A4-027

| Field | Value |
|-------|-------|
| Branch | `eqo/w2-a4-027` |
| Commit | `06604af5347883dd26c98d54e72e49f2f6f58013` |
| Ticket | `T-IGX-055-B` |
| pytest | 1 passed |
| Technical verdict | `PASS_WITH_SCOPED_LIMITATIONS` |
| Pushed | no |
| Tip | `T-IGX-052-A` absent at branch tip |

### Train replays @ `8e146e21cd8c63a47d621edc05d5f0036ba0c9ee`

| Cluster | pytest |
|---------|--------|
| A4-026 | 12 passed |
| A4-017 | 11 passed |
| FC-3 / FC-1 files | 54 passed |
| FC-2 queue worker | 5 passed (`--extra dev-ci`) |

`test_fc1_host_task_routing_posture.py` is **absent** at `8e146e21`.

**REPLAY_ONLY** (zero new commits): FC-1, FC-2, FC-3, and the fifteen-id pass  
(A4-002, A4-004, A4-005, A4-018, A2-009, A2-038, A5-001, A6-002, A6-003, A6-005, A6-007, A6-008).

### EXISTING_IMPLEMENTATION_REQUIRES_REPLAY remainder @ train pin

Artifact: `EQO_REPLAY_REST.json`.

| Slice | Result |
|-------|--------|
| Six named `tests/*.py` files | 64 passed, exit 0 (prior run; not re-run this slice) |
| Rows with named files | EDER-F-0054, -0060, -0078, -0112, -0134, -0159 |
| Remaining 34 obligation IDs | **NO_NAMED_TEST** (no explicit `tests/…/*.py` in register text; EDER-D-0086 has ellipsis path only) |
| pytest this slice | not run (zero new explicit files) |

Evidence does not change `closure_path` or the 1029 path checksum.

**Not closed:** A4-034, A6-006, A6-001.

## F1 view bookkeeping commit

| Field | Value |
|-------|-------|
| Prior F1 view commit | `321957488fcc0c6c65f9172f9de9b2b062ff3720` |
| Trailer | `Co-authored-by Cursor` on that commit — strip before any push |

## NEW_IMPLEMENTATION_REQUIRED queue (33 rows, 2026-11-21)

Artifact: EQO_NEXT_WAVE_QUEUE.json on eqo/w1-f1-view. No new development-based product commits in this slice.

| Status | Count |
|--------|------:|
| ALREADY_DONE | 2 |
| REPLAY_ONLY | 24 |
| BLOCKED_EXACT_REASON | 2 |
| SKIP_NO_GO | 5 |
| IMPLEMENTED_THIS_SLICE | 0 |

SKIP_NO_GO: EQOA-A3-008, EQOA-A3-018 (no implementation go ticket at development pin); EQOA-A6-001 (evidence-gap hard stop); EDER-N-0003 / A4-007-b1b2; EDER-N-0004 / DP-54 (T-IGX-052-E awaiting mandate).

## Artefacts

- `EQO_REMAINING_OBLIGATION_VIEW.json` — 1029-row projection (`lifecycle_state` stays INGESTED; no CLOSED).
- `EQO_REQUAL_CURRENCY.json` — 165-row register-backed currency vs development/train pins.
- `EQO_ORCHESTRATION_STATE.json` — wave map, package SHAs, counters, `remaining_non_executable_stops`.
- `EQO_NEXT_WAVE_QUEUE.json` — per-row disposition for all 33 `NEW_IMPLEMENTATION_REQUIRED` obligations.
- `EQO_REPLAY_REST.json` — named-test replay facts and 34× `NO_NAMED_TEST`.

## Forbidden this slice

FRZ promotion, `EQO_CLOSED`, editing `IIAC_POLICY_ACTIVATION_RECORD.json`, editing `register.jsonl`, `orchestration/packets/` writes, setting any lifecycle to `CLOSED`.

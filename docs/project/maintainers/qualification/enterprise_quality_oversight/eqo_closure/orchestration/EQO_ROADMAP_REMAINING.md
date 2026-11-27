# EQO roadmap — remaining executable vs hard stop

> **RECONCILED 2026-11-27 (W2.1-D, OP-EC-01 OD-W2-DOCS continuation).**
> - Current: development == origin/development == 3190a4487; Wave 2 merges 79f52a89f, a3c6a9d43, f52eb70e8, b7002e24f, plus follow-up commits 7d2cc3d28 (ticket Lifecycle formatting) and 3190a4487 (INTELLIGENCE-X R2 docs, PROPOSED, not Wave 2 scope) are pushed.
> - PR #680 was MERGED 2026-11-21 into arch-freeze/rw-train as 65ef1f196, now an ancestor of development; this supersedes the lines below that place #680 only on rw-train or as an open draft. C16/C21 statuses are unchanged.
> - 8e146e21c (PR #715) is an ancestor of development; AD-AF-DEVTRAIN D-1 (MEP-AF2_v2.1 Appendix V6, signed 2026-11-26) makes development the train, so there is no separate train and any text below saying 'merge 8e146e21 into development forbidden', 'train only' or '164 commits ahead' is historical, not violations.
> - Pins recorded below (dbaa771ac) are historical; the current development pin is 3190a4487.
> - CI at 7d2cc3d28 is red (0 new Wave 2 regressions; downstream jobs never ran): see `orchestration/EQO_W21_CI_DISPOSITION.md`. No G2/R/FCS/M3, not closed, no certification claim.
> - Body below is unchanged historical record.

**Development pin:** `dbaa771ac2e58391dfb0c7c3f4d241b031302b7c` (local `development`; `8e146e21` and `origin/development` are ancestors).  
**Generated:** 2026-11-21 ticket-delta slice (Composer implementer).


## Post-merge replay slice (2026-11-21)

- Merged origin/development into local EQO development at 88c2708c (merge commit with T-IGX-046-F).
- Re-read all STILL_PRESENT / CHERRY_PICK_CONFLICT rows: **11** became ABSENT_ON_DEVELOPMENT; **0** product ports (MERGE_CONFLICT 0); **7** remain STILL_PRESENT (A2-009, A2-038, A4-017, A4-030, A4-044, A5-001, A6-002).
- Cherry-pick ports abandoned; train product posture arrives via origin merge, not feature cherry-picks.

## Ticket-delta slice (2026-11-21 evening)

- Re-read the **7** remaining STILL_PRESENT rows against cited ticket closure text (not train blob replay): all **7** → `ALREADY_MATCHES_TICKET` (no product commit).
- Next roadmap executable (`EQO-PKG-A4-006`): `check_runtime_event_tenant_propagation.py` already references `uaep_executor.py` on this pin — no further file edit without a new ticket row.

## EXECUTABLE_NOW

- Re-read cited product files on local `development` for all **24** `REPLAY_ONLY` rows in `EQO_NEXT_WAVE_QUEUE.json` and record per-row verdicts in `EQO_DEVELOPMENT_REPLAY.json` (no `register.jsonl`, no `lifecycle_state` CLOSED, no `path_counts` change).
- Update orchestration bookkeeping only: `EQO_ROADMAP_REMAINING.md`, `EQO_DEVELOPMENT_REPLAY.json`, `EQO_NEXT_WAVE_QUEUE.json` replay notes, `EQO_ORCHESTRATION_STATE.json` `development_replay` section, `HANDOFF.md` — committed on a development-cut branch and merged with `git merge --no-ff`.
- Where a row is `STILL_PRESENT` and `T-IGX-043-*` names an exact write set, replay train-verified product commits via **cherry-pick onto a new development branch** (feature commits only; never merge `arch-freeze/rw-train` / `8e146e21`). Abort cherry-pick or merge on product conflict; record `CHERRY_PICK_CONFLICT` / `MERGE_CONFLICT`.
- Run at most one named pytest file per ported row: `uv run --extra dev-ci pytest <file>` when a port lands and the ticket names the test.

## HARD_STOP

- **EQO terminal closure (plan §11 / §5 F6):** `EQO_CLOSED_INDEPENDENTLY_VERIFIED`, C01–C25 `FULLY_MET`, FRZ promotion, editing `IIAC_POLICY_ACTIVATION_RECORD.json`, setting register `lifecycle_state` CLOSED — certifier / Operator only.
- **313** `OWNER_DECISION_REQUIRED` rows — no invented decisions; **8** open plan §10 operator gates (implementation go 043-A/B/C/D, write sets, OD-EQO-04/09, push/PR).
- **310** `UNRESOLVED_EXACT_BLOCKER` / **305** `TRIAGE_PENDING` — no reclassification this slice.
- **165** `FORMAL_REQUALIFICATION_REQUIRED` — all **STALE**; must not mark **CURRENT** without full requal run.
- **34** `EXISTING_IMPLEMENTATION_REQUIRES_REPLAY` with **NO_NAMED_TEST** — no invented tests; replay blocked until FCS / ARCH-FREEZE chain (plan §5 F4, §12 ARCH-FREEZE).
- **FCS, PT-Q02–Q13, M3, RB-2, G2**, **EQO-REPLAY-FCS**, full qualification suite, IIAC PASS — not agent-executable.
- **Merge `8e146e21` (or any branch with train as ancestor) into `development`** — forbidden; train line stays replay evidence until FCS.
- **BLOCKED / SKIP** implementation rows: A4-034 (retry budgets), A6-006 (ENT_AUDIT_X semantic skip), A4-040 / OD-EQO-04, Poland-default A4-042, A3-008, A3-018, A6-001, EDER-N-0003, EDER-N-0004.
- **Operator-only:** `git push`, PR create/merge, stripping `Co-authored-by` on unpushed EQO commits before push.
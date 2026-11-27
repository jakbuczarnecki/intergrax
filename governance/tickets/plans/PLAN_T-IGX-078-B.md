# GA.2R (AF-CLOSE): repair the ARCH-FREEZE RW plan for RW-PLAN-AUDIT round 2

Ticket: T-IGX-078-B. Lane kind: DOC + TL + QR.

## Context

GA.2 round 1 returned `RW_PLAN_AUDIT_FAIL` (1 BLOCKER, 12 HIGH, 9 MEDIUM, 11 LOW; any BLOCKER or HIGH fails). GA.3 SCOPE and every product lane wait for a PASS. A third FAIL on the same finding is an escalation. Intended outcome: the tracked RW plan is generated without loss, every audit finding has a fix or a named Operator item, and no evidence is claimed beyond what was run.

## Read scope

- `.tmp/session/arch-freeze/close/` (main checkout): audit report, GA1/GA2 triage, WORKLIST, BOARD, DECISIONS_W0, OP-A.
- `docs/project/maintainers/qualification/arch_freeze_program/` (`AF/` below): RW plan, MEP-AF2 section G and Appendix V6, matrix, `continuation/RED_FIRST_*`, node ledger.
- `scripts/governance/{build_rw_plan,verify_rw_plan_phase_check3,plan_digest,ticket_approval}.py` and their tests; owner tickets and plans.
- Untracked, read only: `.tmp/session/arch-freeze/rw-plan/` (versioned plan), `WP-12/PHASE_CHECKS.md` (source of the 18 CODE_FIX ids).

## Pins and branch

- Branch `claude/clever-newton-6ded37` (this worktree), HEAD = local `development` 32005c212.
- Measurement pin = aabd3925c (as instructed). aabd3925c..32005c212 is docs and one docs ticket (T-IGX-079-A); no overlap with the write set.
- Replay worktrees: detached, slug `ga2r-red`, under `D:/Projekty/_wt/intergrax/_scratch/ga2r-red/{pin,prefix-wt/<sha12>}`. Removed at the end. Existing worktrees of other sessions are not touched.

## Write set (exact)

Tooling (OP-A item 4):
- `scripts/governance/build_rw_plan.py`
- `scripts/governance/verify_rw_plan_phase_check3.py`
- `tests/unit/governance/test_build_rw_plan.py`
- `tests/unit/governance/test_verify_rw_plan_phase_check3.py`

Docs under `AF/`:
- `rw_plan_bindings.v1.json` (new; reviewed input of the generator)
- `rw_plan.v1.json`, `RW_PLAN.md` (generated only, no hand edits)
- `rw_plan_binding_record/INDEX.md` plus verbatim `RW_PLAN_v<N>.md`, only for versions that tickets or plans cite (v6, v13, v14, v15, v18, v19, v21)
- `continuation/RED_FIRST_GA2R_RESULT.{json,md}`, `continuation/RED_FIRST_GA2R_OPERATOR_ITEMS.md` (new)
- `ARCH_FREEZE_NODE_LEDGER.md` (T-AF2-RW-PLAN / -AUDIT rows only, L10)
- `SOURCES.sha256`

Governance:
- `governance/tickets/T-IGX-078-B.md`, `governance/tickets/plans/PLAN_T-IGX-078-B.md`
- New ownership tickets `T-IGX-078-C.md`, `-D.md`, `-E.md` (plan_pending, no Approved line)
- One appended section "RW traceability (GA.2R)" in ticket files only: T-IGX-043-A, -B, -C, -G, -H, -I, -J; 044-F; 046-D, -G; 051-A, -B, -D, -E, -F, -H, -I, -N. Each states: records rows, NT ids and binding pointer; grants no write authority. The generator-derived owner list is final; if it differs, I report it in DONE.

Not written: `intergrax/`, `agents/`, `applications/`, any file under `governance/tickets/plans/` except the new 078-B plan (others are digest-bound), the matrix, the MEP, the RED-CLASSES decision and older RED_FIRST records.

## Approach

### 1. Generator first (H5, D1, L2, L8), with failing tests written first
- Write sets come from each approved plan's explicit write-set section (heading and excluded subsections named per ticket in the bindings file; fenced blocks, tables, bullets; globs kept; adds `.github/`, `governance/`, `docs/`, `pyproject.toml`, `uv.lock`). No `[:80]` cap. Paths are labelled exists / new / glob against the pin tree. A ticket without a plan (046-G) is listed as `NO_PLAN`; mention scans are never presented as write sets.
- D1: regeneration keeps every prior red-first status and all extra fields.
- `regression_scope_source` names the real source (matrix column, matrix Appendix C, binding). `execution_sha` and `generated_at` are fresh; a separate `read_pin` field records aabd3925c.
- The hand-written note in `RW_PLAN.md` moves into the bindings file.

### 2. Plan structure (H1, H2, M9, L11)
- Per-ticket table: lane, state (landed / open), hard predecessors, merge position, mandatory section G tests (node, or "authored by owner ticket"). All train tickets, including 051-H/I, 044-F, 046-D, 046-G, 043-N.
- Full pairwise overlap matrix from the extracted write sets. The generator fails if an overlapping pair has no ordering path in the dependency graph, or the graph has a cycle. Edges come from approved plans (e.g. 051-A after PT-WF, PT-FC1, SUPPLY, PT-FC3, PT-FC2, TRUST) and WORKLIST section 5. Hard-coded hotspots with nonexistent `applications/_shared/` paths are removed.
- Lane = disjoint group, cap 5, checked mechanically; D-4 separately limits active writers to 2. PT-WA becomes an explicit per-ticket edge for W3. Duplicate lane_7 entry removed.

### 3. CODE_FIX mapping and negative tests (H4, B1, H6, H11, H12, M7)
- Tracked table of the 18 ids (with the PHASE_CHECKS source sha256): id, negative test, owner ticket, file present or absent at pin, regression command with real paths. Per-row `tests/` commands for all inventory rows (matrix Appendix C).
- B1: file::node, mechanism and owner replace class markers: SEC-06 = NT-SEC-06-1 (nodes in PLAN_T-IGX-051-B); TYP-01/03 = NT-WF-B01..B06; RPL-04 = NT-WF-C01..C05; TEN-02 = NT-T01-A4-006; TEN-07 = event-id persistence and emitter-gate nodes. Absent files are labelled absent. The checker rejects `NEG-0`, `NEG-P` and "gate extension" as an identifier.
- Re-ownership (new tickets, so scope names each row and no digest-bound plan changes):
  - 078-C, lane K1-B (WL-K1-B-02): TYP-01, TYP-03, RPL-04 product fix and tests (candidate paths: the six `[object]` exposure files and three colocated-adapter files still open at the pin). Closes the H11 hole left by 051-G.
  - 078-D, lane K1-C (WL-K1-C-08): TEN-02, TEN-07, A4-006. The checker's A4-006 ownership contract and its test change with it.
  - 078-E, lane K1-D (fact check, DP-30): PLG-08 with NT-PLG-08-1; relation to RW-GATES 051-F (`NT-RW-GATES-NC-RAISE`) stated.
  - A3-018: stays 046-G, lane K3 (WL-K3-08); ticket names row and NT-G02-02; plan marked required before implementation. FRZ-10: keep the authorised W5/W8 node contract, name the carrying ticket or state that none exists.
- H3: publish the cited versions verbatim (line cites stay valid) as "historical binding authority, as cited; V6 supersedes train rules; live plan is RW_PLAN.md". INDEX maps each of the citing files to version, sha256, section, ticket and AF-CLOSE lane. Each file is scanned for secrets and Polish text first; a file that fails is not published and is reported as BLOCKED.
- M7: per-row traceability in the plan plus the ticket sections above.

### 4. V6 and coverage (H10, M1, M2, M3, M8, L1, L3-L7, L9, L10)
- G2 entry, dependencies, disposition and Markdown rewritten to V6 (development as train, no RW train branch, held paths from G2, pre-G2 items and their order; G2, R, FCS, M3 not set). FRZ-10 paths and order against G2.
- Coverage table: every matrix OPEN row outside the inventory is derived by the generator and needs a reason and owning node (missing entry fails). Matrix status column for inventory rows with the meaning of requalifying a PASS row. FIND-AF2-01..23 mapped or excluded, 23/23 checked. 39-row arithmetic computed. Refs point to evidence records (L3), PRD-08 exit code (L4), REL-09 review note (L5), REG-09 states NT-REG-09-1 uncovered (L6), MD and JSON owners rendered from one field (L7).

### 5. Red-first re-measurement (H7, H8, H9, M4, M6)
Rows (21): BND-04/05/06, CFG-06, CTR-02/06, DEBT-02/03/04, EXE-02, OWN-05, PLG-05, PRD-05, RPL-04, SEC-04/06, TEN-02/07, TYP-01/03/05.

Method (MEP V5.3): lock with `python D:/Projekty/_wt/_locks/_acquire.py`, release with `rm -f`; one `uv run pytest` at a time, several node ids per call, background if long. (a) Green control: existing nodes at the pin. (b) Replay per prefix SHA: detached worktree, overlay only the test file from the fix commit, run the node.

`RED_AT_PREFIX` only if all hold: node collected; fails on a behavioural assertion (not ImportError, FileNotFoundError, or a missing new symbol or field); the assertion exercises the declared FRZ mechanism; the node passes at the pin. Tracked evidence per row: node id, mechanism, prefix and pin SHAs, test blob sha, exit codes, verbatim assertion lines, log sha256.

Otherwise: `EVIDENCE_MISSING` with the reason and the ticket where red-first is due. This is not a waiver. I write no new lane tests (WL-K1-F-23) and add no `RED_FIRST_OUT` or `LATER_WAVE` entry: the RED-CLASSES decision authorises only PLG-02, PLG-07, RPL-04, REL-07, PRD-08 (OUT) and FRZ-10 (later wave). Those six keep their disposition. Matrix test definitions are not changed.
- M4: PRD-05 points to the record that actually supports it, only if my replay confirms it; the missing census gate goes to the owner ticket.
- M6: DEBT-02/03 get a defined seeded negative case (existing seeded node where it fits; DEBT-03 has no node today and is not positive).
- L-level record defects are corrected in the new record and in plan refs; old records stay unchanged.

## Operator items (not mine)

1. M5: verbatim Operator statement or decision-register entry behind OUT / LATER_WAVE.
2. Rows left `EVIDENCE_MISSING`: disposition needed (recommendation: red-first due at the owner ticket, as a new RED-CLASSES entry). Delivered as a ready packet; not applied.
3. TYP-01/03: inverting the EBH-6 Q2-D2 gate is reserved to the Operator (Q-WF-04). DW0-03 paths do not cover these Nexus files.

## Choices the orchestrator can change at GO

- A. New tickets 078-C/D/E (recommended) instead of sections on 051-G, 065-A, 051-F. Confirm these letters are not allocated elsewhere.
- B. Verbatim binding record (recommended; adds large files) instead of superseding and amending each citing ticket.
- C. OP-A item 4 text literally names "scripts/.github/tests tickets in K3 and K0.1 gate marks"; GA.2R is covered only through GA.1 D1 routed to K3. Please confirm before the delegate signs.
- D. GA.2R takes GA.1 D1; K3 drops it. K3-B also writes the checker and its test: GA.2R lands first, K3-B rebases.
- E. Send items 1-3 to the Operator before audit round 2.

## Dependencies

Needs: `GO GA.2R`. Blocks: audit round 2, then GA.3, then all K1/K3 lanes. Approval of T-IGX-078-B comes from the signing delegate only after independent REVIEW_PASS; I write no Approved line.

## Ticket draft (T-IGX-078-B)

- Scope: this plan. Plan digest from `scripts/governance/plan_digest.py` over `PLAN_T-IGX-078-B.md` (this text, tracked copy).
- Test contract: acceptance list below. Out of scope: product paths, new lane tests, matrix/MEP/decision edits, push, PR, merge.
- Implementer session: this session. Lifecycle: plan_pending.
- Before commit 1: `git fetch` and confirm 078-B..E are free on origin and all local refs (B..E were free on this check). Every commit carries `Ref: governance/tickets/T-IGX-078-B.md`; English; no AI attribution.

Commit order: (1) ticket and plan; (2) failing generator/checker tests plus fix; (3) red-first evidence; (4) bindings, regenerated plan, binding record, tickets, ledger, `SOURCES.sha256`.

## Acceptance and verification (one command per step)

1. New tests fail before the fix (kept log), then: `uv run pytest tests/unit/governance/test_build_rw_plan.py tests/unit/governance/test_verify_rw_plan_phase_check3.py tests/unit/qualification/test_run_campaign_b_w2_alpha.py -q`.
2. `uv run ruff check` on the four changed Python files.
3. Generator run twice: identical output apart from `generated_at`.
4. `verify_rw_plan_phase_check3.py`: `PLAN_STRUCTURE=PASS`, 39/39 ownership, 39/39 named negative tests. Red-first result is reported as measured (INCOMPLETE is expected if any row is `EVIDENCE_MISSING`).
5. `verify_arch_freeze_plan_traceability.py`, plan digest match, `sha256sum -c SOURCES.sha256`.
6. `git diff --name-only 32005c212..HEAD` is a subset of the write set; no product path.
7. DONE report carries a closure table: finding id, fix, file:line, with CLOSED / PARTIAL / OPERATOR for all of B1, H1-H12, M1-M9, L1-L11, D1.

## Risks

- Round 2 may still fail on red-first: several declared tests are absent or test a different mechanism (static reading found this for BND-05, BND-06, EXE-02, OWN-05, TYP-01/03/05, RPL-04, DEBT-03/04, SEC-06; replays decide). Honest reclassification removes the evidence mismatch but leaves rows incomplete until item 2 is decided.
- H11/H12: a plan_pending ownership ticket names the rows but has no approved plan yet; the auditor may require one.
- Write-set extraction depends on varied plan formats; per-ticket anchors fail closed when a section is missing or empty.
- Replay environments need a `uv` venv per prefix SHA (time, disk); runs are sequential. A prefix that cannot build is recorded as such.
- `development` may move again; I re-check overlap before DONE and do not rebase the pin.

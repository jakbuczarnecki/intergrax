# P3-IMPL — Phase 3 product lane handoff (preflight @ fe0937517)

| Field | Value |
|---|---|
| Status | **BLOCKED** (P1-SIGN: no ticket `Approved:` lines) |
| Preflight session | explore agent, worktree `hist-base` |
| START_HEAD | `fe09375177bd0fa0d9ebb661b7f9bf64e2f1ed25` |
| Date | 2026-11-08 |
| Orchestrator slice | Phase 1 plan reviews (`hist-cert/plan-reviews`) |

## Gate impact

**Do not open P3 product branches** for a ticket until that ticket has a verified `Approved:` line from a designated signing session ([`P1_SIGN_HANDOFF.md`](P1_SIGN_HANDOFF.md)). This handoff is implementer routing only; it does not grant authority.

## Lane ↔ ticket mapping

| Lane | Ticket | Branch name (planned) | Plan review |
|---|---|---|---|
| C1 | T-IGX-030-B | `hist-cert/c1-030b` | [`plan-030b/PLAN_REVIEW.md`](plan-030b/PLAN_REVIEW.md) PASS |
| C2 | T-IGX-030-C | `hist-cert/c2-030c` | [`plan-030c/PLAN_REVIEW.md`](plan-030c/PLAN_REVIEW.md) PASS |
| C3 | T-IGX-032-A | `hist-cert/c3-032a` | [`plan-032a-r2/PLAN_REVIEW.md`](plan-032a-r2/PLAN_REVIEW.md) PASS |
| C4 | T-IGX-033-A | `hist-cert/c4-033a` | [`plan-033a-r2/PLAN_REVIEW.md`](plan-033a-r2/PLAN_REVIEW.md) PASS |

Integration target after lanes: `hist-cert/product-batch` (P3-INT; SHA TBD).

## Preflight — defect confirmation (read-only @ START_HEAD)

All four defects **still present** at `fe0937517`. Plan reviews and [`HISTORICAL_FINAL_AUTHORITY_AND_DECISIONS.md`](../HISTORICAL_FINAL_AUTHORITY_AND_DECISIONS.md) § baseline reconfirmation align.

### C1 (T-IGX-030-B) — recovery import

- **Finding:** `HIST02-F-013` / FRZ-BND-04 (EBH-4 R1).
- **Site:** `intergrax/runtime/tools/tool_operation_recovery.py:22` imports `RuntimeErrorCode` from `intergrax.runtime.nexus.errors.error_codes` (Nexus encapsulation violation).
- **Fix scope:** contract home in `intergrax/contracts/runtime_error_code.py`, Nexus re-export alias, single-line recovery import change, new contract-home architecture test. EBH-4 gate file byte-identical.
- **Allowed files:** per ticket plan §6 and [`plan-030b/PLAN_REVIEW.md`](plan-030b/PLAN_REVIEW.md).

### C2 (T-IGX-030-C) — unguarded `pool.submit`

- **Finding:** `H01-F015` / TOOL-LIFE residual.
- **Site:** `intergrax/runtime/nexus/tools/invoker.py`, branch `dep_boundary is None` (~1905–1942): after `ext_op.mark_running()` at ~1914, `future = self._execution_pool.submit(...)` at **~1915** has no `try/except BaseException` with `ext_op.mark_failed()` before re-raise. Boundary branch (~1973–1979) already terminalizes submit failures.
- **Fix scope:** guard **only** `pool.submit` in the no-boundary branch; do not change `mark_running()`, `bind_future()`, result handling, or boundary branch.
- **Mechanical apply (after reading current `invoker.py` at lane base):**

  ```text
  git restore --source=29c387879 -- intergrax/runtime/nexus/tools/invoker.py
  ```

  Commit `29c387879` is the historical C3 product hunk (+5/-1 around submit). Re-diff against current file; apply only if the defect is still unguarded (it is at START_HEAD).

- **Red-first:** commit **test** `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` first (expect **red** without product hunk); then product commit (expect green).
- **Dependency:** test imports helpers from the **T-IGX-030-A** C1 helper module already merged on `development` at `fe0937517`. Lane base must include 030-A; do not reimplement helpers in 030-C.
- **Out of scope:** `bind_future` failure residual R-1; any 030-A lifecycle files; new error codes.
- **Allowed files:** `invoker.py` + the one new test file only ([`plan-030c/PLAN_REVIEW.md`](plan-030c/PLAN_REVIEW.md)).

### C3 (T-IGX-032-A) — dual store forward

- **Finding:** `HIST03-F-015`.
- **Site:** `intergrax/skills/skill_execution_pinning_wiring.py:35–38` — `resolve_skill_execution_pinning_wiring` forwards **both** `kv_store` and `document_store` into `wire_skill_execution_pinning_store`, which rejects dual backends.
- **Fix scope (A2):** resolver passes `document_store` into the leaf **only when** `kv_store is None`; KV authoritative. ~3 lines in the one product file; new unit test for precedence. P2-C file byte-identical; leaf dual-backend rejection unchanged.
- **Allowed files:** per plan and [`plan-032a-r2/PLAN_REVIEW.md`](plan-032a-r2/PLAN_REVIEW.md). Step-6 wider run: **four** named test files only (not five); `HIST-F-DUR-01` remains open.

### C4 (T-IGX-033-A) — Tier-0 upward edges (CLI half)

- **Finding:** `HIST02-F-001` / F-002 CLI half.
- **Sites:** `intergrax/cli/external_work.py` — multiple `importlib.import_module(...)` and Tier-3 static edges; exemplar static import at **line 338**:

  `from applications.governed_contractor_application.host.collaborative_work_local_fixture import ...`

- **Explicitly not in ticket:** `intergrax/applications/_shared/product_manifest_registry.py` (tracked blocker `HIST-F-BND-03B`).
- **Fix scope (b2):** command-set seam + Tier-3 `python -m applications...` launcher; plan §3.1 / §3.2 file set; new BND-03 gate without weakened EBH-5/7 assertions.
- **Allowed files:** per [`plan-033a-r2/PLAN_REVIEW.md`](plan-033a-r2/PLAN_REVIEW.md).

## File-disjoint batch (C1–C4)

No shared product paths between lanes at implementation time:

| Lane | Primary product touch |
|---|---|
| C1 | `contracts/runtime_error_code.py`, `nexus/errors/error_codes.py`, `tools/tool_operation_recovery.py` |
| C2 | `nexus/tools/invoker.py` only |
| C3 | `skills/skill_execution_pinning_wiring.py` only |
| C4 | CLI / Tier-3 launcher set per 033-A plan (not `product_manifest_registry.py`) |

Cross-lane ordering: C1 and C2 are independent (`invoker.py` may keep importing `RuntimeErrorCode` via Nexus alias). C3 and C4 disjoint from 030-B/C product paths.

## Implementer checklist (per lane)

1. New worktree detached at `fe0937517` (or lane branch from that SHA).
2. Confirm plan digest matches ticket (commands in each `PLAN_REVIEW.md`).
3. Wait for that ticket's P1-SIGN `Approved:` line.
4. Red-first tests where the plan requires (C2 mandatory; C1/C3/C4 per ticket plan).
5. Record evidence: red log, green log, `git diff --name-only fe0937517..HEAD` ⊆ allowed list.
6. Publish lane branch; handoff child doc under `audits/impl-cN/HANDOFF.md` with implementation SHA (Phase 3 terminal).

## ACT-N11 / IIAC

Product commits are expected to make `verify_iiac_policy_activation --require-in-force` **red** until P5/P6. Do not “fix” ACT-N11 inside product tickets. See [`HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md`](../HIST_IIAC_SHARED_REACTIVATION_AGREEMENT.md).

## Authority pointers

- Master plan P3 sections: [`HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md`](../HISTORICAL_CERTIFICATION_FINAL_MASTER_PLAN.md)
- Adopted decisions: [`HISTORICAL_FINAL_AUTHORITY_AND_DECISIONS.md`](../HISTORICAL_FINAL_AUTHORITY_AND_DECISIONS.md)
- Ticket bodies and plans: `governance/tickets/T-IGX-030-B.md` … `033-A.md` and matching `governance/tickets/plans/PLAN_*.md`

# PLAN-P2C REPORT - T-IGX-032-A (HIST03-F-015), plan only

Planner: fresh session. Worktree `D:/Projekty/_wt/intergrax/plan-p2c`, branch `hist-cert/plan-p2c`, base `origin/development` af323c005. One local commit `b56a566e2147bb55e90b21fe572d96c748d87c17`. Not pushed, no PR, no rebase/reset/amend. No test was run. No product or test file edited.

## Deliverables
- `governance/tickets/T-IGX-032-A.md` - lifecycle `plan_pending`, HELD (ACT-N11 / DR-12=B), no `Approved:` line.
- `governance/tickets/plans/PLAN_T-IGX-032-A.md`
- Plan digest: `2628d1a834d2fcf25f47c55b0d60355811898c9e60337306b557f152d7ba7446` (python -I scripts/governance/plan_digest.py; recomputed from the committed blob: same). Committed blobs ASCII, LF, no backslashes.
- Ticket id: 032-A was free (no file, branch, PR; registry lists it as tentative). BND-03 planner was told 033-A.

## Result
- Recommendation A2: `resolve_skill_execution_pinning_wiring` passes the document store to the leaf only when `kv_store is None` (KV authoritative, document store fallback). The leaf `wire_skill_execution_pinning_store` keeps rejecting both; its existing test `...rejects_dual_backends` stays untouched.
- Refines Q-11 ("pin wiring accepts both"): the sibling convention is resolver-picks / leaf-strict (`profile_resolution/wiring.py:68` vs `persistence.py:344`). Owner SKILL-X and KV authoritative confirmed as the recommendation.
- Rejected: A1 leaf accepts both (inverts an existing contract test), C strict composition supplies one store (untraced consumers of the document store, still misses the other both-store branches). B (host runtime call site) kept as fallback.
- New finding (read, not in Q-11): `is_durable` is a hard-coded True on both pin stores (`execution_pinning_persistence.py:291,:388`), so the durability validator cannot detect an in-memory backing. KV-first protects strict composition; a doc-only strict host is still accepted. Left as residual R1, not fixed.

## Scope
- Product path touched at implementation: ONE file, `intergrax/skills/skill_execution_pinning_wiring.py` (~3 lines). Nothing in agents/ or applications/. None of the six MATERIAL_GOVERNANCE_PATHS, no scripts/.
- New test: `tests/unit/skills/test_skill_execution_pinning_wiring_store_precedence.py`, 8 tests; RED set T1-T3 fails with `not both` before the fix. Probes P-A (revert), P-B (invert precedence), P-C (drop leaf guard).
- Test contract: 7 P2-C tests (file byte-identical) + pinning tests; ONE pytest at a time under the heavy lock, `uv run --extra dev-ci`.
- `git diff --name-only ae467c8c4 HEAD -- intergrax agents applications` is empty at base, so this is the first product-path change after the IIAC audited SHA; ACT-N11 goes red until the single re-activation.

## Could not verify / not done
- Any "after" result; P2-C green after the fix; that no second defect hides behind the first (Q-11 ran -x).
- Wider blast radius is UNMEASURED: research app and the governed-contractor strict `prod_q_r1_*` unit tests go through the same line. Plan step 6 measures five named files before/after (needs Operator confirmation under C10/C15).
- Strict KV = sqlite and strict document store = in-memory taken from Q-11 and a partial read; the "green before d1040f485" claim is documentary only.
- Housekeeping: the repo pre-commit hook ran `uv` and created a gitignored `.venv` in the planner worktree (no Operator approval for that; flagged as in plan-tests).

## Questions for the Operator
1. Confirm A2 over B / A1 (default A2).
2. Confirm owner SKILL-X + KV authoritative as decisions for this ticket.
3. Confirm or cut the bounded wider measurement (5 named files).
4. Should R1 (hard-coded is_durable) become a separate finding/ticket now?

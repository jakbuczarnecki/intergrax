## DRAFT 1 — `T-IGX-023-A` (working name TLX-CORR-A): reconcile Batch B side-effect-safety tests with Batch C terminal semantics (C-1 / F-01)

### Scope
Test-only reconciliation of four stale assertions in `tests/unit/runtime/tools/test_tools_side_effect_safety.py` (current line refs at base `2f6664d53`):
1. `test_side_effect_timeout_marks_uncertain_and_blocks_replay` (`:497-522`; assertion `:518`) — expects `TIMEOUT`.
2. `test_mapped_validation_error_after_executor_marks_uncertain` (`:608-648`; `:642`) — expects `VALIDATION_ERROR`.
3. `test_close_shuts_down_execution_pool` (`:1232-1251`; `:1250`) — expects substring "cannot schedule new futures after shutdown".
4. `test_replay_before_close_preserves_exactly_once` (`:1292-1316`; `:1315`) — same substring.

Basis: sanctioned `ABANDONED` disposition (roadmap §3.0.5 "Abandonment"; Batch C `a5f813a2c`; Batch C tests `…batch_c_stalled_abandoned_recovery.py:329-403, 535-571`). Tests 1–2 depend on **OD-TLX-A**; tests 3–4 depend on **OD-TLX-C** (this draft assumes option A: accept `ABANDONED`/`UNCERTAIN` for a post-`close()` call; if the Operator picks B/C, tests 3–4 are handled by TLX-CORR-C instead and this ticket covers 1–2 only).
Plus (OD-TLX-D): add the module to the standing mandatory L2 inventory so the regression class cannot recur silently (inventory docs are canonical → **proposed patch only**, see below).



### Exact files and minimal change
| File | Change |
|---|---|
| `tests/unit/runtime/tools/test_tools_side_effect_safety.py` | **Only** the four tests above. (1) `error_code == RuntimeErrorCode.ABANDONED`, `error_message == "abandoned"`; keep ledger `UNCERTAIN` and replay→`InvocationUncertaintyError`. (2) `error_code == ABANDONED`; keep `external_effects["n"] == 1`, ledger `UNCERTAIN`, replay→`InvocationUncertaintyError`, `n == 1`. (3)(4) (if OD-TLX-C=A) `error_code == ABANDONED`, `error_message == "abandoned"`, `executor.calls == 1` (never started), plus assert the new key's ledger status is `InvocationStatus.UNCERTAIN`; add a one-line comment citing the OD and that the message `"cannot schedule…"` is intentionally not asserted (ThreadPoolExecutor implementation detail). |
| `intergrax/**` | **No change.** |

### New / negative tests (same file, test-only)
- N1: a **read-only** (`side_effects=False`) tool with the same timeout still returns `TIMEOUT` (guards the boundary of the ABANDONED rule: `must_abandon_*` requires side-effects / non-READ_ONLY). Add only if an equivalent does not already exist — implementer must grep first and record the existing test id instead.
- N2: after-`close()` replay of the *same* key whose ledger is `UNCERTAIN` raises `InvocationUncertaintyError` and `executor.calls` stays unchanged (proves no blind replay of a poisoned key).
- N3: a successful call is still `COMPLETED` and not misclassified `ABANDONED` (exists: `test_post_claim_success_finalizes_completed_once` — cite, do not duplicate).

### Test contract
Single command per step (heavy-test mutex; `-p no:xdist`; `--extra mcp` is not needed for these modules but is harmless):
```text
uv run pytest -p no:xdist \
  tests/unit/runtime/tools/test_tools_side_effect_safety.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_retry_idempotency_unknown_outcome.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_b_real_governance_authorization.py \
  tests/unit/runtime/security/test_p0_safety_8_retry_redelivery_authorization.py \
  tests/unit/runtime/nexus/tools/test_tool_life_x_batch_c_stalled_abandoned_recovery.py -q
```
Expected: 0 failed (Batch B L2 inventory is 63 at Batch B; count at the new SHA is recorded, not assumed). Then the F8 consolidated L2 inventory (12 modules, `TOOL_LIFE_X_R4_FINAL_ROOT_CAUSE…§5`; auditor reproduced 67/67 at the baseline) **replayed at the post-fix SHA** — satisfies the auditor's F-10 "not replayed at base" item. Logs → `.tmp/session/<ticket>/`. Record the exact HEAD SHA.
Verify the diff contains only the allowed test file: `git diff --name-only <base>..HEAD` (+ ticket/doc files if Operator permits) and zero `intergrax/` paths.

### Proposed (not applied) canonical patches — for the authorized integration owner only
- Reconciliation addendum (do **not** rewrite the original verdict): record that `test_tools_side_effect_safety.py` was red from `a5f813a2c`..baseline (4 tests), root cause stale-tests-vs-Batch-C-ABANDONED, Batch B "63 passed" valid only at `60a3ad53b`, fixed at `<new SHA>`.
- Add `test_tools_side_effect_safety.py` to the Batch C/D/E/F8 standing L2 inventory lists.
- Roadmap/closure status wording is **out of scope** for this ticket.

### Out of scope
Any `intergrax/` edit; FRZ-* promotion; closing TOOL-LIFE-X or any stage parent; IIAC paths; modifying the original acceptance/reconciliation text or audit verdicts; Docker/vendor/prod config (C10/C15); changing `ABANDONED` semantics; anything about C-2.

### Risk
Low. Masking risk: updating assertions could hide a real regression if OD-TLX-A is wrong → mitigated by the explicit Operator question, citing the spec text, and keeping all ledger/replay/effect-count assertions. Test-only; no runtime exposure.

### Rollback
`git revert` of the single test commit; module returns to 4 red (known baseline). No data/state migration.

### Signing-session conditions (binding; part of the approved plan text)
Source: OD-TLX-A, OD-TLX-C (Option A), OD-TLX-D signed by SIGN-4_TLX. Narrowing only; no scope added.
1. Test-only. `intergrax/**` MUST NOT change; the diff guard of the Test contract is mandatory evidence.
2. Tests 1-2 change to `ABANDONED` / `"abandoned"` per OD-TLX-A; every non-code assertion (ledger `UNCERTAIN`, replay -> `InvocationUncertaintyError`, effect count `== 1`) MUST be kept. If any of them fails once the code assertion is corrected, STOP and report as a finding; do not weaken it.
3. Tests 3-4 assert `ABANDONED` / `"abandoned"` / `executor.calls == 1` / ledger `UNCERTAIN` as the CURRENT conservative fail-closed behaviour (OD-TLX-C Option A). This is not a finding that NOT_STARTED classification is wrong or right, and it does NOT sanction any durable external-operation record left in RUNNING after a failed `pool.submit` (root-cause section 1.5 stays open and unverified). An implementer who observes such a record reports it as a new finding; no product change under this ticket.
4. N1 is added only if no equivalent test exists (record the existing test id instead).
5. The "Proposed canonical patches" are proposals for the integration owner. This ticket does not publish them and never rewrites the original acceptance/reconciliation verdicts.
6. Replay of the Batch B L2 set, the F8 67/67 inventory and the L3/TL11 17-module set (`--extra mcp`) at the exact post-fix SHA is done by the AUDITOR, never by the implementer (OD-TLX-D).
7. No PASS, closure, FRZ promotion or stage-parent change is implied by this ticket or its approval.

# T-IGX-030-C — HELD — TOOL-LIFE-X: no-boundary `pool.submit` failure terminalises the external-operation record (`29c387879`) + its red-first test (`98b4bf514`)

**HELD - ACT-N11 sequencing (DR-12=B): do not approve or implement until MASTER confirms the IIAC re-audit/re-activation batch.**

## Scope

Product-path batch under the new ID (formerly `T-IGX-024-A`, which collides with the ENT-AUDIT-X stream), landing only on a branch that already contains `T-IGX-030-A`:

1. C3 `29c387879` — `intergrax/runtime/nexus/tools/invoker.py` +5/-1: guard only `pool.submit` in the `dep_boundary is None` branch with `except BaseException: ext_op.mark_failed(); raise`. No new error code, state, API, result mapping or ledger change.
2. C4 `98b4bf514` — new test file `tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py` (red without C3; imports helpers from the `T-IGX-030-A` C1 file).

Exact files: those two. Full plan, order (C3 then C4, red proof first), blast radius, ACT-N11 consequence, test contract, rollback: [`plans/PLAN_T-IGX-030-C.md`](plans/PLAN_T-IGX-030-C.md). Authority (chat, not signature lines): DR-7 option 1, Q-A1 = YES (C4 joins), DR-12 = B (HELD).

No PASS, closure, FRZ promotion or stage-parent change is implied. Independent audit in a fresh session.

## Plan digest

Plan digest: 55d4c9e6e7cc05b27b794e24d71ef1c7ac02a6c4d3026ea838a3deeb007e0103

## Approved

(pending and HELD — no approval line is requested from the Operator for this ticket now; any approval line is written only by the Operator.)

## Test contract

Phase 2 only, after the HOLD is lifted; one process per heavy-lock hold; own exit code captured; logs in `.tmp/session/t-igx-030-c/`:

```text
uv run --extra dev-ci pytest -p no:xdist tests/unit/runtime/nexus/tools/test_tool_life_x_c0_no_boundary_submit_failure_terminal.py -q   # red before C3, green after
uv run --extra dev-ci pytest -p no:xdist <C4 file + side-effect module + Batch B/C modules + 4 invoker/admission modules> -q               # final tree
uv run --extra dev-ci ruff check <2 files> ; uv run --extra dev-ci ruff format --check <2 files>
git diff --name-only <base>..HEAD    # blob-equality guard against 29c387879 / 98b4bf514
```

Informational and expected RED by design (not to be fixed here): `verify_iiac_policy_activation --require-in-force` (ACT-N11).

## Out of scope

`T-IGX-030-A`/`-B` content; any other `intergrax/` hunk; a `bind_future` failure fix (residual R-1 stays); `test_sec_ent.py` (F-006), W5 b2/c/d (F-016); IIAC records/gate; roadmap/FRZ edits; PASS/closure claims; Docker/vendor/production configuration (C10/C15); push, merge, rebase, PR.

## Lifecycle state

plan_pending

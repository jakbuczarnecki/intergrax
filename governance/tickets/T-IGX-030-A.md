# T-IGX-030-A — TOOL-LIFE-X: integrate three unpublished corrective commits under a collision-free ticket ID (HIST-01 N5, Operator DR-7)

## Scope

Mechanical integration, on top of `origin/development` (`492e361c0`), of three local unpublished commits (`cert-wave/publish-v1-with-tlx`):

1. `0adb73380` — test-only: reconcile four stale side-effect-safety tests with ABANDONED semantics (+ new test N2). Formerly `T-IGX-023-A` (ID collides with the published EVOL-X W7-06 ticket).
2. `10089121e` — test-only: streamable-HTTP parent-cancel test made event-driven (+ 2 named negative tests). Formerly `T-IGX-023-B`.
3. `29c387879` — PRODUCT: `intergrax/runtime/nexus/tools/invoker.py` (+5/-1), terminalise the external-operation record when `pool.submit` fails in the no-dependency-boundary branch. Formerly `T-IGX-024-A` (ID used by the ENT-AUDIT-X stream).

Exact files: `tests/unit/runtime/tools/test_tools_side_effect_safety.py`, `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py`, `intergrax/runtime/nexus/tools/invoker.py`. Old ticket/plan files of the source commits are not carried over. Full plan, per-assertion old -> new ledger, alternatives, blast radius and rollback: [`plans/PLAN_T-IGX-030-A.md`](plans/PLAN_T-IGX-030-A.md).

No PASS, closure, FRZ promotion or stage-parent change is implied. Implementer is not the auditor; the independent audit (HIST-01 N6) is run by a fresh session.

## Plan digest

Plan digest: 4680793cfef6ba645f088d05d05ca47455b0ca29743ac035770c5fb83f671d96

## Approved

(pending — the operator approval line is written only by the Operator after plan review; it is intentionally absent from this ticket.)

Operator decision cited as authority for the scope (given in chat 2026-11-08, not a signature line): DR-7 option 1. Open points needing the Operator before approval: plan section 9 (Q1 include `98b4bf514`, Q2 ACT-N11 consequence of the `intergrax/` hunk, Q3 N1).

## Test contract

Phase 2 only, one pytest/ruff process at a time under the campaign heavy lock, pytest's own exit code captured, logs under `.tmp/session/t-igx-030-a/`:

```text
uv run --extra dev-ci pytest -p no:xdist tests/unit/runtime/tools/test_tools_side_effect_safety.py -q
uv run --extra dev-ci --extra mcp pytest -p no:xdist tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py -q    # whole module, 3 separate consecutive runs
uv run --extra dev-ci --extra mcp pytest -p no:xdist <side-effect module + Batch B x3 + Batch C recovery module> -q                  # combined, final tree (exact list in the plan)
uv run --extra dev-ci ruff check <3 files> ; uv run --extra dev-ci ruff format --check <3 files>
git diff --name-only 492e361c0..HEAD    # only the allowed files; blob-equality guard against the source SHAs
```

Informational and expected red (not to be fixed here): `verify_iiac_policy_activation --require-in-force` (ACT-N11, product path changed after the audited SHA).

## Out of scope

`tests/.../test_sec_ent.py` (F-006) and the W5 b2/c/d isolation batch (F-016); the red-first test commit `98b4bf514` unless the Operator extends DR-7; any other `intergrax/` edit; the F-013 fix (`T-IGX-030-B`); ENT-AUDIT-X artefacts; IIAC records/activation; roadmap/FRZ edits; PASS/closure claims; Docker/vendor/production configuration (C10/C15); push, merge, rebase, PR.

## Lifecycle state

plan_pending

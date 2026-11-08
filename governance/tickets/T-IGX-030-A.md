# T-IGX-030-A — TOOL-LIFE-X: integrate two unpublished corrective TEST commits under a collision-free ticket ID (HIST-01 N5, Operator DR-7, re-scoped by DR-12 = B)

## Scope

TESTS-ONLY mechanical integration, on top of `origin/development` (`492e361c0`), of two local unpublished commits (`cert-wave/publish-v1-with-tlx`):

1. `0adb73380` — reconcile four stale side-effect-safety tests with ABANDONED semantics (+ new test N2). Formerly `T-IGX-023-A` (ID collides with the published EVOL-X W7-06 ticket).
2. `10089121e` — streamable-HTTP parent-cancel test made event-driven (+ 2 named negative tests). Formerly `T-IGX-023-B`.

Exact files (nothing under `intergrax/`, `agents/`, `applications/`; ACT-N11 must stay green): `tests/unit/runtime/tools/test_tools_side_effect_safety.py`, `tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py`. Old ticket/plan files of the source commits are not carried over. Full plan, per-assertion old -> new ledger, alternatives, blast radius and rollback: [`plans/PLAN_T-IGX-030-A.md`](plans/PLAN_T-IGX-030-A.md).

The product fix `29c387879` and its red-first test `98b4bf514` are in HELD ticket `T-IGX-030-C`; the F-013 enum move is in HELD ticket `T-IGX-030-B`. ID map: `T-IGX-023-A -> T-IGX-030-A (C1)`, `T-IGX-023-B -> T-IGX-030-A (C2)`, `T-IGX-024-A -> T-IGX-030-C`.

No PASS, closure, FRZ promotion or stage-parent change is implied. The implementer is not the auditor; the independent audit (HIST-01 N6) is run by a fresh session.

## Plan digest

Plan digest: c6d1dca6c129e8f5ce14854698c89e998f600a8d69f11d150649e9591b18e342

## Approved

(pending — the operator approval line is written only by the Operator after plan review; it is intentionally absent from this ticket.)

Operator decisions cited as authority for the scope (given in chat, not signature lines): DR-7 option 1 (two test commits), DR-12 = B (product-path work split out), Q-A3 = no (no N1). No open question blocks approval.

## Test contract

Phase 2 only, one pytest/ruff/uv process per heavy-lock hold, pytest's own exit code captured, logs under `.tmp/session/t-igx-030-a/`, existing `.venv` reused:

```text
uv run --extra dev-ci pytest -p no:xdist tests/unit/runtime/tools/test_tools_side_effect_safety.py -q
uv run --extra dev-ci --extra mcp pytest -p no:xdist tests/unit/applications/test_mcp_catalog_f3_streamable_http_physical_l3.py -q    # whole module, 3 separate consecutive runs
uv run --extra dev-ci --extra mcp pytest -p no:xdist <side-effect module + Batch B x2 + p0_safety_8 + Batch C recovery module> -q        # combined, final tree (exact list in the plan)
uv run --extra dev-ci ruff check <2 files> ; uv run --extra dev-ci ruff format --check <2 files>
git diff --name-only 492e361c0..HEAD    # only the 2 test files + ticket/plan files; blob-equality guard against the source SHAs
# ACT-N11 pre-check after C1+C2 are committed, once each, separate lock holds, expected green:
uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force
uv run --extra dev-ci pytest -p no:xdist tests/unit/governance/test_iiac_activation_gate.py tests/unit/governance/test_iiac_negative_guards.py tests/unit/governance/test_evol_x_real_repo_fresh_clone.py -q
```

## Out of scope

Everything under `intergrax/`, `agents/`, `applications/`; `29c387879` and `98b4bf514` (-> `T-IGX-030-C`); the F-013 fix (-> `T-IGX-030-B`); `test_sec_ent.py` (F-006) and the W5 b2/c/d isolation batch (F-016); N1; ENT-AUDIT-X artefacts; IIAC records/activation and the gate; roadmap/FRZ edits; PASS/closure claims; Docker/vendor/production configuration (C10/C15); push, merge, rebase, PR.

## Lifecycle state

plan_pending

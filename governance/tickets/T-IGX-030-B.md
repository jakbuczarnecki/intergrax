# T-IGX-030-B — HIST02-F-013: restore the EBH-4 R1 nexus-encapsulation gate (move `RuntimeErrorCode` to the contract layer) (Operator DR-5)

## Scope

`tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py::test_ebh_4_r1_production_nexus_imports_outside_ee_are_zero` is red because `intergrax/runtime/tools/tool_operation_recovery.py:22` imports `intergrax.runtime.nexus.errors.error_codes` (TOOL-LIFE-X Batch C, `a5f813a2c`). Fix: define `RuntimeErrorCode` in a new contract-layer module `intergrax/contracts/runtime_error_code.py`, turn `intergrax/runtime/nexus/errors/error_codes.py` into a re-export of the same class (identity preserved, all other importers untouched), and import from the contract module in `tool_operation_recovery.py`. The gate is not weakened, skipped, allow-listed or edited.

Exact files: `intergrax/contracts/runtime_error_code.py` (new), `intergrax/runtime/nexus/errors/error_codes.py`, `intergrax/runtime/tools/tool_operation_recovery.py`, `tests/unit/architecture/test_runtime_error_code_contract_home.py` (new). Full plan, rejected alternatives, importer blast radius, compatibility decision, test contract and rollback: [`plans/PLAN_T-IGX-030-B.md`](plans/PLAN_T-IGX-030-B.md).

No PASS, closure, FRZ promotion (FRZ-BND-01..04 stay OPEN) or stage-parent change is implied. Independent audit follows in a fresh session.

## Plan digest

Plan digest: 9f2a26d18953e10c6193661b581303929ff5ec0132265cf9f96ca5e718f47341

## Approved

(pending — the operator approval line is written only by the Operator after plan review; it is intentionally absent from this ticket.)

Operator decision cited as authority for the scope (given in chat 2026-11-08, not a signature line): DR-5 option 1. Open points needing the Operator before approval: plan section 9 (Q1 ACT-N11 consequence — the fix is necessarily a product-path change; Q2 module location; Q3 alias lifetime).

## Test contract

Phase 2 only, one pytest/ruff process at a time under the campaign heavy lock, pytest's own exit code captured, logs under `.tmp/session/t-igx-030-b/`:

```text
uv run --extra dev-ci pytest -p no:xdist tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py -q    # red proof before, green after; gate file byte-identical
uv run --extra dev-ci pytest -p no:xdist <gate + new test + EBH-2A/2B/2C gates + compat_x R5 inventory consumer> -q
uv run --extra dev-ci pytest -p no:xdist <the 19 test modules importing RuntimeErrorCode> -q
uv run --extra dev-ci ruff check <4 files> ; uv run --extra dev-ci ruff format --check <4 files>
git diff --name-only 492e361c0..HEAD    # only the allowed files
```

Informational and expected red (not to be fixed here): `verify_iiac_policy_activation --require-in-force` (ACT-N11).

## Out of scope

Editing/weakening/skipping/allow-listing the gate; the other failures in `replay_ba07eba.log` (`ebh_5` package C v2 x3, `ue_9br1`); migrating other importers or retiring the old import path; `T-IGX-030-A` content; IIAC/roadmap/FRZ record edits; PASS/closure claims; Docker/vendor/production configuration (C10/C15); push, merge, rebase, PR.

## Lifecycle state

plan_pending

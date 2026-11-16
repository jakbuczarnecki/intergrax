## PLAN `T-IGX-030-B` — HIST02-F-013: restore the EBH-4 R1 nexus-encapsulation gate by moving `RuntimeErrorCode` to the contract layer (Operator decision DR-5 option 1)

**HELD - ACT-N11 sequencing (DR-12=B): do not approve or implement until MASTER confirms the IIAC re-audit/re-activation batch.**

Campaign: Integrax Historical Certification Closure, workstream IMPL-TLX. Implementer != auditor != certifier: this plan claims no PASS, no closure, no FRZ promotion. The gate is NOT weakened, skipped, allow-listed or edited. Operator defaults accepted via MASTER as part of DR-5 option 1: module location `intergrax/contracts/runtime_error_code.py` (former Q2) and keeping the re-export alias (former Q3). The independent auditor re-checks both. File-disjoint from HELD `T-IGX-030-C` (`invoker.py`), which imports `RuntimeErrorCode` through the alias, so either order works and one IIAC cycle can cover both.

### 1. Defect (evidence, re-verified read-only at HEAD `492e361c0`)

- `tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py::test_ebh_4_r1_production_nexus_imports_outside_ee_are_zero` was RED at `ba07ebaa3` (`replay_ba07eba.log`: `runtime.nexus imports outside EE: ['intergrax/runtime/tools/tool_operation_recovery.py']`).
- Cause: `intergrax/runtime/tools/tool_operation_recovery.py:22` has `from intergrax.runtime.nexus.errors.error_codes import RuntimeErrorCode` (TOOL-LIFE-X Batch C, `a5f813a2c`). `runtime/tools/` is outside the EE zone (`runtime/execution`, `runtime/nexus`).
- The enum is defined at `intergrax/runtime/nexus/errors/error_codes.py:8` (`class RuntimeErrorCode(str, Enum)`, 13 members, only imports `enum`). It is a shared vocabulary, not Nexus logic.
- Product-path diff `ba07ebaa3..HEAD` is empty, so the defect is unchanged at HEAD. A static grep shows `tool_operation_recovery.py` is the only real `.py` importer of `intergrax.runtime.nexus` outside the EE zone in `intergrax/`, `agents/`, `applications/` (the other hits are docstrings/strings/notebooks).

### 2. Design choice

Move the class DEFINITION to a new contract-layer module and keep `error_codes.py` as a thin re-export.

- New file `intergrax/contracts/runtime_error_code.py`: the 13 members verbatim (same names, same string values, same order, same bases `(str, Enum)`), license header, short docstring, `__all__ = ["RuntimeErrorCode"]`. Imports only `enum` (contract layer must not import `runtime/`).
- `intergrax/runtime/nexus/errors/error_codes.py`: replaced by `from intergrax.contracts.runtime_error_code import RuntimeErrorCode` + `__all__ = ["RuntimeErrorCode"]`. This is an alias, not a copy: both import paths yield the SAME class object (`is` identity, `isinstance`, equality with the str value, JSON/enum lookup all unchanged).
- `intergrax/runtime/tools/tool_operation_recovery.py:22`: import changed to `from intergrax.contracts.runtime_error_code import RuntimeErrorCode`. No other line in the file changes.

### 3. Alternatives rejected

| Alternative | Why rejected |
|---|---|
| Rewrite all 17 `intergrax/` importers (and 19 test modules) to the new path | Large blast radius for no gate benefit; the other importers are inside the EE zone or go through it. Can be done later mechanically. |
| Move `tool_operation_recovery.py` into `runtime/nexus/` | Breaks the `runtime/tools` seam and its importers (`nexus/tools/invoker.py`, `test_tool_life_x_batch_c_stalled_abandoned_recovery.py`, the compat_x R5 inventory key `intergrax/runtime/tools/tool_operation_recovery.py`); wrong direction for a pure vocabulary type. |
| Import via the existing EE bridge `intergrax/runtime/execution/idempotency_runtime_state_bridge.py` (it already re-exports `RuntimeErrorCode`; `idempotency_pre_effect_coordinator.py` does this) | One-line change but it only launders a Nexus dependency through an EE-zone file; the finding and DR-5 ask for a contract-layer home. Kept only as a documented fallback. |
| Function-local / `TYPE_CHECKING` / dynamic import | The gate walks `ast.walk` (every node); a lazy import is still flagged and would be a dodge. |
| Add `tool_operation_recovery.py` to a gate allow-list, edit the gate, or `skip`/`xfail` it | Forbidden by DR-5. |
| Duplicate the enum in the contract layer, leave Nexus' copy | Two classes: identity/equality breaks across `isinstance` and dict-key use. Rejected. |

### 4. Compatibility decision

Re-export (alias) is kept at `intergrax.runtime.nexus.errors.error_codes.RuntimeErrorCode` indefinitely in this ticket: it guarantees zero behavioural change for the 16 other `intergrax/` modules, `invoker.py`'s four function-local imports, `idempotency_runtime_state_bridge.py`, and the 19 test modules. Retiring the old path is a separate, later ticket. One observable difference: `RuntimeErrorCode.__module__` changes to `intergrax.contracts.runtime_error_code`; phase 2 must `git grep` for `__module__`/pickle/qualname dependence on the old path (phase-1 grep found none in `intergrax/`) and for string references in tests/docs. The only non-code references to the old file are in `docs/project/maintainers/qualification/IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json` (a governance record: not edited here).

### 5. Blast radius (importer list, from `git grep` at HEAD; phase 2 recomputes and records the final list)

Real importers of `intergrax.runtime.nexus.errors.error_codes` in `intergrax/` (all keep working through the alias):
`runtime/execution/host_runtime_config.py`, `runtime/execution/idempotency_runtime_state_bridge.py`, `runtime/nexus/config_sections.py`, `runtime/nexus/errors/classifier.py`, `runtime/nexus/errors/harness_error_taxonomy.py`, `runtime/nexus/orchestration/graph_runner.py`, `runtime/nexus/retry/coordinator.py`, `runtime/nexus/tools/agent_governance_approval_pause_bridge.py`, `runtime/nexus/tools/invoker.py` (1 module-level + 4 function-level), `runtime/nexus/tools/mse_governed_continuation_hitl_bridge.py`, `runtime/nexus/tracing/runtime/runtime_run_retry.py`, `runtime/nexus/tracing/tools/tool_invocation.py`, `runtime/tools/tool_operation_recovery.py` (the only one edited). Indirect via the bridge: `runtime/tools/idempotency_pre_effect_coordinator.py`. Tests importing `RuntimeErrorCode`: 19 modules + helper `tests/system/tools_side_effect_safety/shared/runtime_worker.py` (unchanged). `agents/`, `applications/`, `scripts/`: no `.py` importer.

### 6. Exact files (edit list for phase 2; nothing else may change)

| Path | Action |
|---|---|
| `intergrax/contracts/runtime_error_code.py` | NEW |
| `intergrax/runtime/nexus/errors/error_codes.py` | replace body with the alias |
| `intergrax/runtime/tools/tool_operation_recovery.py` | one import line |
| `tests/unit/architecture/test_runtime_error_code_contract_home.py` | NEW regression test (below) |

Commit trailer: `Ref: governance/tickets/T-IGX-030-B.md`. Two commits: (1) the product change (3 files under `intergrax/`), (2) the regression test. The red proof is the pre-change gate run in step 1 of section 7.

### 7. Test contract (phase 2; ONE pytest/ruff process at a time under `<campaign>\locks\heavy.lock`; capture pytest's OWN exit code via `; echo EXIT=$?` into the log; logs in `.tmp/session/t-igx-030-b/`; no `--basetemp` outside the repo)

New regression test `tests/unit/architecture/test_runtime_error_code_contract_home.py` (positive and negative):
- identity: `intergrax.contracts.runtime_error_code.RuntimeErrorCode is intergrax.runtime.nexus.errors.error_codes.RuntimeErrorCode`;
- frozen vocabulary: the exact 13 `(name, value)` pairs, order included, and `issubclass(..., str)`;
- contract purity (AST): the contract module imports nothing from `intergrax.runtime`, `agents`, `applications` (negative control: an AST check on a synthetic offending snippet is detected);
- no Nexus leak (AST): `tool_operation_recovery.py` contains no `intergrax.runtime.nexus` import (independent of the main gate, no allow-list).

Commands:
1. Red proof BEFORE the change: `uv run --extra dev-ci pytest -p no:xdist tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py -q` expected 1 failed (the production-imports test), record the log.
2. After the change, the same command: expected all green; the gate file must be byte-identical to HEAD (`git diff --stat -- tests/unit/architecture/test_ebh_4_r1_nexus_encapsulation_gate.py` empty).
3. One combined run: gate + new test + contract/layer gates `tests/unit/architecture/test_ebh_2a_public_contract_boundary_gate.py`, `test_ebh_2b_agent_nexus_contract_separation.py`, `test_ebh_2c_runtime_capability_ports.py`, plus the test module that consumes `tests/qualification/compat_x/_compat_x_deprecation_r5_inventory.py` (find with `git grep -l _compat_x_deprecation_r5_inventory -- tests`, record the name).
4. One combined run of the 19 test modules that import `RuntimeErrorCode` (`git grep -l RuntimeErrorCode -- tests`, minus the helper), with `--extra mcp` only if one of them needs it.
5. Lint: `uv run --extra dev-ci ruff check` then `ruff format --check` on the 4 files.
6. Diff guard: `git diff --name-only 492e361c0..HEAD` lists only the 4 files plus the ticket and plan (and the `T-IGX-030-A` files if that ticket is already on the branch).
7. Informational, expected red, DO NOT FIX: `uv run python -m scripts.governance.verify_iiac_policy_activation --require-in-force` — ACT-N11 will report the changed `intergrax/` paths (3 files). Record the lines verbatim.
8. `igx_gov ticket` if feasible.
Independent replay of the full QUAL-X P5 Dimension A manifest / EBH replay belongs to the auditor on a clean pinned checkout (needs Operator budget, C10/C15).

Phase-2 pre-checks that may change the plan (STOP and report, do not improvise): (a) `scripts/governance/iiac_g0_inventory.py` / `verify_iiac_g0_inventory_completeness.py` or the arch-freeze closed-world inventory must not fail on a NEW file under `intergrax/contracts/` (phase 1 did not execute them); (b) the EBH-2A public-contract boundary gate must accept the new contract module.

### 8. Rollback

`git revert` of the two commits restores the pre-change (gate-red) state; the alias design means no data/state migration.

### 9. Open questions for the Operator

- Q1 ACT-N11: the fix is unavoidably a product-path change (`intergrax/`). It makes the product-path diff vs the audit SHA `ae467c8c4` 3 files and turns ACT-N11 red (plus the governance-fast tests that run it). Resolved by Operator DR-12 = B: this ticket is HELD until MASTER confirms the IIAC re-audit/re-activation batch. The gate is never relaxed.
- Q2 (resolved, accepted default): module `intergrax/contracts/runtime_error_code.py`.
- Q3 (resolved, accepted default): keep the re-export alias; retiring it is a separate later ticket.

### 10. Out of scope

Editing/weakening/skipping/allow-listing the gate or any other test; the other four failures seen in `replay_ba07eba.log` (`ebh_5` package C v2 x3, `ue_9br1`) — separate findings; migrating the other importers; roadmap/FRZ/IIAC record edits (incl. `IIAC_GLOBAL_AUTHORITY_IMPACT_MATRIX.json`); `T-IGX-030-A` content; FRZ promotion; PASS/closure claims; Docker/vendor/production config (C10/C15); pushing, merging, opening PRs.

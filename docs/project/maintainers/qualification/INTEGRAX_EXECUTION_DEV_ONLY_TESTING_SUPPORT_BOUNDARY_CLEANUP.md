# INTEGRAx-EXECUTION-DEV-ONLY-TESTING-SUPPORT-BOUNDARY-CLEANUP

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `INTEGRAx-EXECUTION-DEV-ONLY-TESTING-SUPPORT-BOUNDARY-CLEANUP` |
| Follow-up | F-01 (non-blocking debt from Final Architecture Closure) |
| Class | A — dependency hygiene / ownership cleanup |
| Execution Engine | FROZEN — no semantic changes |

## Scope

Remove all `intergrax/` → `testing_support/` imports. Canonicalize shared lab/scaffold/test bootstrap helpers under `intergrax/dev_support/`. Preserve `testing_support` as a thin re-export consumer.

## Repository State

Cleanup branch work on `development`; unrelated WIP (memory entity graph) excluded from this commit.

## Baseline SHA

`325a014c92f1f7b80a7c8ce6ff4d7d31783996b5`

## Import Inventory

| File | Imported symbol | Why needed | Runtime/dev/test | Correct owner |
| --- | --- | --- | --- | --- |
| `intergrax/lab/organization_worker.py` | `build_organization_worker_registry` | §38 lab demo default registry | dev/lab | `intergrax.dev_support.agent_registry_bootstrap` |
| `intergrax/scaffold/new_agent.py` (generated test template) | `canonical_execution_identity_scope` | Agent smoke test identity binding | dev/scaffold → generated tests | `intergrax.dev_support.execution_identity_scope` |
| `intergrax/experiments/workflow.py` (docstring example only) | `build_harness_registry` | Notebook/session example | dev/docs | `intergrax.dev_support.agent_registry_bootstrap` |

Post-audit: **0** executable `intergrax → testing_support` imports (docstring reference updated).

## Finding Classification

| Location | Class |
| --- | --- |
| Lab organization worker registry | **A** — reusable neutral functionality |
| Scaffold generated agent test | **A** — reusable neutral functionality (identity scope) |
| Experiments workflow docstring | **D** — accidental convenience reference in docs |

## Ownership Analysis

| Symbol | Canonical owner | Rationale |
| --- | --- | --- |
| `bootstrap_agent_registry_from_agents`, `build_*_registry` | `intergrax/dev_support/agent_registry_bootstrap.py` | Platform dev/lab composition; Tier-2 agent loading for non-production paths |
| `canonical_run_id_for_tests`, `canonical_execution_identity_scope` | `intergrax/dev_support/execution_identity_scope.py` | Contract-aligned identity binding shared by tests and scaffold |
| `testing_support.agent_registry_bootstrap` | Shim re-export only | Harness backward compatibility |
| `testing_support.builder` | Re-exports identity helpers from dev_support | Harness backward compatibility |

## Selected Design

Introduce `intergrax/dev_support/` as the neutral development-support layer (no pytest, no qualification authority). Move implementations from `testing_support`; reverse dependency so harness imports platform dev-support.

## Dependency Direction Before

```text
intergrax/lab|scaffold|experiments (docs)
    ↓
testing_support (agent_registry_bootstrap, builder)
    ↓
intergrax/runtime, intergrax/contracts
```

## Dependency Direction After

```text
intergrax/lab|scaffold|experiments
    ↓
intergrax/dev_support
    ↓
intergrax/contracts, intergrax/runtime (public surfaces)

testing_support
    ↓
intergrax/dev_support (re-export shims)
```

## Contracts / Abstractions

No new Protocols — existing `AgentRegistry`, execution identity contracts unchanged. Dev-support functions are concrete helpers at a non-production boundary.

## Pluginability Assessment

Unchanged. Registry bootstrap remains explicit composition; no service locator added.

## Layer Boundary Assessment

`intergrax/dev_support` is not wired into production runtime composition roots. `runtime/registry` remains free of Tier-2 agent knowledge (existing gates).

## Production Runtime Impact

**None.** No changes under `intergrax/runtime/execution` or frozen execution paths.

## Qualification Impact

**None.** Qualification DAG authority unchanged; `tests/unit/testing_support/execution_qualification/` PASS.

## Execution Engine Impact

**None.** Frozen semantics preserved.

## Import Cycle Assessment

Cold import verified: `intergrax.dev_support.*`, `intergrax.lab.organization_worker`. No new cycles.

## Architecture Guard

`tests/unit/runtime/architecture/test_intergrax_no_testing_support_import_gate.py` — AST scan of all `intergrax/**/*.py` forbids `testing_support` imports; includes synthetic detection self-test.

## Tests

| Gate | Result |
| --- | --- |
| New intergrax → testing_support guard | PASS |
| Diagnostics import hygiene | PASS |
| EE final arch (`test_ee_final_arch_*`) + U5 zero-bypass | PASS (37) |
| Qualification regression | PASS (216 passed, 7 skipped) |
| Targeted: experiments, org worker, agent registry | PASS |
| UE-10R4.1 execution local-import hygiene | **Pre-existing FAIL** on HEAD (unrelated local imports in `runtime/execution`) |

## Static Quality

Ruff check/format and pyright on `intergrax/dev_support` and touched modules: PASS.

## Changed Files

- `intergrax/dev_support/__init__.py`
- `intergrax/dev_support/agent_registry_bootstrap.py`
- `intergrax/dev_support/execution_identity_scope.py`
- `intergrax/lab/organization_worker.py`
- `intergrax/experiments/workflow.py`
- `intergrax/scaffold/new_agent.py`
- `testing_support/agent_registry_bootstrap.py`
- `testing_support/builder.py`
- `tests/unit/runtime/architecture/test_intergrax_no_testing_support_import_gate.py`
- This document

## Findings

F-01 resolved: dev-only reverse dependency eliminated with canonical ownership in `dev_support`.

## Remaining Debt

- UE-10R4.1 local-import violations in `intergrax/runtime/execution` (pre-existing, out of F-01 scope).
- Legacy / reference surface cleanup (roadmap item after F-01).

## Decision

Proceed with Class A ownership move + AST guard; no Execution Engine reopen.

## Commit SHA

_(filled after commit)_

## Final Verdict

### PASS

```text
DEV-ONLY TESTING_SUPPORT BOUNDARY CLEANUP = PASS
```

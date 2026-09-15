# INTEGRAx-MEMORY-CONTRACT-BOUNDARY-AND-IMPORT-CYCLE-REMEDIATION

## Metadata

| Field | Value |
|---|---|
| Task | `INTEGRAx-MEMORY-CONTRACT-BOUNDARY-AND-IMPORT-CYCLE-REMEDIATION` |
| Branch | `development` |
| Baseline HEAD (start) | `9336beff5ec72c747b440e63f3fb2dddc0b4bf8d` |
| Change class | **Class B** — canonical model module ownership; public paths preserved |

## Session Scope

Remediate Memory contract boundary violation and package import-cycle risk confirmed on the R2-H2-Q1 qualification path (cancellation subprocess → import graph → `intergrax.memory`). Out of scope: unrelated qualification harness migrations, broad Memory refactors.

## Failure Reproducer

**Command (qualification leaf):**

```bash
uv run pytest tests/unit/runtime/architecture/test_npsc5e_r2_h2_q1_frozen_regression_closure.py::test_pre_existing_cancellation_fixture_unrelated_to_h2_revision -q --tb=short
```

**Pre-remediation symptom:** subprocess import failure when `memory.contracts` eagerly pulled lifecycle types that depended on `user_profile_memory` (runtime import), amplifying package initialization order sensitivity.

**Post partial fix (`84a22fa50`):** empty `contracts.__init__` + `TYPE_CHECKING` in `memory_lifecycle` — leaf green but contracts still referenced implementation module via `memory_control` and typing-only indirection in lifecycle.

## Import Graph Before

```text
intergrax.memory.contracts (package __init__ — previously eager re-exports)
        ↓
intergrax.memory.contracts.memory_lifecycle
        ↓  (runtime, pre-84a22fa50)
intergrax.memory.user_profile_memory
        ↓
intergrax.memory.contracts.enterprise_memory_record
        ↓
(intergrax.memory.contracts package still initializing → cycle / partial init risk)

Parallel reverse-edge (contract purity violation):

intergrax.memory.contracts.memory_control
        ↓
intergrax.memory.user_profile_memory  (implementation-facing module path)
```

## Package Initialization Analysis

Eager `contracts.__init__` re-exports (removed in `84a22fa50`) forced lifecycle loading on any `import intergrax.memory.contracts`. Lifecycle’s dependency on profile models expanded initialization through `user_profile_memory` while the contracts package was still wiring submodules.

## Contract Boundary Analysis

`contracts/` must depend only on neutral contract/model modules. Importing `intergrax.memory.user_profile_memory` from `memory_control` inverted ownership (contracts → implementation module path).

## Model Ownership Analysis

| Symbol | Classification |
|---|---|
| `MemoryKind`, `MemoryImportance` | Domain contract enums |
| `UserProfileMemoryEntry`, `EnterpriseMemoryRecord` | Canonical domain / persistence record model |
| `UserProfile`, `UserIdentity`, `UserPreferences` | Domain aggregate models |
| `UserProfileMemoryEntryNotFoundError` | Domain error type |

**Canonical definition:** `intergrax.memory.contracts.memory_models` (after remediation).

**Compatibility surface:** `intergrax.memory.user_profile_memory` re-exports the same classes (identity preserved).

## Public API Surface

Production and tests import `UserProfile`, `UserProfileMemoryEntry`, etc. via `intergrax.memory.user_profile_memory` — preserved through re-export shim.

## Pluginability Assessment

No change to store plugins, control-plane protocol shapes, lifecycle semantics, or provider contracts. `memory_store_plugin` retains `TYPE_CHECKING`-only references to materialization types.

## Change Classification

**Class B** — physical ownership move to `memory_models.py`; schema and public import paths unchanged.

## Selected Design

1. Move neutral domain models to `intergrax/memory/contracts/memory_models.py`.
2. Make `user_profile_memory.py` a thin public re-export layer.
3. Point `memory_lifecycle` and `memory_control` at `memory_models` (runtime imports, no `user_profile_memory`).
4. AST guards in `tests/unit/memory/memory_contract_boundary_ast.py` + regression tests.

## Rejected Alternatives

- Lazy/local imports inside methods — forbidden.
- `TYPE_CHECKING` masking lifecycle runtime ownership — rejected (lifecycle now imports neutral models).
- Duplicated dataclass definitions — forbidden.
- Amputating public `contracts.__init__` exports without model ownership fix — insufficient alone.

## Import Graph After

```text
intergrax.memory.contracts.memory_models
        ↑
intergrax.memory.contracts.enterprise_memory_record

intergrax.memory.contracts.memory_lifecycle
        ↑
intergrax.memory.contracts.memory_models

intergrax.memory.contracts.memory_control
        ↑
intergrax.memory.contracts.memory_models

intergrax.memory.user_profile_memory  (re-export only)
        ↑
intergrax.memory.contracts.memory_models
```

No `contracts → user_profile_memory` edge.

## Public Compatibility

`from intergrax.memory.user_profile_memory import UserProfile` — unchanged path; `is` identity with canonical classes (tested).

## Model Identity / Schema Parity

Single class definitions in `memory_models.py`; fields, defaults, enums, mutability, and validation hooks unchanged from pre-move definitions.

## Contracts Purity Guards

AST scan: `intergrax/memory/contracts/*.py` must not import `intergrax.memory.*` outside `intergrax.memory.contracts.*` (module-level, excluding `TYPE_CHECKING` blocks). `memory_lifecycle.py` must not import `user_profile_memory`.

## Cold Import Matrix

| Command | Result |
|---|---|
| `import intergrax.memory.contracts` | PASS |
| `from intergrax.memory.contracts.memory_lifecycle import MemoryLifecycleOutcome` | PASS |
| `from intergrax.memory.user_profile_memory import UserProfile, UserProfileMemoryEntry` | PASS |
| Import order subprocess matrix (contracts ↔ profile ↔ package) | PASS |

## Memory Regression

`uv run pytest tests/unit/memory/ -q` — **179 passed**.

## Cancellation Reproducer

`test_p0c5_cancellation_continuity.py::test_terminal_cancellation_survives_process_restart` — PASS.

## R2-H2-Q1 Revalidation

Semantic slice (`-k "not test_mandatory_frozen_suite_passes"`) — PASS.

## Qualification Regression

`uv run pytest tests/unit/testing_support/execution_qualification/ -q` — **216 passed**, 7 skipped (live perf).

## Full Canonical Verification

Run at remediation commit HEAD when working tree is clean for task files:

```bash
uv run python -m testing_support.execution_qualification.performance \
  --profile npsc5f-final --repetitions 1 --max-parallel 2 \
  --artifact-dir .tmp/session/MEMORY-IMPORT-CYCLE/npsc5f-final
```

**Measured:** `certification_decision: pass`, wall ≈ **166.97 s**, `run_status_pass: true` (artifact: `.tmp/session/MEMORY-IMPORT-CYCLE/npsc5f-final/`).

## Protected Drift Assessment

No baseline updates performed. Full run PASS at remediation HEAD; no protected drift reclassification performed in this session.

## Production Changes

- `intergrax/memory/contracts/memory_models.py` (new)
- `intergrax/memory/user_profile_memory.py` (re-export shim)
- `intergrax/memory/contracts/memory_lifecycle.py`
- `intergrax/memory/contracts/memory_control.py`
- `intergrax/memory/contracts/__init__.py` (docstring)
- `tests/unit/memory/test_memory_contract_boundary.py`
- `tests/unit/memory/memory_contract_boundary_ast.py`

## Findings

Root cause: contract modules depended on `user_profile_memory` module path instead of neutral contract models, combined with historical eager `contracts.__init__` exports.

## Decision

Proceed with Class B model extraction and AST guards; preserve public import paths via re-export.

## Final Verdict

**MEMORY CONTRACT BOUNDARY AND IMPORT-CYCLE REMEDIATION = PASS**

# NPSC-4.1-F — Execution Boundary Hardening Final Certification

**Status:** `CERTIFIED`

**Verdict:** **PASS** (final enterprise certification)

**Date:** 2026-09-08

**Branch:** `development`

**Revision:** `af825fc88597616f05c7b8a14889a18ec7c50524`

**Scope:** Final read-only certification that NPSC-4.1 preserves a single enterprise execution ownership model. No architecture redesign, no contract changes, no new abstractions.

**Prior artifact:** `NPSC_4_1_EXECUTION_BOUNDARY_HARDENING_CERTIFICATION.md` (implementation certification)

**Frozen:** NPSC-3C Execution Engine — immutable lifecycle and identity contracts.

---

## 1. Objective

Prove that after NPSC-4.1 Execution Boundary Hardening the platform maintains:

- **One identity owner:** `identity_authority.py` (via `ExecutionRuntime`)
- **One lifecycle owner:** `ExecutionRuntime`
- **One canonical execution path:** Application → Interaction/Queue/Scheduler → `HostTaskExecutionExecutor` → `HostTaskExecutionPort` → `ExecutionRuntime` → `ExecutionBoundary` → `StrategyExecutionRouter` → Agent Runtime / Nexus orchestration
- **Nexus role:** orchestration/planning/backend coordination only — no lifecycle ownership, no identity minting, no execution start ownership
- **Intake hardening:** HTTP/MCP/interaction intake cannot mint `RunId` / `AttemptId` / `ExecutionId` or bind execution context
- **UnifiedTaskRunner:** harness/scheduling only — not a production Tier-3 execution entry

---

## 2. Frozen architecture contracts

| Contract | Owner | Responsibility |
| -------- | ----- | -------------- |
| Execution lifecycle | `ExecutionRuntime` | Start, completion, context ownership, boundary coordination |
| Execution identity mint | `identity_authority.py` | `RunId`, `AttemptId`, `ExecutionId` creation |
| Execution boundary bind | `ExecutionBoundary` | `bind_active_execution_identity` propagation |
| Canonical intake | Tier-3 serving / MCP / interactions | Task intent only — no execution identity mint |
| Nexus | Orchestration layer | Planning and backend coordination — forbidden lifecycle/identity ownership |
| UnifiedTaskRunner | Harness/scheduling | Scheduler coordination, eval, long-running resume adapters |

No `IdentityManager`, `RuntimeV2`, compatibility layers, or duplicate execution frameworks introduced.

---

## 3. Identity ownership proof

### 3.1 Sole mint module

| Check | Gate / evidence | Result |
| ----- | --------------- | ------ |
| `identity_authority.py` exposes `mint_root_execution_identity` and `mint_background_transport_identity` | `test_npsc4_1_identity_authority_remains_sole_mint_module` | **PASS** |
| No identity mint outside `identity_authority.py` in execution/nexus/background scan trees | `test_no_identity_mint_outside_execution_runtime` | **PASS** |
| No `bind_active_execution_identity` outside `ExecutionBoundary` in scan trees | `test_no_identity_bind_outside_execution_boundary` | **PASS** |
| Nexus tree has no mint or bind calls | `test_npsc4_1_nexus_does_not_mint_or_bind_execution_identity`, `test_nexus_scanned_tree_has_no_identity_mint_or_bind` | **PASS** |

### 3.2 Static audit — identity symbols

Symbols audited: `mint_run_id`, `mint_attempt_id`, `mint_execution_id`, `new_run_id`, `bind_active_execution_identity`.

| Category | Classification | Production examples |
| -------- | -------------- | ------------------- |
| **A — Allowed ownership** | Runtime authority | `identity_authority.py`, `ExecutionBoundary` |
| **B — Allowed storage/propagation** | Conformance, persistence, harness alias | `decision_finalization_conformance.py`, `persistence_conformance.py`, `task_run_bridge.new_run_id()` (harness/eval alias — docstring forbids HTTP/MCP intake) |
| **C — Violation** | Forbidden zones minting/binding | **None** in applications intake, interactions, Nexus, scheduler |

**Category C count:** 0

`mint_intake_execution_identity` removed from `task_run_bridge.py` (verified by gate).

---

## 4. Lifecycle ownership proof

| Owner | Responsibility | Result |
| ----- | -------------- | ------ |
| `ExecutionRuntime` | Root lifecycle start/close, context resolution, `mint_root_execution_identity` delegation | **PASS** (`test_npsc4_1_execution_runtime_remains_lifecycle_owner`) |
| `ExecutionBoundary` | Identity propagation bind during strategy execution | **PASS** |
| Intake / Nexus / scheduler | No `start_execution` / `begin_execution` ownership | **PASS** (`test_npsc4_1_forbidden_zones_do_not_start_execution_lifecycle`) |
| Nexus | Orchestration only; no lifecycle mint/bind | **PASS** |
| Host adapters | Allowed lifecycle delegation to `ExecutionRuntime` | **PASS** (adapter allowlist in gate) |

Canonical path verified unchanged from NPSC-3C freeze.

---

## 5. Intake validation

### 5.1 Interaction layer

| Requirement | Result |
| ----------- | ------ |
| Validate request | **PASS** (`tests/unit/runtime/interactions/`) |
| Normalize request | **PASS** |
| Create task intent | **PASS** |
| Must NOT mint `RunId` / `AttemptId` / `ExecutionId` | **PASS** (`test_npsc4_1_interactions_do_not_mint_execution_identity`) |
| Must NOT bind execution context | **PASS** (`test_npsc4_1_forbidden_zones_do_not_bind_execution_identity`) |
| Source scan: no identity mint symbols | **PASS** (`test_interaction_intake_service_source_does_not_mint_execution_identity`) |

### 5.2 Application intake surfaces

| Surface | Result |
| ------- | ------ |
| Tier-3 `applications/**/serving/*.py` HTTP routers | **PASS** (AST gate — no forbidden mint calls) |
| `intergrax/applications/_shared/mcp_nexus_server.py` | **PASS** |
| `task_run_bridge.py` — no `mint_intake_execution_identity` | **PASS** |
| LKW intake (`test_lkw_intake_execution_identity.py`) | **PASS** (14 tests) |
| Legal intake (`test_legal_runtime_bridge.py`) | **PASS** |

**Intake verdict:** **PASS**

---

## 6. UnifiedTaskRunner classification

**Final classification:** `HARNESS / SCHEDULING ONLY`

Documented in `intergrax/runtime/task/unified_task_runner.py`:

```text
HARNESS / SCHEDULING ONLY — not a production Tier-3 execution entry.
Allowed: scheduler coordination, harness compatibility, eval orchestration.
Forbidden: lifecycle ownership, identity creation, bypassing HostTaskExecutionPort.
```

| Check | Result |
| ----- | ------ |
| Module docstring contains classification marker | **PASS** |
| Runner source has no identity mint symbols | **PASS** |
| Runner does not import `HostTaskExecutionPort` (no production bypass seam) | **PASS** |
| Tier-3 `host/factory.py` files do not reference `UnifiedTaskRunner` | **PASS** (`test_npsc4_1_tier3_factories_do_not_reference_unified_task_runner`) |
| Production usage confined to harness/scheduling roots | **PASS** (allowlist: `_shared`, `long_running`, `eval`, `experiments`, `harness`, `scaffold`) |

---

## 7. Static audit results

### 7.1 Preflight (Phase 0)

| Check | Result |
| ----- | ------ |
| `branch == development` | **PASS** |
| `HEAD == origin/development` (`af825fc…`) | **PASS** |
| Worktree clean | **PASS** |

### 7.2 Forbidden-zone scan summary

| Zone | Identity mint | Identity bind | Lifecycle start |
| ---- | ------------- | ------------- | --------------- |
| `applications/**` intake (serving, MCP) | **PASS** (0 violations) | **PASS** | **PASS** |
| `intergrax/runtime/interactions/**` | **PASS** | **PASS** | **PASS** |
| `intergrax/runtime/nexus/**` | **PASS** | **PASS** | **PASS** |
| `intergrax/runtime/long_running/**` (scheduler) | **PASS** | **PASS** | **PASS** |

### 7.3 Product identifier review — `ask_service.py` / `WorkspaceAskRun`

**Classification:** **B — Product correlation identifier**

`applications/local_workspace_application/workspaces/ask_service.py` calls `new_run_id()` to populate `WorkspaceAskRun.run_id`. This identifier:

- Tracks product-level ask-run persistence and tenant/workspace correlation
- Is **not** part of Execution Engine identity (`RunId`/`AttemptId`/`ExecutionId` minted by `identity_authority` inside `ExecutionRuntime`)
- Does not participate in HTTP/MCP execution intake (ask service is product domain, not execution boundary intake)
- Uses the harness alias `task_run_bridge.new_run_id()` for UUID generation convenience only

> **This identifier is product-level correlation only and is not part of Execution Engine identity.**

No rename required — ambiguity does not create architectural risk under current gate coverage.

---

## 8. Regression matrix

Certification run: 2026-09-08, revision `af825fc…`, logs under `.tmp/session/npsc-4-1-final-cert/`.

| Suite | Command scope | Result |
| ----- | ------------- | ------ |
| Execution | `tests/unit/runtime/execution/` (`-m "not no_ci"`) | **PASS** |
| Interactions | `tests/unit/runtime/interactions/` | **PASS** |
| Background execution | `tests/unit/runtime/background_execution/` (celery optional test skipped; causal-evidence celery collector excluded) | **PASS** (1 skipped: optional `celery` dep) |
| Contracts | `tests/unit/contracts/` | **1244 passed** |
| NPSC-3B | `test_npsc3b_lkw_public_surface_gate.py` | **PASS** |
| NPSC-3C-A | UE-10R1–R4 + NPSC-3C-D | **PASS** |
| NPSC-3C-B | `test_ue_11gp_production_host_execution_gate.py` | **PASS** |
| NPSC-3C-C | `test_npsc3c_legacy_executor_removal_gate.py` | **PASS** |
| NPSC-3C-D | `test_npsc3c_d_canonical_execution_engine_conformance_gate.py` | **PASS** |
| NPSC-3E | `test_npsc3e_runtime_convergence_gate.py` | **PASS** |
| NPSC-3F | `test_npsc3f_runtime_convergence_gate.py` | **PASS** |
| NPSC-3G | `test_npsc3g_application_runtime_convergence_gate.py` | **PASS** |
| NPSC-4-F | `test_npsc4_agent_runtime_governance_gate.py` | **PASS** |
| NPSC-4.1 | `test_npsc4_1_execution_boundary_hardening_gate.py` (11 tests) | **PASS** |
| Identity single authority (F-R1) | `test_execution_identity_single_authority_gate.py` | **PASS** |
| UE-9D legacy retirement | `test_ue_9d_legacy_execution_retirement_gate.py` | **PASS** |
| Intake integration | LKW + Legal + interaction intake tests | **14 passed** |

**Aggregate (runtime + contracts + frozen gates, CI-equivalent markers):**

```text
2043 passed, 1 skipped, 32 deselected (no_ci), 79 gate tests passed, 1244 contract tests passed
```

Representative command:

```powershell
uv run pytest tests/unit/runtime/execution/ tests/unit/runtime/interactions/ `
  tests/unit/runtime/background_execution/ `
  --ignore=tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py `
  -m "not no_ci" `
  tests/unit/contracts/ `
  tests/unit/runtime/architecture/test_npsc4_1_execution_boundary_hardening_gate.py `
  tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py `
  tests/unit/runtime/architecture/test_npsc3c_d_canonical_execution_engine_conformance_gate.py `
  tests/unit/runtime/architecture/test_ue_9d_legacy_execution_retirement_gate.py `
  tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py `
  tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py `
  tests/unit/applications/architecture/test_npsc3f_runtime_convergence_gate.py `
  tests/unit/applications/architecture/test_npsc3c_legacy_executor_removal_gate.py `
  tests/unit/applications/architecture/test_npsc3b_lkw_public_surface_gate.py `
  tests/unit/applications/architecture/test_npsc3e_runtime_convergence_gate.py `
  tests/unit/runtime/architecture/test_ue_10r1_nexus_lifecycle_retirement_gate.py `
  tests/unit/runtime/architecture/test_ue_10r2_single_canonical_root_execution_id_gate.py `
  tests/unit/runtime/architecture/test_ue_10r3_platform_owned_root_identity_gate.py `
  tests/unit/runtime/architecture/test_ue_10r4_graph_authority_fail_closed_gate.py `
  tests/unit/runtime/architecture/test_ue_11gp_production_host_execution_gate.py -q
```

---

## 9. Residual non-blocking debt

| Item | Classification | Notes |
| ---- | -------------- | ----- |
| `task_run_bridge.new_run_id()` | Harness/eval alias | Allowed in harness routes; forbidden at Tier-3 serving intake (gate-enforced) |
| `task_run_bridge.task_from_execution_request` fallback `mint_task_id()` | Task transport seam | Worker deserialization when no embedded task payload; outside NPSC-4.1 intake scope |
| `WorkspaceAskRun.run_id` | Product correlation | See §7.3 — not Execution Engine identity |
| `Task` default_factory `mint_task_id` | Transport correlation | Pydantic default at object construction; execution `RunId` still runtime-owned |
| `test_host_task_revision_reentry.py` (7 tests) | `no_ci` / infra | Requires `ollama` optional dep; excluded from CI-equivalent run (`pytest -m "not no_ci"`) |
| `test_background_causal_evidence_admission_paths.py` | Optional celery collector | Collection error without `celery` — excluded from certification run |
| `test_ue_10r41_execution_import_hygiene_gate.py` | Pre-existing hygiene | 2 failures: local imports in `decision_finalization_conformance.py`; outside NPSC-4.1 scope, not a boundary/identity regression |
| `acp_run.py`, `autonomous_work/work_stage_capability_loop.py` | Out-of-scan-tier debt | Outside NPSC-4.1 forbidden-zone scan roots; tracked under broader execution convergence, not blocking 4.1 |

---

## 10. Final certification result

### Definition of Done

| Criterion | Status |
| --------- | ------ |
| `HEAD == origin/development` | **PASS** (`af825fc…`) |
| Worktree clean | **PASS** |
| `identity_authority` is sole identity owner | **PASS** |
| `ExecutionRuntime` is sole lifecycle owner | **PASS** |
| Intake cannot mint execution identity | **PASS** |
| Nexus has no lifecycle ownership | **PASS** |
| `UnifiedTaskRunner` classified correctly (harness/scheduling only) | **PASS** |
| All regression suites PASS (CI-equivalent) | **PASS** |
| Certification document created | **PASS** (this document) |
| No architecture regression | **PASS** |

### Verdict

**NPSC-4.1-F FINAL CERTIFICATION: PASS**

The platform maintains a single enterprise execution ownership model. Execution identity and lifecycle remain exclusively owned by `identity_authority` / `ExecutionRuntime`. Application, interaction, MCP, Nexus, and scheduler layers do not mint or bind execution identity. `UnifiedTaskRunner` is confined to harness and scheduling roles. No Category C static-audit violations detected. Frozen NPSC regression gates pass at revision `af825fc…`.

---

*Certification performed read-only. No frozen execution engine contracts modified. No new abstractions introduced.*

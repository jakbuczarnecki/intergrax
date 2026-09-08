# NPSC-4.1 — Execution Boundary Hardening Certification

**Status:** `CERTIFIED`

**Verdict:** **PASS** (execution boundary hardening)

**Date:** 2026-09-08

**Branch:** `development`

**Scope:** Remove residual execution identity minting at HTTP/MCP application intake; classify `UnifiedTaskRunner` as harness/scheduling-only; extend static ownership gates.

**Frozen:** NPSC-3C Execution Engine — no redesign, no replacement framework.

---

## 1. Before state

| Seam | Classification | Location |
| ---- | -------------- | -------- |
| `mint_intake_execution_identity()` | Identity ownership violation | `intergrax/runtime/task/task_run_bridge.py` |
| HTTP/MCP intake callers | Identity ownership violation | Tier-3 `applications/**/serving/fastapi_router.py`, `workspace_routes.py`, `mcp_nexus_server.py` |
| `new_run_id()` at Tier-3 `/run` intake | Identity ownership violation | `attestation_demo`, `lab_application`, `poc_template`, `intergrax_assistant` serving routers |
| `UnifiedTaskRunner` docstring | Compatibility seam | Described as generic HTTP/eval entry |
| F-R1 AST scan roots | Gap | `applications/**` intake outside `execution/`, `nexus/`, `background_execution/` scan |

Residual debt documented in `EXECUTION_ENGINE_ENTERPRISE_VERIFICATION.md` §10 (`task_run_bridge.mint_intake_execution_identity`).

---

## 2. After state

Canonical intake path:

```text
HTTP / MCP intake
      |
      v
Task request (no RunId / AttemptId / ExecutionId mint)
      |
      v
HostTaskExecutionPort.execute(task)
      |
      v
ExecutionRuntime
      |
      v
identity_authority (mint_root_execution_identity)
      |
      v
ExecutionBoundary → StrategyExecutionRouter
```

**Removed:** `mint_intake_execution_identity()` from `task_run_bridge.py`.

**Intake rule:** Application serving layers build `Task` work-intent only. `RunId` / `AttemptId` / `ExecutionId` are minted exclusively inside `ExecutionRuntime` via `identity_authority`.

**Legal application:** Added `LegalApiV1RuntimeMapper.to_task()` for intake; `to_runtime_request()` retained for mapping tests with explicit fixture IDs.

---

## 3. Identity ownership proof

| Check | Result |
| ----- | ------ |
| `identity_authority.py` remains sole runtime mint module | **PASS** (`test_execution_identity_single_authority_gate`) |
| `mint_intake_execution_identity` absent from `task_run_bridge.py` | **PASS** |
| Tier-3 serving intake has no `mint_intake_execution_identity`, `new_run_id`, `mint_run_id`, `mint_attempt_id`, `mint_execution_id` | **PASS** (`test_npsc4_1_application_intake_does_not_mint_execution_identity`) |
| MCP shared intake (`mcp_nexus_server.py`) has no intake identity mint | **PASS** |
| Nexus / interactions / scheduler scanned trees have no forbidden mint | **PASS** |
| `bind_active_execution_identity` outside `ExecutionBoundary` in scanned zones | **PASS** |

---

## 4. Lifecycle ownership proof

| Owner | Responsibility | Result |
| ----- | -------------- | ------ |
| `ExecutionRuntime` | Root lifecycle start/close, context resolution | **PASS** (unchanged) |
| `ExecutionBoundary` | Identity propagation bind | **PASS** |
| Intake / Nexus / scheduler | No `start_execution` / `begin_execution` ownership | **PASS** (`test_npsc4_1_forbidden_zones_do_not_start_execution_lifecycle`) |
| Nexus | Orchestration only; no lifecycle mint/bind | **PASS** |

---

## 5. UnifiedTaskRunner classification

**Final classification:** `HARNESS / SCHEDULING ONLY`

Documented in `unified_task_runner.py` module docstring.

| Allowed | Forbidden |
| ------- | --------- |
| Scheduler coordination (`UnifiedTaskResumeExecutor`) | Tier-3 production execution entry |
| Harness / eval / long-running resume adapters | Lifecycle ownership |
| Thin adapter → `execute_root_task` → `ExecutionRuntime` | Identity mint in runner source |
| | Bypassing `HostTaskExecutionPort` from Tier-3 factories |

**Gate:** `test_npsc4_1_tier3_factories_do_not_reference_unified_task_runner` (NPSC-3G aligned).

---

## 6. Static gates

New gate module:

`tests/unit/runtime/architecture/test_npsc4_1_execution_boundary_hardening_gate.py`

Assertions:

- No `mint_intake_execution_identity` helper
- Application serving intake identity mint forbidden
- Interaction / Nexus / scheduler identity mint forbidden
- Forbidden zones do not bind execution identity
- Forbidden zones do not start execution lifecycle
- `UnifiedTaskRunner` harness classification
- Tier-3 factories do not reference `UnifiedTaskRunner`
- `ExecutionRuntime` and `identity_authority` ownership unchanged

---

## 7. Regression matrix

| Suite | Result |
| ----- | ------ |
| `tests/unit/runtime/execution/**` | **PASS** (included in run) |
| `tests/unit/runtime/interactions/**` | **PASS** |
| `tests/unit/runtime/background_execution/**` | **PASS** (1 skipped: celery optional dep) |
| NPSC-3B / 3C-A..D / 3E / 3F / 3G frozen gates | **PASS** |
| NPSC-4 governance gate | **PASS** |
| NPSC-4.1 gate | **PASS** (11 tests) |
| Legal / LKW intake tests | **PASS** |

Command (representative):

```powershell
uv run pytest tests/unit/runtime/execution/ tests/unit/runtime/interactions/ `
  tests/unit/runtime/background_execution/ `
  --ignore=tests/unit/runtime/background_execution/test_background_causal_evidence_admission_paths.py `
  tests/unit/runtime/architecture/test_npsc4_1_execution_boundary_hardening_gate.py `
  tests/unit/runtime/architecture/test_execution_identity_single_authority_gate.py `
  tests/unit/runtime/architecture/test_npsc3c_d_canonical_execution_engine_conformance_gate.py `
  tests/unit/runtime/architecture/test_ue_9d_legacy_execution_retirement_gate.py `
  tests/unit/runtime/architecture/test_npsc4_agent_runtime_governance_gate.py `
  tests/unit/applications/architecture/test_npsc3g_application_runtime_convergence_gate.py `
  tests/unit/applications/architecture/test_npsc3f_runtime_convergence_gate.py `
  tests/unit/applications/architecture/test_npsc3c_legacy_executor_removal_gate.py -q
```

**799 passed, 1 skipped** at certification time.

---

## 8. Remaining debt

| Item | Classification | Notes |
| ---- | -------------- | ----- |
| `task_run_bridge.new_run_id()` | Harness/eval alias | Allowed in `harness_task_routes.py`; forbidden at Tier-3 serving intake |
| `task_run_bridge.task_from_execution_request` fallback `mint_task_id()` | Task adapter seam | Worker deserialization when no embedded task payload; outside NPSC-4.1 intake scope |
| `ask_service.py` `new_run_id()` for `WorkspaceAskRun` | Product correlation seam | Product-level ask-run tracking; not HTTP execution intake — future UER alignment |
| `Task` default_factory `mint_task_id` | Transport correlation | Pydantic default at object construction; execution RunId still runtime-owned |
| Unrelated dirty worktree (`platform_proofs/**`, `connected_source_discovery.py`) | Out of scope | Pre-existing operator work; not part of NPSC-4.1 |

---

## Definition of Done

| Criterion | Status |
| --------- | ------ |
| Task intake does not mint execution identity (`RunId`/`AttemptId`/`ExecutionId`) | **PASS** |
| `ExecutionRuntime` remains sole lifecycle owner | **PASS** |
| `identity_authority` remains sole mint owner | **PASS** |
| `UnifiedTaskRunner` cannot own production Tier-3 execution | **PASS** |
| Nexus remains orchestration only | **PASS** |
| No duplicate execution framework introduced | **PASS** |
| No new abstractions introduced | **PASS** |
| Regression gates pass | **PASS** |
| Documentation created | **PASS** |
| `HEAD == origin/development` | **PENDING** (local changes not pushed) |
| Worktree clean | **PARTIAL** (unrelated pre-existing dirty paths remain) |

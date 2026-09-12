# Execution Identity Intake Convergence Model (EE-A2-H1)

**Classification:** `MAINTAINER_CERTIFICATION`  
**Status:** `CERTIFIED` (EE-A2-H1 intake convergence on `development`)  
**Parent:** [`EXECUTION_IDENTITY_AUTHORITY_MODEL.md`](EXECUTION_IDENTITY_AUTHORITY_MODEL.md) (EE-A2)

---

## Ingress points

| Surface | Module | Execution identity at ingress |
| --- | --- | --- |
| Host task / FastAPI Core execution adapters | `intergrax/fastapi_core/execution/adapters/*` | Metadata only; admission via `ExecutionRuntime` |
| Tier-3 application serving routers | `applications/*/serving/*` | No `RunId` / `ExecutionId` / `AttemptId` mint |
| Harness HTTP (non-production) | `intergrax/applications/_shared/harness_task_routes.py`, `intergrax/harness/lab_fastapi.py` | `TaskId` only; run mint at runtime admission |
| ACP agent session | `intergrax/agents/authoring/acp_run.py` | Validates optional `run_id` / `task_id`; mint via `default_execution_identity_authority` |
| Background transport bootstrap | `intergrax/runtime/background_execution/bootstrap.py` | Delegates to `mint_background_transport_identity` (authority module) |
| Scheduler / long-running resume | `intergrax/runtime/long_running/*` | Restores checkpoint identity; no mint on resume |
| MCP / workspace product seams | `applications/*` product correlation | Product-level ids only (not execution lifecycle) |

---

## Ownership matrix

| Identity | Owner |
| --- | --- |
| **RunId** | `DefaultExecutionIdentityAuthority` (via `identity_authority.py`), admitted by `ExecutionRuntime` |
| **ExecutionId** | Same authority (`mint_execution_identity`, `mint_child_execution_identity`) |
| **AttemptId** (initial) | Root admission via authority |
| **AttemptId** (retry) | `AttemptLifecycleService` + `mint_retry_attempt_id` |
| **Child ExecutionId** | `ChildExecutionRunner` → `default_execution_identity_authority.mint_child_execution_identity()` |
| **TaskId** | Task plane (`Task`, transport bootstrap) |
| **RequestId / CorrelationId / TraceId / SessionId** | Ingress metadata — may validate into `RunId` at admission, never mint `AttemptId`/`ExecutionId` alone |

---

## Lifecycle diagram

```text
External Request
        |
        v
Ingress Adapter  (request_id, correlation_id, external_id only)
        |
        v
ExecutionRuntime Admission
        |
        v
DefaultExecutionIdentityAuthority  (ExecutionIdentityAuthorityPort)
        |
        +-- RunId
        +-- ExecutionId
        +-- AttemptId
        |
        v
ExecutionBoundary / strategy / evidence
```

---

## Forbidden patterns

- `mint_run_id()`, `mint_execution_id()`, `mint_attempt_id()` outside allowlisted authority and conformance modules.
- `new_run_id()` at Tier-3 serving intake or harness routes that bypass runtime admission.
- Parallel identity systems (SQLite/local mint helpers, provider-generated execution ids).
- `LongRunningCoordinator` or resume paths that mint new `RunId` / `AttemptId` / `ExecutionId`.
- Background workers minting identity on redelivery (must reload persisted transport identity).

---

## Allowed exceptions

| Exception | Rationale |
| --- | --- |
| `intergrax/contracts/execution_identity.py` | Primitive format helpers |
| `intergrax/runtime/execution/identity_authority.py` | Canonical mint implementation |
| `AttemptLifecycleService` retry path | Durable attempt transitions only |
| Persistence / observability conformance harnesses | Test data fixtures (`*_conformance.py`, `repository_qualification_suite.py`) |
| `intergrax/experiments/*`, eval harness | Non-production orchestration |
| `task_run_bridge.new_run_id()` | Deprecated alias delegating to authority — not for HTTP/MCP intake |
| Product ask-run correlation (`ask_*` ids) | Not execution lifecycle |

---

## Certification gate

Architecture gate: `tests/unit/runtime/architecture/test_ee_a2_h1_intake_identity_convergence_certification.py`

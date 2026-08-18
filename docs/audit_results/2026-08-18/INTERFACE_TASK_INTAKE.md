# INTERFACE_TASK_INTAKE — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** INTERFACE_TASK_INTAKE
- **Tier(s):** cross-domain Tier-0 contracts · Tier-1 runtime intake/execution convergence · Tier-3 application host intake surfaces
- **audited_sha:** `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Status:** COMPLETE
- **Auditor:** OpenAI ChatGPT / GPT-5.6 Sol — independent auditor
- **Verdict:** FAIL
- **Architecture doc(s):**
  - `docs/project/architecture/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/architecture/SYSTEM_INVARIANTS.md` (cross-layer identity authority, cited where relevant)
- **Plan doc(s):**
  - `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
- **Scope in:**
  - canonical normalized intake contract adoption (`TaskEnvelope`)
  - TaskId / RunId identity on supported public intake paths
  - execution convergence through `UnifiedTaskRunner` before `NexusLoop`
  - typed intake semantics preservation across intake/runtime boundary
  - typed executor interfaces for critical intake execution capabilities
  - product streaming/async intake parity proof vs configuration claims
  - provider/backend abstraction posture on intake-adjacent ports (no new vendor leakage findings)
- **Scope out:**
  - remediation implementation
  - full security audit
  - re-audit of prior campaign layers
  - duplicate ownership of PBA checkpoint SQLite leak (PBA-01) or TL-FIX-D private `_execution_adapter`
- **Prior audit reference(s):** [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md) (universal execution boundary history); [`TIER_LAYER_BOUNDARIES`](TIER_LAYER_BOUNDARIES.md) (Tier-3 composition); [`PROVIDER_BACKEND_ABSTRACTION`](PROVIDER_BACKEND_ABSTRACTION.md) (checkpoint persistence port consumption)
- **Exact audit-start time:** not captured; preserve date-level precision rather than fabricate one
- **post_sync_sha:** `—` (trace commit follows audit sync commit)

## Executive summary

**Verdict: FAIL.** Six accepted findings (3 HIGH, 3 MEDIUM) show that supported intake surfaces do not structurally converge on one canonical normalized contract before `Task` materialization, several public paths mint `run_...` identifiers as `Task.task_id`, production interaction wiring can execute through direct `NexusLoop.handle_task()` without `UnifiedTaskRunner` guarantees, typed SLA/risk intake semantics flatten into legacy metadata at the intake boundary, critical prepared-task execution relies on `hasattr()` discovery, and the product intake parity gate validates feature flags only—not end-to-end streaming intake through canonical normalization and the public runner. Provider/backend abstraction on intake-adjacent ports is largely preserved; no new independent vendor/backend leakage finding was discovered in this layer.

## Verdict

**FAIL**

## Findings

### AUDIT-20260818-INTERFACE_TASK_INTAKE-01

**Parallel intake contracts — TaskEnvelope not mandatory normalization boundary**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** ITI-FIX-A
- **Claim falsified:** `TaskEnvelope` is the single normalized intake contract through which supported HTTP, CLI/worker, and interaction surfaces converge before `Task` materialization.
- **Observation:** The repository declares `TaskEnvelope` as the single normalized intake contract, but supported surfaces use multiple parallel intake contracts and normalization routes, including `CreateRunRequest`/`ExecutionRequest`, `RuntimeRequest`, `InboundInteraction`, and direct `Task` construction. The `intake_payload_to_envelope` path itself materializes `Task` via `intake_payload_to_task` before returning `TaskEnvelope`. The defect is absence of one mandatory canonical normalization contract—not defective raw JSON at an external edge.
- **Location:**
  - `intergrax/contracts/task_envelope.py:L5-L6,L30` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/fastapi_core/runs/models.py:L22` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/fastapi_core/runs/default_service.py:L13,L45,L52-L57` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/task/task_run_bridge.py:L34-L48` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/interactions/envelope_intake.py:L17-L25` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/interactions/adapter_contract.py:L64-L91` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `applications/legal_application/serving/runtime_bridge.py:L4,L22,L47,L55` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Reproduction:**
  1. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/contracts/task_envelope.py` — single-contract declaration (L5-L6); `TaskEnvelope` model (L30).
  2. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/fastapi_core/runs/default_service.py` — `CreateRunRequest` → `ExecutionRequest` path bypasses `TaskEnvelope` (L13, L45, L52-L57).
  3. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/task/task_run_bridge.py` — `task_from_runtime_request` builds `Task` from `RuntimeRequest` (L34-L48).
  4. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/interactions/envelope_intake.py` — `intake_payload_to_envelope` calls `intake_payload_to_task` then `to_envelope()` (L17-L25).
  5. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/interactions/adapter_contract.py` — `inbound_to_task` direct `Task` construction (L64-L91).
  6. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:applications/legal_application/serving/runtime_bridge.py` — Legal API maps to `RuntimeRequest` (L4, L22, L47, L55).
- **Impact:** Surface-specific edge schemas compete as semantic contracts; intake normalization is not mechanically convergent, weakening cross-surface parity and typed intake guarantees.
- **Confidence:** CONFIRMED

### AUDIT-20260818-INTERFACE_TASK_INTAKE-02

**RunId passed as Task.task_id on supported public intake paths**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT
- **Related classification:** TEST GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** ITI-FIX-B
- **Claim falsified:** `TaskId` and `RunId` are distinct canonical execution identities on all supported public intake paths.
- **Observation:** Several supported entry points call `new_run_id()` and pass the resulting `run_...` identifier as `Task.task_id` even though `Task` validates `task_id` with `validate_task_id()`, which requires `task_...`. Contract tests prove `TaskId != RunId` and the canonical helper exists, but consumer adoption is not mechanically enforced across public surfaces. This is statically **CONFIRMED** as a reachable construction defect; no failing endpoint/integration test was executed by the auditor.
- **Location:**
  - `applications/research_application/serving/fastapi_router.py:L12,L26-L28` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `applications/intergrax_assistant_application/serving/fastapi_router.py:L13,L44-L46` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/applications/_shared/mcp_nexus_server.py:L15,L63-L66` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/applications/_shared/harness_task_routes.py:L22,L70-L72` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `applications/local_workspace_application/serving/fastapi_router.py:L11,L26-L31` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/contracts/execution_identity.py:L41-L42` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/task/task.py:L59-L60` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `tests/unit/contracts/test_execution_identity.py:L62` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Reproduction:**
  1. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:applications/research_application/serving/fastapi_router.py` — `new_run_id()` assigned to `task_id` (L12, L26-L28).
  2. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/applications/_shared/mcp_nexus_server.py` — same pattern (L15, L63-L66).
  3. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/contracts/execution_identity.py` — `validate_task_id` requires `task_` prefix (L41-L42).
  4. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:tests/unit/contracts/test_execution_identity.py` — `run_` rejected as `TaskId` (L62).
  5. `git grep -n "task_id=run_id" 2640d826da6f1a781e798326ff1b21b3a9f7c4cc -- applications/ intergrax/applications/_shared/` — enumerate affected surfaces.
- **Impact:** Identity contract violation on public surfaces; trace/correlation and resume semantics can diverge between task and run lifecycles when `run_...` is used as task identity.
- **Confidence:** CONFIRMED

### AUDIT-20260818-INTERFACE_TASK_INTAKE-03

**Interaction intake can bypass UnifiedTaskRunner to NexusLoop directly**

- **Severity:** HIGH
- **Category:** BOUNDARY VIOLATION
- **Status at publication:** ACCEPTED
- **Remediation block:** ITI-FIX-C
- **Claim falsified:** All supported application execution surfaces converge through `UnifiedTaskRunner` before `NexusLoop`.
- **Observation:** `InteractionIntakeService` can resolve `NexusLoopTaskExecutor` when supplied `nexus_loop`. `NexusLoopTaskExecutor` calls `NexusLoop.handle_task()` directly. Shared production Tier-3 interaction wiring and Legal host make this path reachable even though the host already has a `UnifiedTaskRunner`. Unit coverage explicitly protects this path as backward compatibility. NexusLoop still executes; the direct path does not structurally pass through `UnifiedTaskRunner`-owned guarantees such as `ActiveTaskRegistry`, `llm_tenant_scope`, and canonical runner-level identity/resume handling.
- **Location:**
  - `intergrax/runtime/task/unified_task_runner.py:L15-L16,L20,L50,L71,L78` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/interactions/task_executor.py:L23-L30` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/interactions/intake_service.py:L42,L48,L55-L56` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/applications/_shared/interaction_wiring.py:L17,L21,L37` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `applications/legal_application/host/factory.py:L75,L79,L89,L162,L187` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `tests/unit/runtime/interactions/test_interaction_intake_service.py:L71,L85,L91` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `docs/project/architecture/NEXUS_EXECUTION_FLOW.md:L13,L25` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Reproduction:**
  1. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/interactions/intake_service.py` — `_resolve_executor` returns `NexusLoopTaskExecutor` when `nexus_loop` set (L55-L56).
  2. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/interactions/task_executor.py` — direct `handle_task` (L30).
  3. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/applications/_shared/interaction_wiring.py` — passes `nexus_loop` to service (L37).
  4. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:applications/legal_application/host/factory.py` — Legal wires interaction intake with `nexus_loop` while also building `UnifiedTaskRunner` (L89, L162).
  5. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/task/unified_task_runner.py` — runner guarantees absent on direct path (L50, L71, L78).
  6. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:tests/unit/runtime/interactions/test_interaction_intake_service.py` — backward-compat test (L71, L85, L91).
- **Impact:** Supported production interaction execution can miss canonical runner governance (active-task registry, tenant-scoped LLM metering, runner-level resume/identity handling) while architecture claims universal convergence.
- **Confidence:** CONFIRMED

### AUDIT-20260818-INTERFACE_TASK_INTAKE-04

**Typed intake semantics degrade into flat metadata at intake boundary**

- **Severity:** MEDIUM
- **Category:** IMPLEMENTATION/ARCHITECTURE DRIFT
- **Status at publication:** ACCEPTED
- **Remediation block:** ITI-FIX-A
- **Claim falsified:** Typed intake semantics remain typed canonical execution semantics after `TaskEnvelope` normalization.
- **Observation:** `TaskEnvelope` exposes typed SLA/risk fields, but `Task.from_envelope` serializes them into flat metadata strings and `Task.to_envelope` later reconstructs enums from metadata. `task_metadata_bridge` still carries numerous execution/governance semantics through legacy flat metadata. This layer establishes that typed intake semantics degrade into metadata rather than remaining canonical typed state—not that risk policy is globally bypassed or that this is a security defect.
- **Location:**
  - `intergrax/contracts/task_envelope.py:L41-L42` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/task/task.py:L96-L124` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/task/task_metadata_bridge.py:L4,L99-L100,L252-L274` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/runtime/policy/compliance_profiles.py:L63-L64` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Reproduction:**
  1. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/contracts/task_envelope.py` — typed `sla_class` / `risk_tier` on envelope (L41-L42).
  2. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/task/task.py` — `to_envelope` reads enums from metadata strings; `from_envelope` writes `.value` into metadata (L96-L124).
  3. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/task/task_metadata_bridge.py` — legacy flat metadata bridge for execution/governance options (L4, L99-L100, L252-L274).
  4. `git grep -n "sla_class\|risk_tier\|compliance_template_for_risk_tier" 2640d826da6f1a781e798326ff1b21b3a9f7c4cc -- intergrax/contracts/task_envelope.py intergrax/runtime/task/task.py intergrax/runtime/policy/compliance_profiles.py`
- **Impact:** Canonical typed intake state is not preserved through the intake/runtime boundary; governance and compliance semantics depend on metadata round-trip rather than stable typed fields.
- **Confidence:** CONFIRMED

### AUDIT-20260818-INTERFACE_TASK_INTAKE-05

**Critical execute_prepared capability discovered via hasattr reflection**

- **Severity:** MEDIUM
- **Category:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** ITI-FIX-C
- **Claim falsified:** Critical intake execution capabilities are expressed through explicit typed interfaces.
- **Observation:** `InteractionIntakeService` detects `execute_prepared` dynamically using `hasattr()` even though `TaskPreparationExecutor` Protocol does not declare `execute_prepared`; invocation requires `type: ignore[attr-defined]`.
- **Location:**
  - `intergrax/runtime/interactions/intake_service.py:L97-L99` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Reproduction:**
  1. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/runtime/interactions/intake_service.py` — `hasattr(executor, "execute_prepared")` and `# type: ignore[attr-defined]` (L97-L99).
  2. Compare `TaskPreparationExecutor` Protocol definition in same file (L22-L25) — no `execute_prepared` member.
- **Impact:** Critical prepared-task execution path lacks mechanical type safety; executor capabilities are not contractually enforced at the intake boundary.
- **Confidence:** CONFIRMED

### AUDIT-20260818-INTERFACE_TASK_INTAKE-06

**Product intake parity gate validates flags only—not E2E streaming intake**

- **Severity:** MEDIUM
- **Category:** TEST GAP
- **Related classification:** PROCESS / CLAIM
- **Status at publication:** ACCEPTED
- **Remediation block:** ITI-FIX-D
- **Dependencies:** ITI-FIX-A, ITI-FIX-B, ITI-FIX-C
- **Claim falsified:** The existing product-intake parity gate proves delivered streaming + durable async intake parity.
- **Observation:** Historical registers mark AUDIT-IDEAL-3.2 as Done and product defaults enable `streaming_intake_enabled`, but `check_product_intake_parity.py` validates only configuration flags. The streaming `TaskEnvelope` contract/assembler exists, but the gate does not prove an end-to-end production path from streamed input through canonical normalization and `UnifiedTaskRunner` to `TaskResult`. `assemble_envelope_from_chunks` consumers at audited SHA are unit/gate tests only—not production host wiring.
- **Location:**
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md:L109` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/applications/contracts/application_host.py:L48,L96` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/applications/_shared/intake_wiring.py:L19,L26,L31` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `scripts/maintenance/check_product_intake_parity.py:L17-L24` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `intergrax/contracts/task_envelope_stream.py:L25,L35` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
  - `tests/unit/runtime/architecture/test_ideal_harness_l3_w2_depth_gate.py:L14,L37` @ `2640d826da6f1a781e798326ff1b21b3a9f7c4cc`
- **Reproduction:**
  1. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:scripts/maintenance/check_product_intake_parity.py` — checks `durable_async_index` and `streaming_intake_enabled` flags only (L17-L24).
  2. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:intergrax/applications/contracts/application_host.py` — product defaults set `streaming_intake_enabled=True` (L96).
  3. `git grep -n "StreamingTaskIntake\|assemble_envelope_from_chunks" 2640d826da6f1a781e798326ff1b21b3a9f7c4cc` — contract + unit test only; no production host consumer.
  4. `git show 2640d826da6f1a781e798326ff1b21b3a9f7c4cc:docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md` — AUDIT-IDEAL-3.2 marked Done (L109).
- **Impact:** Maturity and parity claims for streaming product intake lack E2E evidence; flag-only gates can certify configuration without proving canonical intake → runner → result path.
- **Confidence:** CONFIRMED

## Provider/backend abstraction classification matrix

| Concern | Canonical abstraction | Observed pattern | Classification | Notes |
|---------|----------------------|------------------|----------------|-------|
| Run lifecycle store | `RunStore` | port consumed by `DefaultRunService` | ABSTRACTION_PRESERVED | FastAPI run orchestration delegates persistence to store port |
| Run execution trigger | `ExecutionAdapter` | port consumed by `DefaultRunService` | ABSTRACTION_PRESERVED | background execution via adapter contract |
| Interaction intake | `InteractionAdapter` / `InteractionPayloadParser` | adapter composition at wiring | COMPOSITION_ONLY / abstraction preserved | vendor payloads normalized before runtime; no new vendor leak |
| Async task index | `AsyncTaskIndexProtocol` | composition in harness/async wiring | COMPOSITION_ONLY / abstraction preserved | not promoted to independent finding |
| Task checkpoint persistence | `TaskCheckpointPersistence` | SQLite concrete in Nexus paths | EXISTING PBA finding | reference [`AUDIT-20260818-PROVIDER_BACKEND_ABSTRACTION-01`](PROVIDER_BACKEND_ABSTRACTION.md) — **not duplicated** |

**AUDIT-3 discovered no new independent vendor/backend leakage finding** in this layer.

## Falsification log

Targets examined but **not** promoted to findings:

1. **Surface-specific edge JSON schemas** — allowed at external edges when they converge through a canonical normalization boundary; raw JSON at the edge is not itself defective.
2. **NexusLoop execution capability** — direct `handle_task` still executes tasks; finding 03 claims missing runner guarantees, not absent execution.
3. **Risk policy globally bypassed** — finding 04 is typed-semantics degradation only; not classified as security.
4. **No streaming functionality of any kind** — streaming contract/assembler exists; finding 06 is parity-gate evidence gap only.
5. **PBA checkpoint SQLite leak** — owned by PBA-01; not duplicated.
6. **TL-FIX-D private `_execution_adapter`** — owned by Tier-layer audit; not duplicated.

## Prior-audit comparison

Prior campaign layers [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md), [`TIER_LAYER_BOUNDARIES`](TIER_LAYER_BOUNDARIES.md), and [`PROVIDER_BACKEND_ABSTRACTION`](PROVIDER_BACKEND_ABSTRACTION.md) established related universal-execution, Tier-3 composition, and checkpoint-port themes. This layer owns **intake-specific** convergence claims: normalized contract adoption, identity minting on public surfaces, runner-boundary reachability, typed semantics preservation, executor interface discipline, and intake parity proof. No prior canonical Protocol v2.2 `INTERFACE_TASK_INTAKE` immutable snapshot existed before this layer.

## Open questions / blocked items

- Whether `CreateRunRequest`/`ExecutionRequest` should fold into `TaskEnvelope` normalization or remain a parallel FastAPI run lifecycle with explicit non-Nexus semantics — planning only (**ITI-FIX-A**).
- Whether interaction intake should always receive a `UnifiedTaskRunner`-backed executor at composition — deferred to **ITI-FIX-C**.
- E2E streaming proof host selection (which product reference host exercises stream → envelope → runner) — deferred to **ITI-FIX-D**.
- No operator-disputed findings; no blocked evidence collection.

## Operator acceptance

- **Date:** 2026-08-18
- **Accepted findings:** all 6 (`AUDIT-20260818-INTERFACE_TASK_INTAKE-01` … `AUDIT-20260818-INTERFACE_TASK_INTAKE-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

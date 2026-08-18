# IDENTITY_TRUST — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** IDENTITY_TRUST
- **Tier(s):** cross-domain Tier-0 contracts · Tier-1 runtime identity/HITL/delegation · Tier-3 application host authentication surfaces
- **audited_sha:** `6fbc5e4928963ecd386456158b0753662fed209b`
- **Status:** COMPLETE
- **Auditor:** independent ChatGPT platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-18
- **Architecture doc(s):**
  - `docs/project/architecture/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/architecture/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/architecture/RELIABILITY_FAILURE_AND_HITL.md`
  - `docs/project/architecture/UNIFIED_EXECUTION_RUNTIME.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/TIER3_APPLICATION_ENVIRONMENT.md`
  - `docs/project/maintainers/plans/NEXUS_EXECUTION_FLOW.md`
  - `docs/project/maintainers/plans/RELIABILITY_FAILURE_AND_HITL.md`
  - `docs/project/maintainers/plans/UNIFIED_EXECUTION_RUNTIME.md`
- **Scope in:**
  - authenticated principal → execution identity binding
  - delegated authority narrowing (`permission_scopes`)
  - human decision provenance (approver identity)
  - HITL resume surface alignment with exact pause/request correlation
  - execution identity closure (`RunId`/`AttemptId` vs `TaskId`) on residual HITL/lifecycle paths
  - `ActorIdentity` / `RequestIdentity` wire semantics coherence
  - provider/backend abstraction posture on identity-adjacent ports (no new vendor leakage findings)
- **Scope out:**
  - remediation implementation
  - duplicate INTERFACE_TASK_INTAKE public TaskId/RunId minting findings
  - JwtAuthProvider / NoOpJwtVerifier without proven production reachability
  - full security audit of all authentication mechanisms
- **Prior audit reference(s):** [`INTERFACE_TASK_INTAKE`](INTERFACE_TASK_INTAKE.md) (public intake identity); [`PROVIDER_BACKEND_ABSTRACTION`](PROVIDER_BACKEND_ABSTRACTION.md) (IdentityProviderBackend abstraction); [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md) (execution identity continuity)
- **architecture_sync:** COMPLETE after commit A
- **plan_sync:** COMPLETE after commit A
- **post_sync_sha:** — *(pending until commit A exists)*

## Executive summary

**Verdict: FAIL.** Six accepted findings (4 HIGH, 2 MEDIUM) show that verified credentials are not canonically bound to `RequestIdentity` on all Tier-3 intake paths, delegation `permission_scopes` are recorded but not enforced as effective child authority, HITL resolution binds the exact decision but not the authenticated approver, the shared HTTP resume path omits exact pause/request identifiers, residual HITL/lifecycle provenance still substitutes `TaskId` for `RunId`, and `ActorIdentity` remains a paper model with inconsistent wire semantics. Positive controls preserved: FastAPI Core `AuthProvider` → `AuthContext`, G5C exact task/pause/human_request correlation, Legal body-vs-context identity checks, and `IdentityProviderBackend` / `IntegrationProfile` abstraction. No new independent vendor/backend leakage finding in this layer.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 2 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-IDENTITY_TRUST-01

**Authenticated credential is not canonically bound to RequestIdentity**

- **Severity:** HIGH
- **Category:** SECURITY
- **Related classification:** TEST GAP
- **Status at publication:** ACCEPTED
- **Remediation block:** IDT-FIX-A
- **Claim falsified:** An authenticated Tier-3 request produces execution identity exclusively from the verified credential / authenticated principal, and untrusted request fields or metadata cannot redefine tenant/user/principal identity.
- **Observation:** `RequestIdentity` is documented/typed as the authenticated principal. `runtime_request_to_agent_run` derives `tenant_id`/`user_id`/`principal_type`/`auth_subject` from `RuntimeRequest` fields and metadata; metadata `user_id` can override typed `request.user_id`. Harness identity-provider verification currently reduces verified identity to a boolean authentication result and discards the verified `IdentityUser` for downstream execution identity. Shared/product surfaces can construct `Task` `tenant_id`/`user_id` from request body independently of the verified credential. Research is a concrete bounded example; not every product host has this defect.
- **Location:**
  - `intergrax/contracts/agent_run.py` — `RequestIdentity` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/agents/runtime_request_bridge.py` — `runtime_request_to_agent_run` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/applications/_shared/harness_auth.py` — `is_harness_identity_token_valid` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `applications/research_application/serving/fastapi_router.py` — `ResearchRunService.run_pipeline` @ `6fbc5e4928963ecd386456158b0753662fed209b`
- **Reproduction:**
  1. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/contracts/agent_run.py` — `RequestIdentity` authenticated-principal docstring and fields.
  2. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/agents/runtime_request_bridge.py` — metadata `user_id` precedence over `request.user_id`.
  3. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/applications/_shared/harness_auth.py` — `verify_token` result reduced to boolean; verified `IdentityUser` not propagated.
  4. `git show 6fbc5e4928963ecd386456158b0753662fed209b:applications/research_application/serving/fastapi_router.py` — `tenant_id`/`user_id` from request body without credential binding.
- **Impact:** Credential authentication and execution principal are separate sources of truth. This can undermine tenant/user attribution and any downstream memory/policy/security mechanism that trusts `RequestIdentity`.
- **Confidence:** CONFIRMED

### AUDIT-20260818-IDENTITY_TRUST-02

**Delegation permission_scopes are recorded but do not constrain child execution**

- **Severity:** HIGH
- **Category:** SECURITY
- **Status at publication:** ACCEPTED
- **Remediation block:** IDT-FIX-B
- **Claim falsified:** A delegated child execution is restricted to effective authority no broader than the parent and the delegation's declared permission scopes.
- **Observation:** `DelegationSpec` and `SubtaskContract` carry `permission_scopes`. `narrow_delegation_scopes` exists. `GraphExecutor` validates depth / LLM budget / tool budget, but does not apply `permission_scopes` as effective child authority. Child request composition propagates delegation namespace and parent identifiers, not a narrowed effective authority envelope. `DELEGATION_GRANTED` observability records `permission_scopes` although runtime does not prove those scopes are enforced. `CollaborativeWorkAuthorityResolver` is a positive existing pattern for authoritative fail-closed narrowing; do not silently merge the two models in this persistence task.
- **Location:**
  - `intergrax/contracts/delegation.py` — `DelegationSpec.permission_scopes` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/contracts/subtask_contract.py` — `SubtaskContract.permission_scopes` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/interactions/actor_resolution.py` — `narrow_delegation_scopes` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/nexus/execution/graph_executor.py` — `GraphExecutor`, `_emit_delegation_granted`, `max_delegation_depth` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/collaborative_work/authority.py` — `CollaborativeWorkAuthorityResolver` @ `6fbc5e4928963ecd386456158b0753662fed209b`
- **Reproduction:**
  1. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/contracts/delegation.py` — `permission_scopes` on `DelegationSpec`.
  2. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/interactions/actor_resolution.py` — `narrow_delegation_scopes` helper (no production enforcement path established).
  3. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/nexus/execution/graph_executor.py` — depth validation; `DELEGATION_GRANTED` emits `permission_scopes` without downstream enforcement proof.
  4. `git grep -n "permission_scopes" 6fbc5e4928963ecd386456158b0753662fed209b -- intergrax/runtime/nexus/execution/graph_executor.py` — observability-only emission.
- **Impact:** Audit evidence can describe a narrower delegation than the execution path actually enforces.
- **Confidence:** CONFIRMED

### AUDIT-20260818-IDENTITY_TRUST-03

**HITL resolution is bound to the exact decision but not to the authenticated approver**

- **Severity:** HIGH
- **Category:** SECURITY
- **Status at publication:** ACCEPTED
- **Remediation block:** IDT-FIX-C
- **Claim falsified:** A human approval record proves both WHAT exact pause/request was decided and WHO the authenticated/authorized approver was.
- **Observation:** Current G5C HITL correctly binds `task_id` + `pause_id` + `human_request_id` and rejects stale/forged mismatches; this is a positive control and must be explicitly preserved. `HumanApprovalResolution` / persisted `HumanDecisionRecord` do not carry a canonical verified approver principal with auth provenance. Persisted `HumanDecisionRecord.user_id` is populated from `task.user_id`, which identifies the task user rather than necessarily the human who approved. `HUMAN_APPROVAL_RECEIVED` evidence does not establish authenticated approver provenance.
- **Location:**
  - `intergrax/runtime/human/pause.py` — `HumanPauseCoordinator.resolve_human_response` exact-correlation guards @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/task/task_contract.py` — `HumanApprovalResolution` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/human/models.py` — `HumanDecisionRecord`, `build_human_decision_record` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/nexus/orchestration/human_response.py` — `persist_human_decision` (`user_id=task.user_id`) @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/nexus/orchestration/intake_runner.py` — `HUMAN_APPROVAL_RECEIVED` emission @ `6fbc5e4928963ecd386456158b0753662fed209b`
- **Reproduction:**
  1. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/human/pause.py` — pause/request mismatch rejection (positive control).
  2. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/task/task_contract.py` — `HumanApprovalResolution` fields (no approver principal).
  3. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/nexus/orchestration/human_response.py` — `user_id=task.user_id` in persisted record.
  4. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/nexus/orchestration/intake_runner.py` — approval event payload lacks approver provenance.
- **Impact:** The platform can prove which decision was approved but cannot equivalently prove which authenticated human made the decision — material for high-risk side effects, compliance and forensic audit.
- **Confidence:** CONFIRMED

### AUDIT-20260818-IDENTITY_TRUST-04

**Shared HTTP resume path is not aligned with exact HITL pause/request identity**

- **Severity:** MEDIUM
- **Category:** IMPLEMENTATION DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** IDT-FIX-C
- **Claim falsified:** All supported resume surfaces materialize the exact `pause_id` and `human_request_id` required by canonical HITL resolution.
- **Observation:** `NexusIntakeRunner` expects `TaskHumanInput.pause_id` + `human_request_id`. Shared `POST /v1/tasks/{task_id}/resume` accepts resume token and `operator_input`. `resume_task_with_token` copies verdict and `response_text` but does not materialize `pause_id`/`human_request_id` from checkpoint `pause_record`. `DebugHitlResumeService` already demonstrates the correct pattern by extracting those identifiers from the checkpoint.
- **Location:**
  - `intergrax/runtime/nexus/orchestration/intake_runner.py` — `NexusIntakeRunner.run` pause/request correlation @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/applications/_shared/harness_task_routes.py` — shared resume route @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/applications/_shared/task_control.py` — `resume_task_with_token` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/debug/hitl_service.py` — `DebugHitlResumeService.resume_with_human_response` @ `6fbc5e4928963ecd386456158b0753662fed209b`
- **Reproduction:**
  1. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/nexus/orchestration/intake_runner.py` — requires `pause_id` and `human_request_id` on resume.
  2. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/applications/_shared/task_control.py` — sets verdict/response only; no pause/request materialization.
  3. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/debug/hitl_service.py` — extracts `pause_id`/`human_request_id` from `pause_record` (correct pattern).
  4. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/applications/_shared/harness_task_routes.py` — routes to `resume_task_with_token`.
- **Impact:** A shared resume surface can fail or diverge from the new exact-correlation HITL contract even though the canonical G5C core is fail-closed.
- **Confidence:** CONFIRMED

### AUDIT-20260818-IDENTITY_TRUST-05

**HITL and lifecycle provenance still use TaskId as RunId on residual paths**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT
- **Related classification:** OPERABILITY
- **Status at publication:** ACCEPTED
- **Remediation block:** IDT-FIX-D
- **Claim falsified:** Once Nexus binds `ActiveExecutionIdentity`, all runtime/HITL/hook provenance uses the canonical `RunId` and `AttemptId` rather than reconstructing run identity from `TaskId`.
- **Observation:** `NexusLoop` correctly binds `ActiveExecutionIdentity(run_id, attempt_id)`. APPROVE HITL path already uses the bound `run_id` + `attempt_id`. `human_approval_hook_context` and `nexus_lifecycle_hook_context` still construct `run_id` from `task.task_id`. REJECT / ESCALATE paths call `runtime_event_from_task_state` using `task.task_id` as `run_id`. `persist_human_decision` records `run_id=task.task_id`. Canonical `runtime_event_from_task_state` requires validated `RunId` and `AttemptId`. Therefore this is a partial migration, not a claim that all HITL is broken.
- **Location:**
  - `intergrax/contracts/execution_identity.py` — `ActiveExecutionIdentity` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/nexus/nexus_loop.py` — `ActiveExecutionIdentity` binding @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/nexus/orchestration/intake_runner.py` — APPROVE path uses bound identity @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/human/hitl_hooks.py` — `human_approval_hook_context` (`run_id=task.task_id`) @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/hooks/nexus_lifecycle_hooks.py` — `nexus_lifecycle_hook_context` (`run_id=task.task_id`) @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/nexus/orchestration/hitl_runner.py` — REJECT/ESCALATE `runtime_event_from_task_state(..., run_id=task.task_id)` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/nexus/orchestration/human_response.py` — `persist_human_decision` `run_id=task.task_id` @ `6fbc5e4928963ecd386456158b0753662fed209b`
- **Reproduction:**
  1. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/nexus/orchestration/intake_runner.py` — APPROVE uses `execution_identity.require()`.
  2. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/human/hitl_hooks.py` — `run_id=task.task_id` in hook context.
  3. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/nexus/orchestration/hitl_runner.py` — REJECT/ESCALATE events use `task.task_id` as `run_id`.
  4. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/events/trace_bridge.py` — `runtime_event_from_task_state` canonical identity contract.
- **Impact:** Runtime/security/audit provenance can be mis-correlated or contract-invalid on residual HITL/lifecycle branches.
- **Confidence:** CONFIRMED

### AUDIT-20260818-IDENTITY_TRUST-06

**ActorIdentity is a paper identity model with inconsistent wire semantics**

- **Severity:** MEDIUM
- **Category:** ARCHITECTURE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** IDT-FIX-A
- **Claim falsified:** Intergrax has one canonical actor/principal identity model whose producer, resolver and authority semantics are consistently used on production execution paths.
- **Observation:** `ActorIdentity` contains `kind`, `actor_id`, `tenant_id`, `delegated_from` and `permission_scopes`. `resolve_actor_from_task` and `narrow_delegation_scopes` exist but no production use was established in this audit. `TaskEnvelope.with_actor` writes `actor_kind` + `actor_id`. Resolver reads `service_id` for SERVICE, `agent_actor_id` for AGENT and `task.user_id` for USER, so writer and reader do not form one coherent wire contract. `ActorIdentity.allows_scope` treats empty scopes as unrestricted; do not call this independently exploitable because `ActorIdentity` is not currently proven to be the production authority gate.
- **Location:**
  - `intergrax/contracts/actor_identity.py` — `ActorIdentity`, `allows_scope` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/contracts/task_envelope.py` — `TaskEnvelope.with_actor` @ `6fbc5e4928963ecd386456158b0753662fed209b`
  - `intergrax/runtime/interactions/actor_resolution.py` — `resolve_actor_from_task` @ `6fbc5e4928963ecd386456158b0753662fed209b`
- **Reproduction:**
  1. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/contracts/task_envelope.py` — `with_actor` writes `actor_kind`/`actor_id` metadata keys.
  2. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/runtime/interactions/actor_resolution.py` — resolver reads `service_id`/`agent_actor_id`/`task.user_id` by kind.
  3. `git grep -n "resolve_actor_from_task\|resolve_actor_from_envelope" 6fbc5e4928963ecd386456158b0753662fed209b -- intergrax/ applications/` — no production consumer path established.
  4. `git show 6fbc5e4928963ecd386456158b0753662fed209b:intergrax/contracts/actor_identity.py` — empty `permission_scopes` treated as unrestricted in `allows_scope`.
- **Impact:** Platform identity concepts exist in parallel without one mandatory principal/actor spine, encouraging paper abstractions and future inconsistent enforcement.
- **Confidence:** CONFIRMED

## Provider/backend abstraction classification matrix

| Concern | Canonical abstraction | Observed pattern | Classification | Notes |
|---------|----------------------|------------------|----------------|-------|
| Identity verification | `IdentityProviderBackend` | harness + integration profile wiring | ABSTRACTION_PRESERVED | positive control — verified user not propagated to execution identity (finding 01) |
| FastAPI authentication | `AuthProvider` → `AuthContext` | middleware-resolved context | ABSTRACTION_PRESERVED | positive control — not all Tier-3 hosts consume it uniformly |
| Integration profile | `IntegrationProfile` | composition-owned backend selection | ABSTRACTION_PRESERVED | no new vendor leakage |

**AUDIT-5 discovered no new independent vendor/backend leakage finding** in this layer.

## Falsification log

Targets examined but **not** promoted to findings:

1. **All Intergrax authentication is broken** — FastAPI Core has a proper `AuthProvider` → `AuthContext` pattern (`intergrax/fastapi_core/auth/`).
2. **Legal necessarily has the same body-controlled identity defect** — Legal rejects body/context `tenant_id`/`user_id` conflicts (`applications/legal_application/serving/fastapi_router.py`).
3. **Current G5C approval can be replayed onto another pause/request** — exact task/pause/human_request matching is a positive control (`HumanPauseCoordinator.resolve_human_response`).
4. **JwtAuthProvider / NoOpJwtVerifier production defect** — no proven production reachability in this layer.
5. **DelegationSpec bypasses all existing tool policy** — finding 02 concerns missing additional authority narrowing from `permission_scopes`, not absence of all tool gates (`DelegationToolPolicyError` enforces allowlists).
6. **Duplicate INTERFACE_TASK_INTAKE public TaskId/RunId findings** — finding 05 is specifically internal runtime/HITL provenance after valid execution identity is bound.
7. **Provider/backend abstraction** — `IdentityProviderBackend` / `IntegrationProfile` abstraction remains a positive control; no new vendor/backend leakage finding.

## Prior-audit comparison

Prior campaign layers [`INTERFACE_TASK_INTAKE`](INTERFACE_TASK_INTAKE.md), [`PROVIDER_BACKEND_ABSTRACTION`](PROVIDER_BACKEND_ABSTRACTION.md), and [`STRATEGIC_HARNESS_MODEL`](STRATEGIC_HARNESS_MODEL.md) established intake identity minting, provider abstraction, and execution-identity continuity themes. This layer owns **identity/trust-specific** claims: authenticated principal spine, delegated authority enforcement, human approver provenance, resume surface alignment, execution-identity closure on residual paths, and actor/principal model coherence. No prior canonical Protocol v2.2 `IDENTITY_TRUST` immutable snapshot existed before this layer.

## Open questions / blocked items

- Whether harness `IdentityUser` should map directly into `RequestIdentity` or through an explicit Tier-3 bridge type — planning only (**IDT-FIX-A**).
- Whether delegation `permission_scopes` should converge with `CollaborativeWorkAuthorityResolver` semantics or remain a separate Nexus graph authority envelope — planning only (**IDT-FIX-B**).
- Shared resume route authentication requirements for approver provenance — deferred to **IDT-FIX-C**.
- No operator-disputed findings; no blocked evidence collection.

## Operator acceptance

- **Date:** 2026-08-18
- **Accepted findings:** all 6 (`AUDIT-20260818-IDENTITY_TRUST-01` … `AUDIT-20260818-IDENTITY_TRUST-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none
- **Remediation blocks:** IDT-FIX-A, IDT-FIX-B, IDT-FIX-C, IDT-FIX-D — all **ACCEPTED / PLANNED** only; not implemented by this persistence task

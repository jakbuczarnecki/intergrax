# GV-01 — Governance Adoption Sweep (qualification)

**Task:** GV-01-C1R — Clean Qualification & Q20 Ownership Closure  
**AUDITED_HEAD / TESTED_HEAD:** `13db6b2e65c654a2736966aba3f7d61bfc06ee14` (`fix(hitl): fail closed on missing approver during task resume`; matches `origin/development`)  
**Working tree at qualification:** **clean** (before and after C1R documentation commit)  
**GV-01-C1R status:** **CLOSED** (clean tree; Governance-owned production blockers = 0)  
**GV-01 final:** **CLOSED / ENTERPRISE QUALIFIED**

## 1. Canonical governance boundary

| Layer | Canonical entry | Role |
| ----- | ----------------- | ---- |
| Shared pre-effect (non-ToolRuntime) | `MeaningfulSideEffectAuthorizationBoundary.authorize` / `authorize_and_execute` | Fresh collaborative + runtime policy composition; no physical execution |
| Collaborative composition | `CollaborativeWorkEnforcementGate.evaluate` | Workspace / authority / meaningful-side-effect policy layers |
| ToolRuntime (frozen TR-01) | `require_meaningful_side_effect_authorization` + `RuntimeToolInvoker` | Declarative enforcer must be `ENFORCE`; side-effect tools fail closed otherwise |
| Execution admission | `runtime_execution_policy_admission.evaluate_root_execution_admission` | Root execution policy (not a substitute for per-effect fresh auth) |
| Memory (domain) | `MemorySecurityGovernanceService` + `MemoryGovernanceEvaluationRequest` | Typed memory mutation gate (MEM ownership) |
| External Work (production host) | `ExternalWorkAdapter` → boundary immediately before provider dispatch | GR-7-A2/A3 durable intent + ERL observation (committed on baseline SHA) |

Governance **does not** execute providers, tools, or persistence writes. Governance **does not** own retry, provider durability, or reconciliation.

## 2. Fresh side-effect input / output contracts

- **Input:** `MeaningfulSideEffectRequest` (`intergrax/contracts/meaningful_side_effect.py`) — **strict** execution identity spine (`task_id`, `run_id`, `attempt_id`, `execution_id`), action, kinds, scope, tenant, optional decision material.
- **Envelope:** `CollaborativeWorkEnforcementRequest`.
- **Output:** `PolicyDecision` / `PolicyAction` via `MeaningfulSideEffectAuthorizationResult`.

**C1 contract posture:** UNCHANGED — no optional identity fields, no production loosening.

## 3. Meaningful side-effect identity fixtures (C1)

| Item | Result |
| ---- | ------ |
| Canonical helper | `minimal_meaningful_side_effect_request_for_tests` in `tests/unit/runtime/governance/gr3_test_support.py` |
| Stale fixtures corrected | `test_policy_composition.py`, `test_g5c2b1r1_external_work_scope_provenance.py`, `tests/e2e/collaborative_work/harness/scenario_runner.py` |
| Root cause | Pydantic v2 strict `AttemptId` / `ExecutionId` on `MeaningfulSideEffectRequest` after execution-identity spine hardening |

## 4. Retry / redelivery authorization (P0-SAFETY-8)

| Topic | Finding |
| ----- | ------- |
| Root cause | Ordering proof patched `time.sleep` while runtime retry backoff uses `cooperative_delay_seconds` (poll loop → thousands of `sleep` events). |
| Extra authorization event | **C** — test instrumentation artefact (over-broad `authorization` label + wrong delay hook), not duplicate runtime side-effect auth. |
| Model | Each physical retry attempt: `cooperative_delay` (if backoff) → `_require_current_attempt_authorization` (scope + governance gates) → executor. Read-only retries: fresh **attempt** authorization includes scope check; not conflated with meaningful side-effect policy gate unless tool is side-effecting. |
| Retry-safe side effect | `EXPLICITLY_RETRY_SAFE` + flaky executor: N physical attempts → N `attempt_authorization` counts (test `test_explicit_retry_safe_side_effect_retries_when_authorized`). |
| Unsafe side effect | Default side effect: physical calls ≤ 1 despite `max_attempts>1`. |
| Revoked authority | Scope revoked during backoff → second physical call count 0; ordering `attempt_authorization → executor → sleep → attempt_authorization` with `cooperative_delay_seconds` patched. |
| Unknown outcome | Timeout + idempotency: second invoke raises `InvocationUncertaintyError`; physical count stays 1. |
| Fresh auth ≠ idempotency | Preserved — governance ALLOW does not override uncertainty store. |

## 5. Side-effect adoption matrix (representative)

| Domain | Effect | Canonical entry | Fresh auth | Physical owner | Status | Follow-up |
| ------ | ------ | --------------- | ---------: | -------------- | ------ | --------- |
| ToolRuntime | Tool side effect | `RuntimeToolInvoker` | Yes | Tool executor | QUALIFIED | — |
| External Work | Provider mutation | `ExternalWorkAdapter` | Yes (`authorize_and_execute`) | Adapter / provider SDK | QUALIFIED | — |
| Collaborative Work | Work artifact publish | `CollaborativeWorkArtifactService` | Yes | Artifact repository | QUALIFIED | — |
| Decision-bound effects | Decision execution actions | `authorize_and_execute_decision_bound_side_effect` | Yes | Domain callback | QUALIFIED | — |
| Memory | Durable memory write | `MemorySecurityGovernanceService` | Yes | Memory stores | QUALIFIED | MEM-01 architecture |
| Marketplace / tool acquisition | Host activation | `DynamicToolAcquisitionService` | Partial (lifecycle handoff) | Tool host | FOLLOW-UP | PLUG-01 / ME-14 |
| Plugin lifecycle | Install / trust | Plugin host paths | Not unified on MSE | Plugin domain | FOLLOW-UP | PLUG-01 |
| Background / scheduled work | Deferred execution | Enqueue vs execute | Admission ≠ effect time | Execution host | FOLLOW-UP | BG-01 / SCHED-01 |
| Host/API direct | Ad-hoc mutations | Tier-3 composition | Composition-dependent | HOST-01 |
| Secrets / credentials | Rotate / bind | Secrets ports | Domain-specific | FOLLOW-UP | Secrets roadmap |
| Sandbox physical | Non-tool effects | Sandbox boundary | SBX-qualified | NOT_APPLICABLE | SBX-01 |

**UNKNOWN paths:** 0

## 6. External Work GR-7-A3 (committed baseline)

Audited on `365082a4`: fresh governance before provider mutation; durability/reconciliation owned by adapter/ERL — not Governance engine. Unit suite: `agents/external_contractor_adapter/tests/**` + adapter governance tests **PASS** (63 tests in C1 run).

## 7. GV qualification collection vs full repo

| Scope | Result |
| ----- | ------ |
| **GV CORE COLLECTION (C1R)** | **PASS** — `tests/unit/runtime/governance` (186 tests; Q20 deselected as SESSION follow-up) + `test_p0_safety_8_retry_redelivery_authorization.py` + `test_ee_b3_a_governance_bypass_gate` + `test_tool_runtime_authority_closure` + GR-7-A3 host suite (12 tests) |
| **GV COLLECTION ERRORS** | **0** on bounded GV core paths |
| **GV-OWNED FAILURES** | **0** — remaining Nexus intake HITL integration failures are harness / Execution continuation (not Governance contract) |
| **FULL REPO COLLECTION** | **FAIL** — 6 errors (`pytest_plugins` non-top-level conftest; integration syntax) — owners: CI / integration maintainers; does not block GV-01 |

## 8. GV-Q1–Q20 (summary)

| ID | Status | Notes |
| -- | ------ | ----- |
| Q1–Q10 | PASS | Core MSE boundary, GR-3 inner guard, policy composition, P0-SAFETY-8 retry proofs |
| Q11–Q15 | PASS | HITL bridge/matcher/reauth suites (fixture-aligned) |
| Q16 | PASS | `test_ee_b3_a_governance_bypass_gate` |
| Q17 | PASS | External Work adapter governance tests |
| Q18 | FOLLOW-UP | Marketplace shared MSE adoption — ME-14 / PLUG-01 |
| Q19 | FOLLOW-UP | BG-01 / SCHED-01 deferred execution auth |
| Q20 | FOLLOW-UP (SESSION-01 / Execution) | `test_nexus_intake_governed_approval_without_nexus_ae_forwarding` — see §11 |

## 9. Downstream follow-ups

| Owner | Topic |
| ----- | ----- |
| SESSION-01 / Execution | Nexus intake HITL qualification harness: wire `InternalOrchestrationContinuation` like `NexusLoop` (`wire_execution_engine_continuation_dependencies`); register pause via `establish_canonical_hitl_pause` (not legacy `apply_pause` alone); align approve-path grant ordering with `execution_continuation_projection` resume semantics |
| HOST-01 | Host/API convergence |
| PLUG-01 | Plugin lifecycle / marketplace MSE |
| BG-01 / SCHED-01 | Deferred execution fresh auth |
| ME-14 | Marketplace qualification |
| CI | Full-repo collection errors (pytest_plugins layout) |

## 10. Verdict

**GV-01 final:** **CLOSED / ENTERPRISE QUALIFIED** — C1R ran on clean `origin/development` at `13db6b2e`. MSE contract unchanged; P0-SAFETY-8 and GR-3 governance suites green; External Work GR-7-A3 regression green; ToolRuntime authority closure green; **Governance-owned production bypass = 0**. Q20 is explicitly **not** Governance-owned (§11). No Governance Engine redesign; TR-01 frozen.

**NEXT recommended:** **SESSION-01 — Session/Checkpoint SSOT** (canonical execution continuation + intake/HITL harness convergence).

## 11. C1R — Q20 ownership (evidence)

**Question:** Governance contract error (A) vs Nexus/SESSION continuation harness (B)?  
**Verdict:** **B** — TEST-HARNESS + **SESSION / Execution continuation**; **not** Governance authorization semantics.

| Step | Owner |
| ---- | ----- |
| `MeaningfulSideEffectRequest` | Governance contracts |
| `CollaborativeWorkEnforcementRequest` | Governance / collaborative work |
| `compose_governed_continuation_from_enforcement` | Governance bridge (`governed_continuation_bridge`) |
| `bridge_governed_continuation_to_execution_result` | Governance → execution projection bridge |
| `HumanPauseCoordinator.apply_pause` | Human/SESSION projection (**non-authoritative**; docstring: does not establish canonical PAUSED/WAITING) |
| `NexusIntakeRunner.run` → `require_internal_hitl_continuation` | **Execution / Nexus** (`intake_runner.py`) |
| `establish_canonical_hitl_pause` / `ExecutionContinuationPort` | **Execution continuation** (`internal_continuation_orchestration`, `execution_continuation/service.py`) |
| `GovernedContinuationGrantCoordinator` | Governance (grant rules); invoked from intake after canonical resolution |

**Failure on C1R HEAD (`13db6b2e`):**

| Field | Value |
| ----- | ----- |
| Exception | `InternalHitlContinuationCapabilityError` |
| Failing function | `require_internal_hitl_continuation` (`internal_continuation_orchestration.py:72`) |
| Missing state | `NexusIntakeRunner.hitl_continuation is None` in `_build_intake_runner_with_hitl` while production `NexusLoop` wires `InternalOrchestrationContinuation` |
| Expected owner | Execution composition (`nexus_loop.py` wires continuation port + lifecycle driver) |
| Actual gap | Unit test harness omits production wiring; uses legacy `apply_pause` without continuation store registration |
| Governance production impact | **none** — deny/require-human paths and MSE boundary unchanged |
| Blocks GV-01 | **NO** (bounded subsystem qualification; downstream SESSION-01) |

**Q20 follow-up:** SESSION-01 — align intake/HITL tests with canonical continuation lifecycle; optional EE sequencing review for governed grant creation vs `canonical_resume_after_authorization` (approve path).

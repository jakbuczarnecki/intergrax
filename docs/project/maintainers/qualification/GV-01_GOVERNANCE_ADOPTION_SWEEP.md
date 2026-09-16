# GV-01 — Governance Adoption Sweep (qualification)

**Task:** GV-01-C1 — Governance Qualification Fixture & Retry Proof Closure  
**AUDITED_HEAD / TESTED_HEAD (session):** `365082a4e3c789f60864c3c86f73f30fd3baa529` (matches `origin/development` at session start)  
**Working tree at qualification:** **contaminated** — unrelated WIP (memory, governed-contractor host) present locally; GV proof executed on bounded file set only.  
**GV-01-C1 status:** **CORRECTION REQUIRED** (one GV-owned governance test + full-repo collection blockers; no Governance redesign required)

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
| **GV QUALIFICATION COLLECTION** | **PASS** — `tests/unit/runtime/governance`, `tests/unit/runtime/security`, selected policy/HITL/collab tests (278+ collected) |
| **FULL REPO COLLECTION** | **FAIL** — 6 errors (pytest_plugins non-top-level conftest; integration syntax/plugin issues) — not GV-owned framework redesign in C1 |

## 8. GV-Q1–Q20 (summary)

| ID | Status | Notes |
| -- | ------ | ----- |
| Q1–Q10 | PASS | Core MSE boundary, GR-3 inner guard, policy composition, P0-SAFETY-8 retry proofs |
| Q11–Q15 | PASS | HITL bridge/matcher/reauth suites (fixture-aligned) |
| Q16 | PASS | `test_ee_b3_a_governance_bypass_gate` |
| Q17 | PASS | External Work adapter governance tests |
| Q18 | FOLLOW-UP | Marketplace shared MSE adoption — ME-14 / PLUG-01 |
| Q19 | FOLLOW-UP | BG-01 / SCHED-01 deferred execution auth |
| Q20 | FAIL | `test_nexus_intake_governed_approval_without_nexus_ae_forwarding` — canonical execution continuation harness drift (EE intake + HITL); not a Governance contract change |

## 9. Downstream follow-ups

| Owner | Topic |
| ----- | ----- |
| EE / SESSION-01 | Nexus intake HITL tests require `InternalOrchestrationContinuation` + canonical pause registration (`test_gr1_*` nexus path) |
| HOST-01 | Host/API convergence |
| PLUG-01 | Plugin lifecycle / marketplace MSE |
| BG-01 / SCHED-01 | Deferred execution fresh auth |
| ME-14 | Marketplace qualification |
| CI | Full-repo collection errors (pytest_plugins layout) |

## 10. Verdict

**GV-01 final:** **CORRECTION REQUIRED** — C1 closes MSE fixture drift and P0 retry ordering proof; one GR-1 nexus intake qualification test and contaminated local tree block **CLOSED / ENTERPRISE QUALIFIED**. No Governance Engine redesign; layer boundaries preserved; TR-01 frozen.

**NEXT recommended:** bounded GV correction for GR-1 nexus intake test harness **or** SESSION-01 after clean tree.

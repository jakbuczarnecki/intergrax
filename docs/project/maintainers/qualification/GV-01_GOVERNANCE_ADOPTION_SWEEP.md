# GV-01 — Governance Adoption Sweep (qualification)

**Task SHA (audit baseline):** `f606fb325726c34c950c5011c04913721c1d3e02`  
**Origin development at audit start:** `38be8ae40373937afa7e50bf907f24313991d399`  
**Status:** CORRECTION REQUIRED (see report in session; uncommitted GR-7-A3 WIP + qualification test drift)

## 1. Canonical governance boundary

| Layer | Canonical entry | Role |
| ----- | ----------------- | ---- |
| Shared pre-effect (non-ToolRuntime) | `MeaningfulSideEffectAuthorizationBoundary.authorize` / `authorize_and_execute` | Fresh collaborative + runtime policy composition; no physical execution |
| Collaborative composition | `CollaborativeWorkEnforcementGate.evaluate` | Workspace / authority / meaningful-side-effect policy layers |
| ToolRuntime (frozen TR-01) | `require_meaningful_side_effect_authorization` + `RuntimeToolInvoker` | Declarative enforcer must be `ENFORCE`; side-effect tools fail closed otherwise |
| Execution admission | `runtime_execution_policy_admission.evaluate_root_execution_admission` | Root execution policy (not a substitute for per-effect fresh auth) |
| Memory (domain) | `MemorySecurityGovernanceService` + `MemoryGovernanceEvaluationRequest` | Typed memory mutation gate (MEM ownership) |
| External Work (production host) | `ExternalWorkAdapter` → boundary immediately before provider dispatch | GR-7-A2/A3 durable intent + ERL observation |

Governance **does not** execute providers, tools, or persistence writes.

## 2. Fresh side-effect input / output contracts

- **Input:** `MeaningfulSideEffectRequest` (`intergrax/contracts/meaningful_side_effect.py`) — execution identity spine (`task_id`, `run_id`, `attempt_id`, `execution_id`), action, kinds, scope, tenant, optional decision material.
- **Envelope:** `CollaborativeWorkEnforcementRequest`.
- **Output:** `PolicyDecision` / `PolicyAction` (`ALLOW`, `DENY`, `REQUIRE_HUMAN`, `ESCALATE`, …) via `MeaningfulSideEffectAuthorizationResult`; not truthy/falsy.

## 3. Policy snapshot / revision

- Live composition: `RuntimePolicyBundle` + `DeclarativePolicyRuntime` (`intergrax/runtime/policy/policy_bundle.py`).
- Immutable attestation: `PolicyBundleProvenance` on declarative runtime; pinned evidence via existing evidence plane (no second store).

## 4. Side-effect adoption matrix (representative)

| Domain | Effect | Canonical entry | Fresh auth | Physical owner | Evidence | Status | Follow-up |
| ------ | ------ | --------------- | ---------: | -------------- | -------: | ------ | --------- |
| ToolRuntime | Tool side effect | `RuntimeToolInvoker` | Yes (pre-invoke gate) | Tool executor | Tool + policy evidence | QUALIFIED | — |
| External Work | Provider mutation | `ExternalWorkAdapter._run_meaningful_side_effect` | Yes (`authorize_and_execute`) | Adapter / provider SDK | Adapter result + ERL (GR-7-A2) | QUALIFIED | GR-7-A3 durability (WIP) |
| Collaborative Work | Work artifact publish | `CollaborativeWorkArtifactService` | Yes (MP-1 enforcement gate) | Artifact repository | Work artifact records | QUALIFIED | — |
| Decision-bound effects | Decision execution actions | `authorize_and_execute_decision_bound_side_effect` | Yes | Domain callback | Decision governance material | QUALIFIED | — |
| Memory | Durable memory write | `MemorySecurityGovernanceService` | Yes (domain evaluator) | Memory stores | Memory governance outcome | QUALIFIED | MEM architecture N/A for GV |
| Marketplace / tool acquisition | Host activation / registry | `DynamicToolAcquisitionService` | Partial (lifecycle handoff; not shared MSE boundary) | Tool host / registry | Handoff ack | PARTIAL | GV-01 thin adapter or ME-14 re-qualify |
| Plugin lifecycle | Install / trust | Plugin host paths | Not unified on MSE boundary | Plugin domain | Varies | PARTIAL | PLUG-01 |
| Background / scheduled work | Deferred execution | Enqueue vs execute split | Admission ≠ effect time | Execution host | Varies | PARTIAL | BG-01 / SCHED-01 |
| Host/API direct | Ad-hoc mutations | Tier-3 composition | Composition-dependent | Application host | Host tests | PARTIAL | HOST-01 where convergence gap |
| Secrets / credentials | Rotate / bind | Secrets subsystem ports | Domain-specific | Secrets store | Policy-bound | PARTIAL | Secrets roadmap |
| Sandbox physical | Non-tool effects | Sandbox boundary | SBX-qualified isolation | Sandbox runtime | Capability evidence | NOT_APPLICABLE | SBX-01 for isolation depth |

**UNKNOWN paths:** 0 (all rows classified).

## 5. Freshness semantics

Authorization must occur in `authorize_and_execute` (or ToolRuntime invoker) **immediately before** the physical callback. Workflow admission, planning, or enqueue authorization does not substitute.

## 6. HITL

`REQUIRE_HUMAN` / `ESCALATE` → governed continuation; grant must match side-effect identity (`GovernedContinuationGrantCoordinator`). Approval does not call providers directly.

## 7. Downstream follow-ups

| Owner | Topic |
| ----- | ----- |
| GV-01 | Qualification test fixtures (`attempt_id` / `execution_id` on `MeaningfulSideEffectRequest`); P0 retry ordering test drift |
| GR-7-A3 | Durable `ProviderInvocationStore` (uncommitted WIP) |
| PLUG-01 | Plugin trust / lifecycle governance unification |
| BG-01 / SCHED-01 | Execution-time auth for deferred work |
| HOST-01 | Host/API convergence |
| ME-14 | Marketplace acquisition test failures on HEAD |

## 8. Qualification tests (target)

GV-Q1–Q20 mapped to: `test_ee_b3_*`, `test_p0_safety_8_*`, `test_gr3_*`, `test_gr6_*`, `test_gr7_*`, `test_mem_ent10b_*`, enforcement gate suite (currently **fixture drift** on HEAD).

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.
# Use, modification, or distribution without written permission is prohibited.

# ADAPTIVE_HARNESS_INTELLIGENCE - ADAS / Agent Design Search Implementation Plan

**Parent hub:** [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](../ADAPTIVE_HARNESS_INTELLIGENCE.md)  
**Architecture hub:** [`architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md`](../../../architecture/ADAPTIVE_HARNESS_INTELLIGENCE.md)
**Architecture satellite:** [`ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md`](../../../architecture/satellites/ADAPTIVE_HARNESS_INTELLIGENCE_agent_design_search.md)
**ADR:** [`ADR-ADAPT-002`](../../../technical/adr/entries/2026-06-22/ADR-ADAPT-002.md)
**Status:** Planned - Phase **AHI-ADAS-00** complete; implementation begins at **AHI-ADAS-10**  
**Last updated:** 2026-06-22

---

## 1. Pre-implementation architecture audit

**Audit mode:** Mode I / slice audit before runtime implementation.  
**Audit scope:** ADAS architecture satellite, AHI hub canon, AHI plan hub, ADR-ADAPT-002.  
**Audit result:** **Passed with no blocking architecture issues.**

### 1.1 Evidence reviewed

| Artifact | Result |
|----------|--------|
| AHI hub canonical ADAS section | Present; states ADAS is an AHI Tier-1 sub-capability, not a separate layer |
| ADAS architecture satellite | Present; canonical; contains enterprise contracts, evidence bundle, retention, active registration semantics |
| AHI plan hub | Present; contains Phase AHI-ADAS register |
| ADR-ADAPT-002 | Present; accepted; records ADAS-inside-AHI decision |
| AHI existing components | ADAS explicitly reuses signal, utility, governance, verification, process mining, scaffold, observability, and promotion semantics |

### 1.2 Audit verdict

ADAS is ready for implementation planning because the documentation now enforces these invariants:

```text
ADAS lives inside AHI, not beside it.
ADAS is not Tier-3-only.
MAS is a strategy, not the architecture.
Scaffold is the only candidate materialization path.
Candidates are gated, evaluated, archived, and promoted through governed lifecycle stages.
Production mutation requires explicit promotion and human approval by default.
```

### 1.3 Remaining implementation risks

| Risk | Mitigation in this plan |
|------|-------------------------|
| Recreating AHI governance in ADAS | All tasks must reuse existing AHI governance / lifecycle semantics |
| MAS writing arbitrary code | Scaffold bridge and static gate are mandatory before evaluation |
| Candidate archive becoming unauditable | `AgentCandidateEvidenceBundle`, operational envelope, retention, and tenant scoping are required |
| Promotion semantics drifting | Active registration semantics are implemented as explicit modes A–E |
| Cost explosion | Budget policy and stop conditions are required before MAS agents |
| Tier leakage into applications | ADAS Lab is delayed until Tier-1 control plane is complete |

### 1.4 Gate before code

Before implementing **AHI-ADAS-10**, run a scoped audit with these checks:

```text
1. Read AHI hub read-scope block.
2. Read ADAS architecture satellite §1–§13 and §30.
3. Read this plan satellite §1–§8.
4. Confirm no runtime code exists yet under intergrax/runtime/adaptive/agent_design_search/.
5. Confirm Phase AHI-ADAS-00 is marked Done in the plan hub.
6. Confirm ADR-ADAPT-002 is accepted.
7. Start implementation only from AHI-ADAS-10.
```

---

## 2. Phase AHI-ADAS - Agent Design Search

**Goal:** Implement ADAS as an enterprise-grade, governed agent-candidate design loop inside `ADAPTIVE_HARNESS_INTELLIGENCE`.

**Core loop:**

```text
objective
  → baseline
  → candidate draft
  → scaffold bridge
  → static gate
  → offline evaluation
  → utility scoring
  → archive + evidence bundle
  → shadow/canary/promotion request
  → verification
  → keep or rollback
```

**Delivery rule:** One **AHI-ADAS-*** phase or narrowly scoped sub-phase per PR. Each PR must update this plan and include evidence for the touched gates.

**Implementation sequence:**

```text
AHI-ADAS-00  Documentation and ADR                    Done
AHI-ADAS-10  Contracts and candidate archive           Planned
AHI-ADAS-20  Scaffold bridge and static gate           Planned
AHI-ADAS-30  Candidate evaluation and utility          Planned
AHI-ADAS-40  Search controller and strategies          Planned
AHI-ADAS-50  Hooks and lifecycle events                Planned
AHI-ADAS-60  Optional Tier-2 MAS agents                Planned
AHI-ADAS-70  Shadow/canary/promotion bridge            Planned
AHI-ADAS-80  Optional Tier-3 ADAS Lab                  Planned
AHI-ADAS-90  Enterprise hardening                      Planned
```

---

## 3. Normative implementation principles

1. **Extend AHI, do not fork it.** ADAS must live under `intergrax/runtime/adaptive/agent_design_search` and reuse AHI patterns.
2. **No parallel governance stack.** Use existing policy/governance concepts; add adapters only where agent candidates differ from profile versions.
3. **No parallel scaffold.** Candidate materialization must go through existing scaffold via `AgentScaffoldBridge`.
4. **No direct production mutation.** Candidate source starts in sandbox/archive; production routing requires promotion.
5. **Evidence over declaration.** Promotion and verification must be backed by `AgentCandidateEvidenceBundle`.
6. **Human approval by default.** Production promotion requires approval unless a future explicit product gate allows low-risk lab auto-promotion.
7. **Tenant isolation by default.** Candidate archive, objectives, search runs, and bundles are tenant-scoped.
8. **Budget before autonomy.** Search budget policy must exist before MAS agents are allowed to generate multiple candidates.
9. **Static gate before evaluation.** Unsafe candidates are rejected and archived before any evaluation run.
10. **Verification before success.** A promoted candidate is not successful until it passes the verification window.

---

## 4. Out of scope for Phase AHI-ADAS v1

The following are explicitly out of scope until later product decisions:

```text
Deep RL / neural policy training
Foundation model fine-tuning
Direct mutation of production agents
Direct writes to intergrax/runtime by MAS
Automatic production promotion for high-risk candidates
Cross-tenant learning without anonymization and governance approval
Standalone ADAS SaaS layer
Parallel evaluation registry
Parallel PolicyEngine
Parallel tracing system
```

---

## 5. Traceability - architecture section to task IDs

| Architecture topic | Section | Task IDs |
|--------------------|---------|----------|
| Canonical ADAS inside AHI decision | Architecture §1–§7, ADR-ADAPT-002 | AHI-ADAS-00.* |
| Operational envelope | Architecture §13.0 | AHI-ADAS-10.1–10.2 |
| Objective/search/candidate contracts | Architecture §13.1–§13.6 | AHI-ADAS-10.1–10.6 |
| Evidence bundle | Architecture §13.7 | AHI-ADAS-10.7, AHI-ADAS-30.8, AHI-ADAS-70.6 |
| Candidate archive and lineage | Architecture §19 | AHI-ADAS-10.8–10.12 |
| Retention and cleanup | Architecture §19.1 | AHI-ADAS-90.1–90.4 |
| Scaffold bridge | Architecture §9, §12.4 | AHI-ADAS-20.1–20.4 |
| Static gate | Architecture §20 | AHI-ADAS-20.5–20.10 |
| Evaluation model | Architecture §21 | AHI-ADAS-30.1–30.8 |
| Utility model | Architecture §18 | AHI-ADAS-30.6–30.8 |
| Search controller | Architecture §12.1, §15 | AHI-ADAS-40.1–40.6 |
| Strategy protocol / MAS | Architecture §8, §12.2–12.3 | AHI-ADAS-40.7–40.9, AHI-ADAS-60.* |
| Hooks | Architecture §16 | AHI-ADAS-50.1–50.4 |
| Events | Architecture §17 | AHI-ADAS-50.5–50.8 |
| Promotion bridge | Architecture §22 | AHI-ADAS-70.1–70.5 |
| Active registration semantics | Architecture §22.1 | AHI-ADAS-70.6–70.9 |
| Verification and rollback | Architecture §23 | AHI-ADAS-70.10–70.12 |
| ADAS Lab | Architecture §6.4 | AHI-ADAS-80.* |
| Enterprise hardening | Architecture §24–§29 | AHI-ADAS-90.* |

---

## 6. Master deliverables register

### Phase AHI-ADAS-00 - Documentation canon and ADR

**Status:** **Done** (2026-06-22)  
**Purpose:** Establish ADAS as a canonical AHI sub-capability before implementation.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-00.1 | Add ADAS architecture satellite | Done | Critical | Satellite exists and declares canonical AHI placement |
| AHI-ADAS-00.2 | Add canonical ADAS section to AHI hub | Done | Critical | Hub links to satellite and states non-detached placement |
| AHI-ADAS-00.3 | Add Phase AHI-ADAS to plan hub | Done | Critical | Plan hub lists phases AHI-ADAS-00 through AHI-ADAS-90 |
| AHI-ADAS-00.4 | Add ADR-ADAPT-002 | Done | Critical | ADR accepted; rejects separate layer and Tier-3-only alternatives |
| AHI-ADAS-00.5 | Add enterprise contract/evidence/retention/active-registration detail | Done | High | Satellite includes §13.0, §13.7, §19.1, §22.1 |
| AHI-ADAS-00.6 | Add implementation plan satellite | Done | High | This document exists and mirrors ADAS architecture phases |

**Exit gate:** Architecture-only phase complete. No runtime code required.

---

### Phase AHI-ADAS-10 - Contracts and candidate archive

**Status:** Planned  
**Purpose:** Add the minimal typed substrate for ADAS without executing search.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-10.1 | Package skeleton `intergrax/runtime/adaptive/agent_design_search` | Planned | Critical | Importable package; no runtime side effects; copyright headers present |
| AHI-ADAS-10.2 | `AdasOperationalEnvelope` | Planned | Critical | Schema version, audit IDs, approval IDs, trace/correlation, hashes, risk, retention fields |
| AHI-ADAS-10.3 | `AgentDesignObjective` | Planned | Critical | Tenant-scoped objective with baseline, target capability, budgets, approval mode |
| AHI-ADAS-10.4 | `AgentDesignSearchRun` | Planned | Critical | Search lifecycle record with generated candidates, costs, status, stop reason |
| AHI-ADAS-10.5 | `AgentCandidateDraft` | Planned | Critical | Draft candidate with lineage, scaffold pattern, capabilities, contract/prompt/tool policy drafts |
| AHI-ADAS-10.6 | `AgentCandidateRecord` + `AgentCandidateStatus` | Planned | Critical | Immutable candidate record with status, lineage, utility, rejection, promotion refs |
| AHI-ADAS-10.7 | `AgentCandidateEvaluationResult` | Planned | High | Mirrors signal/eval metrics: quality, cost, latency, tokens, tool calls, regression flags, utility delta |
| AHI-ADAS-10.8 | `AgentCandidateEvidenceBundle` | Planned | Critical | Sealed bundle can reconstruct promotion/audit report without scattered recomputation |
| AHI-ADAS-10.9 | `AgentDesignArchive` protocol | Planned | Critical | Append, get, list by tenant/objective/status/lineage; no cross-tenant list by default |
| AHI-ADAS-10.10 | In-memory archive | Planned | High | Unit-testable store with tenant isolation and append-only semantics |
| AHI-ADAS-10.11 | SQLite archive | Planned | Critical | Persists under `build/adaptive_harness/agent_design_search/candidates.db` |
| AHI-ADAS-10.12 | Archive lineage queries | Planned | High | Parent/child lineage listing; rejected and promoted candidates visible |
| AHI-ADAS-10.13 | Package exports | Planned | Medium | `__init__.py` exports contracts and stores consistently with `runtime/adaptive/__init__.py` style |
| AHI-ADAS-10.14 | Unit tests for contracts and archive | Planned | Critical | Validation, tenant isolation, append-only behavior, sealed bundle immutability |

**Gates:**

```text
pytest tests/unit/runtime/adaptive/agent_design_search/
python -m compileall intergrax/runtime/adaptive/agent_design_search
```

**Non-goals:** No scaffold bridge, no evaluation runner, no MAS, no promotion.

---

### Phase AHI-ADAS-20 - Scaffold bridge and static gate

**Status:** Planned  
**Purpose:** Materialize candidate agents safely through the existing scaffold and block unsafe candidates before evaluation.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-20.1 | `AgentScaffoldBridge` protocol/facade | Planned | Critical | Converts `AgentCandidateDraft` to existing scaffold invocation model |
| AHI-ADAS-20.2 | Sandbox materialization path | Planned | Critical | Candidate output under `build/adaptive_harness/agent_design_search/candidates/<candidate_id>` |
| AHI-ADAS-20.3 | Scaffold pattern validation | Planned | High | Allows only known scaffold/cognitive patterns unless registry declares extension |
| AHI-ADAS-20.4 | Scaffold output manifest | Planned | High | Captures files, hashes, contract ref, prompt refs, smoke test refs |
| AHI-ADAS-20.5 | `AgentStaticGate` | Planned | Critical | Runs before evaluation; failed candidates archived with reason |
| AHI-ADAS-20.6 | AgentContract validation | Planned | Critical | Contract exists, schema valid, risk and max steps declared |
| AHI-ADAS-20.7 | Capability validation | Planned | Critical | Capability IDs are syntactically valid and allowed for objective |
| AHI-ADAS-20.8 | Forbidden import/path checks | Planned | Critical | Blocks `applications.*`, runtime edits, direct PolicyEngine bypass, forbidden file paths |
| AHI-ADAS-20.9 | Tool policy validation | Planned | High | Candidate declares allowed/forbidden tools and no unmanaged tool access |
| AHI-ADAS-20.10 | Smoke test presence check | Planned | High | Generated candidate has smoke test or documented evaluation stub |
| AHI-ADAS-20.11 | Static gate result model | Planned | High | Structured result can be embedded into evidence bundle |
| AHI-ADAS-20.12 | Unit tests for bridge/gate | Planned | Critical | Unsafe candidate fixtures rejected; safe scaffolded fixture passes |

**Gates:**

```text
pytest tests/unit/runtime/adaptive/agent_design_search/test_scaffold_bridge.py
pytest tests/unit/runtime/adaptive/agent_design_search/test_static_gate.py
```

**Non-goals:** No offline quality evaluation and no production registration.

---

### Phase AHI-ADAS-30 - Candidate evaluation and utility scoring

**Status:** Planned  
**Purpose:** Compare candidate agents against baselines using existing evaluation, signal, cost, and regression concepts.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-30.1 | `AgentCandidateEvaluator` facade | Planned | Critical | Runs candidate evaluation after static gate only |
| AHI-ADAS-30.2 | Baseline runner adapter | Planned | Critical | Resolves baseline agent/candidate/current profile for comparison |
| AHI-ADAS-30.3 | Candidate runner adapter | Planned | Critical | Runs scaffolded candidate in isolated evaluation host |
| AHI-ADAS-30.4 | Golden scenario adapter | Planned | Critical | Candidate and baseline evaluated on same golden set |
| AHI-ADAS-30.5 | Negative/adversarial scenario adapter | Planned | High | Captures negative and adversarial pass rates |
| AHI-ADAS-30.6 | Cost/latency/token collection | Planned | Critical | Metrics align with `HarnessOutcomeSignal` fields |
| AHI-ADAS-30.7 | `compute_agent_candidate_utility()` | Planned | Critical | Computes `U_agent` with quality/cost/latency/HITL/regression/complexity/novelty/transfer terms |
| AHI-ADAS-30.8 | Evidence bundle assembly for evaluation | Planned | Critical | Evaluation, static gate, cost/security results can be sealed |
| AHI-ADAS-30.9 | Archive write after evaluation | Planned | Critical | Candidate record updated with result, utility delta, failure reasons |
| AHI-ADAS-30.10 | Evaluation report CLI/helper | Planned | Medium | Prints best/worst candidates, deltas, cost summary, gate failures |
| AHI-ADAS-30.11 | Unit/integration tests | Planned | Critical | Candidate vs baseline delta deterministic on fixtures |

**Gates:**

```text
pytest tests/unit/runtime/adaptive/agent_design_search/test_candidate_evaluator.py
pytest tests/integration/runtime/adaptive/agent_design_search/test_candidate_evaluation_flow.py
```

**Non-goals:** No search controller loop and no MAS generation.

---

### Phase AHI-ADAS-40 - Search controller and strategies

**Status:** Planned  
**Purpose:** Orchestrate bounded candidate search while keeping candidate generation pluggable.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-40.1 | `AgentDesignSearchController` | Planned | Critical | Executes objective → candidate → scaffold → gate → eval → archive loop |
| AHI-ADAS-40.2 | `AgentDesignContext` | Planned | Critical | Carries tenant, objective, baseline, archive history, budget, hooks |
| AHI-ADAS-40.3 | `AgentDesignSearchPolicy` | Planned | Critical | max iterations/candidates/costs/no-improvement/complexity enforced |
| AHI-ADAS-40.4 | Stop conditions | Planned | High | Stops on budget, max candidates, no improvement, repeated gate failure, operator cancel |
| AHI-ADAS-40.5 | Ranking and selection | Planned | High | Ranks by utility delta with cost/regression penalties; preserves rejected candidates |
| AHI-ADAS-40.6 | Search result model | Planned | High | Contains generated IDs, best candidate, stop reason, costs, report refs |
| AHI-ADAS-40.7 | `AgentDesignStrategy` protocol | Planned | Critical | Strategy returns `AgentCandidateDraft`, never writes files directly |
| AHI-ADAS-40.8 | `RuleBasedVariantStrategy` | Planned | Medium | Deterministic baseline strategy for tests and first MVP |
| AHI-ADAS-40.9 | `MetaAgentSearchStrategy` stub | Planned | Medium | Interface only; no autonomous MAS agent yet |
| AHI-ADAS-40.10 | Controller tests | Planned | Critical | Search flow archives every candidate and respects stop conditions |

**Gates:**

```text
pytest tests/unit/runtime/adaptive/agent_design_search/test_search_controller.py
pytest tests/unit/runtime/adaptive/agent_design_search/test_search_policy.py
```

**Non-goals:** Tier-2 MAS agents and production promotion.

---

### Phase AHI-ADAS-50 - Hooks and lifecycle events

**Status:** Planned  
**Purpose:** Make ADAS extensible without letting plugins bypass safety boundaries.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-50.1 | `AgentDesignHook` protocol | Planned | Critical | Hooks support search start, candidate generated, scaffolded, gate, eval, archive, rank, promotion request, stop |
| AHI-ADAS-50.2 | Hook dispatcher | Planned | Critical | Ordered hook calls; structured failures; hooks cannot bypass gates |
| AHI-ADAS-50.3 | Custom evaluator hook | Planned | High | Adds evaluator output but cannot mark unsafe candidate safe |
| AHI-ADAS-50.4 | Custom ranker hook | Planned | Medium | Can adjust ranking within policy constraints |
| AHI-ADAS-50.5 | `AgentDesignEvent` model | Planned | Critical | Event includes event_id, timestamp, tenant_id, objective_id, search_run_id, candidate_id, correlation_id, trace_id, actor |
| AHI-ADAS-50.6 | Event emitter adapter | Planned | High | Reuses existing observability/event spine where possible |
| AHI-ADAS-50.7 | Lifecycle event coverage | Planned | High | Emits events for search, candidate, promotion, verification, rollback milestones |
| AHI-ADAS-50.8 | Hook/event tests | Planned | Critical | Hook failures safe-default to reject/archive or stop search |

**Gates:**

```text
pytest tests/unit/runtime/adaptive/agent_design_search/test_hooks.py
pytest tests/unit/runtime/adaptive/agent_design_search/test_events.py
```

**Non-goals:** UI and Tier-2 MAS agents.

---

### Phase AHI-ADAS-60 - Optional Tier-2 MAS agents

**Status:** Planned  
**Purpose:** Add optional agent-based design strategies that consume ADAS contracts but do not own the control plane.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-60.1 | Scaffold `agents/meta_agent_designer` | Planned | High | Created through existing scaffold; Tier-2 only; no runtime writes |
| AHI-ADAS-60.2 | `meta_agent_designer` contract | Planned | High | Outputs `AgentCandidateDraft`-compatible proposal |
| AHI-ADAS-60.3 | MAS prompt assets | Planned | Medium | Prompt instructs MAS to produce structured draft, rationale, measurable change, no direct writes |
| AHI-ADAS-60.4 | Scaffold `agents/candidate_critic` | Planned | Medium | Critiques candidate drafts and evaluation results; no promotion authority |
| AHI-ADAS-60.5 | Scaffold `agents/benchmark_runner` | Planned | Medium | Assists evaluation orchestration; does not bypass evaluator |
| AHI-ADAS-60.6 | MAS strategy adapter | Planned | High | Wraps MAS output as `AgentDesignStrategy` result |
| AHI-ADAS-60.7 | Safety prompt/eval tests | Planned | High | MAS refuses direct runtime edits and self-approval requests |
| AHI-ADAS-60.8 | Integration tests | Planned | Medium | MAS-generated draft flows through scaffold bridge and static gate |

**Gates:**

```text
pytest tests/unit/agents/meta_agent_designer/
pytest tests/integration/runtime/adaptive/agent_design_search/test_mas_strategy_flow.py
```

**Non-goals:** Production auto-promotion.

---

### Phase AHI-ADAS-70 - Shadow / canary / promotion bridge

**Status:** Planned  
**Purpose:** Safely make evaluated candidates routable through governed active-registration semantics.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-70.1 | `AgentCandidatePromotionRequest` | Planned | Critical | Captures candidate, evidence bundle, requested promotion modes, approver, rollback plan |
| AHI-ADAS-70.2 | Promotion gate adapter | Planned | Critical | Requires static gate, evaluation, archive, evidence bundle, approval by default |
| AHI-ADAS-70.3 | Shadow registration bridge | Planned | Critical | Candidate can be routed to shadow evaluation without production traffic |
| AHI-ADAS-70.4 | Canary allocation bridge | Planned | High | Tenant allowlist / traffic percentage; no default broad rollout |
| AHI-ADAS-70.5 | Human approval store/link | Planned | High | Approval record included in evidence bundle |
| AHI-ADAS-70.6 | Active registration mode A - registry pointer | Planned | Critical | Active registry entry points to candidate/package ref; rollback pointer preserved |
| AHI-ADAS-70.7 | Active registration mode E - tenant/application binding | Planned | Critical | Binding links objective/candidate to tenant/application scope |
| AHI-ADAS-70.8 | Optional modes B–D declarations | Planned | Medium | Contract version, routing profile, materialization require explicit mode selection |
| AHI-ADAS-70.9 | Rollback pointer model | Planned | Critical | Previous active registration snapshot restored on failure |
| AHI-ADAS-70.10 | Candidate verification target | Planned | High | Verification loop can compare promoted candidate vs baseline |
| AHI-ADAS-70.11 | Rollback integration | Planned | High | Verification failure triggers rollback request and archives result |
| AHI-ADAS-70.12 | Promotion tests | Planned | Critical | Cannot promote without approval/evidence/static gate/eval; rollback restores prior pointer |

**Gates:**

```text
pytest tests/unit/runtime/adaptive/agent_design_search/test_promotion_bridge.py
pytest tests/integration/runtime/adaptive/agent_design_search/test_candidate_promotion_flow.py
```

**Non-goals:** Full ADAS Lab UI.

---

### Phase AHI-ADAS-80 - Optional Tier-3 ADAS Lab application

**Status:** Planned  
**Purpose:** Provide operator-facing workflow without moving ADAS logic into Tier-3.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-80.1 | Scaffold `applications/adas_lab` | Planned | Medium | App wires ADAS services only; no duplicate archive/eval/governance logic |
| AHI-ADAS-80.2 | Objective creation API/screen | Planned | Medium | Creates `AgentDesignObjective` through Tier-1 service |
| AHI-ADAS-80.3 | Candidate archive browser | Planned | Medium | Tenant-scoped archive browsing; lineage view |
| AHI-ADAS-80.4 | Evidence bundle viewer | Planned | High | Displays sealed bundle: static/eval/cost/security/approval/promotion data |
| AHI-ADAS-80.5 | Approval/rejection workflow | Planned | High | Operator can approve/reject promotion requests; no self-approval by MAS |
| AHI-ADAS-80.6 | Search run dashboard | Planned | Medium | Shows costs, stop reasons, candidate ranking, gate failures |
| AHI-ADAS-80.7 | Rollback action wiring | Planned | High | Operator rollback uses Tier-1 rollback bridge |
| AHI-ADAS-80.8 | Tier boundary tests | Planned | Critical | App imports only public ADAS service/facade; no duplicate stores |

**Gates:**

```text
pytest tests/integration/applications/adas_lab/
```

**Non-goals:** Marketplace, billing, cross-tenant learning.

---

### Phase AHI-ADAS-90 - Enterprise hardening

**Status:** Planned  
**Purpose:** Production-readiness controls for regulated or enterprise deployments.

| ID | Deliverable | Status | Priority | Acceptance |
|----|-------------|--------|----------|------------|
| AHI-ADAS-90.1 | Retention policy enforcement | Planned | High | Archive respects `retention_policy`, risk class, tenant policy |
| AHI-ADAS-90.2 | PII/secret scanning | Planned | Critical | Scan on archive write, bundle seal, export; block/redact unsafe data |
| AHI-ADAS-90.3 | Legal hold support | Planned | High | Prevents compaction/purge while hold active |
| AHI-ADAS-90.4 | Tenant deletion / purge workflow | Planned | High | Tombstone + async purge with legal-hold exceptions |
| AHI-ADAS-90.5 | Cross-tenant isolation tests | Planned | Critical | No archive/evidence/search leakage across tenants |
| AHI-ADAS-90.6 | Budget exhaustion tests | Planned | High | Search stops deterministically and archives stop reason |
| AHI-ADAS-90.7 | Static gate bypass tests | Planned | Critical | Malicious candidates cannot pass via hooks/strategy |
| AHI-ADAS-90.8 | Evidence export | Planned | Medium | Tenant-scoped export of sealed bundles and lineage |
| AHI-ADAS-90.9 | Security review checklist | Planned | High | Checklist covers policy bypass, tool access, secrets, network, tenant isolation |
| AHI-ADAS-90.10 | Closeout report | Planned | High | ADAS enterprise readiness report under `build/adaptive_harness/agent_design_search` |

**Gates:**

```text
pytest tests/security/runtime/adaptive/agent_design_search/
pytest tests/integration/runtime/adaptive/agent_design_search/
```

---

## 7. Minimal viable implementation path

The safest MVP is:

```text
AHI-ADAS-10
  → contracts + archive only

AHI-ADAS-20
  → scaffold bridge + static gate

AHI-ADAS-30
  → offline evaluation + utility + evidence bundle

AHI-ADAS-40
  → deterministic search controller with RuleBasedVariantStrategy
```

Only after this path is stable should MAS agents be introduced.

Recommended MVP stop point:

```text
Generate 3 candidate drafts from a deterministic strategy,
materialize through scaffold sandbox,
run static gate,
run offline evaluation,
archive every result,
produce ranked report.
```

No shadow/canary/promotion in MVP unless explicitly enabled by later phase.

---

## 8. Implementation instructions for Cursor AI

When implementing a phase:

```text
1. Read the AHI hub read-scope block.
2. Read the ADAS architecture satellite sections relevant to the current phase.
3. Read only this plan section for the current phase.
4. Do not audit the whole repo unless needed for immediate dependencies.
5. Reuse existing AHI patterns before adding new abstractions.
6. Keep PRs phase-scoped.
7. Update this plan with status/evidence after each PR.
8. Stop when the phase acceptance criteria are met.
```

Mandatory constraints:

```text
Do not create a new top-level ADAS layer.
Do not create parallel governance/evaluation/trace/scaffold stacks.
Do not let MAS mutate production code.
Do not add Tier-3 ADAS Lab before Tier-1 ADAS is stable.
Do not promote candidates without evidence bundle and approval.
```

---

## 9. Closeout criteria

ADAS implementation can be considered enterprise-ready only when:

```text
1. AHI-ADAS-10 through AHI-ADAS-70 are complete.
2. Candidate archive is tenant-scoped and append-only.
3. Static gate blocks unsafe candidates.
4. Candidate evaluation compares against baseline.
5. Evidence bundles can be sealed and exported.
6. Promotion requires approval by default.
7. Rollback restores prior active registration.
8. Hooks cannot bypass safety gates.
9. MAS is optional and Tier-2 only.
10. ADAS Lab, if present, only wires Tier-1 APIs.
11. Enterprise hardening gates pass.
```

Phase **AHI-ADAS-90** is required before claiming production enterprise readiness.

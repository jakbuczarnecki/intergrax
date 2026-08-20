# ORCHESTRATION — Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** ORCHESTRATION
- **Tier(s):** Tier-3 application orchestration contracts · graph spec · profile wiring · Nexus plan seeding
- **audited_sha:** `a784966681782bc58412af290c2978c1d1f152a3`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 4 HIGH / 1 MEDIUM / 0 LOW
- **Operator decision:** all 5 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/ORCHESTRATION.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/ORCHESTRATION.md`
- **Scope in:**
  - Orchestration → Nexus → UER responsibility split
  - `ApplicationGraphSpec` validation and `graph_spec_to_plan` identity contract
  - `OrchestrationProfile` configuration semantics and fail-fast posture
  - delegation edge provenance in graph-to-plan conversion
  - duplicate `OrchestrationProfile` schema ownership
  - static graph topology validation (cycles)
  - planner/classifier kind fail-fast wiring
  - graph concurrency caps (`max_parallel_nodes`, `max_inflight_nodes`)
- **Scope out:**
  - remediation implementation
  - full Nexus/UER/Reasoning domain re-audit
  - product host operational qualification
  - queue transport internals
- **Prior audit reference(s):** Phase ORCH / ORCH-STRAT / ORCH-CONFIG / ORCH-5 / ORCH-6 closeout (plan **Done** rows); legacy audits `docs/audit_results/2026-06-18/ORCHESTRATION.md`, `docs/audit_results/2026-06-19/ORCHESTRATION.md`
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `—`

## Executive summary

**Verdict: FAIL.** Five accepted findings (4 HIGH, 1 MEDIUM) show graph-node identity can diverge between roster validation and plan construction (`contract_id` vs `agent_id`), execution-affecting orchestration settings silently fall back to defaults, delegation parent provenance can disagree with the delegation edge, two duplicate `OrchestrationProfile` schemas create silent drift risk, and cyclic `ApplicationGraphSpec` topology is accepted until runtime. Positive controls: Orchestration/Nexus/UER split holds; unknown `planner_kind`/`classifier_kind` fail fast; bounded graph concurrency caps exist; runtime cycle detection is controlled via `ExecutionGraphCycleError`. Prior ORCH-* **Done** delivery remains historical fact — this Protocol v2 layer audit records residual contract and architecture gaps beyond harness closeout.

## Verdict

**FAIL** — 0 CRITICAL / 4 HIGH / 1 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-ORCHESTRATION-01

**Graph node identity diverges between roster validation and plan construction**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / CONTRACT DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** ORCH-CONTRACT-INTEGRITY
- **Claim falsified:** Graph node identity resolves once into one canonical executable agent identity before plan construction; validation and `PlanStep` identity use the same resolved identity contract.
- **Observation:** `ApplicationGraphSpec.validate_against_roster()` accepts a `GraphNode` when either `node.agent_id` OR `node.contract_id` matches the manifest roster, but `application_graph_spec_to_nexus_plan()` emits `PlanStep.agent_id = node.agent_id` and does not resolve through `contract_id`. `AgentRegistry.get()` resolves by canonical registered agent/contract id key. A graph can therefore pass static roster validation yet produce a plan pointing at an unresolved `agent_id`.
- **Location:**
  - `intergrax/applications/contracts/graph_spec.py` — `validate_against_roster()`, `GraphNode.agent_id`, `GraphNode.contract_id` @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/applications/_shared/graph_spec_to_plan.py` — `PlanStep(agent_id=node.agent_id, ...)` @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/runtime/registry/agent_registry.py` — `get()`, `get_contract()` keyed by registered id @ `a784966681782bc58412af290c2978c1d1f152a3`
- **Reproduction:**
  1. `git show a784966681782bc58412af290c2978c1d1f152a3:intergrax/applications/contracts/graph_spec.py` — roster check accepts `contract_id` match without requiring `agent_id` in roster.
  2. `git show a784966681782bc58412af290c2978c1d1f152a3:intergrax/applications/_shared/graph_spec_to_plan.py` — plan step uses `node.agent_id` only.
  3. Contrast with `AgentRegistry.get(agent_id)` KeyError when id not registered.
- **Impact:** Valid-looking declarative graphs can seed plans that fail at execution or resolve the wrong agent identity.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ORCHESTRATION-02

**Execution-affecting orchestration configuration silently falls back to defaults**

- **Severity:** HIGH
- **Category:** CONTRACT DEFECT / FAIL-OPEN CONFIGURATION
- **Status at publication:** ACCEPTED
- **Remediation block:** ORCH-CONTRACT-INTEGRITY
- **Claim falsified:** Execution-affecting orchestration configuration is typed and fail-fast; unknown values do not silently change execution semantics.
- **Observation:** `OrchestrationProfile` stores `merge_strategy`, `multi_agent_order`, and `retry_policy_name` as loose strings. `_resolve_merge_strategy()` and `_resolve_multi_agent_order()` catch `ValueError` and return `MergeStrategy.CONCAT` and `MultiAgentOrder.REGISTRY` respectively. `build_nexus_loop_from_environment()` applies default `RetryPolicy(max_retries=3)` unless `retry_policy_name == "strict"` — any other unknown or misspelled value receives the default policy without error. Planner/classifier kinds fail fast via `_normalize_planner_kind` / `_normalize_classifier_kind`.
- **Location:**
  - `intergrax/applications/contracts/environment_profile/sub_profiles.py` — `OrchestrationProfile.merge_strategy`, `multi_agent_order`, `retry_policy_name` @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/applications/_shared/orchestration_wiring.py` — `_resolve_merge_strategy()`, `_resolve_multi_agent_order()` @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/applications/_shared/nexus_factory.py` — `retry_policy = RetryPolicy(max_retries=3)` default branch @ `a784966681782bc58412af290c2978c1d1f152a3`
- **Reproduction:**
  1. `git show a784966681782bc58412af290c2978c1d1f152a3:intergrax/applications/_shared/orchestration_wiring.py` — silent fallback in `_resolve_merge_strategy` / `_resolve_multi_agent_order`.
  2. `git show a784966681782bc58412af290c2978c1d1f152a3:intergrax/applications/_shared/nexus_factory.py` — only `"strict"` alters retry; other values silently use default.
  3. Contrast with `_normalize_planner_kind` raising `OrchestrationWiringError` on unknown kinds.
- **Impact:** Operators can configure orchestration posture that appears valid but executes materially different merge order, multi-agent sequencing, or retry behavior than requested.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ORCHESTRATION-03

**Delegation parent provenance derived from first dependency, not exact delegation edge**

- **Severity:** HIGH
- **Category:** IMPLEMENTATION DEFECT / GRAPH SEMANTICS
- **Status at publication:** ACCEPTED
- **Remediation block:** ORCH-DELEGATION-INTEGRITY
- **Claim falsified:** Delegation parent identity, contract, budget, and provenance derive from the exact `DelegationEdge`; unsupported multi-parent delegation fails static validation rather than ordering accident.
- **Observation:** Graph conversion stores one child delegation contract per target step (`child_delegations[target_step] = contract`) but later sets `parent_node_id` from `parent_candidates = depends_on.get(step_id, [])` and `parent_step_id = parent_candidates[0]`. The `depends_on` list accumulates both `DEPENDS_ON` and `DELEGATES_TO` edges in insertion order. A normal dependency listed before a delegation edge, or multiple delegation sources, can make delegation provenance disagree with the actual delegation edge.
- **Location:**
  - `intergrax/applications/_shared/graph_spec_to_plan.py` — `child_delegations`, `depends_on`, `parent_candidates[0]` @ `a784966681782bc58412af290c2978c1d1f152a3`
- **Reproduction:**
  1. `git show a784966681782bc58412af290c2978c1d1f152a3:intergrax/applications/_shared/graph_spec_to_plan.py` — both edge kinds append to `depends_on[target_step]`; delegation parent taken from first list element.
  2. Construct graph where `DEPENDS_ON` precedes `DELEGATES_TO` for same target — parent provenance follows first dependency, not delegator.
- **Impact:** Delegation metadata and parent linkage can be wrong while graph appears statically valid; multi-parent delegation is resolved by accident rather than explicit policy.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ORCHESTRATION-04

**Duplicate `OrchestrationProfile` schemas with silent drift risk**

- **Severity:** HIGH
- **Category:** ARCHITECTURE DEFECT / DUPLICATE CONTRACT
- **Status at publication:** ACCEPTED
- **Remediation block:** ORCH-CONTRACT-INTEGRITY
- **Claim falsified:** One canonical owner exists for orchestration configuration semantics; two same-purpose duplicate schemas are not acceptable without explicitly different responsibilities and a typed bridge.
- **Observation:** Two independent classes named `OrchestrationProfile` exist in `intergrax/applications/contracts/environment_profile/sub_profiles.py` (application environment contract) and `intergrax/contracts/host_profile_slices.py` (host-profile slice). They duplicate substantially the same fields and semantics. `profile_orchestration_resolver.py` types against the host-profile-slice model while `ApplicationEnvironmentProfile` provides the applications-contract model. Runtime works structurally because both expose compatible Pydantic APIs, but schema ownership is duplicated and can drift silently.
- **Location:**
  - `intergrax/applications/contracts/environment_profile/sub_profiles.py` — `OrchestrationProfile` @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/contracts/host_profile_slices.py` — `OrchestrationProfile` @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/runtime/adaptive/profile_orchestration_resolver.py` — imports host-profile `OrchestrationProfile` @ `a784966681782bc58412af290c2978c1d1f152a3`
- **Reproduction:**
  1. Compare both `OrchestrationProfile` class definitions at audited SHA — overlapping field sets (`merge_strategy`, `multi_agent_order`, `retry_policy_name`, parallelism caps, planner/classifier kinds).
  2. Note adaptive resolver imports `intergrax.contracts.host_profile_slices.OrchestrationProfile`, not applications contract.
- **Impact:** Field additions or semantic changes in one copy may not propagate; operators and adaptive tooling can disagree on orchestration configuration meaning.
- **Confidence:** CONFIRMED

### AUDIT-20260818-ORCHESTRATION-05

**Static `ApplicationGraphSpec` accepts cyclic topology until runtime**

- **Severity:** MEDIUM
- **Category:** VALIDATION GAP / FAIL-LATE
- **Status at publication:** ACCEPTED
- **Remediation block:** ORCH-CONTRACT-INTEGRITY
- **Claim falsified:** Static declarative graph configuration rejects cyclic topology before the host begins serving traffic / before task execution.
- **Observation:** `ApplicationGraphSpec` validates node uniqueness and endpoint existence via `validate_against_roster()` and `_unique_nodes()` but does not reject static dependency cycles. Cycles are discovered later by execution graph construction in `GraphExecutor` via `ExecutionGraphCycleError`. Runtime failure is controlled — not a catastrophic executor defect — but invalid static topology is accepted until task execution.
- **Location:**
  - `intergrax/applications/contracts/graph_spec.py` — `validate_against_roster()`, `_unique_nodes()` @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/runtime/nexus/execution/graph_executor.py` — `ExecutionGraphCycleError` handling @ `a784966681782bc58412af290c2978c1d1f152a3`
  - `intergrax/runtime/nexus/execution/execution_graph.py` — `ExecutionGraphCycleError` @ `a784966681782bc58412af290c2978c1d1f152a3`
- **Reproduction:**
  1. `git show a784966681782bc58412af290c2978c1d1f152a3:intergrax/applications/contracts/graph_spec.py` — no cycle detection in validators.
  2. `tests/unit/runtime/execution/test_execution_graph_cycle.py` — cycle raised at execution graph build, not graph-spec validation.
- **Impact:** Invalid application topology passes static configuration checks and fails only when a task executes.
- **Confidence:** CONFIRMED

## Falsification log (negative results)

1. **Orchestration / Nexus / UER responsibility split invalid** — not falsified; Orchestration owns structure/configuration, Nexus executes tasks through that structure, UER owns per-node execution behavior.
2. **Unknown planner_kind/classifier_kind accepted silently** — not falsified; `_normalize_planner_kind` / `_normalize_classifier_kind` raise `OrchestrationWiringError` on unknown values.
3. **No bounded graph concurrency** — not falsified; `max_parallel_nodes` and `max_inflight_nodes` are wired through orchestration runtime settings to `GraphExecutor`.
4. **Execution graph cycles cause uncontrolled executor failure** — not falsified; `ExecutionGraphCycleError` is caught and handled in `GraphExecutor` (controlled runtime failure).
5. **Prior ORCH-* delivery never occurred** — not falsified; historical **Done** rows remain valid delivery facts; this audit records residual gaps beyond that closeout.

## Prior-audit comparison

First canonical Protocol v2 `ORCHESTRATION` layer snapshot at `a784966681782bc58412af290c2978c1d1f152a3`. Supplements — does not rewrite — Phase ORCH / ORCH-STRAT / ORCH-CONFIG / ORCH-5 / ORCH-6 harness closeout and legacy 2026-06-18/19 orchestration audits. Discoveries are contract/architecture integrity gaps not closed by prior **Done** registers.

## Provider / backend abstraction

`NOT APPLICABLE — ORCHESTRATION scope is collaboration structure, graph spec, and profile wiring; no material external provider/backend substitution boundary in this layer.`

## Positive controls

1. **Orchestration → Nexus → UER split** — documented ownership: Orchestration defines structure; Nexus executes; UER defines per-node runtime behavior @ audited SHA.
2. **Fail-fast planner/classifier kinds** — `orchestration_wiring.py` `_normalize_planner_kind` / `_normalize_classifier_kind` reject unknown kinds @ audited SHA.
3. **Bounded graph concurrency** — `max_parallel_nodes` / `max_inflight_nodes` on `OrchestrationProfile` wired to `GraphExecutor` @ audited SHA.
4. **Controlled runtime cycle handling** — `ExecutionGraphCycleError` raised by execution graph build and handled in `GraphExecutor` @ audited SHA.

**FAIL qualification:** verdict means contract/architecture integrity gaps remain — **not** that the Orchestration control-plane model or prior harness delivery is invalid.

## Root-cause remediation grouping

Planning only — **audit persistence does NOT implement remediation.**

### ORCH-CONTRACT-INTEGRITY — graph identity, typed config, profile ownership, static validation

**Findings:** 01, 02, 04, 05

Canonical graph-node executable identity; typed fail-fast execution-affecting orchestration configuration; single canonical `OrchestrationProfile` ownership; static `ApplicationGraphSpec` cycle rejection before runtime traffic.

### ORCH-DELEGATION-INTEGRITY — exact delegation-edge provenance

**Findings:** 03

Exact delegation-edge provenance; deterministic parent identity; explicit policy for multiple delegation parents; deterministic tests.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `a784966681782bc58412af290c2978c1d1f152a3`; current `development` HEAD was not re-audited.
- ORCH-01 does not prescribe alias compatibility — target invariant is single resolved executable identity unless architecture explicitly chooses otherwise.
- ORCH-05 does not claim uncontrolled executor failure on cycles — only fail-late static validation.
- Tests (`test_orchestration_wiring.py`, `test_graph_spec_to_plan.py`) are supporting evidence, not standalone proof.
- Remediation not performed in this task.

## Open questions / blocked items

- Canonical owner choice for consolidated `OrchestrationProfile` left to implementation (applications contract vs host slice) — must be explicit, not shimmed.
- Multi-parent delegation policy (reject vs explicit model) left to ORCH-DELEGATION-INTEGRITY implementation.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 5 (`AUDIT-20260818-ORCHESTRATION-01` … `AUDIT-20260818-ORCHESTRATION-05`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

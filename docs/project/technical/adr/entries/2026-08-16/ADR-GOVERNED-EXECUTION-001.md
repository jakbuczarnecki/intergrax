# ADR-GOVERNED-EXECUTION-001: Governance Evaluation Points and Enforcement Ownership

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-08-16 |
| **Deciders** | Platform architecture (Governed Execution G1A) |
| **Related** | [`GOVERNED_EXECUTION.md`](../../../architecture/GOVERNED_EXECUTION.md) · [ADR-PLATFORM-PLUGIN-001](../2026-08-14/ADR-PLATFORM-PLUGIN-001.md) · [ADR-POLICY-SIDE-EFFECT-001](../2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md) · [ADR-RUNTIME-POLICY-BUNDLE-001](../2026-07-20/ADR-RUNTIME-POLICY-BUNDLE-001.md) · [ADR-GOVERNED-CONTINUATION-001](../2026-07-20/ADR-GOVERNED-CONTINUATION-001.md) · [RELIABILITY_FAILURE_AND_HITL.md](../../../architecture/RELIABILITY_FAILURE_AND_HITL.md) |

## Context

Governed Execution (G0) froze the public capability name, responsibility boundary (*applications define the rules; Intergrax enforces the execution boundaries*), and the governance-plane mental model. Multiple enforcement families already exist in the runtime with different contracts, decision vocabularies, and maturity.

A read-only audit (G1A) verified that these families are **not** duplicates:

| Family | Primary owner | Role |
| ------ | ------------- | ---- |
| `ToolAccessPolicy` | `intergrax.runtime.nexus.tools.tool_access_policy` | Filters tool plans / capability access before invocation |
| `ToolScopePolicy` | `intergrax.runtime.tools.scope_policy` | Invocation-time authorization by agent / tool |
| `DeclarativePolicyEnforcer` | `intergrax.runtime.policy.declarative_enforcer` | Typed declarative policy at tool invocation |
| `RuntimePolicyEngine` | `intergrax.runtime.policy.runtime_policy_engine` | Live agent decision, interrupt, pre-LLM, pre-output, critic, meaningful side effects |
| `PolicyEngine` | `intergrax.runtime.policy.policy_engine` | Facade over live runtime + replay-oriented evaluators |
| `ExecutionGuard` / `ExecutionPolicyEngine` | `intergrax.runtime.governance` / `intergrax.runtime.replay.policy` | Post-run replay, metrics, regression governance |

`PolicyEngine` docstrings describe a "unified" facade but it fronts only `RuntimePolicyEngine` and optional `ExecutionPolicyEngine`. It does **not** own declarative tool policy, tool access/scope authorization, HITL coordination, or evidence persistence.

Before runtime refactoring (G1B), the platform must freeze: what a Governance Evaluation Point is, who owns each enforcement responsibility, how authorization differs from policy and post-run governance, shared vs point-specific semantics, failure posture, HITL rules, provenance/evidence flow, and why there is no universal `GovernanceEngine`.

**Rejected alternatives:**

- **Single universal GovernanceEngine** - collapses defense-in-depth, mixes authorization with post-run BLOCK semantics, and contradicts existing specialized owners.
- **One runtime enum for all decisions** - replay `ALLOW`/`WARN`/`BLOCK` is not equivalent to live pre-execution `DENY`; timing and follow-up differ.
- **Merge ToolAccessPolicy and ToolScopePolicy** - plan-time filtering and invocation-time authorization are deliberately separate.

## Decision

### 1. Governance Evaluation Point (definition)

A **Governance Evaluation Point** is a named execution boundary at which Intergrax evaluates configured governance state before, during, or after a meaningful execution operation and produces an explicit governance outcome according to that boundary's contract.

Each evaluation point **must** have (as architecture targets; maturity varies):

| Property | Requirement |
| -------- | ----------- |
| Owner | A named module / domain pair with clear responsibility |
| Request / context contract | Typed input for critical paths (see section 7) |
| Decision semantics | Point-specific vocabulary mapped to shared semantics (see section 5) |
| Enforcement behavior | Explicit block, allow, HITL pause, or observe-only |
| Failure posture | Explicit fail-open vs fail-closed (see section 6) |
| Provenance | Policy / rule identity where applicable |
| Evidence | Decision facts feed observability / evidence owners |
| Non-bypass | Agent or model reasoning cannot skip the boundary when wired |

A Governance Evaluation Point is **not** one method, one class, one enum, or one middleware stack.

### 2. Governed Execution ownership model

**Governed Execution** is **one** platform capability composed from **multiple** specialized enforcement owners on a single governance plane:

| Category | Purpose | Examples (current) |
| -------- | ------- | ------------------ |
| **A. Capability / plan authorization** | Whether an actor / capability may reach an execution surface | `ToolAccessPolicy`, `ToolScopePolicy` (where used for access) |
| **B. Live policy evaluation** | Runtime decisions, interrupts, model / output / side-effect evaluation | `RuntimePolicyEngine`, `PolicyEngine` facade (live paths only) |
| **C. Declarative execution policy** | Policy rules bound to a concrete execution point | `DeclarativePolicyEnforcer` at tool invocation |
| **D. Human approval / governed continuation** | Pause, human decision, scoped resume | Canonical Nexus HITL + governed continuation ([ADR-GOVERNED-CONTINUATION-001](../2026-07-20/ADR-GOVERNED-CONTINUATION-001.md), [ADR-PLATFORM-PLUGIN-001](../2026-08-14/ADR-PLATFORM-PLUGIN-001.md)) |
| **E. Post-run governance** | Evaluate completed execution, regression, historical metrics | `GovernanceService`, `ExecutionGuard`, `ExecutionPolicyEngine` |
| **F. Evidence / provenance** | Durable, reviewable representation of decisions and execution | Observability spine, execution evidence contracts ([OBSERVABILITY.md](../../../architecture/OBSERVABILITY.md)) |

No new universal owner replaces these categories. Future work unifies through **shared contracts, evaluation-point semantics, and composition** - not a growing god object.

### 3. Authorization vs policy enforcement vs post-run governance

These concerns **compose sequentially** but are **not interchangeable**:

| Concern | Question | Examples |
| ------- | -------- | -------- |
| **Authorization** | May this principal / agent / capability reach or use this execution surface? | `ToolScopePolicy`, `ToolAccessPolicy` allowlists |
| **Policy enforcement** | Given this request and context, may this particular execution proceed, be changed, escalated, or require human approval? | `RuntimePolicyEngine`, `DeclarativePolicyEnforcer`, meaningful side-effect gate |
| **Post-run governance** | Was completed execution acceptable per replay / regression / governance rules, and what follow-up should occur? | `ExecutionGuard`, `ExecutionPolicyEngine` |

**Frozen rules:**

- Authorization **ALLOW** does **not** imply policy **ALLOW**.
- Policy **ALLOW** does **not** imply every downstream authorization or control passes.
- Post-run **BLOCK** does **not** retroactively mean a pre-action **DENY** (different timing and remediation).

### 4. Evaluation-point classes and matrix

Conceptual classes only - **no runtime enum in G1A/G1B architecture freeze**.

`EXECUTION_INTERRUPT` is **not** a separate top-level class: it shares the live runtime policy owner and `PolicyAction` vocabulary with agent-decision control, but uses the typed `ExecutionInterrupt` request via `evaluate_interrupt`. Document it as a distinct **sub-boundary** under **AGENT_DECISION** / runtime control.

| Class | Owner (today) | Input / context family | Decision vocabulary | Enforcement location | Failure posture | Maturity |
| ----- | ------------- | ------------------------ | ------------------- | -------------------- | --------------- | -------- |
| **AGENT_DECISION** | `RuntimePolicyEngine` via `PolicyEngine.evaluate_decision` | `AgentDecision` + optional opaque `context` dict | `PolicyAction`: ALLOW, DENY, MODIFY, ESCALATE, REQUIRE_HUMAN | UAEP / Nexus decision path | Default allow for unmatched rules; mandatory REQUIRE_HUMAN on configured critical paths | Implemented slice; **weak typed context (G1B)** |
| **AGENT_DECISION** (interrupt sub-boundary) | `RuntimePolicyEngine.evaluate_interrupt` | `ExecutionInterrupt` | Same `PolicyAction` | Interrupt handler path | Blocking interrupt → REQUIRE_HUMAN | Implemented slice |
| **PRE_MODEL** | `RuntimePolicyEngine.evaluate_pre_llm` | tenant / agent / message_count + opaque `context` | `PolicyAction` | Pre-LLM bridge | Empty context → DENY; else largely default allow | Implemented slice; **weak context (G1B)** |
| **TOOL_PLAN_OR_ACCESS** | `ToolAccessPolicy` | `ToolInvocationPlan`, `allowed_tools` | Plan filter (no PolicyAction enum) | Before `ToolRuntime.invoke` | Fail closed when allowlist configured and tool not listed; `allowed_tools is None` → no filter | Implemented |
| **TOOL_INVOCATION** (authorization) | `ToolScopePolicy` | `agent_id`, `tool_id` | Boolean allow / deny → exception | `RuntimeToolInvoker` step 0 | Fail closed when policy configured and not allowed | Implemented |
| **TOOL_INVOCATION** (declarative policy) | `DeclarativePolicyEnforcer` | `PolicyEvaluationContext` | ALLOW, DENY, REQUIRE_HITL + enforcement mode | `RuntimeToolInvoker` before handler | ENFORCE + indeterminate security outcome → block; AUDIT_ONLY may record would-deny | **Reference pattern** - strongest typed contract |
| **MEANINGFUL_SIDE_EFFECT** | `RuntimePolicyEngine.evaluate_meaningful_side_effect` | `MeaningfulSideEffectRequest` | `PolicyAction` | Side-effect authorization composition | **Fail closed** ([ADR-POLICY-SIDE-EFFECT-001](../2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md)) | Implemented mechanism; extensible `context` fields **G1B** |
| **PRE_OUTPUT** | `RuntimePolicyEngine.evaluate_pre_output` | tenant / agent / output_chars + opaque `context` | `PolicyAction` | Pre-output bridge | Empty output → DENY; else default allow | Implemented slice |
| **POST_RUN** | `ExecutionGuard` → `ExecutionPolicyEngine` | `ExecutionMetrics`, `RegressionSignals`, optional `RunDiff` | ALLOW, WARN, BLOCK (`PolicyDecisionType`) | After run completion | Advisory WARN default from config; not pre-execution DENY | Implemented mechanisms |

Critic verdict evaluation (`evaluate_critic_verdict`) maps to **AGENT_DECISION** / runtime control with opaque `critic_governance` context - **G1B typed-context candidate**.

### 5. Shared decision semantics (conceptual only)

Multiple point-specific vocabularies remain valid. Shared **semantic categories** (not one runtime type):

| Shared semantic | Meaning |
| --------------- | ------- |
| **PROCEED** | Execution may continue under constraints |
| **BLOCK** | Governed step must not proceed |
| **REQUIRE_HUMAN** | Canonical HITL before governed continuation |
| **ADJUST_OR_REWRITE** | Replace or adjust proposed content / decision |
| **ESCALATE_OR_REVIEW** | Route to higher review or escalation path |
| **OBSERVE_ONLY** | Record outcome without mandatory block |

**Mappings (illustrative):**

| Source | Maps to |
| ------ | ------- |
| Runtime `ALLOW` | PROCEED |
| Runtime `DENY` | BLOCK |
| Runtime `REQUIRE_HUMAN` | REQUIRE_HUMAN |
| Runtime `MODIFY` | ADJUST_OR_REWRITE |
| Runtime `ESCALATE` | ESCALATE_OR_REVIEW |
| Declarative `ALLOW` / `DENY` / `REQUIRE_HITL` | PROCEED / BLOCK / REQUIRE_HUMAN |
| Replay `ALLOW` | PROCEED / ACCEPT (post-run) |
| Replay `WARN` | OBSERVE_OR_REVIEW |
| Replay `BLOCK` | Post-run follow-up governance - **not** equivalent to pre-execution BLOCK |

Shared semantics do **not** require one enum, one DTO, or one evaluator.

### 6. Failure posture (security-critical)

Failure posture is **evaluation-point-specific**. Platform rules:

**A. Meaningful external side effects - FAIL CLOSED**

Missing identity, missing principal where required, indeterminate rule, invalid decision, unsupported security-sensitive outcome, or no matching authorization rule must **not** silently execute. Current `evaluate_meaningful_side_effect` behavior is canonical.

**B. Authorization boundaries - FAIL CLOSED when restricted**

When access is explicitly configured/restricted, denial is mandatory. Distinguish:

- *No policy configured* (e.g. `allowed_tools is None`, no scope policy) - not automatic denial unless existing semantics define it.
- *Configured policy cannot determine authorization* - treat as denial on restricted paths.

**C. Declarative ENFORCE mode**

Unknown handler or evaluation failure at a security-sensitive enforced rule must not silently allow execution when canonical policy requires fail-closed behavior.

**D. Declarative AUDIT_ONLY mode**

May record would-block / would-deny without blocking, by explicit configuration.

**E. Advisory policy**

Advisory outcomes (`EnforcementLevel.ADVISORY`) do not equal mandatory enforcement.

**F. Pre-model / pre-output**

Do not claim universal fail-closed behavior without point-specific evidence. Security-sensitive boundaries that would cause unauthorized external effects if indeterminate should default to fail-closed.

### 7. Typed context invariant

Every **critical** live Governance Evaluation Point must have an explicit typed context / request contract owned by that evaluation point or its domain.

**Forbidden as target architecture for critical enforcement:**

- `dict[str, Any]` / `Dict[str, Any]` interpreted directly by security-sensitive logic
- Unvalidated string-key semantic bags (e.g. `require_human_on_critical`, `critic_governance`, `phase` in opaque runtime context)

**Allowed temporarily:**

- Opaque dicts at generic plugin ingestion boundaries **if** domain-owned validation occurs before security-sensitive enforcement.

Extensible policy conditions remain plugin/domain-specific via the existing Platform Plugin / `PolicyRuleHandler` architecture. **No second policy plugin system.**

`RuntimePolicyBundle.domain_fragments` may continue to exist; **critical** Governed Execution enforcement must **not** depend on unvalidated opaque fragments. Extension payloads exist only behind domain-owned validation/contracts.

`MeaningfulSideEffectRequest.context` / `correlation` remain extensible; security-sensitive decisions must not depend on opaque unvalidated payload semantics without domain validation (**G1B**).

### 8. HITL contract

Frozen HITL rules (details in [RELIABILITY_FAILURE_AND_HITL.md](../../../architecture/RELIABILITY_FAILURE_AND_HITL.md) and [ADR-PLATFORM-PLUGIN-001](../2026-08-14/ADR-PLATFORM-PLUGIN-001.md)):

- **One** canonical HITL system - no second pause coordinator.
- `REQUIRE_HUMAN` (runtime) and `REQUIRE_HITL` (declarative) map to the same conceptual **REQUIRE_HUMAN** semantic; point-specific names may persist until G1B hardening.
- Approval must be **scoped** (task, run, step, tool, matched rules, provenance digest where wired - see declarative grant satisfaction).
- Approval does **not** generically bypass **DENY**.
- Policy must be **re-evaluated** after resume where canonical HITL architecture requires it.
- HITL is **not** `TOOL_ERROR` or a generic retry substitute.

### 9. Policy identity, provenance, and evidence

For security-relevant governance decisions, where applicable, attribute:

| Field | Notes |
| ----- | ----- |
| Evaluation point | Conceptual class / boundary name |
| Policy / rule identity | e.g. `policy_rule_id`, `matched_rule_ids` |
| Bundle identity | `policy_bundle_id`, version, digest when attested ([ADR-RUNTIME-POLICY-BUNDLE-001](../2026-07-20/ADR-RUNTIME-POLICY-BUNDLE-001.md)) |
| Provenance digest | e.g. declarative `rules_digest_sha256` |
| Execution identity | run / task / step / invocation as appropriate |
| Principal / tenant | When required by the boundary |
| Enforcement outcome | Point-specific decision + enforcement mode |
| Reason(s) | Human- and machine-auditable |

**Governed Execution produces decision facts; Observability / Evidence owns durable reviewable representation.** Do not collapse evaluation and evidence into one capability. No universal evidence DTO in G1A.

### 10. PolicyEngine boundary (explicit)

**`PolicyEngine` is NOT the universal Governed Execution engine.**

It is a **facade** over:

- `RuntimePolicyEngine` - live evaluation
- `ExecutionPolicyEngine` (optional) - replay / post-run oriented evaluation exposed through the same entry type

It **MUST NOT** silently absorb:

- `ToolAccessPolicy`, `ToolScopePolicy`
- `DeclarativePolicyEnforcer`
- HITL coordinator
- Evidence persistence
- Every future evaluation point

Docstring "unified" terminology is **documentation debt** - clarify in G1B without renaming the public capability or turning the facade into `GovernanceEngine`.

### 11. Reference enforcement pattern

**Reference pattern (not universal topology):** `DeclarativePolicyEnforcer` + `RuntimeToolInvoker` (`intergrax/runtime/nexus/tools/invoker.py`).

Demonstrates the well-formed live boundary chain:

```text
typed PolicyEvaluationContext
  → rule evaluation + deterministic aggregation (DENY > REQUIRE_HITL > ALLOW)
  → explicit PolicyEnforcementMode (ENFORCE / AUDIT_ONLY)
  → provenance digest
  → trace / diagnostic evidence
  → block before tool handler side effects
  → scoped HITL approval grant re-evaluation
```

Other evaluation points should converge toward this **contract quality**, not this **implementation topology**. Do not force every boundary through the declarative enforcer.

## Consequences

### Positive

- G1B implementers have a frozen contract without another broad discovery pass.
- Defense-in-depth preserved; authorization, live policy, and post-run governance stay distinct.
- Shared semantics enable correlation without forcing premature type unification.
- Declarative tool path provides a concrete quality bar for typed contexts and provenance.

### Negative

- Multiple decision vocabularies persist until G1B hardening.
- `PolicyEngine` "unified" naming remains confusing until terminology cleanup.
- Runtime policy paths still carry opaque context debt.

## Compliance

- Tier boundaries preserved - no new Tier-0 universal engine.
- G0 naming and responsibility boundary unchanged.
- Platform Plugin architecture reused; no second plugin framework.
- Linked architecture owner updated: [`GOVERNED_EXECUTION.md`](../../../architecture/GOVERNED_EXECUTION.md).

## G1B implementation backlog (frozen scope)

G1B performs **contract hardening** only - no architectural rediscovery. Minimum candidates:

| ID | Concern | G1B action |
| -- | ------- | ---------- |
| **G1B-A** | `RuntimePolicyEngine` opaque contexts | Replace `Dict[str, Any]` contexts used by critical live evaluation (`evaluate_decision`, `evaluate_pre_llm`, `evaluate_pre_output`, `evaluate_critic_verdict`) with typed contracts per evaluation point. |
| **G1B-B** | Runtime policy rule representation | Eliminate `List[Dict[str, Any]]` as the target critical-path rule contract for live engine rules (e.g. meaningful side effect rules). |
| **G1B-C** | `PolicyDecision.audit_payload` | Decide whether `audit_payload` remains extension-only or migrates toward typed evidence metadata for critical paths. |
| **G1B-D** | `RuntimePolicyBundle.domain_fragments` | Ensure critical governance decisions do not read unvalidated opaque fragments; domain handlers validate before enforcement. |
| **G1B-E** | `MeaningfulSideEffectRequest` extensible fields | Domain-specific `context` / `correlation` validated by typed domain contracts before security-sensitive evaluation. |
| **G1B-F** | `PolicyEngine` facade terminology | Clarify facade ownership in docs / types; do not expand into `GovernanceEngine`. |
| **G1B-G** | Evaluation-point provenance | Ensure critical live decisions stamp sufficient identity (rule id, bundle refs, digest) for evidence correlation on all wired paths. |

## Implementation notes

- **G1A:** documentation only - this ADR and [`GOVERNED_EXECUTION.md`](../../../architecture/GOVERNED_EXECUTION.md) section Governance Evaluation Points and ownership.
- **Verification:** `python scripts/maintenance/check_harness_adr.py`

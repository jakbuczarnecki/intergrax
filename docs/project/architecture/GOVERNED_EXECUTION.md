<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Governed Execution

**Governance & Policy Enforcement** — reusable platform mechanisms that enforce configured execution boundaries around agent and model behavior.

Agent and model behavior may propose decisions and actions. The product or application owns the business rule — who may do what, which actions require approval, and what outcomes are acceptable. Intergrax supplies reusable mechanisms that carry identity and context, evaluate configured policy, enforce decisions, pause for human approval when required, and record governance evidence.

**Applications define the rules; Intergrax enforces the execution boundaries.**

> [!NOTE]
> Intergrax is source-available and in active R&D. This document describes the **Governed Execution** platform capability and its conceptual governance plane. It is **not** a production-readiness, enterprise-readiness, security-certification, or complete platform-wide enforcement claim.

Primary audience: Principal / Staff engineers, architects, CTOs, security and governance evaluators, and builders integrating an application with Intergrax.

---

## At a glance

| Concern | What Intergrax provides |
| -------- | --------------------- |
| **Policy definition** | Built-in, application-configured, and plugin-extensible policy rules bound to evaluation contexts |
| **Policy enforcement** | Evaluation at configured boundaries before or after meaningful execution steps |
| **Approval / HITL** | One canonical human-in-the-loop path when policy requires human decision |
| **Tool and action boundaries** | Controlled tool invocation and meaningful side-effect authorization on demonstrated paths |
| **Evidence / provenance** | Governance decisions correlated with execution evidence where mechanisms are wired |
| **Extension** | Policy handlers through the existing platform plugin / policy architecture — not a second plugin framework |

---

## Responsibility boundary

### Application / organization owns

- Business rules and what permission means in product terms
- Approval requirements and organizational risk policy
- Required identity, tenant, and product context
- Acceptance criteria for product outcomes

### Intergrax owns reusable mechanisms for

- Carrying identity and execution context into policy evaluation
- Evaluating configured policy at supported evaluation points
- Enforcing policy decisions (allow, deny, require human, and other supported outcomes)
- Preventing unauthorized execution on wired paths
- Pausing for canonical HITL and scoped governed continuation
- Recording governance evidence where mechanisms are connected

### Agent / model

- Proposes reasoning, decisions, and actions within supplied context
- Does **not** grant business permission or bypass configured boundaries

Intergrax does **not** decide business permissions on behalf of the application.

---

## Governance plane

Conceptual platform model — not a single runtime class or universal wrapper:

```text
                         GOVERNED EXECUTION
                                |
                   +------------+-------------+
                   |                          |
            POLICY DEFINITION          POLICY ENFORCEMENT
                   |                          |
          +--------+---------+                |
          |        |         |                |
       built-in   app     plugin              |
       policies policies  policies            |
          |        |         |                |
          +--------+---------+                |
                   |                          |
                   +------------+-------------+
                                |
                         evaluation point
                                |
             input / model / decision / tool /
                 output / side effect / post-run
                                |
                         policy decision
                                |
          ALLOW / DENY / MODIFY / ESCALATE /
                        REQUIRE_HUMAN
                                |
                    canonical HITL when needed
                                |
                     governed continuation
                                |
                            evidence
```

This is the **governance plane** mental model. Live enforcement, HITL, evidence, and post-run governance remain specialized owners; they are not collapsed into one implementation component.

---

## Policy outcomes

Existing runtime vocabulary (`intergrax.contracts.runtime_policy.PolicyAction`):

| Outcome | Meaning |
| -------- | -------- |
| **ALLOW** | Proceed under configured constraints |
| **DENY** | Block the governed step |
| **MODIFY** | Replace or adjust the proposed decision where supported |
| **ESCALATE** | Route to a higher enforcement or review path where wired |
| **REQUIRE_HUMAN** | Pause for canonical HITL before governed continuation |

Each decision may carry **advisory** or **mandatory** enforcement level. Mandatory enforcement blocks or redirects execution on wired paths; advisory outcomes may surface warnings without stopping execution, depending on host configuration.

**MODIFY**, **ESCALATE**, and uniform mandatory enforcement are **not** claimed at every evaluation boundary. Support is evaluation-point-specific.

---

## Evaluation boundaries

Conceptual boundary classes in the governance plane model:

| Boundary class | Role |
| -------------- | ---- |
| **Model / LLM boundary** | Policy around model invocation and guardrail composition |
| **Agent decision boundary** | Policy on agent-proposed decisions before execution |
| **Tool invocation** | Declarative and runtime policy before tool handlers run |
| **Meaningful external side effect** | Authorization for effects that leave the bounded runtime |
| **Output** | Pre-output policy bridges where wired |
| **Replay / post-run governance** | Post-run evaluation, metrics, and guard mechanisms |

These classes describe **where policy may apply** in the platform model. **Current implementation coverage varies by boundary.** Do not infer a uniform evaluation-point API or complete platform-wide coverage from this list.

---

## Governance Evaluation Points and ownership

Frozen architecture (G1A): [ADR-GOVERNED-EXECUTION-001](../technical/adr/entries/2026-08-16/ADR-GOVERNED-EXECUTION-001.md).

A **Governance Evaluation Point** is a named execution boundary at which Intergrax evaluates configured governance state before, during, or after a meaningful execution operation and produces an explicit governance outcome according to that boundary's contract. It is **not** one class, method, enum, or middleware stack.

**One governance plane, multiple enforcement owners.** Governed Execution composes specialized owners — authorization (`ToolAccessPolicy`, `ToolScopePolicy`), live policy (`RuntimePolicyEngine`, `PolicyEngine` facade for live + replay evaluators), declarative tool policy (`DeclarativePolicyEnforcer`), canonical HITL, post-run governance (`GovernanceService`, `ExecutionGuard`), and evidence/observability — without a universal `GovernanceEngine`.

| Concern | Question |
| ------- | -------- |
| **Authorization** | May this principal / capability reach this execution surface? |
| **Policy enforcement** | Given this request, may this execution proceed, change, escalate, or require human approval? |
| **Post-run governance** | Was completed execution acceptable, and what follow-up is required? |

These compose sequentially; they are **not** interchangeable. Authorization ALLOW does not imply policy ALLOW; post-run BLOCK is not retroactive pre-execution DENY.

**Typed context rule:** critical live evaluation points must move toward explicit typed request/context contracts. Opaque `dict[str, Any]` semantic bags are not the target architecture for security-sensitive enforcement. Plugin/domain extension payloads may exist at ingestion boundaries only behind domain-owned validation ([Platform Plugins](PLATFORM_PLUGINS.md)).

**Failure posture:** security-sensitive indeterminate outcomes at meaningful external side effects and explicitly restricted authorization paths **fail closed**. Declarative `AUDIT_ONLY` may record would-deny without blocking. Other boundaries are evaluation-point-specific (see ADR).

**Reference pattern (not universal topology):** `DeclarativePolicyEnforcer` at `RuntimeToolInvoker` — typed context, deterministic precedence, provenance, enforcement mode, block before handler, scoped HITL. Other boundaries should match this contract quality where critical, not necessarily this implementation path.

**PolicyEngine** is a facade over live `RuntimePolicyEngine` and optional replay `ExecutionPolicyEngine` — **not** the whole of Governed Execution. It does not own tool access/scope, declarative enforcer, HITL, or evidence.

Contract hardening (**G1B**) — **implemented core** on owned live paths (not platform-wide coverage):

- **G1B-1:** typed live policy evaluation contexts for agent decision, pre-model, and critic governance; unused pre-output semantic context removed. Security-sensitive live evaluation on these owned paths no longer depends on opaque `dict` bags.
- **G1B-2:** typed meaningful-side-effect runtime rules (`MeaningfulSideEffectPolicyRule`, explicit `rule_id`, existing `PolicyAction`); `RuntimePolicyEngine` does not parse dynamic type/decision/id strings; fail-closed semantics preserved.
- **G1B-3:** hardened `PolicyDecision` — immutable, extra fields forbidden, explicit canonical provenance; bundle provenance either absent or complete; sha256 digest structurally validated; `audit_payload` remains diagnostic/non-authoritative. `EvaluatedPolicyDecision` remains the bundle-backed typed evidence contract; no duplicate evidence framework.

Not closed by this core: `RuntimePolicyBundle.domain_fragments` hardening, `MeaningfulSideEffectRequest` context/correlation hardening, remaining facade terminology, universal rule catalog, universal evaluation-point coverage, `decision_id` on every policy producer, or durable evidence persistence.

---

## Human-in-the-loop

Intergrax has **one canonical HITL system**. `REQUIRE_HUMAN` connects conceptually to:

```text
policy → pause → human decision → scoped continuation / resume → evidence
```

Canonical owners and invariants:

- Human approval does **not** generically bypass **DENY**
- Authorization and continuation must remain **scoped** to the governed request
- HITL is **not** a generic tool failure or retry substitute
- Do **not** introduce a second HITL runtime

Deeper specification: [RELIABILITY_FAILURE_AND_HITL.md](RELIABILITY_FAILURE_AND_HITL.md) (failure, retry, HITL, governed continuation). Platform plugin admission for policy extensions: [ADR-PLATFORM-PLUGIN-001](../technical/adr/entries/2026-08-14/ADR-PLATFORM-PLUGIN-001.md) (policy handler surface; full third-party production qualification **not** claimed).

---

## Policy extensibility

Policy handlers participate through the **existing** platform plugin and policy architecture:

- Reuse [Platform Plugins](PLATFORM_PLUGINS.md) coordination and domain-owned contracts
- **No second plugin framework** for governance
- Plugin admission, allowlisting, and provenance exist in meaningful slices
- Full production qualification of third-party policy plugins is **not** established

Maintainer roadmap context (not public proof): [PLATFORM_PLUGIN_ENTERPRISE_ROADMAP.md](../maintainers/plans/PLATFORM_PLUGIN_ENTERPRISE_ROADMAP.md).

---

## Policy Catalog

Frozen architecture (G2A): [ADR-GOVERNED-EXECUTION-002](../technical/adr/entries/2026-08-17/ADR-GOVERNED-EXECUTION-002.md).

The **Policy Catalog** is the canonical registry of policy **definitions** available for application selection. It answers *what governance capabilities can this application select?* It is **not** implemented as a runtime catalog in G2A — this section freezes identity and ownership only.

| Question | Concept | Identity |
| -------- | ------- | -------- |
| What can I choose? | Policy Catalog → Policy Definition | `policy_id` + definition version |
| What did this application configure? | Configured rule instance | `rule_id` |
| What policy state is active? | Runtime / immutable bundle | bundle id + bundle version |
| What implements evaluation? | Policy handler | `handler_id` |
| Where is it enforced? | Governance Evaluation Point | point-specific contract (G1A) |

**Frozen flow:**

```text
Policy Catalog
    ↓
Policy Definition (policy_id + version)
    ↓
configured rule (rule_id)
    ↓
runtime bundle
    ↓
handler (handler_id)
    ↓
evaluation point
    ↓
PolicyDecision
```

**Identity separation:** `policy_id` ≠ `rule_id` ≠ `handler_id`.

The catalog describes capability; bundles carry what was configured; handlers execute; evaluation points enforce. Catalog metadata does **not** prove runtime coverage.

**Catalog is not:** `PolicyRuleRegistry`, `RuntimePolicyBundle`, `ImmutableRuntimePolicyBundle`, `PolicyEngine`, `RuntimePolicyEngine`, enforcer, HITL, evidence persistence, or a second plugin framework.

**Catalog vs bundle:** the catalog holds what **can** be selected (e.g. `external_commitment_approval` v2); a configured rule is what the application **did** select (e.g. `finance.contracts.require_cfo`); a runtime bundle is the **active** composed policy state containing that rule. Policy definition version and bundle version are separate — one definition version may appear in many bundles.

**G2B typed contract:** `intergrax.contracts.policy_catalog` implements immutable `PolicyDefinition` metadata — `policy_id`, definition `version`, `display_name`, `description`, `handler_id`, `configuration_contract_id`, and `source` (`built_in` / `plugin`). This answers *what policy capability exists* at the contract level only.

**G2C-1 resolution core:** `intergrax.runtime.policy.catalog.PolicyCatalog` implements deterministic exact `PolicyDefinition` resolution by `(policy_id, version)` — multi-version coexistence, explicit unknown-policy failure, explicit unsupported-version failure, deterministic duplicate conflict rejection, and **no** latest/fallback/downgrade behavior. The catalog may be empty initially; G2C-1 does **not** ship canonical built-in definitions. `PolicyCatalog` does **not** resolve `handler_id` or `configuration_contract_id`; plugin discovery/admission is outside this module. The canonical built-in policy catalog remains **Open**.

**G2C-2A rule / handler identity separation:** on the declarative runtime path, `rule_id` is configured rule instance identity and `handler_id` is runtime handler implementation identity. `PolicyRuleRegistry` resolves handlers by `handler_id`; evidence and outcomes attribute decisions to `rule_id`. G2C-2A-R1 completed active caller and fixture migration after the initial core identity split. No Policy Catalog wiring, no `policy_id` on configured rules, and no canonical built-in `PolicyDefinition` shipped yet — G2C-2B owns first real built-in policy and catalog-to-rule composition.

---

## Existing implementation map

Conceptual pieces mapped to existing mechanisms — **without** blanket maturity claims:

| Concept | Existing mechanism | Notes |
| -------- | ------------------- | ----- |
| Runtime policy contracts | `intergrax.contracts.runtime_policy` — `PolicyAction`, `PolicyDecision`, `EnforcementLevel` | Typed decision vocabulary |
| Policy facade | `intergrax.runtime.policy.PolicyEngine` | Facade over runtime and replay-oriented evaluators |
| Runtime evaluation | `intergrax.runtime.policy.RuntimePolicyEngine` | Interrupt, side-effect, and runtime-bound evaluation |
| Declarative tool-path enforcement | `DeclarativePolicyEnforcer`, declarative policy rules / bundles | DENY and REQUIRE_HITL before tool handler on wired paths |
| Meaningful side effects | `meaningful_side_effect.py`, `meaningful_side_effect_authorization.py` | Side-effect authorization composition |
| Canonical HITL | Nexus interrupt + HITL runtime (see REL canon) | `REQUIRE_HUMAN` / governed continuation |
| Post-run governance | `GovernanceService`, `ExecutionGuard` | Post-run replay, metrics, guard evaluation |
| Policy plugins / handlers | Platform plugin policy surface | Extends definition; enforcement stays at evaluation points |

Owner boundaries stay with each module and domain pair. This table is an orientation map, not an implementation dump.

---

## Current maturity

| Area | Status |
| ---- | ------ |
| Runtime policy decision contracts | **Implemented** — `PolicyDecision` / `PolicyAction` vocabulary |
| Policy facade and runtime engine | **Implemented slices** — bounded evaluation paths |
| Declarative policy on tool path | **Implemented slices** — DENY before handler; REQUIRE_HITL on demonstrated paths |
| Meaningful side-effect authorization | **Implemented mechanism** — not universal every-effect coverage |
| Canonical HITL integration | **Implemented** — bounded paths; not every evaluation point |
| Policy plugin / handler infrastructure | **Implemented slices** — admission / provenance partial |
| Post-run governance | **Implemented mechanisms** — `GovernanceService` / `ExecutionGuard` |
| Governance Evaluation Point architecture (G1A) | **Accepted** — [ADR-GOVERNED-EXECUTION-001](../technical/adr/entries/2026-08-16/ADR-GOVERNED-EXECUTION-001.md) |
| Contract hardening across critical runtime paths (G1B) | **Implemented core** — typed live contexts, typed meaningful-side-effect rules, immutable `PolicyDecision` and explicit bundle provenance invariants |
| Uniform evaluation-point runtime enum / god engine | **Rejected** — multiple owners preserved |
| Policy Catalog architecture (G2A) | **Accepted** — [ADR-GOVERNED-EXECUTION-002](../technical/adr/entries/2026-08-17/ADR-GOVERNED-EXECUTION-002.md) |
| Typed Policy Catalog contracts (G2B) | **Implemented** — immutable `PolicyDefinition` identity/source/configuration-contract metadata |
| Policy Catalog resolution core (G2C-1) | **Implemented** — exact `(policy_id, version)` resolution and deterministic conflict rejection; canonical built-in definitions not yet shipped |
| Declarative rule / handler identity separation (G2C-2A) | **Implemented** — configured rule identity and handler implementation identity are distinct on the declarative runtime path; G2C-2A-R1 completed active caller and fixture migration |
| Canonical built-in policy catalog | **Open** |
| Complete platform-wide coverage | **Not claimed** |
| Dedicated accepted public Governed Execution proof | **Not established** |
| Production qualification | **Not established** |

**Safe summary:** meaningful governance mechanisms and a hardened runtime core exist; coverage, policy catalog, qualification, and accepted public proof remain open.

---

## Relationship to adjacent capabilities

| Capability | Relationship |
| ---------- | ------------- |
| **Governed Execution** | Controls **what execution may proceed** under configured policy |
| **Observability & Auditability** | Records and reconstructs **what happened** — complementary, not interchangeable |
| **Token Optimization** | Optimizes selected context / prompt paths **under policy** |
| **Platform Extensibility** | Packages independent capability extensions, including policy extensions |
| **HITL** | One governance mechanism inside Governed Execution — not the whole capability |

---

## Review / deeper routes

| Topic | Canonical owner |
| ----- | ---------------- |
| Failure, retry, HITL, governed continuation | [RELIABILITY_FAILURE_AND_HITL.md](RELIABILITY_FAILURE_AND_HITL.md) |
| Governance Evaluation Points and enforcement ownership | [ADR-GOVERNED-EXECUTION-001](../technical/adr/entries/2026-08-16/ADR-GOVERNED-EXECUTION-001.md) |
| Policy Catalog identity and ownership | [ADR-GOVERNED-EXECUTION-002](../technical/adr/entries/2026-08-17/ADR-GOVERNED-EXECUTION-002.md) |
| Platform plugins and policy handler admission | [PLATFORM_PLUGINS.md](PLATFORM_PLUGINS.md) · [ADR-PLATFORM-PLUGIN-001](../technical/adr/entries/2026-08-14/ADR-PLATFORM-PLUGIN-001.md) |
| Observability and evidence spine | [OBSERVABILITY.md](OBSERVABILITY.md) · [PROOF_RECEIPTS.md](PROOF_RECEIPTS.md) |
| Public architecture overview | [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) |
| Current bounded evidence | [PROOFS.md](../proofs/PROOFS.md) |
| Runtime architecture hub | [intergrax_runtime_architecture.md](intergrax_runtime_architecture.md) |

Do not treat this document as a replacement for domain pair canon or maintainer plans.

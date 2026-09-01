# ADR-GOVERNED-EXECUTION-002: Policy Catalog Identity, Versioning, and Runtime Ownership

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-08-17 |
| **Deciders** | Platform architecture (Governed Execution G2A) |
| **Related** | [`GOVERNED_EXECUTION.md`](../../../architecture/GOVERNED_EXECUTION.md) · [ADR-GOVERNED-EXECUTION-001](../2026-08-16/ADR-GOVERNED-EXECUTION-001.md) · [ADR-RUNTIME-POLICY-BUNDLE-001](../2026-07-20/ADR-RUNTIME-POLICY-BUNDLE-001.md) · [ADR-PLATFORM-PLUGIN-001](../2026-08-14/ADR-PLATFORM-PLUGIN-001.md) |

## Context

Governed Execution already has configured policy rules, immutable policy bundles, runtime handler registries, policy provenance, and multiple evaluation/enforcement owners ([ADR-GOVERNED-EXECUTION-001](../2026-08-16/ADR-GOVERNED-EXECUTION-001.md)). Contract hardening on owned live paths is **implemented core** (G1B).

The platform still lacks a canonical user-facing model answering:

> What policy capability does the platform offer, which concrete rule did the application configure, and which runtime handler implements that policy?

Existing artifacts serve different roles:

| Artifact | Role today | Not the Policy Catalog |
| -------- | ---------- | ---------------------- |
| `ImmutableRuntimePolicyBundle` / `PolicyBundleRule` | Immutable attested policy **pack** / evidence object | Selection registry |
| `RuntimePolicyBundle` | Live runtime **composition** (tool access, budget, declarative runtime, domain fragments) | Capability catalog |
| `DeclarativePolicyRule` | One configured **rule instance** in a bundle | Policy definition |
| `PolicyRuleRegistry` | Runtime handler registration, resolution, evaluation | Definition catalog |

G2A freezes semantics and identity boundaries **before** any runtime catalog implementation. G2B will freeze typed contracts; G2C will implement built-in catalog and resolution.

## Decision

### 1. Policy Catalog responsibility

The **Policy Catalog** is the canonical registry of policy **definitions** available for application selection and configuration. It is an upstream capability-description and selection boundary - **not** the evaluator.

It will eventually answer queries such as:

- What policies are available?
- What is the stable `policy_id`?
- What version of the policy definition is this?
- What does this policy mean?
- At which governance boundary / category may it apply?
- What configuration shape does it expect?
- What outcomes can it produce?
- Which runtime handler capability is required?
- Is it built-in or provided through an admitted plugin?

G2A freezes **semantics and ownership only**. G2A does **not** freeze an oversized runtime DTO with all of these fields.

### 2. Canonical terminology

| Term | Identity field | Question answered |
| ---- | -------------- | ----------------- |
| **Policy definition** | `policy_id` + definition version | What kind of governance rule is this? |
| **Policy rule instance** | `rule_id` | Which concrete configured rule produced or contributed to this decision? |
| **Policy handler** | `handler_id` | Which runtime implementation interprets this policy type? |

**Frozen inequality:** `policy_id` != `rule_id` != `handler_id`.

Illustrative examples (not mandatory naming conventions):

- `policy_id`: `external_commitment_approval`
- `rule_id`: `finance.contracts.require_cfo`
- `handler_id`: `meaningful_side_effect_approval`

### 3. Identity model and flow

```text
POLICY CATALOG          "What can I choose?"
        |
        v
POLICY DEFINITION       policy_id + definition version
        |
        v
APPLICATION CONFIGURATION
        |
        v
POLICY RULE INSTANCE    rule_id
        |
        v
RUNTIME POLICY BUNDLE / COMPOSITION
        |
        v
POLICY HANDLER          handler_id
        |
        v
GOVERNANCE EVALUATION POINT
        |
        v
PolicyDecision
```

### 4. Catalog vs rule vs bundle vs handler

| Layer | Role | Example |
| ----- | ---- | ------- |
| **Catalog** | What **can** be selected / configured | `external_commitment_approval` v2 |
| **Configured rule** | What the application **did** configure | `finance.contracts.require_cfo` |
| **Runtime bundle** | What policy state is **active** for a runtime / application | bundle containing that rule and effective policy state |
| **Handler** | What **implements** evaluation for the supported definition shape | `meaningful_side_effect_approval` |
| **Evaluation point** | Where enforcement **runs** | e.g. MEANINGFUL_SIDE_EFFECT (conceptual G1A class) |

**Policy Catalog is not:**

- `PolicyRuleRegistry` (implementation registry)
- `RuntimePolicyBundle` or `ImmutableRuntimePolicyBundle` (configured / attested packs)
- `PolicyEngine` or `RuntimePolicyEngine` (evaluators)
- Enforcer, HITL coordinator, evidence persistence, plugin framework, or public Agent Marketplace

Do not create architecture ambiguity around the word "catalog".

### 5. Versioning model

Two versioning concepts - **not interchangeable**:

| Version kind | Tracks | Example |
| ------------ | ------ | ------- |
| **Policy definition version** | Semantic / configuration-contract evolution of the policy definition | `external_commitment_approval@2` |
| **Runtime policy bundle version** | One concrete immutable or composed policy pack identity | `finance-prod@17`, `finance-test@22` |

One policy definition version may appear in many runtime bundles. Do not overload bundle version as policy definition version.

**`rule_id` versioning:** `rule_id` remains the identity of a configured rule instance. Version belongs to the referenced policy definition / configuration contract, not necessarily to the instance name. G2A does **not** require `rule_id` to encode a version or prescribe dotted naming.

### 6. Built-in vs plugin-provided definitions

Policy definitions may originate from:

- built-in platform policy providers
- admitted plugin-provided policy providers

All plugin-provided definitions **must** flow through the existing Platform Plugin admission, provenance, and trust architecture ([ADR-PLATFORM-PLUGIN-001](../2026-08-14/ADR-PLATFORM-PLUGIN-001.md)). The catalog consumes admitted capability metadata; it does **not** bypass Platform Plugins.

**Rejected:** `GovernancePluginRegistry`, `GovernanceCatalogPluginLoader`, a second entry-point framework, or a second plugin admission path.

### 7. Handler ownership

`handler_id` identifies runtime implementation capability. A catalog definition may reference or require a handler capability, but the catalog:

- does not instantiate the handler
- does not execute the handler
- does not own handler lifecycle
- does not own enforcement

Runtime wiring and resolution remain with existing specialized owners (`PolicyRuleRegistry`, `RuntimePolicyEngine`, `DeclarativePolicyEnforcer`, etc.). The catalog is **not** a service locator.

### 8. Policy definition ownership

| Owner | Responsibility |
| ----- | -------------- |
| **Governed Execution** | Canonical Policy Definition contract semantics |
| **Domain / plugin owners** | Provide definitions under admitted extension rules |
| **Application / product** | Select definitions, supply allowed configuration, bind into runtime / policy configuration |
| **Runtime enforcement owners** | Consume validated configuration at evaluation points |

This preserves: **applications define the rules; Intergrax enforces the execution boundaries.**

### 9. Configuration validation boundary

Every catalog policy definition must eventually declare a typed, validated configuration contract (G2B). Architectural invariant aligned with G1A/G1B:

> Security-sensitive evaluation **must not** interpret arbitrary unchecked `dict[str, Any]` catalog configuration.

Dynamic external or plugin configuration may exist at ingestion boundaries, but it must be validated into a definition-owned typed contract **before** enforcement.

### 10. Supported outcomes and evaluation points

A policy definition should conceptually declare which governance outcomes it may emit or support. G2A does **not** invent a new universal decision enum. Reference:

- shared semantic categories and point-specific vocabularies in [ADR-GOVERNED-EXECUTION-001](../2026-08-16/ADR-GOVERNED-EXECUTION-001.md)
- existing `PolicyAction` / `PolicyDecision` contracts for live enforcement

Catalog metadata describes **capability**. Actual `PolicyDecision` / outcome types remain owned by enforcement contracts.

A definition may declare supported Governance Evaluation Point **categories** (conceptual G1A model). A definition stating support for MEANINGFUL_SIDE_EFFECT does **not** wire or prove that evaluation point. Catalog capability metadata != runtime coverage proof.

### 11. Security and failure invariants (architecture targets for G2B/G2C)

1. Unknown `policy_id` must fail validation / resolution before security-sensitive execution.
2. Unsupported definition version must **not** silently downgrade to another version.
3. Missing required handler capability must **not** result in implicit ALLOW.
4. Plugin-provided definitions must retain provider / admission provenance.
5. Catalog registration must eventually detect conflicting `policy_id` + version ownership deterministically.
6. Shipped / built-in policy definitions must **not** be silently overridden by plugins.

G2A documents these requirements; G2A does **not** implement them.

### 12. Clean-cut policy

Intergrax has no production user dependency requiring legacy catalog compatibility. Target architecture is clean-cut.

**Rejected:** alias IDs, legacy policy IDs, fallback string resolution, automatic downgrade, compatibility shim registries.

If future implementation discovers real callers requiring migration, migrate them intentionally rather than preserving weak identity semantics.

### 13. Existing implementation coupling (technical debt)

Today, `PolicyRuleRegistry.evaluate_rule` resolves handlers using `rule.rule_id`:

```python
handler = self._handlers.get(rule.rule_id)
```

This couples **rule instance identity** and **handler lookup identity** on the declarative slice. That is acceptable as **existing implementation state** but is **not** the target catalog identity model.

The target remains `policy_id` != `rule_id` != `handler_id`, even when a current path temporarily uses equal string values. G2A does **not** mandate immediate migration; migration belongs to G2B/G2C after typed contracts are designed.

Similarly, `PolicyHandlerProvenance.rule_id` records handler entry-point metadata using `rule_id` field naming - provenance of admitted handlers, not catalog definition identity.

## Consequences

### Positive

- Unambiguous separation between capability catalog, configured instances, runtime packs, and handlers.
- Clear ownership for G2B typed contracts and G2C runtime catalog without redesigning G1A/G1B enforcement.
- Plugin extension reuses Platform Plugins; no second governance plugin framework.
- Security invariants and clean-cut identity model are explicit before implementation.

### Negative

- Current declarative paths remain coupled until G2B/G2C migration.
- Architects must distinguish catalog metadata from bundle evidence and from evaluation proof.

## Non-goals (G2A)

- Implement `PolicyDefinition`, `PolicyCatalog`, registry code, catalog API, persistence, or configuration parser
- Change `DeclarativePolicyRule`, `PolicyRuleRegistry`, `PolicyBundleRule`, `RuntimePolicyBundle`, `ImmutableRuntimePolicyBundle`, `PolicyEngine`, or `RuntimePolicyEngine`
- Migrate `rule_id` / `handler_id` coupling
- Add built-in policies, plugin loaders, HITL changes, or provenance code changes
- Fix ADR index checker regex for multi-segment IDs

## Follow-up implementation stages

| Stage | Scope |
| ----- | ----- |
| **G2B** | Typed Policy Catalog contracts |
| **G2C** | Built-in catalog and resolution |
| **G3** | Evaluation Point coverage |

## Compliance

- Tier boundaries preserved - catalog semantics only; no runtime code in G2A
- Aligns with [ADR-GOVERNED-EXECUTION-001](../2026-08-16/ADR-GOVERNED-EXECUTION-001.md) evaluation-point and typed-context rules
- [`GOVERNED_EXECUTION.md`](../../../architecture/GOVERNED_EXECUTION.md) updated with Policy Catalog section; public maturity unchanged

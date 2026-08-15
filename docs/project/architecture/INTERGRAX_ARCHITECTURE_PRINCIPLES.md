# INTERGRAX Architecture Principles

**Status:** Canonical platform governance architecture  
**Owner:** Intergrax Platform Architecture  
**Audience:** Platform architects, domain owners, application authors, reviewers, AI coding agents  
**Scope:** Evolution, ownership, adoption, proof, and governance of reusable Intergrax capabilities  
**Applies to:** All existing and future platform domains, Tier-3 applications, product roadmaps, and implementation plans

---

## 1. Purpose

This document defines the architectural principles governing how the Intergrax platform evolves.

Domain architecture documents answer questions such as:

```text
How does a subsystem work?
What contracts does it expose?
Which components own its runtime behavior?
```

This document answers a different class of questions:

```text
When should a new platform domain be created?
Who owns a reusable capability?
When may an application implement infrastructure directly?
What must happen before platform implementation begins?
How are applications used as adopters and proofs?
```

These rules exist to prevent generic infrastructure from becoming embedded inside individual products and to preserve a coherent, modular platform as the number of domains and applications grows.

All platform domains and Tier-3 applications MUST comply with these principles.

---

## 2. Architectural hierarchy

Intergrax architecture is governed through the following hierarchy:

```text
Intergrax Architecture Principles
        │
        ▼
Intergrax Runtime Architecture
        │
        ▼
Platform Domain Architecture
        │
        ▼
Platform Domain Implementation Plans
        │
        ▼
Tier-3 Application Architecture
        │
        ▼
Application Adoption
        │
        ▼
Live Product Proof
```

Each level has a different responsibility.

| Level | Responsibility |
|---|---|
| Architecture principles | Defines how the platform evolves and how ownership is assigned |
| Runtime architecture | Defines the complete Intergrax platform topology |
| Domain architecture | Defines one reusable platform capability |
| Domain implementation plan | Defines the ordered implementation roadmap for that capability |
| Application architecture | Defines product-specific composition and behavior |
| Adoption | Integrates a completed platform capability into an application |
| Product proof | Demonstrates the capability through a real workload |

Lower levels MUST NOT redefine ownership established by higher levels.

---

## 3. Platform capability classification

Before implementing a significant capability, its ownership must be classified.

### 3.1 Application-specific capability

A capability is application-specific when it:

- represents product-specific business behavior,
- is meaningful only inside one application,
- cannot reasonably be reused by other applications,
- depends on domain rules belonging to that product,
- does not provide generic infrastructure.

Examples:

```text
Local Workspace collection naming conventions
Legal-review workflow rules
Product-specific prompts
Application-specific default capabilities
Application-specific UI behavior
```

Application-specific capabilities belong in the application architecture and roadmap.

### 3.2 Platform capability

A capability is platform-wide when it:

- can reasonably be reused by multiple applications,
- provides infrastructure rather than product behavior,
- introduces reusable contracts, lifecycle, policies, adapters, events, or engines,
- solves a recurring problem for Tier-3 applications,
- would otherwise be reimplemented independently by different products.

Examples:

```text
Interaction intake
Application hosting
Proof receipts
Provider integrations
Generic scheduling
Cross-application synchronization
Common lifecycle supervision
Standardized observability
```

Platform capabilities MUST have platform ownership.

---

## 4. PLATFORM-INV-001 — Generic Capability Ownership

> Generic capabilities MUST originate as platform-owned architecture domains or as changes to an existing platform domain.
>
> They MUST NOT originate as application-owned infrastructure.

When application development reveals a reusable capability, implementation in the application must stop until ownership is resolved.

Forbidden outcome:

```text
LKW implements generic hosting
Legal Assistant implements another generic hosting layer
Coding Assistant implements a third hosting layer
```

Required outcome:

```text
Application Hosting platform domain
        │
        ├── Local Workspace adoption
        ├── Legal Assistant adoption
        └── Coding Assistant adoption
```

Applications consume platform capabilities through public contracts.

---

## 5. PLATFORM-INV-002 — Architecture Before Adoption

The required order for a new generic capability is:

```text
1. Capability identification
2. Ownership classification
3. Architecture decision
4. Architecture definition
5. Implementation plan
6. Integration into platform architecture
7. Platform implementation
8. Application adoption
9. Live product proof
```

This order is normative.

Applications MUST NOT implement a generic capability first and move it into the platform later as the intended delivery strategy.

Prototype code may be used for discovery, but it MUST NOT become the canonical architecture or production implementation before platform ownership is established.

---

## 6. PLATFORM-INV-003 — Single Architectural Ownership

Every reusable capability must have exactly one architectural owner.

The owner must provide:

```text
one canonical architecture
one canonical implementation plan
one contract namespace
one implementation roadmap
one acceptance model
one verification strategy
```

Cross-domain dependencies are allowed.

Shared ownership without a clear primary owner is not allowed.

Incorrect:

```text
Part of the capability belongs to LKW
Part belongs to Tier-3
Part belongs to runtime
Part is undocumented
```

Correct:

```text
APPLICATION_HOSTING owns hosting
TIER3_APPLICATION_ENVIRONMENT defines application composition
LKW adopts and proves hosting
```

Feature documentation may coordinate multiple domains, but it does not replace domain ownership.

---

## 7. PLATFORM-INV-004 — Applications Are Platform Adopters

Tier-3 applications are consumers and composition shells for platform capabilities.

Applications own:

```text
business behavior
product-specific configuration
application-specific policies
application-specific capabilities
product UX
application-specific hooks and components
live product proofs
```

Applications do not own:

```text
generic engines
generic lifecycle infrastructure
generic provider abstractions
generic supervisor mechanisms
generic transport frameworks
generic event systems
generic persistence contracts
generic OS abstractions
```

Applications may provide extension implementations through public platform contracts.

Applications MUST NOT create private alternatives to platform mechanisms.

---

## 8. PLATFORM-INV-005 — Proof Follows Implementation

A live product proof validates an implemented platform capability.

A proof does not define the architecture.

The correct sequence is:

```text
architecture
→ implementation
→ adoption
→ proof
```

Not:

```text
application proof script
→ inferred platform architecture
```

A proof should demonstrate:

- real integration,
- real workload behavior,
- public contract usability,
- developer experience,
- operational correctness,
- absence of forbidden bypasses.

The application providing the proof remains the adopter, not the architectural owner.

---

## 9. PLATFORM-INV-006 — Deployment Transparency

Deployment is a platform concern, not an application-semantic concern.

The same Tier-3 application may be executed through different deployment models, including:

```text
standalone execution
hosted execution
batch execution
container execution
future platform deployment models
```

The deployment model may change:

```text
process lifecycle
readiness
health
signals
restart behavior
single-instance enforcement
interaction availability
resource supervision
```

It MUST NOT change:

```text
business logic
Task semantics
capability semantics
agent behavior
Nexus execution
domain results
application policies
```

Applications MUST NOT contain parallel business paths such as:

```python
if hosted:
    execute_hosted_business_logic()
else:
    execute_standard_business_logic()
```

The deployment mechanism wraps the application. It does not redefine it.

---

## 10. PLATFORM-INV-007 — Simplicity First

Public platform APIs should minimize the amount of infrastructure knowledge required from application authors.

The default authoring experience should favor:

```text
one composition root
sensible defaults
declarative configuration
convention over configuration
small extension surfaces
```

Advanced behavior should remain available through explicit extension points.

The recommended author-experience distribution is:

```text
80% of applications:
    one profile and one platform entry point

15% of applications:
    profile plus hooks and components

5% of applications:
    custom policies, plugins, adapters, or providers
```

Platform internals may be complex.

Application authoring must not expose that complexity without necessity.

---

## 11. PLATFORM-INV-008 — One Composition Root Per Domain

A platform domain should provide one clear public composition root.

Examples:

```text
ApplicationEnvironmentProfile
HostedApplicationProfile
IntegrationProfile
ObservabilityProfile
```

Supporting contracts may exist, but application authors should not have to manually coordinate many independent registries and managers for ordinary scenarios.

A public composition root should:

- group related configuration,
- apply defaults,
- validate incompatible combinations,
- expose stable extension points,
- drive framework wiring.

Separate configuration objects are acceptable when nested under or referenced by one canonical root.

---

## 12. PLATFORM-INV-009 — Stable Core, Explicit Extensions

Every major platform subsystem should distinguish between:

```text
stable default behavior
explicit extension contracts
internal implementation details
```

Preferred extension mechanisms are:

```text
profiles
hooks
components
events
policies
plugins
providers
adapters
```

Extensions MUST NOT require bypassing the owning engine.

Custom code should plug into the lifecycle of the subsystem rather than reimplementing it.

Examples:

```text
custom hosting component
custom interaction adapter
custom provider integration
custom lifecycle hook
custom event subscriber
```

Anti-patterns:

```text
custom private event loop
custom private Nexus path
custom process supervisor inside an application
direct vendor SDK use outside the provider boundary
```

---

## 13. PLATFORM-INV-010 — Existing Platform Mechanisms Must Be Reused

New domains and applications must reuse existing canonical mechanisms where their responsibility already exists.

Examples:

```text
Use Task instead of defining another execution request model.
Use the platform event spine instead of creating a private event bus.
Use Integration contracts instead of importing vendor SDKs into applications.
Use ProofReceiptStore instead of persisting proof documents directly.
Use UnifiedTaskRunner instead of calling application agents directly.
```

A new abstraction is justified only when:

- the existing abstraction does not own the required responsibility,
- extending the existing abstraction would violate its boundaries,
- the new abstraction has clear ownership,
- the decision is documented.

---

## 14. Platform domain creation process

A new domain should be created when all of the following are true:

1. The capability is reusable across applications.
2. It has a coherent responsibility boundary.
3. It requires public contracts or a reusable engine.
4. It needs an independent implementation roadmap.
5. Existing domains cannot own it without losing cohesion.
6. It can be verified independently from a single product.

The minimum domain documentation package is:

```text
ADR
docs/project/architecture/<DOMAIN>.md
docs/project/architecture/satellites/<DOMAIN>_*.md when needed
docs/project/maintainers/plans/<DOMAIN>.md
docs/project/maintainers/plans/satellites/<DOMAIN>_*.md when needed
runtime architecture registration
cross-domain ownership references
acceptance and verification matrix
```

A new domain must not begin as an undocumented package namespace.

---

## 15. Extending an existing domain

Not every new generic capability requires a new domain.

An existing domain should be extended when:

- the capability fits its established ownership,
- it reuses the same composition root,
- it does not require a new independent engine,
- it does not create a distinct lifecycle,
- it does not significantly expand the developer mental model,
- the existing implementation plan can own it coherently.

A new domain should be preferred when adding the capability would make the current domain responsible for unrelated concerns.

---

## 16. ADR requirements

An ADR is required when a change introduces one or more of the following:

```text
a new platform domain
a new public composition root
a new execution engine
a new lifecycle model
a new cross-domain ownership boundary
a replacement for an existing canonical mechanism
a new deployment model
a significant compatibility or migration strategy
```

The ADR should document:

- context,
- problem,
- alternatives considered,
- decision,
- ownership,
- consequences,
- compatibility,
- implementation references,
- affected architecture domains.

An ADR does not replace architecture documentation.

It records the decision that authorizes it.

---

## 17. Architecture and plan pairing

Every platform domain must maintain a 1:1 architecture and plan pair:

```text
docs/project/architecture/<DOMAIN>.md
docs/project/maintainers/plans/<DOMAIN>.md
```

Architecture defines:

```text
purpose
boundaries
contracts
components
invariants
flows
extension points
anti-patterns
acceptance
```

The implementation plan defines:

```text
ordered tasks
dependencies
code targets
migration strategy
test strategy
proof strategy
status
```

Architecture must not contain an evolving implementation backlog.

The plan must not redefine architectural ownership.

---

## 18. Hub and satellite documentation

Large domains should use a hub-and-satellite structure.

### Hub

The hub is the default entry point for Cursor and human readers.

It should contain:

- domain purpose,
- core boundaries,
- canonical mental model,
- primary contracts,
- invariants,
- task entry points,
- links to satellites.

### Satellite

Satellites contain:

- extended normative detail,
- large contract matrices,
- OS-specific details,
- migrations,
- implementation registers,
- extensive acceptance criteria.

The default read path should avoid loading every satellite.

Each hub should clearly state which satellite to load for which task.

---

## 19. Domain ownership versus feature coordination

A domain owns implementation truth.

A multi-layer feature coordinates work across domains.

Feature documents may define:

- cross-domain outcomes,
- integration order,
- dependencies,
- combined product capabilities.

Feature documents MUST NOT:

- redefine domain contracts,
- become the only implementation plan for domain-owned work,
- create unclear shared ownership,
- override domain invariants.

Concrete implementation tasks must remain in the owning domain plans.

---

## 20. Application roadmap rules

Application roadmaps should contain:

```text
application-specific capabilities
platform adoption tasks
product integration
product proof
product UX
```

They should not contain implementation of generic platform infrastructure.

When a roadmap item is recognized as generic:

```text
1. Stop the application implementation.
2. Move ownership to a platform domain.
3. Create or update the platform architecture and plan.
4. Implement the platform capability.
5. Return to the application through an adoption task.
```

The application task should then be renamed from:

```text
Implement generic capability
```

to:

```text
Adopt <PLATFORM_DOMAIN> and provide product proof
```

---

## 21. First adopter model

Every major platform capability should have a first adopter.

The first adopter validates:

- API ergonomics,
- wiring complexity,
- extension model,
- compatibility,
- operational behavior,
- real workload behavior,
- reviewer experience.

The first adopter may expose missing platform requirements.

When this happens, fixes should return to the owning platform domain.

The adopter must not accumulate generic workarounds.

---

## 22. Proof requirements

A product proof should establish:

```text
the platform-owned implementation is used
the application does not bypass public contracts
the capability works under a real workload
the result can be independently inspected
the application-specific code remains thin
```

A proof should include guardrails against known bypasses.

Examples:

```text
no direct vendor SDK
no direct agent invocation
no in-memory fallback in a live proof
no static markdown result as source of truth
no application-owned copy of the engine
```

Proof artifacts should be structured and machine-verifiable whenever possible.

---

## 23. Compatibility and migration

Platform evolution should avoid creating permanent parallel systems.

Migration should prefer:

```text
new canonical contract
→ migrate all active consumers
→ remove obsolete contract
```

over:

```text
new contract for one adopter
+ old contract for everyone else
```

Compatibility aliases may exist temporarily only when:

- they refer to the same implementation,
- they contain no independent behavior,
- their purpose is documented,
- their removal path is known.

Parallel engines for the same responsibility are prohibited unless an ADR explicitly authorizes them.

---

## 24. PLATFORM-INV-011 — No Hidden Platform Inside Applications

An application must not become the de facto owner of generic infrastructure because the code physically appeared there first.

Signals of hidden platform infrastructure include:

```text
generic contracts inside applications/<app>/
OS abstractions owned by one product
generic event buses in a product package
reusable supervisor logic under an application namespace
provider-independent persistence engines owned by an application
```

When such code is identified, it should be classified and moved to the appropriate platform domain before further expansion.

---

## 25. PLATFORM-INV-012 — Applications Remain Semantically Portable

An application should remain portable across supported platform deployment and integration models.

Portability means that the application can be:

```text
run directly
hosted continuously
tested in-process
started in a container
adopt future deployment models
```

without rewriting its domain behavior.

Platform wrappers may provide additional operational capabilities, but application semantics remain stable.

---

## 26. PLATFORM-INV-013 — Operational Mechanisms Stay Outside Cognitive Layers

Operational infrastructure must not leak into:

```text
agent cognition
Nexus orchestration decisions
business capability implementations
```

Examples of operational mechanisms:

```text
process supervision
instance locks
OS signals
service registration
restart backoff
readiness probes
health aggregation
```

These belong to platform operational domains such as Application Hosting, Reliability, Observability, or Integrations.

Agents and Nexus should consume stable runtime conditions, not implement them.

---

## 27. PLATFORM-INV-014 — Events Are Shared Platform Infrastructure

Subsystem-specific events should use the canonical Intergrax event model and observability spine.

New domains MUST NOT create isolated private event buses without an ADR.

Events should be:

```text
typed
versioned
correlated
redacted
tenant-aware when applicable
observable
```

Domain events may define domain-specific payloads, but transport and lifecycle should use platform infrastructure.

---

## 28. PLATFORM-INV-015 — Hooks Do Not Become Private Engines

Hooks allow custom reactions at controlled lifecycle points.

Hooks MUST NOT become:

```text
private orchestration loops
parallel execution engines
unbounded background runtimes
untracked provider access
```

A hook may:

- validate,
- enrich,
- block,
- modify,
- initialize,
- clean up,
- emit events.

A hook must remain subordinate to the owning platform engine.

---

## 29. Architecture decision filter

Before implementing a significant capability, answer:

### Question 1

Is the capability exclusively meaningful to one application?

```text
Yes → application ownership may be correct.
No  → continue.
```

### Question 2

Could another Tier-3 application reasonably need it?

```text
Yes → platform capability candidate.
```

### Question 3

Does an existing platform domain already own this responsibility?

```text
Yes → extend that domain.
No  → continue.
```

### Question 4

Does it require reusable contracts, lifecycle, policies, adapters, or an engine?

```text
Yes → new domain candidate.
```

### Question 5

Can it be implemented and tested independently of the discovering application?

```text
Yes → platform ownership is strongly indicated.
```

### Question 6

Is the application currently about to implement generic infrastructure?

```text
Yes → stop and resolve architecture first.
```

---

## 30. Architecture review checklist

A platform architecture review should confirm:

```text
[ ] The capability has one owner.
[ ] Domain versus application responsibility is explicit.
[ ] Existing platform mechanisms are reused.
[ ] The public composition root is identified.
[ ] The default developer experience is simple.
[ ] Advanced extension points are explicit.
[ ] No application-owned parallel engine is introduced.
[ ] Architecture and implementation plan are paired.
[ ] Migration and compatibility are defined.
[ ] A first adopter is named.
[ ] The proof strategy is defined.
[ ] Product proof does not replace platform verification.
[ ] Cross-domain dependencies are linked.
[ ] The capability is registered in runtime architecture.
```

Implementation must not start until the required architecture items are complete.

---

## 31. Implementation review checklist

Before closing a platform implementation task, verify:

```text
[ ] Implementation belongs to the owning platform package.
[ ] Application-specific code remains thin.
[ ] Public contracts match architecture.
[ ] Obsolete competing contracts are removed or explicitly deprecated.
[ ] Focused tests pass.
[ ] Cross-domain regression tests pass.
[ ] Forbidden bypasses are absent.
[ ] Documentation status matches actual implementation.
[ ] The next plan task is explicit.
```

---

## 32. Adoption review checklist

Before closing an application adoption task, verify:

```text
[ ] The application uses platform public contracts.
[ ] No generic engine was copied into the application.
[ ] Product-specific hooks/components are clearly application-owned.
[ ] Standalone application semantics remain unchanged.
[ ] The live proof uses the platform implementation.
[ ] Reviewer evidence is inspectable.
[ ] Guardrails prove no bypass occurred.
[ ] Generic issues discovered during adoption were fixed in the platform domain.
```

---

## 33. Examples

### 33.1 Interaction intake

Discovery:

```text
Applications need multiple interaction channels.
```

Correct evolution:

```text
Platform interaction contracts
→ InteractionIntakeService
→ application adoption
→ LKW proof
```

Incorrect evolution:

```text
LKW-specific Slack parser
→ LKW-specific task model
→ later copied to another application
```

### 33.2 Proof receipts

Discovery:

```text
Product proofs need structured authoritative evidence.
```

Correct evolution:

```text
ProofReceipt contract
→ ProofReceiptStore
→ DocumentStore
→ MongoDB vendor
→ LKW adoption
→ reviewer proof
```

### 33.3 Application hosting

Discovery:

```text
LKW needs to run continuously in the background.
```

Correct evolution:

```text
APPLICATION_HOSTING domain
→ Hosted Application Engine
→ platform lifecycle and supervision
→ LKW adoption
→ always-on product proof
```

Incorrect evolution:

```text
LKW daemon
→ LKW process lock
→ LKW restart loop
→ product-owned OS service implementation
```

---

## 34. Relationship to Application Hosting

Application Hosting is a direct example of these principles.

The capability was discovered through LKW, but:

```text
LKW does not own generic hosting.
APPLICATION_HOSTING owns generic hosting.
LKW becomes the first adopter and proof.
```

The same rule applies to future capabilities discovered through any application.

---

## 35. Governance of these principles

Changes to this document require architectural review.

A new `PLATFORM-INV-*` invariant should be added only when:

- it applies across multiple domains,
- it governs platform evolution or ownership,
- it cannot be expressed as a domain-local invariant,
- it has clear practical consequences.

Domain-local rules belong in domain architecture documents.

Application-specific rules belong in application architecture documents.

---

## 36. Non-goals

This document does not:

```text
define individual runtime contracts
replace domain architecture documents
replace implementation plans
define project-management workflow
define source-control policy
define coding style
define product roadmaps
```

It defines architectural governance and capability ownership.

---

## 37. Final platform evolution model

The canonical Intergrax evolution model is:

```text
Application or product reveals a need
        │
        ▼
Classify the capability
        │
        ├── application-specific
        │       └── implement in the application
        │
        └── reusable platform capability
                │
                ▼
        Establish platform ownership
                │
                ▼
        ADR + Architecture + Plan
                │
                ▼
        Register in platform architecture
                │
                ▼
        Implement in the platform
                │
                ▼
        Adopt in the first application
                │
                ▼
        Execute live product proof
                │
                ▼
        Reuse in further applications
```

This process ensures that Intergrax grows through reusable platform composition rather than duplicated product infrastructure.

---

## 38. Summary

Intergrax applications are clients of the platform.

They compose platform domains, add product-specific behavior, and provide real-world proofs.

They do not own generic infrastructure.

Every reusable capability must have:

```text
one architectural owner
one architecture
one implementation plan
one platform implementation
one or more application adopters
```

The governing sequence is:

```text
architecture
→ implementation
→ adoption
→ proof
```

The long-term scalability of Intergrax depends on preserving this separation.

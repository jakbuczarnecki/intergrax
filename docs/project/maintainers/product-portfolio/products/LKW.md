# Local Knowledge Workspace — Portfolio Control Card

**Document type:** Per-product Portfolio Control index  
**Owner:** Portfolio Control Session  
**Baseline ingested:** MP-13 (2026-08-20)

---

## 1. Identity

| Field | Value |
|-------|-------|
| Product | Local Knowledge Workspace (LKW) |
| Program role | Existing reference product |
| Program State | **ACTIVE** |
| Baseline type | **REFERENCE BASELINE** |
| Baseline commit | `821eb7f6b2096de142822a29abc4546ee387a158` |
| Baseline date | 2026-08-20 |
| Portfolio recommendation | **CONTINUE** |
| Relative priority | **HIGH** |

LKW predates the multi-product reuse experiment. This reference baseline is **not** a retroactive T0 and must not be presented as preregistered reuse evidence.

---

## 2. Authoritative sources

This card indexes Portfolio Control state. It does **not** override product-owned sources.

| Topic | Authoritative document |
|-------|------------------------|
| Product architecture | [ARCHITECTURE.md](../../../../../applications/local_workspace_application/docs/ARCHITECTURE.md) |
| Current execution / status | [IMPLEMENTATION_PLAN.md](../../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |
| Knowledge access | [KNOWLEDGE_ACCESS_ARCHITECTURE.md](../../../../../applications/local_workspace_application/docs/KNOWLEDGE_ACCESS_ARCHITECTURE.md) |
| Hybrid Ask | [HYBRID_ASK_ARCHITECTURE.md](../../../../../applications/local_workspace_application/docs/HYBRID_ASK_ARCHITECTURE.md) |
| Conversation context | [CONVERSATION_CONTEXT_ARCHITECTURE.md](../../../../../applications/local_workspace_application/docs/CONVERSATION_CONTEXT_ARCHITECTURE.md) |
| Platform proof scope | [LKW_PLATFORM_PROOF.md](../../../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) |
| Certification matrix | [LKW_PLATFORM_CERTIFICATION_MATRIX.md](../../../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_CERTIFICATION_MATRIX.md) |
| Governed hybrid knowledge proof | [GOVERNED_HYBRID_KNOWLEDGE_PROOF.md](../../../../../applications/local_workspace_application/docs/proof/GOVERNED_HYBRID_KNOWLEDGE_PROOF.md) |

Where architecture documents retain historical “current state” sections predating later accepted implementation, [IMPLEMENTATION_PLAN.md](../../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) is authoritative for execution status.

---

## 3. Current product baseline

**Product level:** Backend Product Alpha — LKW MVP / Hybrid Knowledge Workspace

At this reference baseline, accepted or implemented product areas include:

- managed / indexed knowledge functionality;
- Workspace Ask;
- provider-neutral Hybrid Ask V2 implementation;
- Knowledge Query Orchestrator path;
- indexed + live evidence types and unified provenance path;
- accepted LKW Knowledge Access foundations;
- Conversation Context implementation accepted by the current implementation plan;
- conversational frontend accepted for the current supported scope;
- durable conversation / thread behavior currently accepted;
- provider-neutral knowledge / capability consumption mechanisms used by LKW.

Not every target in architecture documents is implemented. The implementation plan is authoritative where older architecture “current state” paragraphs predate later accepted work.

---

## 4. Current execution position

**Current direct LKW task:** `LKW-PLUGIN-CAPABILITY-CONFIGURATION-1` — **READY_FOR_REVIEW**

**Next direct task:** `LKW-INDEXED-SOURCE-LIFECYCLE-1` — **PLANNED**

**READY_FOR_REVIEW is not ACCEPTED.** Portfolio Control must not treat review-pending work as closed product evidence.

Major planned or incomplete areas (summary only — see implementation plan for order and gates):

- full Indexed Source lifecycle;
- generic Live Access lifecycle;
- unified inspection / operations;
- natural-language administration completion;
- product hardening;
- deployment / onboarding completion;
- final product acceptance / platform proof;
- LKW 1.0 release gate.

---

## 5. Product-owned responsibility boundary

LKW owns product semantics and user-visible behavior for:

- workspace product meaning;
- Workspace Knowledge Configuration product semantics;
- Indexed Source lifecycle from the product / user perspective;
- Live Access Binding product configuration;
- Query Policy product meaning;
- Hybrid Ask product orchestration;
- freshness, status, and provenance inspection as product behavior;
- frontend behavior;
- conversation / workspace binding semantics;
- product recovery behavior;
- safe source detach and local removal semantics.

These remain product-owned even when implemented through shared platform services. Consumption does not convert them into generic platform abstractions.

---

## 6. Shared-platform responsibilities consumed

Mechanisms **consumed by LKW** where authoritative sources support current use (not claimed as originated by LKW):

- Nexus / shared orchestration boundary;
- application hosting;
- shared Task / runtime / observability infrastructure;
- durable DocumentStore infrastructure;
- durable / background work infrastructure;
- shared memory / conversation mechanisms where actually consumed;
- policy / governance enforcement;
- Vendor Knowledge provider-neutral registration and capability discovery;
- Tenant Connection / Remote Resource concepts and implementations where current code supports them;
- provider-neutral live capability execution;
- provider dispatch and normalized provider contracts;
- shared persistence / recovery mechanisms;
- integration / tool / skill / agent boundaries.

Phrase as **consumed by LKW**, not **created by LKW**. Consumption does not establish historical origin.

---

## 7. Existing proof baseline

### LKW Platform Proof

Bounded evidence includes ([LKW_PLATFORM_PROOF.md](../../../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md)):

- real Intergrax application startup / readiness;
- persistence across restart;
- background work through real infrastructure;
- ProofReceipt-backed evidence;
- Application Hosting behavior;
- file watcher / indexing paths;
- supported Ask / indexed product proof scope.

### Platform certification

Baseline certification state ([LKW_PLATFORM_CERTIFICATION_MATRIX.md](../../../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_CERTIFICATION_MATRIX.md)):

| Profile | Application Hosting | Native host certified |
|---------|---------------------|----------------------|
| Windows native | live-certified | yes |
| Linux Docker runtime | live-certified | no |
| Linux native host | not live-certified | no |
| macOS native | not live-certified | no |

Application Hosting certification is **not** the same as full multi-phase Core Platform Proof certification.

### Governed Hybrid Knowledge Proof

At audited snapshot COMM-5D-R1 hardened proof boundaries ([GOVERNED_HYBRID_KNOWLEDGE_PROOF.md](../../../../../applications/local_workspace_application/docs/proof/GOVERNED_HYBRID_KNOWLEDGE_PROOF.md)):

- real indexed ingestion / search;
- TenantConnection rehydration;
- provider-neutral live capability path;
- canonical `LiveAccessLifecycleService.disable` / runtime authority reload;
- persisted Hybrid Ask run / evidence path;
- normal tenant / workspace scope indexed identity (post-audit doc clarification at baseline HEAD).

---

## 8. Explicit non-claims

This baseline does **not** prove:

- LKW 1.0 completion;
- finished SaaS;
- universal production readiness;
- complete vendor portfolio;
- certification on all OS / deployment topologies;
- customer validation;
- commercial validation;
- that every consumed platform capability originated from LKW;
- cross-product reuse by another independent product.

No M1–M6 reuse scoring applies to LKW. There is no preregistered T0.

---

## 9. Platform-pressure baseline

### A. Historical attribution

No retrospective `PI-*` record is accepted merely from capability consumption. Historical origin must be supported by exact evidence before it can enter [PLATFORM_IMPACT_LEDGER.md](../PLATFORM_IMPACT_LEDGER.md) as an accepted fact.

Do not claim Memory, Hosting, Governance, Vendor Knowledge, or other shared areas were created because of LKW unless exact historical evidence establishes it.

### B. Current known candidate pressure

**Not** an accepted `PI-*` record.

The current implementation plan identifies a bounded Vendor Knowledge problem-radar finding: the shared core does not yet expose a generic durable / indexed eligibility descriptor; LKW reports that dimension as `UNKNOWN` rather than inferring it from capability IDs or source kinds.

| Field | Value |
|-------|-------|
| Status | **G4 CANDIDATE — NOT YET CLASSIFIED** |
| Classification | Not `EXTENDED_GENERALLY` until G4 accepts generalization |

Detail: [IMPLEMENTATION_PLAN.md](../../../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) § B (plugin capability consumption).

---

## 10. Reference-baseline use in future product reviews

For future products, this card enables Portfolio Control to ask:

- Did the capability already exist at the reference baseline?
- Is the new product reusing it unchanged or configured?
- Is it actually LKW-owned semantics and therefore not reusable platform surface?
- Does the new product expose a generality failure?
- Does it force a real shared-platform extension?

---

## 11. Open portfolio questions

- Which capabilities currently consumed by LKW will survive unchanged in non-knowledge-centric applications?
- Which LKW-facing semantics have accidentally been treated as shared platform?
- Will other products consume Vendor Knowledge without LKW Workspace semantics?
- Will governed execution / evidence abstractions survive finance, supply-chain, risk, and deployment workflows?
- Which current LKW problem-radar findings become true G4 platform changes?

---

## 12. Next portfolio state

- Baseline ingestion completed by MP-13 once accepted.
- LKW remains **ACTIVE**.
- Next product work continues according to the authoritative LKW roadmap.
- Future material shared-platform pressure must pass G4.
- No retroactive T0 is created.

---

## Related portfolio documents

| Question | Document |
|----------|----------|
| Live portfolio dashboard | [PORTFOLIO_STATUS.md](../PORTFOLIO_STATUS.md) |
| Platform impact records | [PLATFORM_IMPACT_LEDGER.md](../PLATFORM_IMPACT_LEDGER.md) |
| Program governance | [MULTI_PRODUCT_PROGRAM.md](../MULTI_PRODUCT_PROGRAM.md) |

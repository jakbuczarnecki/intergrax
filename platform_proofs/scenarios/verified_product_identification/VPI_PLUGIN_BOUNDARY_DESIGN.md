# VPI — Plugin Boundary Design

**Task ID:** VPI-PLUGIN-BOUNDARY-DESIGN  
**Scenario slug:** `verified_product_identification`  
**Document role:** Target architecture for scenario extensibility — domain logic as platform plugins, not a hidden in-scenario framework.  
**Status:** Design only (no plugin extraction, no platform core changes in this task).

**Governance (normative):** [PLATFORM_PROOF_AUTHORING_GUIDE.md § Scenario platform integration and pluginability governance](../../PLATFORM_PROOF_AUTHORING_GUIDE.md#scenario-platform-integration-and-pluginability-governance) · [SCENARIO_STRUCTURE.md](../docs/SCENARIO_STRUCTURE.md) · [PLATFORM_CAPABILITY_MAPPING.md](PLATFORM_CAPABILITY_MAPPING.md)

**Code revision anchor:** `platform_proofs/scenarios/verified_product_identification/` application pipeline, ports, composition roots, `integrations/`, `storage_bootstrap/`, `composition/`.

---

## 1. Plugin Boundary Principles

### What a plugin is (target state)

A **scenario plugin** for VPI:

1. **Implements** a **public platform contract** (existing `intergrax.*` port/policy/protocol, or a **promoted** cross-scenario contract introduced after gap classification).
2. Is **replaceable** without editing platform core, pipeline stage order, or execution engine.
3. Is **wired only** at composition roots (`application/pipeline/composition.py`, `retrieval/composition.py`, `application/runtime_composition.py`, scenario `composition/*`) via constructor injection.
4. Depends on **contracts and scenario-owned DTOs** — never on private platform internals or vendor SDKs inside `application/`.
5. Has **scenario-owned contract tests** that assert behavior against the port, not against adapter internals.

### What a plugin is not

| Not a plugin | Why |
| --- | --- |
| Pipeline orchestration (`ProductIdentificationPipelineService`) | Scenario application spine; not swappable domain policy |
| Typed domain models (`ProductCandidate`, `SourceIdentityFact`, terminal enums) | Category C domain logic / value objects |
| Thresholds and feature flags (`ProductIdentificationPipelineConfiguration`, env loaders) | Category D configuration |
| PostgreSQL/Qdrant adapter classes | **Integration adapters** implementing scenario catalog ports + platform `VectorStore` / SQL session — replaceable, but not `intergrax.core.plugins` entries until registration phase |
| Proof evaluator, dataset builders, arena tooling | Proof/dataset lifecycle — outside canonical production pipeline |
| A local plugin registry or lifecycle manager | Forbidden duplicate of platform plugin/host mechanisms |

### When to create a scenario plugin

Create (or extract) a plugin when **all** apply:

- The concern varies by deployment, catalog, or policy (identity rules, fusion weights, verification strictness, clarification discriminators).
- Another implementation must be substitutable **without** changing `ProductIdentificationPipelineService` stage graph.
- The behavior is **domain specialization** of a reusable platform concern (retrieval channel orchestration, rank fusion, verification/decision, evidence collection).

### When to extend the platform instead

Extend platform **public contracts** when:

- Multiple scenarios need the same mechanism with the same lifecycle (multi-channel retrieval coordinator, generic product-stage trace projection, shared RRF utility).
- VPI today duplicates platform math or orchestration concepts (RRF helper vs `intergrax.rag.vectorstore.hybrid.reciprocal_rank_fusion`).
- Terminal authority must align with enterprise Decision System / `evidence_verification` without each scenario inventing a private envelope.

### When to keep logic in VPI only (no plugin surface)

Keep in **Category C** when logic is **pure product-identification semantics** with no stable cross-scenario contract yet: GTIN authority rules, MPN normalization, offer-grain candidate models, clarification discriminator catalogs, query normalization without NL interpreter.

---

## 2. Component Inventory (summary)

| Area | Responsibility | Key dependencies | Current contract | Target |
| --- | --- | --- | --- | --- |
| **query_understanding/** | Normalize typed / future NL input → `ProductIdentificationQuery` | Domain identifiers, extractors | `ProductIdentificationQueryInterpreter`, extractors (Protocol) | C until NL port promoted; optional B for `QueryInterpretationStrategy` |
| **retrieval/** | Four-channel orchestration, per-channel failure semantics | Catalog search ports | `MultiChannelRetrievalPort`, channel DTOs | B → `MultiChannelRetrievalCoordinator` (platform) + VPI orchestrator plugin |
| **fusion/** | Offer-level RRF fusion | Channel batches | `OfferCandidateFusionPort` / `OfferCandidateFusionStrategy` | B → fusion strategy plugin; platform RRF utility (A) |
| **identity/** | Fused offers → hypotheses + source facts | `SourceRecordFetchPort`, strategy | `ProductIdentityHypothesisPort` / `ProductIdentityHypothesisStrategy` | B → product identity hypothesis plugin |
| **identity_evaluation/** | Rerank, surface contradictions pre-verify | Identity DTOs | `IdentityHypothesisEvaluationPort` / `IdentityHypothesisRankingStrategy` | B → ranking plugin |
| **verification/** | Evidence-backed terminal decision | Policies, source facts | `ProductIdentificationVerificationPort`, `IdentityVerificationPolicy`, `ProductIdentificationDecisionPolicy` | B → map to platform decision/evidence contracts when integrated |
| **clarification/** | Discriminator selection when insufficient | Verification outcomes | `ClarificationRequirementSelectionPort`, selection/materiality policies | B → clarification policy plugin |
| **pipeline/** | Stage order, observation recording, config | All stage ports | `build_product_identification_pipeline(...)` DI | C (orchestrator) — not a plugin |
| **observability/** | Stage-ordered `ProductIdentificationObservation` | Sink, clock ports | `ProductIdentificationObservationSink` | B extension → platform diagnostic projection (A/B) |
| **application/contracts/** | Shared immutable DTOs | None (pure types) | Dataclasses | C |
| **application/ports/catalog_search.py** | Provider-neutral catalog ABI | Domain query/result types | Four search ports + `SourceRecordFetchPort` | B adapters; ports stay scenario until catalog ABI promotes |
| **integrations/** | Qdrant, PostgreSQL, embedding bootstrap | Platform `VectorStore`, `EmbeddingProvider`, registry | Adapter classes | A reuse + scenario adapters (not domain plugins) |
| **retrieval/composition.py** | Wire catalog adapters | storage_bootstrap adapters | Factory functions | Composition only |
| **composition/** | Bootstrap / materialization runtime | `intergrax.integrations.*`, proof_data | Orchestrator wiring | A platform + scenario orchestration |
| **contracts/** (package root) | Reserved by scaffold | — | Empty placeholder | D / docs only |

---

## 3. Component Classification Matrix

Taxonomy for this document: **A** Platform core · **B** Platform contract + scenario plugin · **C** Scenario domain logic · **D** Configuration.

| Component | Current Location | Classification | Target Boundary | Reason |
| --- | --- | --- | --- | --- |
| Scenario lab runtime envelope | `application/runtime_composition.py` | A | `build_scenario_lab_runtime` / `ScenarioRuntimeComposition` | Shared scenario execution shell |
| Proof data install / packages | `data_package/`, `intergrax.proof_data` | A | Platform proof_data API | Cross-proof infrastructure |
| Vector search data plane | `integrations/search_store/*`, Qdrant adapter | A + adapter | `VectorStore`, `VectorIndexIdentity` | Platform integration contract |
| Embedding execution | `integrations/embedding/*` | A + adapter | `EmbeddingProvider`, `bind_embedding_provider` | Platform RAG contract |
| PostgreSQL catalog sessions | `storage_bootstrap/adapters/postgresql/*` | A + adapter | Relational session / scenario ports | Infrastructure + port implementation |
| Multi-channel retrieval orchestrator | `application/retrieval/` | B (pending platform contract) | `MultiChannelRetrievalCoordinator` (proposed) ← `VpiMultiChannelRetrievalPlugin` | Cross-scenario pattern; today scenario `MultiChannelRetrievalPort` |
| Catalog channel adapters | `storage_bootstrap/adapters/*`, Qdrant vector adapter | B (adapter) | Scenario ports (`ExactIdentifierLookupPort`, …) | Swappable reference stack |
| Retrieval fusion (RRF) | `application/fusion/` | B | `RankFusionStrategy` (scenario port today) ← `OfferLevelRrfFusionPlugin`; reuse platform RRF math | Domain weights at offer grain; dedupe platform helper |
| Product identity hypothesis former | `application/identity/` | B | `ProductIdentityHypothesisPort` ← `ProductIdentityHypothesisPlugin` | Core VPI domain specialization |
| Identity hypothesis evaluation | `application/identity_evaluation/` | B | `IdentityHypothesisEvaluationPort` ← `ProductIdentityRankingPlugin` | Replaceable ranking/contradiction policy |
| Product identification verification | `application/verification/` | B | `ProductIdentificationVerificationPort` ← `ProductIdentificationVerificationPlugin`; optional `intergrax.contracts.evidence_verification` | Terminal authority + material constraints |
| Clarification discriminator selection | `application/clarification/` | B | `ClarificationRequirementSelectionPort` ← `ClarificationSelectionPlugin` | Business allow-list of attributes |
| Query understanding service | `application/query_understanding/` | C (+ future B) | Stay in VPI; optional plugin when NL interpreter lands | GTIN/MPN/constraint semantics |
| Pipeline orchestration | `application/pipeline/service.py` | C | `ProductIdentificationPipelineService` | Application spine — not replaceable plugin |
| Domain models & source facts | `application/domain/`, `application/contracts/source_identity_fact.py` | C | Immutable scenario types | Product identity semantics |
| Catalog normalization | `application/catalog/` | C | Normalization helpers | WDC/catalog-specific |
| Stage observations schema | `application/observability/` | B → platform hook | `ProductIdentificationObservation*` → diagnostic contributor | Reusable trace projection gap |
| Pipeline / fusion / retrieval config | `application/pipeline/contracts.py` (configuration dataclass), env loaders | D | Configuration only | Not plugins |
| Agent adapter skeleton | `application/agent.py` | C | Thin delegate to pipeline | Composition wiring only |
| Bootstrap orchestrator | `storage_bootstrap/`, `composition/bootstrap_runtime.py` | C (tooling) | Operator path | Not hot-path plugin |
| Arena / qualification / dataset | `arena/`, `qualification/`, `dataset/` | C | Proof and data lifecycle | Out of production plugin set |

---

## 4. Plugin Contract Design

**Naming rule:** Platform column lists **target** public contract (existing or evolution). **Current state:** implementations are in-repo Protocols under `application/*/strategy.py`, `application/pipeline/contracts.py`, and `application/ports/catalog_search.py` — logical plugins, not yet `platform_plugins` manifests.

| Plugin (target name) | Platform contract (target) | Input | Output | Replaceable |
| --- | --- | --- | --- | --- |
| `VpiMultiChannelRetrievalPlugin` | `MultiChannelRetrievalPort` → future `MultiChannelRetrievalCoordinator` | `MultiChannelRetrievalRequest` | `MultiChannelRetrievalResult` | YES — inject at `build_product_identification_pipeline(retrieval_service=...)` |
| `OfferLevelRrfFusionPlugin` | `OfferCandidateFusionPort` / `OfferCandidateFusionStrategy` | `OfferCandidateFusionRequest` | `FusedOfferCandidateCollection` | YES — `fusion_service=` |
| `ProductIdentityHypothesisPlugin` | `ProductIdentityHypothesisPort` / `ProductIdentityHypothesisStrategy` | `ProductIdentityHypothesisRequest` | `ProductIdentityHypothesisCollection` | YES — `identity_service=` |
| `ProductIdentityRankingPlugin` | `IdentityHypothesisEvaluationPort` / `IdentityHypothesisRankingStrategy` | `IdentityHypothesisEvaluationRequest` | `RankedIdentityHypothesisCollection` | YES — `identity_evaluation_service=` |
| `ProductIdentificationVerificationPlugin` | `ProductIdentificationVerificationPort`; optional `intergrax.contracts.evidence_verification` | `ProductIdentificationVerificationRequest` | `ProductIdentificationVerificationOutcome` | YES — `verification_service=` |
| `ClarificationSelectionPlugin` | `ClarificationRequirementSelectionPort` / `ClarificationRequirementSelectionStrategy` | `ClarificationSelectionRequest` | `ClarificationSelectionResult` | YES — `clarification_service=` |
| `ProductIdentificationQueryUnderstandingPlugin` (future) | `ProductIdentificationQueryInterpreter` | Raw query envelope | `ProductIdentificationQuery` | YES — when interpreter wired |
| `CatalogExactLookupAdapter` | `ExactIdentifierLookupPort` | `ExactIdentifierQuery` | `ExactIdentifierLookupResult` | YES — integration boundary |
| `CatalogVectorSearchAdapter` | `VectorCandidateSearchPort` + platform `VectorStore` | `VectorSearchQuery` | Vector candidate batch | YES — Qdrant vs future PgVector |
| `ProductIdentificationObservationSinkPlugin` | `ProductIdentificationObservationSink` → future platform diagnostic sink | `ProductIdentificationObservation` | void / persistence | YES — `observation_sink=` |

### Illustrative typed contract shape (design-only)

Plugins must use frozen dataclasses and Protocols — no `dict[str, object]` or `Any` in public boundaries.

```python
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True, slots=True)
class IdentityContext:
    """Scenario-owned; maps from ProductIdentityHypothesisRequest."""
    ...


@dataclass(frozen=True, slots=True)
class IdentityResult:
    """Scenario-owned; maps to ProductIdentityHypothesisCollection."""
    ...


class ProductIdentityHypothesisStrategy(Protocol):
    def form_hypotheses(self, request: IdentityContext) -> IdentityResult: ...
```

Today VPI already uses equivalent types under `application/identity/contracts.py`; promotion to platform would **rename/envelope**, not change pipeline semantics.

---

## 5. Dependency Direction

### Required (target)

```text
intergrax Platform Core (execution, integrations, RAG, applications baseline)
        ^
        | implements / uses public contracts only
intergrax Platform Contracts (VectorStore, EmbeddingProvider, scenario runtime, …)
        ^
        | implements specialization (injection)
VPI Scenario Plugins (fusion, identity, verification, …)
        ^
        | orchestrates via ports
VPI Application (ProductIdentificationPipelineService, domain models)
        ^
        | adapters only
VPI Integrations (PostgreSQL, Qdrant, embedding bootstrap)
```

### Forbidden

```text
VPI Application
 |
 +-- intergrax.runtime.nexus.* / private _shared internals (except allowed scenario baseline imports)
 +-- vendor SDKs inside application/pipeline or application/domain
 +-- parallel execution engine / plugin registry / observability spine
```

**Verified today:** `application/pipeline` core has **zero** `intergrax` imports; vendor code is confined to `integrations/` and `storage_bootstrap/adapters/` (see architecture conformance tests).

---

## 6. Replacement Capability

| Plugin / port | Swap A → B without changing pipeline / execution / platform core? | Evidence |
| --- | --- | --- |
| Multi-channel retrieval | YES | `MultiChannelRetrievalPort` mock in unit tests; `retrieval_service=` parameter |
| Offer fusion | YES | `fusion_service=`; alternate `OfferCandidateFusionStrategy` |
| Identity hypothesis | YES | `identity_service=`; strategy injection in `build_product_identity_hypothesis_service` |
| Identity evaluation | YES | `identity_evaluation_service=` |
| Verification | YES | `verification_service=`; policies composable inside service |
| Clarification | YES | `clarification_service=` |
| Observation sink | YES | `NoOpProductIdentificationObservationSink` vs future persistent sink |
| Catalog adapters | YES | `retrieval/composition.py` builds alternate port implementations |
| Pipeline stage order | NO (by design) | Changing order is application change — not plugin substitution |
| Scenario runtime / ReflexAgent shell | NO without platform scenario API change | Platform-owned envelope |

**Gap:** `VerifiedProductIdentificationAgent` is not yet delegating to `ProductIdentificationPipelineService` — replacement at **agent** boundary is incomplete; **pipeline** DI already satisfies plugin replacement for stage implementations.

---

## 7. Platform Evolution Opportunities

| Gap | Decision | Reason |
| --- | --- | --- |
| Generic multi-channel retrieval coordinator (peer channels + failure semantics) | Platform extension | EPUR/VPI-style scenarios; avoid N copies of orchestration |
| Shared RRF / rank-fusion math | Platform extension (utility) | VPI duplicates `reciprocal_rank_contribution` vs RAG hybrid helper |
| Product-stage observability vs `RetrievalTrace` / diagnostic spine | Platform extension | Project `ProductIdentificationObservation` without duplicate fields |
| Generic evidence graph for material identity checks | Platform extension (optional) | Reuse `evidence_verification` patterns; keep product rules in plugins |
| GTIN/MPN/hypothesis materiality rules | Scenario plugin | Domain-specific |
| `ProductIdentificationDecision` terminal enum | Scenario domain (+ optional Decision System mapping) | Outcome vocabulary is product-identification specific |
| `MetadataFilter` range / structured SQL parity | Platform extension | Structured catalog channel needs provider-neutral filters |
| Platform plugin manifest registration (`intergrax.core.plugins`) | Platform extension (hosting) | Physical packaging after contracts stabilize |
| Agent → pipeline wiring | Scenario composition | Not platform core |
| Proof evaluator + `PlatformProofEvidence` | Platform reuse | Proof infrastructure |

**Core platform modifications for this design:** **NONE** (all items are phased evolution or scenario work).

---

## 8. Anti Pattern Analysis

| Check | Result | Notes |
| --- | --- | --- |
| Duplicated platform capability | **FAIL** (contained) | Local RRF math; multi-channel orchestration outside `RetrievalService` — justified but should converge |
| Scenario framework creation | **PASS** | Ports + composition roots; no local plugin registry or execution engine |
| Hidden coupling | **PASS** (pipeline) / **WARN** (agent) | Pipeline DI clean; agent not wired to pipeline yet |
| Vendor leakage | **PASS** | Enforced by `test_vpi_architecture_conformance` import boundaries |
| Missing abstraction | **FAIL** (documented) | Decision System / `evidence_verification` not integrated; platform trace bridge missing |

---

## 9. Plugin Implementation Roadmap

Design task only — phases below are **implementation** work, out of scope for VPI-PLUGIN-BOUNDARY-DESIGN.

### Phase 1 — Contract alignment

- Map each VPI port to existing or proposed `intergrax.*` contract names in `SCENARIO_SPEC.md` / `PLATFORM_CAPABILITY_MAPPING.md`.
- Resolve RRF duplication (call shared platform helper or document exception).
- Document platform vs scenario DTO ownership for verification/evidence.

### Phase 2 — Plugin extraction

- Move default implementations behind explicit `*Plugin` modules (same behavior, clearer boundary).
- Keep `build_product_identification_pipeline` as the single pipeline composition root.
- Introduce query-understanding interpreter injection when NL path lands.

### Phase 3 — Platform registration

- Optional `platform_plugins` manifests when a second scenario shares the same port.
- Register adapters separately from domain policy plugins.

### Phase 4 — Contract tests

- One test module per port proving replacement (mock A vs mock B) without pipeline edits.
- Extend architecture conformance for plugin isolation imports.

### Phase 5 — Reference scenario validation

- Wire agent → pipeline → observation sink on lab runtime.
- Complete proof evaluator path; re-run Scenario Architecture Review checklists in `SCENARIO_SPEC.md`.

---

## 10. Related artifacts

| Artifact | Role |
| --- | --- |
| [PLATFORM_CAPABILITY_MAPPING.md](PLATFORM_CAPABILITY_MAPPING.md) | As-built platform reuse audit |
| [SCENARIO_SPEC.md](SCENARIO_SPEC.md) | Normative capability adoption & pluginability checklists |
| [docs/PLATFORM_PLUGINABILITY_PROOF.md](docs/PLATFORM_PLUGINABILITY_PROOF.md) | DI / replacement pointers (when populated) |
| `application/pipeline/composition.py` | Canonical pipeline DI |
| `application/runtime_composition.py` | Scenario lab runtime DI |

---

## Document control

| Field | Value |
| --- | --- |
| Authoring task | VPI-PLUGIN-BOUNDARY-DESIGN |
| Implements code | No |
| Supersedes | Nothing — complements PLATFORM_CAPABILITY_MAPPING |
| Next review trigger | Phase 1 contract alignment start or platform retrieval coordinator ADR |

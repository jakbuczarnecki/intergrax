# VPI — Platform Capability Mapping Audit

**Task ID:** VPI-PLATFORM-CAPABILITY-MAPPING-AUDIT  
**Scenario slug:** `verified_product_identification`  
**Document role:** Architecture decision record proving VPI extends Integrax through platform contracts and scenario-owned ports/adapters—not an isolated application with parallel platform mechanics.

**Code anchors (discovery revision):** `intergrax/` (Tier-0 contracts, integrations, RAG, applications baseline, proof_data, core plugins) · `platform_proofs/scenarios/verified_product_identification/` (application pipeline, integrations, storage_bootstrap, data_package, retrieval composition).

**Related:** [SCENARIO_SPEC.md](SCENARIO_SPEC.md) (Intergrax fit table) · [PRODUCTION_PIPELINE_AND_OBSERVABILITY.md](PRODUCTION_PIPELINE_AND_OBSERVABILITY.md) · [BOOTSTRAP_ARCHITECTURE.md](BOOTSTRAP_ARCHITECTURE.md) · [REFERENCE_PROVIDER_DECISION.md](REFERENCE_PROVIDER_DECISION.md)

---

## 1. Scenario Problem Definition

### Business problem

Industrial-scale product catalogs (here: 3.77M WDC offers) mix heterogeneous identifiers, incomplete attributes, and near-identical variants. Users describe parts in natural language or with partial codes. **Semantic top-k retrieval is not product identity.** Wrong variant selection causes procurement errors, RMAs, and downtime.

VPI must:

1. Generate candidates through **independent retrieval channels** (exact identifiers, lexical, structured attributes, dense vectors).
2. Fuse and rank for recall-quality ordering without treating fusion score as verification.
3. **Verify** material identity constraints against **traceable catalog source evidence** (`record_json` / `SourceIdentityFact`).
4. Return bounded outcomes: `VERIFIED`, `NO_MATCH`, `AMBIGUOUS`, `INSUFFICIENT_INFORMATION`, with optional clarification loop.

### Required capabilities

| Capability area | Requirement |
| --- | --- |
| **Query understanding** | Typed `ProductIdentificationQuery` (verification context + optional search text); future NL → typed bridge |
| **Retrieval** | Multi-channel orchestration with per-channel failure semantics; provider-neutral catalog search ports |
| **Fusion** | Offer-level RRF-style fusion with channel attribution |
| **Identity** | Hypothesis formation from fused candidates + immutable source facts |
| **Identity evaluation** | Reranking / contradiction surfacing before terminal verification |
| **Verification** | Evidence-backed terminal decision with abstention |
| **Clarification** | Discriminator selection when evidence insufficient |
| **Evidence / provenance** | Per-requirement support/contradiction/missing tied to `SourceRecordRef` |
| **Observability** | Stage-ordered `ProductIdentificationObservation` on production path |
| **Data / proof** | `proof_data` packages, storage bootstrap, future `PlatformProofEvidence` evaluator |
| **Runtime envelope** | Scenario lab entry via `scenario_runtime_baseline` + `ReflexAgent` (adapter skeleton) |

### Constraints

- **Provider neutrality:** `application/domain`, `application/pipeline`, and orchestration **must not** import `integrations/` or vendor SDKs (`test_vpi_architecture_conformance.py`).
- **No cluster_id as identity truth** in pipeline or observability.
- **Separation:** retrieval quality ≠ verification; fusion score ≠ confidence.
- **Status:** production pipeline (5C12) is implemented; agent adapter and proof evaluator wiring are **not** complete (`application/agent.py` raises `NotImplementedError`; README proof run N/A).

---

## 2. Platform Capability Adoption

Mandatory summary table (expanded detail in §3–§6).

| Capability | Platform Contract | VPI Implementation | Classification | Status |
| --- | --- | --- | --- | --- |
| **Retrieval** | `VectorStore`, `VectorStoreScope`, `MetadataFilter` (`intergrax.integrations.contracts.vector_store`); `EmbeddingProvider` (`intergrax.rag.embedding.contracts.embedding_provider`); PostgreSQL session (`intergrax.integrations.providers.relational_store.postgresql.session`); optional `RetrievalService` / `RetrieverRegistry` (`intergrax.rag.retrieval`) — **not used** for VPI multi-channel | Scenario: `MultiChannelRetrievalPort` / `MultiChannelRetrievalService` (`application/retrieval/`); ports `ExactIdentifierLookupPort`, `LexicalCandidateSearchPort`, `StructuredCandidateSearchPort`, `VectorCandidateSearchPort` (`application/ports/catalog_search.py`); adapters in `storage_bootstrap/adapters/*`, `integrations/search_store/qdrant_vector_candidate_search_adapter.py`; wiring `retrieval/composition.py` | **Reuse** (vector + embedding + SQL session) + **Scenario internal** (4-channel orchestration & catalog ports) | **PASS** (boundary correct); **GAP** vs generic `RetrievalService` |
| **Identity** | `intergrax.contracts.capability_catalog.identity` (generic capability identity keys—not product SKU semantics); no platform product-identity SPI | `ProductIdentityHypothesisPort`, `ProductIdentityHypothesisService`, `ProductIdentityHypothesisRequest` (`application/identity/`); evidence via `SourceRecordFetchPort` + `SourceIdentityFact` | **Scenario internal** (Category D); optional future **Scenario plugin** | **PASS** |
| **Verification** | `intergrax.contracts.evidence_verification` + Decision System lifecycle (`intergrax.contracts.decision.integration.*`) — **generic**, not wired to VPI outcomes | `ProductIdentificationVerificationPort`, `ProductIdentificationVerificationService`, `ProductIdentificationDecision` (`application/verification/`); policies in `decision_policy.py`, `direct_source_evidence.py` | **Scenario internal** (Category D); optional **Reuse** of decision envelope later | **PASS** (by design); Decision System **not integrated** |
| **Evidence** | `KnowledgeDocument` (`intergrax.knowledge.contracts.document`); `ExternalEffectEvidence` / execution evidence (`intergrax.contracts.execution_evidence.*`, `intergrax.contracts.external_operations.evidence`) — different domain | `SourceIdentityFact`, `HypothesisRejectionEvidence`, `VerifiedRequirementEvidence`, `ContradictedRequirementEvidence` (`application/contracts/`, `application/verification/`); bootstrap writes via `KnowledgeDocument` in `integrations/search_store/platform_bootstrap_adapter.py` | **Reuse** at index bootstrap only; **Scenario internal** for identity evidence graph | **PASS** |
| **Observability** | `RetrievalTrace` (RAG); application wiring via `wire_application_observability`, `wire_terminal_execution_diagnostics` (`intergrax.applications._shared.*`); `TraceEvent` / diagnostic assembly on scenario runtime | `ProductIdentificationObservationSink`, `ProductIdentificationObservation`, `ProductIdentificationObservationRecorder` (`application/observability/`); pipeline mapping `application/pipeline/observation_mapping.py` | **Scenario extension** (Category B trace schema); platform spine **available** via `build_scenario_lab_runtime` but **not** bound to pipeline sink today | **PASS** (scenario contract complete); **GAP** bridge to platform diagnostic spine |

### Additional platform surfaces actively reused (supporting)

| Surface | Platform contract / module | VPI usage |
| --- | --- | --- |
| Scenario runtime | `ScenarioRuntimeComposition`, `execute_scenario_task`, `build_scenario_lab_runtime` (`intergrax.applications._shared.scenario_runtime_baseline`, `scenario_runtime_profiles`) | `application/scenario.py`, `application/runtime_composition.py` |
| Agent authoring | `ReflexAgent`, `AgentStepContext`, `CapabilityMatchResult` (`intergrax.agents.authoring.*`, `intergrax.contracts.*`) | `application/agent.py` (skeleton only) |
| Vector index admin | `VectorIndexIdentity`, index administration contracts (`intergrax.integrations.contracts.vector_index_administration`) | Qdrant bootstrap, `qdrant_index_identity_resolver.py`, data pack composition |
| Proof data packages | `intergrax.proof_data` (`ProofDataPackageDescriptor`, checksums, install API) | `data_package/*` |
| Integration registry | `intergrax.integrations.registry.catalog`, `IntegrationCategory` | `integrations/embedding/bootstrap.py` |
| Embedding registry | `EmbeddingProfile`, `EmbeddingProviderExecutionConfig`, `bind_embedding_provider` | `application/config/*`, `integrations/embedding/intergrax_adapter.py` |

---

## 3. Classification Rules

### CATEGORY A — Existing Platform Capability (REUSE)

Platform contract is authoritative; VPI consumes without redefining semantics.

```text
VectorStore / EmbeddingProvider / PostgreSQL session / proof_data / scenario_runtime_baseline
        |
        v
VPI integrations & storage_bootstrap adapters (Qdrant, PostgreSQL, HF embedding)
```

**Examples in repo:** `QdrantVectorCandidateSearchAdapter` opens `open_qdrant_vector_store` and queries via `VectorStore`; `IntergraxEmbeddingBootstrapAdapter` calls `bind_embedding_provider`; `setup_data.py` / `data_package/install.py` use `intergrax.proof_data`.

### CATEGORY B — Platform Extension Through Contract

Capability is cross-scenario; VPI defines a **scenario trace or port** that could later align with or promote into platform contracts.

```text
ProductIdentificationObservation*  (scenario)
        |
        +--> future: projection into RetrievalTrace / diagnostic spine
```

**Decision:** **platform extension** (document only—no core change in this audit).

### CATEGORY C — Scenario Plugin

Domain-specific logic behind a **replaceable port**; could be packaged as `platform_plugins` entry points later (pattern: EPUR `erl_integration/plugins/*`), but VPI today uses **in-repo Protocol injection**, not `intergrax.core.plugins` registration.

```text
OfferCandidateFusionPort ──> OfferCandidateFusionStrategy (RRF)
ProductIdentityHypothesisPort ──> ProductIdentityHypothesisStrategy
IdentityHypothesisEvaluationPort ──> evaluation strategy
ProductIdentificationVerificationPort ──> verification_policy
ClarificationRequirementSelectionPort ──> selection_strategy
```

**Decision:** **scenario plugin** (logical SPI; physical location `application/*/composition.py`).

### CATEGORY D — Scenario Internal Logic

Pure product-identification semantics; stays in VPI.

- `application/domain/*` (candidates, identifiers, source refs)
- `application/query_understanding/*` (normalization; no NL parser yet)
- `application/catalog/*` (attribute normalization, search representation)
- Terminal outcome enums and clarification discriminator policy
- Dataset / arena / qualification tooling under `dataset/`, `arena/`, `qualification/`

---

## 4. Plugin Boundary Analysis

| Candidate | Why plugin (domain boundary) | Contract (VPI port / SPI) | Replacement possible |
| --- | --- | --- | --- |
| **Product identity hypothesis former** | Maps fused offers → hypotheses + source facts; product semantics | `ProductIdentityHypothesisPort` | **YES** — swap `build_product_identity_hypothesis_service` strategy |
| **Identity hypothesis evaluator** | Rerank/contradiction before verification | `IdentityHypothesisEvaluationPort` | **YES** |
| **Product identification verifier** | Material constraint check vs catalog evidence | `ProductIdentificationVerificationPort` | **YES** |
| **Clarification discriminator selector** | Business allow-list of distinguishing attributes | `ClarificationRequirementSelectionPort` | **YES** |
| **Offer candidate fusion** | Multi-channel RRF at offer grain | `OfferCandidateFusionPort` | **YES** |
| **Multi-channel retrieval orchestrator** | Exact+lexical+structured+vector policy | `MultiChannelRetrievalPort` | **YES** (in-process); not a platform plugin today |
| **Catalog search channel adapters** | Provider-specific SQL/Qdrant | `ExactIdentifierLookupPort`, `LexicalCandidateSearchPort`, `StructuredCandidateSearchPort`, `VectorCandidateSearchPort` | **YES** — reference PostgreSQL + Qdrant adapters |
| **Product evidence collector** | Fetch `record_json` + derive `SourceIdentityFact` | `SourceRecordFetchPort` + identity strategy | **YES** |
| **Observation sink** | Proof-grade stage trace persistence | `ProductIdentificationObservationSink` | **YES** — `NoOp` vs future file/DB sink |
| **Query understanding (future)** | NL → `ProductIdentificationQuery` | Not yet a port; pipeline uses typed query only | **YES** when `RAW_QUERY` path lands |

**Not scenario plugins (platform-owned):** `VectorStore` data plane, Qdrant index administration, embedding provider runtime, PostgreSQL session factory, scenario Nexus lab runtime assembly.

---

## 5. Architecture Anti-Pattern Review

| Check | Result | Evidence |
| --- | --- | --- |
| **Duplicated platform mechanism** | **FAIL** (minor, contained) | VPI implements `reciprocal_rank_contribution` in `application/fusion/contracts.py` instead of importing `intergrax.rag.vectorstore.hybrid.reciprocal_rank_fusion`. Multi-channel retrieval reimplements orchestration outside `RetrievalService` / `RetrieverRegistry` (justified by four independent catalog channels, but duplicates RAG orchestration concept). |
| **Private framework inside scenario** | **PASS** | No parallel plugin registry, execution engine, or vector-store ABI in VPI. Ports + composition roots (`application/pipeline/composition.py`, `retrieval/composition.py`) follow hexagonal style enforced by unit tests. |
| **Vendor coupling** | **PASS** (with declared reference stack) | `qdrant_client` / PostgreSQL access only under `integrations/` and `storage_bootstrap/adapters/`; `application/` imports are scenario-local except allowed platform types (agent, scenario runtime, embedding config). |
| **Direct platform internals dependency** | **PASS** | Pipeline core has **zero** `intergrax` imports. Only `application/agent.py` touches `intergrax.runtime.task.TaskContext`. No imports of `intergrax.runtime.nexus.*` or private `_shared` modules from domain/pipeline. |
| **Missing abstraction** | **FAIL** (documented gaps) | `VerifiedProductIdentificationAgent` not wired to `ProductIdentificationPipelineService`; platform Decision System and `evidence_verification` not used for terminal authority; generic RAG `RetrievalTrace` not emitted alongside scenario observations. |

---

## 6. Platform Evolution Opportunities

Recommendations only—**no implementation** in this audit.

| Gap | Classification | Recommendation |
| --- | --- | --- |
| No generic **multi-channel retrieval orchestrator** (exact + lexical + structured + vector as peer inputs) | Platform extension | Introduce optional `MultiChannelRetrievalCoordinator` contract in `intergrax.rag.retrieval` that accepts channel result lists; VPI could delegate policy while keeping catalog ports |
| **MetadataFilter** equality-only / no range queries | Platform extension | Extend `MetadataFilter` + provider SQL for numeric ranges if structured retrieval moves into vector ABI |
| **PgVector** dense-only / ANN index gaps | Platform adapter | Extend provider schema (HNSW/IVFFlat) per SCENARIO_SPEC gap table; VPI reference path uses Qdrant today |
| **Product-stage observability** vs `RetrievalTrace` | Platform extension | Define diagnostic contributor or trace projection hook so `ProductIdentificationObservation` can map to platform diagnostic spine without duplicating fields |
| **Terminal authority** for identification outcomes | Scenario plugin + optional reuse | Keep `ProductIdentificationDecision` in scenario; optionally map terminal envelope to Decision System lifecycle for enterprise audit parity (EPUR pattern) |
| **RRF helper duplication** | Platform adapter (utility) | VPI fusion should call `reciprocal_rank_fusion` or shared math helper to avoid drift |
| **Agent adapter** | Scenario internal | Thin `ReflexAgent` delegating to `ProductIdentificationPipelineService.run` — composition only |
| **Proof evaluator** | Reuse | Wire `PlatformProofEvidence` v3 + scenario evaluator consuming pipeline result + observation sink (per SCENARIO_SPEC § O) |
| **Platform plugin packaging** | Scenario plugin | If multiple scenarios share identity verification patterns, promote `ProductIdentificationVerificationPort` implementations to `platform_plugins` with manifest—**not justified yet** (single scenario) |

---

## 7. Core Platform Modifications

**NONE** required for current VPI architecture. All gaps are addressable via scenario adapters, optional future contracts, or provider extensions listed above.

---

## 8. Discovery Reference — Platform Contracts (Tier-0)

Non-exhaustive list verified in code discovery:

- **Execution / lifecycle:** `intergrax.contracts.execution_request`, `execution_phase`, `execution_identity`, `governed_execution_result`, `execution_interrupt`, `resilience_policy`
- **Agents / capabilities:** `intergrax.contracts.agent_step`, `agent_step_context`, `capability`, `intergrax.agents.agent_contract`
- **Applications:** `intergrax.applications.contracts.application_host`, `intergrax.applications._shared.scenario_runtime_baseline`
- **Integrations:** `intergrax.integrations.contracts.vector_store`, `vector_index_administration`, `vector_index_metadata`, `SearchProvider`, relational store configs
- **RAG:** `intergrax.rag.retrieval.retrieval_service`, `intergrax.rag.retrievers.registry.retriever_registry`, `intergrax.rag.vectorstore.hybrid.reciprocal_rank_fusion`, `intergrax.rag.embedding.contracts.embedding_provider`
- **Knowledge:** `intergrax.knowledge.contracts.document.KnowledgeDocument`
- **Evidence / decisions:** `intergrax.contracts.evidence_verification`, `intergrax.contracts.decision.integration.engine`, `intergrax.contracts.execution_evidence.receipt.ProofReceipt`
- **Enterprise reliability:** `intergrax.contracts.enterprise_reliability.*` (used by other scenarios; not VPI pipeline)
- **Plugins:** `intergrax.core.plugins.*`, `intergrax.applications.contracts.platform_plugin_evidence` (scenario runtime can load plugins; VPI does not register domain plugins)
- **Proof data:** `intergrax.proof_data`

---

## 9. Discovery Reference — VPI Modules Analyzed

| Area | Path | Role |
| --- | --- | --- |
| **application/contracts** | `application/contracts/` | Typed queries, failures, identification context, source facts |
| **application/pipeline** | `application/pipeline/` | `ProductIdentificationPipelineService`, stage orchestration |
| **application/retrieval** | `application/retrieval/` | `MultiChannelRetrievalService` |
| **application/fusion** | `application/fusion/` | Offer-level RRF fusion |
| **application/identity** | `application/identity/` | Hypothesis formation |
| **application/identity_evaluation** | `application/identity_evaluation/` | Reranking / contradiction |
| **application/verification** | `application/verification/` | Terminal decision |
| **application/clarification** | `application/clarification/` | Discriminator selection |
| **application/observability** | `application/observability/` | Stage observations |
| **application/query_understanding** | `application/query_understanding/` | Normalization (typed path) |
| **application/ports** | `application/ports/catalog_search.py` | Catalog search ABI |
| **retrieval** | `retrieval/composition.py` | Adapter wiring root |
| **integrations** | `integrations/search_store/*`, `integrations/embedding/*`, `integrations/catalog_store/*` | Platform contract adapters |
| **storage_bootstrap** | `storage_bootstrap/` | Provider-neutral bootstrap orchestration |
| **scripts** | `scripts/` | Operator/diagnostics entrypoints (out of pipeline hot path) |
| **application/agent + scenario** | `application/agent.py`, `application/scenario.py` | Platform scenario lab envelope |

---

## 10. Conclusion

VPI **is** structured as a platform extension:

1. **Infrastructure and portable ABIs** (vector store, embeddings, relational sessions, proof data, scenario runtime) come from `intergrax/`.
2. **Product identification semantics** live in scenario ports and domain models, injected at composition boundaries—not reimplemented as a second platform.
3. **Replaceability** is demonstrated by `build_product_identification_pipeline(...)` constructor injection and catalog port adapters.
4. Remaining risks are **integration completeness** (agent, proof evaluator, platform trace bridge)—not architectural isolation.

# VPI — Platform Capability Decision Review

**Task ID:** VPI-PLATFORM-CAPABILITY-DECISION-REVIEW  
**Scenario slug:** `verified_product_identification`  
**Document role:** Architecture decision record — target ownership split between Intergrax platform and VPI scenario plugins/domain logic.  
**Status:** Decision complete (no production code changes in this task).

**Sources:** [PLATFORM_PROOF_AUTHORING_GUIDE.md](../../PLATFORM_PROOF_AUTHORING_GUIDE.md) · [SCENARIO_STRUCTURE.md](../docs/SCENARIO_STRUCTURE.md) · [PLATFORM_CAPABILITY_MAPPING.md](PLATFORM_CAPABILITY_MAPPING.md) · [VPI_PLUGIN_BOUNDARY_DESIGN.md](VPI_PLUGIN_BOUNDARY_DESIGN.md) · VPI `application/`, `composition/`, `integrations/` · Tier-0 `intergrax/{core,retrieval,rag,evidence,observability,plugins,execution,contracts}`.

---

## 1. Executive Summary

### What was analyzed

End-to-end VPI production pipeline capabilities: multi-channel catalog retrieval, offer-level rank fusion, product identity hypothesis lifecycle, evidence-backed verification, stage observability, and their relationship to existing Intergrax platform contracts (`VectorStore`, `EmbeddingProvider`, `RetrievalService`, `reciprocal_rank_fusion`, `evidence_verification`, scenario runtime observability wiring, `proof_data`, integration registry).

### Decisions taken

| Theme | Decision |
| --- | --- |
| **Retrieval orchestration** | **Promote (phased):** generic multi-channel **coordinator** + per-channel failure semantics → platform; **catalog channel ports**, query shapes, and retrieval **policy** remain VPI/scenario. |
| **Rank fusion** | **Promote (utility + optional strategy port):** RRF math and generic N-list fusion primitive → platform (`intergrax.rag`); **offer-grain fusion strategy** (`OfferCandidateFusionStrategy`) remains scenario plugin. |
| **Identity resolution** | **Split:** platform may later host hypothesis/evidence **mechanisms**; **GTIN/MPN/variant semantics** stay VPI domain + plugins. |
| **Verification** | **Split:** platform **decision/evidence lifecycle** (`evidence_verification`, Decision System integration) for audit spine; **material identity rules** (`IdentityVerificationPolicy`) stay VPI plugin. |
| **Observability** | **Promote (projection hook):** platform diagnostic spine receives scenario stage traces; **VPI observation schema** stays scenario-owned until projection contract exists. |

### What was not changed

No plugin extraction, no module moves, no platform core patches, no execution engine or retrieval runtime changes, no dataset/proof pipeline edits. This document is the mandatory gate before any implementation.

---

## 2. Capability Ownership Matrix

| Capability | Current Owner | Target Owner | Decision | Reason |
| --- | --- | --- | --- | --- |
| Vector store / embedding execution | Platform (`VectorStore`, `EmbeddingProvider`) | Platform | **Keep (Reuse)** | Portable ABI; VPI adapters in `integrations/` |
| Single-path RAG retrieval (`RetrievalService`) | Platform | Platform | **Keep** | Document/knowledge RAG; not catalog 4-channel model |
| Multi-channel retrieval orchestrator | VPI (`MultiChannelRetrievalService`) | Platform coordinator + VPI policy/plugin | **Promote (phased)** | EPUR/VPI pattern; reuse across scenarios |
| Catalog search channel ports (exact/lexical/structured/vector) | VPI (`application/ports/catalog_search.py`) | VPI (scenario ABI) → optional promote if 2+ scenarios share | **Keep (scenario)** | Catalog-specific DTOs; swappable adapters |
| Channel adapter implementations (PostgreSQL, Qdrant) | VPI (`storage_bootstrap/`, `integrations/`) | VPI integration | **Keep** | Reference stack; not platform domain |
| RRF mathematical primitive | Platform (`reciprocal_rank_fusion`) + duplicate in VPI | Platform only | **Promote (dedupe)** | Anti-pattern: per-scenario RRF copies |
| Offer-level RRF fusion strategy | VPI (`OfferCandidateFusionStrategy`) | VPI plugin | **Keep (plugin)** | Offer grain, channel weights, domain ordering |
| Query understanding (typed normalization) | VPI | VPI domain | **Keep** | Product query semantics |
| Product identity hypothesis formation | VPI (`ProductIdentityHypothesisPort`) | VPI plugin | **Keep (plugin)** | Product semantics + source facts |
| Identity hypothesis evaluation / rerank | VPI (`IdentityHypothesisEvaluationPort`) | VPI plugin | **Keep (plugin)** | Pre-verify ranking policy |
| Source record fetch / evidence facts | VPI (`SourceRecordFetchPort`, `SourceIdentityFact`) | VPI (+ platform `KnowledgeDocument` at bootstrap) | **Keep (scenario)** | Identity evidence graph is product-specific |
| Terminal verification rules | VPI (`IdentityVerificationPolicy`) | VPI plugin | **Keep (plugin)** | “Is this the same product?” |
| Terminal decision assembly | VPI (`ProductIdentificationDecisionPolicy`) | VPI domain; optional map to Decision System | **Keep (+ optional reuse)** | Outcome vocabulary is VPI-specific |
| Evidence verification lifecycle (generic) | Platform (`intergrax.contracts.evidence_verification`) | Platform | **Reuse (not wired)** | Enterprise audit when integrated |
| Stage observations (`ProductIdentificationObservation*`) | VPI | VPI schema + platform projection | **Extend platform hook** | Application Observability Test |
| Scenario lab runtime / Nexus baseline | Platform (`build_scenario_lab_runtime`) | Platform | **Keep** | Shared envelope |
| Proof data packages | Platform (`intergrax.proof_data`) | Platform | **Keep** | Cross-proof infrastructure |
| Pipeline orchestration | VPI (`ProductIdentificationPipelineService`) | VPI application spine | **Keep** | Not a plugin |
| Plugin registry / host | Platform (`intergrax.core.plugins`) | Platform | **Keep** | No scenario-local registry |

---

## 3. Decision Criteria

Every row above was evaluated against:

| Criterion | Question |
| --- | --- |
| **Reuse** | Do other scenarios need the same mechanism (multi-channel retrieval, RRF math, diagnostic projection)? |
| **Ownership** | Who defines meaning — platform (mechanism) vs scenario (product/catalog semantics)? |
| **Stability** | Is the public contract stable enough to promote without churn? |
| **Complexity** | Should platform own lifecycle, monitoring, and configuration for this concern? |
| **Replaceability** | Can implementations swap behind a port without pipeline or core edits? |

---

## 4. Capability Deep Dive (Phase 2)

### 4.1 Retrieval Capability

**As-built:** VPI `MultiChannelRetrievalService` coordinates four independent channels via `ExactIdentifierLookupPort`, `LexicalCandidateSearchPort`, `StructuredCandidateSearchPort`, `VectorCandidateSearchPort`, with per-channel `SKIPPED` / success / failure semantics. Platform `RetrievalService` + `RetrieverRegistry` serve unified RAG document retrieval — **not** used for VPI’s peer-channel catalog model (documented in `PLATFORM_CAPABILITY_MAPPING.md` §2).

**Decision:** **Model A (target)** — not full replacement of VPI service by `RetrievalService`.

```text
Platform: MultiChannelRetrievalCoordinator (proposed)
    ├── channel execution envelope (status, timing hooks, optional RetrievalTrace bridge)
    └── accepts abstract channel invocations (scenario-supplied callables or port registry)

VPI: MultiChannelRetrievalPolicy / plugin
    ├── which channels run for a given ProductIdentificationQuery
    ├── limits, skip rules, failure tolerance
    └── wires scenario catalog ports at composition roots (retrieval/composition.py)
```

**Rationale:** Reuse (other catalog/search scenarios), lifecycle/monitoring at coordinator layer, configuration split (platform defaults + scenario policy). **Forbidden:** moving catalog query DTOs or WDC normalization into platform.

### 4.2 Rank Fusion Capability

**As-built:** `OfferCandidateFusionStrategy` implements offer-level RRF using local `reciprocal_rank_contribution` in `application/fusion/contracts.py`. Platform ships `intergrax.rag.vectorstore.hybrid.reciprocal_rank_fusion` for doc-id ranked lists.

**Decision:**

| Layer | Owner | Artifact |
| --- | --- | --- |
| Generic RRF / N ranked-list fusion | **Platform** | `reciprocal_rank_fusion` (existing); optional thin `intergrax.rag.retrieval.fusion` module for typed channel-agnostic fusion |
| Offer-grain fusion + channel attribution | **VPI plugin** | `OfferCandidateFusionPort` / `OfferCandidateFusionStrategy` |

**Rationale:** Eliminates Scenario A/B/C each owning RRF math; keeps fusion **policy** (offer key, channel ordering, limits) in VPI.

### 4.3 Identity Resolution Capability

| Concern | Target owner |
| --- | --- |
| Hypothesis collection lifecycle, immutable source refs as DTO pattern | Scenario today; **optional** platform “hypothesis envelope” only if second scenario shares shape |
| Evidence aggregation from catalog `record_json` | VPI `SourceRecordFetchPort` + strategies |
| GTIN authority, MPN rules, variant/materiality | **VPI domain + `ProductIdentityHypothesisPlugin`** |
| `cluster_id` as non-truth | VPI invariant (stay out of platform) |

**Decision:** **Do not** promote product identity semantics to platform. Promote only cross-scenario **mechanisms** after a second consumer exists.

### 4.4 Verification Capability

| Mechanism (platform candidate) | Policy (VPI plugin) |
| --- | --- |
| Evidence collection patterns, decision lifecycle, audit receipt | `IdentityVerificationPolicy.verify_hypothesis` |
| Optional `evidence_verification` + Decision System engine | `ProductIdentificationDecisionPolicy` terminal outcomes |

**Decision:** No standalone “Verification Engine” in platform for product identity. Use existing **Decision System** and `evidence_verification` as **optional enterprise envelope**; VPI keeps `ProductIdentificationVerificationPort` as the scenario SPI. Map terminal decisions in a later integration phase — not a new platform engine.

### 4.5 Observability Capability

**As-built:** `ProductIdentificationObservationSink` records stage-ordered observations on the production path. Platform provides `wire_application_observability`, `RetrievalTrace`, scenario lab runtime — **not** bound to VPI sink today.

**Decision:** Introduce (document-only) **platform diagnostic projection contributor** so VPI (and Execution, Agents, Decision, Proof) emit into one spine without duplicating fields. VPI retains schema ownership until projection contract is defined.

---

## 5. Final Architecture Model

```text
                 PLATFORM
        Generic Capabilities
 ┌──────────────────────────────────────┐
 │ Retrieval (RetrievalService, future   │
 │   MultiChannelRetrievalCoordinator)  │
 │ Rank fusion primitives (RRF)           │
 │ Evidence / decision contracts          │
 │ Observability wiring + diagnostic spine│
 │ Execution + plugin registry          │
 │ Integrations (VectorStore, SQL, …)   │
 └──────────────────────────────────────┘
                 CONTRACTS
        (public intergrax.* + scenario ports)
                 VPI
 ┌──────────────────────────────────────┐
 │ Product semantics & domain models    │
 │ Identity rules (hypothesis, evaluate)│
 │ Verification & clarification policies│
 │ Multi-channel retrieval policy       │
 │ Offer fusion strategy                │
 │ Scenario workflow (pipeline spine)   │
 │ Catalog port adapters (reference)    │
 └──────────────────────────────────────┘
```

**Dependency direction:** `integrations` → platform contracts; `application` → scenario ports/DTOs; composition roots wire plugins. No vendor SDKs in `application/pipeline` or `application/domain`.

---

## 6. Platform Extension Candidates

| Candidate | Priority | Why |
| --- | --- | --- |
| `MultiChannelRetrievalCoordinator` | **High** | Cross-scenario orchestration + channel failure semantics |
| Diagnostic spine / stage trace projection | **High** | Enterprise observability + Application Observability Test |
| RRF / rank-fusion utility consolidation | **Medium** | Remove VPI duplicate; prevent N-scenario drift |
| `MetadataFilter` / structured SQL parity | **Medium** | Structured catalog channel at scale |
| Generic hypothesis/evidence envelope | **Low** | Wait for second scenario consumer |
| `platform_plugins` manifest hosting | **Low** | After contracts stabilize |

---

## 7. Plugin Model

| Plugin | Contract | Owner | Replaceable |
| --- | --- | --- | --- |
| `VpiMultiChannelRetrievalPlugin` | `MultiChannelRetrievalPort` → future coordinator delegate | VPI | YES (`retrieval_service=`) |
| `OfferLevelRrfFusionPlugin` | `OfferCandidateFusionPort` / `OfferCandidateFusionStrategy` | VPI | YES (`fusion_service=`) |
| `ProductIdentityHypothesisPlugin` | `ProductIdentityHypothesisPort` | VPI | YES (`identity_service=`) |
| `ProductIdentityRankingPlugin` | `IdentityHypothesisEvaluationPort` | VPI | YES (`identity_evaluation_service=`) |
| `ProductIdentificationVerificationPlugin` | `ProductIdentificationVerificationPort` / `IdentityVerificationPolicy` | VPI | YES (`verification_service=`) |
| `ClarificationSelectionPlugin` | `ClarificationRequirementSelectionPort` | VPI | YES (`clarification_service=`) |
| `ProductIdentificationObservationSinkPlugin` | `ProductIdentificationObservationSink` → future platform sink | VPI | YES (`observation_sink=`) |
| Catalog channel adapters | `ExactIdentifierLookupPort`, … | VPI integrations | YES (`retrieval/composition.py`) |

**Note:** Logical plugins today are in-repo Protocol implementations; physical `platform_plugins` entries deferred until multi-scenario reuse justifies manifests.

---

## 8. Anti Patterns

| Problem | Result | Notes |
| --- | --- | --- |
| Scenario framework duplication | **PASS** | Ports + composition; no local execution engine or plugin registry |
| Platform leakage (product domain in core) | **PASS** | Pipeline core has zero `intergrax` imports; product types stay in scenario |
| Missing abstraction | **FAIL** (documented) | Coordinator, diagnostic bridge, Decision System wiring gaps — addressed via roadmap, not ad-hoc code |
| Vendor coupling | **PASS** | Qdrant/PostgreSQL confined to `integrations/` and `storage_bootstrap/` |
| Wrong ownership (RRF duplicated) | **FAIL** (contained) | VPI local `reciprocal_rank_contribution` vs platform `reciprocal_rank_fusion` — dedupe in Phase 1 |
| Per-scenario RRF engines | **FAIL** (preventive) | Decision: platform primitive + scenario strategy only |

---

## 9. Implementation Roadmap

### Phase 1 — Platform contracts (design → ADR → minimal API)

- ADR for `MultiChannelRetrievalCoordinator` shape (channel status, timing, optional trace).
- ADR for diagnostic projection contributor (VPI observations → platform spine).
- Consolidate RRF: VPI calls platform helper; no new fusion framework.
- Document Decision System mapping for `ProductIdentificationDecision` (optional).

**Out of scope for Phase 1:** moving VPI modules, changing `RetrievalService` behavior.

### Phase 2 — Reference plugins

- Extract/default implementations behind explicit `*Plugin` modules (behavior unchanged).
- Contract tests per port (mock A vs mock B) without pipeline edits.

### Phase 3 — Migration

- VPI orchestrator delegates channel envelope to platform coordinator; policy stays in scenario.
- Wire observation projection to lab runtime diagnostics.
- Optional Decision System envelope for terminal outcomes.

### Phase 4 — Scenario validation

- Agent → pipeline → sink on lab runtime.
- Proof evaluator + `PlatformProofEvidence`; re-run Scenario Architecture Review checklists in `SCENARIO_SPEC.md`.

---

## 10. Core Platform Modifications (this decision)

**NONE** required to honor ownership split. Phase 1+ work is explicit, gated evolution — not preemptive framework building.

---

## 11. GitHub Audit Checklist (post-merge verification)

Verify against repository `development` (or release branch):

- [ ] VPI `application/pipeline` still has no vendor imports and no forbidden platform internals.
- [ ] `MultiChannelRetrievalService` remains scenario-owned until coordinator contract lands.
- [ ] `intergrax.rag.vectorstore.hybrid.reciprocal_rank_fusion` exists and is the canonical RRF primitive.
- [ ] `PLATFORM_CAPABILITY_MAPPING.md` and this document agree on classification A/B/C/D.
- [ ] Composition roots: `application/pipeline/composition.py`, `retrieval/composition.py`, `composition/bootstrap_runtime.py`.
- [ ] No new scenario-local plugin registry or execution engine.
- [ ] Enterprise paths: `evidence_verification` and scenario runtime observability remain available but optional for VPI.

---

## Document control

| Field | Value |
| --- | --- |
| Authoring task | VPI-PLATFORM-CAPABILITY-DECISION-REVIEW |
| Implements code | No |
| Supersedes | Nothing — decision layer atop PLATFORM_CAPABILITY_MAPPING + VPI_PLUGIN_BOUNDARY_DESIGN |
| Next review trigger | Start of Phase 1 ADR for retrieval coordinator or diagnostic spine |

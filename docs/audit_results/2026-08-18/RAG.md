# RAG - Platform Audit

## Metadata

- **Campaign date:** 2026-08-18
- **Layer code:** RAG
- **Constituent domains:** RAG (retrieval · profiles · retriever ABI · GraphRAG wiring · observability)
- **Tier(s):** Tier-0 `intergrax/rag/` · Tier-1 touchpoints `intergrax/tools/providers/rag/` · `intergrax/runtime/nexus/context/`
- **audited_sha:** `81b344411596d4a4187193d97b20f610e21ca3ac`
- **Status:** COMPLETE
- **Auditor:** independent platform audit
- **Verdict:** FAIL
- **Counts:** 0 CRITICAL / 3 HIGH / 3 MEDIUM / 0 LOW
- **Operator decision:** all 6 ACCEPTED 2026-08-20
- **Architecture doc(s):**
  - `docs/project/architecture/RAG.md`
- **Plan doc(s):**
  - `docs/project/maintainers/plans/RAG.md`
- **Scope in:**
  - `RetrievalService` canonical Tier-0 retrieval authority and `VectorStoreScope` contract
  - `RetrievalRequest` resource and scope fields
  - `BaseRetrieverManager.retrieve()` → `List[RetrievalHit]` ABI vs `RetrievalService` candidate adaptation
  - `RagProfile` frozen configuration and `rag_profile_from_env()` parsing
  - `production_rag_profile()` vs `production_graph_rag_profile()` naming semantics
  - `validate_graph_rag_production_wiring()` GraphRAG readiness gate
  - retrieval telemetry tenant identity on spans and metrics
  - `rag.retrieve` Tool path and Nexus `ContextBuilder` scope enforcement (positive control)
  - RAG / Memory / Context Engineering responsibility separation (positive control)
  - publication fencing / `SourceOperation` scoped identity (positive control)
  - historical RAG-FINAL / RAG-PROD / RAG-LIVE qualification evidence (positive control - not re-audited as failures)
- **Scope out:**
  - remediation implementation
  - second RAG runtime or duplicate `RetrievalService`
  - universal re-qualification of all vector/graph providers beyond documented bounds
  - Memory or Context Engineering domain re-audit beyond RAG touchpoints
  - silent runtime clamping of invalid production configuration in scattered callers
- **Prior audit reference(s):** Protocol v2 [`INTEGRATIONS`](INTEGRATIONS.md) (`INTEGRATIONS-RUNTIME-BINDING-INTEGRITY` - coordinate RAG-05); historical RAG-FINAL / RAG-PROD / RAG-LIVE **Done** / **CLOSED** rows remain valid qualification facts
- **architecture_sync:** COMPLETE
- **plan_sync:** COMPLETE
- **post_sync_sha:** `bbe52a6355799929b27bc97802aa44acef7300c1`

## Executive summary

**Verdict: FAIL.** Six accepted findings (3 HIGH, 3 MEDIUM) show the canonical `RetrievalService` does not require authoritative `VectorStoreScope` before provider retrieval; `RetrievalService` retains a duck-typed legacy candidate path inconsistent with the canonical `RetrievalHit` ABI and reranker strictness; `RagProfile` / `RetrievalRequest` lack bounded resource-policy validation; `production_rag_profile()` is semantically misleading versus the durable GraphRAG preset; GraphRAG production validation can succeed without proving an actual Integration graph-store binding; and retrieval telemetry reads a non-existent `tenant_id` request field instead of `request.scope.tenant_id`. Positive controls: RAG / Memory / Context Engineering separation is sound; `VectorStoreScope` and `RetrievalHit` are strong native ABIs; `rag.retrieve` and Nexus `ContextBuilder` require tenant scope and detect conflicts; publication fencing has scoped identity and durable coordinator evidence; architecture honestly limits production qualification scope; live Qdrant/PgVector/Chroma/Neo4j evidence within documented bounds is not invalidated; no second RAG runtime is required. Protocol v2 residual contract defects are distinct from bounded qualification evidence - remediation is **PLANNED**, not implemented.

## Verdict

**FAIL** - 0 CRITICAL / 3 HIGH / 3 MEDIUM / 0 LOW

## Findings

### AUDIT-20260818-RAG-01

- **Severity:** HIGH
- **Category:** SECURITY / AUTHORITY BOUNDARY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** RAG-SCOPE-CONTRACT-INTEGRITY
- **Claim falsified:** Canonical production `RetrievalService` requires authoritative `VectorStoreScope` before any provider retrieval.
- **Substance:** `RetrievalRequest` defines `scope: VectorStoreScope | None = None`. `RetrievalService` only rejects an incapable retriever when a scope was supplied. When `request.scope` is `None`, the canonical Tier-0 service invokes the retriever without scope. The audited `rag.retrieve` Tool path and Nexus `ContextBuilder` independently require/resolve tenant scope, but the canonical `RetrievalService` contract itself does not preserve that invariant.
- **Evidence:**
  - `intergrax/rag/retrieval/retrieval_request.py` - `scope: VectorStoreScope | None = None`
  - `intergrax/rag/retrieval/retrieval_service.py` - scoped-capability check only when `request.scope is not None`; `retrieve_kwargs["scope"]` added only when scope present; unscoped `retrieve()` call otherwise
  - `intergrax/tools/providers/rag/service.py` - `tenant_scope_required` when tenant unresolved (positive contrast)
  - `intergrax/runtime/nexus/context/context_builder.py` - `tenant_scope_required` when tenant unresolved (positive contrast)
- **Confidence:** HIGH - direct code path; production callers may enforce scope while canonical service permits ambient unscoped retrieval.
- **Target invariant:** Canonical production retrieval authority must require an authoritative `VectorStoreScope` before provider retrieval. If unscoped evaluation/lab retrieval remains required, expose it through an explicitly non-production/test surface or typed execution mode rather than making absence of scope an ambient valid state. Do not create a second `RetrievalService`.

### AUDIT-20260818-RAG-02

- **Severity:** HIGH
- **Category:** CONTRACT / PROVENANCE INTEGRITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** RAG-SCOPE-CONTRACT-INTEGRITY
- **Claim falsified:** One canonical retriever result ABI - `RetrievalHit` - with provenance preserved through `RetrievalService` regardless of reranker configuration.
- **Substance:** `BaseRetrieverManager.retrieve()` canonically returns `List[RetrievalHit]`. `RetrievalHit` is immutable and carries a validated `KnowledgeDocument`, score/rank, scope, provenance and identity. `RetrievalService` nevertheless retains `_candidates_to_chunks(candidates: List[Any])`. For non-`RetrievalHit` objects it duck-types content/id/score/rank/metadata and constructs `RetrievalChunk` without the canonical `KnowledgeDocument` scope/provenance path. When reranking is active the service already requires every candidate to be `RetrievalHit`, creating inconsistent contract enforcement depending on reranker configuration.
- **Evidence:**
  - `intergrax/rag/retrievers/contracts/base_retriever_manager.py` - `retrieve()` → `List[RetrievalHit]`
  - `intergrax/rag/retrieval/retrieval_service.py` - `TypeError` when reranking and not all `RetrievalHit`; `_candidates_to_chunks()` duck-typing path when rerank disabled
  - `intergrax/rag/retrievers/contracts/base_retriever.py` - `RetrievalHit` immutable ABI
- **Confidence:** HIGH - rerank-on vs rerank-off provenance strictness divergence is explicit in code.
- **Target invariant:** One canonical retriever result ABI: `RetrievalHit`. All production/native `RetrieverManager` implementations must return `RetrievalHit` or fail contract validation. Remove/segregate the loose legacy candidate adapter; do not allow reranker enablement to determine provenance strictness. Clean-cut preferred - no real-user legacy requirement exists.

### AUDIT-20260818-RAG-03

- **Severity:** HIGH
- **Category:** CONFIGURATION / RESOURCE GOVERNANCE DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** RAG-CONFIGURATION-QUALIFICATION-INTEGRITY
- **Claim falsified:** `RagProfile` and `RetrievalRequest` are typed, fail-fast resource-policy contracts with explicit production-safe ranges and cross-field invariants.
- **Substance:** `RagProfile` is a frozen dataclass but does not validate value ranges for resource-sensitive fields including `prefetch_top_k`, `final_top_k`, `max_context_chars`, `graph_rag_hops` / `graph_rag_seed_top_k`, `agentic_max_iterations`, ingest size limits, and agentic latency/score controls. `rag_profile_from_env()` uses permissive integer/float parsing with no bounds. `RetrievalRequest` likewise allows `top_k` / `final_top_k` / `prefetch_k` without positive bounded validation. `RetrievalService` forwards resolved `prefetch_k` to retriever execution.
- **Evidence:**
  - `intergrax/rag/profiles/rag_profile.py` - frozen dataclass fields without range validators; `rag_profile_from_env()` permissive `_env_int` parsing
  - `intergrax/rag/retrieval/retrieval_request.py` - unbounded top-k fields
  - `intergrax/rag/retrieval/retrieval_service.py` - `prefetch_k` forwarded to `retrieve(top_k=prefetch_k)`
- **Confidence:** HIGH - no fail-fast bounds at profile/request construction; dangerous values can reach runtime retrieval.
- **Target invariant:** `RagProfile` and `RetrievalRequest` must be typed, fail-fast resource-policy contracts with explicit production-safe ranges and cross-field invariants (`prefetch >= final`, positive limits, bounded iterations/hops, finite thresholds). Invalid explicit env configuration must fail startup/config validation rather than silently produce dangerous or nonsensical runtime values. Do not silently clamp arbitrary production configuration in scattered runtime callers.

### AUDIT-20260818-RAG-04

- **Severity:** MEDIUM
- **Category:** API / CONFIGURATION SEMANTICS DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** RAG-CONFIGURATION-QUALIFICATION-INTEGRITY
- **Claim falsified:** Production-named public presets provide production-qualified posture.
- **Substance:** `production_rag_profile()` is named as a production preset, but its own docstring describes a harness/lab GraphRAG preset and it configures the in-memory graph store backend. The actual durable GraphRAG preset is `production_graph_rag_profile()`.
- **Evidence:**
  - `intergrax/rag/profiles/rag_profile.py` - `production_rag_profile()` docstring: harness/lab GraphRAG preset with in-memory graph store; `production_graph_rag_profile()` durable neo4j preset
- **Confidence:** HIGH - naming contradicts docstring and durable preset split.
- **Target invariant:** Production-named public presets must provide production-qualified posture. Rename the in-memory preset to an unambiguously lab/harness name or otherwise remove the semantic trap. Because clean-cut policy applies, do not preserve a misleading alias solely for compatibility unless a real consumer requires it.

### AUDIT-20260818-RAG-05

- **Severity:** MEDIUM
- **Category:** VALIDATION GAP / FAIL-LATE
- **Status at publication:** ACCEPTED
- **Remediation block:** RAG-CONFIGURATION-QUALIFICATION-INTEGRITY
- **Claim falsified:** Production GraphRAG readiness proves consistency of requested `RagProfile` backend, actual `IntegrationProfile` graph-store binding, and approved provider qualification.
- **Substance:** `validate_graph_rag_production_wiring()` rejects an unapproved profile backend and rejects an unapproved `graph_store_slug` only when a slug is supplied. Thus `profile.graph_store_backend = approved durable backend` (for example neo4j) with `graph_store_slug = None` returns validation success despite the function/doc contract saying a durable production graph backend is required.
- **Evidence:**
  - `intergrax/rag/profiles/rag_profile.py` - `validate_graph_rag_production_wiring()` early success when slug is `None` after backend slug check only
- **Confidence:** HIGH - configuration string alone can pass without Integration binding evidence.
- **Target invariant:** Production GraphRAG readiness must prove consistency of requested `RagProfile` backend, actual Integration graph-store binding, and approved provider qualification. A configuration string alone is not evidence of a bound durable provider. Coordinate with accepted `INTEGRATIONS-RUNTIME-BINDING-INTEGRITY` and do not invent a parallel integration resolver.

### AUDIT-20260818-RAG-06

- **Severity:** MEDIUM
- **Category:** OBSERVABILITY / AUDITABILITY DEFECT
- **Status at publication:** ACCEPTED
- **Remediation block:** RAG-OBSERVABILITY-IDENTITY
- **Claim falsified:** Scoped retrieval telemetry uses canonical execution scope tenant identity.
- **Substance:** `RetrievalRequest` has no `tenant_id` field; canonical tenant identity is carried in `request.scope.tenant_id`. `RetrievalService` spans and retrieval metrics attempt to read `attribute_access.optional(request, "tenant_id", None)`. That field does not exist, so scoped retrieval can be emitted as no tenant and `_record_retrieval_metrics` falls back to `_platform`.
- **Evidence:**
  - `intergrax/rag/retrieval/retrieval_request.py` - no `tenant_id` field; scope carries tenant
  - `intergrax/rag/retrieval/retrieval_service.py` - span attributes and `_record_retrieval_metrics(tenant_id=attribute_access.optional(request, "tenant_id", None))`
- **Confidence:** HIGH - telemetry reads wrong identity surface.
- **Target invariant:** Observability identity must come from the exact canonical execution/retrieval scope. Scoped retrieval telemetry uses `request.scope.tenant_id`, with an explicit non-tenant label only for intentionally unscoped lab/evaluation execution. Do not duplicate tenant identity as another independently writable request field.

## Positive controls / falsification log

| Control | Result |
|---------|--------|
| RAG / Memory / Context Engineering responsibility separation | NOT falsified - sound |
| `VectorStoreScope` typed, immutable; validates tenant/routing scope | NOT falsified |
| Reserved tenant/namespace/workspace routing fields cannot be tunneled through normal `MetadataFilter` conditions | NOT falsified |
| `RetrievalHit` strong native provider-neutral ABI | NOT falsified |
| Canonical `rag.retrieve` tool path requires tenant scope and detects conflicts | NOT falsified |
| Nexus `ContextBuilder` requires tenant scope and detects request/config/session conflicts | NOT falsified |
| Publication fencing / `SourceOperation` coordination - scoped identity, generations, CAS-backed durable coordinator | NOT falsified |
| Architecture limits production qualification and public proof scope honestly | NOT falsified |
| Findings do not invalidate live Qdrant/PgVector/Chroma/Neo4j qualification evidence within documented bounds | NOT falsified |
| No second RAG runtime required | NOT falsified |

## Qualification evidence vs Protocol v2 residual defects

Bounded qualification evidence (RAG-PROD-13, RAG-LIVE-15A–15E, RAG-FINAL harness gates, enterprise handoff limits) remains valid historical and environment-specific proof within documented bounds. Protocol v2 findings are **residual contract, configuration, and observability defects** on the canonical native path - they do not retract live provider qualification rows or claim universal backend SLO. Remediation must close contract gaps without erasing qualification artifacts.

## Root-cause remediation grouping

### RAG-SCOPE-CONTRACT-INTEGRITY - fail-closed scoped retrieval and one native result ABI

**Findings:** `AUDIT-20260818-RAG-01`, `AUDIT-20260818-RAG-02`

One fail-closed scoped retrieval boundary on canonical `RetrievalService` and one provenance-preserving `RetrievalHit` → `RetrievalChunk` path without ambient duck-typed legacy candidates.

### RAG-CONFIGURATION-QUALIFICATION-INTEGRITY - bounded resource policy and GraphRAG qualification binding

**Findings:** `AUDIT-20260818-RAG-03`, `AUDIT-20260818-RAG-04`, `AUDIT-20260818-RAG-05`

Bounded `RagProfile` / `RetrievalRequest` contracts, unambiguous production vs harness preset naming, and GraphRAG production validation that binds profile intent to actual Integration graph-store binding. Coordinate RAG-05 with `INTEGRATIONS-RUNTIME-BINDING-INTEGRITY`.

### RAG-OBSERVABILITY-IDENTITY - telemetry from canonical retrieval scope

**Findings:** `AUDIT-20260818-RAG-06`

Tenant and audit telemetry derives from `request.scope.tenant_id` (or explicit lab/unscoped label), not a duplicate writable request field.

## Evidence limitations / scope limitations

- Evidence bound exclusively to `audited_sha` `81b344411596d4a4187193d97b20f610e21ca3ac`; current `development` HEAD was not re-audited beyond persistence sync.
- Tests are supporting evidence, not standalone proof of production qualification at full provider scale.
- Remediation not performed in this task.
- Historical RAG plan **Done** rows and qualification handoffs remain valid - not rewritten.

## Open questions / blocked items

- 01: exact lab/evaluation execution mode surface for intentionally unscoped retrieval - operator decision deferred to remediation.
- 05: binding proof coordinates with INTEGRATIONS-3B / `INTEGRATIONS-RUNTIME-BINDING-INTEGRITY` - no parallel resolver.
- No operator-disputed findings.

## Operator acceptance

- **Date:** 2026-08-20
- **Accepted findings:** all 6 (`AUDIT-20260818-RAG-01` … `AUDIT-20260818-RAG-06`)
- **Deferred:** none
- **Disputed:** none
- **Rejected:** none
- **Withdrawn:** none

## No-remediation statement

This artifact persists accepted audit observations, architecture target invariants, and planned remediation blocks only. **No production source, test, CI, or script changes were made.** No finding is marked IMPLEMENTED, VERIFIED, or CLOSED.

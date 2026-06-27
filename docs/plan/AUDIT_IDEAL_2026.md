# Ideal Architecture Gap Register — Post-L3 Audit (2026-06-09)

**Architecture target:** [`guides/IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §0–§26  
**Platform canon (scope):** [`architecture/PLATFORM_FOUNDATION.md`](../architecture/PLATFORM_FOUNDATION.md) · cross-domain register (no 1:1 pair)  
**Audit map:** [`guides/INTEGRAX_HARNESS_AUDIT_MAP.md`](../guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §8 (32 layers)  
**Baseline:** **32/32 L3** (`scripts/gates/harness_maturity_report.py`, IDEAL-L3 W2 Done)  
**Hub:** [`plan/PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) · Band **2az** · queue **§6.1au**  
**Debt register:** [`guides/ARCHITECTURE_DEBT_REGISTER.md`](../guides/ARCHITECTURE_DEBT_REGISTER.md)  
**Status:** **AUDIT-IDEAL complete** (2026-06-18) — **90/90 Done** · **0 Deferred §6.3** · **0 Planned**

> **Scope:** Close gaps between **L3 Production Harness OS** and **full ideal architecture** (modular, elastic, complete). Harness infrastructure only unless row is explicitly **Deferred §6.3** (product).

---

## Delivery rule

One **AUDIT-IDEAL-\*** ID per PR → update this register + affected domain plan row → `pytest -m gate` + relevant depth gate green.

---

## Phase waves

| Wave | Focus | Priority |
|------|-------|----------|
| **W0** | Register + domain plan sync + hub cross-ref | — |
| **W1** | P0 harness depth (memory org, ECP doc sync, registry durable) | P0 |
| **W2** | P1 L4 + elasticity (AHI evidence, reasoning, shadow eval, DX HTTP) | P1 |
| **W3** | P2 completeness (modality live, cost forecast, prompt UI, swarm templates) | P2 |
| **W4** | Band 3 product (Tier-3 daemon, dashboard, K.1/K.2) | **Done** |

---

## Master register (32 layers → tasks)

| ID | Layer | AUDIT § | Gap (vs IDEAL) | Priority | Domain plan | Status |
|----|-------|---------|----------------|----------|-------------|--------|
| AUDIT-IDEAL-1.1 | Strategic Harness Model | 1 | Operationalize quarterly strategy review (IDEAL-1.2 process, not docs-only) | P2 | `PLATFORM_FOUNDATION` | **Done** |
| AUDIT-IDEAL-1.2 | Strategic Harness Model | 1 | Architecture health metrics as live signals (modularity, debt index) | P2 | `PLATFORM_FOUNDATION` | **Done** |
| AUDIT-IDEAL-2.1 | Tier boundaries | 2 | Continuous tier-boundary gate maintenance (no drift) | P3 | `PLATFORM_FOUNDATION` | **Done** (gates exist) |
| AUDIT-IDEAL-3.1 | Task intake | 3 | Canonical `TaskEnvelope` type consolidation (`Task` + `RuntimeRequest` alias) | P1 | `ORCHESTRATION` · `TIER3` | **Done** |
| AUDIT-IDEAL-3.2 | Task intake | 3 | Product host intake parity (streaming + durable async index default) | P2 | `TIER3_APPLICATION_ENVIRONMENT` | **Done** |
| AUDIT-IDEAL-4.1 | Identity & trust | 4 | Cryptographic signing / audit-protect for critical actions | P2 | `UNIFIED_EXECUTION_RUNTIME` | **Done** |
| AUDIT-IDEAL-4.2 | Identity & trust | 4 | Hard tenant storage isolation (Postgres multi-tenant RFC → ship) | P1 | `UNIFIED_EXECUTION_RUNTIME` | **Done** |
| AUDIT-IDEAL-5.1 | Policy & governance | 5 | Pre-output policy hooks on all LLM response paths | P1 | `UNIFIED_EXECUTION_RUNTIME` | **Done** |
| AUDIT-IDEAL-5.2 | Policy & governance | 5 | Compliance profile templates per regulated domain class | P2 | `UNIFIED_EXECUTION_RUNTIME` | **Done** |
| AUDIT-IDEAL-5.3 | Policy & governance | 5 | Governance health dashboard (GOV-PROD.1) | P4 | `OBSERVABILITY` | **Done** |
| AUDIT-IDEAL-6.1 | LLM adapters | 6 | Structured output validation on 100% reference + certified agent paths | P1 | `LLM_ADAPTERS` | **Done** |
| AUDIT-IDEAL-6.2 | LLM adapters | 6 | Live cost/latency/quality model routing (AHI integration prod path) | P2 | `LLM_ADAPTERS` · `ADAPTIVE_HARNESS_INTELLIGENCE` | **Done** — M-LLM-X.5.3 · `check_live_model_routing_wiring.py` |
| AUDIT-IDEAL-6.3 | LLM adapters | 6 | Central `ModelCatalog` + unified context window resolution | P0 | `LLM_ADAPTERS` | **Done** — M-LLM-X.1.7 · `CatalogCapabilityAdapter` |
| AUDIT-IDEAL-6.4 | LLM adapters | 6 | Tokenizer-consistent context preflight (adapter path) | P0 | `LLM_ADAPTERS` · `CONTEXT_ENGINEERING` | **Done** — M-LLM-X.3.2 · `count_message_tokens(adapter=)` |
| AUDIT-IDEAL-6.5 | LLM adapters | 6 | Profile failover chain on retriable provider errors | P1 | `LLM_ADAPTERS` · `RELIABILITY_FAILURE_AND_HITL` | **Done** — M-LLM-X.4.1–4.4 |
| AUDIT-IDEAL-6.6 | LLM adapters | 6 | ACP `StepLLMRouter` backed by `LLMAdapter` (single DX) | P1 | `LLM_ADAPTERS` · `NEXUS_EXECUTION_FLOW` | **Done** — M-LLM-X.5.4 |
| AUDIT-IDEAL-6.7 | LLM adapters | 6 | Developer `USAGE.md` + startup validation | P2 | `LLM_ADAPTERS` · `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** — M-LLM-X.7.2 · `check_llm_profile_runtime.py` + doctor |
| AUDIT-IDEAL-7.1 | Cognition | 7 | Ship `ReasoningProfile` contract + environment wire | P1 | `REASONING_AND_COGNITION` | **Done** |
| AUDIT-IDEAL-7.2 | Cognition | 7 | Complete `allow_dynamic_replan` runtime path | P1 | `REASONING_AND_COGNITION` | **Done** |
| AUDIT-IDEAL-7.3 | Cognition | 7 | Reasoning failure taxonomy on all planner kinds | P2 | `REASONING_AND_COGNITION` | **Done** |
| AUDIT-IDEAL-8.1 | Execution runtime | 8 | Long-running workflow resume E2E on product hosts | P2 | `NEXUS_EXECUTION_FLOW` | **Done** |
| AUDIT-IDEAL-8.2 | Execution runtime | 8 | Checkpoint introspection API for ops (beyond lab) | P2 | `NEXUS_EXECUTION_FLOW` | **Done** |
| AUDIT-IDEAL-9.1 | Orchestration | 9 | Production queue adapter (beyond SQLite scaffold) | P1 | `ORCHESTRATION` | **Done** |
| AUDIT-IDEAL-9.2 | Orchestration | 9 | Swarm + peer-to-peer coordination graph templates | P2 | `ORCHESTRATION` | **Done** |
| AUDIT-IDEAL-9.3 | Orchestration | 9 | Dynamic execution strategy selection (L4 hook) | P2 | `ORCHESTRATION` · `ADAPTIVE_HARNESS_INTELLIGENCE` | **Done** |
| AUDIT-IDEAL-10.1 | Subagents | 10 | Evaluator-loop standard node in product graph specs | P2 | `NEXUS_EXECUTION_FLOW` | **Done** |
| AUDIT-IDEAL-10.2 | Subagents | 10 | Budget delegation enforcement on all delegation paths | P2 | `NEXUS_EXECUTION_FLOW` | **Done** |
| AUDIT-IDEAL-11.1 | Tools | 11 | Sandboxed execution for code / side-effectful tools | P1 | `TOOLS` | **Done** |
| AUDIT-IDEAL-11.2 | Tools | 11 | MCP / function-schema export for shipped tool catalog | P2 | `TOOLS` | **Done** |
| AUDIT-IDEAL-11.3 | Tools | 11 | Oversized-tool lint enforcement in CI (adoption sweep) | P2 | `TOOLS` | **Done** |
| AUDIT-IDEAL-12.1 | Skills | 12 | LangGraph-compatible skill pack import path | P2 | `SKILLS` | **Done** |
| AUDIT-IDEAL-12.2 | Skills | 12 | Dynamic skill selection L4 hook (AHI) | P2 | `SKILLS` · `ADAPTIVE_HARNESS_INTELLIGENCE` | **Done** |
| AUDIT-IDEAL-13.1 | Integrations | 13 | Integration marketplace catalog + trust scoring | P3 | `INTEGRATIONS` | **Done** |
| AUDIT-IDEAL-13.2 | Integrations | 13 | Catalog hot-reload without host restart | P3 | `INTEGRATIONS` | **Done** |
| AUDIT-IDEAL-14.1 | RAG | 14 | Graph RAG as default production retrieval profile | P1 | `RAG` · `MEMORY` | **Done** |
| AUDIT-IDEAL-14.2 | RAG | 14 | Retrieval poisoning defense live on product hosts | P1 | `MEMORY` | **Done** |
| AUDIT-IDEAL-14.3 | RAG | 14 | Wire `RagProfile.query_expansion` to retrieval path | P0 | `RAG` | **Done** |
| AUDIT-IDEAL-14.4 | RAG | 14 | Dual-index + hierarchical retriever default bootstrap | P1 | `RAG` | **Done** — M-RAG.24 · `check_rag_hierarchical_bootstrap.py` |
| AUDIT-IDEAL-14.5 | RAG | 14 | Retrieval poisoning defense on `rag.retrieve` catalog path | P1 | `RAG` · `UNIFIED_EXECUTION_RUNTIME` | **Done** — M-RAG.25 · `check_rag_catalog_poisoning_defense.py` |
| AUDIT-IDEAL-14.6 | RAG | 14 | Large-corpus async ingest (stream / job orchestration) | P1 | `RAG` | **Done** |
| AUDIT-IDEAL-14.7 | RAG | 14 | OpenTelemetry spans on RAG retrieve + ingest hot path | P2 | `RAG` · `OBSERVABILITY` | **Done** |
| AUDIT-IDEAL-15.1 | Memory | 15 | Org memory 2.5 (organizational LTM scope) | **P0** | `MEMORY` | **Done** |
| AUDIT-IDEAL-15.2 | Memory | 15 | Episodic / semantic / procedural memory taxonomy (`MemoryKind` uplift) | P1 | `MEMORY` | **Done** |
| AUDIT-IDEAL-15.3 | Memory | 15 | Entity graph memory ship (beyond RFC — MEM-DEPTH-5.1) | P2 | `MEMORY` | **Done** |
| AUDIT-IDEAL-16.1 | Context | 16 | Online context drift monitoring + alerts | P1 | `MEMORY` | **Done** |
| AUDIT-IDEAL-16.2 | Context | 16 | Semantic compression strategy in production profiles | P2 | `MEMORY` | **Done** |
| AUDIT-IDEAL-17.1 | Prompt registry | 17 | Prompt approval workflow (beyond registry metadata) | P2 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-17.2 | Prompt registry | 17 | Prompt diff / compare API for all managed prompts | P2 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-18.1 | Agent assembly | 18 | `ModalityProfile` mandatory on certified agents | P1 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-18.2 | Agent assembly | 18 | Cross-host agent reuse certification test suite | P2 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-19.1 | Registry | 19 | Durable cross-host registry snapshot store (DEBT-19-01) | **P0** | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-19.2 | Registry | 19 | Capability negotiation at runtime resolve | P2 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-20.1 | Capability graph | 20 | Product CI blast-radius check on tool/skill changes | P1 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-20.2 | Capability graph | 20 | Policy change impact visualization CLI | P2 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-21.1 | Observability | 21 | Causal diagnostics beyond trace bridge (ops tooling) | P2 | `OBSERVABILITY` | **Done** |
| AUDIT-IDEAL-21.2 | Observability | 21 | Quality / governance / cost health dashboard contracts | P2 | `OBSERVABILITY` | **Done** |
| AUDIT-IDEAL-21.3 | Observability | 21 | Unified product observability dashboard | P4 | `OBSERVABILITY` | **Done** |
| AUDIT-IDEAL-22.1 | Reliability | 22 | Compensation flows on product side-effect paths | P1 | `RELIABILITY_FAILURE_AND_HITL` | **Done** |
| AUDIT-IDEAL-22.2 | Reliability | 22 | Partial results contract on all reference hosts | P2 | `RELIABILITY_FAILURE_AND_HITL` | **Done** |
| AUDIT-IDEAL-23.1 | Security | 23 | Immutable multi-region security audit trail | P2 | `UNIFIED_EXECUTION_RUNTIME` | **Done** |
| AUDIT-IDEAL-23.2 | Security | 23 | Retrieval poisoning + tool injection live on product hosts | P1 | `UNIFIED_EXECUTION_RUNTIME` | **Done** |
| AUDIT-IDEAL-24.1 | Cost | 24 | Cost forecasting from historical run patterns | P2 | `UNIFIED_EXECUTION_RUNTIME` | **Done** |
| AUDIT-IDEAL-24.2 | Cost | 24 | Automated cost optimization recommendations (AHI) | P2 | `UNIFIED_EXECUTION_RUNTIME` · `ADAPTIVE_HARNESS_INTELLIGENCE` | **Done** |
| AUDIT-IDEAL-24.3 | Cost | 24 | CPU/memory/concurrency quotas with tenant fairness | P2 | `UNIFIED_EXECUTION_RUNTIME` · `ELASTIC_CAPACITY_AND_SCALING` | **Done** |
| AUDIT-IDEAL-25.1 | Evaluation | 25 | Shadow eval path automation (DEBT-25-01) | P1 | `CRITIC_VERIFICATION` | **Done** |
| AUDIT-IDEAL-25.2 | Evaluation | 25 | Human review sample queue (beyond CLI) | P2 | `CRITIC_VERIFICATION` | **Done** |
| AUDIT-IDEAL-25.3 | Evaluation | 25 | Context/RAG eval blocking product release CI | P1 | `CRITIC_VERIFICATION` | **Done** |
| AUDIT-IDEAL-26.1 | CI / gates | 26 | Architecture-boundary chaos job in weekly CI | P2 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** |
| AUDIT-IDEAL-26.2 | CI / gates | 26 | Simulation tests for multi-agent contention | P2 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** |
| AUDIT-IDEAL-27.1 | DX | 27 | Trace Explorer interactive UI (beyond lab APIs) | P2 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** |
| AUDIT-IDEAL-27.2 | DX | 27 | Replay environment HTTP API on product hosts | P1 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** |
| AUDIT-IDEAL-27.3 | DX | 27 | Agent simulator on product hosts (not CLI-only) | P2 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** |
| AUDIT-IDEAL-27.4 | DX | 27 | Visual builder / graph editor (Phase 2 UI) | P3 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** |
| AUDIT-IDEAL-28.1 | Tier-3 hosts | 28 | Durable async queue default beyond SQLite (DEBT-28-01) | P1 | `TIER3_APPLICATION_ENVIRONMENT` | **Done** |
| AUDIT-IDEAL-28.2 | Tier-3 hosts | 28 | Queue worker scaffold-default (`INCLUDE_QUEUE_WORKER`) | P1 | `TIER3_APPLICATION_ENVIRONMENT` | **Done** |
| AUDIT-IDEAL-28.3 | Tier-3 hosts | 28 | LKW hybrid daemon (CFG-14) | P4 | `TIER3_APPLICATION_ENVIRONMENT` | **Done** |
| AUDIT-IDEAL-28.4 | Tier-3 hosts | 28 | Business agents K.1/K.2 certification + deploy | P4 | `TIER3_APPLICATION_ENVIRONMENT` | **Done** |
| AUDIT-IDEAL-29.1 | Modality | 29 | Live Triton / HF Inference endpoints (replace placeholders) | P1 | `MODALITY` | **Done** |
| AUDIT-IDEAL-29.2 | Modality | 29 | Plane C vision inference E2E on product worker pools | P2 | `MODALITY` | **Done** |
| AUDIT-IDEAL-30.1 | Ops / SLO | 30 | Sync `architecture/ELASTIC_CAPACITY_AND_SCALING.md` §22 after ECP-DEPTH | **P0** | `ELASTIC_CAPACITY_AND_SCALING` | **Done** |
| AUDIT-IDEAL-30.2 | Ops / SLO | 30 | Real deploy SLO window evidence (`W_OPS_RELEASE_CYCLES>=2` prod) | P1 | `OBSERVABILITY` · `EXPERIMENTATION` | **Done** |
| AUDIT-IDEAL-30.3 | Ops / SLO | 30 | On-call ownership model for production components | P2 | `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | **Done** |
| AUDIT-IDEAL-30.4 | Ops / SLO | 30 | Celery/K8s production-scale adapters (beyond stub/beta) | P2 | `ELASTIC_CAPACITY_AND_SCALING` | **Done** |
| AUDIT-IDEAL-31.1 | Agent lifecycle | 31 | Owner/on-call mandatory on all certified agents | P1 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-31.2 | Agent lifecycle | 31 | Evaluation results required before production promotion (enforce) | P1 | `AGENT_CONTRACTS_AND_ASSEMBLY` | **Done** |
| AUDIT-IDEAL-32.1 | Doc governance | 32 | Living architecture debt burn-down tied to milestones | P2 | `PLATFORM_FOUNDATION` | **Done** |
| AUDIT-IDEAL-32.2 | Doc governance | 32 | Scorecard auto-sync on plan row change (extend IDEAL-32.3) | P2 | `PLATFORM_FOUNDATION` | **Done** |
| AUDIT-IDEAL-AHI.1 | Adaptive Harness L4 | 25 | 30-day L4 closed-loop evidence on ≥3 golden scenarios (real deploy) | P1 | `ADAPTIVE_HARNESS_INTELLIGENCE` | **Done** |
| AUDIT-IDEAL-AHI.2 | Adaptive Harness L4 | 25 | Bounded policy learning without governance drift | P2 | `ADAPTIVE_HARNESS_INTELLIGENCE` | **Done** |
| AUDIT-IDEAL-AHI.3 | Adaptive Harness L4 | 25 | Capability marketplace readiness (trust, certification, billing) | P3 | `ADAPTIVE_HARNESS_INTELLIGENCE` | **Done** |

---

## Domain routing

| Domain plan | AUDIT-IDEAL IDs |
|-------------|-----------------|
| [`PLATFORM_FOUNDATION.md`](PLATFORM_FOUNDATION.md) | 1.1, 1.2, 2.1, 32.1, 32.2 |
| [`ORCHESTRATION.md`](ORCHESTRATION.md) | 3.1, 9.1, 9.2, 9.3 |
| [`UNIFIED_EXECUTION_RUNTIME.md`](UNIFIED_EXECUTION_RUNTIME.md) | 4.1, 4.2, 5.1, 5.2, 23.1, 23.2, 24.1, 24.2, 24.3 |
| [`LLM_ADAPTERS.md`](LLM_ADAPTERS.md) | 6.1, 6.2, 6.3, 6.4, 6.5, 6.6, 6.7 |
| [`REASONING_AND_COGNITION.md`](REASONING_AND_COGNITION.md) | 7.1, 7.2, 7.3 |
| [`NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) | 8.1, 8.2, 10.1, 10.2 |
| [`TOOLS.md`](TOOLS.md) | 11.1, 11.2, 11.3 |
| [`SKILLS.md`](SKILLS.md) | 12.1, 12.2 |
| [`INTEGRATIONS.md`](INTEGRATIONS.md) | 13.1, 13.2 |
| [`RAG.md`](RAG.md) | 14.1 (shared), 14.3–14.7 |
| [`MEMORY.md`](MEMORY.md) | 14.1, 14.2, 15.1, 15.2, 15.3, 16.1, 16.2 |
| [`AGENT_CONTRACTS_AND_ASSEMBLY.md`](AGENT_CONTRACTS_AND_ASSEMBLY.md) | 17.1, 17.2, 18.1, 18.2, 19.1, 19.2, 20.1, 20.2, 31.1, 31.2 |
| [`OBSERVABILITY.md`](OBSERVABILITY.md) | 5.3, 21.1, 21.2, 21.3, 30.2 |
| [`RELIABILITY_FAILURE_AND_HITL.md`](RELIABILITY_FAILURE_AND_HITL.md) | 22.1, 22.2 |
| [`CRITIC_VERIFICATION.md`](CRITIC_VERIFICATION.md) | 25.1, 25.2, 25.3 |
| [`EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md`](EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE.md) | 26.1, 26.2, 27.1, 27.2, 27.3, 27.4, 30.2, 30.3 |
| [`TIER3_APPLICATION_ENVIRONMENT.md`](TIER3_APPLICATION_ENVIRONMENT.md) | 3.2, 28.1, 28.2, 28.3, 28.4 |
| [`MODALITY.md`](MODALITY.md) | 29.1, 29.2 |
| [`ELASTIC_CAPACITY_AND_SCALING.md`](ELASTIC_CAPACITY_AND_SCALING.md) | 24.3, 30.1, 30.4 |
| [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md) | 6.2, 9.3, 12.2, 24.2, AHI.1, AHI.2, AHI.3 |

---

## Completion criteria (phase AUDIT-IDEAL)

Phase **AUDIT-IDEAL** closes incrementally:

1. All **P0** rows **Done** with gate evidence.
2. **P1** rows ≥ **80% Done** or explicitly deferred with §6.3 / debt register entry.
3. `uv run pytest -m gate -q` green.
4. `scripts/gates/harness_maturity_report.py` remains **32/32 L3+**.
5. Domain plan rows synced (no orphan AUDIT-IDEAL IDs).

**ADR policy:** New ADR only when contract changes; depth-only gates → **no ADR needed** per row.

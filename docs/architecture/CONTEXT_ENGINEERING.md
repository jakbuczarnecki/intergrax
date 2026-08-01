# Context Engineering

**Status:** Canonical architecture (domain pair 1:1)  
**Hub:** [`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
**Plan (1:1):** [`plan/CONTEXT_ENGINEERING.md`](../plan/CONTEXT_ENGINEERING.md)  
**Target:** [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §16  
**Audit layer:** 16 (Context Engineering)  
**Audit instruction:** [`audit/CONTEXT_ENGINEERING.md`](../audit/CONTEXT_ENGINEERING.md)  
**ADR:** [`ADR-CTX-001`](../adr/entries/2026-06-12/ADR-CTX-001.md) · [`ADR-MEM-001`](../adr/entries/2026-06-08/ADR-MEM-001.md) (Context Compiler budget semantics)  
**Related:** [`architecture/MEMORY.md`](MEMORY.md) (stores + lifecycle) · [`architecture/UNIFIED_CONTEXT_LIFECYCLE.md`](UNIFIED_CONTEXT_LIFECYCLE.md) (single budget authority + lifecycle) · [`architecture/RAG.md`](RAG.md) (retrieval) · [`architecture/TOOLS.md`](TOOLS.md) (tool outputs) · [`architecture/NEXUS_EXECUTION_FLOW.md`](NEXUS_EXECUTION_FLOW.md) (turn narrative) · [`architecture/OBSERVABILITY.md`](OBSERVABILITY.md) (event spine) · [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix L
**Implementation (as-built):** `intergrax/context/` · `intergrax/runtime/nexus/context/` · `intergrax/runtime/architecture/context_engineering.py` · `intergrax/contracts/context_assembly.py` · `applications/_shared/context_*`  
**Last architecture pass:** 2026-06-17 — **Full Harness LC** (re-validates iteration III); CE-LLM-X doc sync

---

## Cursor read scope (token budget)

**Do not read this entire file in one session** (CONTEXT_ENGINEERING canon).

- **Implement / audit default:** context assembly engine + scoring (§1–§7). Extended §8+: [`satellites/CONTEXT_ENGINEERING_extended_depth.md`](satellites/CONTEXT_ENGINEERING_extended_depth.md).
- **Use** table of contents below — `Read` with offset/limit per §.
- **Plan hub:** [`plan/CONTEXT_ENGINEERING.md`](../plan/CONTEXT_ENGINEERING.md) (scoped §6 only).
- **Audit slice:** [`guides/audit_slices/CONTEXT_ENGINEERING.md`](../guides/audit_slices/CONTEXT_ENGINEERING.md).
- **Max reads:** at most **one** file >5k tokens per session unless RESUME cites more.

---


## Architecture satellites (read on demand)

Large § blocks moved out of the architecture hub to reduce Cursor context use.
Load **only** the satellite matching your task or cited §.

| Satellite | Contents |
|-----------|----------|
| [`satellites/CONTEXT_ENGINEERING_extended_depth.md`](satellites/CONTEXT_ENGINEERING_extended_depth.md) | extended depth |

> **Cursor context budget:** read hub read-scope block + **at most one** satellite per session.

## 1. Purpose

Context Engineering (CE) is the **Tier-1 Harness engine** that decides **what information reaches the LLM** at a given execution step: which fragments to include, in what order, under what token budget, with full provenance and observability.

CE is **not** a memory store. It **consumes** outputs from:

| Source domain | What CE receives |
|---------------|------------------|
| [`MEMORY`](MEMORY.md) | Session history, LTM search hits, task KV reads (via providers) |
| [`RAG`](RAG.md) | Retrieved chunks + citations |
| [`TOOLS`](TOOLS.md) | Tool result blocks, workspace reads |
| [`ORCHESTRATION`](ORCHESTRATION.md) | Graph prior outputs, shared task context, delegation summaries |
| [`UNIFIED_EXECUTION_RUNTIME`](UNIFIED_EXECUTION_RUNTIME.md) | Policy overlays, system instructions, guardrail context |
| [`REASONING_AND_COGNITION`](REASONING_AND_COGNITION.md) | Optional `objective` / plan slice for step-aware ranking |
| [`MODALITY`](MODALITY.md) | Attachment / media summaries within budget |

**Rules:**

- Tier-2 agents MUST NOT hand-assemble production prompts from unbounded history.
- Tier-3 configures CE via `ContextProfile` and optional **plugin registration** — not agent imports of Nexus internals.
- Vendor-specific retrieval stays in Tier-0 (RAG); CE orchestrates **injection into the LLM window**.

```text
Tier-3 ContextProfile + ContextEnginePreset + context_plugins[]
  → context_runtime_bridge / context_wiring
  → ContextEngine.assemble(ContextAssemblyRequest)
  → messages_for_llm + AssembledContext (provenance, budget diag)
  → CoreLLMStep / Agent.run step
```

---

## 2. Production readiness verdict (2026-06-12, post CE-EXT)

| Question | Answer |
|----------|--------|
| Is CE **production-ready** as a **budgeted assembly spine**? | **Yes — L3+ engine / L3 control plane** (UAEP/ACP hybrid compile paths remain; see §8.3) |
| Is **`ContextCompiler` on the production hot path**? | **Yes (ACP)** — `StepLLMRouter` + `compile_service` before LLM; **Yes (graph)** when `ContextEngine` + `llm_adapter` wired; **Partial (UAEP)** — session `build_context` not yet full `assemble()` |
| Is there a **unified plugin catalog**? | **Yes** — `intergrax/context/` + `bootstrap_context_catalog()` + `BuiltinContextPlugin` (13 providers) |
| Is **step-aware** assembly implemented? | **Yes (ACP/graph events)** — `step_kind` / `step_index` on `ContextAssemblyRequest` + `context_assembly.v2`; ranker boosts by step |
| Is **workspace/codebase** context production-grade? | **Yes (MVP)** — workspace provider + FORMAT merge + orchestrator on graph codebase preset |
| Observability on assembly path? | **L3** — unified `CONTEXT_ASSEMBLED` v2 (CE-3.11); `CONTEXT_CANDIDATE_*` on engine assemble when `event_bus` wired (CE-9.1); OTel span shim + `check_context_otel_span_registry.py` |
| Can authors register custom providers without forking Nexus? | **Yes** — `register_context_plugin()` + `context_plugin_ids` on `ContextProfile` |

**Remaining:** deferred CE-9.5/9.6, CE-10.4–10.5, CE-12.1–12.3 — see §16.

---

## 3. Maturity score (audit map L0–L4)

| Dimension | Score | Notes |
|-----------|-------|-------|
| Control plane (`ContextProfile`, bridges, wiring) | **L3** | CTX + `context_presets.py` + `check_context_engine_wiring.py` |
| Global budget / never-overflow | **L3** | `ContextCompiler` on ACP LLM path + graph engine assemble; UAEP session path hybrid; preflight uses `adapter.count_messages_tokens` (M-LLM-X.3 / CE-LLM-X) |
| Provenance + assembly events | **L3** | `CONTEXT_ASSEMBLED` v2 with `engine_id` + `step_kind`; graph + UAEP aligned (CE-3.11) |
| Quality scoring (relevance/freshness/confidence) | **L3** | `DefaultContextRanker` + `evaluate_context_engineering()` gate (CE-10.1) |
| Plugin extensibility | **L3** | Catalog + FORMAT merge + **CE-PROV-WIRE** live collect for all §8.4 builtins (handle-gated where noted) |
| Step-aware selection | **L3** | ACP `AgentStepContext` + ranker table; graph uses `node.capability` as `step_kind` |
| Codebase-scale preset | **L3** | `CodebaseContextEngine` + workspace provider; 1k-file gate test |
| Interactive multi-hop context loop | **L2** | `ContextOrchestrator` on codebase preset only (CE-8) |
| OTel on compile hot path | **L2.5** | Span registry + engine shim; full OTel SDK wiring optional |
| Regression / drift gates | **L2.5** | `context_regression_benchmark.py`; preset baselines deferred (CE-10.4) |

**Overall:** **L3+ engine / L3 control plane** — CE-EXT S0–S12 complete; **L4** adaptive ranking deferred to [`ADAPTIVE_HARNESS_INTELLIGENCE.md`](ADAPTIVE_HARNESS_INTELLIGENCE.md).

---

## 4. Design principles

| Principle | Meaning in Intergrax |
|-----------|----------------------|
| **Compiler, not concatenation** | CE runs a deterministic pipeline with recorded stages — not string joins in agent code |
| **Budget-first** | Global input token budget derived from `llm_adapter.context_window_tokens` minus output reserve |
| **Step-scoped** | Assembly is keyed by **execution step** (UAEP step index or graph node + phase) — **Done** (CE-4) |
| **Source-agnostic plugins** | Each fragment has a `ContextSourceProvider`; Memory is one provider among many |
| **Provenance everywhere** | Every included/excluded fragment traceable to `source_id`, `source_type`, `degradation_step` |
| **Policy-governed** | `BEFORE_CONTEXT_BUILD` hooks + `pre_context_policy_audit` — no silent policy bypass |
| **Observable by default** | Events, structured logs, spans on collect/rank/budget/format — not optional debug |
| **Environment-driven** | `ContextProfile` on `ApplicationEnvironmentProfile` — Tier-3 owns presets |
| **Fail-safe degradation** | [`DegradationLadder`](MEMORY.md) order normative — never silent overflow |
| **Agents stay dumb about window** | Agents declare *needs* (`required_context_sources` on contract — future); CE satisfies |

---

## 5. Domain boundaries

| Concern | Owner | CE role |
|---------|-------|---------|
| Persist facts | MEMORY | Read via `MemoryContextProvider` |
| Retrieve documents | RAG | Read via `RagContextProvider` |
| Execute actions | TOOLS | Read tool output blocks via `ToolOutputContextProvider` |
| Graph priors | ORCHESTRATION | `ContextManager` / `GraphPriorContextProvider` |
| Prompt assets | AGENT_CONTRACTS (Prompt Registry) | `SystemInstructionsProvider` |
| Policy text | UAEP | `PolicyOverlayProvider` |
| **Compose LLM window** | **CONTEXT_ENGINEERING** | **Owns entire read-path orchestration** |

**Anti-pattern:** documenting CE inside MEMORY Layer C long-term — Layer C spec lives **here**; MEMORY links to this doc for read-path.

---

## 6. Tier placement

```text
Tier-0  intergrax/context/              contracts, plugins, ranker, dedup, orchestrator (CE-1 — shipped)
Tier-1  intergrax/runtime/nexus/context/  DefaultNexusContextEngine, ContextManager, ContextCompiler
Tier-3  applications/_shared/context_*      profile bridges, presets, wiring
```

| Component | Tier | Rationale |
|-----------|------|-----------|
| `ContextEngine`, `DefaultNexusContextEngine`, `CodebaseContextEngine` | 1 | Nexus turn-critical (CE-3 — shipped) |
| `ContextCompiler`, `DegradationLadder` | 1 | Budget allocator on ACP + engine assemble paths |
| `ContextSourceProvider` Protocol | 0 / providers | `intergrax/context/providers/` + app entry points |
| `context/quality.py` (scoring) | 0 | Shared types; shim in `context_engineering.py` |
| `ContextProfile` | 3 contract | Environment composition root |

---

## 7. Core domain model

### 7.1 Context fragment lifecycle

```text
ContextAssemblyRequest
  → COLLECT   (providers emit ContextFragment candidates)
  → NORMALIZE (schema_version, dedup keys, content_hash)
  → SCORE     (relevance, freshness, confidence → composite)
  → FILTER    (thresholds, policy, poisoning rules)
  → RANK      (mandatory first, then score desc)
  → BUDGET    (token allocation + degradation ladder)
  → COMPRESS  (summarize tiers, semantic compression flag)
  → FORMAT    (ChatMessage[] or AgentContextBundle.message)
  → VALIDATE  (preflight never-overflow + citation requirements)
  → EMIT      (events, spans, trace records)
```

### 7.2 Primary types (Tier-0 contracts — CE-1 shipped)

```python
# intergrax/context/contracts.py

class ContextFragmentSource(str, Enum):
    TASK_MESSAGE = "task_message"
    SYSTEM_INSTRUCTIONS = "system_instructions"
    SESSION_HISTORY = "session_history"
    SESSION_HISTORY_SEMANTIC = "session_history_semantic"  # MEM-VEC-2.4 — episodic vector recall hits
    LONGTERM_MEMORY = "longterm_memory"
    RAG = "rag"
    WEBSEARCH = "websearch"
    TOOL_OUTPUT = "tool_output"
    GRAPH_PRIOR = "graph_prior"
    SHARED_CONTEXT = "shared_context"
    ATTACHMENT = "attachment"
    POLICY_OVERLAY = "policy_overlay"
    WORKSPACE = "workspace"
    CUSTOM = "custom"

@dataclass(frozen=True)
class ContextFragment:
    fragment_id: str
    source: ContextFragmentSource
    source_id: str              # stable id for provenance
    content: str
    token_estimate: int
    relevance_score: float      # 0..1
    freshness_score: float
    confidence_score: float
    mandatory: bool             # never drop unless hard trim
    metadata: dict[str, Any]    # citations, path, line_range, tool_call_id
    content_hash: str           # dedup

@dataclass(frozen=True)
class ContextAssemblyRequest:
  """What CE needs to assemble context for ONE model call."""
    trace_id: str
    run_id: str
    task_id: str
    tenant_id: str
    # Step identity (CE-4)
    assembly_scope: Literal["uaep_turn", "graph_node", "delegation_child", "acp_step"]
    step_index: int | None
    graph_node_id: str | None
    step_kind: str | None         # e.g. plan | tool_call | synthesize | explore
    objective: str                # current sub-goal (may differ from full task message)
    # Policy
    decision_profile: ContextDecisionProfile
    budget_policy: ContextBudgetPolicy
    assembly_options: TaskContextAssemblyOptions
    # Capability hints
    required_sources: frozenset[ContextFragmentSource]
    excluded_sources: frozenset[ContextFragmentSource]
    # Runtime handles (injected by Nexus — not serialized)
    runtime_config: RuntimeConfig  # via context var / engine ctx — not in logs

@dataclass(frozen=True)
class AssembledContext:
    messages: list[ChatMessage]
    fragments_included: list[ContextFragment]
    fragments_excluded: list[tuple[ContextFragment, str]]  # reason
    provenance: list[ContextProvenance]
    total_tokens: int
    budget_tokens: int
    degradation_steps: tuple[str, ...]
    schema_version: str = "assembled_context.v1"
```

### 7.3 As-built types (today)

| Type | Module | Status |
|------|--------|--------|
| `ContextCandidate` | `context_compiler_models.py` | **Shipped** — message-index based |
| `AgentContextBundle` | `context_manager.py` | **Shipped** — graph path |
| `ContextProvenance` | `context_models.py` | **Shipped** |
| `ContextChunkSignal` | `context_engineering.py` | **Shipped** — quality eval only |
| `ContextAssemblyRequest` | `intergrax/context/contracts.py` | **Shipped** — step fields populated on ACP/graph (CE-4) |
| `ContextFragment` | `intergrax/context/contracts.py` | **Shipped** |
| `ContextPluginRegistry` | `intergrax/context/registry.py` | **Shipped** |
| `AgentContextHints` | `intergrax/contracts/agent_context_hints.py` | **Shipped** (CE-5.1) |

---

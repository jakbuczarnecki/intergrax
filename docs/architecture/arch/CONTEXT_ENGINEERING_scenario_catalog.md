# CONTEXT_ENGINEERING — §12+ scenarios & control

**Parent hub:** [`CONTEXT_ENGINEERING.md`](../CONTEXT_ENGINEERING.md)

## 12. Harness integration

### 12.1 Control plane (Tier-3)

```text
ApplicationEnvironmentProfile
  └── context_profile
        ├── assembly_options → TaskContextAssemblyOptions
        ├── budget_policy → ContextBudgetPolicy
        ├── decision → ContextDecisionProfile
        ├── engine_preset / engine_ref / context_plugin_ids
        └── enable_rag / enable_websearch / compression flags

materialize_runtime_config()
  └── context_runtime_bridge.py
        → RuntimeConfig.context_budget_policy
        → task_context_assembly_options
        → context_decision_profile

wire_application_environment()
  └── context_wiring.py
        → resolve_context_manager_from_environment()
        → resolve_context_engine_from_environment()
        → resolve_context_orchestrator_from_environment()  # codebase preset only

build_nexus_loop_from_environment()
  └── NexusLoop(context_manager=...)  # optional context_engine override (CE-3.5)
```

### 12.2 Hooks

| HookPoint | When | CE behaviour |
|-----------|------|--------------|
| `BEFORE_CONTEXT_BUILD` | Before `assemble()` | Middleware may MODIFY runtime_state; BLOCK stops assembly |
| `AFTER_CONTEXT_BUILD` | After `assemble()` | Audit / enrich diagnostics |

Registered in `hook_registry.py` parity map · UAEP `run_before` / `run_after`.

### 12.3 Policy

| Check | Module |
|-------|--------|
| Pre-context policy gate | `runtime/policy/context_assembly_policy.py` (`run_pre_context_policy_gate`) |
| Static wiring audit | `pre_context_policy_audit.py` (CI marker scan) |
| Retrieval poisoning | RAG security profile — fragments filtered before inject |
| Secret redaction | `ApplicationSecurityProfile` on trace payloads |

### 12.4 Runtime events

| Event | Phase | Payload schema | Storage | Path |
|-------|-------|----------------|---------|------|
| `CONTEXT_ASSEMBLED` | `CONTEXT_BUILDING` | `context_assembly.v2` (preferred) | TraceStore | Graph + UAEP (CE-3.11, CE-4.5) |
| `CONTEXT_TRIMMED` | `CONTEXT_BUILDING` | `context_assembly.v2` | TraceStore | Graph |
| `CONTEXT_BUILT` | `CONTEXT_BUILDING` | `context_assembly.v1` (alias registry) | TraceStore | **Deprecated** — use `CONTEXT_ASSEMBLED` |
| `CONTEXT_CANDIDATE_COLLECTED` | `CONTEXT_BUILDING` | `context_candidate.v1` | TraceStore | `DefaultNexusContextEngine` when `event_bus` in provider handles (CE-9.1) |
| `CONTEXT_CANDIDATE_DROPPED` | `CONTEXT_BUILDING` | `context_candidate.v1` | TraceStore | Dedup phase + counters (`context_counters.py`) |
| `CONTEXT_VALIDATION_FAILED` | `CONTEXT_BUILDING` | `validation.v1` | TraceStore | Policy/validator failures before `assemble()` raises |

Payload fields (assembled):

```yaml
trace_id, run_id, task_id, tenant_id
assembly_scope, step_index, graph_node_id, step_kind
total_tokens, budget_tokens, degradation_steps[]
fragments_included_count, fragments_excluded_count
provenance_summary: [{source_type, source_id, token_estimate}]
engine_id, plugin_ids[]
```

### 12.5 Logging

Structured logger namespace: `intergrax.context.engine`

| Level | When |
|-------|------|
| DEBUG | Per-provider collect counts, timing |
| INFO | Assembly complete — tokens, degradation |
| WARNING | Validation near-limit, quality threshold suppressions |
| ERROR | Validation failed, provider exception (fail closed or degrade per profile) |

**Never log** raw fragment content at INFO in production profiles — use `trace_id` + `fragment_id` only.

### 12.6 Observability / tracing

| Signal | Status |
|--------|--------|
| OTel spans | **Shipped (shim)** — `context/tracking/context_spans.py`; gate: `check_context_otel_span_registry.py` |
| Metrics | **Opt-in** — `runtime/observability/context_counters.py` (`INTERGRAX_CONTEXT_METRICS`) |
| Gate scripts | `check_context_otel_span_registry.py`, `check_context_engine_wiring.py`, `intergrax doctor` |
| Dashboards | Deferred (CE-9.6) — link when OBS product dashboard slice ships |

### 12.7 Cost attribution

Context assembly CPU time and optional LLM summarization calls **MUST** attribute to `run_id` / `task_id` via V-COST hooks when semantic compression invokes LLM — **deferred** until semantic compression hot path (CE-9.5).

---

## 13. Interactive / multi-hop assembly (codebase-scale)

For Cursor-class behaviour, CE ships an **optional orchestration loop** (CE-8 — codebase preset):

```text
ContextOrchestrator (max_hops=3, latency_budget_ms)
  hop 1: coarse workspace index search → candidate paths
  hop 2: read symbols / chunks via ToolOutputProvider
  hop 3: re-rank + assemble
```

| Component | Role |
|-----------|------|
| `WorkspaceContextProvider` | Merkle + chunks (`context/providers/workspace_index.py`) |
| `rag.retrieve` | Semantic fallback |
| `explore` delegation | Wide search child agent |
| `ContextOrchestrator` | Bounded loop — not unbounded agent while-true |

**Guardrails:** `max_hops`, `max_collect_latency_ms`, policy on total tool reads per assembly.

---

## 14. Use cases

### 14.1 Short chat Q&A

- Preset: `default`
- Sources: session + LTM (if prefer) + optional RAG
- Degradation: unlikely; `OFF` compression

### 14.2 Long session support

- **Episodic semantic recall** (when `MemoryProfile.enable_session_vector_index`) — retrieve relevant past turns from the session turn vector index before assembling chronological history; see [`MEMORY.md`](MEMORY.md) §7.1.1 (MEM-VEC-2.*)
- `HistoryLayer` SUMMARIZE_OLDEST on the **remaining** chronological tail when episodic + budget still overflow
- CE drops lowest-scored optional fragments (`SESSION_HISTORY_SEMANTIC` before mandatory user turn) per degradation ladder

### 14.3 Document Q&A

- RAG provider dominant; history minimal
- Citations in fragment metadata → output grounding

### 14.4 Multi-agent legal workflow

- Graph priors with `SUMMARY_ONLY` tier
- Preset `regulated_minimal`

### 14.5 Codebase feature implementation

- Preset `codebase`
- Plugins: workspace + graph_prior + tool_output
- Orchestrator 2–3 hops
- Budget: 128k window — never full repo dump

### 14.6 Delegated research child

- Preset `explore_child`
- Parent sees synthesis fragment only via `DelegationResult`

---

## 15. Author extension guide (summary)

Full detail: [`EXTENSION_AUTHOR_GUIDE.md`](../guides/EXTENSION_AUTHOR_GUIDE.md) §Context (CE-12.1 deferred) · [`AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix L · `intergrax/context/plugin.py`.

```python
class MyCodebaseContextPlugin:
    plugin_id = "acme.codebase"

    def register(self, registry: ContextPluginRegistry) -> None:
        registry.add_provider(WorkspaceIndexProvider(...))
        registry.set_ranker(CodeProximityRanker(...))

# applications/myapp/host/environment_profile.py
profile = ApplicationEnvironmentProfile(
    context_profile=ContextProfile(
        engine_preset="custom",
        engine_ref="myapp.context.CodebaseContextEngine",
        context_plugin_ids=["acme.codebase"],
    ),
)
```

**Forbidden:**

- Subclassing agents to concatenate prompts
- Importing `ContextCompiler` from Tier-2 agents
- Bypassing `BEFORE_CONTEXT_BUILD` for production hosts

---

## 16. Engine depth audit register (2026-06-12, post CE-EXT)

| ID | Status | Finding (original) | Resolution |
|----|--------|-------------------|------------|
| GAP-CTX-01 | **Closed** | No `ContextSourceProvider` Protocol + registry | CE-1, CE-2 shipped |
| GAP-CTX-02 | **Closed** | No unified `ContextEngine.assemble()` | CE-3.7 graph engine path; hybrid UAEP/ACP remain |
| GAP-CTX-03 | **Closed** | No injectable `ContextEngine` on assembly path | CE-3.4, CE-3.5 |
| GAP-CTX-04 | **Closed** | No step-aware `ContextAssemblyRequest` | CE-4, CE-5.1 |
| GAP-CTX-05 | **Closed** | Quality scoring not in hot path | CE-10.1 ranker gate |
| GAP-CTX-06 | **Closed** | Workspace spike only | CE-7 workspace provider |
| GAP-CTX-07 | **Closed** | No `ContextOrchestrator` | CE-8 codebase preset |
| GAP-CTX-08 | **Closed** | `classify_candidates` string heuristics | **CE-10.3** — CE tag prefix + legacy fallback (2026-06-14) |
| GAP-CTX-09 | **Closed** | No OTel spans on hot path | CE-9.2 span registry + shim |
| GAP-CTX-10 | **Closed** | No `CONTEXT_CANDIDATE_*` bus events | CE-9.1 engine emission via `context_skill_recording` |
| GAP-CTX-11 | **Closed** | `ContextBuilder` name collision | CE-3.6 `SessionRagContextBuilder` alias |
| GAP-CTX-12 | **Deferred** | L4 adaptive ranking | AHI plan |
| GAP-CTX-13 | **Closed** | `ContextCompiler` not on hot path | CE-3.9 ACP + graph engine |
| GAP-CTX-14 | **Closed** | `CONTEXT_BUILT` vs `CONTEXT_ASSEMBLED` split | CE-3.11 unified v2 |
| GAP-CTX-15 | **Closed** | Provider `collect()` fragments not merged into LLM window | CE-FMT-1 `formatter.py` + engine merge |
| GAP-CTX-16 | **Closed** | `ContextOrchestrator` not on Nexus graph hot path | CE-8.2b `ContextManager` + wiring |
| GAP-CTX-17 | **Closed** | `engine_ref=custom` not resolved | `context_engine_resolver.load_context_engine` |
| GAP-CTX-18 | **Closed** | Preset engines lack §8.5 behavior | `preset_engines.py` |
| GAP-CTX-19 | **Closed** | Registry formatter unused | `BuiltinContextPlugin` sets `DefaultContextFormatter` |
| GAP-CTX-20 | **Closed** | 8 builtin stub `collect()` returned `[]` on engine path | **CE-PROV-WIRE** B1–B4 Done (2026-06-14) |

**Traceability:** [`plan/CONTEXT_ENGINEERING.md`](../plan/CONTEXT_ENGINEERING.md) · **CE-ALIGN** Done · **CE-PROV-WIRE** Done.

---

## 17. Module inventory

| Module | Tier | Role |
|--------|------|------|
| `intergrax/context/contracts.py` | 0 | `ContextFragment`, `ContextAssemblyRequest`, `AssembledContext` |
| `intergrax/context/protocols.py` | 0 | Provider / ranker / engine protocols |
| `intergrax/context/registry.py` | 0 | `ContextPluginRegistry` |
| `intergrax/context/plugin.py` | 0 | `register_context_plugin()` |
| `intergrax/context/bootstrap.py` | 0 | `bootstrap_context_catalog()` |
| `intergrax/context/ranker.py` | 0 | `DefaultContextRanker` (step + quality gate) |
| `intergrax/context/dedup.py` | 0 | `dedup_fragments_by_hash()` |
| `intergrax/context/orchestrator.py` | 0 | `ContextOrchestrator` (codebase preset) |
| `intergrax/context/quality.py` | 0 | `evaluate_context_engineering()` |
| `intergrax/context/providers/builtin.py` | 0 | `BuiltinContextPlugin` (all §8.4 providers live via legacy bridge) |
| `intergrax/context/providers/legacy_bridge.py` | 0 | Legacy collector adapters (**CE-PROV-BRIDGE** — B1 shipped) |
| `intergrax/context/providers/workspace.py` | 0 | `WorkspaceContextProvider` |
| `intergrax/context/providers/workspace_index.py` | 0 | Merkle workspace index |
| `intergrax/context/providers/session_semantic_recall.py` | 0 | Episodic vector recall provider |
| `intergrax/context/tracking/context_spans.py` | 0 | CE OTel span names + shim |
| `intergrax/runtime/nexus/context/context_engine.py` | 1 | `DefaultNexusContextEngine` |
| `intergrax/runtime/nexus/context/codebase_engine.py` | 1 | `CodebaseContextEngine` preset |
| `intergrax/runtime/nexus/context/compile_service.py` | 1 | ACP compiler hot-path helpers |
| `intergrax/runtime/nexus/context/context_validator.py` | 1 | `DefaultContextValidator` + preflight |
| `intergrax/runtime/nexus/context/fragment_bridge.py` | 1 | `ContextCandidate` ↔ `ContextFragment` |
| `intergrax/runtime/nexus/context/graph_assembly.py` | 1 | Graph `ContextAssemblyRequest` builder |
| `intergrax/runtime/nexus/context/context_compiler.py` | 1 | Budget allocator + degradation |
| `intergrax/runtime/nexus/context/context_preflight.py` | 1 | Never-overflow preflight |
| `intergrax/runtime/nexus/context/context_manager.py` | 1 | Graph assembly + async engine path |
| `intergrax/runtime/nexus/context/context_assembler.py` | 1 | Graph message compose helpers |
| `intergrax/runtime/nexus/context/context_builder.py` | 1 | Session RAG; `SessionRagContextBuilder` alias |
| `intergrax/runtime/policy/context_assembly_policy.py` | 1 | `run_pre_context_policy_gate()` |
| `intergrax/runtime/observability/context_counters.py` | 1 | Opt-in assembly metrics |
| `intergrax/runtime/nexus/tools/plan_context_invocation.py` | 1 | ACP catalog context injection |
| `intergrax/runtime/architecture/context_engineering.py` | 1 | Quality scoring shim |
| `intergrax/runtime/architecture/context_regression_benchmark.py` | 1 | Regression harness |
| `intergrax/memory/workspace_index_spike.py` | 0 | Legacy RFC — superseded by `context/providers/workspace_index.py` |
| `intergrax/contracts/context_assembly.py` | 0 | `TaskContextAssemblyOptions` |
| `intergrax/contracts/agent_context_hints.py` | 0 | `AgentContextHints` (CE-5.1) |
| `intergrax/applications/_shared/context_presets.py` | 3 | Tier-3 preset helpers |
| `intergrax/applications/_shared/context_runtime_bridge.py` | 3 | Profile → RuntimeConfig |
| `intergrax/applications/_shared/context_wiring.py` | 3 | Engine + manager resolution |
| `intergrax/agents/authoring/context_assembly_bridge.py` | 2 | ACP `ContextAssemblyRequest` builder |
| `intergrax/agents/uaep.py` | 2 bridge | UAEP hooks + `CONTEXT_ASSEMBLED` v2 |
| `intergrax/runtime/nexus/context/runtime_state_handle_bridge.py` | 1 | RuntimeState → CE provider metadata sync (CE-HANDLE-FILL) |
| `scripts/check_context_engine_wiring.py` | — | CI preset resolution gate |
| `scripts/check_context_builtin_providers.py` | — | CI builtin collect wiring gate (CE-PROV-GATE) |
| `scripts/check_context_otel_span_registry.py` | — | CI span wiring gate |

---

## 18. Anti-patterns

| Anti-pattern | Correct approach |
|--------------|------------------|
| Agent builds `prompt = history + rag + tools` | `ContextEngine.assemble()` |
| Full session dump to LLM | HistoryLayer + CE budget |
| Silent fragment drop | Emit `CONTEXT_CANDIDATE_DROPPED` with reason |
| RAG chunks without citation metadata | `ContextFragment.metadata.citations` |
| Custom CE fork in `agents/` | `ContextPlugin` in application package |
| Storing CE diagnostics in task KV | Trace + RuntimeEvent only |

---

## 19. Related documents

| Document | Relationship |
|----------|--------------|
| [`architecture/MEMORY.md`](MEMORY.md) | Stores + lifecycle — CE consumes via providers |
| [`architecture/RAG.md`](RAG.md) | Retrieval — CE consumes chunks |
| [`plan/CONTEXT_ENGINEERING.md`](../plan/CONTEXT_ENGINEERING.md) | Implementation register |
| [`guides/AGENT_CREATION_GUIDE.md`](../guides/AGENT_CREATION_GUIDE.md) Appendix L | Author control plane |
| [`IDEAL_HARNESS_AI_ARCHITECTURE.md`](../guides/IDEAL_HARNESS_AI_ARCHITECTURE.md) §16 | Target vision |
| [`ADR-CTX-001`](../adr/entries/2026-06-12/ADR-CTX-001.md) | Domain split decision |

---

*End of Context Engineering architecture canon.*

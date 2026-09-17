# ADR-MEM-XINT-002: Unified Information & Context Authority (Memory × CE × RAG × Tools)

| Field | Value |
|-------|-------|
| **Status** | Accepted (architecture) |
| **Date** | 2026-09-17 |
| **Deciders** | Platform / Memory × Context integration |
| **Related** | [`MEMORY_ARCHITECTURE.md`](../../../../architecture/MEMORY_ARCHITECTURE.md) · [`CONTEXT_ENGINEERING.md`](../../../../architecture/CONTEXT_ENGINEERING.md) · [`ADR-UCL-001`](../../2026-08-01/ADR-UCL-001.md) · MEM-XINT-1 audit (HEAD `60ba65bb86e5a74a821f90c8d6a2881e5ea2c83e`) · Memory baseline `4e92d14c58f02200518786c59c9128b3068e4bc5` |

## Context

MEM-XINT-1 confirmed cross-layer ownership gaps:

| ID | Finding |
|----|---------|
| MXINT-01 | Runtime recall bypasses `MemoryControlPlane.recall` via `SessionManager.search_user_longterm_memory` → `UserProfileManager.search_longterm_memory` |
| MXINT-02 | `ltm.write_fact` / `ltm.search` fall back to direct `UserProfileManager` when the control plane is absent |
| MXINT-03 | Parallel pipelines: CE `ContextEngine.assemble` **and** legacy `ChatMessage` injection (`insert_context_before_last_user`, prompt builders) |
| MXINT-04…09 | Duplicate LTM injection, missing cross-source conflict policy, missing score normalization, weak legacy provenance, SessionTurnIndex E2E gaps, `ltm.search` keyword fallback |

Enterprise Memory (MEM-ENT-1…16) established **`MemoryControlPlane`** as the semantic mutation/recall boundary for durable user memory. Context Engineering (CE) already owns **`ContextEngine.assemble`** as the composition gate on graph/UAEP paths, but Nexus tool loops still inject RAG/LTM/tool traces outside that gate.

This ADR defines the **target authority model** and **migration boundaries**. It does **not** implement enforcement (MXINT-3…6).

## Problem

Without a single durable-memory authority and a single final-context authority:

- Governance diverges (plane vs manager vs prompt builder).
- Token budget is split (CE compiler vs legacy injected messages).
- Provenance cannot answer “which Memory entry / RAG chunk / tool output reached the model?”
- Cross-source conflicts (e.g. Memory says Warsaw, live tool says Kraków) have no deterministic policy stage.

## Current state (code-aligned snapshot)

**Memory read (MXINT-01):** `populate_request_memory_recall_metadata` and `run_longterm_memory_context` call `SessionManager.search_user_longterm_memory`, not `MemoryControlPlane.recall`. CE consumes LTM via handles (`LTM_ENTRIES_HANDLE` / metadata keys) populated from that path.

**Memory write (MXINT-02):** `intergrax/tools/providers/ltm/service.py` uses the plane when wired; otherwise `manager.add_memory_entry` / `search_longterm_memory` / keyword scan.

**Context (MXINT-03):** `DefaultNexusContextEngine.assemble` collects providers, hash-dedups, ranks, compiles budget. In parallel, `memory_context_invocation.run_longterm_memory_context`, `plan_context_invocation` (RAG), and `catalog_context` / `inject_tool_traces_system_context` inject `ChatMessage` blocks directly.

**Session episodic:** `SessionSemanticRecallProvider` reads `session_vector_hits` handles filled by `SessionManager.search_session_semantic_recall` (SessionTurnIndex store). Session **turn log** authority remains `SessionStorage` behind `SessionManager`.

### Diagram 1 — Current split architecture

```mermaid
flowchart TB
    subgraph Runtime["Nexus runtime"]
        RR[RuntimeRequest / RuntimeState]
        SM[SessionManager]
        UPM[UserProfileManager.search_longterm_memory]
        POP[populate_request_memory_recall_metadata]
        LEG[run_longterm_memory_context + prompt builders]
        INS[insert_context_before_last_user]
        RAGB[rag_prompt_builder]
        TOOL[inject_tool_traces_system_context]
    end
    subgraph CE["Context Engineering"]
        CEgate[ContextEngine.assemble]
        PROV[builtin.* providers + legacy_bridge handles]
        COMP[ContextCompiler budget]
    end
    subgraph Mem["Memory domain"]
        MCP[MemoryControlPlane recall/remember]
    end
    RR --> POP
    POP --> SM --> UPM
    RR --> LEG --> SM
    LEG --> INS
    RR --> RAGB --> INS
    RR --> TOOL
    POP --> PROV
    PROV --> CEgate --> COMP
    LEG -.->|duplicate LTM| INS
    MCP -.->|ltm.tools when wired| Tools
    MCP -.x bypassed on runtime recall| SM
```

## Decision

### Formal ownership answers

| Question | Owner |
|----------|--------|
| Who owns **final model context**? | **Context Engineering** — `ContextEngine.assemble` → `AssembledContext.messages` (single composition gate after migration) |
| Who is **semantic authority for durable user memory**? | **`MemoryControlPlane`** (`remember` / `recall` / governed lifecycle) |
| Who owns **recall governance** for user durable memory? | **`MemoryControlPlane`** (before any CE fragment emission) |
| Can runtime fetch canonical LTM via manager directly? | **NO** (target). Only typed Memory capability / control plane contract |
| Who owns **token budget** for model-facing context? | **Context Engineering** (`ContextBudgetSnapshot`, `ContextCompiler`, unified accounting) |
| Who owns **cross-source conflict resolution**? | **Context Engineering** — pluggable **`ContextConflictResolver`** stage (MXINT-5); must not mutate Memory |
| Who owns **cross-source score normalization**? | **Context Engineering** — pluggable **`ContextScoreNormalizer`** (MXINT-5) |
| Where do **adapters** live? | **Application / runtime composition** (`intergrax/runtime/nexus/context/*`, `intergrax/applications/_shared/*`) — bridge typed domain results → `ContextFragment`, not CE → concrete stores |

**No god object:** reject `UnifiedInformationManager`, `SuperContextMemoryService`, or monolithic knowledge services.

**CE is not storage:** CE must not write UserProfile, RAG stores, tool state, or canonical session logs.

**Memory is not a compiler:** recall returns **`MemoryControlRecallResult`** (typed items), not `ChatMessage` or prompt blocks.

**RAG is not Memory:** retrieval yields evidence; durable memory only via explicit `remember`.

**Tool output is ephemeral** until an explicit `remember` path.

### Diagram 2 — Target information flow

```mermaid
flowchart TB
    ID[RequestIdentity + scope]
    ID --> MR[MemoryControlPlane.recall]
    ID --> STI[SessionTurnIndex recall metadata]
    ID --> RAG[RAG RetrievalResult]
    ID --> TOOL[ToolExecutionResult / ToolModelObservation]
    MR --> H1[Typed recall handle]
    STI --> H2[Episodic hit handle]
    RAG --> H3[Chunk handle]
    TOOL --> H4[Observation handle]
    H1 --> AD1[Memory → ContextFragment adapter]
    H2 --> AD2[Session episodic provider]
    H3 --> AD3[RAG provider adapter]
    H4 --> AD4[Tool provider adapter]
    AD1 --> CSP[ContextSourceProvider.collect]
    AD2 --> CSP
    AD3 --> CSP
    AD4 --> CSP
    CSP --> CE[ContextEngine.assemble]
    CE --> NORM[Normalize scores]
    NORM --> DED[Dedup identity + content]
    DED --> CONF[Conflict resolve]
    CONF --> RANK[Rank]
    RANK --> BUD[Budget + compression]
    BUD --> COMP[Compiler / formatter]
    COMP --> FMC[FINAL MODEL CONTEXT]
```

### Diagram 3 — Memory read/write authority

```mermaid
flowchart LR
    subgraph Writes["Durable writes"]
        TWR[Tool / agent remember intent]
        TWR --> MCPW[MemoryControlPlane.remember]
        MCPW --> GOVW[governance + lifecycle]
        GOVW --> CAN[Canonical UserProfile / stores]
    end
    subgraph Reads["Governed recall"]
        TR[Tool ltm.search / runtime recall]
        TR --> MCPR[MemoryControlPlane.recall]
        MCPR --> GOVR[recall governance]
        GOVR --> RES[MemoryControlRecallResult]
        RES --> AD[Composition adapter]
        AD --> FR[ContextFragment LONGTERM_MEMORY]
    end
    UPM[UserProfileManager direct] -.->|FORBIDDEN target path| CAN
    SM[SessionManager.search_user_longterm_memory] -.->|MIGRATE to MCPR| UPM
```

### Canonical ownership table

| Concern | Canonical owner | Replaceable contract | Forbidden bypass |
|--------|-----------------|----------------------|------------------|
| Durable memory write | `MemoryControlPlane.remember` | Memory capability / projection plugins | `UserProfileManager.add_memory_entry` from tools/runtime fallback |
| Memory recall | `MemoryControlPlane.recall` | Recall strategies behind plane | `SessionManager.search_user_longterm_memory` as semantic boundary |
| Session episodic recall | SessionTurnIndex + plane-adjacent metadata APIs | Index store / recall metadata builders | Treating index as canonical session history |
| Session history (turn log) | `SessionStorage` via `SessionManager` | Storage backends | CE mutating session store |
| RAG retrieval | RAG integration / retrieval service | Rank fusion, retrievers | `rag_prompt_builder` → direct model messages (target) |
| Tool execution | Tool runtime + governance | Tool catalog | Tool output → durable memory without `remember` |
| Final context assembly | `ContextEngine.assemble` | Engine id, provider registry | Parallel legacy `insert_context_before_last_user` for same sources |
| Token budget | CE `ContextCompiler` + budget policy | `ContextBudgetAllocator`, UCL artifacts | Legacy injected tokens outside CE accounting |
| Cross-source conflict | CE pipeline stage | `ContextConflictResolver` (MXINT-5) | Ad-hoc merge in prompt builders |
| Score normalization | CE pipeline stage | `ContextScoreNormalizer` (MXINT-5) | Raw vector score == memory score assumptions |

### Data flow table

| Source | Source contract | Governance before CE | CE adapter/provider | Final authority |
|--------|-----------------|----------------------|---------------------|-----------------|
| User LTM | `MemoryControlRecallResult` | Plane recall governance | `builtin.longterm_memory` (+ composition adapter from plane output) | CE assembly |
| Session turns | `SessionStorage` messages / revision | Session scope | `builtin.session_history` | CE assembly |
| SessionTurnIndex | Recall metadata rows | Index scope + identity | `SessionSemanticRecallProvider` | CE assembly |
| RAG | `RetrievalResult` / chunk records | RAG scope/disclosure | `builtin.rag` / `fragments_from_rag_chunks` | CE assembly |
| Tools | `ToolExecutionResult`, `IterativeToolOutputBlock` | Tool governance + untrusted boundary | `builtin.tool_output` | CE assembly |
| System instructions | Policy / harness | Policy gate | `builtin.system_instructions` | CE assembly |

### Mutation flow table

| Mutation | Allowed boundary | Forbidden boundary |
|----------|------------------|-------------------|
| Remember user fact | `MemoryControlPlane.remember` | CE, RAG indexer, tool handler → profile store |
| Forget / supersede | Plane lifecycle APIs | Direct store delete from CE |
| Session turn append | `SessionManager` / `SessionStorage` | CE |
| RAG index update | RAG pipeline | Automatic from tool output |
| Tool state | Tool runtime | Memory plane without explicit remember |

### Diagram 4 — Cross-source policy pipeline (target CE inner pipeline)

```mermaid
flowchart TD
    COL[Provider collect + canonicalize]
    COL --> PRE[Pre/post collect policy gate]
    PRE --> HASH[Identity dedup content_hash]
    HASH --> SEM[Semantic dedup optional strategy]
    SEM --> NORM[ScoreNormalizer per source_type]
    NORM --> CONF[ConflictResolver]
    CONF --> RANK[ContextRanker]
    RANK --> ALLOC[BudgetAllocator + mandatory/preferred/optional]
    ALLOC --> OVER[Overflow: drop optional / compress / truncate allowed types]
    OVER --> FMT[ContextFormatter + compile_service]
    FMT --> OUT[AssembledContext + provenance manifest]
```

**Ordering invariant:** normalize → dedup (identity then semantic) → conflict → rank → budget → compile. **Deterministic** tie-break: `normalized_score`, `source_priority` (policy), `freshness_score`, `source_id`, `fragment_id`.

**Raw scores preserved:** normalization adds fields; does not erase provider scores in metadata.

### Diagram 5 — Migration phases

```mermaid
flowchart LR
    P0[MEM-ENT + MEM-XINT-1 audit]
    P3[MXINT-3 Memory boundary enforcement]
    P4[MXINT-4 Single CE composition]
    P5[MXINT-5 Cross-source policy layer]
    P6[MXINT-6 Cross-layer E2E certification]
    P0 --> P3 --> P4 --> P5 --> P6
```

Each phase: reversible flags, working runtime, **no permanent dual-path**; removal condition documented per flag.

### MXINT-01/02/03 — current vs target vs migration

**MXINT-01**

| | Flow |
|---|------|
| CURRENT | Runtime → `SessionManager.search_user_longterm_memory` → `UserProfileManager.search_longterm_memory` → metadata handles + legacy injection |
| TARGET | Runtime/tool → `MemoryControlPlane.recall` → typed result → composition adapter → `LTM_ENTRIES` handle → `builtin.longterm_memory` → CE only |
| MIGRATION | MXINT-3: rewire `populate_request_memory_recall_metadata` / deprecate semantic gateway on `SessionManager`; thin adapter keeps API surface if needed |

**MXINT-02**

| | Flow |
|---|------|
| CURRENT | `ltm.write_fact` / `ltm.search` → plane **or** manager/keyword fallback |
| TARGET | Tools require plane + `RequestIdentity`; fail closed if missing |
| MIGRATION | MXINT-3: remove fallback branches; host wiring must inject plane |

**MXINT-03**

| | Flow |
|---|------|
| CURRENT | CE assemble **+** `user_longterm_memory_prompt_builder`, `rag_prompt_builder`, `insert_context_before_last_user`, tool system injection |
| TARGET | All sources → handles/fragments → **`ContextEngine.assemble` only** |
| MIGRATION | MXINT-4: feature-flag legacy injection off; route RAG/LTM/tool traces through providers; unified budget |

### Legacy path classification

| Path | State | Action | Migration target |
|------|-------|--------|------------------|
| `MemoryControlPlane.recall/remember` | ACTIVE | KEEP | Canonical |
| `ContextEngine.assemble` | ACTIVE | KEEP | Single gate |
| `ContextSourceProvider` + `ContextPluginRegistry` | ACTIVE | KEEP | Pluggable providers |
| `legacy_bridge` handle adapters | ACTIVE | MIGRATE | Typed handles from plane/RAG/tools (not raw manager dicts) |
| `populate_request_memory_recall_metadata` | ACTIVE | MIGRATE | Plane-backed recall population |
| `SessionManager.search_user_longterm_memory` | ACTIVE | DEPRECATE (semantic) | Thin adapter → `MemoryControlPlane.recall` (internal) |
| `run_longterm_memory_context` + LTM prompt builder | ACTIVE | DEPRECATE | CE provider only |
| `DefaultRagPromptBuilder` + plan RAG inject | ACTIVE | DEPRECATE | `RAG_CHUNKS_HANDLE` → CE |
| `insert_context_before_last_user` | ACTIVE | DEPRECATE | CE formatter output |
| `inject_tool_traces_system_context` | ACTIVE | MIGRATE | CE tool channel policy; untrusted by default |
| `ltm.search` / `ltm.write_fact` manager fallback | ACTIVE | REMOVE (target) | Plane required |
| `_keyword_hits` in ltm.search | ACTIVE | REMOVE | Plane recall only |
| `tools/providers/memory/service.py` manager search | ACTIVE | MIGRATE | Plane recall |

Deprecation states: **ACTIVE → MIGRATION (flag) → DEPRECATED → REMOVE**.

### Contracts

#### Reuse (NO NEW CORE CONTRACT REQUIRED for CE-1.x core)

Keep and extend in place:

| Contract | Role |
|----------|------|
| `ContextFragment` | Universal CE candidate; already has `source`, `source_id`, scores, `mandatory`, `provider_provenance`, `metadata` |
| `ContextFragmentSource` | Maps to authority class (see below) |
| `ContextAssemblyRequest` | Scoped assembly input (`tenant_id`, trace/run/task ids) |
| `ContextProviderContext.handles` | Typed runtime handles (not serialized) — **`request_identity`** must be supplied by composition layer |
| `ContextSourceProvider` | Provider plugin boundary |
| `ContextAssemblyProvenance` / `AssembledContext.provenance` | Lineage v2 |
| `MemoryControlRecallResult` / `MemoryControlRememberRequest` | Memory recall/write semantics |
| `RequestIdentity` | Spine for scope; no synthetic identity |

#### Minimal extensions (MXINT-4/5 — design only)

1. **Normative `ContextFragment.metadata` keys** (documented constants, not unbounded dict at adapter boundaries for new code):

   - `authority_class`: `canonical_memory` \| `derived_memory` \| `rag_evidence` \| `tool_observation` \| `session_episodic` \| `system_context`
   - `trust_tier`, `sensitivity`, `scope_ref` (tenant/user/session)
   - `raw_relevance_score`, `normalized_relevance_score`
   - `memory_entry_id`, `rag_chunk_id`, `tool_call_id`, `session_message_id` (source-specific ids)

2. **`ContextScoreNormalizer` protocol** (CE plugin, MXINT-5).

3. **`ContextConflictResolver` protocol** (CE plugin, MXINT-5) — no Memory writes.

4. **`ContextSemanticDeduper` protocol** (optional).

5. **`ContextAssemblyDecisionRecord`** optional manifest on `AssembledContext` (vNext field).

#### Contract impact matrix

| Contract | Keep | Extend | Deprecate | New |
|----------|------|--------|-----------|-----|
| `ContextFragment` | ✓ | metadata keys + raw/normalized scores | | |
| `ContextAssemblyProvenance` | ✓ | transformation steps (optional) | | |
| `AssembledContext` | ✓ | optional decision manifest | | |
| `ContextRanker` | ✓ | consume normalized scores | | |
| `ContextSourceProvider` | ✓ | | | |
| `MemoryControlPlane` | ✓ | | | |
| `SessionManager.search_user_longterm_memory` | | adapter-only | public semantic use | |
| RAG/tool prompt builders | | | direct model injection | |
| | | | | `ContextScoreNormalizer` (MXINT-5) |
| | | | | `ContextConflictResolver` (MXINT-5) |

### Pluginability map

| Mechanism | Contract | Default strategy | Replaceable |
|-----------|----------|------------------|-------------|
| Memory source provider | `ContextSourceProvider` (`builtin.longterm_memory`) | Handle rows from plane adapter | ✓ registry |
| RAG source provider | `ContextSourceProvider` (`builtin.rag`) | `fragments_from_rag_chunks` | ✓ |
| Tool observation provider | `ContextSourceProvider` (`builtin.tool_output`) | `IterativeToolOutputBlock` | ✓ |
| Session episodic provider | `SessionSemanticRecallProvider` | vector hits handle | ✓ |
| Ranker | `ContextRanker` | `DefaultContextRanker` | ✓ |
| Budget allocator | `ContextBudgetAllocator` / compiler | `ContextCompiler` | ✓ |
| Deduper | hash + future semantic | `dedup_fragments_by_hash` | ✓ MXINT-5 |
| Conflict resolver | `ContextConflictResolver` | none (pass-through) | ✓ MXINT-5 |
| Score normalizer | `ContextScoreNormalizer` | identity mapping initially | ✓ MXINT-5 |
| Compiler | `compile_service` / formatter | Nexus defaults | ✓ |

### Governance boundaries

Memory governance before recall handles; RAG/tool governance before CE; CE inclusion policy for sensitivity/channel/budget/injection.

**Trust authority:** domains propose trust; CE policy finalizes model-facing trust. Providers must not self-elevate to privileged system channels.

### Layer diagram

```text
Tier-0 domain contracts (Memory, Context, RAG, Tools)
        ↑
Tier-1 implementations (plane, stores, retrievers, tool runtime)
        ↑
Application / Nexus composition adapters (recall → fragment, retrieval → fragment)
        ↑
ContextEngine.assemble (final authority)
```

**Import rule:** CE imports source **contracts**, not `UserProfileManager` concrete types. Memory must not import CE.

### SessionTurnIndex

Derived episodic source for CE; **not** canonical session history. Canonical turn log = **`SessionStorage`** via **`SessionManager`**.

### Observability

Reuse runtime event bus / CE recording hooks; reuse `run_id`, `step_id`, `session_id`, `trace_id`.

### Security

Untrusted RAG/tool content defaults non-privileged; no control of memory authority or tool permissions.

### Compatibility

Legacy APIs as thin adapters to plane; no dual governance; migration-only feature flags with removal conditions.

## Rejected alternatives

1. **Keep manager fallback for resilience** — resilience ≠ authority bypass.
2. **ToolRuntime as final composition owner** — duplicates CE.
3. **Monolithic UnifiedInformationService** — breaks bounded contexts.
4. **Simple numeric scale normalization** — incomparable scores.
5. **Automatic tool output → Memory** — governance drift.

## Consequences

### Positive

Single audit story; MXINT-3…6 independently certifiable; aligns MEM-ENT + CE providers.

### Negative

Hot-path Nexus migration; MXINT-5 pipeline must stay deterministic and LLM-free by default.

## Implementation roadmap

| Phase | Scope |
|-------|--------|
| **MXINT-3** | Plane-only LTM read/write; SessionManager adapter to plane. |
| **MXINT-4** | Remove duplicate ChatMessage injection; CE-only composition. |
| **MXINT-5** | Normalizer, dedup, conflict, unified budget, provenance manifest. |
| **MXINT-6** | E2E certification all sources. |

## Hard invariants (target)

Memory persistent authority = Control Plane; final context = CE; no synthetic identity; CE never writes Memory; Memory never imports CE; scope preserved; external data not privileged by default.

## Strategy points (pluginable)

Ranking, budget, source priority, normalization, dedup, conflict, compression, optional semantic judge.

## Certification plan

MXINT-3 plane enforcement tests; MXINT-4 single assemble path; MXINT-5 deterministic fixtures; MXINT-6 integration suite.

## Compliance

Tier boundaries preserved; MEM-ENT invariants authoritative; aligns with ADR-UCL-001 single-budget direction.

## Implementation notes

Documentation only in MEM-XINT-2. Bounded regression: see `.tmp/session/MEM-XINT-2/pytest.log`.

# Local Knowledge Workspace (LKW) — architecture

**Status:** Scaffold + architecture baseline (2026-06-07) — Wave 0 product reprioritization  
**Tier:** Tier-3 application (`local_workspace_application`)  
**Agents:** Tier-2 `local_indexer`, `local_search`, `local_synthesizer`  
**Canonical plan row:** [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md` §6.3a LKW.*](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated)  
**Local plan:** [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

---

## 1. Strategic purpose

**Local Knowledge Workspace (LKW)** is the first **business product environment** on Intergrax after harness platform maturity. Its role is dual:

1. **Product:** Give a user a local, safe assistant over their own files — search, gather context, produce structured outputs (reports, emails, estimates).
2. **Harness validation:** Exercise the Agent OS on a real, observable workload without external market APIs (unlike deferred K.1 Problem Radar / K.2 Vendor Discovery).

LKW validates: RAG ingest/retrieve/index lifecycle, document parsing, shadow workspace, multi-agent orchestration, memory, policy, trace, MCP/HTTP serving, and Tier-3 composition — while surfacing platform gaps early.

**Strategic frame:** [`docs/INTERGRAX_DEVELOPMENT_STRATEGY.md`](../../docs/INTERGRAX_DEVELOPMENT_STRATEGY.md) — explicit product reprioritization after Appendix A sign-off.

---

## 2. Problem statement

Users store project knowledge across folders (PDF, DOCX, XLSX, TXT, email exports). They need to:

| Need | Example |
|------|---------|
| **Find** | „Znajdź dokumenty o projekcie X / rozliczeniu Y” |
| **Gather** | „Zbierz dane z folderów A i B dotyczące kosztorysu” |
| **Synthesize** | „Przygotuj mail / sprawozdanie / kosztorys wg szablonu” |
| **Safety** | Nic nie usuwać ani nie nadpisywać w oryginalnych plikach użytkownika |

LKW solves this with **read-heavy indexing + semantic retrieval + isolated write artifacts**, orchestrated by Nexus.

---

## 3. Solution overview

```text
┌─────────────────────────────────────────────────────────────────────────┐
│  User client (HTTP / MCP / future desktop shell)                        │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │ POST /v1/local_workspace/run
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-3  local_workspace_application                                    │
│  manifest · environment_profile · tool_wiring · factory · MCP           │
└───────────────────────────────┬─────────────────────────────────────────┘
                                │
                                ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-1  Nexus Agent OS                                                 │
│  Intake → Plan → Graph (index → search → synthesize) → Trace → Result   │
└───────┬─────────────────┬─────────────────┬─────────────────────────────┘
        │                 │                 │
        ▼                 ▼                 ▼
┌──────────────┐  ┌──────────────┐  ┌──────────────────┐
│ local_indexer│  │ local_search │  │ local_synthesizer│
│ Tier-2       │  │ Tier-2       │  │ Tier-2           │
└──────┬───────┘  └──────┬───────┘  └────────┬─────────┘
       │                 │                    │
       └─────────────────┴────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────────────────┐
│  Tier-0  Platform — four-layer stack (canon §7.1.6–§7.1.8)              │
│  Integration → Tool → Skill → Agent                                     │
│  Docling · SQLite · vector store · rag.* · workspace.* · local.* skills │
└─────────────────────────────────────────────────────────────────────────┘
```

### 3.2 Four-layer composition (Integration → Tool → Skill → Agent)

LKW follows the canonical Intergrax stack — agents never call vendor SDKs; Tier-3 selects backends and enables catalog surfaces:

```text
IntegrationProfile (Tier-3 host)
  ├── document_parser=docling     → DocumentParser backend
  ├── vector_store=inmemory|chroma → VectorStore for RAG index
  ├── relational_store=sqlite     → trace, session, task memory
  ├── rerank_provider=cohere_rerank → optional rerank on rag.retrieve
  └── observability_backend=otel  → traces (optional)

ToolProfile (Tier-3 host/tool_wiring.py)
  └── enabled tool_ids → ToolRegistry + ToolWiringContext

SkillProfile (Tier-3 environment)
  └── enabled_bundles: harness (LKW.0) · local (LKW.2 planned)
      → resolves skill_ids → allowed_tools + prompt refs on AgentContract

AgentContract (Tier-2)
  └── skill_ids[] + capabilities[] → UAEP steps invoke tools via ToolRuntime
```

**Rule:** Tier-3 **wires** integrations and tools; Tier-2 agents **declare** `skill_ids` on `AgentContract`; skills **compose** tool packs + prompts + policy fragments. See [`docs/SKILLS.md`](../../docs/SKILLS.md) · [`docs/TOOLS.md`](../../docs/TOOLS.md) · [`docs/INTEGRATIONS.md`](../../docs/INTEGRATIONS.md).

### 3.1 Trust zones (filesystem safety)

| Zone | Purpose | Mechanism | Mutations |
|------|---------|-----------|-----------|
| **Read zone** | User documents (allowlisted paths) | `rag.ingest_document`, `document.parse`; future `filesystem.*` read-only | **None** on user FS |
| **Artifact zone** | Reports, drafts, exports | `workspace.*` on **shadow workspace** | Only under `INTERGRAX_SHADOW_ROOT` |
| **Sandbox zone** | Risky experiments | `sandbox.exec` (opt-in per task) | Isolated under `INTERGRAX_SANDBOX_ROOT` |

**Rule:** LKW agents MUST NOT write to user home directories. All deliverables go to shadow workspace unless the user explicitly promotes an export path in a future Wave.

---

## 4. Agent roster and capabilities

| Agent | Module | Capability | Responsibility |
|-------|--------|------------|----------------|
| **LocalIndexerAgent** | `agents/local_indexer/` | `local.workspace.index` | Discover paths (Wave 1: explicit), parse, chunk, embed, index via `rag.ingest_document` |
| **LocalSearchAgent** | `agents/local_search/` | `local.workspace.search` | Semantic + metadata-filtered retrieval via `rag.retrieve`; rank and package evidence |
| **LocalSynthesizerAgent** | `agents/local_synthesizer/` | `local.workspace.synthesize` | LLM synthesis from retrieved context; write artifacts to shadow workspace |

**Pipeline capability (graph-level):** `local.workspace.pipeline` — multi-step intent routing index → search → synthesize (Wave 2). Documented here; wired via Nexus `AgentGraph` / delegation like `research.pipeline`.

Agent architecture docs:

- [`agents/local_indexer/ARCHITECTURE.md`](../../agents/local_indexer/ARCHITECTURE.md)
- [`agents/local_search/ARCHITECTURE.md`](../../agents/local_search/ARCHITECTURE.md)
- [`agents/local_synthesizer/ARCHITECTURE.md`](../../agents/local_synthesizer/ARCHITECTURE.md)

---

## 5. Integrations, tools, and skills

### 5.1 Integrations (`IntegrationProfile`)

**Baseline preset:** `IntegrationProfile.legal_product()` — RAG + document parsing without mandatory web search (unlike `research_product()`).

| `IntegrationCategory` slot | Slug (default) | Role in LKW | Wired via |
|--------------------------|----------------|-------------|-----------|
| `relational_store` | `sqlite` | Trace DB, session state, task memory persistence | `wire_application_environment` → `memory_wiring` |
| `vector_store` | `inmemory` | RAG chunk index (dev); replace with `chroma` for durable local index | `rag_runtime_bridge` → `ToolWiringContext.vectorstore_manager` |
| `document_parser` | `docling` | PDF/DOCX/XLSX parsing inside `rag.ingest_document` / `document.parse` | `CatalogDocumentParser` — infra slot, not auto-exposed as agent tool |
| `rerank_provider` | `cohere_rerank` | Optional rerank after hybrid retrieval in `rag.retrieve` | `RetrievalService` / `RagProfile` |
| `observability_backend` | `otel` (optional) | Export traces when OTLP enabled on environment profile | `host/environment_profile.py` |
| `object_storage` | `filesystem` (Wave 4) | Export shadow artifacts / checkpoint blobs | `storage.*` tools when enabled |
| `message_bus` | queue slug (Wave 4) | Background ingest jobs | `message_bus.*` when worker enabled |

**Override:** `INTERGRAX_INTEGRATION_PROFILE_JSON` — e.g. swap `vector_store` to `chroma` for persistent local index.

**Explicitly excluded in baseline:** `search_provider` (web), `collaboration` (mail APIs) — LKW is local-first.

Authoring: [`docs/AGENT_CREATION_GUIDE.md` Appendix K](../../docs/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) · catalog: [`docs/INTEGRATIONS.md`](../../docs/INTEGRATIONS.md).

### 5.2 Tools (`ToolProfile` + `host/tool_wiring.py`)

Tier-3 enables tools; agents invoke them through `BoundToolGateway` / `ctx.invoke_tool()` — never direct integration imports.

#### Host-wide tool allowlist (`_LKW_BASE_TOOL_IDS`)

| tool_id | Bundle | Composes (integration slot) | LKW role |
|---------|--------|----------------------------|----------|
| `rag.ingest_document` | `rag` | `vector_store` + `document_parser` + embedding managers | Index local files |
| `rag.retrieve` | `rag` | `vector_store` + optional `rerank_provider` | Semantic search |
| `rag.list_collections` | `rag` | `vector_store` | Index diagnostics |
| `rag.list_documents` | `rag` | `vectorstore_manager` | Paginated index inventory |
| `rag.get_document` | `rag` | `vectorstore_manager` | Fetch indexed chunk by id |
| `rag.check_index_status` | `rag` | `vectorstore_manager` | Index readiness probe |
| `document.parse` | `document` | `document_parser` | Ad-hoc parse without full ingest |
| `document.parse_preview` | `document` | `document_parser` | Bounded parse preview (no ingest) |
| `workspace.read_file` | `workspace` | runtime `ShadowWorkspace` | Read shadow artifacts |
| `workspace.write_file` | `workspace` | runtime `ShadowWorkspace` | Write drafts/reports |
| `workspace.list_files` | `workspace` | runtime `ShadowWorkspace` | List artifacts |
| `workspace.snapshot` | `workspace` | runtime `ShadowWorkspace` | Point-in-time snapshot |
| `workspace.delete_file` | `workspace` | runtime `ShadowWorkspace` | Remove draft revisions in shadow only |
| `workspace.search` | `workspace` | runtime `ShadowWorkspace` | Grep across shadow artifacts |
| `memory.read` / `memory.write` / `memory.list_keys` | `memory` | `relational_store` + task memory | Session working state |
| `cache.get` / `cache.set` | `cache` | optional KV backend | Dedup parse/embedding keys |

**Env-gated (settings):** `LOCAL_WORKSPACE_ENABLE_RAG` → `rag.retrieve`; `LOCAL_WORKSPACE_ENABLE_RAG_INGEST` → `rag.ingest_document`.

**Filesystem browse (T6 / LKW.3 Done):** when `INTERGRAX_ALLOWED_READ_ROOTS` or `allowed_read_roots` is set, host auto-enables `filesystem.list`, `filesystem.glob`, `filesystem.read_text`, `filesystem.stat`.

**Explicitly disabled:** `websearch.*`, `openai.file_search.*` — external retrieval out of scope for LKW baseline.

Catalog reference: [`docs/TOOLS.md`](../../docs/TOOLS.md) · wiring: [`host/tool_wiring.py`](host/tool_wiring.py).

### 5.3 Skills (`SkillProfile` + `AgentContract.skill_ids`)

Skills are **composable packs** (tools + prompt instruction ids + optional policy fragments). The LLM does not call skills directly — Nexus resolves `skill_ids` into `allowed_tools` at register time.

#### Enabled today (LKW.0)

| Bundle | `skill_bundles` | Purpose |
|--------|-----------------|---------|
| `harness` | `["harness"]` on `ApplicationEnvironmentProfile` | Platform smoke packs (`harness.tool_smoke`, `harness.trace_read`, …) — harness validation only |

Environment: [`manifest.py`](manifest.py) · [`host/environment_profile.py`](host/environment_profile.py).

#### Planned domain bundle (LKW.2) — `intergrax/skills/providers/local/`

| `skill_id` | Agent | `tool_ids` | `prompt_instruction_ids` |
|------------|-------|------------|----------------------------|
| `local.workspace.index` | `local_indexer` | `rag.ingest_document`, `document.parse`, `rag.list_collections` | `local.workspace.index.system` |
| `local.workspace.search` | `local_search` | `rag.retrieve`, `rag.list_collections`, `cache.get`, `cache.set` | `local.workspace.search.system` |
| `local.workspace.synthesize` | `local_synthesizer` | `workspace.read_file`, `workspace.write_file`, `workspace.list_files`, `workspace.search`, `memory.read` | `local.workspace.synthesize.system` |
| `local.workspace.pipeline` | graph intent (all three) | union of above (via `requires_skills`) | orchestration prompt refs |

**Agent wiring (LKW.2):** each `AgentContract` gains `skill_ids=[...]`; register via `registry.register(agent, skill_registry=..., tool_registry=...)`. Until then, agents use scaffold `skills=[]` and rely on host `ToolProfile` only.

Skill authoring: [`docs/SKILLS.md`](../../docs/SKILLS.md) · Appendix J in [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane).

### 5.4 Per-agent Integration / Tool / Skill matrix

| Agent | Integrations consumed (indirect) | Primary tools | Skill (LKW.2) |
|-------|----------------------------------|---------------|---------------|
| **LocalIndexerAgent** | `document_parser`, `vector_store`, embedding managers | `rag.ingest_document`, `document.parse`, `rag.list_collections` | `local.workspace.index` |
| **LocalSearchAgent** | `vector_store`, `rerank_provider` | `rag.retrieve`, `cache.*`, `memory.*` | `local.workspace.search` |
| **LocalSynthesizerAgent** | runtime shadow workspace (not integration slug) | `workspace.*`, `memory.read` | `local.workspace.synthesize` |

### 5.5 Runtime wiring path (Tier-3 → Tier-1 → Tier-2)

```text
wire_application_environment(manifest, environment, settings)
  ├── bootstrap_application_integration_catalog()
  ├── probe_integration_profile_health()
  ├── resolve_rag_stack_for_environment()     # ContextProfile.enable_rag=true
  ├── build_application_tool_wiring()           # ToolProfile → ToolRegistry
  ├── build_application_skill_wiring()        # SkillProfile → SkillRegistry
  └── ApplicationBuildContext                 # passed to agent factories

build_application_registry(manifest, build_context, builders)
  └── register agents; resolve skill_ids → allowed_tools

NexusLoop → AgentEngine → UAEP ctx.invoke_tool(ToolRequest(tool_name="rag.retrieve", ...))
```

### 5.6 Environment profile summary

- `ApplicationEnvironmentProfile.product_defaults(profile_id="local_workspace.product")`
- `skill_bundles=["harness"]` (LKW.0); extend with `"local"` at LKW.2
- `integration_profile=IntegrationProfile.legal_product()`
- `ContextProfile(enable_rag=True, enable_websearch=False)`
- `with_harness_memory()` — STM/LTM hooks for long sessions
- OTLP optional on `observability_profile` + `IntegrationProfile` OTEL slot

See [`host/environment_profile.py`](host/environment_profile.py).

---

## 6. Local OS runtime and interaction model

LKW is a **local execution environment** (Tier-3 host + Nexus) that runs **in the background on the user's machine** (Windows, Linux, macOS). The user can submit work **at any time**; agents are spawned by Nexus on demand. Chat apps (Slack, Teams) are **interaction surfaces** — not the runtime — they deliver commands and receive summaries.

### 6.1 Design principle: compute local, control multi-channel

| Layer | Where it runs | What it holds |
|-------|---------------|---------------|
| **Execution** | User OS (localhost) | Nexus, agents, RAG index, shadow workspace, file access |
| **Interaction** | Slack / Teams / HTTP / MCP / tray (optional) | Commands, status, HITL prompts, short answers |
| **Cloud** | Optional (Slack API, LLM API) | Messaging + inference only — **not** user file storage |

**Privacy default:** document chunks and embeddings stay on disk under user control. Slack receives **commands and condensed answers**, not full file dumps.

### 6.2 Background process topology

```text
┌──────────────────────────────────────────────────────────────────────────┐
│  User workstation (Windows / Linux / macOS)                              │
│                                                                          │
│  ┌─────────────────────┐    ┌──────────────────────┐                   │
│  │ LKW Host (always-on) │    │ Indexer worker (opt) │                   │
│  │ local_workspace_app  │    │ file watcher + queue │                   │
│  │ :8020 localhost      │    │ Wave LKW.7           │                   │
│  └──────────┬──────────┘    └──────────┬───────────┘                   │
│             │                          │                                 │
│             ▼                          ▼                                 │
│  ┌─────────────────────────────────────────────────────────────┐        │
│  │ Local data plane                                             │        │
│  │ Chroma/SQLite index · shadow_workspaces/ · trace DB          │        │
│  └─────────────────────────────────────────────────────────────┘        │
│             ▲                                                            │
│             │ POST /v1/local_workspace/run                               │
│             │ POST /v1/interactions/intake  (Slack / Teams / JSON)       │
│             │ MCP /v1/...                                                │
│  ┌──────────┴──────────┐   ┌─────────────┐   ┌──────────────────┐     │
│  │ Tray / CLI (LKW.8)  │   │ Cursor MCP  │   │ Slack / Teams    │     │
│  │ (optional UI)       │   │ (local)     │   │ (remote surface) │     │
│  └─────────────────────┘   └─────────────┘   └────────┬─────────┘     │
└──────────────────────────────────────────────────────────┼───────────────┘
                                                           │
                                              Slack Socket Mode or HTTPS tunnel
                                              (outbound from daemon — no public IP required)
```

#### OS service packaging (LKW.6)

| OS | Recommended mechanism | Notes |
|----|----------------------|-------|
| **Windows** | Windows Service or scheduled task + tray helper | User-session for file access; avoid SYSTEM account for home-folder indexing |
| **Linux** | `systemd` user unit (`lkw.service`) | `After=network.target`; restart on failure |
| **macOS** | `launchd` LaunchAgent (`~/Library/LaunchAgents/`) | Full Disk Access may be required for user folders |

Host entrypoint (today): `uvicorn local_workspace_application.host.main:app`. Production packaging: single binary or `uv run` wrapper in service unit.

#### Always-on responsibilities

1. **Listen** for user tasks (HTTP, MCP, interaction intake).
2. **Maintain** local RAG index (background ingest — LKW.7).
3. **Run** Nexus graph on demand (search now, synthesize on request).
4. **Notify** on completion / HITL (`notification_channel=slack` on long-running tasks).
5. **Persist** checkpoints for pause/resume ([`docs/INTERGRAX_IMPLEMENTATION_PLAN.md` Appendix F.4](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md)).

### 6.3 Interaction surfaces (how the user talks to LKW)

| Surface | Status | Endpoint / mechanism | Best for |
|---------|--------|----------------------|----------|
| **Local HTTP** | Scaffold **Done** | `POST /v1/local_workspace/run` | Scripts, tray, local integrations |
| **Local MCP** | Scaffold **Done** | `/mcp` on same host | Cursor / IDE at desk |
| **Interaction intake** | Platform **Done**; LKW host **planned LKW.6** | `POST /v1/interactions/intake` | Slack slash commands, Teams, lab JSON |
| **Slack outbound** | Platform **Done** | `INTERGRAX_SLACK_WEBHOOK_URL`, HITL templates | Alerts, approvals, result snippets |
| **Debug CLI** | Platform **Done** | `python -m intergrax.debug` | Operators |
| **Tray / native UI** | **Deferred LKW.8** | Calls localhost HTTP/MCP | Folder picker, status icon |

**Rule:** every surface normalizes to a Nexus `Task` — same agents, same policy, same trace. See [`applications/USAGE.md` §4b](../USAGE.md) · canon §18.

### 6.4 Slack as interaction channel — recommended architecture

**Yes — Slack fits LKW**, using existing Intergrax **interaction + notification** integrations (`slack` slug). Slack is **not** the execution environment; the **local LKW daemon** is.

#### Reference flow (slash command)

```text
User in Slack:  /lkw search dokumenty o projekcie Alpha
       │
       ▼
Slack Events API  ──►  (A) Socket Mode client in LKW daemon   [preferred: no inbound port]
                    or (B) HTTPS tunnel → localhost:8020/v1/interactions/intake
       │
       ▼
InteractionIntakeService  +  SlackInteractionAdapter
       │  verify signature · parse slash payload · map text → Task
       ▼
NexusLoop.handle_task(capability=local.workspace.search, message=...)
       │
       ▼
LocalSearchAgent → rag.retrieve (local Chroma index)
       │
       ▼
Reply to Slack (response_url / chat.postMessage) — citations + short summary only
```

**Platform primitives to reuse (no Nexus fork):**

| Primitive | Module / doc | LKW use |
|-----------|--------------|---------|
| `InteractionIntakeService` | `runtime/interactions/intake_service.py` | Inbound Slack → `Task` |
| `SlackInteractionAdapter` | `integrations/providers/notification_channel/slack/` | Channel id `slack` |
| `wire_interaction_intake_service` | `applications/_shared/interaction_wiring.py` | Enable on `local_workspace_application` factory |
| `TaskLongRunningOptions.notify_channel="slack"` | plan Appendix F.4 | HITL + long ingest jobs |
| Organization worker runbook | plan §H.6 | Prior art for slash → Nexus → resume |

**Example intake (lab-equivalent, today):**

```bash
curl -s -X POST "http://127.0.0.1:8020/v1/interactions/intake?execute=true&tenant=U1" \
  -H "Content-Type: application/json" \
  -d '{"command":"/lkw","text":"search projekt Alpha","user_id":"U1","team_id":"T1"}'
```

LKW.6 wires this on the product host (mirror `lab_application` / `legal_application` interaction flags).

#### Slack connectivity modes

| Mode | Pros | Cons | LKW recommendation |
|------|------|------|---------------------|
| **Socket Mode** | No public URL; daemon initiates outbound WebSocket | Requires Slack app + bot token in local config | **Default for desktop daemon** |
| **HTTPS tunnel** (ngrok, Cloudflare Tunnel) | Quick dev | Extra dependency; URL rotation | Dev / demo only |
| **Slack notifications only** | Simple webhook | User cannot command from Slack | Phase 1 fallback — local MCP/HTTP for commands, Slack for HITL |

#### Slack command mapping (convention)

| User text (after `/lkw`) | `Task.context.capability` | Agent |
|--------------------------|----------------------------|-------|
| `index <path>` | `local.workspace.index` | `local_indexer` |
| `search <query>` | `local.workspace.search` | `local_search` |
| `draft email\|report\|estimate …` | `local.workspace.synthesize` | `local_synthesizer` |
| free text (default) | `local.workspace.pipeline` | graph (LKW.2) |

`tenant_id` / `user_id` from Slack identity map to Intergrax task scope for memory and index partitions.

#### What must NOT go through Slack

- Raw file uploads containing full document corpora (use local index instead).
- Shadow workspace binary artifacts (link to local export path or summary).
- Unredacted secrets from parsed files.

### 6.5 Task timing: foreground vs background

| Pattern | Trigger | Nexus behaviour |
|---------|---------|-----------------|
| **Interactive** | User message (Slack, HTTP, MCP) | Sync or async run; reply when `COMPLETED` or `WAITING_FOR_HUMAN` |
| **Background index** | File watcher / cron (LKW.7) | `message_bus.enqueue` → worker runs `local.workspace.index`; Slack notify on batch complete |
| **Long-running synthesize** | Large report | `TaskLongRunningOptions` + checkpoint; user resumes via Slack `approve` / HTTP |

User can **always** submit a new interactive task while background indexing runs — Nexus queue + idempotency prevent duplicate ingests (LKW.7).

### 6.6 Integration profile extension for Slack (LKW.6)

Extend `IntegrationProfile` on LKW host (in addition to `legal_product()` RAG slots):

```text
notification_channel = slack    # HITL + completion alerts
interaction_surface  = slack    # inbound slash / events (via intake router)
```

Env (mirror legal/lab): `LOCAL_WORKSPACE_INCLUDE_INTERACTIONS=true`, `LOCAL_WORKSPACE_INTERACTION_SURFACE=slack`, Slack signing secret + bot token for Socket Mode.

### 6.7 Implementation waves (product capabilities)

| Wave | Deliverable |
|------|-------------|
| **LKW.1** | Domain UAEP steps; explicit-path ingest + search |
| **LKW.2** | Multi-agent graph + `local.workspace.*` skills |
| **LKW.3** | `filesystem.*` read tools + allowlist |
| **LKW.6** | Always-on local daemon packaging (systemd / launchd / Windows) + interaction intake on host |
| **LKW.6b** | Slack Socket Mode client + slash command mapping |
| **LKW.7** | Background file watcher + incremental index + optional Slack batch notifications |
| **LKW.8** | Tray / folder-picker UI (calls localhost API) |

---

## 7. Request and data flows

### 7.1 Index flow

```text
Task(capability=local.workspace.index, metadata={source_paths: [...]})
  → LocalIndexerAgent UAEP steps
  → invoke rag.ingest_document per path
  → ParserPipeline + chunk + embed + vector store
  → StepOutput(metadata: {num_chunks, collection_id, parser_trace})
```

### 7.2 Search flow

```text
Task(capability=local.workspace.search, message="znajdź dokumenty o projekcie X")
  → LocalSearchAgent
  → rag.retrieve(query, metadata filters)
  → Package evidence chunks + citations (path, page, chunk_id)
```

### 7.3 Synthesize flow

```text
Task(capability=local.workspace.synthesize, metadata={template: "email"|"report"|...})
  → LocalSynthesizerAgent
  → LLM with retrieved context (from graph handoff or prior step)
  → workspace.write_file("draft.md", content)
  → metadata: {shadow_workspace_id, artifact_paths}
```

### 7.4 Pipeline flow (Wave 2)

```text
Task(capability=local.workspace.pipeline, intent=local_workspace_full)
  → Nexus graph: DELEGATES_TO indexer? → search → synthesizer
  → SharedTaskContext carries evidence + artifact refs
```

---

## 8. Tier-3 composition map

| File | Role |
|------|------|
| [`manifest.py`](manifest.py) | Roster, capabilities, `LOCAL_WORKSPACE_APPLICATION_MANIFEST` |
| [`host/environment_profile.py`](host/environment_profile.py) | RAG-on, websearch-off product profile |
| [`host/tool_wiring.py`](host/tool_wiring.py) | LKW tool allowlist |
| [`host/settings.py`](host/settings.py) | `LOCAL_WORKSPACE_*` env, RAG flags |
| [`host/wiring.py`](host/wiring.py) | Registry + `wire_application_environment` |
| [`host/factory.py`](host/factory.py) | FastAPI Core + MCP |
| [`serving/fastapi_router.py`](serving/fastapi_router.py) | `/run`, `/agents` |
| [`mcp/server.py`](mcp/server.py) | FastMCP mount |

**No agent logic in Tier-3** — only wiring. Domain steps live in `agents/*/steps/`.

Authoring guide: [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) · Tier-3: [`applications/USAGE.md`](../USAGE.md).

---

## 9. Configuration

### 9.1 Environment variables (`.env.example`)

| Variable | Default | Purpose |
|----------|---------|---------|
| `LOCAL_WORKSPACE_BACKEND_PORT` | `8020` | HTTP port |
| `LOCAL_WORKSPACE_DEFAULT_AGENT_ID` | `local_search` | Default roster agent |
| `LOCAL_WORKSPACE_ENABLE_RAG` | `true` | Enable `rag.retrieve` |
| `LOCAL_WORKSPACE_ENABLE_RAG_INGEST` | `true` | Enable `rag.ingest_document` |
| `INTERGRAX_SHADOW_ROOT` | `build/shadow_workspaces` | Artifact isolation root |
| `INTERGRAX_ALLOWED_READ_ROOTS` | *(planned Wave 3)* | Comma-separated read allowlist |

### 9.2 Task metadata conventions (Wave 1+)

```python
Task(
    metadata={
        "shadow_workspace": True,
        "source_paths": ["D:/Docs/project_a/report.pdf"],
        "collection_id": "user_u1_workspace",
        "synthesis_template": "email",
    }
)
```

---

## 10. Security and governance

- **Read-only user FS** in Waves 1–2 (ingest reads; no writes)
- **Shadow workspace** mandatory for synthesizer outputs
- **HITL** optional for sensitive exports (`REQUEST_HUMAN`) — [`docs/AGENT_CREATION_GUIDE.md` Appendix A](../../docs/AGENT_CREATION_GUIDE.md#appendix-a--human-in-the-loop)
- **Cost governance:** `CostProfile` on environment; embedding batch limits per ingest job
- **Trace:** all tool calls via Nexus trace DB — debug with `intergrax.debug` CLI

---

## 11. Observability and verification

```bash
# Host smoke
uv run pytest applications/local_workspace_application/local_workspace_application_tests -q

# Agent smoke
uv run pytest agents/local_indexer/tests agents/local_search/tests agents/local_synthesizer/tests -q

# Run host
uv run uvicorn local_workspace_application.host.main:app --host 127.0.0.1 --port 8020
```

Deploy triad: `docker/`, `BUILD_AND_DEPLOY.md` — gate `test_application_deploy_triad.py`.

---

## 12. Implementation waves (plan traceability)

| ID | Deliverable | Status |
|----|-------------|--------|
| **LKW.0** | Scaffold agents + application + architecture doc | **Done** (this document) |
| **LKW.1** | Wave 1 — single-path ingest + search smoke | Planned |
| **LKW.2** | Multi-agent graph + pipeline capability | Planned |
| **LKW.3** | Tier-0 `filesystem.*` read tools + allowlist policy | **Done** (T6) |
| **LKW.4** | Background ingest queue (message_bus worker) | Planned |
| **LKW.6** | Local OS daemon + interaction intake on host | Planned |
| **LKW.6b** | Slack Socket Mode + slash → Nexus mapping | Planned |
| **LKW.7** | File watcher + incremental index + Slack batch notify | Planned |
| **LKW.8** | Tray / file-picker UI (localhost client) | Deferred product |

Registered in [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md` §6.3a](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md#63a-business-backlog-register-consolidated).

---

## 13. Known platform gaps (honest audit)

| Gap | Impact | Mitigation (Wave) |
|-----|--------|-------------------|
| No file watcher | No auto re-index | LKW.4 worker |
| In-memory vector store default | Index lost on restart | Chroma + `INTEGRATION_PROFILE_JSON` |
| Windows path / OneDrive edge cases | Parser failures | Test matrix in LKW.1 acceptance |
| Qdrant/Chroma lack `list_document_ids` | `rag.list_documents` empty/unsupported on some backends | Use InMemory for dev; extend provider bindings in follow-up |

These gaps are **expected** — LKW exists to discover and close them without Nexus forks.

---

## 14. References

| Topic | Document |
|-------|----------|
| Agent workflow | [`docs/AGENT_CREATION_GUIDE.md`](../../docs/AGENT_CREATION_GUIDE.md) |
| Integration catalog | [`docs/INTEGRATIONS.md`](../../docs/INTEGRATIONS.md) |
| Tools catalog | [`docs/TOOLS.md`](../../docs/TOOLS.md) |
| Skill Library | [`docs/SKILLS.md`](../../docs/SKILLS.md) |
| Tools & skills control plane | [`docs/AGENT_CREATION_GUIDE.md` Appendix J](../../docs/AGENT_CREATION_GUIDE.md#appendix-j--tools--skills-control-plane) |
| RAG control plane | [`docs/AGENT_CREATION_GUIDE.md` Appendix K](../../docs/AGENT_CREATION_GUIDE.md#appendix-k--integration--rag-control-plane) |
| Shadow workspace | [`docs/AGENT_CREATION_GUIDE.md` Appendix B](../../docs/AGENT_CREATION_GUIDE.md#appendix-b--shadow-workspace-and-sandbox) |
| Multi-agent graphs | [`docs/AGENT_CREATION_GUIDE.md` Appendix C](../../docs/AGENT_CREATION_GUIDE.md#appendix-c--multi-agent-graphs) |
| Nexus execution flow | [`docs/NEXUS_EXECUTION_FLOW_REFERENCE.md`](../../docs/NEXUS_EXECUTION_FLOW_REFERENCE.md) |
| Implementation plan | [`docs/INTERGRAX_IMPLEMENTATION_PLAN.md`](../../docs/INTERGRAX_IMPLEMENTATION_PLAN.md) |
| Quickstart | [`README.md`](README.md) · [`BUILD_AND_DEPLOY.md`](BUILD_AND_DEPLOY.md) |

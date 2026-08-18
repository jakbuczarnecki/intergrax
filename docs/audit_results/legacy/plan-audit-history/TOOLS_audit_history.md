> **Migrated (AUDIT-PROTOCOL-RESET-R2):** Historical plan-satellite audit register.
> **Original path:** docs\project\maintainers\plans\satellites\TOOLS_implementation_history.md
> **Original role:** Plan satellite — audit history + LC closeout
> **Canonical audit ownership:** docs/audit_results/ (this file is historical evidence only)

# TOOLS — audit history + LC closeout

**Parent hub:** [`TOOLS.md`](../TOOLS.md)

## Phase LEG — Legacy tool plan boolean closeout

**Status:** **Done** (2026-06-02) — **3/3** deliverables Done (LEG-1–2); gate **612 passed**

**Audit basis:** Phase O.5a residual; `check_legacy_tool_plan_booleans.py`; Appendix J §J.6.

**Priority ladder:** **Band 2o** (§4.0) — closed; default queue = **§6.1** maintenance.

### LEG — Master register

| ID | Area | Deliverable | Status | Modules | Acceptance |
|----|------|-------------|--------|---------|------------|
| LEG-1 | LEG1 | **`tool_invocation_plan_from_capability_payload`** — gateway maps booleans → `tool_ids` without `from_legacy` | **Done** | `tool_runtime.py`, `tool_gateway.py` | `test_capability_payload_tool_plan.py` |
| LEG-2 | LEG2 | **Engine planner `tool_ids`** — parser populates `EnginePlan.tool_ids`; schema optional `tool_ids` | **Done** | `engine_planner_parse.py`, `nexus_llm_plan_builder.py` | `test_engine_plan_json_parser.py` |
| LEG-3 | LEG3 | **`plan_from_like` canonical path** — `from_tool_ids` only; `tool_gateway` removed from audit grandfather | **Done** | `tool_runtime.py`, `check_legacy_tool_plan_booleans.py` | audit script green |

**Residual:** `ToolInvocationPlan.from_legacy()` retained in `tool_runtime.py` for explicit deprecation tests only; `EnginePlan.use_rag`/`use_websearch` remain on LLM schema for backward-compatible planner output.

---

---

## Phase TS — Tools & skills control plane closeout

**Status:** **Done** (2026-06-02) — **5/5** deliverables Done (TS-DOC.* + TS-1–3); gate **589 passed**

**Audit basis:** [`INTEGRAX_HARNESS_AUDIT_MAP.md`](guides/INTEGRAX_HARNESS_AUDIT_MAP.md) §11–§12; author map: `guides/AGENT_CREATION_GUIDE.md` **Appendix J**.

**Priority ladder:** **Band 2k** (§4.0) — closed; default queue = **§6.1** maintenance.

**Execution order:** [§6.2bc](.#62bc-phase-ts-execution-order-band-2k--closed) · queue: [§6.1c](.#61c-harness-implementation-queue--toolsskills-closeout-closed)

**Delivery rule:** One **TS-*** ID per PR → update master table + §6.1c + paydown log below → `pytest -m gate` + §6.1 scripts green.

### TS — Master register

| ID | Area | Deliverable | Status | Priority | Modules | Acceptance |
|----|------|-------------|--------|----------|---------|------------|
| TS-DOC.1 | TS0 | **Appendix J** — tools & skills control plane map (§J.1–J.7) | **Done** | High | `guides/AGENT_CREATION_GUIDE.md` | TOC + verification table |
| TS-DOC.2 | TS0 | **Cross-ref sync** — plan, README, AUDIT_MAP §11–§12, audit prompt ref #7 | **Done** | Medium | `docs/*` | Links resolve |
| TS-1 | TS1 | **`catalog_runtime_bridge.py`** — `tool_profile` / `skill_profile` on `RuntimeConfig` via `materialize_runtime_config` | **Done** | **Critical** | `catalog_runtime_bridge.py`, `runtime_config_bridge.py`, `config.py` | `test_catalog_runtime_bridge.py` |
| TS-2 | TS2 | **Harness host LLM wiring** — `resolve_llm_adapter(env)` → `build_nexus_loop_from_environment` | **Done** | High | `harness_host_runtime.py` | `test_harness_host_runtime_llm.py` |
| TS-3 | TS3 | **`SkillResolverProtocol`** — typed contract for skill composition resolution | **Done** | Medium | `skills/resolver.py`, `contract_resolution.py` | existing skill resolver tests green |

**Residual (not TS scope — track separately):** legacy `use_rag`/`use_websearch` booleans in `engine_planner` / `tool_gateway` (deprecation warnings; `check_legacy_tool_plan_booleans.py`).

### TS — Paydown log

| Date | TS ID | Summary |
|------|-------|---------|
| 2026-06-02 | TS-DOC.1, TS-DOC.2 | Appendix J + cross-refs; AUDIT_MAP §11–§12 authoring map |
| 2026-06-02 | TS-1, TS-2, TS-3 | Catalog runtime bridge, harness LLM wiring, SkillResolverProtocol; gate **589** |

**Phase TS complete when:** TS-1–3 + TS-DOC.* **Done**; §6.1c queue closed; Appendix J has no “planned wiring” gaps; gate **589** green. **Status: complete (2026-06-02).**

---

## Phase TOOL-ENG-DOC — Tool engine documentation canon (Band 2ar / 2bb)

**Status:** **Done** (2026-06-12) — **7/7** DOC rows · pipeline · selection modes · invocation patterns · selection plugin · graph boundary  
**Prerequisites:** Phase TS **Done** · Phase O **Done** · Phase LEG **Done**  
**Goal:** Canon in [`architecture/TOOLS.md`](../architecture/TOOLS.md) for selection (L6), orchestration (2a), atomic invoke (2b), logging — plus plugin extensibility  
**ADR:** **No ADR needed** for DOC rows; implementation rows TOOL-ENG-13/14/16/26 require ADR at code merge

| ID | Deliverable | Status | Priority | Module / doc | Acceptance |
|----|-------------|--------|----------|--------------|------------|
| TOOL-ENG-DOC.1 | **Tool execution pipeline** — diagram + phase table + entry paths | **Done** | Critical | `architecture/TOOLS.md` | select → orchestrate → invoke → log |
| TOOL-ENG-DOC.2 | **Component naming** — Tool engine vs `ToolRuntime` | **Done** | High | same | §Tool engine table |
| TOOL-ENG-DOC.3 | **Cross-ref sync** — FLOW §15, AUDIT_MAP §11, Appendix J | **Done** | Medium | `docs/*` | Links resolve |
| TOOL-ENG-DOC.4 | **Selection modes** — standard / semantic / hierarchical | **Done** | Critical | `architecture/TOOLS.md`, FLOW §15 | §modes |
| TOOL-ENG-DOC.5 | **Invocation patterns** — single / parallel / ReAct / chain / graph boundary | **Done** | Critical | `architecture/TOOLS.md`, FLOW §15.1, ORCH §50.4 | §patterns |
| TOOL-ENG-DOC.6 | **Graph vs tool-pattern boundary** | **Done** | High | `ORCHESTRATION.md`, `NEXUS_EXECUTION_FLOW.md` | §50.4 + §15.1 |
| TOOL-ENG-DOC.7 | **Selection plugin model** — `ToolSelectionStrategy`, surfaces A/B/C | **Done** | Critical | `architecture/TOOLS.md` | §selection plugin |

### TOOL-ENG-DOC traceability

| Pipeline phase | Canon section | Runtime modules |
|----------------|---------------|-----------------|
| Selection L6 | §[modes](../architecture/TOOLS.md#tool-selection-modes-production-strategies) · §[plugin model](../architecture/TOOLS.md#tool-selection-plugin-model-l6-extensibility) · FLOW §15 | `ToolSelectionStrategy`, `resolve_planner_allowed_tool_ids` |
| Planning L6b | §Multi-tool execution · §patterns | `ToolPlanningService`, `ToolPlannerProtocol` |
| Orchestration 2a | §[Invocation patterns](../architecture/TOOLS.md#tool-invocation-patterns-production-orchestration) · FLOW §15.1 | `ToolInvocationPattern` **Done** (TOOL-ENG-16), `run_bounded_tool_loop` / `resolve_invocation_pattern()` |
| Atomic invoke 2b | §pipeline · §42.12 gateway | `RuntimeToolInvoker`, `ToolRuntime` |
| Logging | §pipeline · FLOW §17 · OBS | `trace_event`, `TOOL_*`, `run_bounded_tool_loop` / `ctx.invoke_tool` |
| Gaps | §[Engine gap register](../architecture/TOOLS.md#engine-gap-register-canon) | Phase **TOOL-ENG** master register |

### TOOL-ENG-DOC — Paydown log

| Date | ID | Summary |
|------|-----|---------|
| 2026-06-08 | TOOL-ENG-DOC.1–3 | Tool execution pipeline §; cross-refs |
| 2026-06-11 | TOOL-ENG-DOC.4 | Selection modes canon; TOOL-ENG-13/14/15 |
| 2026-06-12 | TOOL-ENG-DOC.5–7 | Invocation patterns + selection plugin + ORCH/FLOW boundary |

---

### 6.1d Harness implementation queue — tool engine docs (closed)

**Purpose:** Phase **TOOL-ENG-DOC** (Band 2ar) documentation closeout. **Closed 2026-06-08**.

| Order | ID | Type | Status | Deliverable | Acceptance |
|-------|-----|------|--------|-------------|------------|
| 1 | **TOOL-ENG-DOC.1–2** | Docs | **Done** | Architecture pipeline § + naming | Select / invoke / log covered |
| 2 | **TOOL-ENG-DOC.3** | Docs | **Done** | Cross-ref sync | FLOW, AUDIT_MAP, Appendix J |

---

---

### Phase O — Tool Library & Unified Tool Model (Tier-0)

**Canon:** §7.1.6–§7.1.7, §22, §42.12  
**Goal:** Ship a reusable **Tool Library** catalog (mirror Integration Library) and migrate legacy pipeline flags (`use_rag`, `use_websearch`) to explicit catalog tools.

**Prerequisite:** Phase M.3 (`IntegrationProfile`) available; tool engine (`ToolRegistry`, `RuntimeToolInvoker`) exists.

**Catalog reference:** [`architecture/TOOLS.md`](architecture/TOOLS.md)

**Delivery rule:** One domain or migration slice per iteration — implement → gate → update `architecture/TOOLS.md` → next step.

| # | Deliverable | Status | Canon | Notes |
|---|-------------|--------|-------|-------|
| O.0 | Architecture & catalog documented | **Done** | §7.1.6–§7.1.7, §22 | Runtime canon + `architecture/TOOLS.md` + this section (2026-05-30) |
| O.1 | Extended `ToolContract` | **Done** | §22 | `ToolRiskLevel`, `ToolRetryPolicy`, metadata fields; invoker timeout/retry/trace (2026-05-30) |
| O.2 | `ToolCatalog` + `ToolProfile` + `ToolWiringContext` | **Done** | §7.1.6 | `intergrax/tools/registry`; `build_registry_from_profile`; RuntimeConfig wiring (2026-05-30) |
| O.3 | Context tools: `rag.retrieve`, `websearch.query` | **Done** | §7.1.7, §22.1 | `providers/rag`, `providers/websearch` (2026-05-30) |
| O.4 | Reference domain: `jira.*` tools | **Done** | §7.1.6 | `get_issue`, `add_comment`, `search_tasks` over `IssueTracker` (2026-05-30) |
| O.4b | Catalog domain bundles: `confluence.*`, `notify.send`, observability, `sandbox.exec` | **Done** | §7.1.6 | All first-party catalog tools registered (2026-05-30) |
| O.5 | **Unified tool model migration** | **Done** | §7.1.7, §22.2 | `tool_ids` on plans; RagStep/WebsearchStep → catalog shims (2026-05-30) |
| O.6 | Schema exporters (OpenAI + MCP) | **Done** | §7.1.6 | `tools/exporters`; MCP catalog mount on lab/poc_template (2026-05-30) |
| O.7 | Migrate legacy `ToolBase` → `ToolContract` | **Done** | §5.2.2 | `ChatAgent` → registry; `tools_base` deprecated (2026-05-30) |
| O.8 | `ToolProfile` in Tier-3 scaffold | **Done** | §7.4.8 | `tool_wiring.py` template; lab + poc_template reference (2026-05-30) |
| O.9 | Agent Creation Guide Appendix E update | **Done** | — | Unified model + ToolProfile examples (2026-05-30) |
| O.10 | Gate tests for catalog conformance | **Done** | — | `tests/unit/tools/providers` — all catalog bundles (2026-05-30) |
| O.11 | Phase P wave 2 context tools: `websearch.read_url`, `confluence.search` | **Done** | §7.1.7, §22.1 | `providers/websearch/read_url_*`, confluence alias (2026-05-30) |
| O.12 | Phase P wave 3 tools: `websearch.fetch_batch`, `rag.list_collections`, `observability.query_traces` | **Done** | §7.1.7, §22.1 | Extended `ObservabilityBackend.query_traces`, vector `list_collections` (2026-05-30) |

#### O — Step-by-step implementation sequence

Execute **strictly in order** for foundation (O.1–O.4); O.5–O.10 may overlap after O.4 reference tools land.

| Step | ID | Action | Done when |
|------|-----|--------|-----------|
| 1 | O.1 | Extend `ToolContract` + update `RuntimeToolInvoker` for new fields | Unit tests pass; backward compatible defaults |
| 2 | O.2 | Add `tools/registry/catalog.py`, `profile.py`, `ToolWiringContext` dataclass | `register_default_tools()` no-op registry; profile enables subset |
| 3 | O.3 | Implement `providers/rag` and `providers/websearch` handlers | **Done** — `rag.retrieve`, `websearch.query` + tests |
| 4 | O.4 | Implement `providers/jira` bundle (3 tools) | **Done** — conformance tests with mocked `IssueTracker` |
| 4b | O.4b | Implement remaining catalog bundles (`confluence`, `notify`, `observability`, `sandbox`) | **Done** — all tool_ids in `register_default_tools()` |
| 5 | O.5a | Add `tool_ids` to plan models; map legacy booleans → tool_ids | **Done** — `ToolInvocationPlan`, `LegalToolPlan` |
| 6 | O.5b | `rag.retrieve` (catalog) / `websearch.query` (catalog) delegate to catalog tools | **Done** — `catalog_context.py` shim |
| 7 | O.5c | Update `LegalToolPlan` / engine plans to tool list | **Done** — bridge passes `tool_ids` |
| 8 | O.6 | MCP + OpenAI exporters from single catalog | **Done** — `tools/exporters` |
| 9 | O.7 | Remove `ToolBase` usage from production paths | **Done** — `ChatAgent` uses registry `ToolRegistry` |
| 10 | O.8–O.10 | Scaffold, docs, gate | **Done** |

#### O.4 — Adding a new tool provider (checklist)

Copy into every `tools/providers/<domain>/USAGE.md`:

```text
[ ] 1. Define Input/Output Pydantic models (LLM-friendly field names)
[ ] 2. Implement ToolHandler — compose integration contract(s), no vendor SDK
[ ] 3. Build ToolContract per tool (description tuned for model selection)
[ ] 4. register_<domain>_tools(registry, ctx: ToolWiringContext)
[ ] 5. Register in tools/registry/catalog.py
[ ] 6. Unit tests with fakes (no live vendor in default gate)
[ ] 7. Wire in lab or poc_template via ToolProfile
[ ] 8. Update architecture/TOOLS.md status + this plan tracker
```

#### T-EXPAND — Integration bridge catalog expansion (2026-06-07) — **Done**

**Goal:** Close the integration→tool coverage gap (~78% integrations without LLM tools) by shipping provider-agnostic bundles that compose existing `IntegrationCategory` contracts.

| Wave | Bundles | Tools | Status |
|------|---------|------:|--------|
| T1 (DX / runtime-bound) | `workspace`, `memory`, `knowledge`, `document`, `browser`, `storage` (get) | 12 | **Done** |
| T2 (prod harness) | `storage` (+put/presigned/delete), `issues`, `platform` | 10 | **Done** |
| T3 (async / graph / collab / cache) | `message_bus`, `graph`, `collaboration`, `cache` | 8 | **Done** |

**Delivered:**

- **67** catalog `tool_id` values · **28** shipped bundles (`shipped_plugins.py`)
- Typed `ToolWiringContext` slots for all new integration categories
- `TaskMemoryViewBinding` protocol (avoids Tier-0 ↔ UAEP import cycle)
- UAEP `runtime_bound_catalog.py` for `workspace.*` / `memory.*` (mirrors `sandbox.exec`)
- `extend_tool_profile_for_integration()` P6 auto-enable (excludes ingest-only `document_parser`)
- Gate: **909** passed (`uv run pytest -m gate -q`)

**Follow-up (2026-06-07) — Done:**

- `IssueCreator` protocol + `issues.create_issue` (no `getattr` in GitLab tool path)
- `harness.integration_bridge_smoke` skill pack + resolver test fix (skills vs tools `build_registry_from_profile`)
- Lab harness `wire_lab_tools(harness=True)` enables runtime-bound + bridge tools
- PoC template `extend_tool_profile_for_integration()` wiring
- MCP full-catalog export smoke (130 tools)

#### T-EXPAND T4 — Agent Builder Essentials (2026-06-07) — **Done**

**Goal:** Close highest-ROI integration→tool gaps for agent/environment builders (SQL, document JSON, RAG lifecycle, workspace DX, collaboration read path, auto-enable wiring).

| Bundle | Tools | Status |
|--------|------:|--------|
| `database` | `database.query`, `database.execute` | **Done** |
| `records` | `records.get`, `records.put`, `records.delete`, `records.query` | **Done** |
| `rag` (+2) | `rag.delete_documents`, `rag.describe_collection` | **Done** |
| `workspace` (+2) | `workspace.delete_file`, `workspace.search` | **Done** |
| `collaboration` (+4) | `collaboration.list_messages`, `get_message`, `list_calendar`, `get_user` | **Done** |
| wiring | `relational_store` / `document_store` ctx slots; auto-enable notify/obs/database/records/collaboration | **Done** |

**Delivered:** **81** catalog `tool_id` values · **30** shipped bundles.

#### T-EXPAND T5 — Production Harness Ops (2026-06-07) — **Done**

**Goal:** Production harness operations for identity, persisted run trace read, integration health probes, online evaluation registry, and platform/security extensions.

| Bundle | Tools | Status |
|--------|------:|--------|
| `identity` | `identity.verify_token`, `identity.get_user`, `identity.list_tenants` | **Done** |
| `harness` | `harness.get_run`, `harness.list_runs`, `harness.get_run_cost`, `harness.get_run_events` | **Done** |
| `health` | `health.check_integration`, `health.check_profile` | **Done** |
| `eval` | `eval.record_observation`, `eval.list_observations`, `eval.summarize_release` | **Done** |
| `security` (+1) | `security.summarize_findings` | **Done** |
| `platform` (+1) | `platform.put_secret` | **Done** |
| wiring | `trace_reader` / `evaluation_registry` / `integration_profile` ctx slots; runtime-bound `harness.*`; observability bundle promoted STABLE | **Done** |

**Delivered:** **95** catalog `tool_id` values · **34** shipped bundles.

#### T-EXPAND T6 — LKW Filesystem + Harness Economics (2026-06-07) — **Done**

**Goal:** LKW read-only filesystem browse (LKW.3), V-COST/billing tool surface, rerank/cache/CRM/platform extensions.

| Bundle | Tools | Status |
|--------|------:|--------|
| `filesystem` | `filesystem.list`, `filesystem.glob`, `filesystem.read_text`, `filesystem.stat` | **Done** |
| `billing` | `billing.record_usage`, `billing.list_usage` | **Done** |
| `cost` | `cost.get_run_budget`, `cost.check_quota` | **Done** |
| `crm` | `crm.get_account`, `crm.list_contacts`, `crm.list_tickets` | **Done** |
| `platform` (+1) | `platform.delete_secret` | **Done** |
| `rag` (+1) | `rag.rerank` | **Done** |
| `cache` (+2) | `cache.delete`, `cache.list_keys` | **Done** |
| wiring | `read_allowlist_roots` ctx slot; runtime-bound `cost.*`; LKW auto-enable filesystem | **Done** |

**Delivered:** **110** catalog `tool_id` values · **38** shipped bundles.

#### T-EXPAND T7 — Index Lifecycle + Async Queue (2026-06-07) — **Done**

**Goal:** RAG index inspection, async task queue ops, observability range/tail, eval release compare, cost forecast.

| Bundle | Tools | Status |
|--------|------:|--------|
| `message_bus` (+2) | `message_bus.list_tasks`, `message_bus.cancel` | **Done** |
| `rag` (+3) | `rag.list_documents`, `rag.get_document`, `rag.check_index_status` | **Done** |
| `document` (+1) | `document.parse_preview` | **Done** |
| `observability` (+2) | `metrics.query_range`, `logs.tail` | **Done** |
| `eval` (+1) | `eval.compare_releases` | **Done** |
| `cost` (+1) | `cost.forecast_spend` | **Done** |
| contracts | `TaskQueue.cancel` / `list_tasks`; `VectorStoreDocumentListerBinding` | **Done** |
| wiring | auto-enable message_bus + observability extensions; runtime-bound `cost.forecast_spend` | **Done** |

**Delivered:** **120** catalog `tool_id` values · **38** shipped bundles.

#### T-EXPAND T8 — Governance + Agent Safety + LKW write (2026-06-07) — **Done**

**Goal:** Read-only HITL ops, allowlisted filesystem write, RAG metadata search/purge, schema introspection, CI/CD workflow ops.

| Bundle | Tools | Status |
|--------|------:|--------|
| `hitl` (+3, new) | `hitl.list_pending`, `hitl.get_decision`, `hitl.summarize_queue` | **Done** |
| `filesystem` (+1) | `filesystem.write_text` | **Done** |
| `rag` (+2) | `rag.search_by_metadata`, `rag.purge_collection` | **Done** |
| `database` (+1) | `database.describe_schema` | **Done** |
| `records` (+1) | `records.describe_collection` | **Done** |
| `platform` (+2) | `platform.list_workflow_runs`, `platform.cancel_workflow_run` | **Done** |
| contracts | `HumanDecisionStoreBinding`; `CiCdBackend.list/cancel`; `VectorstoreIndexLifecycleBinding.search/purge` | **Done** |
| wiring | LKW auto-enable write + RAG maintenance; integration profile CI/CD + schema tools | **Done** |

**Delivered:** **130** catalog `tool_id` values · **39** shipped bundles.

#### T-EXPAND T9 — Async orchestration + interaction (2026-06-07) — **Done**

**Goal:** Workflow run ops, notify batch, collaboration write-back, websearch cache invalidation, harness run diff/export, interaction session reads.

| Bundle | Tools | Status |
|--------|------:|--------|
| `workflow` (+2) | `workflow.list_runs`, `workflow.cancel_run` | **Done** |
| `notify` (+1) | `notify.send_batch` | **Done** |
| `collaboration` (+2) | `collaboration.reply_message`, `collaboration.create_event` | **Done** |
| `websearch` (+1) | `websearch.invalidate_cache` | **Done** |
| `harness` (+2) | `harness.compare_runs`, `harness.export_run_bundle` | **Done** |
| `interaction` (+2, new) | `interaction.list_sessions`, `interaction.get_last_input` | **Done** |
| contracts | `WorkflowOrchestratorBackend.list/cancel`; `CollaborationSuite.reply/create`; `WebSearchCacheBinding` | **Done** |
| wiring | integration profile workflow/collaboration/notify extensions; `session_storage` via `session_tool_wiring.py` + `SessionStorageToolBinding` | **Done** |

**Delivered:** **140** catalog `tool_id` values · **40** shipped bundles.

**Verification:** `152 passed` (`tests/unit/tools/providers` + exporters) · `check_harness_no_getattr.py` OK · MCP full-catalog export smoke (**140** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{workflow,notify,collaboration,websearch,harness,interaction}`

#### T-EXPAND T10 — LKW storage bridge + deferred scheduling (2026-06-07) — **Done**

**Goal:** Close T8/T9 deferred tools (`workspace.export_artifact`, `notify.schedule`) and extend builder/LKW ops without new bundles.

| Bundle | Tools | Status |
|--------|------:|--------|
| `workspace` (+2) | `workspace.export_artifact`, `workspace.import_artifact` | **Done** |
| `notify` (+1) | `notify.schedule` | **Done** |
| `interaction` (+1) | `interaction.get_session_history` | **Done** |
| `eval` (+1) | `eval.export_observations` | **Done** |
| `storage` (+1) | `storage.exists` | **Done** |
| `memory` (+1) | `memory.delete_key` | **Done** |
| `pagerduty` (+1) | `pagerduty.acknowledge_incident` | **Done** |
| `message_bus` (+1) | `message_bus.purge_completed` | **Done** |
| `records` (+1) | `records.count` | **Done** |
| contracts | `ScheduledNotificationBinding`; `SessionStorageBinding.get_session_history`; `TaskMemoryViewBinding.delete`; `TaskQueue.purge_completed` | **Done** |
| wiring | `notify_tool_wiring.py` + `PolicyScopedMemoryView.delete` | **Done** |

**Delivered:** **150** catalog `tool_id` values · **40** shipped bundles.

**Verification:** `164 passed` (`tests/unit/tools/providers` + exporters) · `check_harness_no_getattr.py` OK · MCP full-catalog export smoke (**150** tools)

**Closeout notes (accepted platform limits):**

| Area | Platform behavior | Product follow-up |
|------|-------------------|-------------------|
| `notify.schedule` | Records deferred delivery in `ScheduledNotificationBinding` (in-memory default via Tier-3 wiring) | Production dispatcher/cron in application host |
| `message_bus.purge_completed` | **Done** — KV task index on broker queues (`rabbitmq`, `kafka`); Celery unchanged | Residual: Celery result-backend purge |
| `pagerduty.acknowledge_incident` | **Done** — `PagerDutyEventsClient.acknowledge_incident` + adapter + typed `PagerDutyIncidentChannel` | — |

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{workspace,notify,interaction,eval,storage,memory,pagerduty,message_bus,records}`

#### T-EXPAND T11 — HITL write path + cloud/vector store ops (2026-06-07) — **Done**

**Goal:** Close T8/T10 deferred governance and integration-bridge gaps without product scope.

| Bundle | Tools | Status |
|--------|------:|--------|
| `hitl` (+2) | `hitl.submit_response`, `hitl.list_for_task` | **Done** |
| `notify` (+2) | `notify.list_scheduled`, `notify.cancel_scheduled` | **Done** |
| `cloud_platform` (new) | `cloud_platform.health`, `cloud_platform.resolve` | **Done** |
| `vector_store` (new) | `vector_store.count`, `vector_store.delete`, `vector_store.list_collections`, `vector_store.health` | **Done** |
| contracts | `HumanDecisionStoreBinding.record` / `list_for_task`; `ScheduledNotificationBinding.cancel_scheduled` | **Done** |
| wiring | `ToolWiringContext.cloud_platform`; `IntegrationProfile` cloud platform resolution | **Done** |

**Delivered:** **160** catalog `tool_id` values · **42** shipped bundles.

**Verification:** provider unit tests + MCP full-catalog export smoke (**160** tools) · `check_harness_no_getattr.py` OK

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{hitl,notify,cloud_platform,vector_store,health}`

#### T-EXPAND T12 — Integration slot health + notify dispatcher (2026-06-07) — **Done**

**Goal:** Close post-T11 harness ops gaps (category health probes, scheduled notify dispatch, Celery purge index).

| Bundle | Tools | Status |
|--------|------:|--------|
| `health` (+9) | `health.check_object_storage`, `health.check_key_value_cache`, `health.check_message_bus`, `health.check_graph_store`, `health.check_identity_provider`, `health.check_relational_store`, `health.check_wiki_knowledge`, `health.check_search_provider`, `health.check_notification_channel` | **Done** |
| `notify` (+1) | `notify.dispatch_due` | **Done** |
| queue | Celery optional KV task index + `purge_completed` | **Done** |
| contracts | `ScheduledNotificationBinding.mark_delivered` | **Done** |
| planner | LEG-DEPTH — remove `use_rag`/`use_websearch` from LLM schema; deprecation trace | **Done** |
| observability | OBS-DEPTH.2 trace bridge phase gate; live emit via `runtime_event_bus` | **Done** |

**Delivered:** **170** catalog `tool_id` values · **42** shipped bundles.

#### T-EXPAND T13 — CRIT-V eval tools (2026-06-07) — **Done**

**Goal:** Ship semantic verification tools for Phase CRIT-V (PEV verify depth) without Nexus orchestrator wiring.

| Bundle | Tools | Status |
|--------|------:|--------|
| `eval` (+2) | `eval.judge`, `eval.trajectory` | **Done** |

**Delivered:** **172** catalog `tool_id` values · **42** shipped bundles.

**Verification:** `test_eval_critic_tools.py` · `test_catalog_expansion.py` (172) · MCP export smoke (**172** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · [`architecture/CRITIC_VERIFICATION.md`](architecture/CRITIC_VERIFICATION.md)

#### T-EXPAND T14 — Agent Builder DX introspection (2026-06-08) — **Done**

**Goal:** Runtime/catalog introspection for agent builders — discover tools, agents, and skill resolution without reading source.

| Bundle | Tools | Status |
|--------|------:|--------|
| `catalog` | `catalog.list_tools`, `catalog.describe_tool` | **Done** |
| `agent` | `agent.list_agents`, `agent.get_contract` | **Done** |
| `skill` | `skill.resolve` | **Done** |

**Delivered:** **175** catalog `tool_id` values · **45** shipped bundles.

#### T-EXPAND T15 — Sandbox execution depth (2026-06-08) — **Done**

**Goal:** Close `SANDBOX_REQUIRED_TOOLS` policy gap (`code.exec`, `script.run`, `browser.run`) and sandbox self-discovery.

| Bundle | Tools | Status |
|--------|------:|--------|
| `sandbox` (+4) | `code.exec`, `script.run`, `browser.run`, `sandbox.list_operations` | **Done** |
| runtime | `AGENT_BUILDER_SANDBOX_OPERATIONS` + `run_python` / `run_script` / `browser_fetch` session ops | **Done** |

**Delivered:** **179** catalog `tool_id` values · **45** shipped bundles.

**ADR:** **No ADR needed** — extends existing sandbox session ops; policy constants already referenced in `sandbox_runtime.py`.

#### T-EXPAND T16 — Memory & context builder surface (2026-06-08) — **Done**

**Goal:** Agent-facing LTM, task memory search, and context budget helpers.

| Bundle | Tools | Status |
|--------|------:|--------|
| `ltm` (new) | `ltm.search`, `ltm.write_fact` | **Done** |
| `memory` (+1) | `memory.search` | **Done** |
| `context` (new) | `context.summarize`, `context.estimate_tokens` | **Done** |
| bindings | `UserProfileManagerBinding` on `ToolWiringContext` | **Done** |

**Delivered:** **184** catalog `tool_id` values · **47** shipped bundles.

#### T-EXPAND T17 — Integration completeness (2026-06-08) — **Done**

**Goal:** HTTP allowlist client, interaction reply, issue update, RAG preview dry-run.

| Bundle | Tools | Status |
|--------|------:|--------|
| `http` (new) | `http.request` | **Done** |
| `interaction` (+1) | `interaction.post_reply` | **Done** |
| `issues` (+1) | `issues.update_issue` | **Done** |
| `rag` (+1) | `rag.preview_retrieval` | **Done** |
| contracts | `HttpClientBackend`, `IssueUpdater`, `AllowlistHttpClient` | **Done** |

**Delivered:** **190** catalog `tool_id` values · **48** shipped bundles.

**Verification:** `test_t14_t17_builder_tools.py` · `test_catalog_expansion.py` (190) · MCP export smoke (**190** tools)

Canon: [architecture/TOOLS.md](architecture/TOOLS.md) · handlers under `intergrax/tools/providers/{catalog,agent,skill_tool,ltm,context_tool,http}`


**Problem (Phase O):** Two parallel mechanisms — boolean plan flags dispatching pipeline steps vs `ToolRegistry` for function tools.

**Phase O outcome:** Unified **contracts** (`tool_ids` on plans, catalog shims for rag/websearch). **Phase TOOL-ENG** closes runtime **dispatch** and **gateway** gaps.

### Dispatch state — actual vs target

```text
LEGACY (deprecated, still mapped):
  plan.use_rag=True        → RagStep → catalog_context → rag.retrieve
  plan.use_websearch=True  → WebsearchStep → catalog_context → websearch.query
  plan.use_tools=True      → ToolsStep → ToolPlanningService → RuntimeToolInvoker

ACTUAL (2026-06-10):
  plan.tool_ids=["rag.retrieve", "websearch.query"]
      → normalized() sets use_rag / use_websearch → pipeline steps

  plan.tool_ids=["jira.search_tasks", "database.query"]
      → catalog_dispatch → RuntimeToolInvoker (TOOL-ENG-1 **Done**)
      → use_tools=True runs ToolsStep with planner allow-list from plan `tool_ids` (TOOL-ENG-4)

  ctx.invoke_tool(ToolRequest(tool_name="jira.get_issue"))
      → catalog_dispatch via RuntimeToolGateway (TOOL-ENG-2 **Done**)

TARGET (remaining TOOL-ENG):
  Multi-iteration tool loop (TOOL-ENG-6)
  Optional multi-iteration tool loop (TOOL-ENG-6)
```

**Compatibility (O.5a / LEG):** `ToolInvocationPlan.from_legacy(use_rag=…)` maps booleans to default tool_ids. Deprecation trace when legacy-only booleans used.

**Context injection:** `rag.retrieve` and `websearch.query` set `injects_context=true`; pipeline merges via `catalog_context` + `run_bounded_tool_loop` / `ctx.invoke_tool` system inject (§22.1).

**Configuration reference:** [`architecture/TOOLS.md`](../architecture/TOOLS.md) — [Runtime configuration reference](../architecture/TOOLS.md#runtime-configuration-reference), [Multi-tool execution](../architecture/TOOLS.md#multi-tool-execution-semantics), [§42.12 gateway](../architecture/TOOLS.md#4212-gateway-surface-toolrequest).

**Out of scope (TOOL-ENG):**

- Domain-specific tools inside `agents` (Tier-2; register via `ToolProvider` if reusable)
- New integration categories (Phase M)
- Product-only tool packs (§6.3 / Phase K)

---

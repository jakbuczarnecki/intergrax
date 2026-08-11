# LKW PRODUCT-3 First-Run Onboarding Gap Audit

## 1. Executive verdict

**Status:** READY_FOR_REVIEW  
**Task:** LKW-PRODUCT-3A — FIRST-RUN ONBOARDING UX/API GAP AUDIT  
**Mode:** discovery / gap analysis only — no production code changed

**Overall first-run readiness:** Backend and application-service foundations from PRODUCT-1/PRODUCT-2 are sufficient for a thin-client first-run journey over existing durable workspace, intake, sync, and Ask boundaries. The product-facing gap is not missing core lifecycle logic but missing **HTTP projection**, **orchestration contract**, and **UI** for the daily-use path.

| Classification | Count (15 journey steps) |
|---|---|
| **READY** | 4 |
| **PARTIAL** | 10 |
| **MISSING** | 1 |
| **OUT_OF_SCOPE_PRODUCT_3** | 0 |

**Verdict:** Implement PRODUCT-3 as thin clients over existing services. Do **not** add a dedicated onboarding state machine or first-run database mutations. Derive setup phase from workspace list, source inventory, knowledge configuration projection, and operation status. Expose existing `KnowledgeInspectionService` / `KnowledgeOperationsService` on HTTP where a unified indexed/live surface is required.

**Architecture decision required:** **yes** — one bounded decision on the **document inspect/open** product boundary (see §9). No new subsystem required.

**Git context:** branch `development`; required ancestor `66e9df07bf8f86555033f9a6bbb5279ea6e1321d` present at audit HEAD; unrelated dirty work not present at commit time.

---

## 2. Existing reusable foundation

| Area | Component | Public boundary | Persistence | Restart / resume | Thin-client reusable |
|---|---|---|---|---|---|
| Host launch & deployment readiness | `LocalWorkspaceHostLifecycle`, `mount_local_workspace_readiness_routes` | `GET /v1/local_workspace/liveness`, `GET /v1/local_workspace/readiness` | Host lifecycle in process | Host readiness survives client restart; not workspace setup state | Yes — deployment gate only |
| Workspace lifecycle | `ManagedWorkspaceService`, `mount_managed_workspace_routes` | `POST/GET/DELETE /v1/local_workspace/workspaces`, `GET /v1/local_workspace/workspaces/{id}` | SQLite via `ManagedWorkspaceRepository` | Durable per tenant | Yes |
| Indexed intake (managed file) | `ManagedFileIntakeService`, `KnowledgeIntakeService` | `POST /v1/local_workspace/workspaces/{id}/knowledge/files` | `KnowledgeInput`, `WorkspaceOperation`, `WorkspaceSource` | Intake + operation durable; ingestion re-queued on worker availability | Yes |
| Indexed intake (web URL) | `WebUrlIntakeService` | `POST /v1/local_workspace/workspaces/{id}/knowledge/web-urls` | Same intake model | Same | Yes |
| Indexed intake (source candidate) | `SourceCandidateIntakeService` | `GET/POST …/source-candidates` | Same | Same | Yes |
| Legacy local folder register + sync | `ManagedWorkspaceService` | `POST …/sources`, `POST …/sources/{id}/sync` | `WorkspaceSource`, `WorkspaceOperation` | Manual sync required after register; operations durable | Yes |
| Connected indexed sources | `mount_connected_source_knowledge_routes` | `POST …/knowledge/indexed-sources`, `POST …/knowledge/indexed-sources/{id}/sync`, remote-resource discovery | Knowledge configuration + source bindings | Configuration revision durable; sync via operations | Yes — provider-neutral |
| Connection attach | `mount_knowledge_connection_attachment_routes` | `PUT …/knowledge/connections/{connection_ref}` (attach), `DELETE` (detach) | Knowledge configuration mutations | Revision-tracked | Yes — provider-neutral |
| Live access bindings | `mount_knowledge_live_access_routes` | `POST/DELETE …/knowledge/live-access-bindings` | Knowledge configuration | Revision-tracked | Yes — provider-neutral |
| Knowledge configuration projection | `mount_knowledge_query_policy_routes` | `GET …/knowledge-configuration`, query-policy routes | Committed configuration head | Durable | Yes — indexed vs live bindings visible here |
| Sync / ingestion progress | `ManagedWorkspaceRepository` operations | `GET /v1/local_workspace/operations/{operation_id}` | `WorkspaceOperation` with counters and timestamps | Running ops may be marked failed on host restart (`recover_running_operations_for_tenant`) | Yes — per operation |
| Ask (indexed, v1) | `WorkspaceAskService` | `POST …/workspaces/{id}/ask`, `GET …/asks/{run_id}` | `WorkspaceAskRepository` | Ask runs durable | Yes — primary PRODUCT-3 first-run Ask path |
| Ask (hybrid, v2) | `WorkspaceAskServiceV2` | `POST /v2/local_workspace/workspaces/{id}/ask` | Hybrid ask run store | Durable | Yes — later milestone; needs query policy |
| Citations | `WorkspaceAskResponseV1` / assembler | Embedded in Ask response | In ask run record | Durable with run | Yes |
| Knowledge inventory (service only) | `KnowledgeInspectionService` | **No HTTP route** — `app.state.lkw_knowledge_inspection_service` | Derived from configuration + lifecycle | Durable underlying state | **No** — must be projected |
| Knowledge operations (service only) | `KnowledgeOperationsService` | **No HTTP route** — `app.state.lkw_knowledge_operations_service` | Lifecycle-owned mutations | Durable | **No** — must be projected |
| NL administration (service only) | `KnowledgeAdministrationService` | **No HTTP route** — Slack/companion wiring | Uses inspection + operations | Conversation-bound | Bot-only today |
| Conversation workspace selection | `ConversationWorkspaceSelectionService` | **No generic HTTP** — Slack/conversation layer | `ConversationContextRepository` | Durable per conversation binding | Slack/bot only — not general HTTP “current workspace” |

**PRODUCT-2 closure:** Windows zero-to-value quickstart (`run-lkw-product-quickstart-*`) already exercises create workspace → managed-file intake → poll operation → Ask v1 → citation → persisted run readback. That path is **script-orchestrated**, not a reusable product onboarding contract.

---

## 3. Target first-run journey matrix

| Step | Journey step | Class | Existing capability | Gap |
|---|---|---|---|---|
| 1 | User launches LKW | **READY** | OS quickstart scripts; Docker bootstrap; `GET /v1/local_workspace/readiness` | Product UI launcher not in LKW app (scripts/Docker only) |
| 2 | Clear first-run / welcome state | **MISSING** | None at product/API layer | No welcome/onboarding contract; no LKW frontend |
| 3 | Create or select workspace | **PARTIAL** | `POST/GET /workspaces` | No server-side “current workspace” for generic HTTP clients; selection is client-held `workspace_id` or conversation-bound |
| 4 | Prompted to add first knowledge source | **PARTIAL** | Multiple intake APIs + source candidates | No product orchestration API/UI guiding first source choice |
| 5 | Understand Indexed vs Live | **PARTIAL** | `GET /knowledge-configuration` separates indexed bindings vs live-access bindings; inventory service has `mode` | Unified provider-neutral inventory not on HTTP; `GET /sources` is legacy-oriented list |
| 6 | Configure / authenticate source | **PARTIAL** | Managed upload (no auth); local folder path policy; connection attach + connected discovery | Provider auth lives behind connection capabilities; no single first-run setup wizard contract |
| 7 | Start or confirm sync/indexing | **READY** | Intake auto-enqueues ingestion; `POST …/sync` for folder/connected sources | `register_local_folder_source` does **not** auto-sync — extra step |
| 8 | See progress / state | **PARTIAL** | `GET /operations/{id}` with counters | No workspace-scoped operation list HTTP; `error_code` not exposed in `OperationResponseV1` |
| 9 | Clear READY state | **PARTIAL** | `WorkspaceSourceStatus.READY`, operation `completed`, inventory `summary.active` (service) | No workspace-level “setup complete / can Ask” product contract on HTTP |
| 10 | Suggested example question | **PARTIAL** | Source labels/descriptions in candidates, configuration projections, `safe_source_label` | No deterministic suggestion helper; adequate metadata for client-side generic template |
| 11 | User asks question | **READY** | `POST …/workspaces/{id}/ask` (v1) | v2 hybrid requires query policy configuration |
| 12 | Grounded answer with citation | **READY** | `WorkspaceAskResponseV1` with `citations[]` | `insufficient_evidence` is valid outcome when no usable evidence |
| 13 | Inspect / open cited source | **PARTIAL** | Citation includes `document_id`, `source_id`, `file_name`, `source_path`, excerpt | No `GET document` / open-original HTTP action; path usefulness varies by source type |
| 14 | Later launch — no incorrect onboarding restart | **PARTIAL** | All setup state durable in repository | Clients must derive phase from APIs; must not use client-only onboarding flags |
| 15 | Incomplete setup resumes correctly | **PARTIAL** | Durable workspace, sources, operations, configuration | No explicit resume pointer; client re-derives from same signals as step 14 |

---

## 4. First-run state derivation model

### Derivable from existing durable state?

**Yes**, for all setup phases, without `onboarding_step = N` persistence.

| Setup phase | Derivation signals (HTTP-available today) | Service-only enrichments |
|---|---|---|
| No workspace | `GET /workspaces` → empty list | — |
| Workspace exists, no source | `GET /sources` empty **and** `GET /knowledge-configuration` has no indexed/live bindings | Inventory `summary.total == 0` |
| Source configured, not synchronized | `WorkspaceSource.status == registered` **or** no completed sync/ingestion operation | Inventory `sync_state` |
| Sync/indexing in progress | Source `syncing`/`processing` **or** operation `queued`/`running`/`processing`/`accepted` | Active ingestion locator |
| Source requires attention | Source `error` **or** inventory `attention_required` / `lifecycle_state` error | `last_error_code` on inventory item |
| Usable indexed knowledge exists | Any source `ready` with `last_sync_at` **or** completed ingestion with `documents_indexed > 0` | — |
| Live source available | `GET /knowledge-configuration` → active `live_access_bindings` | Inventory `mode == live`, `runtime_available` |
| User can start asking | Host `readiness.accepts_new_work` **and** indexed readiness above **or** tolerate `insufficient_evidence` | Hybrid v2 additionally needs query policy |

### Explicit onboarding state machine required?

**No.** Derivation is sufficient if PRODUCT-3 exposes:

1. Existing list/get endpoints (already present), and  
2. HTTP projection of `KnowledgeInventoryV1` (service exists), and  
3. Optional **setup snapshot** endpoint that composes the above without new persistence (implementation choice in PRODUCT-3B).

Rationale: workspace, source, configuration revision, and operation records already form a durable, restart-safe product state. Onboarding is a **view** over that state, not a separate lifecycle.

---

## 5. Workspace gaps

### Ready

- Create: `POST /v1/local_workspace/workspaces` (`ManagedWorkspaceService.create_workspace`)
- List / get: `GET /workspaces`, `GET /workspaces/{workspace_id}`
- Delete: `DELETE /workspaces/{workspace_id}`
- Tenant scoping: `X-Tenant-Id` / request context via `resolve_tenant_id`

### Gaps

| Gap | Classification | Minimal contract | Owner | Backend work? |
|---|---|---|---|---|
| Default / current workspace for HTTP clients | **PARTIAL** | Client stores `workspace_id`; optional future user-preference API is **not** required for PRODUCT-3 if single-workspace-first UX creates on first run | Application / product | No — unless multi-workspace UX is mandatory on day one |
| Conversation-bound workspace selection | **READY** (Slack) / **N/A** (HTTP) | `ConversationWorkspaceSelectionService` — durable server-side for conversation clients only | Conversation layer | No for HTTP first-run |
| First-run workspace creation via public API | **READY** | Already supported | `ManagedWorkspaceService` | No |

---

## 6. Source / connection gaps

### Ready

| Source type | Create boundary | Auto index/sync |
|---|---|---|
| Managed file upload | `POST …/knowledge/files` | Yes — intake dispatches `KnowledgeIngestionJob` |
| Web URL | `POST …/knowledge/web-urls` | Yes |
| Source candidate (local folder) | `POST …/knowledge/source-candidates/{id}` | Yes |
| Local folder (legacy API) | `POST …/sources` | **No** — requires `POST …/sources/{id}/sync` |
| Connected indexed source | `POST …/knowledge/indexed-sources` | Create + optional `…/sync` |
| Connection attach | `PUT …/knowledge/connections/{connection_ref}` | Enables provider-neutral setup |
| Live access | `POST …/knowledge/live-access-bindings` | Live — query-time, not indexed |

**Indexed vs live representation:** `GET …/knowledge-configuration` returns `indexed_source_bindings` and `live_access_bindings` separately. `KnowledgeInventoryItemV1.mode` (`INDEXED` | `LIVE`) exists in service layer.

**Status fields:** `WorkspaceSourceStatus`: `registered`, `syncing`, `processing`, `ready`, `error`. Inventory adds `lifecycle_state`, `sync_state`, `runtime_available`, `available_actions`.

### Gaps

| Gap | Classification | Minimal contract | Owner | Backend work? |
|---|---|---|---|---|
| Unified knowledge inventory on HTTP | **PARTIAL** | `GET …/knowledge/inventory` projecting `KnowledgeInventoryV1` | Application serving | Yes — HTTP projection only |
| Knowledge operations on HTTP | **PARTIAL** | `POST …/knowledge/items/{id}/operations` projecting `KnowledgeOperationsService` | Application serving | Yes — HTTP projection only |
| NL administration on HTTP | **PARTIAL** | Optional; Slack bot already uses service | Conversation / bot | No for HTTP PRODUCT-3 |
| Provider-specific onboarding in shared orchestration | **N/A** | Forbidden by vendor rule | — | No |

---

## 7. Sync / progress gaps

### Ready

- First sync trigger: intake paths enqueue automatically; folder/connected via explicit sync POST
- Async operation ID: returned on intake accept and sync accept
- Status API: `GET /v1/local_workspace/operations/{operation_id}`
- Progress fields: `files_discovered`, `files_processed`, `files_failed`, `documents_indexed`, `documents_unchanged`, timestamps
- Terminal states: `completed`, `failed`
- Retry: via `KnowledgeOperationsService` (`RETRY_SYNC`) — service only today
- Durability: operations persisted; host restart may fail in-flight ops (`interrupted_by_host_restart`)

### Gaps

| Gap | Classification | Minimal contract | Owner | Backend work? |
|---|---|---|---|---|
| List operations for workspace | **PARTIAL** | `GET …/workspaces/{id}/operations` or filter on inventory item | Application serving | Yes — thin list over repository |
| `error_code` in HTTP operation response | **PARTIAL** | Add `error_code` to `OperationResponseV1` (model already has field) | Application serving | Yes — schema projection |
| User-actionable retry from HTTP | **PARTIAL** | Expose inventory `available_actions` + operations execute | Application serving | Yes — with inventory HTTP |
| Percent / ETA progress | **OUT_OF_SCOPE** | Not in current operation model | — | No for PRODUCT-3 |

---

## 8. Ask / readiness / citation gaps

### Ask readiness — “Can this workspace meaningfully Ask now?”

**Derivable today (no new readiness API strictly required):**

| Signal | Source |
|---|---|
| Host can accept work | `GET /readiness` → `accepts_new_work` |
| Indexed data likely available | Any `source.status == ready` or completed ingestion op with `documents_indexed > 0` |
| Live available | Active live bindings in `GET /knowledge-configuration` |
| Stale / unavailable | `source.status == error`; inventory `attention_required`; `runtime_available == false` |
| No usable evidence | Ask returns `status: insufficient_evidence` |

**Insufficient for polished UX without HTTP inventory:** unified `attention_required` and `available_actions` live only in `KnowledgeInspectionService`.

**Architecture note:** Prefer derivation + inventory HTTP over a new `workspace_readiness` subsystem. If PRODUCT-3C needs a single poll endpoint, implement a **composed setup snapshot** (read-only) in the application layer.

### Citations

**Contract (`WorkspaceAskCitationV1`):** `evidence_id`, `document_id`, `source_id`, `workspace_id`, `source_path`, `file_name`, `excerpt`, `score`, `chunk_id`, optional `location.page`.

### Gaps

| Gap | Classification | Minimal contract | Owner | Backend work? |
|---|---|---|---|---|
| Open / inspect original source | **PARTIAL** | **ARCHITECTURE_DECISION_REQUIRED** — see below | Application + product | Depends on decision |
| Hybrid Ask v1 first-run default | **READY** (indexed) | v1 Ask works without query policy | Ask service | No |
| Suggested question API | **PARTIAL** | Client template: e.g. “What does {label} contain?” from source candidate `label` / `description` | Product UI | No — deterministic client sufficient |

### ARCHITECTURE_DECISION_REQUIRED — document inspect / open

| Item | Detail |
|---|---|
| Missing capability | User-actionable “open cited source” across managed upload, local folder, web URL, connected sources |
| Why existing boundaries are insufficient | Ask citations carry `source_path`/`file_name` but no authenticated HTTP retrieval; paths may be host-internal for managed uploads; connected sources need provider-neutral open semantics |
| Minimal decision | Choose one: (A) host-mediated `GET …/documents/{document_id}` with safe display payload + optional external URL; (B) citation-only display without open for managed/connected in PRODUCT-3; (C) defer open to provider-specific deep links behind inventory metadata |
| Do not invent | Separate document subsystem or direct vector-store reads from UI |

---

## 9. Error UX gaps (first-run blocking only)

Technical `detail` strings returned today that block onboarding without user-actionable mapping:

| Error / detail | Where | First-run impact |
|---|---|---|
| `idempotency_key_required` | File/web/candidate intake | Blocks upload without header |
| `tenant_id_required` | Connection attach routes | Blocks connection setup |
| `knowledge_configuration_if_match_required` | Configuration mutations | Blocks attach/indexed create without `If-Match: WKC/{revision}` |
| `managed_file_storage_unavailable` | File upload | Blocks primary first-run path |
| `source_candidate_registry_unavailable` | Candidates | Blocks candidate path |
| `sync_enqueue_failed` / `enqueue_failed` | Sync / intake | Blocks indexing with generic 502 |
| `search_evidence_incomplete` / `ask_persistence_failed` | Ask | Blocks success path |
| Raw `ValueError` / `run_error: {ExceptionName}` | Legacy routes | Non-actionable |
| Operation `error` string without `error_code` in HTTP | `GET /operations/{id}` | Hard to map to “retry” / “fix connection” |

Not PRODUCT-3 scope: full PRODUCT-8 error catalog redesign.

---

## 10. Minimal implementation plan

1. **HTTP projection layer** — expose `KnowledgeInventoryV1` and `KnowledgeOperationsService.execute` on managed workspace routes (no new domain logic).
2. **Setup snapshot (optional, read-only)** — compose workspace list + inventory summary + host readiness for one first-run poll (derivation only).
3. **Operation response hardening** — surface `error_code`; add workspace operation list if inventory alone is insufficient for progress UX.
4. **First-run UI / orchestration** — thin client (web or shell) implementing welcome → workspace → source picker → progress → Ask; no backend onboarding state.
5. **Document open decision** — implement chosen boundary from architecture decision before promising step 13 in UI.
6. **Suggested question** — deterministic client template from existing labels; no LLM suggestion service.

---

## 11. Explicit non-goals

- Implementing onboarding UI in this audit task
- New `onboarding_step` persistence or first-run DB mutations
- Provider-specific (Slack/Google/MS) logic in shared first-run orchestration
- Full Hybrid Ask v2 as default first-run path
- Full PRODUCT-8 error redesign
- Vendor Knowledge internals beyond public connection/indexed/live boundaries
- Public Docs redesign, LCI, Token Optimization
- Repo-wide search or new subsystems (Mongo/Qdrant direct client access)

---

## 12. PRODUCT-3 proposed task breakdown

| Task | Purpose |
|---|---|
| **PRODUCT-3B — Knowledge surface HTTP projection** | Add `GET /knowledge/inventory`, knowledge operation execute route, and operation list/`error_code` projection over existing inspection/operations services and repository. Unblocks indexed/live unified view and retry without duplicating lifecycle logic. |
| **PRODUCT-3C — Setup snapshot & first-run orchestration contract** | Optional read-only setup snapshot endpoint (derivation-only) plus documented client orchestration sequence (workspace → intake → poll → Ask). No new persistence. |
| **PRODUCT-3D — First-run product UI / welcome flow** | Provider-neutral welcome and step UX as thin HTTP client over 3B + existing workspace/intake/ask routes. Includes workspace create/select (client-held `workspace_id`). |
| **PRODUCT-3E — Citation inspect/open + acceptance** | Implement architecture decision for document open; harden first-run error mapping; live acceptance proving resume after restart and no false onboarding restart. |

**Dependency order:** 3B → 3C (if snapshot needed) → 3D → 3E.

---

## Audit metadata

**Files read (15, no repo-wide search):**

1. `applications/local_workspace_application/README.md`
2. `applications/local_workspace_application/serving/fastapi_router.py`
3. `applications/local_workspace_application/host/readiness.py`
4. `applications/local_workspace_application/serving/readiness_routes.py`
5. `applications/local_workspace_application/serving/workspace_routes.py` (targeted sections)
6. `applications/local_workspace_application/serving/knowledge_connected_source_routes.py` (targeted sections)
7. `applications/local_workspace_application/workspaces/models.py` (targeted sections)
8. `applications/local_workspace_application/serving/workspace_schemas.py` (targeted sections)
9. `applications/local_workspace_application/workspaces/knowledge_inspection_operations_service.py` (targeted sections)
10. `applications/local_workspace_application/workspaces/knowledge_intake.py` (targeted sections)
11. `applications/local_workspace_application/workspaces/service.py` (targeted sections)
12. `applications/local_workspace_application/workspaces/conversation_workspace_selection_service.py` (targeted sections)
13. `applications/local_workspace_application/serving/knowledge_query_policy_routes.py` (targeted sections)
14. `applications/local_workspace_application/serving/knowledge_connection_attachment_routes.py` (targeted sections)
15. `docs/project/product/lkw/USER_JOURNEY.md` (targeted sections)

**Targeted discovery only:** `grep` / `glob` on `applications/local_workspace_application/` for routes, symbols, and onboarding keywords — not repo-wide semantic search.

**Files changed:** `docs/project/product/lkw/PRODUCT_3_FIRST_RUN_GAP_AUDIT.md` (this file only).

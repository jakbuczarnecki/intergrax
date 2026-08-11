# LKW PRODUCT-3 First-Run Onboarding Gap Audit

## 1. Executive verdict

**Status:** CLOSED  
**Task:** LKW-PRODUCT-3A — FIRST-RUN ONBOARDING UX/API GAP AUDIT (superseded by PRODUCT-3 final closeout §13)  
**Mode:** discovery / gap analysis — no production code changed in 3A; PRODUCT-3 closed at `41ec991713a0445bc2e4302f2bfb8e1fefb5c27f`

**Overall first-run readiness:** Backend and application-service foundations from PRODUCT-1/PRODUCT-2 are sufficient for a **Slack-first** thin-client first-run journey over existing durable workspace, intake, sync, and Ask boundaries. The product-facing gap is not missing core lifecycle logic but missing **Slack conversational UX** orchestration over accepted HTTP projection and setup-snapshot capabilities.

| Classification | Count (15 journey steps) |
|---|---|
| **READY** | 4 |
| **PARTIAL** | 10 |
| **MISSING** | 1 |
| **OUT_OF_SCOPE_PRODUCT_3** | 0 |

**Verdict:** Implement PRODUCT-3D as Slack thin client over existing services and accepted PRODUCT-3B/3C capabilities. Do **not** add a dedicated onboarding state machine or first-run database mutations. Derive setup phase from workspace list, source inventory, knowledge configuration projection, operation status, and `setup-snapshot`.

**Architecture decision required:** **no** — citation inspect/open is **resolved** (host-mediated provider-neutral document boundary; see §8). PRODUCT-3E implements/proves the chosen architecture.

**Accepted milestones (do not reopen):** PRODUCT-2 **CLOSED**; PRODUCT-3B **CLOSED** (HTTP projection); PRODUCT-3C **CLOSED** (setup snapshot). The cancelled web-first **LKW-PRODUCT-3D — FIRST-RUN PRODUCT UI AND WELCOME FLOW** produced no implementation and must not be represented as executed.

**Git context:** branch `development`; required ancestor `607083e728ab718ddf10ef97aa8f3a37425797a3` at contract-correction base; unrelated dirty work preserved outside this task.

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
| Knowledge inventory (service only) | `KnowledgeInspectionService` | `GET …/knowledge/inventory` projecting `KnowledgeInventoryV1` | Derived from configuration + lifecycle | Durable underlying state | **Yes** — **PRODUCT-3B CLOSED** |
| Knowledge operations (service only) | `KnowledgeOperationsService` | Knowledge operation execute + workspace operation list HTTP | Lifecycle-owned mutations | Durable | **Yes** — **PRODUCT-3B CLOSED** |
| Setup snapshot | `WorkspaceSetupSnapshotService` | `GET …/workspaces/{id}/setup-snapshot` | Derivation-only composition | Durable underlying state | **Yes** — **PRODUCT-3C CLOSED** |
| NL administration (service only) | `KnowledgeAdministrationService` | **No HTTP route** — Slack/companion wiring | Uses inspection + operations | Conversation-bound | Bot-only today |
| Conversation workspace selection | `ConversationWorkspaceSelectionService` | **No generic HTTP** — Slack/conversation layer | `ConversationContextRepository` | Durable per conversation binding | Slack/bot only — not general HTTP “current workspace” |

**PRODUCT-2 closure:** Windows zero-to-value quickstart (`run-lkw-product-quickstart-*`) already exercises create workspace → managed-file intake → poll operation → Ask v1 → citation → persisted run readback. That path is **script-orchestrated installation proof**, not the primary daily-use client and not a reusable Slack first-run contract.

---

## 3. Target first-run journey matrix

| Step | Journey step | Class | Existing capability | Gap |
|---|---|---|---|---|
| 1 | User launches LKW | **READY** | OS quickstart scripts; Docker bootstrap; `GET /v1/local_workspace/readiness` | Slack app install/connect is separate vendor surface; host readiness gate sufficient for backend |
| 2 | Clear first-run / welcome state | **MISSING** | None at Slack conversational UX layer | No Slack welcome/onboarding conversational contract; backend derivation exists via setup-snapshot |
| 3 | Create or select workspace | **PARTIAL** | `POST/GET /workspaces`; Slack `ConversationWorkspaceSelectionService` | Slack first-run workspace selection UX not productized; HTTP clients use client-held `workspace_id` |
| 4 | Prompted to add first knowledge source | **PARTIAL** | Multiple intake APIs + source candidates; Slack attachments | No Slack conversational orchestration guiding first source choice |
| 5 | Understand Indexed vs Live | **PARTIAL** | `GET /knowledge-configuration`; `GET …/knowledge/inventory` with `mode` (**PRODUCT-3B CLOSED**) | Slack conversational explanation of indexed vs live not productized |
| 6 | Configure / authenticate source | **PARTIAL** | Managed upload (no auth); local folder path policy; connection attach + connected discovery; Slack attachments | Provider auth lives behind connection capabilities; no Slack first-run setup conversational contract |
| 7 | Start or confirm sync/indexing | **READY** | Intake auto-enqueues ingestion; `POST …/sync` for folder/connected sources | `register_local_folder_source` does **not** auto-sync — extra step |
| 8 | See progress / state | **PARTIAL** | `GET /operations/{id}`; workspace operation list + `error_code` (**PRODUCT-3B CLOSED**); `GET …/setup-snapshot` (**PRODUCT-3C CLOSED**) | Slack progress/attention UX not productized (**PRODUCT-3D**) |
| 9 | Clear READY state | **PARTIAL** | `WorkspaceSourceStatus.READY`, operation `completed`, inventory `summary.active`, setup-snapshot phase (**PRODUCT-3C CLOSED**) | Slack READY-state conversational UX not productized (**PRODUCT-3D**) |
| 10 | Suggested example question | **PARTIAL** | Source labels/descriptions in candidates, configuration projections, `safe_source_label` | No Slack suggested-first-question UX (**PRODUCT-3D**); adequate metadata for client-side template |
| 11 | User asks question | **READY** | `POST …/workspaces/{id}/ask` (v1) | v2 hybrid requires query policy configuration |
| 12 | Grounded answer with citation | **READY** | `WorkspaceAskResponseV1` with `citations[]` | `insufficient_evidence` is valid outcome when no usable evidence |
| 13 | Inspect / open cited source | **PARTIAL** | Citation includes `document_id`, `source_id`, `file_name`, `source_path`, excerpt | Host-mediated `GET …/documents/{document_id}` not implemented — **PRODUCT-3E**; architecture **RESOLVED** |
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

**No.** Derivation is sufficient. **PRODUCT-3B** and **PRODUCT-3C** are **CLOSED**:

1. HTTP projection of `KnowledgeInventoryV1` and knowledge operations (**PRODUCT-3B**).
2. Read-only **setup snapshot** endpoint composing workspace/inventory/readiness signals (**PRODUCT-3C**).

**PRODUCT-3D** implements Slack conversational UX over these capabilities without new persistence.

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
| Unified knowledge inventory on HTTP | **READY** | `GET …/knowledge/inventory` projecting `KnowledgeInventoryV1` | Application serving | **No** — **PRODUCT-3B CLOSED** |
| Knowledge operations on HTTP | **READY** | Knowledge operation execute + workspace operation list | Application serving | **No** — **PRODUCT-3B CLOSED** |
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
| List operations for workspace | **READY** | Workspace-scoped operation list HTTP | Application serving | **No** — **PRODUCT-3B CLOSED** |
| `error_code` in HTTP operation response | **READY** | `error_code` in `OperationResponseV1` | Application serving | **No** — **PRODUCT-3B CLOSED** |
| User-actionable retry from HTTP | **READY** | Inventory `available_actions` + operations execute | Application serving | **No** — **PRODUCT-3B CLOSED** |
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

**Insufficient for polished Slack UX without conversational orchestration:** unified `attention_required` and `available_actions` are on HTTP inventory (**PRODUCT-3B CLOSED**) but not yet surfaced in Slack first-run UX (**PRODUCT-3D**).

**Architecture note:** Derivation + inventory HTTP + setup snapshot (**PRODUCT-3C CLOSED**) — no new `workspace_readiness` subsystem.

### Citations

**Contract (`WorkspaceAskCitationV1`):** `evidence_id`, `document_id`, `source_id`, `workspace_id`, `source_path`, `file_name`, `excerpt`, `score`, `chunk_id`, optional `location.page`.

### Gaps

| Gap | Classification | Minimal contract | Owner | Backend work? |
|---|---|---|---|---|
| Open / inspect original source | **PARTIAL** | Host-mediated provider-neutral `GET …/documents/{document_id}` — **RESOLVED** architecture; **PRODUCT-3E** implementation | Application + product | Yes — PRODUCT-3E |
| Hybrid Ask v1 first-run default | **READY** (indexed) | v1 Ask works without query policy | Ask service | No |
| Suggested question API | **PARTIAL** | Slack/client template: e.g. “What does {label} contain?” from source candidate `label` / `description` | Slack PRODUCT-3D | No — deterministic client sufficient |

### RESOLVED — document inspect / open (PRODUCT-3E implements)

| Item | Detail |
|---|---|
| Decision status | **RESOLVED** — not `ARCHITECTURE_DECISION_REQUIRED` |
| Chosen architecture | **Host-mediated provider-neutral document inspect/open boundary** |
| Conceptual endpoint | `GET /.../documents/{document_id}` (exact final route may be established in PRODUCT-3E) |
| Safe contract concept | document/source identity, display name, source type, source label, logical source location, provenance, page/location, bounded preview/metadata, optional `external_url`/provider deep-link where capability exists |
| Forbidden | UI/Slack direct Qdrant reads; arbitrary host filesystem exposure; arbitrary raw local path exposure; vendor-specific citation routing in Slack; separate document subsystem solely for citation opening |
| Why citations alone are insufficient | Ask citations carry `source_path`/`file_name` but no authenticated HTTP retrieval; paths may be host-internal for managed uploads; connected sources need provider-neutral open semantics |

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
| Operation `error` string without `error_code` in HTTP | `GET /operations/{id}` | Hard to map to “retry” / “fix connection” in Slack UX (**PRODUCT-3E**) |

Not PRODUCT-3 scope: full PRODUCT-8 error catalog redesign.

---

## 10. Minimal implementation plan (remaining PRODUCT-3 work)

1. **LKW-PRODUCT-3D — SLACK FIRST-RUN PRODUCT EXPERIENCE** — Slack conversational
   UX: welcome → workspace selection/creation → first knowledge via
   Slack-compatible path → snapshot-driven sync state → attention/recovery →
   READY → suggested first question → Ask → grounded response + citation
   display. Consumes **PRODUCT-3B** and **PRODUCT-3C** capabilities. No web
   frontend. No new onboarding persistence.
2. **LKW-PRODUCT-3E — CITATION INSPECT/OPEN + ERROR/RESUME ACCEPTANCE** —
   implement/prove host-mediated `GET …/documents/{document_id}`; bounded
   user-facing error behavior in Slack; restart/resume acceptance; first-run
   end-to-end acceptance.
3. **Suggested question** — deterministic Slack/client template from existing
   labels; no LLM suggestion service.

**Already delivered (do not reopen):** PRODUCT-3B HTTP projection; PRODUCT-3C
setup snapshot.

---

## 11. Explicit non-goals

- Implementing Slack PRODUCT-3D in this audit task
- Web-first first-run UI or welcome flow (cancelled invalid PRODUCT-3D)
- New `onboarding_step` persistence or first-run DB mutations
- Provider-specific (Slack/Google/MS) logic in shared first-run orchestration
- Full Hybrid Ask v2 as default first-run path
- Full PRODUCT-8 error redesign
- Vendor Knowledge internals beyond public connection/indexed/live boundaries
- Public Docs redesign, LCI, Token Optimization
- Repo-wide search or new subsystems (Mongo/Qdrant direct client access)

---

## 12. PRODUCT-3 proposed task breakdown

| Task | Status | Purpose |
|---|---|---|
| **PRODUCT-3B — Knowledge surface HTTP projection** | **CLOSED** | `GET /knowledge/inventory`, knowledge operation execute, operation list, and `error_code` projection over existing inspection/operations services. |
| **PRODUCT-3C — Setup snapshot & first-run orchestration contract** | **CLOSED** | Read-only `setup-snapshot` endpoint (derivation-only) plus documented client orchestration sequence. No new persistence. |
| **LKW-PRODUCT-3D — SLACK FIRST-RUN PRODUCT EXPERIENCE** | **CLOSED** | Slack conversational first-run over 3B + 3C + existing workspace/intake/ask routes. |
| **LKW-PRODUCT-3E — Citation inspect/open + error/resume acceptance** | **CLOSED** | Host-mediated document inspect/open; bounded Slack error behavior; restart/resume acceptance. |
| **PRODUCT-3 FINAL CLOSEOUT** | **CLOSED** | Accepted at `41ec991713a0445bc2e4302f2bfb8e1fefb5c27f`; matrix §13. |
| **LKW-PRODUCT-4 — SLACK DAILY-USE PRODUCT EXPERIENCE** | **NEXT** | Daily Slack UX for workspace selection, inventory, source state, sync, disable/enable/detach, Ask, citations/open, freshness, attention, basic settings — using shared backend capabilities. Not generic “real product UI”; no web frontend required for LKW 1.0. |

**Dependency order:** 3B (**CLOSED**) → 3C (**CLOSED**) → 3D (**CLOSED**) → 3E (**CLOSED**) → PRODUCT-3 closeout (**CLOSED**) → PRODUCT-4.

**Cancelled / invalid:** **LKW-PRODUCT-3D — FIRST-RUN PRODUCT UI AND WELCOME FLOW** (web-first) — no implementation; must not be represented as executed.

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

**Files changed:** `docs/project/product/lkw/PRODUCT_3_FIRST_RUN_GAP_AUDIT.md` (this file); aligned with Slack-first contract correction in `PRODUCT_CONTRACT.md` and `USER_JOURNEY.md`.

---

## 13. PRODUCT-3 final closeout (LKW-PRODUCT-3-FINAL-CLOSEOUT)

**Final status:** **CLOSED**  
**Accepted closing commit:** `41ec991713a0445bc2e4302f2bfb8e1fefb5c27f`  
**Required ancestor:** `580015167baa62868ed08623aed8d6d68f39001e`  
**Live Slack acceptance:** **NOT_RUN_ENVIRONMENT** (no canonical local Slack stack/credentials in this audit session; automated/integration evidence accepted).  
**Bounded test suite:** 105 passed (`test_conversation_first_run_application`, `test_interaction_application_service`, `test_workspace_setup_snapshot`, `test_document_inspect_service`, `test_conversation_citation_inspect`, `test_interaction_response_renderer`, `test_conversation_setup_onboarding`, `test_conversation_workspace_selection_service`).

### 15-step acceptance matrix

| Step | Journey | Status | Evidence | User-visible behavior | Later owner |
|---|---|---|---|---|---|
| 1 | Launch / availability | **READY** | PRODUCT-2 **CLOSED**; `GET /v1/local_workspace/readiness` | User reaches running LKW via accepted install path | PRODUCT-11 (multi-OS clean machine) |
| 2 | First contact / welcome | **READY** | `test_first_dm_without_workspace_selection_shows_welcome`; `test_welcome_first_dm_no_workspace` | Welcome + create/select guidance; no IDs/jargon; attachments blocked pre-workspace | — |
| 3 | Create / select workspace | **READY** | `ConversationWorkspaceSelectionService` + `ConversationContextRepository`; `test_conversation_workspace_selection_service` | List/create/select/switch via Slack DM; durable conversation context (not `InMemorySlackWorkspaceSelectionStore` on wired companion path) | — |
| 4 | Add first knowledge | **READY** | `test_attachment_intake_is_executed_when_snapshot_is_not_ready` | Slack file attachment → Knowledge Intake → async preparation | — |
| 5 | Indexed vs Live understanding | **READY_WITH_LATER_BOUNDARY** | `test_no_knowledge_guidance`; setup snapshot phases | Managed-file first-run is indexed-only; no false “all knowledge is live” claim | PRODUCT-4/6 |
| 6 | Configure / authenticate source | **READY_WITH_LATER_BOUNDARY** | Managed upload requires no vendor auth | Sufficient for PRODUCT-3 first-run | PRODUCT-5 |
| 7 | Start preparation / sync | **READY** | `test_finish_success_appends_snapshot_guidance_after_action` | Attachment triggers canonical preparation/indexing | — |
| 8 | Progress / state | **READY** | `test_syncing_preparation_state`; `test_workspace_setup_snapshot` | Honest snapshot-derived preparation state; no fake %/ETA | PRODUCT-7 (background notifications) |
| 9 | READY state | **READY** | `test_ready_can_ask_suggested_question`; `test_ready_cannot_ask_blocks_question` | READY from snapshot; Ask CTA only when `can_ask=true` | — |
| 10 | Suggested question | **READY** | `test_generic_suggested_question_without_label`; setup snapshot | Deterministic suggested first question from snapshot | — |
| 11 | Ask | **READY** | `test_ready_question_is_planned_and_executed`; plan-aware gating tests | Planner → `WorkspaceAsk` → shared `WorkspaceAskService`; non-Ask admin actions still executable | — |
| 12 | Grounded answer + citations | **READY** | `test_renderer_includes_safe_ask_citation`; Ask insufficient-evidence paths | Answer + citation labels; insufficient evidence is normal | — |
| 13 | Inspect / open citation | **READY** | `test_conversation_citation_inspect`; `test_document_inspect_service` | “show source 1” / “open citation 2”; safe name, location, preview, optional URL; inspect-only when no safe open target | — |
| 14 | Restart does not restart onboarding | **READY** | `test_creates_selection_and_reconstructs_from_same_store`; `test_restart_reconstructs_recent_turns_before_planning`; `test_citation_inspect_survives_service_recreation`; `test_repeated_snapshot_has_no_persisted_onboarding_state` | Workspace selection, READY, citation context survive service recreation; no `onboarding_step` persistence | — |
| 15 | Incomplete setup resumes | **READY** | Snapshot phase tests (`NO_KNOWLEDGE`, `SYNCING`, `ATTENTION_REQUIRED`); `test_question_gated_when_snapshot_not_ready_for_ask` | Correct next state/action after recreation; no wizard reset | — |

### Later boundaries (explicitly not blocking PRODUCT-3)

| Area | Owner |
|---|---|
| Vendor OAuth and polished provider configuration | PRODUCT-5 |
| Full daily knowledge UX (inventory, scoping, freshness polish) | PRODUCT-4 / PRODUCT-6 |
| Background sync notifications | PRODUCT-7 |
| General failure catalog / recovery redesign | PRODUCT-8 |
| Clean-machine unfamiliar-user live proof | PRODUCT-11 |

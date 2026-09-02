# LKW PRODUCT-4 Slack Daily-Use Gap Audit

## 1. Executive user-facing verdict

**Status:** READY_FOR_REVIEW  
**Task:** LKW-PRODUCT-4A - SLACK DAILY-USE UX GAP AUDIT  
**Mode:** discovery / gap analysis only - no production code changed

**Overall daily-use readiness after PRODUCT-3 closeout:** A user who completed Slack first-run onboarding can **Ask**, receive **grounded answers with citations**, **inspect/open cited sources**, and **switch/create workspaces** through the canonical conversational path. The gap for everyday Slack use is **not** missing core Ask or citation backends; it is missing **daily knowledge inventory, lifecycle control, freshness visibility, and source-scoped Ask** on the conversational path, plus **presentation gaps** where executor data exists but Slack responses do not expose it.

| Classification | Count (29 audited capabilities) |
|---|---|
| **READY** | 6 |
| **WIRING_GAP** | 14 |
| **UX_GAP** | 5 |
| **REUSABLE_CAPABILITY_GAP** | 2 |
| **LATER_PRODUCT_BOUNDARY** | 2 |

**Verdict:** Implement PRODUCT-4 as Slack thin-client wiring and presentation over **accepted PRODUCT-3B inventory/operations HTTP projection**, **`KnowledgeAdministrationService`** (NL + HMAC confirmation), and existing planner/executor/renderer - **without** new Slack-only lifecycle truth. Source-scoped Ask requires a **new reusable application contract** before conversational wiring. Staleness/scheduling semantics and vendor OAuth remain later milestones.

**Architecture decisions required:** **yes** - (1) source-scoped Ask reusable contract; (2) whether workspace delete needs the same class of confirmation as knowledge detach.

**Git context:** branch `development`; required ancestor `c99819c962fe98bf081bfefaee1c7496f18bd06d` present at audit start; unrelated dirty work preserved outside this task.

---

## 2. Audited daily-use journey

After PRODUCT-3 first-run, a normal user should manage workspace context, inspect knowledge, run lifecycle actions, see freshness/problems, Ask (including scoped Ask where product-supported), and adjust basic non-vendor configuration - all via Slack DM without HTTP/curl.

**What works today in Slack (post-onboarding):**

- Workspace **create**, **switch** (`workspace.activate` → `ConversationWorkspaceSelectionService`), and **list** (count only in renderer).
- **Ask** (`workspace.ask` → `WorkspaceAskService` v1), **citations** in renderer, **insufficient-evidence** outcome, **citation inspect/open** (`citation.inspect` → `ConversationCitationContextService` + `DocumentInspectService`).
- First-run **attention/sync guidance** via `WorkspaceSetupSnapshotService` + `ConversationSetupOnboardingPresenter` (still appended on many turns when phase is `READY`).

**What does not work for daily use:**

- Unified **knowledge inventory** (indexed + live bindings) is not a conversational action; `source.list` returns legacy `WorkspaceSource` rows only.
- **Sync / retry / disable / enable / detach** exist in `KnowledgeOperationsService` and `KnowledgeAdministrationService` but are **not** planner actions and are **not** wired in `companion.py` executor composition.
- **Freshness** (`last_successful_sync_at`, `runtime_available`) exists on `KnowledgeInventoryItemV1` but is not rendered in Slack.
- **Source-scoped Ask** has no planner field, no `WorkspaceAskService.ask()` parameter, and conversation path uses indexed v1 whole-workspace search only.
- **Workspace list / active workspace** artifacts carry names and `is_active`; renderer shows counts only.

---

## 3. Capability matrix

| # | User need | Existing backend/application capability | Slack/conversation wiring | Result | Evidence | Minimal PRODUCT-4 change |
|---|---|---|---|---|---|---|
| W1 | List workspaces with recognizable names | `ManagedWorkspaceService.list_workspaces`; `workspace.list` planned action | Planner + executor populate `workspaces[]` with `name`, `is_active` | **UX_GAP** | `interaction_executor.py` `WorkspaceListPlannedAction`; `interaction_response_renderer.py` `workspace.list` → count only | Extend renderer (and prompt examples) to list numbered workspace names and mark active |
| W2 | See which workspace is currently selected | `ConversationWorkspaceSelectionService`; `active_workspace_id` on planning request; activate artifact | Active name shown after `workspace.activate`; not consistently on other turns | **UX_GAP** | `interaction_application_service.py` `_build_planning_request`; renderer `Active workspace:` insert | Show active workspace in inventory/status responses and optionally a lightweight status prefix |
| W3 | Switch workspace | `workspace.activate` + `ConversationWorkspaceSelectionService.select_personal_workspace` | Wired in executor + companion | **READY** | `interaction_executor.py`; `test_conversation_first_run_application.py` `test_workspace_switch_is_executed_when_snapshot_is_not_ready` | - |
| W4 | Create workspace | `workspace.create` → `ManagedWorkspaceService.create_workspace` | Wired | **READY** | `interaction_executor.py`; PRODUCT-3 §13 step 3 | - |
| W5 | Delete workspace safely | `workspace.delete` → `ManagedWorkspaceService.delete_workspace` (immediate, no confirmation token) | Planner action exists; no confirmation flow | **REUSABLE_CAPABILITY_GAP** | `interaction_executor.py` `WorkspaceDeletePlannedAction`; `service.py` `delete_workspace` | Product decision: reusable confirmation for workspace delete, or exclude from daily UX; if included, reuse HMAC-style pattern analogous to knowledge detach |
| K1 | List all knowledge sources (inventory) | `KnowledgeInspectionService.list_items` → `KnowledgeInventoryV1`; HTTP `GET …/knowledge/inventory` | No `knowledge.inventory.list` (or administration LIST) conversational action | **WIRING_GAP** | `knowledge_inspection_operations_service.py`; `workspace_routes.py` inventory route; no match in `interaction_models.py` `PlannedAction` | Add planned action + executor calling inspection service (or `KnowledgeAdministrationService` LIST) |
| K2 | Understand source name, type, indexed vs live | `KnowledgeInventoryItemV1` (`display_label`, `mode`, `source_kind`, `provider_id`) | Not projected to Slack | **WIRING_GAP** | `KnowledgeInventoryItemV1` fields; `source.list` returns only legacy `source_id/source_type/status` | Renderer rows from inventory item projection |
| K3 | See usable vs unusable | `lifecycle_state`, `enabled`, `detached`, `sync_state`, `runtime_available` on inventory item | Not on conversational path | **WIRING_GAP** | `knowledge_inspection_operations_service.py`; `workspace_setup_snapshot_service.py` `_item_usable` | Inventory list/detail presentation |
| K4 | See attention-required sources | Inventory `summary.attention_required`; snapshot `SetupAttentionV1` (first item only) | Snapshot guidance only; no inventory list | **WIRING_GAP** | `workspace_setup_snapshot_service.py` `_attention_for_items`; onboarding presenter | Daily inventory + attention list; optional filter via administration LIST |
| L1 | Sync / refresh a source | `KnowledgeOperationsService` `SYNC`; HTTP `POST …/knowledge/items/{id}/operations` | Not planner action; `KnowledgeAdministrationService` not in companion | **WIRING_GAP** | `KnowledgeOperationV1.SYNC`; `knowledge_administration_service.py`; `companion.py` executor wiring | Wire administration or typed `knowledge.operation.execute` action |
| L2 | Retry failed sync | `RETRY_SYNC` operation + inventory `available_actions` | Not on conversational path | **WIRING_GAP** | `KnowledgeOperationV1`; `test_knowledge_surface_http.py` | Same as L1 |
| L3 | Disable source | `DISABLE` operation | Not on conversational path | **WIRING_GAP** | `knowledge_administration_service.py` `DeterministicKnowledgeAdministrationIntentInterpreter` | Same as L1 |
| L4 | Enable source | `ENABLE` operation | Not on conversational path | **WIRING_GAP** | Same | Same as L1 |
| L5 | Detach with safe confirmation | `DETACH` + `HmacKnowledgeAdministrationConfirmationCodec`; HTTP 409 `knowledge_admin_confirmation_required` | Confirmation codec used on HTTP detach; **not** Slack conversation | **WIRING_GAP** | `workspace_routes.py` `execute_knowledge_item_operation`; `knowledge_administration_service.py` | Conversational detach → confirmation token round-trip via durable conversation context (reuse codec, not Slack-only state) |
| F1 | Last successful sync / last activity | `KnowledgeInventoryItemV1.last_successful_sync_at` | Not rendered in Slack | **WIRING_GAP** | `knowledge_inspection_operations_service.py` line 88 | Show formatted timestamp in inventory/detail lines |
| F2 | Staleness / “is this current?” semantics | Partial signals (`sync_state`, `last_successful_sync_at`); no canonical stale flag | Not in Slack | **LATER_PRODUCT_BOUNDARY** | PRODUCT_CONTRACT PRODUCT-7; no `stale` field on inventory | PRODUCT-4 may show last sync; stale policy → PRODUCT-7 |
| F3 | Live runtime availability | `runtime_available` on inventory item | Not in Slack | **WIRING_GAP** | `KnowledgeInventoryItemV1.runtime_available` | Inventory row for live bindings |
| F4 | Honest unknown when freshness unavailable | Nullable fields on inventory item | Not surfaced | **WIRING_GAP** | Inventory model nullability | Renderer uses “unknown” when null |
| P1 | Know which source has a problem | Snapshot `attention.knowledge_item_id` without `display_label` | User sees error class, not source name | **UX_GAP** | `SetupAttentionV1`; `conversation_setup_onboarding.py` `_attention_lines` | Map `knowledge_item_id` → `display_label` in attention UX |
| P2 | Understandable explanation | `last_error_code` → mapped messages in onboarding presenter | Only in snapshot attention path | **UX_GAP** | `_ATTENTION_ERROR_MESSAGES` | Extend mappings; surface in daily inventory |
| P3 | Execute recovery without logs | `available_actions` on inventory item | Text hints only (`_ATTENTION_ACTION_MESSAGES`); no executable wiring | **WIRING_GAP** | `conversation_setup_onboarding.py` action messages vs no executor | Wire L1–L5 |
| A1 | Normal Ask | `workspace.ask` → `WorkspaceAskService.ask` | Wired; gated when snapshot blocks Ask | **READY** | `interaction_executor.py`; PRODUCT-3 §13 steps 11–12 | - |
| A2 | Source-scoped Ask | Indexed v1 searches whole workspace collection; hybrid v2 + query policy on HTTP only | No `source_ids` on `WorkspaceAskPlannedAction`; executor calls `ask()` without scope | **REUSABLE_CAPABILITY_GAP** | `ask_service.py` `ask()` params; `WorkspaceAskPlannedAction` fields; hybrid not in companion | Define reusable Ask scope contract (indexed binding / legacy source); planner + executor; citations must stay scoped |
| A3 | Citations in answer | `WorkspaceAskService` citations in artifact | Renderer lists safe file names | **READY** | `test_interaction_response_renderer.py` `test_renderer_includes_safe_ask_citation` | - |
| A4 | Inspect / open citation | `citation.inspect` | Wired | **READY** | `test_conversation_citation_inspect.py` | - |
| A5 | Insufficient evidence behavior | `AskRunStatus.INSUFFICIENT_EVIDENCE` | Renderer safe message | **READY** | `interaction_response_renderer.py` `workspace.ask` branch | - |
| C1 | Basic daily configuration (non-vendor) | `knowledge.connections.list`, `knowledge.resources.list`, `knowledge.capabilities.list` for discovery | Wired for intake/discovery; not “settings” UX | **WIRING_GAP** | `interaction_executor.py` knowledge discovery actions | Clarify product copy; optional inventory-driven status command |
| C2 | Query policy / hybrid mode tuning | `WorkspaceQueryPolicyService`; HTTP query-policy routes | Not conversational | **LATER_PRODUCT_BOUNDARY** | PRODUCT_CONTRACT PRODUCT-5/6; `knowledge_query_policy_routes.py` | PRODUCT-4 excludes vendor auth; hybrid policy UX → PRODUCT-6 |
| UX1 | Non-repetitive daily responses | `ConversationSetupOnboardingPresenter.render_snapshot_guidance` | Appends READY body + suggested question on every success when phase is `READY` | **UX_GAP** | `interaction_application_service.py` `_append_setup_guidance`; `conversation_setup_onboarding.py` READY branch | Daily mode: suppress or shorten READY append after first-run complete |

---

## 4. Existing reusable foundations

| Area | Component | Reachable from Slack today | Notes |
|---|---|---|---|
| Conversational orchestration | `ConversationInteractionApplicationService` | Yes (`companion.py`) | Canonical path; setup snapshot + onboarding presenter |
| Planner / executor / renderer | `ConversationInteractionPlanner`, `ConversationInteractionExecutor`, `ConversationInteractionResponseRenderer` | Yes | Action union in `interaction_models.py` - no knowledge lifecycle actions |
| Workspace selection | `ConversationWorkspaceSelectionService` + `ConversationContextRepository` | Yes | Durable per conversation |
| Setup / attention derivation | `WorkspaceSetupSnapshotService` | Yes (read-only UX) | First-run oriented; not full daily inventory |
| Knowledge inventory | `KnowledgeInspectionService` / `KnowledgeInventoryV1` | HTTP only | PRODUCT-3B closed |
| Knowledge operations | `KnowledgeOperationsService` | HTTP only | Execute + operation list |
| NL administration + confirmation | `KnowledgeAdministrationService` + `HmacKnowledgeAdministrationConfirmationCodec` | **No** (constructed in `workspace_routes.py`, stored on `app.state`, not passed to executor) | Acceptance tests only |
| Indexed Ask v1 | `WorkspaceAskService` | Yes | Whole-workspace scope only |
| Hybrid Ask v2 | `WorkspaceAskServiceV2` / hybrid stack | HTTP v2 only | Not companion wiring |
| Citations context | `ConversationCitationContextService` | Yes | |
| Document inspect/open | `DocumentInspectService` | Yes | PRODUCT-3E closed |
| Legacy source list | `ManagedWorkspaceService.list_sources` | Yes via `source.list` | Not unified inventory |

---

## 5. Confirmed gaps

### Wiring gaps (Slack cannot reach existing capability)

1. Unified knowledge inventory and lifecycle (sync, retry, disable, enable, detach+confirmation).
2. Freshness and runtime fields from inventory.
3. Executable recovery actions (not hint text only).
4. Presentation of workspace list/active workspace despite rich executor artifacts.

### UX gaps (reachable but inadequate for daily use)

1. Workspace list shows count, not names/active marker.
2. Attention UX lacks source display label.
3. Repetitive READY/suggested-question append on ongoing daily turns.
4. Error explanations only in snapshot attention subset.

### Reusable capability gaps (no canonical contract yet)

1. **Source-scoped Ask** - neither `WorkspaceAskPlannedAction` nor `WorkspaceAskService.ask()` accepts scope; indexed search uses workspace-wide collection; conversation does not use hybrid v2.
2. **Workspace delete confirmation** - immediate delete; unlike knowledge detach HMAC flow.

### Honest unresolved (bounded audit stop)

- **Indexed Ask internal source filter at search layer:** Not verified whether Qdrant/metadata filters could scope without new product contract; audit treats scoped Ask as missing at application/planner boundary per task instruction.

---

## 6. Architectural decisions required

| Decision | Why | Options (discussion only) |
|---|---|---|
| Source-scoped Ask contract | Product workflow #4 requires scoping; no planner/ask parameter today | (a) `source_id` / `knowledge_item_id` on Ask action + indexed retrieval filter; (b) query-policy-only live scope via hybrid v2; (c) both with unified scope model |
| Workspace delete in daily UX | Destructive, no confirmation | Exclude from PRODUCT-4 daily surface vs reusable confirmation parity with detach |
| Administration integration style | `KnowledgeAdministrationService` exists but orphaned | Typed `knowledge.*` planned actions vs administration `handle()` behind single action |
| Daily vs first-run guidance | Same presenter appends READY on every turn | Derive “onboarding complete” from snapshot phase `READY` + suppress repeat guidance |

**Invariant checks (future PRODUCT-4 must preserve):**

- Slack remains thin client over application services.
- No duplicate lifecycle/configuration truth in Slack.
- Tenant/workspace isolation via existing context + services.
- Detach confirmation via reusable HMAC codec (not Slack-only confirmation store).
- Durable conversation context for citation and workspace selection survives restart.

---

## 7. Explicit later-product boundaries

| Area | Milestone | PRODUCT-4A stance |
|---|---|---|
| Vendor OAuth / provider onboarding | PRODUCT-5 | Do not pull into PRODUCT-4 |
| Broad Ask quality / hybrid polish | PRODUCT-6 | Source-scoped Ask gap flagged; full hybrid daily UX deferred |
| Stale indication / scheduled sync / background retry | PRODUCT-7 | Show `last_successful_sync_at` only; no new stale model |
| General failure catalog redesign | PRODUCT-8 | Use existing `last_error_code` + inventory actions |
| Migration / backups | PRODUCT-9 | Out of scope |
| Observability dashboards | PRODUCT-10 | Out of scope |
| Clean-machine live Slack proof | PRODUCT-11 | Out of scope |

---

## 8. Recommended coherent implementation blocks for PRODUCT-4

Do not split into artificial microtasks. Group by dependency and user cohesion:

### Block A - Daily knowledge surface in Slack (inventory, freshness, attention)

- Add conversational path to `KnowledgeInspectionService` (list/show) with renderer rows: label, mode, state, `last_successful_sync_at`, `runtime_available`, `available_actions`.
- Fix workspace list/active presentation in renderer.
- Tune `_append_setup_guidance` for daily mode (no repetitive READY spam).
- Map attention `knowledge_item_id` → `display_label`.

**Depends on:** existing PRODUCT-3B inventory model only.

### Block B - Daily knowledge lifecycle in Slack (sync, retry, disable, enable, detach)

- Wire `KnowledgeAdministrationService` or typed operations into planner/executor.
- Reuse `HmacKnowledgeAdministrationConfirmationCodec` for detach confirmation round-trip in conversation context (token in user message or structured confirm action).
- Planner prompt rules for NL sync/disable/enable/detach/list.

**Depends on:** Block A for item identity resolution.

### Block C - Source-scoped Ask (after architecture decision)

- New reusable Ask scope on application service + `WorkspaceAskPlannedAction`.
- Executor + citation scoping verification.
- Optional later: hybrid v2 companion wiring (PRODUCT-6).

**Depends on:** architecture decision §6; independent of Block B except shared inventory labels.

**Suggested sequencing:** A → B → (decision) C.

**PRODUCT-4B / 4C split justified only if:** Block C decision or implementation latency blocks A+B delivery.

---

## 9. Proposed next task

**LKW-PRODUCT-4 - SLACK DAILY-USE PRODUCT EXPERIENCE (Block A + B)**

Implement daily Slack inventory/freshness/attention presentation and lifecycle wiring over accepted inspection/operations/administration services. Defer source-scoped Ask to **PRODUCT-4C** pending Ask-scope architecture decision.

---

## 10. Evidence / read inventory

### Canonical product docs (2)

1. `applications/local_workspace_application/docs/product/PRODUCT_CONTRACT.md`
2. `applications/local_workspace_application/docs/product/PRODUCT_3_FIRST_RUN_GAP_AUDIT.md`

### Production / source files read (12 - budget respected)

1. `applications/local_workspace_application/conversation/interaction_application_service.py` (sections)
2. `applications/local_workspace_application/conversation/interaction_executor.py` (sections)
3. `applications/local_workspace_application/conversation/interaction_models.py` (sections)
4. `applications/local_workspace_application/conversation/interaction_response_renderer.py` (sections)
5. `applications/local_workspace_application/conversation/conversation_setup_onboarding.py`
6. `applications/local_workspace_application/conversation/interaction_prompt.py` (grep + sections)
7. `applications/local_workspace_application/slack_companion/companion.py` (sections)
8. `applications/local_workspace_application/workspaces/conversation_workspace_selection_service.py` (sections)
9. `applications/local_workspace_application/workspaces/knowledge_inspection_operations_service.py` (sections)
10. `applications/local_workspace_application/workspaces/knowledge_administration_service.py` (sections)
11. `applications/local_workspace_application/workspaces/workspace_setup_snapshot_service.py` (sections)
12. `applications/local_workspace_application/workspaces/ask_service.py` (sections)

**Targeted discovery (no full read):** `workspace_routes.py` - inventory route, knowledge operation execute, detach confirmation via grep; `serving/workspace_routes.py` symbols only.

### Tests read (8 - budget respected)

1. `tests/conversation/test_conversation_first_run_application.py` (grep)
2. `tests/conversation/test_conversation_citation_inspect.py` (sections)
3. `tests/conversation/test_interaction_application_service.py` (grep)
4. `tests/conversation/test_interaction_response_renderer.py` (grep)
5. `tests/workspaces/test_conversation_workspace_selection_service.py` (grep)
6. `tests/serving/test_knowledge_surface_http.py` (grep)
7. `tests/serving/test_workspace_setup_snapshot.py` (grep)
8. `tests/workspaces/test_knowledge_administration_service.py` (grep)

### Validation

- `git diff --check` - run at commit time.
- Symbol references verified via targeted grep against paths above.
- No full test suite run (documentation-only task).

---

## Audit metadata

**Files changed:** `applications/local_workspace_application/docs/product/PRODUCT_4_SLACK_DAILY_USE_GAP_AUDIT.md` (this file only).

**Concurrent work:** preserve unrelated repository changes; stage only this file at commit.

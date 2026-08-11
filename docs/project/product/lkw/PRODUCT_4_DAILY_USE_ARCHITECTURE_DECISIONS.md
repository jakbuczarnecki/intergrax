# LKW PRODUCT-4 — Daily-Use Architecture Decisions (ARCH-1)

## 1. User-facing decision summary

**Status:** READY_FOR_REVIEW  
**Task:** LKW-PRODUCT-4-ARCH-1 — FREEZE DAILY-USE ASK SCOPE AND DESTRUCTIVE CONFIRMATION CONTRACTS  
**Mode:** bounded architecture decision / evidence only — no production code changed

Two open decisions from the accepted PRODUCT-4A audit are **frozen**:

| Decision | Outcome |
|---|---|
| **Source-scoped Ask** | Canonical scope identity is **`knowledge_item_id`** from the unified inventory. Reusable application contract: **`KnowledgeAskScopeV1`** (`knowledge_item_ids`). **Indexed** scoped Ask is enforceable today via inventory validation + post-retrieval `source_id` filtering. **Live** scoped Ask is **deferred** — not enforceable on the current companion Ask path. |
| **Workspace delete** | **Confirmation required** before any destructive delete on the daily conversational path. Existing HMAC confirmation is **knowledge-specific** but **safely extractable** into a reusable destructive-action confirmation boundary. Until wired, **`workspace.delete` must fail closed** (no unconfirmed execution). |

**PRODUCT-4 sequencing (if both resolve):**

1. **PRODUCT-4B** — daily inventory, freshness, attention, lifecycle, confirmation UX (including workspace delete), workspace daily UX  
2. **PRODUCT-4C** — source-scoped Ask using the frozen `KnowledgeAskScopeV1` contract (indexed mode initially)  
3. **PRODUCT-4** final acceptance

**Git context:** branch `development`; required ancestor `23388770b97c475f5f46932da73022eeb7ce120e` verified at task start.

---

## 2. Source-scoped Ask decision

### Status: **RESOLVED**

Indexed scoped Ask is architecturally resolved with a bounded follow-up for application-layer filtering. Live scoped Ask is explicitly deferred.

### Canonical scope identity

**`knowledge_item_id`** on `KnowledgeInventoryItemV1` — not legacy `source_id`, provider IDs, Slack IDs, Qdrant IDs, or storage IDs.

Inventory encoding (verified):

- Indexed: `indexed:{indexed_source_binding_id}` via `indexed_knowledge_item_id()`
- Live: `live:{live_access_binding_id}` via `live_knowledge_item_id()`
- Parsed by `_parse_knowledge_item_id()` → `(KnowledgeAccessModeV1, binding_id)`

Each inventory item also carries `source_id` (indexed), `indexed_source_binding_id`, `live_access_binding_id`, and binding refs — these are **derivation fields**, not canonical Ask scope identity.

### Exact reusable contract

**No existing canonical multi-item Ask scope type was found.** `KnowledgeOperationCommandV1` binds a single `knowledge_item_id` for mutations; `KnowledgeAdministrationFilterV1` is list filtering only; `WorkspaceAskPlannedAction` has no scope field.

**Frozen minimum reusable contract:**

```text
KnowledgeAskScopeV1
  knowledge_item_ids: tuple[str, ...]   # non-empty, unique, bounded
```

Optional future field (not required for PRODUCT-4C freeze): `mode_hint` — **not frozen**; mode is derived per item from inventory.

Planner may propose scope; **planner output is never authorization**.

### Validation authority

Application layer **after** tenant/workspace context is established:

| Check | Authority |
|---|---|
| `tenant_id` / `workspace_id` scope | Existing workspace authority (`ManagedWorkspaceService` / conversation executor tenant binding) |
| Each `knowledge_item_id` belongs to tenant/workspace | `KnowledgeInspectionService.get_item()` |
| Item exists | same |
| Item eligible for Ask | Inventory-derived usability (indexed: not detached/disabled + lifecycle usable per `_item_usable` / `_indexed_usable`; live: `_live_usable` when live path exists) |
| Disabled / detached / unavailable semantics | Inventory fields `enabled`, `detached`, `lifecycle_state`, `runtime_available`, `sync_state` |
| Malformed / stale / unauthorized scope | Fail closed — reject before retrieval |

Validated scope yields **allowed indexed `source_id` set** (for indexed items) used for evidence enforcement.

### Indexed enforcement path

**Retrieval boundary today:** `WorkspaceAskService.ask()` → task `local.workspace.search` with metadata `tenant_id`, `workspace_id`, `collection_id=workspace_id`, `query`, `top_k` — **whole-workspace collection**, no source/binding filter in task metadata.

**Search agent metadata keys** (`agents/local_search/steps/search_job.py` `_LKW_SEARCH_METADATA_KEYS`): `query`, `collection_id`, `top_k`, `tenant_id`, `user_id`, `workspace_id` — **no `source_id` or `knowledge_item_id` filter**.

**Evidence mapping:** `map_search_hits()` in `search_evidence.py` verifies hits against repository document refs and **already drops** hits when `source_id` on evidence disagrees with `ref.source_id` (line 103–104). It accepts no scope parameter today.

**Minimum safe indexed enforcement (application layer, no new retrieval subsystem):**

1. Validate `KnowledgeAskScopeV1` via inventory.  
2. Resolve indexed items → `source_id` set.  
3. Run existing workspace-wide search.  
4. **Post-filter** hits to `source_id ∈ allowed_source_ids` (extend `map_search_hits` with optional `allowed_source_ids` **or** filter in `WorkspaceAskService` after mapping).  
5. Assembly + `project_ask_citations()` only on filtered evidence.

**Qdrant/metadata pre-filter at search layer:** not required for PRODUCT-4C; not verified as available without new task metadata contract. Efficiency optimization is a later concern.

### Live / hybrid enforcement path

| Path | Current state | PRODUCT-4C |
|---|---|---|
| **Indexed v1 Ask** (`WorkspaceAskService`) | Companion-wired; indexed search only | **Supported** for scoped Ask (post-filter path above) |
| **Live** | `live:{binding_id}` inventory items; execution via `hybrid_ask_execution` / live handlers with `live_access_binding_id` | **Not supported** — companion does not use hybrid/live Ask |
| **Hybrid v2** | HTTP-only; not companion wiring | **Deferred** (PRODUCT-6) |

**One provider-neutral scope type** (`KnowledgeAskScopeV1`) can **represent** both modes at the contract level, but **cannot safely enforce both** through a single retrieval path today. PRODUCT-4C must initially support **only indexed scoped Ask**.

### Citation invariant

- Citations projected only from **validated, scope-filtered** evidence (`project_ask_citations` on filtered hits).  
- Citations expose workspace `source_id` and safe file names from verified document refs — **not** raw provider/storage/vector IDs to Slack.  
- If scoped retrieval yields no verified evidence → `INSUFFICIENT_EVIDENCE` (existing behavior).  
- Citations must not reference evidence outside validated scope (enforced by pre-assembly filter).

### Unsupported / deferred modes

- Live scoped Ask (`live:*` items)  
- Hybrid v2 scoped Ask  
- Scoped Ask via legacy `source_id` without inventory `knowledge_item_id`  
- Scoped Ask via planner-only `source.list` / `WorkspaceSource` rows  
- Search-layer pre-filter without application validation  
- Multi-workspace scope

### Exact implementation boundary for PRODUCT-4C

**In scope:**

1. Add `KnowledgeAskScopeV1` (frozen shape above).  
2. Extend `WorkspaceAskPlannedAction` with optional `ask_scope: KnowledgeAskScopeV1 | None` (whole-workspace when absent).  
3. Extend `WorkspaceAskService.ask(..., scope: KnowledgeAskScopeV1 | None = None)`.  
4. Add scope validation (inventory-backed) before `_retrieve_verified_evidence`.  
5. Extend evidence mapping with `allowed_source_ids` post-filter for indexed items.  
6. Wire executor to pass validated scope; verify citation scoping in tests.

**Out of scope (PRODUCT-4C):**

- Live/hybrid scoped retrieval  
- New search task metadata / Qdrant filter  
- Slack-only scope encoding  
- Planner as authorization

**Bounded follow-up (feasibility, not architecture block):** optional `allowed_source_ids` parameter on `map_search_hits` vs inline filter — implementation choice within boundary.

---

## 3. Workspace-delete decision

### Status: **RESOLVED**

### Confirmation required

**Frozen product rule:** `workspace.delete` on the daily conversational path is allowed **only** through reusable explicit confirmation (two-step: issue token → confirm with token). **Single-turn unconfirmed deletion is rejected.**

Current state: `WorkspaceDeletePlannedAction` → `ConversationInteractionExecutor` → `ManagedWorkspaceService.delete_workspace()` **immediately** — no confirmation.

### Existing reusable confirmation evidence

`KnowledgeAdministrationService` + `HmacKnowledgeAdministrationConfirmationCodec`:

| Property | Evidence |
|---|---|
| Stateless HMAC token | `issue()` / `verify()` — URL-safe payload + HMAC-SHA256 signature |
| Binding fields | `tenant_id`, `workspace_id`, `knowledge_item_id`, `operation`, `expected_revision`, `expires_at` |
| TTL | Default 5 minutes (`timedelta(minutes=5)`) |
| Tamper protection | Signature mismatch → `knowledge_admin_confirmation_invalid` |
| Expiry | `expires_at <= now` → `knowledge_admin_confirmation_expired` |
| Cross-workspace replay | Token from workspace A rejected in workspace B (`test_confirmation_cannot_cross_workspace_and_parser_is_replaceable`) |
| Stale target | `item.revision != confirmation.expected_revision` → `knowledge_admin_confirmation_stale` |
| Detach flow | DETACH issues token; resume with same token executes (`test_detach_confirmation_is_signed_bound_stale_and_restart_safe`) |
| HTTP parity | `test_destructive_operation_requires_confirmation` in `test_knowledge_surface_http.py` |

`Workspace` model has **no lifecycle revision** — only `updated_at` (and metadata fields). Stale binding for workspace delete must use a **state version** derived from workspace record (e.g. `updated_at` epoch seconds), not `KnowledgeRevisionKindV1`.

**Not reusable directly:** payload schema and `_verify_confirmation()` are bound to `knowledge_item_id` + `KnowledgeOperationV1`. Workspace delete target is the workspace itself.

**Slack companion** has local delete-confirm rendering/parsing (`slack_companion/workflow.py`, `rendering.py`) — **not** the canonical confirmation primitive; must not become the daily-use confirmation store.

### Reuse / generalization decision

**Extract, do not duplicate crypto.**

Minimum reusable boundary:

```text
DestructiveActionConfirmationV1
  tenant_id: str
  workspace_id: str          # context workspace (for delete: same as target)
  action_kind: str           # e.g. "knowledge.detach", "workspace.delete"
  target_id: str             # knowledge_item_id or workspace_id
  expected_state_version: int # item.revision or workspace updated_at epoch
  expires_at: datetime

HmacDestructiveActionConfirmationCodec
  # Same HMAC/TTL mechanics as HmacKnowledgeAdministrationConfirmationCodec
  # Payload schema version 2 (or shared codec with version discriminator)
```

**Migration approach (preserve knowledge-detach behavior):**

1. Introduce generic codec + confirmation model.  
2. `HmacKnowledgeAdministrationConfirmationCodec` becomes adapter **or** thin wrapper mapping `KnowledgeAdministrationConfirmationV1` ↔ `DestructiveActionConfirmationV1` with `action_kind="knowledge.detach"` (or operation-specific kinds) and `target_id=knowledge_item_id`.  
3. `KnowledgeAdministrationService._verify_confirmation` continues equivalent binding checks via adapter.  
4. New `WorkspaceDeletionService` (or executor helper) uses same codec + secret for `action_kind="workspace.delete"`, `target_id=workspace_id`, `expected_state_version=int(workspace.updated_at.timestamp())`.  
5. Shared secret configuration (same as `knowledge_admin_confirmation_secret` on HTTP routes) — **one secret, one crypto primitive**.

### Binding fields (minimum)

| Field | Workspace delete |
|---|---|
| Tenant | `tenant_id` |
| Target | `target_id` = `workspace_id` to delete |
| Context workspace | `workspace_id` (must match target) |
| Operation kind | `action_kind = workspace.delete` |
| State version | `expected_state_version` from `Workspace.updated_at` |
| Expiry | `expires_at` (short TTL, same class as knowledge detach) |
| Cryptographic authenticity | HMAC signature over canonical payload |

### Expiry / replay semantics

| Scenario | Behavior |
|---|---|
| Expired token | Reject — `confirmation_expired` (fail closed) |
| Tampered token | Reject — `confirmation_invalid` |
| Wrong tenant / workspace / target / action_kind | Reject — `confirmation_invalid` |
| Workspace changed after issue (`updated_at` differs) | Reject — `confirmation_stale` |
| Valid token within TTL, workspace unchanged | Allow delete (idempotent delete after success: second attempt → workspace not found / already deleted) |
| Replay within TTL before delete | Same as knowledge detach: **allowed** until state version changes or TTL expires — not a second independent secret system |

Do **not** weaken existing HMAC semantics (signature, expiry, binding equality checks).

### Failure behavior

| State | Behavior |
|---|---|
| Confirmation not wired for daily UX | **Fail closed** — do not expose executable unconfirmed `workspace.delete` on conversational path |
| Invalid / expired / stale token | Reject; no deletion |
| Valid confirmation | `ManagedWorkspaceService.delete_workspace()` (existing destructive implementation) |

### Exact implementation boundary (PRODUCT-4B)

**In scope:**

1. Generic `DestructiveActionConfirmationV1` + `HmacDestructiveActionConfirmationCodec` (extracted from knowledge codec).  
2. Knowledge detach adapter — existing tests must pass unchanged.  
3. Workspace delete confirmation issue/verify around `delete_workspace`.  
4. Conversational two-step: `workspace.delete` planned action returns `confirmation_required` + token; separate confirm path (resume action or confirmation token on follow-up) executes delete.  
5. Block immediate delete in `ConversationInteractionExecutor` until confirmation supplied.

**Out of scope:**

- Slack-only in-memory pending-delete map as source of truth  
- Duplicate confirmation secrets  
- Second unrelated confirmation protocol

---

## 4. Rejected alternatives

| Alternative | Rejection |
|---|---|
| Legacy `source_id` as canonical Ask scope | Product-neutral inventory uses `knowledge_item_id`; `source_id` is indexed derivation only |
| Provider-specific / Slack IDs as scope | Violates provider-neutral inventory contract |
| Slack-only retrieval filter | Ask scope must be reusable across Slack, Teams, web, mobile, CLI |
| Search-layer scope without inventory validation | Planner/retrieval cannot authorize; fails tenant/workspace/item eligibility |
| Live scoped Ask on PRODUCT-4C companion path | No live Ask execution on companion; would invent parallel subsystem |
| Slack-only confirmation store for workspace delete | Existing Slack workflow local state is not durable reusable confirmation |
| In-memory confirmation token mapping | Stateless HMAC tokens required (knowledge detach pattern) |
| Immediate workspace deletion on conversational path | Destructive; violates frozen product rule |
| Duplicate confirmation crypto / second secret system | Extract generic codec; one secret |
| Reusing knowledge codec verbatim for workspace delete without generalization | Payload binds `knowledge_item_id` + `KnowledgeOperationV1` — wrong target semantics |
| `workspace.delete` excluded from daily UX without fail-closed guard | Must not leave silent immediate delete executable |

---

## 5. Required reusable capability additions

Only items **genuinely missing** today:

| Addition | Reason |
|---|---|
| **`KnowledgeAskScopeV1`** | No existing multi-item Ask scope type |
| **Inventory-backed Ask scope validation** | Authorization cannot live in planner or raw retrieval |
| **`allowed_source_ids` post-filter** on evidence path (`map_search_hits` or equivalent) | Search task has no source filter; scoped indexed Ask needs application enforcement |
| **`DestructiveActionConfirmationV1` + generic HMAC codec** | Knowledge codec is not directly reusable for workspace delete |
| **Workspace delete confirmation orchestration** | Two-step flow missing on conversational path |

No new retrieval subsystem, Slack-only stores, or duplicate crypto required.

---

## 6. Final architecture status

| Decision | Status | Notes |
|---|---|---|
| Source-scoped Ask — canonical contract | **RESOLVED** | `knowledge_item_id` + `KnowledgeAskScopeV1` |
| Source-scoped Ask — indexed enforcement | **RESOLVED** | Post-filter via validated `source_id` set; search layer unscoped |
| Source-scoped Ask — live/hybrid | **RESOLVED (deferred)** | Explicitly out of PRODUCT-4C; not ARCHITECTURE_BLOCKED |
| Workspace delete confirmation | **RESOLVED** | Generic extraction + two-step conversational flow |
| Workspace delete — direct codec reuse | **RESOLVED (rejected)** | Generalization required; pattern reuse approved |

No **ARCHITECTURE_BLOCKED** items — bounded implementation follow-ups remain within PRODUCT-4B/4C.

---

## 7. Consequence for PRODUCT-4 implementation sequence

1. **PRODUCT-4B** can proceed: inventory/lifecycle/freshness/attention UX; wire `KnowledgeAdministrationService` confirmation for detach; implement **generic destructive confirmation** and **workspace delete two-step**; fix workspace list/active presentation; suppress repetitive READY guidance. **Do not** ship unconfirmed `workspace.delete` on conversational executor.  
2. **PRODUCT-4C** proceeds after 4B (or in parallel once scope contract is frozen — now frozen): implement `KnowledgeAskScopeV1` + indexed scoped Ask only; extend planner/executor/`WorkspaceAskService`; citation scoping tests.  
3. **Live scoped Ask** and **hybrid daily UX** remain PRODUCT-6 / later — not blockers for PRODUCT-4B or indexed PRODUCT-4C.  
4. **PRODUCT-4 final acceptance** after 4B + 4C + audit criteria from PRODUCT-4A.

---

## 8. Evidence / read inventory

### Accepted PRODUCT-4 audit (1)

1. `docs/project/product/lkw/PRODUCT_4_SLACK_DAILY_USE_GAP_AUDIT.md`

### Production / source files (10 — budget respected)

1. `applications/local_workspace_application/workspaces/ask_service.py`
2. `applications/local_workspace_application/conversation/interaction_models.py` (sections: `WorkspaceAskPlannedAction`, `WorkspaceDeletePlannedAction`)
3. `applications/local_workspace_application/workspaces/knowledge_inspection_operations_service.py` (sections: inventory model, `indexed_knowledge_item_id`, `live_knowledge_item_id`, `_indexed_item`)
4. `applications/local_workspace_application/workspaces/knowledge_administration_service.py` (sections: confirmation models, codec, detach flow, `_verify_confirmation`)
5. `applications/local_workspace_application/workspaces/service.py` (section: `delete_workspace`)
6. `applications/local_workspace_application/workspaces/search_evidence.py`
7. `agents/local_search/steps/search_job.py` (section: `_LKW_SEARCH_METADATA_KEYS`)
8. `applications/local_workspace_application/workspaces/workspace_setup_snapshot_service.py` (section: `_item_usable`)
9. `applications/local_workspace_application/workspaces/models.py` (section: `Workspace`)
10. `applications/local_workspace_application/workspaces/ask_answer_assembler.py` (grep: `project_ask_citations`)

**Targeted discovery (grep / symbols only, not full read):**

- `applications/local_workspace_application/conversation/interaction_executor.py` — `WorkspaceAskPlannedAction`, `WorkspaceDeletePlannedAction` execution
- `applications/local_workspace_application/workspaces/hybrid_ask_execution.py` — live path symbols
- `applications/local_workspace_application/slack_companion/workflow.py` — Slack-local delete confirm (rejected pattern)

### Focused tests (4 — budget 6)

1. `applications/local_workspace_application/tests/workspaces/test_knowledge_administration_service.py` — `test_detach_confirmation_is_signed_bound_stale_and_restart_safe`, `test_confirmation_cannot_cross_workspace_and_parser_is_replaceable`
2. `applications/local_workspace_application/tests/serving/test_knowledge_surface_http.py` — `test_destructive_operation_requires_confirmation` (grep)

### Validation

- `git diff --check` — run at commit time.
- Symbol/path references verified against files read above.
- No full test suite (documentation-only task).
- Document internally consistent: canonical identities, enforcement paths, and deferred modes aligned with evidence.

---

## Document metadata

**Files changed:** `docs/project/product/lkw/PRODUCT_4_DAILY_USE_ARCHITECTURE_DECISIONS.md` (this file only at commit).

**Concurrent work:** preserve unrelated repository changes; stage only this file at commit.

# LKW PRODUCT-4 — Daily-Use Architecture Decisions (ARCH-1)

## 1. User-facing decision summary

**Status:** READY_FOR_REVIEW (R1 corrections applied)
**Task:** LKW-PRODUCT-4-ARCH-1 — FREEZE DAILY-USE ASK SCOPE AND DESTRUCTIVE CONFIRMATION CONTRACTS
**Mode:** bounded architecture decision / evidence only — no production code changed

Two open decisions from the accepted PRODUCT-4A audit are **frozen**:

| Decision | Outcome |
|---|---|
| **Source-scoped Ask** | Canonical scope identity is **`knowledge_item_id`** from the unified inventory. Reusable application contract: **`KnowledgeAskScopeV1`** (`knowledge_item_ids`). **Indexed** scoped Ask requires **pre-ranking retrieval filter** (not yet wired on the LKW Ask path) **plus** defensive post-retrieval filter. **Live** scoped Ask is **deferred** — not enforceable on the current companion Ask path. **PRODUCT-4C blocked** until retrieval-level scoped filtering is available. |
| **Workspace delete** | **Confirmation required** before any destructive delete on the daily conversational path. Existing HMAC confirmation is **knowledge-specific** but **safely extractable** into a reusable destructive-action confirmation boundary. **`workspace_revision` + CAS required** for production-grade state binding (no `int(updated_at.timestamp())`). Until wired, **`workspace.delete` must fail closed** (no unconfirmed execution). |

**PRODUCT-4 sequencing (if both resolve):**

1. **PRODUCT-4B** — daily inventory, freshness, attention, lifecycle, confirmation UX (including workspace delete), workspace daily UX
2. **PRODUCT-4C** — source-scoped Ask using `KnowledgeAskScopeV1` + **`KnowledgeRetrievalScopeV1`** (indexed mode; **after** retrieval-level scoped filter is wired)
3. **PRODUCT-4** final acceptance

**Git context:** branch `development`; required ancestor `e3f45e7cbe10a14bac640bab4c7e096b482d983d` verified at task start (R1).

---

## 2. Source-scoped Ask decision

### Status: **RESOLVED_WITH_REQUIRED_CAPABILITY** (indexed enforcement)

Canonical Ask scope contract is frozen. **Indexed semantic correctness** requires retrieval-level scoped filtering before `top_k` truncation; application post-filter alone is a **security invariant**, not sufficient scoped retrieval. Live scoped Ask is explicitly deferred.

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

Validated scope yields **allowed indexed `source_id` set** (for indexed items) used for **retrieval scope** and **defensive evidence enforcement**.

### Indexed enforcement path (frozen)

**End-to-end invariant:**

```text
knowledge_item_id
  → inventory authorization (KnowledgeAskScopeV1)
  → validated retrieval scope (allowed source_id / binding identifiers)
  → PRE-RANKING retrieval filter (before ranking / top_k truncation)
  → defensive post-filter (security invariant)
  → assembly
  → scoped citations
```

**Why post-filter alone is insufficient:** whole-workspace search → `top_k` → post-filter can drop all hits when relevant scoped sources never entered the candidate set. Result: false `INSUFFICIENT_EVIDENCE` despite in-scope evidence existing.

**Retrieval boundary today:** `WorkspaceAskService._retrieve_verified_evidence()` → task `local.workspace.search` with metadata `tenant_id`, `workspace_id`, `collection_id=workspace_id`, `query`, `top_k` — **whole-workspace collection**, no source/binding filter in task metadata.

**Search agent** (`agents/local_search/steps/search_job.py`):

- `_LKW_SEARCH_METADATA_KEYS`: `query`, `collection_id`, `top_k`, `tenant_id`, `user_id`, `workspace_id` — **no `source_id` / `knowledge_item_id` filter keys**.
- `run_search_job()` builds `rag.retrieve` `tool_input` with `query`, `top_k`, `tenant_id`, `user_id`, `workspace_id` only — **no source scope**.

**`rag.retrieve` contract today** (`intergrax/tools/providers/rag/contracts.py` `RagRetrieveInput`): `query`, `top_k`, `session_id`, `user_id`, `tenant_id`, `workspace_id`, `score_threshold` — **no `source_id` / `source_ids` field**. `_build_metadata_scope()` adds only `session_id` and `user_id` to `MetadataFilter` conditions.

**Integration-layer filter primitive (exists, not wired on LKW Ask path):** `perform_rag_retrieve()` passes `metadata_filter=MetadataFilter(conditions=where)` into `RetrievalRequest` before ranking/`top_k`. `MetadataFilter` (`intergrax/rag/vectorstore/contracts/native_vectorstore.py`) is provider-neutral equality filtering; Qdrant/provider translation stays inside vectorstore integration. **No application-to-retrieval contract currently passes validated `source_id` scope into this filter on the companion Ask path.**

**Evidence mapping:** `map_search_hits()` in `search_evidence.py` verifies hits against repository document refs and drops hits when `source_id` disagrees with `ref.source_id`. It accepts no scope parameter today — this is **defensive post-filter only**.

**Frozen minimum indexed enforcement (correct semantics):**

1. Validate `KnowledgeAskScopeV1` via inventory (fail closed).
2. Resolve indexed items → validated `allowed_source_ids` (and binding identifiers as needed for retrieval-safe scope).
3. Pass validated retrieval scope into search **before** ranking/`top_k` (see `KnowledgeRetrievalScopeV1` below).
4. Run scoped retrieval (not whole-workspace-only candidate selection when scope is narrowed).
5. **Defensive post-filter** hits to `source_id ∈ allowed_source_ids` (extend `map_search_hits` with optional `allowed_source_ids` **or** filter in `WorkspaceAskService` after mapping).
6. Assembly + `project_ask_citations()` only on scope-validated evidence.

**Do not** classify application post-filter alone as complete indexed scoped Ask enforcement.

### Live / hybrid enforcement path

| Path | Current state | PRODUCT-4C |
|---|---|---|
| **Indexed v1 Ask** (`WorkspaceAskService`) | Companion-wired; indexed search only | **Blocked until retrieval-level scoped filter wired**; post-filter remains required defense-in-depth |
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
- Search-layer scope without inventory validation
- Post-filter-only scoped retrieval (defense-in-depth only; not semantic correctness)
- Multi-workspace scope

### Required reusable retrieval contract (genuine gap)

No existing application contract passes validated source scope to search before `top_k`. Freeze minimum contract:

```text
KnowledgeRetrievalScopeV1
  allowed_source_ids: tuple[str, ...]   # validated indexed source_id set; non-empty when scope narrowed
```

Propagation (integration boundary — no Qdrant-specific application model):

1. `WorkspaceAskService` / task metadata carries validated `KnowledgeRetrievalScopeV1` (or equivalent allowed-source fields).
2. `local.workspace.search` / `run_search_job()` forwards scope into `rag.retrieve`.
3. `RagRetrieveInput` (or task-metadata adapter) accepts validated source scope.
4. `perform_rag_retrieve()` maps scope into `MetadataFilter` / `RetrievalRequest.metadata_filter` **before** ranking/`top_k`.
5. Provider-specific filter translation remains inside `intergrax/rag/vectorstore/` integration.

**PRODUCT-4C may proceed only after steps 1–4 are implemented.** Defensive post-filter remains mandatory.

### Exact implementation boundary for PRODUCT-4C

**Prerequisite (blocks PRODUCT-4C):** wire `KnowledgeRetrievalScopeV1` through Ask → `local.workspace.search` → `rag.retrieve` → `MetadataFilter` before `top_k`.

**In scope:**

1. Add `KnowledgeAskScopeV1` (frozen shape above).
2. Extend `WorkspaceAskPlannedAction` with optional `ask_scope: KnowledgeAskScopeV1 | None` (whole-workspace when absent).
3. Extend `WorkspaceAskService.ask(..., scope: KnowledgeAskScopeV1 | None = None)`.
4. Add scope validation (inventory-backed) before `_retrieve_verified_evidence`.
5. Pass `KnowledgeRetrievalScopeV1` into retrieval (pre-ranking filter).
6. Extend evidence mapping with `allowed_source_ids` **defensive** post-filter for indexed items.
7. Wire executor to pass validated scope; verify citation scoping in tests.

**Out of scope (PRODUCT-4C):**

- Live/hybrid scoped retrieval
- Slack-only scope encoding
- Planner as authorization
- Qdrant-specific application contracts (provider translation stays in integration layer)

**Bounded follow-up (within PRODUCT-4C after prerequisite):** `allowed_source_ids` on `map_search_hits` vs inline filter — implementation choice; both pre-ranking filter and post-filter required.

---

## 3. Workspace-delete decision

### Status: **RESOLVED_WITH_REQUIRED_CAPABILITY**

Confirmation orchestration pattern is frozen. **Workspace state-version authority** requires a new monotonic revision (preferred) or exact full-precision persisted identity — **not** `int(updated_at.timestamp())`.

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

### Workspace state-version investigation

| Option | Finding |
|---|---|
| **A. Monotonic revision / CAS on `Workspace`** | **Not found.** `Workspace` (`models.py`) has `updated_at` only — no `revision`, `expected_revision`, ETag, or CAS. `ManagedWorkspaceRepository.put_workspace()` is unconditional `_put`; `delete_workspace()` is unconditional delete. Production `put_workspace` is only called from `ManagedWorkspaceService.create_workspace()` — **`updated_at` is set once at creation and not bumped on later mutations.** |
| **B. Durable state fingerprint** | **Not found** for workspace shell. `KnowledgeConfigurationHead.committed_revision` has CAS (`replace_knowledge_configuration_head_if_match`) but is knowledge-configuration state, not workspace lifecycle identity. Knowledge inventory items expose `revision` (reused by detach confirmation) — wrong target semantics for workspace delete. |
| **C. Minimum safe contract** | Add **`workspace_revision: int`** (monotonic, bumped on any workspace-shell mutation, verified with CAS on `put_workspace`) as **`expected_state_version` for `workspace.delete`**. **Forbidden:** `int(workspace.updated_at.timestamp())` or any second-resolution truncation. |

**Persistence precision (if `updated_at` used temporarily):** `ManagedWorkspaceRepository._put()` serializes via `model_dump(mode="json")` — datetime round-trips at JSON precision. Comparison must use **exact canonical persisted `updated_at` string equality** (full precision), never epoch-second `int`. **Limitation:** because `updated_at` is currently write-once at create, it does **not** detect intervening workspace-shell mutations — **not production-grade** without `workspace_revision`.

**Rejected:** `int(updated_at.timestamp())` — truncates sub-second precision; two state changes within one second can share the same version and fail to invalidate confirmation.

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
  expected_state_version: int # knowledge item.revision; workspace.workspace_revision (NOT epoch seconds)
  expires_at: datetime

HmacDestructiveActionConfirmationCodec
  # Same HMAC/TTL mechanics as HmacKnowledgeAdministrationConfirmationCodec
  # Payload schema version 2 (or shared codec with version discriminator)
```

**Migration approach (preserve knowledge-detach behavior):**

1. Introduce generic codec + confirmation model.
2. `HmacKnowledgeAdministrationConfirmationCodec` becomes adapter **or** thin wrapper mapping `KnowledgeAdministrationConfirmationV1` ↔ `DestructiveActionConfirmationV1` with `action_kind="knowledge.detach"` (or operation-specific kinds) and `target_id=knowledge_item_id`.
3. `KnowledgeAdministrationService._verify_confirmation` continues equivalent binding checks via adapter.
4. New `WorkspaceDeletionService` (or executor helper) uses same codec + secret for `action_kind="workspace.delete"`, `target_id=workspace_id`, `expected_state_version=workspace.workspace_revision` (after revision field + CAS added).
5. Shared secret configuration (same as `knowledge_admin_confirmation_secret` on HTTP routes) — **one secret, one crypto primitive**.

**PRODUCT-4B prerequisite:** add `workspace_revision` to `Workspace` + monotonic bump/CAS on `put_workspace` before workspace-delete confirmation can be production-grade.

### Binding fields (minimum)

| Field | Workspace delete |
|---|---|
| Tenant | `tenant_id` |
| Target | `target_id` = `workspace_id` to delete |
| Context workspace | `workspace_id` (must match target) |
| Operation kind | `action_kind = workspace.delete` |
| State version | `expected_state_version` = `workspace.workspace_revision` (monotonic int; exact equality) |
| Expiry | `expires_at` (short TTL, same class as knowledge detach) |
| Cryptographic authenticity | HMAC signature over canonical payload |

### Expiry / replay semantics

| Scenario | Behavior |
|---|---|
| Expired token | Reject — `confirmation_expired` (fail closed) |
| Tampered token | Reject — `confirmation_invalid` |
| Wrong tenant / workspace / target / action_kind | Reject — `confirmation_invalid` |
| Workspace changed after issue (`workspace_revision` differs) | Reject — `confirmation_stale` |
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
3. Add `workspace_revision` to `Workspace` + monotonic bump/CAS on `put_workspace`.
4. Workspace delete confirmation issue/verify around `delete_workspace` using `workspace_revision`.
5. Conversational two-step: `workspace.delete` planned action returns `confirmation_required` + token; separate confirm path executes delete.
6. Block immediate delete in `ConversationInteractionExecutor` until confirmation supplied.

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
| Post-filter-only indexed scoped Ask | Semantically incorrect; misses in-scope evidence excluded by whole-workspace `top_k` |
| `int(updated_at.timestamp())` for workspace delete state version | Truncates sub-second precision; two changes in one second share version |
| Second-resolution workspace state identity | Not production-grade destructive confirmation |

---

## 5. Required reusable capability additions

Only items **genuinely missing** today:

| Addition | Reason |
|---|---|
| **`KnowledgeAskScopeV1`** | No existing multi-item Ask scope type |
| **Inventory-backed Ask scope validation** | Authorization cannot live in planner or raw retrieval |
| **`KnowledgeRetrievalScopeV1`** + wiring through `local.workspace.search` → `rag.retrieve` → `MetadataFilter` | Pre-ranking scoped retrieval not available on LKW Ask path today |
| **`allowed_source_ids` defensive post-filter** on evidence path (`map_search_hits` or equivalent) | Security invariant after retrieval; not substitute for pre-ranking filter |
| **`workspace_revision` + CAS on `put_workspace`** | No monotonic workspace lifecycle revision; `updated_at` is write-once at create |
| **`DestructiveActionConfirmationV1` + generic HMAC codec** | Knowledge codec is not directly reusable for workspace delete |
| **Workspace delete confirmation orchestration** | Two-step flow missing on conversational path |

No Slack-only stores or duplicate crypto required. Retrieval uses existing `MetadataFilter` integration primitive once application contract is wired.

---

## 6. Final architecture status

| Decision | Status | Notes |
|---|---|---|
| Source-scoped Ask — canonical contract | **RESOLVED** | `knowledge_item_id` + `KnowledgeAskScopeV1` |
| Source-scoped Ask — indexed enforcement | **RESOLVED_WITH_REQUIRED_CAPABILITY** | Pre-ranking `KnowledgeRetrievalScopeV1` wiring required; defensive post-filter retained |
| Source-scoped Ask — live/hybrid | **RESOLVED (deferred)** | Explicitly out of PRODUCT-4C |
| Workspace delete confirmation — crypto/orchestration | **RESOLVED** | Generic extraction + two-step conversational flow |
| Workspace delete — state version authority | **RESOLVED_WITH_REQUIRED_CAPABILITY** | `workspace_revision` + CAS required; forbid `int(timestamp())` |
| Workspace delete — direct codec reuse | **RESOLVED (rejected)** | Generalization required; pattern reuse approved |

**PRODUCT-4C blocked** until retrieval-level scoped filtering is wired. **PRODUCT-4B** may proceed but workspace-delete confirmation needs `workspace_revision` for production-grade stale binding.

---

## 7. Consequence for PRODUCT-4 implementation sequence

1. **PRODUCT-4B** can proceed: inventory/lifecycle/freshness/attention UX; wire `KnowledgeAdministrationService` confirmation for detach; implement **generic destructive confirmation** and **workspace delete two-step**; add **`workspace_revision` + CAS** on workspace persistence; fix workspace list/active presentation; suppress repetitive READY guidance. **Do not** ship unconfirmed `workspace.delete` on conversational executor.
2. **PRODUCT-4C** proceeds **only after** `KnowledgeRetrievalScopeV1` is wired through Ask → search → `rag.retrieve` pre-ranking filter (may overlap late 4B once prerequisite lands): implement `KnowledgeAskScopeV1` + indexed scoped Ask; defensive post-filter; citation scoping tests.
3. **Live scoped Ask** and **hybrid daily UX** remain PRODUCT-6 / later — not blockers for PRODUCT-4B.
4. **PRODUCT-4 final acceptance** after 4B + 4C + audit criteria from PRODUCT-4A.

---

## 8. Evidence / read inventory

### Accepted PRODUCT-4 audit (1)

1. `docs/project/product/lkw/PRODUCT_4_SLACK_DAILY_USE_GAP_AUDIT.md`

### Production / source files (8 — budget respected)

1. `applications/local_workspace_application/workspaces/ask_service.py` (section: `_retrieve_verified_evidence`)
2. `agents/local_search/steps/search_job.py` (sections: `_LKW_SEARCH_METADATA_KEYS`, `run_search_job` → `rag.retrieve` `tool_input`)
3. `intergrax/tools/providers/rag/contracts.py` (section: `RagRetrieveInput`)
4. `intergrax/tools/providers/rag/service.py` (sections: `perform_rag_retrieve`, `_build_metadata_scope`)
5. `intergrax/rag/vectorstore/contracts/native_vectorstore.py` (section: `MetadataFilter`)
6. `applications/local_workspace_application/workspaces/models.py` (section: `Workspace`)
7. `applications/local_workspace_application/workspaces/repository.py` (sections: `put_workspace`, `_put` / `model_dump(mode="json")`)
8. `applications/local_workspace_application/workspaces/service.py` (sections: `create_workspace`, `delete_workspace`)

**Targeted discovery (grep / symbols only, not full read):**

- `applications/local_workspace_application/workspaces/knowledge_administration_service.py` — `expected_revision`, detach confirmation pattern
- `applications/local_workspace_application/workspaces/search_evidence.py` — `map_search_hits` defensive `source_id` check

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

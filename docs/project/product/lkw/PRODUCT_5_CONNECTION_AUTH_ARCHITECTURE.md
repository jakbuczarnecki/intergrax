# LKW PRODUCT-5B-ARCH-1 — Connection & Authorization Architecture

**Task:** LKW-PRODUCT-5B-ARCH-1  
**Status:** READY_FOR_REVIEW  
**Mode:** architecture decision only — no production code changed  
**Authority:** [`PRODUCT_5_REAL_VENDOR_GAP_AUDIT.md`](PRODUCT_5_REAL_VENDOR_GAP_AUDIT.md) (PRODUCT-5A, accepted), [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md)  
**Required ancestor:** `2159a54906d45dcc7f845608e0f499d7ee2edf3b`

---

## 1. Executive decision

**Freeze:** User-facing tenant connection creation, authentication, reconnect, revoke, and credential binding are owned by a **reusable application-layer orchestration service** in `local_workspace_application`, backed by existing Vendor Knowledge durable primitives (`TenantConnection`, `SecretsStore`, `TenantConnectionRehydrator`). Provider-specific OAuth and credential mechanics live **behind a provider auth adapter protocol** in `intergrax/integrations/providers/*`. Slack, web, mobile, Teams, CLI, and HTTP clients are **thin clients** of the same contract — no Slack-owned OAuth or connection business state.

**Google Workspace PRODUCT-5 discovery MVP:** **Option B** — user-supplied Google resource URL (or equivalent stable identifier) resolved and validated by the backend into a provider-neutral `opaque_candidate_ref`; **not** full Drive/shared-drive browsing in PRODUCT-5.

**PRODUCT-5B may proceed** without inventing architecture. Cross-track items CT-2 (Drive enumeration) and CT-3 (M365 principal derivation) are deferred to provider integration + PRODUCT-5E/5F; CT-1 is **resolved here** as application orchestration + provider adapters.

---

## 2. Responsibility / layer diagram

```text
┌─────────────────────────────────────────────────────────────────────────────┐
│  Thin clients (Slack conversation, HTTP, future web/mobile/Teams/CLI)       │
│  ConversationInteractionApplicationService · FastAPI routes                 │
└───────────────────────────────────┬─────────────────────────────────────────┘
                                    │ typed connection actions / HTTP
                                    ▼
┌─────────────────────────────────────────────────────────────────────────────┐
│  APPLICATION ORCHESTRATION (local_workspace_application)                    │
│  TenantConnectionProductOrchestrationService  [NEW in PRODUCT-5B]           │
│    · list providers · begin/complete auth · bind manual · reconnect · revoke│
│    · secrets bind · TenantConnection create/update · trigger rehydrate      │
│  TenantConnectionAuthorizationTransactionStore  [NEW, durable]              │
│  WorkspaceConnectionAttachmentService          [EXISTING — unchanged owner] │
│  WorkspaceRemoteResourceDiscoveryService       [EXISTING — unchanged owner] │
│  KnowledgePluginConfigurationService           [EXISTING — safe list/read]  │
└───────────────┬─────────────────────────────┬───────────────────────────────┘
                │                             │
                ▼                             ▼
┌───────────────────────────────┐  ┌──────────────────────────────────────────┐
│  RUNTIME VENDOR KNOWLEDGE      │  │  PROVIDER INTEGRATION (per vendor)        │
│  (intergrax/runtime/           │  │  TenantConnectionAuthProvider adapter     │
│   vendor_knowledge)            │  │    [NEW protocol + per-provider impl]     │
│  TenantConnection              │  │  TenantConnectionIntegrationFactory       │
│  TenantConnectionService       │  │    [EXISTING — runtime construction]      │
│  TenantConnectionRepository    │  └──────────────────────────────────────────┘
│  TenantConnectionRehydrator    │
│  TenantConnectionPort          │
│  SafeTenantConnectionV1        │
└───────────────┬────────────────┘
                │
                ▼
┌───────────────────────────────┐
│  SecretsStore (credential_ref) │
│  Document store (connections + auth transactions) │
└───────────────────────────────┘
```

### Layering decisions (concerns 1–12)

| # | Concern | Owner | Notes |
|---|---------|-------|-------|
| 1 | Product/application orchestration | `TenantConnectionProductOrchestrationService` | Single reusable entry for all clients |
| 2 | Provider-specific OAuth mechanics | Provider `TenantConnectionAuthProvider` adapters | No OAuth in conversation/planner/renderer |
| 3 | Durable connection metadata | `TenantConnection` + `DocumentStoreTenantConnectionRepository` | Secret-free config; immutable identity fields |
| 4 | Secrets/token persistence | `SecretsStore` via `credential_ref` | Tokens never in connection record or product API |
| 5 | OAuth transient state | `TenantConnectionAuthorizationTransaction` (durable store) | Required for multi-worker / restart survival |
| 6 | PKCE / state / CSRF verification | Orchestration service validates; adapter generates | Fail closed on mismatch, expiry, replay |
| 7 | Callback handling | HTTP route → `complete_connection_authorization` | Single product callback surface per deployment |
| 8 | Token refresh | Provider adapter + runtime integration on use; reconnect path for hard expiry | Not exposed to product clients |
| 9 | Reconnect | Orchestration `reconnect_connection` | Re-auth or credential refresh per provider policy |
| 10 | Revoke | Orchestration `revoke_connection` | `REVOKED` status + secret deletion + runtime deregister |
| 11 | Workspace attachment | `WorkspaceConnectionAttachmentService` | Unchanged; consumes existing `connection_ref` |
| 12 | Remote-resource discovery | `WorkspaceRemoteResourceDiscoveryService` | Unchanged; requires attached connection |

---

## 3. Reusable application contract

### Service

**`TenantConnectionProductOrchestrationService`** (application layer, tenant-scoped factory pattern matching `TenantConnectionService`).

All operations are **tenant-bound** (`tenant_id` from request context / execution context). Outputs use **`SafeTenantConnectionV1`** or dedicated safe DTOs — never raw `TenantConnection` with `credential_ref`.

### Operations

| Operation | Purpose | Safe inputs | Safe outputs |
|-----------|---------|-------------|--------------|
| `list_supported_connection_providers` | Product picker / capability discovery | `tenant_id`, optional `audience` | `provider_id`, `integration_kind`, `auth_mode` (`oauth_delegated` \| `manual_credential_binding`), `safe_display_name`, `supported_scopes_summary` |
| `begin_connection_authorization` | Start OAuth or manual binding session | `tenant_id`, `provider_id`, `redirect_uri` (allowlisted), optional `safe_display_name`, optional `connection_ref` (reconnect) | `authorization_transaction_ref` (opaque), `authorization_url` (OAuth only), `expires_at`, `required_user_action` (`redirect` \| `present_manual_instructions`) |
| `complete_connection_authorization` | OAuth callback or manual credential submit | `tenant_id`, `authorization_transaction_ref`, callback `code`/`state` OR manual `credential_payload` (provider-scoped, never logged) | `SafeTenantConnectionV1`, `disposition` (`created` \| `reconnected` \| `already_exists`) |
| `get_connection` | Read one connection | `tenant_id`, `connection_ref` | `SafeTenantConnectionV1` |
| `list_connections` | List tenant connections | `tenant_id`, optional `administrative_status`, `limit` | `SafeTenantConnectionV1[]` |
| `reconnect_connection` | Re-establish credentials for ACTIVE connection | `tenant_id`, `connection_ref`, `redirect_uri` | Same as `begin_connection_authorization` (new transaction) |
| `revoke_connection` | Irreversible disconnect | `tenant_id`, `connection_ref`, idempotency key | `SafeTenantConnectionV1` with `REVOKED` |

**Delegation:** `get_connection` / `list_connections` may delegate to existing `TenantConnectionPort` / `TenantConnectionService.get_safe` / `list_safe`.

### Opaque references

| Reference | Semantics | Exposure |
|-----------|-----------|----------|
| `connection_ref` | Stable tenant-scoped connection identity | Product-safe |
| `authorization_transaction_ref` | One-time OAuth/manual session correlator | Product-safe; not a secret |
| `credential_ref` | Secrets store path | **Internal only** — never in API, Slack, or logs |
| `connected_principal_ref` | Provider account/principal identity for rehydration and discovery | Product-safe opaque string |

### Lifecycle states

Uses existing **`TenantConnectionAdministrativeStatus`**: `ACTIVE`, `DISABLED`, `REVOKED`.

| Phase | State |
|-------|-------|
| Authorization in progress | No `TenantConnection` yet; durable `AuthorizationTransaction` record |
| Connection usable | `ACTIVE` + secret present + runtime rehydrated |
| Administratively paused | `DISABLED` — no runtime registration; reconnect allowed |
| Terminated | `REVOKED` — immutable; secrets deleted; no reconnect |

### Error taxonomy (stable product codes)

| Code | Meaning | Retryable |
|------|---------|-----------|
| `connection_provider_unsupported` | Unknown `provider_id` | No |
| `connection_provider_misconfigured` | Host missing OAuth client config | No |
| `authorization_transaction_not_found` | Unknown or consumed transaction | No |
| `authorization_transaction_expired` | Past `expires_at` | Yes (begin again) |
| `authorization_state_invalid` | CSRF/state/PKCE mismatch | No |
| `authorization_callback_replay` | One-time completion already consumed | No |
| `authorization_redirect_not_allowed` | `redirect_uri` not on deployment allowlist | No |
| `credential_binding_invalid` | Manual payload rejected by provider adapter | Yes |
| `connection_already_exists` | Duplicate principal/provider binding | No |
| `connection_not_found` | Unknown `connection_ref` | No |
| `connection_not_active` | Operation requires ACTIVE | No |
| `connection_revoked` | Terminal state | No |
| `connection_version_conflict` | Optimistic concurrency | Yes |
| `connection_runtime_unavailable` | Rehydration failed after create | Yes |
| `tenant_mismatch` | Cross-tenant binding attempt | No |

### Idempotency

- **`complete_connection_authorization`:** transaction record is **consume-once**; replay returns `authorization_callback_replay` (not a second connection).
- **`revoke_connection`:** idempotent — repeated revoke on `REVOKED` returns same safe view.
- **Connection create:** duplicate `(tenant_id, provider_id, connected_principal_ref)` policy enforced at orchestration layer (exact uniqueness rule per provider adapter metadata).

### Authorization-state correlation

Each `begin_connection_authorization` creates a durable **`TenantConnectionAuthorizationTransaction`** containing:

- `authorization_transaction_ref` (public opaque ID)
- `tenant_id`, `provider_id`
- `state` (CSRF), `pkce_verifier` (when applicable)
- `redirect_uri`, `nonce` (if provider requires)
- `connection_ref` target (create vs reconnect)
- `created_at`, `expires_at` (default **15 minutes**)
- `consumed_at` (null until complete)

Callback **`state`** must match stored value; **`authorization_transaction_ref`** may be embedded in state or looked up by state index. Completion atomically: validate → exchange token (adapter) → write secret → create/update `TenantConnection` → rehydrate → mark consumed.

### Expiry semantics

| Artifact | TTL / rule |
|----------|------------|
| Authorization transaction | 15 min default; fail closed after expiry |
| OAuth access token | Provider-managed; refresh via adapter, not product API |
| `REVOKED` connection | Permanent; new connect creates new `connection_ref` |

### HTTP mapping (PRODUCT-5B)

Thin FastAPI routes under existing workspace serving composition (pattern: `knowledge_connection_attachment_routes.py`):

```text
GET  /v1/local_workspace/knowledge/connection-providers
POST /v1/local_workspace/knowledge/connections/authorize/begin
POST /v1/local_workspace/knowledge/connections/authorize/complete   # callback + manual submit
GET  /v1/local_workspace/knowledge/connections
GET  /v1/local_workspace/knowledge/connections/{connection_ref}
POST /v1/local_workspace/knowledge/connections/{connection_ref}/reconnect
POST /v1/local_workspace/knowledge/connections/{connection_ref}/revoke
```

OAuth browser callback may use a dedicated public path (e.g. `/oauth/callback/{provider_id}`) that forwards to `complete_connection_authorization` after allowlist validation.

### LKW conversation path (PRODUCT-5C — not 5B)

```text
Slack → ConversationInteractionApplicationService
      → typed connection action (e.g. knowledge.connection.begin_authorize)
      → TenantConnectionProductOrchestrationService
      → provider auth adapter
      → TenantConnection + rehydrate
      → (later) WorkspaceConnectionAttachmentService / discovery
```

No provider SDK, OAuth URLs, or token handling in `interaction_plan_compiler`, `interaction_executor` business logic beyond dispatching to orchestration service.

---

## 4. Provider-specific responsibility matrix

| Concern | Slack (knowledge source) | Google Workspace | Microsoft 365 |
|---------|--------------------------|------------------|---------------|
| **PRODUCT-5 auth mode** | `manual_credential_binding` | `oauth_delegated` | `oauth_delegated` |
| **Credential shape in secrets** | JSON `{app_token, bot_token}` | JSON OAuth token bundle (access + refresh + expiry metadata) | JSON OAuth token bundle |
| **Secret-free config** | Optional `api_timeout_seconds` | Empty (existing factory constraint) | `client_id`, optional `graph_base_url`, `timeout_seconds` — **no** `client_secret` in config |
| **connected_principal_ref** | Slack workspace/team ID | Google account `sub` or stable email | Entra user OID / UPN |
| **OAuth/PKCE** | N/A | Required (offline refresh) | Required (delegated scopes) |
| **Token refresh** | Static bot tokens (Slack model) | Adapter refresh on runtime use | Adapter refresh on runtime use |
| **Revoke** | Mark REVOKED + delete secret; optional Slack token revocation API best-effort | Revoke refresh token via Google adapter + REVOKED | Revoke via Graph adapter + REVOKED |
| **Reconnect** | Re-submit bot/app tokens | New OAuth flow | New OAuth flow |
| **Discovery (PRODUCT-5)** | `SlackRemoteResourceDiscoveryStrategy` — READY | URL/ID resolution MVP (5E), not Drive browse | Requires `connected_principal_ref` → mailbox user (5F) |
| **Dual-role invariant** | `slack_companion_*` UI credentials **independent** from `TenantConnection` | — | — |
| **Factory (runtime)** | `SlackTenantConnectionIntegrationFactory` | `GoogleWorkspaceTenantConnectionIntegrationFactory` | `Ms365GraphTenantConnectionIntegrationFactory` |
| **Auth adapter owner** | `intergrax/integrations/providers/conversation_channel/slack/` | `intergrax/integrations/providers/collaboration_suite/google_workspace/` | `intergrax/integrations/providers/collaboration_suite/ms365_graph/` |

### Provider auth adapter protocol (new)

```python
# Conceptual — intergrax/runtime/vendor_knowledge or integrations contract
class TenantConnectionAuthProvider(Protocol):
    provider_id: str
    integration_kind: IntegrationCategory
    auth_mode: Literal["oauth_delegated", "manual_credential_binding"]

    def begin_authorization(...) -> ProviderAuthorizationBeginResult: ...
    def complete_authorization(...) -> ProviderAuthorizationCompleteResult: ...
    def refresh_credentials_if_needed(...) -> None: ...  # internal/runtime
    def revoke_remote_credentials(...) -> None: ...     # best-effort
    def derive_connected_principal_ref(...) -> str: ...
    def build_credential_secret_payload(...) -> str: ...  # JSON for SecretsStore
    def build_secret_free_config(...) -> Mapping[str, JsonValue]: ...
```

Registry parallels `TenantConnectionIntegrationFactoryRegistry`.

---

## 5. Security invariants (frozen)

Production **fail-closed** invariants for PRODUCT-5 connection/auth:

1. **OAuth state** — cryptographically random, bound to `authorization_transaction_ref` and `tenant_id`.
2. **PKCE** — S256 required for Google and M365 delegated OAuth; verifier never leaves server storage.
3. **CSRF / callback correlation** — callback `state` must match durable transaction; `tenant_id` and `provider_id` must match transaction.
4. **Callback expiry** — transactions expire (default 15 min); expired completion rejected.
5. **One-time completion** — `consumed_at` set atomically; replays rejected.
6. **Replay rejection** — same `code` + transaction cannot complete twice.
7. **Tenant binding** — all operations enforce `tenant_id`; cross-tenant transaction lookup impossible.
8. **Provider binding** — transaction `provider_id` must match adapter and created connection.
9. **Redirect allowlist** — `redirect_uri` must match deployment-configured allowlist (per client surface).
10. **Credential secret separation** — secrets only in `SecretsStore`; `TenantConnection` validated secret-free.
11. **No tokens in URLs/logs/product artifacts** — callbacks use `code` only; logs redact credentials; product DTOs use `SafeTenantConnectionV1`.
12. **Token refresh ownership** — provider adapters + runtime integration; never returned to clients.
13. **Revoke behavior** — `REVOKED` is terminal; secrets deleted; runtime deregistered; workspace attachments remain but discovery fails with `connection_not_active` until detach.
14. **Reconnect semantics** — only for `ACTIVE` or `DISABLED`; creates new authorization transaction; updates same `connection_ref` credential in place with version bump.
15. **Account/principal identity binding** — `connected_principal_ref` set at completion from provider identity; duplicate active connection per principal rejected.

---

## 6. Durability model

### Durable

| Artifact | Store | Purpose |
|----------|-------|---------|
| `TenantConnection` | Document store via `DocumentStoreTenantConnectionRepository` | Connection metadata, `credential_ref`, `connected_principal_ref`, status |
| Credential material | `SecretsStore` at `credential_ref` | Tokens / manual credentials |
| `TenantConnectionAuthorizationTransaction` | Document store (new collection) | OAuth/manual session state, PKCE verifier, CSRF state |
| Workspace attachment | `WorkspaceKnowledgeConfigurationV1` | Links workspace → `connection_ref` |

### Ephemeral

| Artifact | Rule |
|----------|------|
| Browser OAuth redirect session | User agent only |
| In-memory PKCE/state | **Forbidden** as sole store — must persist in authorization transaction |

### Restart / multi-worker

- Host bootstrap (`connected_source_host_wiring.py`) continues to rehydrate via `TenantConnectionRehydrator` for configured tenant IDs.
- **Authorization transactions must survive process restart** — document store backed, not process-local dict.
- Multiple workers: transaction consume uses optimistic locking or atomic compare-and-set on `consumed_at`.

### Post-create path

```text
complete_authorization
  → SecretsStore.put_secret(credential_ref)
  → TenantConnectionService.create / update
  → TenantConnectionRehydrator.rehydrate_tenant (target tenant)
  → connection available to WorkspaceConnectionAttachmentService
```

---

## 7. Google Workspace MVP discovery decision

**Decision: Option B — URL / stable identifier resolution MVP for PRODUCT-5 public support.**

| Option | Verdict |
|--------|---------|
| A. Full Drive/shared-drive browsing | **Rejected for PRODUCT-5** — not evidenced in LKW (`RemoteResourceTypeV1` has calendar/docs/sheets only); CT-2 provider ownership; exceeds installation persona for first connect |
| B. User supplies Google resource URL or ID → backend validates → `opaque_candidate_ref` | **Accepted for PRODUCT-5** |

### Rationale

- Backend evidence: `GoogleWorkspaceKnownResourceDiscoveryStrategy` resolves **one configured resource** per type; `GoogleWorkspaceKnownResourceCatalog` is host-in-memory today.
- Supported indexed kinds (evidenced): `google_workspace_calendar`, `google_workspace_docs`, `google_workspace_sheets`.
- `PRODUCT_CONTRACT` installation persona: no internal APIs; pasting a share link or selecting from a **short guided list** is acceptable; full Drive picker is not required for gate 2.
- Drive enumeration remains **CT-2 / PRODUCT-6+** unless product revises scope.

### PRODUCT-5E contract (consumes 5B connection)

New application operation (5E, not 5B): **`register_google_workspace_resource_candidate`**

- Input: `connection_ref`, `resource_url_or_id`, inferred or selected `resource_type`
- Provider adapter validates accessibility with connected credentials
- Output: durable catalog entry + standard `opaque_candidate_ref` for `POST …/knowledge/indexed-sources`
- Attach/index/sync follow existing canonical flow

---

## 8. Cross-track ownership

| ID | Capability | Owner | LKW contract expectation |
|----|------------|-------|--------------------------|
| **CT-1** | User-facing OAuth / delegated auth orchestration | **SHARED_VENDOR_KNOWLEDGE_OWNED** — `TenantConnectionProductOrchestrationService` + provider auth adapters | LKW calls orchestration API only; no local OAuth duplication |
| **CT-2** | Google Drive / shared-drive enumeration | **PROVIDER_INTEGRATION_OWNED** (future) | LKW consumes standard `RemoteResourceDiscovery` when available; PRODUCT-5 uses URL resolution (7) instead |
| **CT-3** | M365 user principal → mailbox/team defaults | **PROVIDER_INTEGRATION_OWNED** — principal derivation in M365 auth adapter at `complete_authorization` | LKW discovery strategies read `connected_principal_ref` from `SafeTenantConnectionV1`; remove host `msgraph_mailbox_user_id` injection in 5F |

### Duplication rule

LKW **must not** implement Graph identity resolution, Google Drive traversal, or OAuth token exchange. It **must** implement thin HTTP/conversation dispatch and workspace/discovery orchestration already present.

---

## 9. PRODUCT-5B exact implementation boundary

**One coherent block:** *Product connection & credential orchestration* — reusable across all clients.

### PRODUCT-5B implements

1. **`TenantConnectionProductOrchestrationService`** with full contract (§3).
2. **`TenantConnectionAuthorizationTransaction`** model + document-store repository.
3. **`TenantConnectionAuthProvider`** protocol + registry.
4. **Provider auth adapters:**
   - Slack: `manual_credential_binding` (guided app/bot token JSON).
   - Google Workspace: `oauth_delegated` (authorize URL, callback, refresh token storage).
   - M365: `oauth_delegated` (replace app-only secret product path; delegated user token).
5. **HTTP routes** (§3 HTTP mapping) mounted in workspace serving composition.
6. **Host wiring** — secrets store, transaction store, orchestration service, callback allowlist settings.
7. **Unit/HTTP tests** for orchestration, transaction expiry, replay rejection, tenant isolation, safe DTO guarantees.
8. **On successful complete:** secrets bind → `TenantConnection` create/update → `TenantConnectionRehydrator` for tenant.

### Reuses (no redesign)

- `TenantConnection`, `TenantConnectionService`, `DocumentStoreTenantConnectionRepository`
- `SecretsStore`
- `TenantConnectionRehydrator`, `TenantConnectionIntegrationFactory` / registry
- `TenantConnectionPort`, `SafeTenantConnectionV1`
- `connected_source_host_wiring.py` bootstrap pattern
- `knowledge_connection_attachment_routes.py` HTTP composition patterns

### Cross-track prerequisites consumed

- **CT-1:** implemented in 5B (resolved).
- **CT-2:** not required for 5B.
- **CT-3:** M365 adapter sets `connected_principal_ref` in 5B; discovery wiring fix in **5F**.

### Likely owned modules (5B)

```text
applications/local_workspace_application/workspaces/tenant_connection_product_orchestration.py   [NEW]
applications/local_workspace_application/workspaces/tenant_connection_authorization_transaction.py [NEW]
applications/local_workspace_application/serving/tenant_connection_routes.py                  [NEW]
intergrax/integrations/providers/*/tenant_connection_auth.py                                  [NEW per provider]
applications/local_workspace_application/workspaces/connected_source_host_wiring.py           [extend wiring]
applications/local_workspace_application/serving/workspace_routes.py                            [mount routes]
applications/local_workspace_application/tests/.../test_tenant_connection_product_*.py        [NEW]
```

### Remains after 5B

| Block | Scope |
|-------|-------|
| **5C** | Slack conversational connect journey — planner/executor actions calling orchestration + attachment + indexed-source create |
| **5D** | Slack knowledge source productization + real Slack qualification |
| **5E** | Google URL resolution, durable known-resource catalog API, guided UX |
| **5F** | M365 discovery without host mailbox/team injection; principal-bound strategies |
| **5G** | Cross-vendor real-provider acceptance harness |

---

## 10. Rejected alternatives

| Alternative | Why rejected |
|-------------|------------|
| Slack workflow/planner owns OAuth state and token exchange | Violates thin-client rule; not reusable by HTTP/web |
| OAuth inside `TenantConnectionService` (runtime) | Runtime is administrative lifecycle only; no product auth flows today |
| Process-local / in-memory OAuth state | Fails multi-worker and restart survival for callback flow |
| New Slack-specific connection microservice | Duplicates application boundaries; contradicts PRODUCT_CONTRACT single API surface |
| Expose `credential_ref`, tokens, or secrets store keys in product API | Security invariant violation |
| Full Google Drive browse in PRODUCT-5 | No LKW backend evidence; CT-2 scope; delays 5B/5E |
| Keep M365 app-only `client_secret` as product auth | SECURITY_GAP in 5A; incompatible with installation persona delegated consent |
| Embed Graph/Google identity logic in `WorkspaceRemoteResourceDiscoveryService` | Violates provider integration ownership (CT-3, CT-2) |
| LKW conversation executor calls provider SDKs directly | Violates layering; blocks non-Slack clients |

---

## Git context

| Field | Value |
|-------|-------|
| Task | LKW-PRODUCT-5B-ARCH-1 |
| Document only | `docs/project/product/lkw/PRODUCT_5_CONNECTION_AUTH_ARCHITECTURE.md` |

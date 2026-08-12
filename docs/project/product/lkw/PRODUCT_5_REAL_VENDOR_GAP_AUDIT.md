# LKW PRODUCT-5A — Real Vendor Experience Gap Audit

**Task:** LKW-PRODUCT-5A  
**Status:** READY_FOR_REVIEW  
**Mode:** discovery / gap audit only — no production code changed  
**Authority:** [`PRODUCT_CONTRACT.md`](PRODUCT_CONTRACT.md) — PRODUCT-5 must verify Slack, Google Workspace, and Microsoft 365 end to end before public support is claimed.

---

## 1. Executive verdict

**Overall status:** READY_FOR_REVIEW  
**Public support claimable today:** **NO** for all three first-public-product vendors.

| Vendor | Overall status | Public support? | Qualification verdict |
|--------|----------------|-----------------|------------------------|
| **Slack (knowledge source)** | BACKEND_READY_PRODUCT_GAP | **NO** | NOT_QUALIFIED |
| **Google Workspace** | BACKEND_READY_PRODUCT_GAP | **NO** | NOT_QUALIFIED |
| **Microsoft 365** | BACKEND_READY_PRODUCT_GAP | **NO** | NOT_QUALIFIED |

**Core finding:** Reusable Vendor Knowledge backend (tenant connections, rehydration, remote discovery, indexed binding, sync, inventory, indexed Ask, lifecycle) is **substantially implemented and HTTP-proven with mocks/fakes**. The missing PRODUCT-5 layer is **user-reachable connection/auth orchestration** and **Slack conversational wiring** for connect → discover → attach → sync — without manual secrets, internal IDs, or host-operator injection.

**Slack dual-role invariant preserved:** Slack-as-UI (PRODUCT-3/4) is separate from Slack-as-knowledge-source (this audit). UI bot credentials (`slack_companion_*`) and knowledge-source credentials (`TenantConnection` + secrets store) are independent.

**Git context (preflight):**

| Field | Value |
|-------|-------|
| Branch | `development` |
| Start/final HEAD | `4bea0fc8248ba677fcec72d3f256b792b34288e1` |
| `origin/development` | `4bea0fc8248ba677fcec72d3f256b792b34288e1` |
| Required ancestor `57c0985b…` | present (`ANCESTOR_OK`) |
| Unrelated dirty work | preserved |

---

## 2. Cross-vendor product journey

### 2.1 What exists (reusable path)

```text
[durable TenantConnection in document store + credential_ref in secrets store]
→ TenantConnectionRehydrator registers runtime integration
→ POST …/workspaces/{id}/connections/{connection_ref}  (workspace attach)
→ GET  …/knowledge/connections/{connection_ref}/remote-resources
→ POST …/knowledge/indexed-sources  (opaque_candidate_ref)
→ POST …/knowledge/indexed-sources/{id}/sync
→ KnowledgeInventoryV1 (HTTP + Slack knowledge.inventory.list)
→ indexed Ask + citations
→ knowledge.operation.execute (sync/disable/enable/detach) in Slack
```

**Canonical components:** `TenantConnection` / `TenantConnectionRehydrator`, `WorkspaceConnectionAttachmentService`, `WorkspaceRemoteResourceDiscoveryService`, `WorkspaceKnowledgeAccessService`, `ManagedWorkspaceConnectedSourceSyncService`, `KnowledgeInspectionService`, `KnowledgeOperationsService`, `KnowledgePluginConfigurationService`.

### 2.2 What is missing for the installation persona

| Gap | Classification |
|-----|----------------|
| No HTTP or Slack action to **create** a tenant connection (OAuth/callback or safe credential binding) | REUSABLE_CAPABILITY_GAP (product orchestration) |
| No Slack actions for workspace connection attach, indexed-source create from remote candidate, or guided vendor connect | WIRING_GAP |
| Google resource discovery is **known-resource catalog**, not Drive browse | UX_GAP + PRODUCTION_QUALIFICATION_GAP |
| M365 discovery requires `msgraph_mailbox_user_id` / `msgraph_teams_channel_team_id` at **host mount**, not from user connection | WIRING_GAP + UX_GAP |
| SharePoint / OneDrive / Drive not in LKW `RemoteResourceTypeV1` discovery | LATER_PRODUCT_BOUNDARY (Drive live exists in runtime only) |
| No real-provider CI proof (all e2e uses fakes/mocks) | BLOCKED_EXTERNAL_PROOF / NOT_QUALIFIED |

### 2.3 Slack PRODUCT-4 reuse

PRODUCT-4 closed wiring is **reusable** for post-connection daily use:

- `knowledge.connections.list`, `knowledge.resources.list`, `knowledge.capabilities.list`
- `knowledge.inventory.list`, `knowledge.operation.execute`
- `workspace.ask` (scoped ask when targets supplied), `citation.inspect`

**Not wired for vendor onboarding:** connection create, OAuth redirect, workspace connection attach, indexed-source create from `opaque_candidate_ref`.

---

## 3. Gap matrices (V1–V12)

Legend: **READY** · **WIRING_GAP** · **UX_GAP** · **REUSABLE_CAPABILITY_GAP** · **SECURITY_GAP** · **PRODUCTION_QUALIFICATION_GAP** · **LATER_PRODUCT_BOUNDARY**

### 3.1 Slack knowledge source

| Stage | Capability | Production component | Product reachability | Evidence | Classification | Minimal PRODUCT-5 change | Later dependency |
|-------|------------|----------------------|----------------------|----------|----------------|--------------------------|------------------|
| V1 | Supported vendor messaging | `KnowledgePluginConfigurationService.list_connections` | Slack lists only if connection pre-exists | `interaction_executor.py` | WIRING_GAP | First-run/daily copy + empty-state guidance | — |
| V2 | Real connection / auth | `TenantConnection` + `SlackTenantConnectionIntegrationFactory` (bot/app tokens in secrets) | Requires operator `TenantConnectionService.create` + secrets | `slack/tenant_connection_factory.py` | UX_GAP | Product OAuth or guided token binding HTTP + Slack flow | — |
| V2 | Token persistence / refresh | `DocumentStoreTenantConnectionRepository` + `SecretsStore` | Backend only | `connected_source_host_wiring.py` | READY | Expose reconnect in product surface | — |
| V2 | Revocation | `TenantConnectionAdministrativeStatus.REVOKED` | No product revoke API | `tenant_connections.py` | WIRING_GAP | Connection revoke HTTP + Slack | — |
| V3 | Remote resource discovery | `SlackRemoteResourceDiscoveryStrategy` → `list_accessible_conversations_page` | HTTP when connection attached | `connected_source_discovery_slack.py`, `test_slack_connected_source_http.py` | READY | Wire via existing `knowledge.resources.list` | — |
| V4 | Resource selection / attachment | `POST …/knowledge/indexed-sources` | HTTP only; no Slack planner action | `knowledge_connected_source_routes.py` | WIRING_GAP | Conversational indexed-source create action | — |
| V5 | Access modes | Slack conversation: INDEXED + LIVE (+ DURABLE) | Live not daily Slack UX | `test_slack_connected_source_end_to_end.py` (modes assert) | INDEXED READY; LIVE backend-only | Honest mode labels in inventory | PRODUCT-6 live Ask |
| V6 | Indexed data availability | `ConnectedSourceSyncService` + materializer | HTTP path proven | e2e tests (mock backend) | PRODUCTION_QUALIFICATION_GAP | Real-provider proof | — |
| V7 | Knowledge inventory | `KnowledgeInventoryV1` | Slack `knowledge.inventory.list` | PRODUCT-4 executor | READY (after attach) | — | — |
| V8 | Indexed Ask + citations | `WorkspaceAskService` | Slack `workspace.ask` | `test_slack_connected_source_end_to_end.py` | PRODUCTION_QUALIFICATION_GAP | Real Slack workspace proof | — |
| V9 | Lifecycle | `KnowledgeOperationsService` | Slack `knowledge.operation.execute` | PRODUCT-4 | READY | — | — |
| V10 | Restart / rehydration | `TenantConnectionRehydrator` | Automatic when `tenant_connection_bootstrap_tenant_ids` set | `connected_source_host_wiring.py`, cross-provider e2e | READY | Document install contract | — |
| V11 | Failure / recovery | `ConnectedSourceDiscoveryError`, inventory `last_error_code` | Partial Slack mapping | onboarding presenter + inventory | WIRING_GAP | Vendor-specific user messages | PRODUCT-8 |
| V12 | Disconnect / control | Workspace detach: `DELETE …/connections/{ref}`; item detach via operations | HTTP; partial Slack | `knowledge_connection_attachment_routes.py` | WIRING_GAP | Slack detach connection + distinguish item vs connection | — |

**Slack knowledge source verdict:** **PRODUCT-WIRING-ONLY** at backend; **NOT QUALIFIED** for public support.

### 3.2 Google Workspace

| Stage | Capability | Production component | Product reachability | Evidence | Classification | Minimal PRODUCT-5 change | Later dependency |
|-------|------------|----------------------|----------------------|----------|----------------|--------------------------|------------------|
| V1 | Vendor supported messaging | Same as Slack | No without connection | — | WIRING_GAP | Product copy | — |
| V2 | Auth | `GoogleWorkspaceTenantConnectionIntegrationFactory` (JSON credential in secrets) | Operator-only create | `google_workspace/tenant_connection_factory.py` | UX_GAP | OAuth delegated auth product flow | CROSS_TRACK if platform owns OAuth |
| V3 | Resource discovery | `GoogleWorkspaceKnownResourceDiscoveryStrategy` + in-memory catalog | **No browse**; catalog populated out-of-band | `connected_source_discovery_google_workspace.py`, e2e `catalog.register()` | UX_GAP | Drive picker or URL-to-resource binding API | CROSS_TRACK: Google Drive discovery |
| V4 | Attach / binding | HTTP indexed-source create | HTTP only | `test_google_workspace_lkw_e2e.py` | WIRING_GAP | Slack + registration API | — |
| V5 | Access modes | Calendar, Docs, Sheets — **INDEXED only** in LKW extension | No live Google in LKW catalog | `vendor_knowledge_extension_composition.py` | INDEXED only | Declare boundaries | PRODUCT-6 if live added |
| V6 | Indexed sync → documents | Google materializers + sync | Mock transport e2e | `test_google_workspace_lkw_e2e.py` **PASS** | PRODUCTION_QUALIFICATION_GAP | Real Google proof | BLOCKED_EXTERNAL_PROOF |
| V7 | Inventory | `KnowledgeInventoryV1` | After indexed create | e2e | READY (post-attach) | — | — |
| V8 | Ask + citations | Search + Ask over indexed | e2e | same test | PRODUCTION_QUALIFICATION_GAP | Real provider | — |
| V9 | Lifecycle | `KnowledgeOperationsService` | Slack when in inventory | PRODUCT-4 | READY | — | — |
| V10 | Rehydration | `TenantConnectionRehydrator` + Google factory | e2e restart | `test_google_workspace_lkw_e2e.py` | READY | — | — |
| V11 | Failure recovery | Discovery `candidate_inaccessible` | HTTP errors only | discovery strategy | WIRING_GAP | User-facing recovery | — |
| V12 | Disconnect | Workspace connection detach + item detach | HTTP | attachment routes | WIRING_GAP | Slack wiring | — |

**Supported resource kinds (evidenced):** `google_workspace_calendar`, `google_workspace_docs`, `google_workspace_sheets`. **Not implemented in LKW:** Google Drive folder/file browse.

**Google Workspace verdict:** **BACKEND-ONLY** for indexed exact resources; **NOT QUALIFIED**.

### 3.3 Microsoft 365

| Stage | Capability | Production component | Product reachability | Evidence | Classification | Minimal PRODUCT-5 change | Later dependency |
|-------|------------|----------------------|----------------------|----------|----------------|--------------------------|------------------|
| V1 | Vendor messaging | connections.list | Pre-existing connection only | — | WIRING_GAP | Product copy | — |
| V2 | Auth | `Ms365GraphTenantConnectionIntegrationFactory` (client_id config + client_secret credential) | Operator-only; app-only secret model | `ms365_graph/tenant_connection_factory.py` | UX_GAP + SECURITY_GAP | Delegated OAuth product flow | — |
| V3 | Resource discovery | Mail folder, Teams chat, Teams channel, Calendar strategies | Requires `msgraph_mailbox_user_id` / `msgraph_teams_channel_team_id` on host mount | `connected_source_discovery_msgraph.py`, `workspace_routes.py` | WIRING_GAP | Derive mailbox/team from connection principal | — |
| V4 | Attach | HTTP indexed-source | HTTP only | cross-provider e2e | WIRING_GAP | Slack indexed-source create | — |
| V5 | Access modes | Indexed: mail, teams_chat, teams_channel, calendar; Live capabilities in runtime | Live/hybrid in tests, not Slack daily UX | `test_vendor_knowledge_cross_provider_e2e.py` **PASS** | INDEXED READY; LIVE → PRODUCT-6 | Honest inventory modes | PRODUCT-6 |
| V6 | Indexed availability | MS Graph materializers | Mock graph client | cross-provider e2e | PRODUCTION_QUALIFICATION_GAP | Real Graph proof | BLOCKED_EXTERNAL_PROOF |
| V7 | Inventory | `KnowledgeInventoryV1` | Post-attach | e2e | READY | — | — |
| V8 | Indexed Ask | Workspace search/ask | e2e | cross-provider | PRODUCTION_QUALIFICATION_GAP | Real provider | — |
| V9 | Lifecycle | Knowledge operations | Slack PRODUCT-4 | — | READY | — | — |
| V10 | Rehydration | Rehydrator + Graph factory | e2e | cross-provider | READY | — | — |
| V11 | Failure recovery | VendorKnowledgeError mapping | Partial | live adapters | WIRING_GAP | Product errors | PRODUCT-8 |
| V12 | Disconnect | Detach routes | HTTP | attachment routes | WIRING_GAP | Slack | — |

**Implemented in LKW (evidenced):** Teams Chat, Teams Channel, Mail folders, Calendar. **Not in LKW discovery:** SharePoint, OneDrive (runtime has `MSGRAPH_DRIVE_*` live only — `LATER_PRODUCT_BOUNDARY`). **Outlook** covered via mail folders, not full mailbox UX.

**Microsoft 365 verdict:** **BACKEND-ONLY**; **NOT QUALIFIED**.

---

## 4. Cross-vendor summary

| Vendor | Connection | Discovery | Attach | Indexed | Live | Lifecycle | Restart | Real-provider proof | Overall |
|--------|------------|-----------|--------|---------|------|-----------|---------|---------------------|---------|
| Slack | UX_GAP | READY (HTTP) | WIRING_GAP | READY (mock) | Backend only | READY | READY | NOT_QUALIFIED | BACKEND_READY_PRODUCT_GAP |
| Google | UX_GAP | UX_GAP (known-resource) | WIRING_GAP | READY (mock) | N/A in LKW | READY | READY | NOT_QUALIFIED | BACKEND_READY_PRODUCT_GAP |
| M365 | UX_GAP | WIRING_GAP (host injection) | WIRING_GAP | READY (mock) | Backend only | READY | READY | NOT_QUALIFIED | BACKEND_READY_PRODUCT_GAP |

---

## 5. Security matrix

| # | Control | Slack | Google | M365 |
|---|---------|-------|--------|------|
| 1 | Credential storage | Secrets store + `credential_ref` — READY | Same — READY | Same — READY |
| 2 | Secret exposure in config | `TenantConnection` secret-free validation — READY | READY | READY |
| 3 | OAuth / PKCE / CSRF | **No product OAuth** — SECURITY_GAP | **No OAuth** — SECURITY_GAP | App secret in credential blob — SECURITY_GAP |
| 4 | Tenant/account binding | `tenant_id` on connection — READY | READY | READY |
| 5 | Token refresh | Slack tokens static in factory — partial | Access token static in tests — UX_GAP | Client secret static — UX_GAP |
| 6 | Revocation | Model supports REVOKED; no product API — WIRING_GAP | Same | Same |
| 7 | Reconnect | Rehydration only — WIRING_GAP | Same | Same |
| 8 | Least privilege | Slack bot scopes external to LKW — operator responsibility | Same | Graph permissions external |
| 9 | Remote resource authz | Discovery revalidates candidate scope — READY | Known-resource only | Graph API errors mapped |
| 10 | Workspace binding | Connection attachment + opaque refs signed — READY | READY | READY |
| 11 | Cross-tenant isolation | `tenant_id` enforced on all ports — READY | READY | READY |
| 12 | Log/error leakage | Fail-closed credential redaction in factories — READY | READY | READY |
| 13 | Restart recovery | Rehydrator — READY | READY | READY |
| 14 | Revoked remote resource | Manifest tombstones / `candidate_inaccessible` — READY | READY | READY |

---

## 6. Production qualification

| Evidence type | Slack | Google | M365 |
|---------------|-------|--------|------|
| Unit / adapter tests | `test_slack_connected_source.py`, VK sync units | `test_google_workspace_lkw_e2e.py`, GW unit factories | `test_msgraph_*_knowledge_sync.py`, cross-provider e2e |
| HTTP contract tests | `test_slack_connected_source_http.py` | e2e HTTP | cross-provider e2e |
| Restart / rehydration | `TenantConnectionRehydrator` in host wiring | **PASS** `test_google_workspace_rehydrated_*` | **PASS** cross-provider rehydration |
| Auth expiry / revocation | Not product-tested | Not product-tested | Not product-tested |
| Lifecycle tests | PRODUCT-4 knowledge operations | e2e sync/disable paths | e2e |
| End-to-end LKW product | `test_slack_connected_source_end_to_end.py` — **FAIL** this run (env); mock-based | **PASS** | **PASS** `test_cross_provider_three_mode_e2e` |
| Real-provider tests | **None in repo** | **None** | **None** |

**Commands run (this audit):**

```text
uv run pytest applications/local_workspace_application/tests/workspaces/test_slack_connected_source_end_to_end.py -q  → FAILED (136s, assertion truncated)
uv run pytest …/test_google_workspace_lkw_e2e.py::test_google_workspace_rehydrated_calendar_docs_sheets_search_ask -q  → PASSED
uv run pytest …/test_vendor_knowledge_cross_provider_e2e.py::test_cross_provider_three_mode_e2e -q  → PASSED
```

**External proof:** unavailable in this environment — all qualification is mock/fake-provider based.

**Verdicts:** Slack **NOT_QUALIFIED**; Google **NOT_QUALIFIED**; M365 **NOT_QUALIFIED** (backend **BACKEND_READY_PRODUCT_GAP**).

---

## 7. PRODUCT-5 vs PRODUCT-6 boundary

| Owned by PRODUCT-5 | Owned by PRODUCT-6+ |
|--------------------|---------------------|
| Real vendor connect/auth (product orchestration) | Broad live/hybrid Ask UX |
| Resource discovery + honest mode declaration | Query-policy tuning |
| Selection / attachment / first indexed knowledge | Daily cross-mode behavior |
| Lifecycle, reconnect, disconnect (product surfaces) | Rich live freshness UX |
| Real-provider qualification gates | — |

**Architectural dependency:** Vendors expose LIVE capabilities in runtime today. PRODUCT-5 can ship **indexed-only** public support without PRODUCT-6, provided inventory honestly shows `indexed` and does not imply live freshness.

---

## 8. Cross-track dependencies

| ID | Missing capability | Expected reusable contract | Why LKW should not implement locally |
|----|-------------------|---------------------------|--------------------------------------|
| CT-1 | User-facing OAuth / delegated auth orchestration (authorize URL, callback, PKCE, token exchange, credential_ref write) | Platform- or application-neutral `TenantConnection` create/update HTTP with secrets binding | Duplicates provider auth safety work; belongs in Vendor Knowledge platform or shared integration layer |
| CT-2 | Google Drive / shared-drive remote enumeration | `RemoteResourceDiscovery` for `google_workspace_drive` (or scoped file URL resolution) | Google integration provider ownership |
| CT-3 | M365 user principal resolution from OAuth token | Connection `connected_principal_ref` → mailbox user / default team | Graph integration ownership; LKW should consume, not embed Graph identity logic |

---

## 9. Implementation roadmap (derived blocks)

Dependency order — **no microtasks**:

1. **PRODUCT-5B — Product connection & credential orchestration**  
   Tenant-connection create/list/revoke HTTP (or thin orchestration service), OAuth/callback flows per vendor family, secrets binding, reconnect. Prerequisite for all vendors. Blocks: UX_GAP on V2 for all.

2. **PRODUCT-5C — Conversational vendor connect journey (Slack thin client)**  
   Reuse PRODUCT-4 discovery actions; add planner/executor for: workspace connection attach, remote-resource pick, indexed-source create, setup-snapshot progress. No vendor-specific Slack logic — call existing HTTP/application services.

3. **PRODUCT-5D — Slack knowledge source productization**  
   Real remote conversation discovery (already backend-ready), qualification with real Slack workspace, dual-role credential separation documented in install path.

4. **PRODUCT-5E — Google Workspace productization**  
   Depends on CT-2 or accepted “paste resource URL / pick from guided list” MVP; replace in-memory-only `GoogleWorkspaceKnownResourceCatalog` registration with durable product API; OAuth via 5B.

5. **PRODUCT-5F — Microsoft 365 productization**  
   Remove host-injected `msgraph_mailbox_user_id` / `msgraph_teams_channel_team_id`; bind from connection principal; limit PRODUCT-5 scope to evidenced workloads (Teams chat/channel, mail folder, calendar).

6. **PRODUCT-5G — Cross-vendor real-provider acceptance**  
   Gated real-provider test harness (credentials in CI secret store or manual gate); blocks public support claim per `PRODUCT_CONTRACT.md` gate 11–12.

---

## 10. Architecture ownership questions (unresolved)

1. **OAuth ownership:** Application-layer LKW routes vs platform `TenantConnection` admin API — needs explicit ADR before 5B.
2. **Google MVP discovery:** Full Drive browse vs URL/deep-link binding for PRODUCT-5 — product decision, not code discovery.

---

## 11. Read inventory

**Product docs (4):** `PRODUCT_CONTRACT.md`, `USER_JOURNEY.md`, `PRODUCT_4_SLACK_DAILY_USE_GAP_AUDIT.md` (partial), this file.

**Production files (≤28):** `connected_source_discovery*.py` (4), `connected_source_host_wiring.py`, `connected_source_wiring.py`, `connected_source_models.py`, `vendor_knowledge_extension_composition.py`, `knowledge_connection_attachment_service.py`, `knowledge_connection_attachment_routes.py`, `knowledge_connected_source_routes.py`, `knowledge_plugin_configuration_service.py`, `tenant_connections.py`, `tenant_connection_capabilities.py`, `google_workspace/tenant_connection_factory.py`, `ms365_graph/tenant_connection_factory.py`, `slack/tenant_connection_factory.py`, `interaction_executor.py` (partial), `interaction_models.py` (partial), `interaction_plan_compiler.py` (partial), `workspace_routes.py` (partial), `host/settings.py` (partial), `slack_companion/companion.py` (partial).

**Test files (≤18):** `test_slack_connected_source_http.py`, `test_slack_connected_source_end_to_end.py`, `test_google_workspace_lkw_e2e.py`, `test_vendor_knowledge_cross_provider_e2e.py`, `test_knowledge_connection_attachment_routes.py`, `test_vendor_knowledge_scoped_source_seam_qualification.py`, `test_slack_connected_source.py`, `test_lkw_product_acceptance.py`, `test_acme_reference_external_provider_proof.py` (partial).

---

## 12. Files changed

- `docs/project/product/lkw/PRODUCT_5_REAL_VENDOR_GAP_AUDIT.md` (this file only)

**Commit:** none (audit document created; commit not requested).

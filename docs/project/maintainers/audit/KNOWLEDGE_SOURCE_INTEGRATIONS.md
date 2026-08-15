# Vendor Knowledge Facade — Reuse Audit

**Status:** `DONE / READY_FOR_REVIEW`  
**Task:** `VENDOR-KNOWLEDGE-FACADE-AUDIT-1`  
**Branch:** `development`  
**Architecture:** [`../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md)
**Plan:** [`../plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md)

---

## 1. Verdict

The platform already contains the mechanisms required to resolve existing provider/category integrations, execute durable logical tasks, persist tenant-scoped state through `DocumentStore`, and resolve secrets through the `secrets_store` category.

The facade must reuse those mechanisms.

The proven missing platform surface is limited to:

1. vendor-neutral knowledge models and facade/adapter ports;
2. a small source-adapter registry, separate from the integration catalog;
3. tenant-scoped source/connection bindings above `IntegrationProfile`;
4. facade-specific normalized errors;
5. later checkpoint, lease and remote-item repositories over the existing `DocumentStore` contract;
6. missing low-level read/change methods in selected existing vendor integrations.

Do not create:

- a `knowledge_source` integration category;
- duplicate public vendor integrations;
- a second integration resolver/catalog;
- a new queue framework;
- a new generic persistence framework;
- vendor-specific RAG pipelines.

---

## 2. Reuse matrix

| Concern | Existing mechanism | Decision | Proven gap |
|---|---|---|---|
| Integration selection | `IntegrationProfile` with category slots and `IntegrationBinding` | `REUSE` | Profile selects one application-level integration per category; it is not a multi-connection source-binding store. |
| Integration construction | `resolve_from_profile()` / `resolve()` and integration catalog | `REUSE` | Facade needs a resolver port around this behavior so tests can inject pre-built integrations without duplicating catalog logic. |
| Provider/category validation | `IntegrationCategory`, catalog entry categories and mismatch errors | `REUSE` | None for basic integration resolution. |
| Source-adapter selection | No matching registry exists | `MINIMAL NEW` | Add a private/platform service registry keyed by provider, integration category and source kind. It must not construct integrations. |
| Secrets | `IntegrationCategory.SECRETS_STORE`, `IntegrationProfile.secrets_store`, `SecretsStore` protocol | `REUSE` | Add only opaque `connection_ref` / `credential_ref` semantics and an injected connection resolver. No secret values in bindings. |
| Durable task delivery | `DocumentStoreTaskQueue` | `REUSE` | Queue has idempotent enqueue and restart recovery, but no delayed retry/backoff policy. |
| Worker dispatch | `DocumentStoreTaskWorker` + `TaskExecutionRegistry` | `REUSE` | Register sync handlers later; do not create a second worker framework. |
| Tenant isolation in queue | Tenant partition and tenant-bearing task request | `REUSE` | Facade contracts and repositories must preserve tenant checks end to end. |
| Durable facade state | Provider-neutral `DocumentStore` contract and repository pattern | `REUSE FOUNDATION` | Add facade-owned repository ports/implementation for bindings, checkpoints and remote-item state. Do not import the LKW repository. |
| Existing LKW persistence | `ManagedWorkspaceRepository` | `PATTERN ONLY` | It is application-owned and must not become a dependency of the platform facade. |
| Existing LKW sync runtime | `ManagedWorkspaceSyncRuntime` | `INTEGRATION REFERENCE` | It proves queue/worker reuse, but vendor sync orchestration must remain platform-neutral until the convergence task. |
| Integration errors | `IntegrationError`, `IntegrationDependencyError`, configuration/category/unknown errors | `REUSE BASE` | Add normalized facade codes for authentication, authorization, rate limit, cursor invalidation, item not found/revoked and unsupported capability. |
| Retry classification | Worker marks failures; integration dependency error exists | `EXTEND LATER` | Explicit retryable/non-retryable classification and backoff scheduling are not currently provided by the queue. |
| Lease/concurrency | Queue atomically claims tasks | `PARTIAL REUSE` | A source-level lease is still required to prevent two distinct tasks synchronizing the same binding concurrently. |
| Checkpoint atomicity | No facade checkpoint repository | `MINIMAL NEW LATER` | Persist proposed cursor only after durable page completion. |
| ACL retrieval/enforcement | No shared vendor-knowledge ACL envelope | `MINIMAL NEW` | Contract is needed; retrieval-time enforcement remains a later LKW convergence task. |

---

## 3. Integration resolution decision

### Reuse

The facade must resolve existing integrations through an injected resolver whose production implementation delegates to `IntegrationProfile.resolve()` / `resolve_from_profile()`.

Required semantic port:

```python
class VendorIntegrationResolver(Protocol):
    def resolve(
        self,
        *,
        integration_kind: IntegrationCategory,
        provider_id: str,
        connection_ref: str | None,
    ) -> object:
        ...
```

The exact signature may be adjusted in the contract task, but these rules are fixed:

- the integration catalog remains authoritative;
- a pre-built integration instance remains valid for tests and explicit wiring;
- the facade does not call provider factories directly;
- the facade validates that the resolved provider/category matches the binding;
- no provider-specific conditional chain is allowed in the facade.

### Gap

`IntegrationProfile` is an application composition profile. It contains one slot per category and therefore cannot by itself represent:

- two Jira connections for one tenant;
- multiple Microsoft 365 tenants;
- several Confluence sites;
- multiple independently configured scopes using one integration;
- connection expiry/revocation state;
- source-specific safe display metadata.

Therefore `IntegrationProfile` is reused for integration resolution, while a separate tenant-scoped facade binding references the selected integration and connection.

---

## 4. Source adapter registry decision

A new registry is justified, but it is not another integration registry.

Purpose:

```text
resolved existing integration
+
(provider_id, integration_kind, source_kind)
→ source adapter
```

Recommended key:

```text
(provider_id, integration_kind, source_kind)
```

Examples:

```text
(jira, issue_tracker, issues)
(confluence, wiki_knowledge, pages)
(ms365_graph, collaboration_suite, drive)
(ms365_graph, collaboration_suite, mail)
(databricks, relational_store, unity_catalog)
```

Rules:

- adapters are registered explicitly during composition;
- adapters receive an existing integration instance;
- adapters do not resolve secrets;
- adapters do not instantiate vendor clients;
- adapters do not persist checkpoints;
- duplicate keys fail at registration;
- missing keys fail deterministically;
- registry has no global import-time side effects.

`TaskExecutionRegistry` is not suitable for this purpose because it maps logical task names to worker handlers, not provider/source capabilities.

---

## 5. Secrets and connection boundary

Reuse the existing `SecretsStore` protocol and `secrets_store` profile category.

The facade binding may persist only opaque references:

```text
binding_id
tenant_id
provider_id
integration_kind
source_kind
connection_ref
credential_ref
safe_display_name
scope
status
configuration_version
```

Forbidden durable fields:

```text
access_token
refresh_token
api_key
client_secret
password
authorization_header
signed_download_url
```

A later connection resolver may obtain secret material from `SecretsStore` and inject/configure the existing vendor integration. The facade models and logs never receive the raw secret value.

---

## 6. Queue and worker decision

Reuse:

```text
DocumentStoreTaskQueue
DocumentStoreTaskWorker
TaskExecutionRegistry
TaskRequest / TaskHandle
```

They already provide:

- durable task records through provider-neutral `DocumentStore`;
- tenant partitioning;
- idempotent enqueue by idempotency key;
- atomic pending-to-running claim;
- task attempt count;
- interrupted-running recovery on host restart;
- logical task handler registration.

Do not create a vendor-facade-specific queue.

Proven gaps for a later sync task:

- delayed retry and backoff scheduling;
- retryable/non-retryable error classification;
- source-level lease independent of task claim;
- page-level checkpoint commit;
- partial-page recovery policy.

These gaps belong to `VENDOR-KNOWLEDGE-SYNC-*`, not the first contract task.

---

## 7. Durable state decision

Use the provider-neutral `DocumentStore` contract and the same tenant-partitioned repository style already proven by LKW and the durable task queue.

Do not reuse `ManagedWorkspaceRepository` directly because it is owned by `local_workspace_application`.

Later facade repositories should be isolated behind ports such as:

```text
KnowledgeSourceBindingRepository
KnowledgeSyncCheckpointRepository
KnowledgeRemoteItemStateRepository
KnowledgeSourceLeaseRepository
```

Production implementations may share one `DocumentStore`, but partitions and model ownership remain separate from LKW.

No persistence repository is added in the first contract task.

---

## 8. Error boundary decision

Reuse existing integration errors as causes, but expose a stable facade error envelope.

Minimum normalized codes for the contract:

```text
configuration_error
integration_not_found
integration_category_mismatch
adapter_not_found
unsupported_capability
authentication_failed
authorization_denied
rate_limited
dependency_unavailable
remote_item_not_found
remote_item_revoked
invalid_cursor
invalid_scope
tenant_mismatch
invalid_provider_response
```

The envelope must be safe for logs and application status. It must not expose tokens, authorization headers, credential paths containing secrets or unsafe response bodies.

---

## 9. Current vendor capability gaps

### Jira

Current public integration supports issue lookup/search and mutations. It does not yet expose the complete low-level behavior required for durable synchronization:

- explicit page cursor/offset result;
- updated-since or changelog traversal;
- deletion/revocation semantics;
- attachment inventory/content;
- permission visibility;
- stable revision envelope.

These methods must later be added to the existing `JiraIssueTrackerIntegration` or a private read component behind it, not to a second Jira integration.

### Confluence

Current public integration supports page lookup and search. Missing for durable synchronization:

- bounded page inventory with continuation;
- version/revision envelope;
- attachments;
- archived/deleted state;
- permissions/visibility;
- robust rich-text normalization input.

Extend the existing `ConfluenceWikiKnowledgeIntegration` path later.

### Microsoft Graph

Current collaboration integration exposes mail, calendar and directory operations. It does not yet expose Drive/SharePoint knowledge operations:

- site/drive/folder scope inspection;
- drive item delta;
- binary content download;
- item permissions;
- Graph tombstones;
- ETag/cTag revision state.

Add those low-level methods to the existing Microsoft Graph integration/private client boundary before implementing the drive adapter.

### Databricks

Current integration is relational SQL-oriented (`connect`, `execute`, `fetch_all`, `close`). It does not expose Unity Catalog, workspace tree, volumes, notebooks, lineage or change-feed knowledge behavior.

Databricks remains deferred until its source kind and domain scope are selected precisely.

---

## 10. Placement decision

The first code slice should live in a platform-neutral package and must not import LKW.

Recommended location after contract review:

```text
intergrax/runtime/vendor_knowledge/
    models.py
    contracts.py
    errors.py
```

The audit deliberately does not create:

```text
registry.py
facade.py
repositories.py
sync.py
providers/
```

Those files belong to later scoped tasks.

The package name is a platform service namespace, not an integration category.

---

## 11. Exact next task

### `VENDOR-KNOWLEDGE-FACADE-CONTRACT-1`

**Type:** small code task  
**Status:** `DONE`

Allowed initial scope:

```text
intergrax/runtime/vendor_knowledge/__init__.py
intergrax/runtime/vendor_knowledge/models.py
intergrax/runtime/vendor_knowledge/contracts.py
intergrax/runtime/vendor_knowledge/errors.py
tests/unit/runtime/vendor_knowledge/
```

Semantic deliverables:

- strict tenant-aware source binding reference;
- scope and capability models;
- stable remote item identity separated from revision;
- page/cursor result;
- binary, rich-text and structured content envelope;
- provenance/deep link;
- ACL envelope;
- normalized facade error;
- `VendorKnowledgeAdapter` protocol;
- `VendorKnowledgeFacade` protocol;
- `VendorIntegrationResolver` protocol.

Explicitly out of scope:

- implementation of a registry or facade;
- changes to `IntegrationCategory`;
- changes to existing provider integrations;
- secrets resolution implementation;
- persistence;
- queue/worker registration;
- retries/checkpoints/leases;
- Jira, Confluence, Graph or Databricks adapters;
- LKW and RAG changes.

---

## 12. Current marker

```text
DONE:    VENDOR-KNOWLEDGE-FACADE-ARCH-1
DONE:    VENDOR-KNOWLEDGE-FACADE-PLAN-1
DONE:    VENDOR-KNOWLEDGE-FACADE-AUDIT-1
DONE:    VENDOR-KNOWLEDGE-FACADE-CONTRACT-1
NEXT:    VENDOR-KNOWLEDGE-FACADE-CORE-1
LATER:   VENDOR-KNOWLEDGE-CONNECTION-1
LATER:   VENDOR-KNOWLEDGE-SYNC-1A
DEFERRED: LKW-CONNECTED-SOURCE-1
```

## Three-mode reuse audit

**Date:** 2026-07-31
**Task:** `VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1`
**Type:** docs-only architecture audit append

### Verdict

The existing provider/category integration architecture is suitable as a shared foundation for indexed RAG, durable materialization and live access.

The currently implemented Vendor Knowledge Facade and Sync Coordinator directly cover the durable synchronization/materialization path.

They do not yet constitute the complete live-query capability layer.

### Findings

1. Existing integrations are correctly reusable.
2. Jira and Confluence already expose separate operational search/get methods and knowledge inventory/content methods.
3. Microsoft Graph exact reads and typed references are reusable for live exact access.
4. Graph delta and reconciliation methods are not substitutes for live search.
5. `VendorKnowledgeAdapter.read_page()` must not be used as an artificial live search engine.
6. Live access requires typed capability contracts and a validated executor.
7. Database materialization requires an injected durable sink, not LKW ownership.
8. RAG requires the LKW/application ingestion path after durable normalized delivery.
9. Live results remain ephemeral unless explicitly promoted.
10. The largest cross-mode security gap remains explicit authorization and ACL policy.

### Gap classification

Future audits and tasks must use these classifications instead of calling every missing behavior an "adapter gap":

```text
PROVIDER PRIMITIVE GAP
DURABLE MATERIALIZATION GAP
RAG BRIDGE GAP
LIVE CAPABILITY GAP
AUTHORIZATION / ACL GAP
```

### Current marker (post-append)

```text
DONE:    VENDOR-KNOWLEDGE-THREE-MODE-REUSE-ARCH-1
IN_PROGRESS: MSGRAPH-KNOWLEDGE-ADAPTERS-1
  NEXT:  MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT
DEFERRED: LKW-CONNECTED-SOURCE-1
PLANNED: VENDOR-LIVE-CAPABILITY-CONTRACT-1
PLANNED: VENDOR-LIVE-CAPABILITY-EXECUTOR-1
  (planned after adapter-family completion; not part of the immediate adapter task)
```

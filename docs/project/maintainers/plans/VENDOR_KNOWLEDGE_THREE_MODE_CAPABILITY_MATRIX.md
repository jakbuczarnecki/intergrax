# Vendor Knowledge Three-Mode Capability Matrix

**Task:** `VENDOR-KNOWLEDGE-PROVIDER-COVERAGE-1`
**Status:** `ACCEPTED / CLOSED`
**Branch:** `development`  
**Input:** [`VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md`](VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md)  
**Roadmap:** [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md)
**Live architecture:** [`VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md`](../../architecture/VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md)

## 1. Executive summary

This document is the canonical capability matrix at the exact
`provider_family × source_kind × mode` boundary. VK-6 audits and registers
existing provider/source-kind coverage; it does not inflate or activate a
missing mode.

The accepted adapter-family input proves provider adapters and durable
reconciliation foundations, not automatically application materialization,
indexed ingestion or live access. Current repository evidence proves:

- generic Indexed / RAG infrastructure and an application indexed path;
- generic Durable / Storage / Materialization sink, checkpoint, reconciliation
  and recovery foundations;
- generic Live / Realtime binding, capability, executor, limits, normalized
  evidence and receipt foundations;
- the accepted `ARCH-1` shared live delta is implemented and closed through
  `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1`;
- accepted provider-neutral Indexed bridge proofs for Slack
  `slack_conversation`, Microsoft Graph `teams_chat`, `mail`, `teams_channel`
  and `calendar` provider-owned materialization bridges, with application
  Search/Ask closeout for the Graph family;
- provider/source-kind live handlers and registrations are implemented for the
  five Microsoft Graph bounded list capabilities: Drive, Mail, Teams Channel,
  Teams Chat and Calendar, plus the three Slack conversation live capabilities;
- a canonical default plugin composition covers every implemented adapter:
  Microsoft Graph, Slack, Google Workspace, Jira and Confluence;
- Atlan and Power BI have no Vendor Knowledge implementation, while Databricks
  has only a relational integration and no selected Vendor Knowledge source kind;

Microsoft Graph Drive, Mail, Teams Channel, Teams Chat and Calendar live access
and Slack `slack_conversation` live access are `ACCEPTED / CLOSED`; all other
live provider/source-kind rows remain conservative and unimplemented.

The canonical mode names for this roadmap are **Indexed / RAG**, **Durable /
Storage / Materialization** and **Live / Realtime**. The machine-readable
column identifiers `indexed`, `durable` and `live` below are retained for
compatibility with the matrix format and mean those canonical modes.

## 2. Mode definitions

### Indexed / RAG (`INDEXED`)

Provider content is durably or controllably materialized, converted to
canonical content, written to an index, refreshed, removed when supported,
retained with provenance, and retrieved through an application query path.
Generic local-file ingestion is not Vendor Knowledge provider proof.

### Durable / Storage / Materialization (`DURABLE_MATERIALIZATION`)

Provider content is synchronized into durable application-controlled storage
through a source binding, initial sync, continuation or reconciliation,
deterministic and idempotent delivery, checkpointing, recovery and explicit
application ownership. The generic DocumentStore sink proves the runtime
contract; it does not by itself prove a production application sink.

### Live / Realtime (`LIVE`)

An authorized application invokes a typed, provider/source-kind-registered,
validated read-only capability. Execution is bounded by item, byte and time
limits, returns normalized ephemeral evidence and an execution receipt, and
does not rely on prior durable ingestion.

## 3. Evidence and status rules

Evidence precedence is:

1. accepted end-to-end mode proof;
2. production application wiring plus focused tests;
3. provider-specific runtime wiring;
4. generic platform implementation;
5. accepted architecture;
6. roadmap wording.

VK-6 mode statuses use only the required vocabulary:

| Status | Matrix meaning |
|---|---|
| `ACCEPTED` | Exact mode has production wiring, focused proof and accepted closeout. |
| `FOUNDATION_ONLY` | Generic contracts/runtime or low-level reads exist without exact mode wiring. |
| `IMPLEMENTED_UNREVIEWED` | Implementation exists but this audit did not establish complete acceptance proof. |
| `UNSUPPORTED` | The capability intentionally does not exist for the source kind. |
| `UNPROVEN` | The repository does not establish the capability or its semantics. |

The historical detail table may retain its older evidence vocabulary, but it is
not authoritative for current VK-6 status.

The audit inventory is authoritative for source kinds. Databricks has no
source-kind mode row because no Vendor Knowledge source kind has been selected;
its implementation state is explicitly recorded below.

## 4. Platform foundation summary

| Mode | Generic foundation | Boundary |
|---|---|---|
| Indexed | `YES` — LKW document ingestion, canonical document indexing, vector retrieval and query/Ask paths | Local-file/LKW capability is not transferred to Vendor Knowledge providers. |
| Durable materialization | `YES` — `DocumentStore`, idempotent sink, remote-item state, checkpoints, leases, reconciliation, queue/worker and recovery | Provider adapter/sync proof is not application-owned materialization proof. |
| Live | `YES` — current typed binding/catalog contracts, exact handler registry, validated executor, bounded limits, normalized evidence, receipts and retention | `FOUNDATION-1` provides the accepted shared boundary; the Graph family closeout verifies five provider-specific list handlers on the same integration and executor. |

## 5. VK-6 authoritative provider/source-kind coverage

`ACCEPTED` means technically accepted for the documented scope only; it is not
a commercial, GA, SLA, exhaustive-history or complete-ACL claim.

LKW readiness status: Slack `slack_conversation`, Microsoft Graph `teams_chat`,
`mail`, `teams_channel` and `calendar` are **`LKW_READY`**. The Graph
application proof uses one durable `ms365_graph / collaboration_suite`
TenantConnection, `credential_ref`, restart rehydration, discovery/binding,
sync, provider materialization, generic KnowledgeDocument/index/Search/Ask
handoff and the existing Live handlers. Microsoft Graph `drive` remains
**`FOUNDATION_ONLY`** for Indexed: the adapter exposes file content only as
`BINARY`, while folders/packages have no content representation, so no
truthful text materialization is claimed; the exact blocker is
`REQUIRES_GENERIC_BINARY_CONTENT_EXTRACTION_CAPABILITY`. These readiness labels
are not commercial GA/SLA claims. Graph item/per-user ACL semantics remain
unproven/limited; content/history is bounded and unsupported attachments,
hosted content or richer traversal remain excluded.

### Microsoft Graph family closeout

Microsoft Graph has one canonical Vendor Knowledge identity:
`provider_id=ms365_graph` with integration category
`COLLABORATION_SUITE`. Drive, Mail, Teams Channel, Teams Chat and Calendar
reuse one durable `TenantConnection` and its `credential_ref` through
`Ms365GraphTenantConnectionIntegrationFactory`,
`TenantConnectionRehydrator` and `KnowledgeConnectionRegistry`. Mailbox,
team and calendar scope values are discovery/source configuration, not separate
provider credential lifecycles.

The generic LKW path contains no Graph credential handling, direct Graph
execution, Graph-specific Search or Ask implementation, or provider business
switch. Graph discovery, materialization and Live registration remain at the
approved provider composition boundary. `LKW_READY` means the documented
bounded technical path is accepted; it does not mean commercial GA, complete
ACL coverage or exhaustive Microsoft Graph feature coverage.

| provider | source kind | adapter | DURABLE | INDEXED | LIVE | plugin declaration | runtime registration | proof level | deletion semantics | ACL status | known limitations | evidence |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Microsoft Graph | `drive` | ACCEPTED | FOUNDATION_ONLY | FOUNDATION_ONLY | ACCEPTED | DURABLE, LIVE | adapter + Graph live bundle | FOCUSED_TESTS | authoritative delta/removal | provider, tenant, source binding; item/per-user UNPROVEN | files are exposed as `BINARY` and folders/packages have no content representation; no truthful text materializer or binary extraction is claimed; exact blocker `REQUIRES_GENERIC_BINARY_CONTENT_EXTRACTION_CAPABILITY` | `test_msgraph_drive_knowledge_adapter.py`, `test_msgraph_drive_knowledge_sync.py`, `test_msgraph_drive_live.py` |
| Microsoft Graph | `mail` | ACCEPTED | ACCEPTED | ACCEPTED | ACCEPTED | DURABLE, INDEXED, LIVE | adapter + provider materializer + generic bridge + Graph live bundle | LKW_READY / FOCUSED_TESTS | folder-scoped delta/removal | provider, tenant, source binding; item/per-user UNPROVEN | folder-scoped sync; bounded body/metadata projection; thread messages, attachment inventory and attachment bytes excluded; per-user ACL unproven; commercial GA/SLA not implied | `test_msgraph_mail_knowledge_adapter.py`, `test_msgraph_mail_knowledge_sync.py`, `test_connected_source_materializer.py`, `test_vendor_knowledge_cross_provider_e2e.py`, `test_msgraph_mail_live.py` |
| Microsoft Graph | `teams_channel` | ACCEPTED | ACCEPTED | ACCEPTED | ACCEPTED | DURABLE, INDEXED, LIVE | adapter + provider materializer + generic bridge + Graph live bundle | LKW_READY / FOCUSED_TESTS | explicit `deletedDateTime` | provider, tenant, source binding; item/per-user UNPROVEN | one bounded root post; replies, all history, attachments and hosted content excluded; per-user ACL unproven; commercial GA/SLA not implied | `test_msgraph_teams_channel_knowledge_adapter.py`, `test_msgraph_teams_channel_knowledge_sync.py`, `test_connected_source_materializer.py`, `test_msgraph_teams_channel_live.py`, `test_vendor_knowledge_cross_provider_e2e.py` |
| Microsoft Graph | `teams_chat` | ACCEPTED | ACCEPTED | ACCEPTED | ACCEPTED | DURABLE, INDEXED, LIVE | adapter + materializer + Graph live bundle | LKW_READY / FOCUSED_TESTS | authoritative `DELETED` | tenant/source binding; item/per-user UNPROVEN / limited | bounded fixed-window content/history; mentions, reactions, attachments and hosted content excluded where unsupported; commercial GA/SLA not implied | `test_msgraph_teams_chat_knowledge_adapter.py`, `test_msgraph_teams_chat_knowledge_sync.py`, `test_msgraph_teams_chat_live.py`, `test_connected_source_materializer.py`, `test_vendor_knowledge_cross_provider_e2e.py` |
| Microsoft Graph | `calendar` | ACCEPTED | ACCEPTED | ACCEPTED | ACCEPTED | DURABLE, INDEXED, LIVE | adapter + provider materializer + generic bridge + Graph live bundle | LKW_READY / FOCUSED_TESTS | primary delta + window/removal semantics | provider, tenant, source binding; item/per-user UNPROVEN | bounded event projection; primary delta and non-primary snapshot/window semantics remain source-owned; full traversal and richer recurrence/content coverage are not claimed; per-user ACL unproven; commercial GA/SLA not implied | `test_msgraph_calendar_knowledge_adapter.py`, `test_msgraph_calendar_knowledge_sync.py`, `test_connected_source_materializer.py`, `test_msgraph_calendar_live.py`, `test_vendor_knowledge_cross_provider_e2e.py` |
| Slack | `slack_conversation` | ACCEPTED | ACCEPTED | ACCEPTED | ACCEPTED | DURABLE, INDEXED, LIVE | adapter + materializer + Slack live bundles | LKW_READY / FOCUSED_TESTS | DELETION_UNPROVEN; no tombstones | tenant/source binding; complete per-user ACL UNPROVEN | bounded configured channels; native search, exhaustive history and files deferred | `test_slack_connected_source_end_to_end.py`, `test_slack_conversation_knowledge_sync.py`, `test_slack_live.py`, `test_tenant_connection_factory.py` |
| Google Workspace | `drive` | ACCEPTED | FOUNDATION_ONLY | FOUNDATION_ONLY | UNSUPPORTED | DURABLE | adapter | FOCUSED_TESTS | snapshot absence not authoritative | provider, tenant, source binding; item ACL UNPROVEN | no application sink, index bridge or live handler | `test_google_workspace_drive_knowledge_adapter.py`, `test_google_workspace_drive_knowledge_sync.py` |
| Google Workspace | `docs` | ACCEPTED | FOUNDATION_ONLY | FOUNDATION_ONLY | UNSUPPORTED | DURABLE | adapter | FOCUSED_TESTS | snapshot absence not authoritative | provider, tenant, source binding; item ACL UNPROVEN | known-document scope; no discovery, index bridge or live handler | `test_google_workspace_docs_knowledge_adapter.py`, `test_google_workspace_docs_knowledge_sync.py` |
| Google Workspace | `sheets` | ACCEPTED | FOUNDATION_ONLY | FOUNDATION_ONLY | UNSUPPORTED | DURABLE | adapter | FOCUSED_TESTS | snapshot absence not authoritative | provider, tenant, source binding; item ACL UNPROVEN | known-spreadsheet scope; no index bridge or live handler | `test_google_workspace_sheets_knowledge_adapter.py`, `test_google_workspace_sheets_knowledge_sync.py` |
| Google Workspace | `calendar` | ACCEPTED | FOUNDATION_ONLY | FOUNDATION_ONLY | UNSUPPORTED | DURABLE | adapter | FOCUSED_TESTS | snapshot absence not authoritative | provider, tenant, source binding; item ACL UNPROVEN | adapter/reconciliation only; no index bridge or live handler | `test_google_workspace_calendar_knowledge_adapter.py`, `test_google_workspace_calendar_knowledge_sync.py` |
| Jira | `issues` | ACCEPTED | FOUNDATION_ONLY | FOUNDATION_ONLY | UNSUPPORTED | DURABLE | adapter | FOCUSED_TESTS | no authoritative tombstones | provider, tenant, source binding; item ACL UNPROVEN | project reconciliation only; no incremental feed, index bridge or live handler | `test_jira_knowledge_adapter.py`, `test_jira_knowledge_sync.py` |
| Confluence | `pages` | ACCEPTED | FOUNDATION_ONLY | FOUNDATION_ONLY | UNSUPPORTED | DURABLE | adapter | FOCUSED_TESTS | no authoritative tombstones | provider, tenant, source binding; item ACL UNPROVEN | space reconciliation only; no incremental feed, index bridge or live handler | `test_confluence_knowledge_adapter.py`, `test_confluence_knowledge_sync.py` |

### Explicitly unimplemented providers

| provider | implementation state | supported modes | classification |
|---|---|---|---|
| Atlan | no Vendor Knowledge adapter, plugin or runtime | none | UNSUPPORTED / NOT IMPLEMENTED |
| Power BI | no Vendor Knowledge adapter, plugin or runtime | none | UNSUPPORTED / NOT IMPLEMENTED |
| Databricks | relational-store integration exists, but no Vendor Knowledge adapter and no source kind is selected | none | UNSUPPORTED / NOT IMPLEMENTED |

`VK1-GAP-07` is **CLOSED**: provider coverage is now deterministic, truthful and
aligned with the accepted platform boundaries. Frontend neutrality remains
VK-7 and complete cross-provider product E2E remains VK-8.

The following legacy detail table is retained for traceability; the VK-6 table
above is authoritative for current status.

## 5A. Historical detailed provider/source-kind matrix

| provider_family | integration_identity | source_kind | adapter_status | indexed_status | indexed_platform_foundation | indexed_provider_wiring | indexed_application_wiring | indexed_refresh | indexed_removal | indexed_provenance | indexed_proof | indexed_gap | durable_status | durable_platform_foundation | durable_provider_wiring | durable_application_sink | durable_checkpoint | durable_recovery | durable_proof | durable_gap | live_status | live_platform_foundation | live_provider_wiring | live_executor | live_limits | live_evidence | live_receipt | live_application_wiring | live_proof | live_gap | commercially_supported_modes | primary_evidence | next_action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Microsoft Graph | `ms365_graph / collaboration_suite` | `drive` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No truthful Graph-to-index content exists: files are `BINARY`, folders/packages have no content representation; preserve accepted delta/full-reconciliation and ACL limits. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Accepted adapter/reconciliation exists, but no production application-owned sink was found. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Drive list/query only; text search and binary extraction are unsupported, child read is not applicable. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE`; focused live tests | Preserve the accepted Drive list proof and exact operation matrix; add indexing only with a truthful textual representation. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `mail` | `ACCEPTED` | `IMPLEMENTED_UNREVIEWED` | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Provider-owned Mail materialization feeds the generic sink/index bridge and the rehydrated application connection→discovery→binding→sync→Search/Ask→Live proof passes. | `PARTIAL` | YES | YES | YES | YES | YES | FOCUSED_TESTS | Folder-scoped reconciliation and application sink wiring are proven; thread messages, attachment inventory and bytes remain excluded. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Mail live remains bounded to the existing folder-scoped handler; item/per-user ACL and GA/SLA claims remain out of scope. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL`; `test_connected_source_materializer.py`; `test_vendor_knowledge_cross_provider_e2e.py` | Preserve folder/removal semantics and Mail limitations. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `teams_channel` | `ACCEPTED` | `IMPLEMENTED_UNREVIEWED` | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Provider-owned bounded root-post materialization and generic sink/index wiring plus the rehydrated application Search/Ask proof pass without widening the root-post contract. | `PARTIAL` | YES | YES | YES | YES | YES | FOCUSED_TESTS | Adapter reconciliation remains source-owned and explicit `deletedDateTime` semantics are preserved. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Live remains limited to one root post; replies, all history, attachments and hosted content remain deferred. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`; `test_connected_source_materializer.py`; `test_vendor_knowledge_cross_provider_e2e.py`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL` | Preserve the root-post contract and explicit deletion semantics. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `teams_chat` | `ACCEPTED` | `ACCEPTED` | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | VK-4 generic bridge and canonical Graph Teams Chat materialization/index proof are accepted; the rehydrated Graph Teams Chat LKW E2E closes the application Search/Ask proof gap. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Adapter fixed-window snapshot/reconciliation exists, but application-owned durable sink coverage remains representative. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Fixed-window metadata-only list; item/per-user ACL semantics remain unproven/limited; content/history is bounded; mentions, reactions, attachments and hosted content remain excluded where unsupported; commercial GA/SLA is not implied. | NONE | `applications/local_workspace_application/tests/workspaces/test_connected_source_materializer.py`; `tests/unit/runtime/vendor_knowledge/test_durable_lifecycle_cross_provider.py`; `applications/local_workspace_application/tests/workspaces/test_vendor_knowledge_cross_provider_e2e.py`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT` | Preserve fixed-window and explicit-deletion semantics plus the LKW_READY limitations. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `calendar` | `ACCEPTED` | `IMPLEMENTED_UNREVIEWED` | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Provider-owned accepted event/metadata materialization and generic sink/index wiring plus the rehydrated application Search/Ask proof pass while preserving separate primary delta and non-primary snapshot/window lifecycles. | `PARTIAL` | YES | YES | YES | YES | YES | FOCUSED_TESTS | Primary delta and non-primary snapshot/window reconciliation remain separate, with source-owned removal semantics. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | One binding-selected bounded event projection; full traversal and richer event content remain deferred. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`; `test_connected_source_materializer.py`; `test_vendor_knowledge_cross_provider_e2e.py`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR` | Preserve the separate reconciliation lifecycles and bounded event projection. |
| Slack | `slack / conversation_channel` | `slack_conversation` | `ACCEPTED` | `ACCEPTED` | YES | YES | YES | YES | NO | YES | FOCUSED_TESTS | Indexed HTTP→sync→materialization→index→Search/Ask proof passes with replay/recovery; Slack removal propagation remains unproved. | `ACCEPTED` | YES | YES | YES | YES | YES | FOCUSED_TESTS | Removal propagation and complete per-user ACL enforcement remain unproved. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Bounded recent configured-channel activity, bounded multi-channel Ask, fetched-evidence filtering, one-page thread summary and exact message read are supported; native search, exhaustive history, arbitrary accessible-channel discovery, permissions and file/attachment reads remain deferred. | NONE | `test_slack_connected_source_end_to_end.py`; `test_connected_source_continuation.py`; `test_connected_source_recovery.py`; `test_slack_live.py`; `test_slack_ask_orchestration.py` | Keep ACL, native search, exhaustive history, removal propagation and attachment/file-body claims conservative. |
| Google Workspace | `google_workspace / collaboration_suite` | `drive` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Adapter and sync tests exist, but no Google application binding, index bridge or indexed proof exists. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Provider adapter/reconciliation exists; application-owned materialization is not wired. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Google Drive live handler or application invocation. | NONE | `google_workspace_drive` adapter/sync tests named by accepted audit; roadmap Google status is stale | Correct status only through matrix routing; route application proof separately if product support is approved. |
| Google Workspace | `google_workspace / collaboration_suite` | `docs` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Known-document adapter does not establish indexed discovery, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | One-item reconciliation exists; broad discovery, deletion semantics and application sink are absent. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Google Docs live handler or application invocation. | NONE | `google_workspace_docs` adapter/sync tests named by accepted audit; current integration | Keep exact-known-resource limits explicit; do not copy stale “no adapters” wording as capability evidence. |
| Google Workspace | `google_workspace / collaboration_suite` | `sheets` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Known-spreadsheet adapter does not establish indexed discovery, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | One-item reconciliation exists; broad discovery, deletion semantics and application sink are absent. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Google Sheets live handler or application invocation. | NONE | `google_workspace_sheets` adapter/sync tests named by accepted audit; current integration | Keep exact-known-resource limits explicit; route application proof only if approved. |
| Google Workspace | `google_workspace / collaboration_suite` | `slides` | `PLANNED` | `PLANNED` | YES | NO | NO | NO | NO | NO | UNPROVEN | Shared enum/transport only; no source-specific read, adapter, index bridge or application proof. | `FOUNDATION_ONLY` | YES | NO | NO | YES | YES | UNPROVEN | Generic sync runtime exists, but no Slides provider adapter or application sink exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | Generic live platform exists without a Slides registration or handler. | NONE | Accepted audit inventory; `GoogleWorkspaceSourceKind.SLIDES`; roadmap planned task | Keep planned; do not activate a Slides implementation task from this matrix. |
| Google Workspace | `google_workspace / collaboration_suite` | `calendar` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Adapter and sync-focused tests exist, but no Google application binding, index bridge or indexed proof exists. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Adapter/reconciliation exists, but no production application-owned sink was found. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | NO | UNPROVEN | No Google Calendar live handler or application invocation. | NONE | `GoogleWorkspaceCalendarKnowledgeAdapter` and focused adapter/sync tests; no application sink, index bridge or live registration | Route application-mode proof separately; do not infer indexed/live parity from adapter presence. |
| Google Workspace | `google_workspace / collaboration_suite` | `mail` | `PLANNED` | `PLANNED` | YES | NO | NO | NO | NO | NO | UNPROVEN | Shared enum/transport only; no Gmail read, adapter, index bridge or application proof. | `FOUNDATION_ONLY` | YES | NO | NO | YES | YES | UNPROVEN | Generic sync runtime exists, but no Gmail provider adapter or application sink exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | Generic live platform exists without a Gmail registration or handler. | NONE | Accepted audit inventory; `GoogleWorkspaceSourceKind.MAIL`; roadmap planned task | Keep planned; select a precise Gmail contract before routing implementation. |
| Google Workspace | `google_workspace / collaboration_suite` | `chat` | `PLANNED` | `PLANNED` | YES | NO | NO | NO | NO | NO | UNPROVEN | Shared enum/transport only; no Chat read, adapter, index bridge or application proof. | `FOUNDATION_ONLY` | YES | NO | NO | YES | YES | UNPROVEN | Generic sync runtime exists, but no Chat provider adapter or application sink exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | Generic live platform exists without a Chat registration or handler. | NONE | Accepted audit inventory; `GoogleWorkspaceSourceKind.CHAT`; roadmap planned task | Keep planned; select a precise Chat contract before routing implementation. |
| Jira | `jira / issue_tracker` | `issues` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Adapter exact reads and sync do not prove Jira index binding, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Project reconciliation exists, but no incremental feed, application sink or provider-specific materialization proof exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Jira live registration or handler; exact issue reads are not live mode. | NONE | `JIRA-KNOWLEDGE-ADAPTER-1`; adapter/sync tests named by accepted audit | Keep deferred capabilities explicit; assess application binding only through a separately approved task. |
| Confluence | `confluence / wiki_knowledge` | `pages` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Adapter exact reads and sync do not prove Confluence index binding, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Space reconciliation exists, but no incremental feed, application sink or provider-specific materialization proof exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Confluence live registration or handler; exact page reads are not live mode. | NONE | `CONFLUENCE-KNOWLEDGE-ADAPTER-1`; adapter/sync tests named by accepted audit | Keep deferred capabilities explicit; assess application binding only through a separately approved task. |

### Databricks decision note

Databricks has no source-kind mode row: **classification is blocked by
source-kind selection**. The explicit provider decision above is authoritative;
the roadmap must select exactly one source kind before a provider/source-kind
mode row is created. Unity Catalog metadata, workspace tree, volume files and
query snapshots remain hypothetical and are not classified here.

## 6. Indexed-mode findings

The repository has a real generic index/RAG path and a real LKW connected
source path. The accepted provider-specific materializers cover Slack and
Microsoft Graph Teams Chat:

- `connected_source_wiring.py` registers all implemented Vendor Knowledge
  adapters and binds the sync/index services;
- `connected_source_materializer.py` converts Slack structured records into
  safe indexed documents;
- `test_slack_connected_source_end_to_end.py` exercises discovery, binding,
  synchronization, vector-store population, Search and Ask citations;
- `test_connected_source_materializer.py` proves Graph Teams Chat resolution,
  canonical `KnowledgeDocument` materialization and entry into the generic
  index service;
- the Slack application task is `ACCEPTED / CLOSED` through its focused
  continuation, recovery, materialization and indexed Search/Ask proof.

Google, Jira and Confluence source kinds have no exact provider-to-index bridge
in the inspected application wiring. Generic LKW local-file indexing and
generic RAG infrastructure are not transferred to those rows.

Removal propagation is deliberately not inferred from adapter reconciliation.
Slack has no removal tombstones in the accepted audit; Graph removal semantics
remain source-specific; Google Docs/Sheets, Jira and Confluence lack the
required provider/application removal path.

## 7. Durable-materialization findings

The generic runtime provides a conditional DocumentStore sink, delivery
identity, item state, checkpoints, source leases, reconciliation runs,
queue/worker continuation and recovery states. Provider adapters and focused
sync tests exist for the accepted Graph family, Slack, Google
`drive`/`docs`/`sheets`, Jira and Confluence.

Representative application-owned materialization/index proof covers Slack and
Microsoft Graph Teams Chat through the provider-neutral coordinator→sink path.
For adapter rows without that representative proof, `FOUNDATION_ONLY` means
adapter/reconciliation layers are present while application materialization is
not proven; this is an intentional VK-6 boundary, not a platform regression.

The DocumentStore runtime itself is not counted as provider-specific
application materialization.

## 8. Live-mode findings

The application contains provider-neutral live binding validation,
capability descriptors, an exact handler registry, bounded execution,
normalized result items, byte/item/time limits, safe retention and receipts.
Hybrid Ask can orchestrate indexed and live evidence.

Microsoft Graph has five accepted bounded provider/source-kind live handlers
and registrations: Drive, Mail, Teams Channel, Teams Chat and Calendar. Slack
has its accepted bounded conversation live family. Other provider/source-kind
rows are explicitly `UNSUPPORTED` for Live. Exact provider reads,
remote-resource descriptors and adapter tests do not change the operation
matrix: an adapter exact read is not a live capability.

### Slack bounded Ask closeout

`SLACK-LIVE-DISCOVERY-AND-ASK-READINESS-1` is `ACCEPTED / CLOSED` only for
bounded recent configured-channel Ask. Its execution is two-phase: stage 1
executes list calls for active current-workspace bindings; stage 2 parses only
normalized Slack list results, filters/ranks actual roots, and executes at most
three globally selected binding-owned thread reads through the shared executor.
Coverage is execution-derived, including attempted calls, root/reply counts,
binding ownership, truncation and deterministic partial reasons. Root and reply
evidence remains transient and is not indexed or durably persisted.

This row does not claim native Slack workspace search, exhaustive history,
arbitrary token-accessible channel discovery, files/attachments or organization-
  wide Ask Slack. Indexed / RAG and Durable / Storage / Materialization
  lifecycle status remains independent.

## 9. Commercially supported mode claims

Current commercial claims must remain source-kind specific:

- **Indexed:** no broad Vendor Knowledge commercial claim.
  Slack `slack_conversation` and Microsoft Graph `teams_chat` have accepted
  provider-neutral indexed bridge proofs; this does not establish full provider
  coverage.
- **Durable materialization:** provider-neutral durable application
  materialization is accepted through `KnowledgeSyncSink` /
  `DocumentStoreDurableKnowledgeSyncSink`. Slack structured-record durable
  materialization and Microsoft Graph `teams_chat` coordinator→sink proofs are
  accepted. Broader provider application wiring and commercial coverage remain
  outside this claim (→ VK-6 / VK-8).
- **Live:** no Vendor Knowledge provider/source kind is commercially
  supported; technically accepted Graph and Slack Live rows remain scoped to
  their bounded handlers.

Prohibited claims: all Vendor Knowledge providers support indexed mode, all
adapters have production materialization, or all providers support live
access.

## 10. Cross-provider gaps

1. The provider-neutral indexed bridge is accepted for Slack
   `slack_conversation` and Microsoft Graph `teams_chat`; broader application
   wiring remains to expand (→ VK-6/VK-8).
2. Provider adapters and reconciliation are connected to a provider-neutral
   durable sink port; broader production application hosts beyond the DocumentStore
   durable sink and representative Slack/Teams Chat indexed paths remain to
   expand (→ VK-6/VK-8).
3. VK-6 proves registration completeness for the accepted Graph and Slack Live
   source kinds; Google, Jira and Confluence remain explicitly unsupported.
4. Indexed removal, provenance-preserving refresh and application query proof
   are established for the representative Slack and Teams Chat bridge paths;
   broader product E2E proof remains deferred to VK-8.
5. Google `drive`, `docs`, `sheets` and `calendar` have Vendor Knowledge
   adapter implementations despite stale roadmap wording; they still lack
   provider-neutral application-owned indexed wiring and all four remain
   explicitly unsupported for Live.
6. Microsoft Graph Mail low-level attachment reads must remain distinct from
   adapter-level attachment inventory and from any three-mode claim.
7. Databricks source-kind selection is still a roadmap decision.

## 11. Follow-up sequencing — superseded by the canonical platform roadmap

The matrix records evidence and gaps; it does not activate implementation.
Its former follow-up sequence is superseded by the canonical VK-1–VK-9 order in
[`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md):

1. `VENDOR-KNOWLEDGE-UNIFIED-THREE-MODE-CONTRACT-AUDIT-1`;
2. `VENDOR-KNOWLEDGE-PLUGIN-CAPABILITY-CONTRACT-1`;
3. `VENDOR-KNOWLEDGE-DURABLE-LIFECYCLE-CLOSEOUT-1`;
4. `VENDOR-KNOWLEDGE-INDEXED-BRIDGE-1`;
5. `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-CLOSEOUT-1`;
6. `VENDOR-KNOWLEDGE-PROVIDER-COVERAGE-1`;
7. `VENDOR-KNOWLEDGE-FRONTEND-NEUTRALITY-PROOF-1`;
8. `VENDOR-KNOWLEDGE-CROSS-PROVIDER-E2E-1`;
9. `VENDOR-KNOWLEDGE-PLATFORM-CLOSEOUT-1`.

The current matrix records Atlan and Power BI as not implemented and Databricks
as lacking a selected Vendor Knowledge source kind; no capability is inferred
for any of them. The accepted Slack connected-source result remains
`LKW-SLACK-CONNECTED-SOURCE-1` **ACCEPTED / CLOSED** and is not reopened.

## 12. Evidence appendix

### Canonical input and roadmap

- [`VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md`](VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md)
  — accepted inventory, source-specific adapter/reconciliation boundaries and
  evidence references.
- [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md)
  — roadmap status, Slack LKW correction state, generic durable/live plans
  and Google contradiction.

### Generic platform and application surfaces

- `intergrax/runtime/vendor_knowledge/sync_contracts.py`
- `intergrax/runtime/vendor_knowledge/sync_document_store.py`
- `intergrax/runtime/vendor_knowledge/sync_runtime.py`
- `applications/local_workspace_application/workspaces/connected_source_materializer.py`
- `applications/local_workspace_application/workspaces/knowledge_ingestion.py`
- `applications/local_workspace_application/workspaces/document_indexing.py`
- `applications/local_workspace_application/workspaces/knowledge_live_access_service.py`
- `applications/local_workspace_application/workspaces/hybrid_ask_execution.py`
- `applications/local_workspace_application/workspaces/connected_source_wiring.py`
- `intergrax/integrations/providers/collaboration_suite/google_workspace/integration.py`

### Focused proof surfaces

- `applications/local_workspace_application/tests/workspaces/test_slack_connected_source_end_to_end.py`
  — Slack discovery, sync, index, Search, Ask, citation and retry path.
- `applications/local_workspace_application/tests/workspaces/test_knowledge_live_access_service.py`
  — provider-neutral live binding validation and safe configuration lifecycle.
- `applications/local_workspace_application/tests/workspaces/test_knowledge_access_indexed_live_reuse_proof.py`
  — provider-neutral indexed/live reuse and boundary proof.
- `tests/unit/integrations/providers/collaboration_suite/google_workspace/test_calendar.py`
  — current Google Calendar exact read/integration proof; not Vendor Knowledge
  adapter or mode proof.

The accepted audit's provider-specific adapter tests are referenced rather than
re-read here; they prove adapter/sync layers, not the higher application modes.

---

## 13. Historical unified live capability rollout evidence — SUPERSEDED BY VK-6/VK-9

### `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1`

**Status:** `ACCEPTED / CLOSED`

Historical architecture reference:
[`VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md`](../../architecture/VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md)
— `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1` —
`READY_FOR_REVIEW`.

This addendum is retained for historical traceability only. The VK-6 matrix
above is authoritative for current selective capability status; this evidence
does not convert any `FOUNDATION_ONLY` row into an implemented provider
capability or create current sequencing.

Shared live foundation:

```text
current production foundation: implemented
  LiveCapabilityDescriptorV1
  tenant-safe capability catalog
  durable Live Access Binding lifecycle
  evidence-plan validation
  LiveCapabilityHandlerV1 protocol
  exact handler registry
  provider-neutral executor
  connection integration resolver
  basic item/byte budgets
  normalized result/evidence models
  receipt-only retention

FOUNDATION-1 frozen shared delta: implemented and ACCEPTED / CLOSED
  canonical source_kind assertion and validation
  contract_version across descriptor/handler/request/result/binding
  strict capability-specific request models
  request/result schema resolution
  ValidatedLiveCapabilityCallV1 or equivalent typed call
  atomic descriptor-handler-schema registration
  missing-pair and duplicate-pair validation
  provider page/request/upstream/content budgets
  expanded provider-neutral error taxonomy
  source-kind-aware result/evidence provenance
  ordered item-identity-aware receipt hashing
  safe-locator validation/filtering
  shared registration bootstrap
  shared contract test suite

FOUNDATION-1-REVIEW-FIX-2: implemented and ACCEPTED / CLOSED
  canonical live execution contracts owned by Vendor Knowledge runtime
  application executor consumes the exact runtime contracts
  runtime live modules have no LKW application import
  runtime-only handler construction and strict outcome proof

MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE: ACCEPTED / CLOSED
  bounded Drive list/query is registered through the shared live executor

MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL: ACCEPTED / CLOSED
  bounded mailbox-folder message list/query is registered through the same
  shared live executor; body, thread and attachment reads remain deferred

MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL: ACCEPTED / CLOSED
  bounded root-post metadata list is registered through the same shared live executor

MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT: ACCEPTED / CLOSED
  bounded fixed-window metadata-only message list is registered through the same shared live executor

MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR: ACCEPTED / CLOSED
  bounded binding-selected initial metadata-only page is registered through the same shared live executor

provider-specific production handlers:
  implemented for Microsoft Graph Drive, Mail, Teams Channel, Teams Chat and Calendar list

provider-specific production registrations:
  implemented for Microsoft Graph Drive, Mail, Teams Channel, Teams Chat and Calendar list

all other provider/source-kind live handlers and registrations:
  not implemented

cross-provider production proof: ACCEPTED / CLOSED through VK-8
```

Rows without an accepted or review-ready provider task use the same planned
boundary. The `PLANNED:vk...` values in the legacy matrix are planning
placeholders, not canonical capability IDs.
`ARCH-1` freezes the provider-neutral request/result contract decisions and
exact capability naming. `FOUNDATION-1` implements and validates the shared
schemas, typed call, registration, budgets, provenance, locator and receipt
boundary; all five Microsoft Graph live rows are accepted and closed, while
other provider/source-kind rows remain `FOUNDATION_ONLY` until their provider
task is accepted.

| provider | source_kind | capability_id | search/list support | exact-read support | resource scope | request schema | result schema | timeout | item budget | byte budget | evidence mapping | safe locator | receipt behavior | retention | descriptor registration | handler registration | proof status | commercial status |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Microsoft Graph | `drive` | `vendor.ms365_graph.drive.list` | unsupported by provider / supported list-query / unsupported exact read | unsupported by provider / not applicable child / deferred content | one known drive ID from binding | `MsGraphDriveListLiveRequestV1` | `LiveCapabilityExecutionResultV1` | executor deadline | restrictive minimum | restrictive minimum | normalized metadata/item identity | filtered provider web locator | receipt only | ephemeral/receipt-only | registered | registered | `ACCEPTED / CLOSED` | search and exact item read are unsupported; child read is not applicable; content read is deferred because provider bytes do not fit the textual live result boundary |
| Microsoft Graph | `mail` | `vendor.ms365_graph.mail.list` | unsupported by provider / supported list-query / unsupported exact read | unsupported by provider / deferred thread / deferred child and content | one binding-derived opaque mailbox-folder scope | `MsGraphMailListLiveRequestV1` | `LiveCapabilityExecutionResultV1` | executor deadline | restrictive minimum | restrictive minimum | normalized metadata/item identity | filtered provider web locator | receipt only | ephemeral/receipt-only | registered | registered | `ACCEPTED / CLOSED` | search, exact read and thread read are unsupported; attachment inventory/content and bounded content read remain deferred |
| Microsoft Graph | `teams_channel` | `vendor.ms365_graph.teams_channel.list` | bounded list, at most one root post | search unsupported; replies/thread/content/attachment reads deferred | one binding-fixed team + channel | `MsGraphTeamsChannelListLiveRequestV1` | `LiveCapabilityExecutionResultV1` | one call | one result item | one provider item | root-post metadata/deletion | provider-safe locator only | receipt only | ephemeral/receipt-only | registered | registered | `ACCEPTED / CLOSED` | bounded root-post metadata list; replies, full traversal and content remain deferred |
| Microsoft Graph | `teams_chat` | `vendor.ms365_graph.teams_chat.list` | bounded fixed-window list | exact read unsupported; content/attachments deferred | one binding-derived opaque chat/window scope | `MsGraphTeamsChatListLiveRequestV1` | `LiveCapabilityExecutionResultV1` | one call | 50 items | 50 provider items | metadata/deletion | provider-safe locator only | receipt only | ephemeral/receipt-only | registered | registered | `ACCEPTED / CLOSED` | metadata-only message list; bodies, mentions, reactions and hosted content excluded |
| Microsoft Graph | `calendar` | `vendor.ms365_graph.calendar.list` | bounded initial list; primary delta or non-primary snapshot selected by adapter | exact read unsupported; content/attachments deferred | one binding-derived opaque calendar/window scope | `MsGraphCalendarListLiveRequestV1` | `LiveCapabilityExecutionResultV1` | one call | 200 items | 200 provider items | metadata/window-removal | provider-safe locator only | receipt only | ephemeral/receipt-only | registered | registered | `ACCEPTED / CLOSED` | one initial page only; complete traversal and continuation replay deferred |
| Slack | `slack_conversation` | `vendor.slack.slack_conversation.list`; `vendor.slack.slack_conversation.thread.read`; `vendor.slack.slack_conversation.read` | bounded root list; one-page thread read; exact read; bounded search unsupported | one exact message; bounded text only when safe; file/attachment reads deferred | one canonical opaque conversation scope | strict Slack list/thread/read request v1 schemas | `LiveCapabilityExecutionResultV1` | one executor call per operation | 1 / 15 / 1 | finite shared byte budgets | normalized metadata; bounded exact text; safe item identity | shared safe locator filtering | receipt only | ephemeral/receipt-only | registered | registered | `ACCEPTED / CLOSED` | search, full traversal, authoritative permissions and file/attachment reads remain unsupported or deferred |
| Jira | `issues` | `PLANNED:vk.jira.issues.search/read` | planned | planned | one Jira project | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | planned | planned | `FOUNDATION_ONLY` | NONE |
| Confluence | `pages` | `PLANNED:vk.confluence.pages.search/read` | planned | planned | one Confluence space | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | planned | planned | `FOUNDATION_ONLY` | NONE |
| Google Workspace | `drive` | `PLANNED:vk.google_workspace.drive.search/read` | gated | gated | selected Drive resource | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | gated | gated | `FOUNDATION_ONLY` | NONE |
| Google Workspace | `docs` | `PLANNED:vk.google_workspace.docs.search/read` | gated | gated | one known document | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | gated | gated | `FOUNDATION_ONLY` | NONE |
| Google Workspace | `sheets` | `PLANNED:vk.google_workspace.sheets.search/read` | gated | gated | one known spreadsheet | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | gated | gated | `FOUNDATION_ONLY` | NONE |
| Google Workspace | `calendar` | `PLANNED:vk.google_workspace.calendar.search/read` | gated | gated | calendar + bounded time window | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | gated | gated | `FOUNDATION_ONLY` | NONE |
| Google Workspace | `slides` | `PLANNED:vk.google_workspace.slides.search/read` | gated | gated | one known presentation | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | gated | gated | `FOUNDATION_ONLY` | NONE |
| Google Workspace | `mail` | `PLANNED:vk.google_workspace.mail.search/read` | gated | gated | mailbox + bounded folder/query scope | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | gated | gated | `FOUNDATION_ONLY` | NONE |
| Google Workspace | `chat` | `PLANNED:vk.google_workspace.chat.search/read` | gated | gated | one authorized space + window | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | gated | gated | `FOUNDATION_ONLY` | NONE |

`gated` means that the exact source kind must first pass
`GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1`; it is not an implementation
claim. Shared budget semantics and strict schemas are provided by the accepted
foundation; provider-specific tasks must prove their own bounded behavior.

### Microsoft Graph Drive live operation matrix

```text
bounded search: UNSUPPORTED_BY_PROVIDER
bounded list/query: SUPPORTED
exact item read: UNSUPPORTED_BY_PROVIDER
child read: NOT_APPLICABLE
bounded content read: DEFERRED
```

The supported list/query operation uses the existing
`read_drive_delta_page` boundary once per live call and returns metadata-only
text. Content remains deferred because the provider surface returns binary
bytes, the shared live result is textual, and the adapter does not propagate
the live per-item byte budget.

### Microsoft Graph Mail live operation matrix

```text
bounded search: UNSUPPORTED_BY_PROVIDER
bounded list/query: SUPPORTED through read_mail_messages_delta_page
exact item read: UNSUPPORTED_BY_PROVIDER
thread read: UNSUPPORTED_BY_PROVIDER
child read / attachment inventory: DEFERRED
bounded content read: DEFERRED
```

The Mail capability reads exactly one adapter page for one opaque
mailbox-folder binding and emits metadata-only deterministic JSON. It does not
follow or expose continuation, read message bodies, infer global deletion, or
claim attachment inventory/content from `has_attachments`.

### Google readiness gate

#### `GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1`

**Status:** `PLANNED`

The gate evaluates each of the seven exact Google source kinds independently.
For every row it requires:

```text
stable source_kind identity
shared Google Workspace integration reuse
typed source-specific read surface
bounded provider read
safe provider references
stable remote item identity
provider error normalization
no secret-bearing public models
Vendor Knowledge adapter availability where required
focused source-specific tests
current task/review status
```

Allowed outcomes are:

```text
READY_FOR_LIVE_ROLLOUT
BLOCKED_BY_CORE_READ_SURFACE
BLOCKED_BY_ADAPTER
BLOCKED_BY_PROVIDER_SEMANTICS
BLOCKED_BY_REVIEW
```

The gate is not yet run, so every Google row above is `FOUNDATION_ONLY` with
activation `gated`. A ready Drive/Docs/Sheets/Calendar subset may activate
independently; Slides, Mail or Chat readiness does not block that subset, and
an unfinished source kind must not be represented as ready by family inference.
The Google family closeout requires an accepted live implementation or an
explicit deferred/excluded decision for every source kind in the accepted core
family scope.

### Ownership and family closeout

Google core remains owned by its separate workstream:

```text
integration primitives
source-specific read surfaces
pagination/cursors
typed provider models
durable Vendor Knowledge adapters
source-specific adapter tests
Google family implementation closeout
```

This rollout owns Google live integration:

```text
LiveCapabilityDescriptorV1 declarations
LiveCapabilityHandlerV1 implementations
provider-neutral request/result contracts
handler and catalog registration
effective budgets and timeouts
provider error normalization
normalized evidence and safe locators
receipts and retention
cross-provider tests
live family closeout
```

No Google-specific executor, registry, receipt mechanism or direct LKW-to-Google
path is allowed. No provider-specific live framework or duplicate provider
client is allowed for any family.

Provider tasks may not redefine the shared live semantics. They enter only
through the accepted `FOUNDATION-1` boundary and own source-specific
availability, strict request models, descriptors, handlers, provider read
mapping, bounded invocation, error/result/locator mapping, focused tests and
registration. No provider client or credential is introduced by the live
handler task.

Every provider family closeout verifies shared integration reuse, tenant-safe
connection resolution, exact source-kind and resource-scope isolation,
read-only enforcement, bounded request/results, timeout behavior, normalized
errors/evidence, safe locators, private receipts, credential non-disclosure,
contract tests and production proof. Google additionally proves that all seven
handlers reuse the existing shared `GoogleWorkspaceCollaborationSuiteIntegration`.

### Historical Live rollout evidence — SUPERSEDED BY VK-6/VK-9

The following rollout order and status record is retained for traceability only.
It is superseded by the accepted VK-6 capability matrix and VK-9 platform
closeout and must not be interpreted as current sequencing or active work.

Historical rollout sequence:

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1 — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITIES-1-FAMILY-CLOSEOUT — READY_FOR_REVIEW
```

Historically, the complete order was frozen in
`KNOWLEDGE_SOURCE_INTEGRATIONS.md`, from this rollout plan through `ARCH-1`,
the shared `FOUNDATION-1`, Graph, Slack, Jira, Confluence, the Google
readiness gate and independently gated Google source tasks, to both family
closeouts and:

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1 — PLANNED
```

That historical audit intent is superseded. The current capability truth is
the VK-6 matrix above; future Jira, Confluence or Google implementation is
separate provider/product expansion and is not VK-10.

### Historical rollout status — SUPERSEDED / TRACEABILITY ONLY

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1: ACCEPTED / CLOSED
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1: READY_FOR_REVIEW
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR: ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITIES-1-FAMILY-CLOSEOUT: READY_FOR_REVIEW
SLACK-KNOWLEDGE-LIVE-CAPABILITIES-1: ACCEPTED / CLOSED
SLACK-KNOWLEDGE-LIVE-CAPABILITIES-1-FAMILY-CLOSEOUT: ACCEPTED / CLOSED
other Microsoft Graph live tasks: PLANNED
Slack live task: ACCEPTED / CLOSED
Slack bounded Ask readiness: ACCEPTED / CLOSED
Jira live task: PLANNED
Confluence live task: PLANNED
GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1: PLANNED
Google source-kind live tasks: PLANNED / GATED_BY_CORE_READINESS
Google live family closeout: PLANNED
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1: PLANNED
```

## 14. VK-9 platform closeout reconciliation

`VENDOR-KNOWLEDGE-PLATFORM-CLOSEOUT-1` is **ACCEPTED / CLOSED**. The
provider/source-kind matrix above remains authoritative for selective coverage:
`ACCEPTED`, `FOUNDATION_ONLY`, `UNSUPPORTED` and `NOT IMPLEMENTED` are truthful
capability statuses, not platform defects.

Current canonical authority is:

- VK-9 platform closeout reconciliation — authoritative for platform state;
- the VK-6 provider/source-kind capability matrix — authoritative for current
  selective capability status;
- historical Live rollout sections — superseded and retained for traceability
  only.

The matrix is consumed together with the final closeout in
[`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md). The
platform has zero `PLATFORM_BLOCKER` items. Durable, Indexed and Live remain
independently composable through explicit plugin mode declarations; provider
ACL, deletion/tombstone, content/history, commercial and SDK/API packaging
limitations remain outside the platform-completeness claim.

The authoritative roadmap state is:

```text
VENDOR KNOWLEDGE PLATFORM
VK-1 through VK-9 — COMPLETE
NEXT: NONE
```

Future provider/source-kind expansion is separate provider/product work only
and must not be represented as VK-10.

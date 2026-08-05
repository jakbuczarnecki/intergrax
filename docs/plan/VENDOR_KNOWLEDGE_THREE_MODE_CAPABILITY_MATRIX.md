# Vendor Knowledge Three-Mode Capability Matrix

**Task:** `VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1`  
**Status:** `READY_FOR_REVIEW`  
**Branch:** `development`  
**Input:** [`VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md`](VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md)  
**Roadmap:** [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md)
**Live architecture:** [`VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md`](../architecture/VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md)

## 1. Executive summary

This document is the canonical capability matrix at the exact
`provider_family × source_kind × mode` boundary. It does not implement,
activate or accept any missing mode.

The accepted adapter-family input proves provider adapters and durable
reconciliation foundations, not automatically application materialization,
indexed ingestion or live access. Current repository evidence proves:

- generic indexed/RAG infrastructure and an application indexed path;
- generic durable sink, checkpoint, reconciliation and recovery foundations;
- generic live binding, capability, executor, limits, normalized evidence and
  receipt foundations;
- the accepted `ARCH-1` shared live delta is implemented and closed through
  `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1`;
- a provider-specific application indexed path only for Slack
  `slack_conversation`, with final accepted LKW closeout still unresolved;
- provider/source-kind live handlers and registrations are implemented for the
  five Microsoft Graph bounded list capabilities: Drive, Mail, Teams Channel,
  Teams Chat and Calendar;

Microsoft Graph Drive, Mail, Teams Channel, Teams Chat and Calendar live access
are `ACCEPTED / CLOSED`; all other live provider/source-kind rows remain
conservative and unimplemented.

## 2. Mode definitions

### `INDEXED`

Provider content is durably or controllably materialized, converted to
canonical content, written to an index, refreshed, removed when supported,
retained with provenance, and retrieved through an application query path.
Generic local-file ingestion is not Vendor Knowledge provider proof.

### `DURABLE_MATERIALIZATION`

Provider content is synchronized into durable application-controlled storage
through a source binding, initial sync, continuation or reconciliation,
deterministic and idempotent delivery, checkpointing, recovery and explicit
application ownership. The generic DocumentStore sink proves the runtime
contract; it does not by itself prove a production application sink.

### `LIVE`

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

Mode statuses use only the required vocabulary:

| Status | Matrix meaning |
|---|---|
| `ACCEPTED` | Exact mode has production wiring, focused proof and accepted closeout. |
| `IMPLEMENTED_UNREVIEWED` | Implementation and focused proof exist; accepted mode closeout is absent. |
| `PARTIAL` | A meaningful subset exists, but the user-facing workflow is incomplete. |
| `FOUNDATION_ONLY` | Generic contracts/runtime or low-level reads exist without exact mode wiring. |
| `PLANNED` | Explicitly routed work is not implemented. |
| `DEFERRED` | Intentionally postponed work has no active implementation route. |
| `NOT_IMPLEMENTED` | No implementation or committed roadmap route exists. |
| `NOT_APPLICABLE` | The mode genuinely does not apply to the source contract. |

Subcolumns use only `YES`, `NO`, `PARTIAL`, `NOT_APPLICABLE` and `UNPROVEN`.
`UNPROVEN` describes evidence; it is never used as a mode status.

The audit inventory is authoritative for source kinds. Databricks is excluded
from the exact matrix because no source kind has been selected.

## 4. Platform foundation summary

| Mode | Generic foundation | Boundary |
|---|---|---|
| Indexed | `YES` — LKW document ingestion, canonical document indexing, vector retrieval and query/Ask paths | Local-file/LKW capability is not transferred to Vendor Knowledge providers. |
| Durable materialization | `YES` — `DocumentStore`, idempotent sink, remote-item state, checkpoints, leases, reconciliation, queue/worker and recovery | Provider adapter/sync proof is not application-owned materialization proof. |
| Live | `YES` — current typed binding/catalog contracts, exact handler registry, validated executor, bounded limits, normalized evidence, receipts and retention | `FOUNDATION-1` provides the accepted shared boundary; the Graph family closeout verifies five provider-specific list handlers on the same integration and executor. |

## 5. Exact provider/source-kind matrix

| provider_family | integration_identity | source_kind | adapter_status | indexed_status | indexed_platform_foundation | indexed_provider_wiring | indexed_application_wiring | indexed_refresh | indexed_removal | indexed_provenance | indexed_proof | indexed_gap | durable_status | durable_platform_foundation | durable_provider_wiring | durable_application_sink | durable_checkpoint | durable_recovery | durable_proof | durable_gap | live_status | live_platform_foundation | live_provider_wiring | live_executor | live_limits | live_evidence | live_receipt | live_application_wiring | live_proof | live_gap | commercially_supported_modes | primary_evidence | next_action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Microsoft Graph | `ms365_graph / collaboration_suite` | `drive` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No Graph-to-index bridge or application query proof; preserve accepted delta/full-reconciliation and ACL limits. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Accepted adapter/reconciliation exists, but no production application-owned sink was found. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Drive list/query only; search and exact read are unsupported, child read is not applicable, content read is deferred. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE`; focused live tests | Preserve the accepted Drive list proof and exact operation matrix. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `mail` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No folder-scoped Graph Mail index bridge; attachment presence is not attachment inventory or bytes. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Accepted reconciliation is folder-scoped; no production application-owned sink was found. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Mail live handler and registration are implemented; body, thread and attachment reads remain deferred. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL`; focused live tests | Preserve folder-scoped delta/removal and attachment non-goals. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `teams_channel` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No channel-to-index bridge or application proof; deletion evidence remains explicit `deletedDateTime`. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Adapter snapshot/reconciliation is proven, but application sink ownership is absent. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Live remains limited to one root post and does not list replies or all channel messages. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL`; focused live tests | Preserve explicit deletion semantics and the bounded root-post list. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `teams_chat` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No chat-to-index bridge or application proof; fixed-window scope does not prove live or indexed lifecycle. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Adapter fixed-window snapshot/reconciliation exists, but application sink ownership is absent. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | Fixed-window metadata-only list; bodies, mentions, reactions, attachments and hosted content remain excluded. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT`; focused live tests | Preserve fixed-window and explicit-deletion semantics. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `calendar` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No Calendar-to-index bridge or application proof; primary delta and non-primary snapshot paths must remain separate. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Accepted primary/non-primary reconciliation exists, but no production application-owned sink was found. | `ACCEPTED / CLOSED` | YES | YES | YES | YES | YES | YES | YES | FOCUSED_TESTS | One binding-selected initial metadata-only page; complete traversal, replay and event content remain deferred. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`; `MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR`; focused live tests | Preserve primary-delta versus non-primary-snapshot semantics and window removals. |
| Slack | `slack / conversation_channel` | `slack_conversation` | `IMPLEMENTED_UNREVIEWED` | `IMPLEMENTED_UNREVIEWED` | YES | YES | YES | PARTIAL | NO | YES | PARTIAL | Current HTTP→sync→index→Search/Ask proof exists; final accepted LKW/recovery closeout and removal propagation remain unresolved. | `PARTIAL` | YES | YES | YES | YES | PARTIAL | PARTIAL | Connected-source application sink and retry path exist; final crash-safe recovery/closeout is still unresolved. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Slack live handler registration; exact reads alone are insufficient. | NONE | `test_slack_connected_source_end_to_end.py`; `connected_source_wiring.py`; `connected_source_materializer.py`; `LKW-SLACK-CONNECTED-SOURCE-1` remains `CHANGES_REQUIRED` | Complete the already-routed Slack LKW proof before any commercial indexed claim; keep live unimplemented. |
| Google Workspace | `google_workspace / collaboration_suite` | `drive` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Adapter and sync tests exist, but no Google application binding, index bridge or indexed proof exists. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Provider adapter/reconciliation exists; application-owned materialization is not wired. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Google Drive live handler or application invocation. | NONE | `google_workspace_drive` adapter/sync tests named by accepted audit; roadmap Google status is stale | Correct status only through matrix routing; route application proof separately if product support is approved. |
| Google Workspace | `google_workspace / collaboration_suite` | `docs` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Known-document adapter does not establish indexed discovery, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | One-item reconciliation exists; broad discovery, deletion semantics and application sink are absent. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Google Docs live handler or application invocation. | NONE | `google_workspace_docs` adapter/sync tests named by accepted audit; current integration | Keep exact-known-resource limits explicit; do not copy stale “no adapters” wording as capability evidence. |
| Google Workspace | `google_workspace / collaboration_suite` | `sheets` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Known-spreadsheet adapter does not establish indexed discovery, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | One-item reconciliation exists; broad discovery, deletion semantics and application sink are absent. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Google Sheets live handler or application invocation. | NONE | `google_workspace_sheets` adapter/sync tests named by accepted audit; current integration | Keep exact-known-resource limits explicit; route application proof only if approved. |
| Google Workspace | `google_workspace / collaboration_suite` | `slides` | `PLANNED` | `PLANNED` | YES | NO | NO | NO | NO | NO | UNPROVEN | Shared enum/transport only; no source-specific read, adapter, index bridge or application proof. | `FOUNDATION_ONLY` | YES | NO | NO | YES | YES | UNPROVEN | Generic sync runtime exists, but no Slides provider adapter or application sink exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | Generic live platform exists without a Slides registration or handler. | NONE | Accepted audit inventory; `GoogleWorkspaceSourceKind.SLIDES`; roadmap planned task | Keep planned; do not activate a Slides implementation task from this matrix. |
| Google Workspace | `google_workspace / collaboration_suite` | `calendar` | `FOUNDATION_ONLY` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Current source-specific Calendar read/integration exists, but no Vendor Knowledge adapter, index bridge or app proof exists. | `FOUNDATION_ONLY` | YES | NO | NO | YES | YES | UNPROVEN | Exact Calendar read is not durable materialization; adapter, sync and application sink are absent. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | Generic live platform exists without Google Calendar registration or handler. | NONE | Current `GoogleCalendarKnowledgeReader` and integration/test; no runtime Vendor Knowledge Calendar adapter | Record the roadmap contradiction; do not infer adapter or three-mode parity from the exact read. |
| Google Workspace | `google_workspace / collaboration_suite` | `mail` | `PLANNED` | `PLANNED` | YES | NO | NO | NO | NO | NO | UNPROVEN | Shared enum/transport only; no Gmail read, adapter, index bridge or application proof. | `FOUNDATION_ONLY` | YES | NO | NO | YES | YES | UNPROVEN | Generic sync runtime exists, but no Gmail provider adapter or application sink exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | Generic live platform exists without a Gmail registration or handler. | NONE | Accepted audit inventory; `GoogleWorkspaceSourceKind.MAIL`; roadmap planned task | Keep planned; select a precise Gmail contract before routing implementation. |
| Google Workspace | `google_workspace / collaboration_suite` | `chat` | `PLANNED` | `PLANNED` | YES | NO | NO | NO | NO | NO | UNPROVEN | Shared enum/transport only; no Chat read, adapter, index bridge or application proof. | `FOUNDATION_ONLY` | YES | NO | NO | YES | YES | UNPROVEN | Generic sync runtime exists, but no Chat provider adapter or application sink exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | Generic live platform exists without a Chat registration or handler. | NONE | Accepted audit inventory; `GoogleWorkspaceSourceKind.CHAT`; roadmap planned task | Keep planned; select a precise Chat contract before routing implementation. |
| Jira | `jira / issue_tracker` | `issues` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Adapter exact reads and sync do not prove Jira index binding, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Project reconciliation exists, but no incremental feed, application sink or provider-specific materialization proof exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Jira live registration or handler; exact issue reads are not live mode. | NONE | `JIRA-KNOWLEDGE-ADAPTER-1`; adapter/sync tests named by accepted audit | Keep deferred capabilities explicit; assess application binding only through a separately approved task. |
| Confluence | `confluence / wiki_knowledge` | `pages` | `IMPLEMENTED_UNREVIEWED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | Adapter exact reads and sync do not prove Confluence index binding, lifecycle or application query wiring. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Space reconciliation exists, but no incremental feed, application sink or provider-specific materialization proof exists. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Confluence live registration or handler; exact page reads are not live mode. | NONE | `CONFLUENCE-KNOWLEDGE-ADAPTER-1`; adapter/sync tests named by accepted audit | Keep deferred capabilities explicit; assess application binding only through a separately approved task. |

### Databricks decision note

Databricks has no matrix row: **mode classification blocked by source-kind
selection**. The roadmap must select exactly one source kind before any
provider/source-kind/mode row is created. Unity Catalog metadata, workspace
tree, volume files and query snapshots remain hypothetical and are not
classified here.

## 6. Indexed-mode findings

The repository has a real generic index/RAG path and a real LKW connected
source path. The latter is provider-specific only for Slack:

- `connected_source_wiring.py` registers the Slack Vendor Knowledge adapter and
  binds the sync/index services;
- `connected_source_materializer.py` converts Slack structured records into
  safe indexed documents;
- `test_slack_connected_source_end_to_end.py` exercises discovery, binding,
  synchronization, vector-store population, Search and Ask citations;
- the roadmap still marks the Slack application task
  `CHANGES_REQUIRED`, so the exact mode is `IMPLEMENTED_UNREVIEWED`, not
  `ACCEPTED`.

No Microsoft Graph, Google, Jira or Confluence source kind has an exact
provider-to-index bridge in the inspected application wiring. Generic LKW
local-file indexing and generic RAG infrastructure are not transferred to
those rows.

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

Only the Slack connected-source path proves a production application-owned
materialization/index sink in the inspected application surfaces. Even there,
the final crash-safe recovery and accepted LKW closeout remain open. For all
other adapter rows, `PARTIAL` means adapter/reconciliation layers are present
while application materialization is not proven.

The DocumentStore runtime itself is not counted as provider-specific
application materialization.

## 8. Live-mode findings

The application contains provider-neutral live binding validation,
capability descriptors, an exact handler registry, bounded execution,
normalized result items, byte/item/time limits, safe retention and receipts.
Hybrid Ask can orchestrate indexed and live evidence.

The Microsoft Graph `drive` row has one accepted provider/source-kind live
handler: bounded list/query through the existing `read_drive_delta_page`
boundary. The Microsoft Graph `mail` row has one review-ready provider/source-
kind live handler and registration for bounded mailbox-folder list/query.
Other provider/source-kind rows remain `FOUNDATION_ONLY`. Exact provider reads,
remote-resource descriptors and live binding tests do not change the operation
matrix: an adapter exact read is not a live capability.

## 9. Commercially supported mode claims

Current commercial claims must remain source-kind specific:

- **Indexed:** no generally accepted Vendor Knowledge commercial claim.
  Slack `slack_conversation` is an `IMPLEMENTED_UNREVIEWED` demonstrable LKW
  path, but its final accepted proof is unresolved.
- **Durable materialization:** adapter-level synchronization is broader than
  production application materialization. The repository does not support a
  broad commercial claim for Graph, Slack, Google, Jira or Confluence
  application-owned materialization; Slack is `PARTIAL`.
- **Live:** no Vendor Knowledge provider/source kind is commercially
  supported; all rows are `FOUNDATION_ONLY`.

Prohibited claims: all Vendor Knowledge providers support indexed mode, all
adapters have production materialization, or all providers support live
access.

## 10. Cross-provider gaps

1. There is no provider/source-kind index bridge except the partial,
   unreviewed Slack LKW path.
2. Provider adapters and reconciliation are not uniformly connected to
   application-owned durable sinks.
3. There is no provider/source-kind live handler registration.
4. Indexed removal, provenance-preserving refresh and application query proof
   are not established for non-Slack rows.
5. Google `drive`, `docs` and `sheets` are implemented adapter candidates
   despite stale roadmap wording; Google `calendar` now has an exact
   source-specific read in the current tree but still lacks a Vendor
   Knowledge adapter.
6. Microsoft Graph Mail low-level attachment reads must remain distinct from
   adapter-level attachment inventory and from any three-mode claim.
7. Databricks source-kind selection is still a roadmap decision.

## 11. Follow-up sequencing

The matrix records gaps; it does not activate them. The documented sequence is:

1. accept this matrix after external review;
2. finish the already-routed Slack connected-source corrections and obtain
   the final indexed proof;
3. decide whether application-owned materialization is a supported product
   workflow for each non-LKW provider family;
4. route provider/source-kind index or materialization tasks only after the
   product boundary is approved;
5. route live capability contracts/handlers and application proofs separately;
6. select one Databricks source kind before any Databricks mode work.

These are sequencing findings, not newly activated implementation tasks.

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

## 13. Unified live capability rollout matrix

### `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-PLAN-1`

**Status:** `ACCEPTED / CLOSED`

Canonical architecture:
[`VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md`](../architecture/VENDOR_KNOWLEDGE_LIVE_CAPABILITY_ROLLOUT.md)
— `VENDOR-KNOWLEDGE-LIVE-CAPABILITY-ROLLOUT-ARCH-1` —
`READY_FOR_REVIEW`.

This addendum is the canonical live-rollout planning view. It does not convert
any `FOUNDATION_ONLY` row into an implemented provider capability.

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

cross-provider production proof: not implemented
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
| Slack | `slack_conversation` | `PLANNED:vk.slack.slack_conversation.search/read` | planned | planned | one authorized conversation | `LiveCapabilityRequestV1` (`ARCH-1`) | `NormalizedLiveResultV1` (`ARCH-1`) | effective policy | effective policy | effective policy | source/item/locator | opaque provider-safe | receipt only | ephemeral/receipt-only | planned | planned | `FOUNDATION_ONLY` | NONE |
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

### Canonical rollout order and final audit

Immediate next task:

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FOUNDATION-1 — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1A-DRIVE — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1B-MAIL — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1C-TEAMS-CHANNEL — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1D-TEAMS-CHAT — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITY-1E-CALENDAR — ACCEPTED / CLOSED
MSGRAPH-KNOWLEDGE-LIVE-CAPABILITIES-1-FAMILY-CLOSEOUT — READY_FOR_REVIEW
```

The complete order is frozen in
`KNOWLEDGE_SOURCE_INTEGRATIONS.md`, from this rollout plan through `ARCH-1`,
the shared `FOUNDATION-1`, Graph, Slack, Jira, Confluence, the Google
readiness gate and independently gated Google source tasks, to both family
closeouts and:

```text
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1 — PLANNED
```

The final audit matrix must retain the columns shown above and include all
source kinds that passed their readiness gates plus every explicitly deferred
source kind and its reason. Databricks remains excluded because no exact
`source_kind` has been selected; Power BI and Atlan are outside this rollout.

### Rollout status

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
other Microsoft Graph live tasks: PLANNED
Slack live task: PLANNED
Jira live task: PLANNED
Confluence live task: PLANNED
GOOGLE-WORKSPACE-KNOWLEDGE-LIVE-READINESS-GATE-1: PLANNED
Google source-kind live tasks: PLANNED / GATED_BY_CORE_READINESS
Google live family closeout: PLANNED
VENDOR-KNOWLEDGE-LIVE-CAPABILITY-FAMILY-AUDIT-1: PLANNED
```

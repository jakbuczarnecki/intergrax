# Vendor Knowledge Three-Mode Capability Matrix

**Task:** `VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1`  
**Status:** `READY_FOR_REVIEW`  
**Branch:** `development`  
**Input:** [`VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md`](VENDOR_KNOWLEDGE_ADAPTER_FAMILY_AUDIT.md)  
**Roadmap:** [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md)

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
- a provider-specific application indexed path only for Slack
  `slack_conversation`, with final accepted LKW closeout still unresolved;
- no provider/source-kind live handler registration.

No row has an `ACCEPTED` mode status. Microsoft Graph adapter acceptance is
preserved, while the higher application-mode claims remain conservative.

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
| Live | `YES` — typed binding/catalog contracts, exact handler registry, validated executor, limits, normalized evidence, receipts and retention | No provider/source-kind live handler registration or accepted live proof was found. |

## 5. Exact provider/source-kind matrix

| provider_family | integration_identity | source_kind | adapter_status | indexed_status | indexed_platform_foundation | indexed_provider_wiring | indexed_application_wiring | indexed_refresh | indexed_removal | indexed_provenance | indexed_proof | indexed_gap | durable_status | durable_platform_foundation | durable_provider_wiring | durable_application_sink | durable_checkpoint | durable_recovery | durable_proof | durable_gap | live_status | live_platform_foundation | live_provider_wiring | live_executor | live_limits | live_evidence | live_receipt | live_application_wiring | live_proof | live_gap | commercially_supported_modes | primary_evidence | next_action |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Microsoft Graph | `ms365_graph / collaboration_suite` | `drive` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No Graph-to-index bridge or application query proof; preserve accepted delta/full-reconciliation and ACL limits. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Accepted adapter/reconciliation exists, but no production application-owned sink was found. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Graph source-kind live handler or application invocation. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`; accepted audit input; sync runtime | Keep adapter scope; do not reopen Graph; await a separately routed application-mode task. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `mail` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No folder-scoped Graph Mail index bridge; attachment presence is not attachment inventory or bytes. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Accepted reconciliation is folder-scoped; no production application-owned sink was found. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No Graph Mail live handler; low-level reads are not live mode. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`; accepted audit input; Mail semantics in roadmap | Preserve folder-scoped delta/removal and attachment non-goals. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `teams_channel` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No channel-to-index bridge or application proof; deletion evidence remains explicit `deletedDateTime`. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Adapter snapshot/reconciliation is proven, but application sink ownership is absent. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No provider live registration or application call path. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`; accepted audit input | Preserve explicit deletion semantics; do not infer absence-based removal. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `teams_chat` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No chat-to-index bridge or application proof; fixed-window scope does not prove live or indexed lifecycle. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Adapter fixed-window snapshot/reconciliation exists, but application sink ownership is absent. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No provider live registration or application call path. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT`; accepted audit input | Preserve fixed-window and explicit-deletion semantics. |
| Microsoft Graph | `ms365_graph / collaboration_suite` | `calendar` | `ACCEPTED` | `FOUNDATION_ONLY` | YES | NO | NO | NO | NO | NO | UNPROVEN | No Calendar-to-index bridge or application proof; primary delta and non-primary snapshot paths must remain separate. | `PARTIAL` | YES | YES | NO | YES | YES | PARTIAL | Accepted primary/non-primary reconciliation exists, but no production application-owned sink was found. | `FOUNDATION_ONLY` | YES | NO | YES | YES | YES | YES | NO | UNPROVEN | No provider/source-kind live handler. | NONE | `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR`; accepted review correction; audit input | Preserve primary delta versus non-primary snapshot semantics. |
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

No inspected integration registers a provider/source-kind live handler. No
exact row therefore exceeds `FOUNDATION_ONLY`. Exact provider reads,
remote-resource descriptors and live binding tests do not change this result:
an adapter exact read is not a live capability.

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

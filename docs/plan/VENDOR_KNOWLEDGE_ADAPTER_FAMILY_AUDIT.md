# Vendor Knowledge Adapter Family Audit

**Task:** `VENDOR-KNOWLEDGE-ADAPTER-FAMILY-AUDIT-1`
**Status:** `READY_FOR_REVIEW`
**Branch:** `development`
**Canonical roadmap:** [`KNOWLEDGE_SOURCE_INTEGRATIONS.md`](KNOWLEDGE_SOURCE_INTEGRATIONS.md)

## 1. Executive summary

The repository proves five active Vendor Knowledge families or explicit
provider adapters:

- **Microsoft Graph** — the accepted reference family with `drive`, `mail`,
  `teams_channel`, `teams_chat` and `calendar`.
- **Slack** — foundation, adapter and durable reconciliation exist; the LKW
  connected-source path remains `CHANGES_REQUIRED`, and live access is absent.
- **Google Workspace** — `drive`, `docs` and `sheets` have production adapters
  and focused sync proof, despite stale roadmap wording. `slides`, `calendar`,
  `mail` and `chat` exist only in the shared low-level source-kind/transport
  surface.
- **Jira** — the `issues` adapter supports bounded project reconciliation and
  exact structured issue reads.
- **Confluence** — the `pages` adapter supports bounded space reconciliation
  and exact rich-text page reads.

Databricks is a roadmap entry, not an adapter family yet: no precise
`source_kind` has been selected. No family is proven end-to-end through all
seven layers. In particular, provider adapter and durable-sync proof must not
be read as proof of application binding, indexed LKW use, or live access.

## 2. Audit scope and evidence rules

The inventory is bounded to provider families discovered through the canonical
roadmap, the Vendor Knowledge adapter exports/registry, provider integration
identities, provider knowledge-read registrations and provider-specific tests.
No repository-wide semantic crawl was used.

Evidence precedence:

1. production implementation and provider-specific tests;
2. runtime adapter exports and explicit registry registration;
3. accepted architecture/status records;
4. current roadmap wording.

The Vendor Knowledge registry is instance-local and explicit. An exported
registration helper plus a provider-specific registry/facade test proves
runtime reachability for an adapter; it does not prove automatic application
bootstrap.

`IMPLEMENTED_UNREVIEWED` means production code and focused proof exist, but
this audit did not find an accepted family closeout for that row. `ACCEPTED`
is used only where the accepted Microsoft Graph records explicitly cover the
row. `PLANNED` rows are not user-facing support.

## 3. Layer model

The audit keeps these layers separate:

1. integration foundation;
2. typed knowledge read surface;
3. Vendor Knowledge adapter and facade registration;
4. durable synchronization and reconciliation;
5. application materialization and connected-source lifecycle;
6. indexed/RAG mode;
7. bounded live mode.

`YES`, `NO`, `PARTIAL`, `NOT_APPLICABLE` and `UNPROVEN` are the only capability
values used below. The matrix records adapter/runtime facts; it does not
promote a lower layer into a higher layer.

For the requested user workflow, active rows can represent a provider
connection, a precise typed source and an exact read. Broad discovery/listing
is proven for Graph, Slack, Jira, Confluence and Google Drive; Google Docs and
Sheets are exact-known-resource adapters rather than broad resource discovery.
Initial durable synchronization is proven for active rows, while incremental
sync and safe removal detection are source-specific. Application materialization
is only partial, indexed access is absent or incomplete, and live access is
absent for every family.

## 4. Family summary

| Family | Integration identity | Proven layers | Current boundary |
|---|---|---|---|
| Microsoft Graph | `ms365_graph` / `collaboration_suite` | 1–4, accepted per source kind | Application binding, indexed LKW bridge and live mode are outside the accepted adapter closeout; ACL and several attachment capabilities are intentionally unsupported. |
| Slack | `slack` / `conversation_channel` | 1–4 within bounded conversation windows | LKW connected source is `CHANGES_REQUIRED`; final indexed proof and live capability are not complete. |
| Google Workspace | `google_workspace` / `collaboration_suite` | 1–4 for `drive`, `docs`, `sheets`; foundation only for four additional kinds | Roadmap still says runtime work is planned; source selection/application wiring for active adapters and all three modes remain unproven. |
| Jira | `jira` / `issue_tracker` | 1–4 for `issues` | No incremental feed, tombstones, ACL/attachment projection or application/three-mode proof. |
| Confluence | `confluence` / `wiki_knowledge` | 1–4 for `pages` | No incremental feed, tombstones, ACL/attachment projection or application/three-mode proof. |
| Databricks | no Vendor Knowledge identity | none | Source-kind selection is deferred; do not infer Unity Catalog, workspace, volume or query-snapshot adapters. |

Jira and Confluence are not merged into an invented Atlassian runtime family:
the repository exposes separate provider identities and separate adapter
registrations.

## 5. Exact provider-family × source-kind matrix

The Databricks row is a decision record, not an exact source-kind row:
`[none selected]` is deliberately not a runtime identifier and is excluded
from exact source-kind validation.

| provider_family | integration_identity | source_kind | status | integration_foundation | knowledge_read_surface | vendor_adapter | initial_sync | incremental_sync | full_reconciliation | missing_item_or_tombstone_semantics | exact_content_read | structured_content | binary_content | attachment_or_related_inventory | application_materialization | indexed_mode | live_mode | primary_proof | known_gap | gap_class | commercial_impact | recommended_follow_up |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| Microsoft Graph | `ms365_graph` / `collaboration_suite` | `drive` | ACCEPTED | YES | YES | YES | YES | YES | YES | YES | YES | NO | YES | NOT_APPLICABLE | PARTIAL | NO | NO | `MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE`; accepted family closeout | Complete ACL/inheritance projection and LKW/live wiring are absent; permission limitation is intentional. | APPLICATION_INTEGRATION_GAP | LOW | Route through the next matrix; preserve accepted adapter scope and do not reopen Graph. |
| Microsoft Graph | `ms365_graph` / `collaboration_suite` | `mail` | ACCEPTED | YES | YES | YES | YES | YES | YES | YES | YES | YES | NO | NO | PARTIAL | NO | NO | `MSGRAPH-KNOWLEDGE-ADAPTERS-1B-MAIL`; accepted reconciliation finalization | Adapter preserves attachment presence, not attachment inventory/bytes; folder removal is not global mailbox deletion. | INTENTIONAL_NON_GOAL | LOW | Keep folder-scoped deletion and deferred attachment scope explicit in the matrix. |
| Microsoft Graph | `ms365_graph` / `collaboration_suite` | `teams_channel` | ACCEPTED | YES | YES | YES | YES | NO | YES | YES | YES | YES | NO | YES | PARTIAL | NO | NO | `MSGRAPH-KNOWLEDGE-ADAPTERS-1C-TEAMS-CHANNEL`; accepted family closeout | Only explicit `deletedDateTime` is a tombstone; no Graph delta, authoritative ACL or attachment bytes. | INTENTIONAL_NON_GOAL | LOW | Preserve explicit-deletion semantics; no implementation follow-up implied. |
| Microsoft Graph | `ms365_graph` / `collaboration_suite` | `teams_chat` | ACCEPTED | YES | YES | YES | YES | NO | YES | YES | YES | YES | NO | YES | PARTIAL | NO | NO | `MSGRAPH-KNOWLEDGE-ADAPTERS-1D-TEAMS-CHAT`; accepted family closeout | Fixed `lastModifiedDateTime` window; delta/absence deletion, ACL and attachment bytes are unsupported. | INTENTIONAL_NON_GOAL | LOW | Preserve fixed-window and explicit-deletion semantics. |
| Microsoft Graph | `ms365_graph` / `collaboration_suite` | `calendar` | ACCEPTED | YES | YES | YES | YES | PARTIAL | YES | PARTIAL | YES | YES | NO | YES | PARTIAL | NO | NO | `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` through accepted review correction | Primary calendar has delta/tombstones; non-primary calendars use snapshot reconciliation; ACL and attachment bytes remain absent. | INTENTIONAL_NON_GOAL | LOW | Preserve primary/non-primary split in the next matrix. |
| Slack | `slack` / `conversation_channel` | `slack_conversation` | IMPLEMENTED_UNREVIEWED | YES | YES | YES | YES | NO | YES | NO | YES | YES | NO | YES | PARTIAL | PARTIAL | NO | `SLACK-KNOWLEDGE-FOUNDATION-1`; `test_slack_conversation_knowledge_adapter.py`; `test_slack_conversation_knowledge_sync.py` | No removal tombstones; LKW connected-source recovery/indexed proof remains `CHANGES_REQUIRED`; live capability absent. | APPLICATION_INTEGRATION_GAP | MEDIUM | Complete the already-routed Slack LKW corrections before claiming a user indexed workflow; keep live as planned. |
| Google Workspace | `google_workspace` / `collaboration_suite` | `drive` | IMPLEMENTED_UNREVIEWED | YES | YES | YES | YES | YES | YES | YES | YES | NO | YES | NOT_APPLICABLE | PARTIAL | NO | NO | `test_google_workspace_drive_knowledge_adapter.py`; `test_google_workspace_drive_knowledge_sync.py` | Current roadmap incorrectly says no Google knowledge capability is implemented; application binding, indexed and live proof are absent. | DOCUMENTATION_GAP | MEDIUM | Reconcile roadmap status in the next matrix; then route application proof only if claimed. |
| Google Workspace | `google_workspace` / `collaboration_suite` | `docs` | IMPLEMENTED_UNREVIEWED | YES | YES | YES | YES | NO | YES | NO | YES | YES | NO | NOT_APPLICABLE | PARTIAL | NO | NO | `test_google_workspace_docs_knowledge_adapter.py`; `test_google_workspace_docs_knowledge_sync.py` | Exact known-document read and one-item reconciliation exist; no broad resource discovery, deletion semantics, application binding or three-mode proof; roadmap wording is stale. | DOCUMENTATION_GAP | MEDIUM | Reconcile roadmap wording and make source-selection limits explicit. |
| Google Workspace | `google_workspace` / `collaboration_suite` | `sheets` | IMPLEMENTED_UNREVIEWED | YES | YES | YES | YES | NO | YES | NO | YES | YES | NO | NOT_APPLICABLE | PARTIAL | NO | NO | `test_google_workspace_sheets_knowledge_adapter.py`; `test_google_workspace_sheets_knowledge_sync.py` | Exact known-spreadsheet read and one-item reconciliation exist; no broad resource discovery, deletion semantics, application binding or three-mode proof; roadmap wording is stale. | DOCUMENTATION_GAP | MEDIUM | Reconcile roadmap wording and make source-selection limits explicit. |
| Google Workspace | `google_workspace` / `collaboration_suite` | `slides` | PLANNED | PARTIAL | PARTIAL | NO | NO | NO | NO | NO | NO | NO | NO | UNPROVEN | NO | NO | NO | `GoogleWorkspaceSourceKind.SLIDES` and shared transport service root only | No source-specific read client, descriptor mapping, adapter, sync proof or application wiring. | IMPLEMENTATION_GAP | LOW | Keep planned; do not claim support until a source-specific read and adapter task is routed. |
| Google Workspace | `google_workspace` / `collaboration_suite` | `calendar` | PLANNED | PARTIAL | PARTIAL | NO | NO | NO | NO | NO | NO | NO | NO | UNPROVEN | NO | NO | NO | `GoogleWorkspaceSourceKind.CALENDAR` and shared transport service root only | No source-specific read client, descriptor mapping, adapter, sync proof or application wiring. | IMPLEMENTATION_GAP | LOW | Keep planned; do not infer parity with Microsoft Graph Calendar. |
| Google Workspace | `google_workspace` / `collaboration_suite` | `mail` | PLANNED | PARTIAL | PARTIAL | NO | NO | NO | NO | NO | NO | NO | NO | UNPROVEN | NO | NO | NO | `GoogleWorkspaceSourceKind.MAIL` and shared transport service root only | No Gmail read surface, adapter, sync proof or application wiring. | IMPLEMENTATION_GAP | LOW | Keep planned; route only after a precise Gmail contract is selected. |
| Google Workspace | `google_workspace` / `collaboration_suite` | `chat` | PLANNED | PARTIAL | PARTIAL | NO | NO | NO | NO | NO | NO | NO | NO | UNPROVEN | NO | NO | NO | `GoogleWorkspaceSourceKind.CHAT` and shared transport service root only | No Chat read surface, adapter, sync proof or application wiring. | IMPLEMENTATION_GAP | LOW | Keep planned; route only after a precise Chat contract is selected. |
| Jira | `jira` / `issue_tracker` | `issues` | IMPLEMENTED_UNREVIEWED | YES | YES | YES | YES | NO | YES | NO | YES | YES | NO | NO | PARTIAL | NO | NO | `JIRA-KNOWLEDGE-ADAPTER-1`; `test_jira_knowledge_adapter.py`; `test_jira_knowledge_sync.py` | Comments, attachments, ACL, deletion/revocation projection and incremental feed are deferred; no application/three-mode proof. | APPLICATION_INTEGRATION_GAP | LOW | Keep deferred capabilities explicit; assess application binding only through the planned matrix. |
| Confluence | `confluence` / `wiki_knowledge` | `pages` | IMPLEMENTED_UNREVIEWED | YES | YES | YES | YES | NO | YES | NO | YES | NO | NO | NO | PARTIAL | NO | NO | `CONFLUENCE-KNOWLEDGE-ADAPTER-1`; `test_confluence_knowledge_adapter.py`; `test_confluence_knowledge_sync.py` | Blog posts, attachments, comments, labels, ACL, deletion/revocation projection and incremental feed are deferred; no application/three-mode proof. | APPLICATION_INTEGRATION_GAP | LOW | Keep deferred capabilities explicit; assess application binding only through the planned matrix. |
| Databricks | no identity selected | `[none selected]` | PLANNED | NO | NO | NO | NO | NO | NO | NO | NO | NO | NO | UNPROVEN | NO | NO | NO | `DATABRICKS-KNOWLEDGE-ADAPTER-1` (`DEFERRED`) | The roadmap has not selected Unity Catalog metadata, workspace tree, volume files or an approved query snapshot. | ROADMAP_DECISION_REQUIRED | LOW | Select one exact source kind before designing or claiming an adapter. |

## 6. Gap classification

### Implementation gaps

- Google Workspace `slides`, `calendar`, `mail` and `chat` have only shared
  enum/transport support; source-specific read contracts and adapters are
  absent.
- No Databricks adapter can be scoped until the source-kind decision is made.

### Proof gaps

- Google `drive`, `docs` and `sheets` have focused adapter/sync tests, but no
  accepted roadmap closeout was found for those rows.
- Slack has strong platform proof, but the final LKW indexed Search/Ask and
  crash-safe connected-source proof is still under correction.
- No provider-specific evidence proves all seven layers for any family.

### Runtime wiring gaps

The adapter registry is intentionally explicit and instance-local, so lack of
an import-time global registry is not a defect. The remaining wiring gap is
above that boundary: provider-specific application source binding is not
proven for the non-LKW adapters, and the Slack LKW bridge is not accepted.

### Application integration gaps

The generic durable sync/sink path is reusable, but application lifecycle,
connected-source binding and provider-specific indexed ingestion are not
uniformly wired. Live capability registration/execution is not proven for any
row.

### Documentation gaps

1. The roadmap says Google Workspace runtime tasks remain planned and that no
   Google knowledge capability is implemented, while current production
   adapters and sync tests exist for `drive`, `docs` and `sheets`.
2. The roadmap describes low-level Microsoft Graph Mail attachment inventory
   and bounded ordinary file-attachment reads as implemented, while the
   Mail adapter matrix says attachment inventory and binary bytes remain
   deferred. This audit preserves the conservative adapter-level result:
   presence is not attachment inventory.

### Intentional non-goals

Permissions/authoritative ACLs, absence-based deletion where the provider
does not prove deletion, deferred attachments/comments/custom projections,
and all live-mode features are unsupported scope unless a source row says
otherwise. These are not implementation bugs or blockers by themselves.

### Roadmap decisions required

Databricks must select exactly one source kind before any adapter design.
The next formal roadmap route remains only
`VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1`.

## 7. Commercial blockers

No `BLOCKER` is proven: the repository does not currently claim that every
provider has complete connected-source, indexed and live workflows. The
highest practical risks are:

- **MEDIUM:** Slack's incomplete LKW connected-source/indexed proof.
- **MEDIUM:** stale Google Workspace status can cause an adopter to miss
  existing `drive`, `docs` and `sheets` capability.
- **LOW:** application binding and optional source expansion for the other
  families; Databricks source selection.

Accepted adapter-level unsupported scope is `NONE` impact when it is explicit
and does not contradict the claimed workflow.

## 8. Non-blocking intentional gaps

The following are safe to keep deferred when the public contract remains
honest: provider ACL projection, Graph attachment bytes/recursive attachment
expansion, Jira/Confluence comments and attachments, incremental feeds for
Jira/Confluence/Slack/Google Docs/Sheets, absence-based deletion without
provider evidence, and all live-access execution.

## 9. Documentation contradictions

The Google Workspace contradiction is actionable because the code and tests
prove three adapters while the roadmap denies implementation. It should be
resolved as a documentation/status correction, without changing provider code
in this task.

The Microsoft Graph Mail contradiction is narrower: low-level knowledge-read
capability and Vendor Knowledge adapter capability are different layers. The
roadmap should say so explicitly rather than using one sentence for both.
No Graph production correction is made here.

## 10. Recommended follow-up order

1. Correct the two status/wording contradictions above; avoid optimistic
   claims.
2. Complete the already-routed Slack connected-source corrections and obtain
   final indexed proof.
3. Run `VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1` using this matrix as
   its input, keeping adapter, durable, indexed and live columns separate.
4. Decide whether application binding is a supported product workflow for
   Jira, Confluence, Google and Microsoft Graph before creating provider
   implementation tasks.
5. Select one Databricks source kind; do not create hypothetical adapter rows.
6. Treat permissions, attachment expansion and optional source kinds as
   follow-ups only after their user-facing contract is approved.

Only the three-mode matrix is formally routed by this audit; the other items
are findings or candidate follow-ups, not newly activated roadmap tasks.

## 11. Entry criteria for `VENDOR-KNOWLEDGE-THREE-MODE-CAPABILITY-MATRIX-1`

The next task can start when:

- every row has an exact provider identity and exact runtime or explicitly
  planned `source_kind`;
- the Google three-adapter status contradiction is recorded and not silently
  treated as a missing implementation;
- Databricks remains a source-kind decision, not a hypothetical adapter;
- accepted Microsoft Graph rows remain unchanged and serve as the reference;
- Slack's platform foundation is not mistaken for final LKW proof;
- indexed, durable-materialization and live columns remain independent;
- every positive claim in the next matrix points to code, runtime registration,
  provider-specific tests or an accepted status record.

## 12. Evidence appendix

### Runtime and integration surfaces

- `intergrax/runtime/vendor_knowledge/registry.py` — explicit instance-local
  key `(provider_id, integration_kind, source_kind)` and adapter resolution.
- `intergrax/runtime/vendor_knowledge/adapters/__init__.py` — exported
  registration helpers for the 11 active adapters.
- `intergrax/runtime/vendor_knowledge/adapters/slack_conversation.py` —
  `slack_conversation`, structured records, bounded conversation inventory,
  exact reads and attachment metadata.
- `intergrax/runtime/vendor_knowledge/adapters/jira_issues.py` —
  `issues`, structured issue mapping, paging and reconciliation.
- `intergrax/runtime/vendor_knowledge/adapters/confluence_pages.py` —
  `pages`, rich-text mapping, paging and reconciliation.
- `intergrax/runtime/vendor_knowledge/adapters/google_workspace_drive.py`,
  `google_workspace_docs.py`, `google_workspace_sheets.py` — the three
  registered Google adapters and their declared capabilities.
- `intergrax/integrations/providers/collaboration_suite/google_workspace/contracts.py`
  — exact low-level Google source kinds: `drive`, `docs`, `sheets`, `slides`,
  `calendar`, `mail`, `chat`.
- `docs/plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md` — accepted Microsoft Graph
  registry keys, source matrices, Jira/Confluence deferred scope, Slack
  status, Google roadmap wording and Databricks deferral.

### Representative provider-specific proof

- Slack: `tests/unit/integrations/providers/conversation_channel/slack/test_knowledge_reads.py`,
  `tests/unit/runtime/vendor_knowledge/test_slack_conversation_knowledge_adapter.py`,
  `tests/unit/runtime/vendor_knowledge/test_slack_conversation_knowledge_sync.py`.
- Jira: `tests/unit/integrations/providers/issue_tracker/test_jira_knowledge_read.py`,
  `tests/unit/runtime/vendor_knowledge/test_jira_knowledge_adapter.py`,
  `tests/unit/runtime/vendor_knowledge/test_jira_knowledge_sync.py`.
- Confluence: `tests/unit/integrations/providers/wiki_knowledge/test_confluence_knowledge_read.py`,
  `tests/unit/runtime/vendor_knowledge/test_confluence_knowledge_adapter.py`,
  `tests/unit/runtime/vendor_knowledge/test_confluence_knowledge_sync.py`.
- Google Drive: `tests/unit/runtime/vendor_knowledge/test_google_workspace_drive_knowledge_adapter.py`,
  `tests/unit/runtime/vendor_knowledge/test_google_workspace_drive_knowledge_sync.py`.
- Google Docs: `tests/unit/runtime/vendor_knowledge/test_google_workspace_docs_knowledge_adapter.py`,
  `tests/unit/runtime/vendor_knowledge/test_google_workspace_docs_knowledge_sync.py`.
- Google Sheets: `tests/unit/runtime/vendor_knowledge/test_google_workspace_sheets_knowledge_adapter.py`,
  `tests/unit/runtime/vendor_knowledge/test_google_workspace_sheets_knowledge_sync.py`.
- Microsoft Graph accepted proof/status references are the exact roadmap
  entries `MSGRAPH-KNOWLEDGE-ADAPTERS-1A-DRIVE` through
  `MSGRAPH-KNOWLEDGE-ADAPTERS-1E-CALENDAR` and the accepted reconciliation
  finalization records linked in the canonical roadmap. Graph production code
  was not reopened for this audit.

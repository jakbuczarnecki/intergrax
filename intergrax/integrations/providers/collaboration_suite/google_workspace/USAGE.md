# Google Workspace (google_workspace)

Category: `collaboration_suite`

## Single public entrypoint

- **`GoogleWorkspaceCollaborationSuiteIntegration`** in `integration.py` is the only public provider class.
- Legacy catalog factories are compatibility shims delegating to `GoogleWorkspaceCollaborationSuiteIntegration`.
- Contract factory: `create_google_workspace_collaboration_suite_integration()`.

Do **not** create parallel public integrations such as `GoogleDriveIntegration`, `GoogleDocsIntegration`, `GoogleSheetsIntegration`, `GoogleCalendarIntegration`, `GoogleSlidesIntegration`, `GmailKnowledgeIntegration`, `GoogleChatKnowledgeIntegration` or `GoogleWorkspaceKnowledgeIntegration`.

## Current implementation honesty

Present today:

- public `GoogleWorkspaceCollaborationSuiteIntegration` shell
- `google_workspace` collaboration-suite manifest (`BETA`)
- legacy `CollaborationSuite` client delegation
- basic mail/calendar/directory-shaped public contract methods
- provider registration/catalog structure

Not implemented: production Google OAuth, Google API client construction, Drive inventory, Docs/Sheets/Slides content reads, Calendar/Gmail/Chat knowledge synchronization, Google Vendor Knowledge adapters, Google live capabilities, LKW Google Connected Sources, Google Search/Ask proof.

## Knowledge architecture (frozen - `GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1`)

Canonical architecture: [`docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../../../docs/project/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) §13.8.

Implementation plan: [`docs/project/maintainers/plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../../../docs/project/maintainers/plans/KNOWLEDGE_SOURCE_INTEGRATIONS.md) Phase 10.

One integration, seven planned source kinds (independent scope/cursor semantics each):

```text
(google_workspace, collaboration_suite, drive)
(google_workspace, collaboration_suite, docs)
(google_workspace, collaboration_suite, sheets)
(google_workspace, collaboration_suite, calendar)
(google_workspace, collaboration_suite, slides)
(google_workspace, collaboration_suite, mail)
(google_workspace, collaboration_suite, chat)
```

**Canonical durable resource ownership:**

Drive may discover all Drive-hosted resources; discovery does **not** determine durable `source_kind`. The platform derives canonical binding kind server-side. The frontend must not choose or override `source_kind`.

| Google resource class | Canonical durable `source_kind` |
|---|---|
| Google-native document (Docs) | `docs` |
| Google-native spreadsheet (Sheets) | `sheets` |
| Google-native presentation (Slides) | `slides` |
| Ordinary uploaded/stored file | `drive` |
| Drive folder / My Drive / Shared Drive scope | `drive` |
| Google Calendar / calendar-event scope | `calendar` |
| Gmail scope | `mail` |
| Google Chat space / conversation scope | `chat` |

**Drive discovery flow:**

```text
Drive inventory / discovery
→ inspect authoritative Google resource type
→ derive canonical target source_kind server-side
→ issue provider-neutral Remote Resource candidate
→ create only the canonical KnowledgeSourceBinding
```

**Stable resource identity:**

```text
provider_id = google_workspace
connection_ref
canonical Google resource type
stable Google resource ID
```

Rename/move (where Google preserves ID) do not change identity. Export/download URL is never identity. Revision/ETag/modified time/content hash are change state - not identity. The same native Google file must not become unrelated `drive` and `docs`/`sheets`/`slides` durable objects.

**Overlapping-binding policy (first proof):** explicit selected resources only; broad Drive/folder synchronization deferred. Future broad scopes require Option A (reject overlapping binding in same workspace) or Option B (canonical deduplication record) - Option B not chosen until Vendor Knowledge and LKW ownership models support it safely.

**Separation of concerns:**

```text
provider integration          → GoogleWorkspaceCollaborationSuiteIntegration
Vendor Knowledge adapters     → GoogleWorkspace*KnowledgeAdapter (thin, per source kind)
Live Capability adapters      → planned; separate from durable sync
LKW Connected Source          → generic Connected Source pattern (proved by Slack)
```

Drive read surface owns inventory, hierarchy, resource classification, folder/drive traversal, ordinary binary content and change-feed primitives. Docs, Sheets and Slides read surfaces own typed native content extraction, native structure and exact native content reads. A Drive adapter may call a shared typed native-content primitive internally only when the durable binding remains canonical and duplication is prevented.

**Foundation prerequisites (activation gates - not satisfied):** `GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1` becomes **ACCEPTED** (currently **READY_FOR_REVIEW**); Google Workspace runtime implementation starts only after `LKW-SLACK-KNOWLEDGE-PROOF-1` becomes **ACCEPTED** (currently **PLANNED**); canonical Tenant Connection / credential-reference boundary; `SecretsStore`-owned credentials; runtime integration rehydration/resolution; Vendor Knowledge binding/registry/sync contracts. Connection Catalog and rehydration owned by `LKW-KNOWLEDGE-ACCESS-1` - no second Connection system or Google-only credential store.

**Execution placement (vertically incremental):** Google Workspace runtime implementation starts only after `LKW-SLACK-KNOWLEDGE-PROOF-1` becomes **ACCEPTED** → Foundation → each read surface + matching adapter + contract proof (Drive → Docs → Sheets → Calendar) → LKW Connected Source → LKW proof → remaining surfaces (Slides, Mail, Chat) → other provider expansion.

**First LKW proof target:** one Google account; one Doc, one Sheet, one Calendar resource and optionally one ordinary Drive file synchronized; indexed Search/Ask with citations; no Google API calls during Ask after durable sync. First sources default to `PERSONAL_ONLY`.

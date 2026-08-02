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

## Knowledge architecture (frozen — `GOOGLE-WORKSPACE-KNOWLEDGE-ARCH-1`)

Canonical architecture: [`docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../../docs/architecture/KNOWLEDGE_SOURCE_INTEGRATIONS.md) §13.8.

Implementation plan: [`docs/plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md`](../../../../docs/plan/KNOWLEDGE_SOURCE_INTEGRATIONS.md) Phase 10.

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

**Separation of concerns:**

```text
provider integration          → GoogleWorkspaceCollaborationSuiteIntegration
Vendor Knowledge adapters     → GoogleWorkspace*KnowledgeAdapter (thin, per source kind)
Live Capability adapters      → planned; separate from durable sync
LKW Connected Source          → generic Connected Source pattern (proved by Slack)
```

Drive owns resource discovery and storage hierarchy. Docs, Sheets and Slides own typed native content reads. Drive may discover native Docs/Sheets/Slides items; content is read through the appropriate typed surface. Stable provider identity is separate from revision; download URLs are not durable identity.

**Execution placement:** complete accepted Slack Knowledge vertical → Google proof-critical path (Foundation → Drive/Docs/Sheets/Calendar read surfaces and adapters → LKW Connected Source → LKW proof) → remaining surfaces (Slides, Mail, Chat) → other provider expansion.

**First LKW proof target:** one Google account; one Doc, one Sheet, one Calendar resource and optionally one ordinary Drive file synchronized; indexed Search/Ask with citations; no Google API calls during Ask after durable sync. First sources default to `PERSONAL_ONLY`.

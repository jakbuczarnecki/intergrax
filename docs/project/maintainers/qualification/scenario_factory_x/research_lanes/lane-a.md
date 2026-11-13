> Evidence record committed per review R-W1-03; produced 2026-11-12 by a research subagent; partial page reads are stated in the matrix header.
# SFX W1 competitive research - Lane A (n8n, Zapier, Make)

Ticket: T-IGX-037-B. Dimensions D-01..D-14 per `R0_SECTIONS.md` §4. Sources: vendor primary docs only (docs/help centers, developer docs, official app catalogue pages). Access date for all sources: 2026-11-12.

Rubric: 0 = documented absence; 1 = basic/manual; 2 = solid documented feature; 3 = leading/automated; `UNKNOWN` = no primary evidence.

Cross-cutting note: none of the three vendors publishes an SLA or a measured time-to-first-success figure in the pages reviewed. D-12 scores reflect documented onboarding aids, not measured TTFS. SLA is not scored inside D-13.

---

## n8n

Positioning: a node-graph workflow automation platform (cloud or self-hosted) with AI agent nodes, a natural-language AI Workflow Builder and a chat-based n8n Assistant. Accessed 2026-11-12.

**Interaction model: hybrid (canvas + chat), with a code path.** Workflows are node graphs on a canvas. The AI Workflow Builder and n8n Assistant build and edit them from chat, and the output is a normal editable workflow. An MCP server also offers TypeScript Workflow SDK creation. Sources: https://docs.n8n.io/build/ways-of-building-workflows/ai-assistant ; https://docs.n8n.io/connect/connect-to-n8n-mcp-server/mcp-server-tools-reference

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 Business discovery | 1 | The Assistant plans the workflow before building and may ask for clarification. The docs describe no structured business discovery process. | https://docs.n8n.io/build/ways-of-building-workflows/ai-assistant |
| D-02 NL workflow generation | 3 | The AI Workflow Builder creates, refines and debugs workflows from natural language. It selects, places and configures nodes, then asks you to review credentials and parameters. | https://docs.n8n.io/build/ways-of-building-workflows/ai-workflow-builder |
| D-03 Configuration model | 2 | Workflows are nodes, parameters and connections. The REST API accepts that JSON, and the MCP server accepts validated TypeScript SDK code. | https://docs.n8n.io/connect/n8n-api/workflow |
| D-04 Integration catalogue & auth | 2 | "Hundreds" of nodes. For other APIs, the HTTP Request node uses generic credentials (Basic, Header, OAuth1/2). Credentials are per service and reusable. | https://docs.n8n.io/integrations/custom-operations/ |
| D-05 HITL | 3 | Per-tool human review for AI Agent tool calls: approve or block, routed via Chat, Slack, Teams, Gmail and others. The approver can differ from the chat user. | https://docs.n8n.io/advanced-ai/human-in-the-loop-tools/ |
| D-06 Secrets handling | 3 | External secret stores (Vault, AWS, Azure, GCP, 1Password, Infisical) are referenced as `$secrets.<vault>.<name>`. They resolve only in credential fields, can be project-scoped, and are Enterprise-only. | https://docs.n8n.io/administer/manage-credentials/use-external-secret-stores |
| D-07 Testing/evaluation | 3 | Dataset-driven light and metric-based evaluations: Correctness and Helpfulness (AI-scored), string similarity, tools used, custom metrics. There is also data pinning/mocking for development runs. | https://docs.n8n.io/advanced-ai/evaluations/metric-based-evaluations/ |
| D-08 Versioning/diff | 3 | Workflow history with restore and named versions. A visual diff compares two versions on the canvas. The MCP server has a versions-diff tool. Retention depends on the plan. | https://docs.n8n.io/workflows/history/ |
| D-09 Deployment/environments | 2 | Git-backed environments with push/pull and branch-per-environment. Promotion is one-way (dev to prod). There is no workflow conflict detection. It is an Enterprise feature. | https://docs.n8n.io/administer/use-source-control-and-environments/push-and-pull-changes |
| D-10 Run results/observability | 3 | An executions list plus "Debug in editor" that loads a failed run's data for re-run. Log streaming sends workflow, node and audit events to webhook, syslog or Sentry. | https://docs.n8n.io/build/understand-workflows/understand-executions/debug-executions |
| D-11 ROI/business reporting | 3 | Insights dashboard: production executions, failure rate, run time, time saved. Time saved can be fixed per workflow or dynamic via Time Saved nodes. API endpoint available. | https://docs.n8n.io/administer/observe-and-log/track-usage-with-insights |
| D-12 Usability/TTFS | 2 | Example prompts in the AI Workflow Builder and iterative "Execute and refine". No measured TTFS is published. | https://docs.n8n.io/build/ways-of-building-workflows/ai-workflow-builder |
| D-13 Production readiness | 3 | RBAC with custom project roles, SAML/OIDC/LDAP SSO, audit event log streaming, execution-data redaction, self-hosting option. No SLA confirmed. | https://docs.n8n.io/enterprise-hub |
| D-14 IDE/code-first parity | 3 | MCP server lets IDE agents validate, create, test (with pin data), execute, publish and diff workflows from TypeScript Workflow SDK code. | https://docs.n8n.io/connect/connect-to-n8n-mcp-server/mcp-server-tools-reference |

**Patterns worth borrowing**
- Credential-blind AI generation: the builder sends prompts, node definitions and the workflow to the LLM, never credential details. The user reviews credentials after generation. https://docs.n8n.io/build/ways-of-building-workflows/ai-workflow-builder
- Business value is declared inside the workflow: Time Saved nodes (or a fixed per-run estimate) count only on successful production runs. This feeds an Insights ROI-style dashboard. https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.timesaved
- Evaluation as a workflow branch: an Evaluation Trigger plus Set Metrics, with "check if evaluating" gating so metric cost runs only in tests. https://docs.n8n.io/advanced-ai/evaluations/metric-based-evaluations/
- One tool surface for UI and code: the same MCP tools (validate, create from code, test with pin data, versions diff) make IDE parity with the canvas possible. https://docs.n8n.io/connect/connect-to-n8n-mcp-server/mcp-server-tools-reference

**Limitations / anti-patterns**
- Git environments cannot detect workflow conflicts, and a pull overwrites local work. The docs ask users to treat Git as the source of truth and push immediately. https://docs.n8n.io/administer/use-source-control-and-environments/push-and-pull-changes
- Key governance features are plan-gated: external secrets, full history, environments and the Insights dashboard sit on Business or Enterprise. Plan statements conflict across pages. https://docs.n8n.io/administer/observe-and-log/track-usage-with-insights
- The n8n Assistant has only "limited" Enterprise readiness: partial RBAC/project scoping and partial environments/source-control support. https://docs.n8n.io/build/ways-of-building-workflows/ai-assistant

---

## Zapier

Positioning: a SaaS automation platform: Zaps, AI by Zapier steps, Agents (being migrated into AI by Zapier), Canvas for process mapping, Copilot for AI building, and early-access Next Gen Zaps. Accessed 2026-11-12.

**Interaction model: hybrid (chat + linear editor/canvas), plus a separate diagram canvas.** Copilot builds Zaps from plain-language chat. Canvas diagrams whole systems, including manual steps. Next Gen Zaps add a visual canvas, a TypeScript code view and a prompt box. Sources: https://help.zapier.com/hc/en-us/articles/23503999825421-Build-Zap-workflows-faster-using-AI-powered-Copilot-Beta ; https://help.zapier.com/hc/en-us/articles/49005047515021-Use-the-Workflow-Manager-to-build-and-view-your-Next-Gen-Zaps

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 Business discovery | 2 | Canvas turns a described goal into an AI system suggestion with manual and automated steps and "additional steps to consider". "Build it" then creates the assets. | https://help.zapier.com/hc/en-us/articles/19880280846221-Create-a-canvas-to-visualize-your-automated-system |
| D-02 NL workflow generation | 3 | Copilot builds Zaps from plain-language descriptions, recommends apps, triggers and actions, and refines through chat. It is in open beta. | https://help.zapier.com/hc/en-us/articles/23503999825421-Build-Zap-workflows-faster-using-AI-powered-Copilot-Beta |
| D-03 Configuration model | 2 | Trigger/action steps with field mapping. Next Gen Zaps show the same workflow as a visual diagram and as TypeScript code (early access). | https://help.zapier.com/hc/en-us/articles/49005047515021-Use-the-Workflow-Manager-to-build-and-view-your-Next-Gen-Zaps |
| D-04 Integration catalogue & auth | 3 | Agents and Zaps use actions from a library of 9,000+ apps through managed app connections. | https://help.zapier.com/hc/en-us/articles/24393442652557-Build-an-agent-in-Zapier-Agents |
| D-05 HITL | 3 | The Human in the Loop app has approval and data-collection steps: reviewer notifications (email or Slack), timeout, a decline action, and a Decision output for branching. It is a premium app. | https://help.zapier.com/hc/en-us/articles/38731463206029-Request-approval-to-keep-your-workflow-running-with-Human-in-the-Loop |
| D-06 Secrets handling | 2 | App credentials are stored encrypted in account-owned connections. API by Zapier injects credentials at runtime, and connections can be locked to approved hostnames. No external vault is documented. | https://help.zapier.com/hc/en-us/articles/36818633398157-App-connections-on-Zapier |
| D-07 Testing/evaluation | 1 | Step tests create test records. Action tests are live and change real apps. The trigger and Filter/Paths must be tested before publishing. No evaluation suite. | https://help.zapier.com/hc/en-us/articles/18811411817741-Test-Zap-steps |
| D-08 Versioning/diff | 2 | Drafts and versions with rollback through "Edit from this version". Old versions are immutable. Retention is 1 month (Pro) to 1 year (Enterprise). No diff view is documented. | https://help.zapier.com/hc/en-us/articles/14094586364941-Restore-your-Zap-to-a-prior-version-with-version-rollback |
| D-09 Deployment/environments | 1 | No staging environment is documented. Assets are copied or JSON-exported between accounts and arrive off. Connections must be re-authenticated and webhook URLs change. | https://help.zapier.com/hc/en-us/articles/8496275081357-Move-Zaps-between-accounts |
| D-10 Run results/observability | 3 | Zap history across the account. Early-access Agentic Management reviews runs and suggests or auto-applies tested fixes as new versions. | https://help.zapier.com/hc/en-us/articles/49007548829069-Agentic-Management-in-Next-Gen-Zaps |
| D-11 ROI/business reporting | 2 | The Enterprise Analytics dashboard includes an ROI calculator: time saved assumes 2 min per task (adjustable). It also shows top usage by Zap, app and member, and error rates. | https://help.zapier.com/hc/en-us/articles/25444544607373-Review-your-account-usage-in-the-analytics-dashboard |
| D-12 Usability/TTFS | 2 | Quick start offers pre-built Zap templates ("Try it") and plain-language Copilot guidance. No measured TTFS is published. | https://help.zapier.com/hc/en-us/articles/22234847450893-Zap-workflows-quick-start-guide |
| D-13 Production readiness | 3 | Owner, Super Admin, Admin and Member roles, plus asset roles, SAML SSO, SCIM, and an audit log with 12-month Enterprise retention. No SLA confirmed. | https://help.zapier.com/hc/en-us/articles/39698983334797-User-roles-and-permissions-in-Team-and-Enterprise-accounts |
| D-14 IDE/code-first parity | 2 | Next Gen Zaps code view (TypeScript, early access). The Platform CLI builds and tests integrations locally, but not workflows. No local workflow IDE is documented. | https://docs.zapier.com/integrations/build-cli/overview |

**Patterns worth borrowing**
- Discovery canvas mixing manual and automated steps: the AI proposes a system from a goal and suggests extra steps. Selected parts are then turned into real assets. https://help.zapier.com/hc/en-us/articles/19880280846221-Create-a-canvas-to-visualize-your-automated-system
- A declared "Workflow intent" field: the self-healing agent checks every proposed fix against it. Healing modes are Off, Suggest or Auto, and risky fix classes always need approval. https://help.zapier.com/hc/en-us/articles/49007548829069-Agentic-Management-in-Next-Gen-Zaps
- HITL as a typed step: a structured Decision output, a timeout policy (skip or end) and a decline policy (continue or stop) make approvals branchable and testable. https://help.zapier.com/hc/en-us/articles/38731463206029-Request-approval-to-keep-your-workflow-running-with-Human-in-the-Loop
- Publish locks the config, and edits happen in drafts: published agents and Zaps are immutable, and changes go through a draft and then a new version. https://help.zapier.com/hc/en-us/articles/42070243063053-Publish-and-manage-agent-versions

**Limitations / anti-patterns**
- Tests are not sandboxed: Next Gen test runs execute live actions against real connections. https://help.zapier.com/hc/en-us/articles/49005047515021-Use-the-Workflow-Manager-to-build-and-view-your-Next-Gen-Zaps
- Product-surface churn: Agents are being auto-converted into Zaps with one AI by Zapier step, with feature gaps (chat triggers, some knowledge sources). https://help.zapier.com/hc/en-us/articles/47402591569805-Migrating-from-Agents-to-AI-by-Zapier
- No environment promotion: moving between accounts drops connections and changes webhook URLs. https://help.zapier.com/hc/en-us/articles/8496275081357-Move-Zaps-between-accounts

---

> Correction R-W1-05: (1) The Zapier D-04 "9,000+ apps" figure is not supported by the cited page, which says only "thousands of apps"; read "thousands of apps". (2) Zapier Copilot is labelled "(Beta)" on the cited page, not "open beta" (D-02 row). Scores unchanged.

## Make

Positioning: a visual scenario-builder automation platform with Maia (a conversational AI builder, public beta), Make AI Agents (open beta), Make Grid (landscape map) and an MCP server. Accessed 2026-11-12.

**Interaction model: hybrid (canvas + chat).** Scenarios are module graphs in the Scenario Builder. Maia builds and modifies them from chat, with a Plan mode that previews the scenario before building. Source: https://help.make.com/introduction-to-maia-by-make

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 Business discovery | 1 | Maia Plan mode proposes a plan with a scenario preview (Reject, Revise, or Accept and build). Make Grid maps existing automations, not business problems. | https://help.make.com/introduction-to-maia-by-make |
| D-02 NL workflow generation | 3 | Maia builds and modifies scenarios and AI agents from prompts, troubleshoots using run history, and remembers user preferences. Public beta on paid plans. | https://help.make.com/introduction-to-maia-by-make |
| D-03 Configuration model | 2 | Scenarios serialize as JSON blueprints (modules, settings, mapped values). They can be exported, imported, and created or updated via the CLI and API. | https://help.make.com/blueprints |
| D-04 Integration catalogue & auth | 2 | Dedicated app modules handle auth automatically. A generic HTTP module supports OAuth 2.0 for other services, and custom apps define their own connections. App count unconfirmed in docs. | https://help.make.com/connect-to-any-web-service-using-oauth-20 |
| D-05 HITL | 2 | The Human in the Loop (Enterprise) app creates review requests (with a review URL). A trigger fires on approve, adjust or cancel, routable to Slack or email. | https://www.make.com/integrations/human-in-the-loop-enterprise |
| D-06 Secrets handling | 2 | Passwords are stored encrypted. Enterprise "credential requests" let others create connections without exposing secrets. Keys and certificates are managed under Credentials. | https://help.make.com/credential-requests |
| D-07 Testing/evaluation | 1 | Manual testing: "Run once" with a previous run's trigger data, agent chat tests (consume credits) and a Reasoning tab. No evaluation suite. | https://help.make.com/create-your-first-ai-agent |
| D-08 Versioning/diff | 2 | Version history keeps saved versions 60 days, with descriptions, author and timestamp, plus restore and crash recovery. No diff view is described. | https://help.make.com/restore-and-recover-scenario |
| D-09 Deployment/environments | 1 | No built-in promotion is documented. Blueprints move scenarios between teams, but connections do not carry over and connection IDs must be swapped. | https://help.make.com/blueprints |
| D-10 Run results/observability | 3 | Incomplete executions store failed runs, which are resolved from history or retried automatically for transient errors. Grid shows a live dependency map of scenarios and data stores. | https://help.make.com/manage-incomplete-executions |
| D-11 ROI/business reporting | 1 | The Enterprise Analytics dashboard ranks scenarios by operations used and usage change. Credit usage history is per run. No time-saved or ROI metric. | https://help.make.com/analytics-dashboard |
| D-12 Usability/TTFS | 2 | Maia is available on paid plans and as a 30-day trial on Free, building from a stated goal inside the Scenario Builder. No measured TTFS is published. | https://help.make.com/introduction-to-maia-by-make |
| D-13 Production readiness | 2 | Enterprise audit logs (12 months, org and team level), OIDC/SAML SSO, and organization and team roles with Enterprise custom roles. No SLA or SCIM confirmed. | https://help.make.com/audit-logs |
| D-14 IDE/code-first parity | 2 | Make CLI creates, updates and runs scenarios from blueprints. The cloud MCP server manages scenarios. A VS Code extension (Git-friendly) covers custom apps. | https://developers.make.com/make-cli/make-cli/make-cli-reference/scenarios |

**Patterns worth borrowing**
- A plan-then-build gate in the AI builder: Plan mode shows a scenario preview with Reject, Revise, or Accept and build before anything changes. https://help.make.com/introduction-to-maia-by-make
- Failed runs become first-class resumable objects: "incomplete executions" are fixed in place and re-run, with auto-retry for transient error classes only. https://help.make.com/manage-incomplete-executions
- Credential requests: a builder asks the credential owner to authorize a connection that the builder can use but never read. https://help.make.com/credential-requests
- An auto-generated estate map (Grid) shows which scenarios read or write a shared table before it changes. It respects per-user view permissions. https://help.make.com/make-grid-public-beta-is-now-live

**Limitations / anti-patterns**
- Maia cannot view other scenarios, reference their configurations, or connect to subscenarios or webhooks. Chat sessions are kept only 15 days. https://help.make.com/introduction-to-maia-by-make
- Restoring a version does not save it, and versions expire after 60 days. This is an easy way to lose a rollback. https://help.make.com/restore-and-recover-scenario
- Blueprint portability excludes connections, so promotion needs a scripted connection-ID swap. https://help.make.com/replace-connections-across-multiple-modules

---

## Lane A design-implication candidates

- CI-A-01: Credential-blind generation (the LLM sees schema and workflow, never secrets; credentials are reviewed after generation) → keeps the SFX Studio and IDE generator inside secret-handling rules → https://docs.n8n.io/build/ways-of-building-workflows/ai-workflow-builder
- CI-A-02: A discovery canvas that mixes manual and automated steps before generation → gives SFX a business-discovery artifact (D-01) that is separate from the executable scenario → https://help.zapier.com/hc/en-us/articles/19880280846221-Create-a-canvas-to-visualize-your-automated-system
- CI-A-03: A plan-then-build gate (Reject, Revise, Accept) on AI changes → an explicit operator approval point before scenario mutation, which fits ticket and plan approval discipline → https://help.make.com/introduction-to-maia-by-make
- CI-A-04: A declared "workflow intent" used as a guardrail for AI fixes, with Off, Suggest or Auto healing and mandatory approval for risky fix classes → the scenario spec can carry intent that validates repairs → https://help.zapier.com/hc/en-us/articles/49007548829069-Agentic-Management-in-Next-Gen-Zaps
- CI-A-05: Value instrumentation inside the scenario (fixed or dynamic time-saved, counted only on successful production runs) → native ROI reporting (D-11) without an external BI → https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.timesaved
- CI-A-06: Evaluations as a first-class scenario mode (dataset, metrics, "is evaluating" gating) → SFX testability with cheap production paths → https://docs.n8n.io/advanced-ai/evaluations/metric-based-evaluations/
- CI-A-07: Typed HITL steps with Decision output, timeout and decline policies, and per-tool approval → HITL becomes testable and branchable, not ad-hoc → https://help.zapier.com/hc/en-us/articles/38731463206029-Request-approval-to-keep-your-workflow-running-with-Human-in-the-Loop ; https://docs.n8n.io/advanced-ai/human-in-the-loop-tools/
- CI-A-08: Failed runs as resumable "incomplete executions", with retry only for transient error classes → bounded, operator-visible recovery semantics → https://help.make.com/manage-incomplete-executions
- CI-A-09: One tool surface (validate, create from code, test with pin data, diff, publish) shared by UI and IDE agents → Studio/IDE parity (D-14) by construction → https://docs.n8n.io/connect/connect-to-n8n-mcp-server/mcp-server-tools-reference
- CI-A-10: A visual version diff plus immutable published versions edited only through drafts → reviewable scenario changes (D-08) → https://docs.n8n.io/workflows/history/ ; https://help.zapier.com/hc/en-us/articles/42070243063053-Publish-and-manage-agent-versions
- CI-A-11 (anti-pattern): Live, unsandboxed test runs and environment moves that drop connections → SFX needs a sandboxed test mode and an environment-binding layer for connections → https://help.zapier.com/hc/en-us/articles/49005047515021-Use-the-Workflow-Manager-to-build-and-view-your-Next-Gen-Zaps ; https://help.make.com/blueprints
- CI-A-12: Credential requests, where the owner authorizes and the builder uses but cannot read → a delegated-connection pattern for multi-role SFX projects → https://help.make.com/credential-requests

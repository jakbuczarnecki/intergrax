> Evidence record committed per review R-W1-03; produced 2026-11-12 by a research subagent; partial page reads are stated in the matrix header.
# SFX W1 competitive research - Lane C

Scope: Dify, Flowise, Langflow, Relevance AI, Vellum, Gumloop, StackAI. Access date for every source: 2026-11-12.
Method: vendor primary docs, official changelogs/release pages and vendor-owned pages only. Evidence cells are paraphrases. Score key: 0 = documented absence (also used when the vendor's own docs index or onboarding path shows no such step), 1 = basic/manual, 2 = solid documented feature, 3 = leading/automated, UNKNOWN = no primary evidence.

## Activity / status check

| Product | Status | Primary evidence |
|---|---|---|
| Dify | Active; docs current at app-package format 1.19.x | https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md |
| Flowise | Active; acquired by Workday (Aug 2025); latest release flowise@3.1.4 | https://github.com/FlowiseAI/Flowise/releases , https://blog.flowiseai.com/workday-flowise |
| Langflow | Active; docs at 1.13.x with 1.14.x "Next" | https://docs.langflow.org/release-notes |
| Relevance AI | Active | https://relevanceai.com/docs/build/invent/invent.md |
| Vellum | PIVOTED. On 2026-05-07 vellum.ai relaunched as a personal AI assistant. The workflow-platform docs (docs.vellum.ai) are still online; last changelog entry Feb 2026. Scored below as the docs.vellum.ai platform; status for new buyers is uncertain. | https://www.vellum.ai/blog/introducing-vellum , https://docs.vellum.ai/changelog |
| Gumloop | Active but repositioned as "an AI agent platform"; current docs do not describe the earlier node canvas. Scored as the agent platform. | https://docs.gumloop.com/ |
| StackAI | Active; docs moved to docs.stackai.com (307 redirect from docs.stack-ai.com) | https://docs.stackai.com/llms.txt |

## Dify

Open-source platform for agents, agentic workflows and chatbots, published as web apps, APIs or MCP; cloud or self-hosted (accessed 2026-11-12).

**Interaction model**: canvas (node workflows/chatflows), with a conversational "Build mode" for agents only, so hybrid for agents. Source: https://docs.dify.ai/en/quick-start.md , https://docs.dify.ai/en/cloud/use-dify/build/new-agent/build.md

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 0 | Quick start goes straight to model-provider setup and a blank workflow; no business-discovery step is documented. | https://docs.dify.ai/en/quick-start.md |
| D-02 | 1 | Build mode lets an agent configure its own prompt, skills, files and env vars from chat; workflow generation from NL is not documented. | https://docs.dify.ai/en/cloud/use-dify/build/new-agent/build.md |
| D-03 | 2 | Canvas apps serialise to a YAML DSL (app.yaml inside a .ifpkg ZIP); draft vs published separation. | https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md |
| D-04 | 2 | Tool plugins with workspace credentials, overridable per app/node; MCP tools support OAuth dynamic client registration and per-caller header forwarding. | https://docs.dify.ai/en/cloud/use-dify/workspace/tools.md |
| D-05 | 3 | Human Input node: web-app or email delivery (no account needed), button-per-branch, required fields, configurable timeout branch (default 3 days). | https://docs.dify.ai/en/cloud/use-dify/nodes/human-input.md |
| D-06 | 2 | Workflow environment variables support a Secret type for API keys and passwords. | https://enterprise-docs.dify.ai/en/3.8.x/use/workflow/variables.md |
| D-07 | 1 | Single-step run, variable inspect, last-run data and CSV batch runs; no native evaluation/test-suite feature found. | https://docs.dify.ai/en/cloud/use-dify/debug/step-run.md |
| D-08 | 2 | Version history with naming, release notes and restore-to-draft; side-by-side diff is not described. | https://docs.dify.ai/en/cloud/use-dify/build/version-control.md |
| D-09 | 2 | Publish as web app, API or MCP server; Enterprise ships Helm, Terraform and air-gapped installs; no dev/staging/prod promotion described. | https://dify.ai/dify-enterprise |
| D-10 | 3 | Run history with per-node inputs/outputs, timing and tracing; native integrations for Langfuse, LangSmith, Opik, Arize, Phoenix and W&B Weave. | https://docs.dify.ai/en/cloud/use-dify/debug/history-and-logs.md |
| D-11 | 1 | Dashboard shows messages, active users, interactions per session and token usage; no ROI or satisfaction report. | https://docs.dify.ai/en/cloud/use-dify/monitor/analysis.md |
| D-12 | 2 | Official 30-minute quick start in four steps, building a nine-node workflow from blank. | https://docs.dify.ai/en/quick-start.md |
| D-13 | 3 | Enterprise: SAML/OIDC SSO, SCIM, workflow-level RBAC, tamper-evident audit logs to SIEM, multi-tenant quotas, Sev-1 1-hour response. | https://dify.ai/dify-enterprise |
| D-14 | 2 | difyctl CLI for terminals, scripts, CI and coding agents; YAML DSL; plugin SDK; Service API. | https://docs.dify.ai/en/cli/overview.md |

**Patterns worth borrowing**
- HITL node with multi-channel delivery (web app and email to non-users), action buttons as branches, and an explicit timeout branch. https://docs.dify.ai/en/cloud/use-dify/nodes/human-input.md
- Agent "Build mode" edits a read-only config panel through chat, then Apply/Discard, and writes a `build_note.md` as cross-session memory. https://docs.dify.ai/en/cloud/use-dify/build/new-agent/build.md
- Portable, versioned app package (ZIP plus YAML DSL) with explicit cross-version import rules. https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md

**Limitations / anti-patterns**
- Export can embed Secret-type values in plain text inside the package if the option is enabled. https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md
- difyctl sign-ins expire and need a browser to renew, which hampers unattended CI. https://docs.dify.ai/en/cli/overview.md
- No documented environment promotion or version diff. https://docs.dify.ai/en/cloud/use-dify/build/version-control.md

> Correction R-W1-05: (1) The Dify secret export is an optional toggle (the limitation above applies only when it is enabled). (2) "1.19.x" in the status table is the Dify application version, not an app-package format version. Scores unchanged.

## Flowise

Open-source visual builder for agents and LLM workflows (Assistant, Chatflow, Agentflow builders), now owned by Workday (accessed 2026-11-12).

**Interaction model**: canvas (Agentflow V2 node graph), plus a form-style Assistant builder. Source: https://docs.flowiseai.com/ , https://docs.flowiseai.com/using-flowise/agentflowv2.md

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 0 | Capability overview lists builders, tracing, evals and HITL; no business-discovery feature. | https://docs.flowiseai.com/ |
| D-02 | 2 | Release notes reference "Agentflow generation" validation (3.0.13) and a Generate button fix (3.1.1); official docs do not describe the feature. | https://github.com/FlowiseAI/Flowise/releases |
| D-03 | 2 | Node graph (LLM, Agent, Condition, Loop, Human Input, Execute Flow); static/runtime variables; per-request `overrideConfig`. | https://docs.flowiseai.com/using-flowise/variables.md |
| D-04 | 2 | LangChain and LlamaIndex node catalogues plus Zapier; credentials shareable across workspaces. | https://docs.flowiseai.com/integrations.md |
| D-05 | 2 | Human Input node approve/reject with optional feedback; tool-call approval on Agent tools; public trace link for outside approvers. | https://docs.flowiseai.com/tutorials/human-in-the-loop.md |
| D-06 | 2 | Credentials encrypted with a key stored locally or in AWS Secrets Manager; key can be pinned via env var. | https://docs.flowiseai.com/configuration/environment-variables.md |
| D-07 | 2 | Datasets plus text, numeric and LLM evaluators; re-run and compare across versions; Cloud/Enterprise only. | https://docs.flowiseai.com/using-flowise/evaluations.md |
| D-08 | 0 | Docs index has no flow version-history page; only migration guides and node-version upgrade notes. | https://docs.flowiseai.com/llms.txt |
| D-09 | 2 | Deployment guides and queue mode with workers; production page does not cover dev/staging/prod promotion. | https://docs.flowiseai.com/configuration/running-in-production.md |
| D-10 | 3 | Step tracing for Agentflow V2; Langfuse, LangSmith, Arize, Phoenix, Opik, Lunary, LangWatch; Prometheus/Grafana/OTel metrics. | https://docs.flowiseai.com/using-flowise/analytics.md |
| D-11 | 1 | Metrics cover API requests and flow/prediction counts; evaluation charts show tokens and latency; no business reporting. | https://docs.flowiseai.com/using-flowise/monitoring.md |
| D-12 | 2 | Three builders; Assistant is described as the simplest way to build an agent with tools and RAG. | https://docs.flowiseai.com/ |
| D-13 | 2 | Workspaces with granular custom roles and login-activity view (Cloud/Enterprise); SSO via Microsoft, Google, Auth0 (Enterprise). | https://docs.flowiseai.com/using-flowise/workspaces.md |
| D-14 | 1 | REST API with Chatflows CRUD and prediction; CLI mirrors GUI tasks; no SDK documented. | https://docs.flowiseai.com/api-reference.md |

**Patterns worth borrowing**
- Evaluation runs warn when a flow changed since the last run and support re-run plus comparison. https://docs.flowiseai.com/using-flowise/evaluations.md
- Approval checkpoint between an agent's tool decision and tool execution for side-effecting tools. https://docs.flowiseai.com/tutorials/human-in-the-loop.md
- Per-flow, pluggable tracing provider selected in flow settings and overridable per API call. https://docs.flowiseai.com/using-flowise/analytics.md

**Limitations / anti-patterns**
- Evaluations, workspaces and RBAC are gated to Cloud/Enterprise plans. https://docs.flowiseai.com/using-flowise/evaluations.md
- No documented flow version history or environment promotion. https://docs.flowiseai.com/configuration/running-in-production.md

> Correction R-W1-05: The Flowise acquisition by Workday (status table and section intro) is UNCONFIRMED: the cited blog URL (https://blog.flowiseai.com/workday-flowise) could not be loaded in the independent review (TLS failure from the fetch tool), so no verified source exists for it.

## Langflow

Open-source Python framework with a visual editor for agents, MCP and RAG flows; flows run headless via LFX (accessed 2026-11-12).

**Interaction model**: hybrid (canvas editor plus graph-aware Langflow Assistant chat pane). Source: https://docs.langflow.org/langflow-assistant

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 0 | Docs start from flows, components and templates; no business-discovery step documented. | https://docs.langflow.org/ |
| D-02 | 3 | Assistant builds whole flows or single components from natural-language prompts, using the open flow's graph as context. | https://docs.langflow.org/langflow-assistant |
| D-03 | 2 | Flows are fully serialisable; components are Python, including custom components. | https://docs.langflow.org/concepts-flows |
| D-04 | 2 | Component bundles, MCP client/server, and persistent Google Workspace, Microsoft 365 and Slack connections (1.13). | https://docs.langflow.org/release-notes |
| D-05 | 2 | Human Input component: custom actions as branches, stable action_id for API, fallback output on timeout; agent tool approval. | https://docs.langflow.org/human-input |
| D-06 | 2 | Global variables encrypted in DB; Credential type is masked; can be sourced from environment variables. | https://docs.langflow.org/configuration-global-variables |
| D-07 | 1 | Playground testing is manual and interactive; no built-in automated evaluation described. | https://docs.langflow.org/concepts-playground |
| D-08 | 2 | Version History: save, read-only preview, restore with draft backup, export or delete versions; no diff described. | https://docs.langflow.org/concepts-flows |
| D-09 | 2 | Docker, Kubernetes and cloud deployment guides; `lfx serve` endpoints; Flow DevOps SDK commands for validation and deployment. | https://docs.langflow.org/lfx-overview |
| D-10 | 3 | Built-in traces record span inputs, outputs, latency and tokens; Langfuse and Arize integrations documented. | https://docs.langflow.org/traces |
| D-11 | 1 | Only technical span metrics (tokens where available); cost is not mentioned. | https://docs.langflow.org/traces |
| D-12 | 2 | Prebuilt templates such as Basic Prompting and Vector Store RAG as starting points. | https://docs.langflow.org/concepts-flows |
| D-13 | 2 | 1.13 adds users/teams/roles via API/CLI and append-only audit log; RBAC needs an external authorization plugin. | https://docs.langflow.org/authentication-overview |
| D-14 | 2 | LFX runs flows headless (`lfx run`), serves APIs, scaffolds CI templates; authoring remains visual-first. | https://docs.langflow.org/lfx-overview |

**Patterns worth borrowing**
- Build in a visual editor, export JSON, execute with a stateless runtime that needs no DB or UI. https://docs.langflow.org/lfx-overview
- Assistant runs as a separate built-in flow with its own model; docs warn whole-flow builds cost more tokens. https://docs.langflow.org/langflow-assistant
- HITL actions get stable IDs derived from labels, so API clients resume deterministically. https://docs.langflow.org/human-input

**Limitations / anti-patterns**
- Without an authorization plugin every authenticated user has full access. https://docs.langflow.org/authentication-overview
- No automated evaluation; testing is manual Playground chat. https://docs.langflow.org/concepts-playground

## Relevance AI

AI workforce platform: agents, tools, triggers and multi-agent workforces, built with the Invent forward-deployed agent (accessed 2026-11-12).

**Interaction model**: hybrid (conversational Invent builder plus node-based Workforce canvas). Source: https://relevanceai.com/docs/build/invent/invent.md , https://relevanceai.com/docs/build/workforces/build-an-ai-workforce/introduction-to-nodes.md

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 3 | Invent works through the process with you: inputs, systems, judgment calls, exceptions, success criteria; reads it back and flags gaps. | https://relevanceai.com/docs/build/invent/invent.md |
| D-02 | 3 | From a description Invent generates agents, tools, triggers, workforces and evals, all in draft for review. | https://relevanceai.com/docs/build/invent/invent.md |
| D-03 | 2 | Assets are agents, tools, knowledge and workforces; per-edge behaviour settings configure approval and autonomy. | https://relevanceai.com/docs/build/workforces/workforce-features/approvals-and-escalations.md |
| D-04 | 2 | No hard limit on integrations, multiple accounts per service, API tool steps and webhooks for uncovered services. | https://relevanceai.com/docs/integrations/introduction.md |
| D-05 | 3 | Per-edge modes: auto run (capped), approval required, or let agent decide; escalations via Slack/email; Task View inbox. | https://relevanceai.com/docs/build/workforces/workforce-features/approvals-and-escalations.md |
| D-06 | 2 | API keys, certificates and service accounts stored encrypted (AES-256 at rest); org- and project-level API keys. | https://relevanceai.com/docs/admin/security.md |
| D-07 | 3 | Test sets with simulated users, reusable LLM Checks, minimum pass-rate publish gate, sampled production Monitor dashboards. | https://relevanceai.com/docs/build/agents/evals/introduction.md |
| D-08 | 2 | Every draft save and publish creates a version with author; restore lands in draft; no diff described. | https://relevanceai.com/docs/build/agents/build-your-agent/version-history.md |
| D-09 | 1 | Draft vs live published version only; no environment stages documented. | https://relevanceai.com/docs/build/agents/build-your-agent/version-history.md |
| D-10 | 2 | Analytics on tasks, tasks to review, actions, error rate per agent; Workforce Task View with history. | https://relevanceai.com/docs/enterprise/analytics.md |
| D-11 | 2 | Per-agent credits/task and task volume; docs suggest using them to show value; time saved and ROI not measured. | https://relevanceai.com/docs/enterprise/analytics.md |
| D-12 | 3 | Conversational Invent builds the full asset set in draft from a process description. | https://relevanceai.com/docs/build/invent/invent.md |
| D-13 | 3 | Org/project/asset-scoped RBAC (Enterprise), SSO, SOC 2 Type II, region choice (US/EU/AU), audit events. | https://relevanceai.com/docs/enterprise/rbac.md |
| D-14 | 1 | JS/TS SDK loads, messages and runs agents/workforces; defining agents in code is not covered. | https://relevanceai.com/docs/sdk/introduction.md |

**Patterns worth borrowing**
- Structured discovery interview (inputs, systems, judgment calls, exceptions, success criteria) before generation, then playback and gap detection. https://relevanceai.com/docs/build/invent/invent.md
- Evals as a publish gate: required test sets with minimum pass rate, optionally blocking publish. https://relevanceai.com/docs/build/agents/evals/introduction.md
- Graduated autonomy per edge, including a "let agent decide" mode with described escalation logic. https://relevanceai.com/docs/build/workforces/workforce-features/approvals-and-escalations.md
- Generator also produces the evals each agent must pass before launch. https://relevanceai.com/docs/build/invent/invent.md

**Limitations / anti-patterns**
- RBAC Enterprise-only and in gradual rollout. https://relevanceai.com/docs/enterprise/rbac.md
- No environments or code-first authoring; SDK is run-only. https://relevanceai.com/docs/sdk/introduction.md

## Vellum

Workflow, prompt and evaluation platform for AI products (docs.vellum.ai); company relaunched as a personal assistant on 2026-05-07, so platform status is uncertain (accessed 2026-11-12).

**Interaction model**: hybrid plus schema/code-first (visual Workflow editor, conversational Agent Builder, Python Workflows SDK with two-way sync). Source: https://docs.vellum.ai/developers/workflows-sdk/introduction.md

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 2 | Vellum for Agents: state a goal, answer follow-up questions, confirm the plan, then a first-draft agent is built. | https://www.vellum.ai/blog/introducing-vellum-for-agents |
| D-02 | 3 | Agent Builder proposes a plan from an outcome or builds your plan step by step into a Workflow. | https://docs.vellum.ai/product/agent-builder/agent-builder-sme.md |
| D-03 | 3 | Strictly typed graph defined in code or UI, kept in two-way sync. | https://docs.vellum.ai/developers/workflows-sdk/introduction.md |
| D-04 | 2 | 22 native integrations by Oct 2025, authenticated inside Vellum; Agent Builder prompts to connect them; Composio optional. | https://docs.vellum.ai/changelog/2025/2025-10 |
| D-05 | 1 | Human escalation documented only as a routing pattern to a separate output; no native pause/approval node. | https://docs.vellum.ai/product/workflows/common-architectures/human-escalation.md |
| D-06 | 2 | Workspace Secrets referenced by per-environment Environment Variables; per-environment API keys. | https://docs.vellum.ai/product/deployments/environments.md |
| D-07 | 3 | Test Suites with metrics (exact match, premade, custom) scored 0–1, plus Online Evaluations on every execution. | https://docs.vellum.ai/product/evaluation/quantitative-evaluation.md |
| D-08 | 3 | Release Reviews work like PR reviews; approval required before protected tags; SDK enables git-based code history. | https://docs.vellum.ai/product/deployments/release-reviews.md |
| D-09 | 3 | Development/Staging/Production Environments with own release history and keys; Promote copies a tested release forward. | https://docs.vellum.ai/product/deployments/environments.md |
| D-10 | 3 | Trace and Graph views per node, aggregated latency and cost, Actuals/feedback API, Datadog and webhooks. | https://docs.vellum.ai/product/deployments/observability.md |
| D-11 | 1 | Per-execution LLM cost totals; no business or ROI reporting described. | https://docs.vellum.ai/product/monitoring/execution-cost-tracking.md |
| D-12 | 2 | Conversational goal-to-draft agent flow, marketed as "Lovable, for agents". | https://www.vellum.ai/blog/introducing-vellum-for-agents |
| D-13 | 2 | Six predefined workspace roles; self-hosting including air-gapped; SSO and audit not in docs. | https://docs.vellum.ai/product/security/rbac-permissions.md |
| D-14 | 3 | Open-source SDK: code-first or UI-first, push/pull CLI, two-way sync. | https://docs.vellum.ai/developers/workflows-sdk/introduction.md |

**Patterns worth borrowing**
- Two-way code/UI sync (push/pull) so IDE and visual editor edit one artifact. https://docs.vellum.ai/developers/workflows-sdk/introduction.md
- Environment promotion of the same release, with per-environment secrets and API keys. https://docs.vellum.ai/product/deployments/environments.md
- Release Reviews gating protected tags such as Production. https://docs.vellum.ai/product/deployments/release-reviews.md

**Limitations / anti-patterns**
- Vendor pivoted to a personal assistant in May 2026; platform continuity is a vendor-risk signal. https://www.vellum.ai/blog/introducing-vellum
- Requested-changes feedback must be relayed outside Vellum; reviewer loop is not closed in-product. https://docs.vellum.ai/product/deployments/release-reviews.md
- No native HITL pause node. https://docs.vellum.ai/product/workflows/common-architectures/human-escalation.md

> Correction R-W1-05: The cited Vellum blog page says only that Vellum introduced a Personal Intelligence, dated 2026-05-07. The words "relaunched" / "PIVOTED" / "relaunched as a personal assistant" in the status table, in this section and in CI-C-17 are not supported and are withdrawn.

## Gumloop

AI agent platform: agents with connectors, skills, triggers and subagents, run via chat, Slack, Teams, email or API (accessed 2026-11-12).

**Interaction model**: conversational (guided agent setup and the Gumball agent that creates/updates agents). Source: https://docs.gumloop.com/core-concepts/agents.md

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 1 | Guided setup asks what a new agent should handle and configures itself from the answer. | https://docs.gumloop.com/core-concepts/agents.md |
| D-02 | 2 | Gumball creates/updates agents and schedule/webhook triggers from one request, bundling changes into a single approval card; no interview. | https://docs.gumloop.com/core-concepts/gumball.md |
| D-03 | 2 | Agent = connectors, skills, knowledge sources, triggers, subagents. | https://docs.gumloop.com/core-concepts/agents.md |
| D-04 | 3 | OAuth or API-key connectors with personal, team and agent-owned ownership; agent-owned users never receive the credential. | https://docs.gumloop.com/help/connectors/personal-vs-team-vs-agent-owned.md |
| D-05 | 3 | Per-app approval modes, CEL App Rules on tool arguments; approve via web, Slack, email or iMessage. | https://docs.gumloop.com/core-concepts/human_in_the_loop.md |
| D-06 | 3 | Encrypted personal/team Vault; secrets bound to env-var names; agent sees names, never values. | https://docs.gumloop.com/core-concepts/agent_sandbox_and_secrets.md |
| D-07 | 2 | AI evaluator grades transcripts against criteria (pass/flag/notify); no test cases or version comparison. | https://docs.gumloop.com/core-concepts/evaluations.md |
| D-08 | 1 | Agent versions listable and retrievable via API; CLI exports versions; no diff described. | https://docs.gumloop.com/cli/overview.md |
| D-09 | 1 | Agents automated via triggers, schedules and API; no environment stages documented. | https://docs.gumloop.com/ |
| D-10 | 2 | Each task keeps its own conversation, history, files and credit cost. | https://docs.gumloop.com/core-concepts/agents.md |
| D-11 | 2 | Enterprise Organization Insights covers adoption, credit spend and outcomes. | https://docs.gumloop.com/previews/enterprise-features/organization_insights.md |
| D-12 | 2 | New agents start with guided conversational setup, skippable in preferences. | https://docs.gumloop.com/core-concepts/agents.md |
| D-13 | 3 | SSO/SCIM, audit logging, custom roles per connector/tool/model, app policies, managed tunnels, static egress IPs. | https://docs.gumloop.com/enterprise-features/overview.md |
| D-14 | 2 | CLI lists/creates/updates agents, runs sessions and evals, `--json` for scripts; Python and JS SDKs. | https://docs.gumloop.com/cli/overview.md |

**Patterns worth borrowing**
- Three credential ownership modes (personal, team, agent-owned) for connectors. https://docs.gumloop.com/help/connectors/personal-vs-team-vs-agent-owned.md
- Argument-conditioned approval rules (CEL), e.g. approve only external-domain email. https://docs.gumloop.com/core-concepts/human_in_the_loop.md
- Builder agent cannot disable tool approvals and batches all changes into one approval card. https://docs.gumloop.com/core-concepts/gumball.md
- Secret binding by name so model context never carries values. https://docs.gumloop.com/core-concepts/agent_sandbox_and_secrets.md

**Limitations / anti-patterns**
- Docs do not guarantee sandbox code cannot leak a bound secret. https://docs.gumloop.com/core-concepts/agent_sandbox_and_secrets.md
- No test-case suites or version comparison in evaluations. https://docs.gumloop.com/core-concepts/evaluations.md

> Correction R-W1-04: The D-06 row (score 3) and the pattern "Secret binding by name so model context never carries values" overstate the cited page. The page says Gumloop provides secret names to the agent and supplies the values to code at runtime as environment variables, and states that this "is not a guarantee that code cannot expose a credential". Values therefore reach the agent code, and leakage is not excluded. The corrected D-06 score is 2 (see the matrix, section 5 D-06). CI-C-11 ("model sees names, never values") holds only for the model context, not for sandbox code.

## StackAI

Enterprise platform to turn manual processes into governed AI agents, deployable multi-tenant, VPC or on-prem (accessed 2026-11-12).

**Interaction model**: hybrid (canvas workflow builder plus Prompt-to-Flow / Ask AI generation and editing). Source: https://docs.stackai.com/guides-and-quickstarts/workflow-builder/ai-assisted-workflow-editing.md

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 1 | Entry is "describe the process you want"; no structured discovery interview documented. | https://docs.stackai.com/stackai-tools/workflow-and-catalog-tools/generate-stackai-workflow.md |
| D-02 | 3 | Describe a process, get a proposed workflow; Ask AI proposes node/edge edits with warnings, then review and apply. | https://docs.stackai.com/guides-and-quickstarts/workflow-builder/ai-assisted-workflow-editing.md |
| D-03 | 2 | Projects comprise workflow nodes, edges and an interface; env variables attach per stage. | https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/project-version-history-and-diffs.md |
| D-04 | 3 | Catalogue lists 514 provider entries with connection methods, actions and triggers; custom OAuth scopes supported. | https://docs.stackai.com/integrations/catalog.md |
| D-05 | 2 | Ask Human step pauses chat for input/choice; cannot answer in unattended schedules; deploy approvals per stage. | https://docs.stackai.com/stackai-tools/communication-and-human-input/ask-human-in-chat-interface.md |
| D-06 | 2 | One key with separate masked values for Development, Staging, Production; HashiCorp Vault integration listed. | https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/create-environment-variables.md |
| D-07 | 2 | Reviewed datasets, LLM-judge evaluators with pass/fail or custom labels, experiments, production quality monitors. | https://docs.stackai.com/guides-and-quickstarts/evaluations-and-observability/configure-an-evaluator.md |
| D-08 | 3 | "Review changes" diffs draft vs saved by Nodes, Edges and Interface; version preview and restore. | https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/project-version-history-and-diffs.md |
| D-09 | 3 | Stages each with own deployment; pull request then per-stage required approvals before deploy. | https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/deployments-and-approval-requirements.md |
| D-10 | 2 | Recorded runs show status, duration, input, output, errors with traceback, node-level values. | https://docs.stackai.com/guides-and-quickstarts/evaluations-and-observability/debug-a-recorded-run.md |
| D-11 | 1 | Org analytics shows runs and tokens (not billing); ROI appears only in marketing case studies. | https://docs.stackai.com/guides-and-quickstarts/evaluations-and-observability/organization-analytics.md |
| D-12 | 2 | First workflow in about seven steps; templates offered instead of blank canvas. | https://docs.stackai.com/guides-and-quickstarts/workflow-builder/first-input-to-output-workflow.md |
| D-13 | 3 | Custom roles and groups, SSO, audit logs (allow/deny) via OTel export; SOC 2 II, HIPAA, ISO 27001; on-prem. | https://docs.stackai.com/guides-and-quickstarts/organization-administration/audit-logs.md |
| D-14 | 1 | API runs published projects, KB, evals, SCIM; workflow authoring via API or SDK not documented. | https://docs.stackai.com/stackai-api/stackai-api-developer-guide.md |

**Patterns worth borrowing**
- Structural diff by Nodes / Edges / Interface before publish. https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/project-version-history-and-diffs.md
- Pull-request deploy with per-stage required-approval counts; previous version stays live until approved. https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/deployments-and-approval-requirements.md
- AI edit review checklist: check reference warnings, keep template fields as live references, run the full path. https://docs.stackai.com/guides-and-quickstarts/workflow-builder/ai-assisted-workflow-editing.md

**Limitations / anti-patterns**
- Ask Human cannot respond in unattended schedules; chat-only channel. https://docs.stackai.com/stackai-tools/communication-and-human-input/ask-human-in-chat-interface.md
- Masking hides values on screen only; revealable in UI. https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/create-environment-variables.md

## Lane C design-implication candidates

CI-C-01: Structured discovery interview (inputs, systems, judgment calls, exceptions, success criteria) with playback and gap detection before generation → SFX D-01 needs a business-to-scenario intake that precedes any config → https://relevanceai.com/docs/build/invent/invent.md
CI-C-02: Generator emits the acceptance evals alongside the scenario → a generated SFX scenario should ship with its own test set so "testable" is by construction → https://relevanceai.com/docs/build/invent/invent.md
CI-C-03: Eval pass-rate as a publish gate → SFX promotion should be blocked by failing scenario tests, not only by review → https://relevanceai.com/docs/build/agents/evals/introduction.md
CI-C-04: All generated changes land in draft and go through one review/apply card → SFX generation must never mutate a live scenario; diff then apply → https://docs.gumloop.com/core-concepts/gumball.md
CI-C-05: Structural diff by nodes / edges / interface → SFX Studio and IDE need a semantic scenario diff, not a raw text diff → https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/project-version-history-and-diffs.md
CI-C-06: Two-way code/UI sync over one artifact → SFX Studio and IDE toolkit must share one canonical scenario schema to reach parity (D-14) → https://docs.vellum.ai/developers/workflows-sdk/introduction.md
CI-C-07: Environment promotion of an identical release with per-environment secrets and keys → SFX sandbox (LAB) to PRODUCTION_ATTACHED promotion should copy config, rebind secrets → https://docs.vellum.ai/product/deployments/environments.md
CI-C-08: PR-style release review and per-stage required approvals → SFX production attach needs governed approval tied to policy → https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/deployments-and-approval-requirements.md
CI-C-09: HITL node with multi-channel delivery, button-as-branch and explicit timeout branch → SFX HITL contract should model timeout as a first-class path → https://docs.dify.ai/en/cloud/use-dify/nodes/human-input.md
CI-C-10: Argument-conditioned approval rules (CEL) and graduated autonomy modes → SFX tool policies should express "approve only when X", not on/off → https://docs.gumloop.com/core-concepts/human_in_the_loop.md
CI-C-11: Secrets bound by name; model sees names, never values → SFX config schema should reference secrets, never inline them → https://docs.gumloop.com/core-concepts/agent_sandbox_and_secrets.md
CI-C-12: Anti-pattern - exports that can embed secrets in plain text → SFX export must strip secret values unconditionally → https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md
CI-C-13: Credential ownership modes (personal / team / agent-owned) → SFX integration binding must state whose identity a scenario acts as → https://docs.gumloop.com/help/connectors/personal-vs-team-vs-agent-owned.md
CI-C-14: Stateless headless runtime that executes exported definitions (no UI or DB) → SFX CI and IDE runs need a headless executor over the same artifact → https://docs.langflow.org/lfx-overview
CI-C-15: Stable action IDs for HITL decisions → SFX resume APIs need deterministic decision identifiers → https://docs.langflow.org/human-input
CI-C-16: Gap - no lane-C vendor documents ROI or time-saved reporting; best is per-agent credits/task → SFX D-11 business reporting is a differentiation opportunity → https://relevanceai.com/docs/enterprise/analytics.md
CI-C-17: Vendor-continuity risk (Vellum pivot, Gumloop canvas removal, Flowise acquisition) → SFX should not depend on any lane-C product as a backend → https://www.vellum.ai/blog/introducing-vellum

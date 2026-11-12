> Evidence record committed per review R-W1-03; produced 2026-11-12 by a research subagent; partial page reads are stated in the matrix header.
# SFX W1 research - Lane B (vendor primary sources)

Ticket: T-IGX-037-B. Access date for every source: 2026-11-12. Dimensions D-01..D-14 follow R0 §4.
Scoring: 0 = documented absence, 1 = basic/manual, 2 = solid documented feature, 3 = leading/automated, UNKNOWN = no primary evidence found.

Name checks (primary sources):
- **Microsoft Copilot Studio**: the name is current. The Overview page is dated 2026-10-27 ([Learn](https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio)).
- **Retool**: Retool Workflows and Retool Agents are current ([Agents docs](https://docs.retool.com/agents), [Workflows docs](https://docs.retool.com/workflows)).
- **OpenAI Agent Builder / AgentKit**: OpenAI deprecated Agent Builder on 2026-06-03, and it **shuts down on 2026-11-30**. Evals became read-only on 2026-10-31 and shut down on 2026-11-30. ChatKit remains. OpenAI names the Agents SDK and ChatGPT Workspace Agents as replacements ([Deprecations](https://developers.openai.com/api/docs/deprecations)).
- **Google Vertex AI Agent Builder**: this is **now part of Gemini Enterprise Agent Platform** ("evolution of Vertex AI"). Agent Engine became **Agent Runtime** (release notes, 2026-04-22) ([name changes](https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes), [overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview)). ADK keeps its name.
- **AWS**: Amazon Bedrock Agents is still documented. **Amazon Bedrock AgentCore** is the current modular platform: Harness, Runtime, Gateway, Identity, Observability, Evaluations, Optimization, Policy, Registry and others ([AgentCore overview](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)).

---

## Microsoft Copilot Studio

Positioning: a graphical, low-code studio for agents, workflows and agent flows on Power Platform, with a choice of harness (GitHub Copilot, Customer Conversation, standard). Accessed 2026-11-12.

**Interaction model**: Hybrid. Users can describe a business goal in natural language (preview), and Copilot Studio proposes the agents and workflows. A drag-and-drop canvas and VS Code YAML editing are also available ([Overview](https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio)).

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | 2 | Guidance asks makers to define business value and metrics before building. It is guidance documentation, not an in-product discovery step. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/agent-business-value-define-value |
| D-02 | 3 | The user describes a business goal. Studio picks the mix of agents and workflows and refines it through follow-up conversation (preview). | https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio |
| D-03 | 2 | Topics, tools, triggers and workflows live as agent components. They can be cloned locally as YAML files with validation. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/visual-studio-code-extension-edit-agent-components |
| D-04 | 3 | Power Platform prebuilt (standard/premium) and custom connectors serve as tools, flow actions or knowledge. They default to end-user credentials, with an optional maker-credentials mode. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/advanced-connectors |
| D-05 | 3 | Agent flows include human-intervention actions such as approvals and requests for information. AI approvals can combine with human stages. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/flows-overview |
| D-06 | 2 | Secret-type environment variables reference Azure Key Vault. Each agent or environment is allowed through AllowedBots or AllowedEnvironments tags. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/kit-configure-secrets |
| D-07 | 3 | Single-response and conversational test sets can be manual, imported or AI-generated. Graders cover quality, keyword and tool use. Runs can be compared and are API-triggerable for CI/CD. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-agent-evaluation-rest-api |
| D-08 | 1 | Agents move as unmanaged solutions. The ALM guidance points to Azure DevOps/GitHub for source control. No native diff is documented. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/alm |
| D-09 | 3 | The recommended setup is dev, test and production environments. Power Platform pipelines automate solution deployment between environments. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/authoring-solutions-import-export |
| D-10 | 2 | Analytics monitor performance, custom metrics and session outcomes. A separate page covers monitoring of autonomous agents. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-improve-agent-health |
| D-11 | 3 | A savings calculator records per-run or per-tool time and money saved. It produces an ROI tile, computed against successful runs. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-cost-savings |
| D-12 | 2 | Users create an agent by describing it in plain language, then test it in a test pane before publishing. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio |
| D-13 | 3 | Purview audit telemetry covers invocations and tool calls. DLP data policies, role-based admin and an agent inventory are available, with a multitenant mode in preview. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/security-and-governance |
| D-14 | 2 | A VS Code extension clones an agent, edits its YAML with IntelliSense and syncs back. It works alongside GitHub Copilot or Claude Code. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/visual-studio-code-extension-overview |

**Patterns worth borrowing**
- Goal-first natural-language build that produces several assets (agents plus workflows) in one session, then refines them conversationally ([Overview](https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio)).
- An ROI calculator tied to real successful runs, with baselines the owner enters per run or per tool, recomputed after the fact when inputs change ([Savings](https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-cost-savings)).
- Test-case generation from the agent's own design and knowledge, plus a REST API so CI/CD can run evals ([Eval create](https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-agent-evaluation-create), [Eval API](https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-agent-evaluation-rest-api)).
- Round-trip between Studio and IDE: clone, edit YAML locally, sync back ([VS Code ext](https://learn.microsoft.com/en-us/microsoft-copilot-studio/visual-studio-code-extension-overview)).

**Limitations / anti-patterns**
- Solution export can miss components such as topics and knowledge stored separately. Imported copies do not track later source changes ([Import/export](https://learn.microsoft.com/en-us/microsoft-copilot-studio/authoring-solutions-import-export)).
- A maker-credentials mode can expose more data than an end user should see. When end-user credentials are enforced, autonomous or scheduled runs fail ([Maker credentials](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configure-no-maker-authentication)).
- Three harnesses and three flow formats (workflows, agent flows, topics) fragment the mental model, and features differ per harness ([Overview](https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio)).

---

## Retool (Workflows, Agents, AI/Assist)

Positioning: an internal-tools platform. Workflows handle scheduled and webhook automations. Agents "automate human work with AI" by combining deterministic code with LLM steps. Accessed 2026-11-12.

**Interaction model**: Hybrid. Workflows use a block-based canvas ("connect pre-built query blocks"). Agents are configured through a form with instructions and tools. Assist gives prompt-driven app generation ([Workflows](https://docs.retool.com/workflows), [Agents overview](https://docs.retool.com/agents/concepts/overview)).

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | UNKNOWN | No primary evidence of business-discovery tooling. | — |
| D-02 | 1 | Assist generates and edits classic apps from prompts. Its FAQ says it cannot yet operate on workflows or agents. | https://docs.retool.com/apps/guides/assist/ |
| D-03 | 2 | Agents are configured through form-based instructions, a model and tools. Workflows are built from resource, code, loop and response blocks. | https://docs.retool.com/agents/concepts/overview |
| D-04 | 2 | Native integrations plus REST/GraphQL/SOAP resources, with OAuth 2.0 (auth code, PKCE, client credentials). Agents can also use MCP-server tools. | https://docs.retool.com/data-sources/concepts/intro/ |
| D-05 | 2 | User Tasks add human checkpoints to workflows. Agent tools can require consent, but consent-gated tools are unavailable when the agent is called from a workflow. | https://docs.retool.com/workflows/reference/glossary |
| D-06 | 2 | Admin-only configuration variables can be secrets. Secrets are usable only in resources and workflows and are never exposed to the frontend. | https://docs.retool.com/org-users/guides/configuration/config-vars |
| D-07 | 2 | Agent datasets hold test cases scored by programmatic or LLM-as-judge reviewers. Two eval runs can be compared side by side. | https://docs.retool.com/agents/concepts/evals |
| D-08 | 3 | Git Source Control (GitHub, GitLab, Bitbucket, Azure Repos) covers apps, workflows and resources through PRs. Releases are stored as git tags. | https://docs.retool.com/source-control |
| D-09 | 2 | Multi-instance releases use a per-instance manifest (dev/uat/prod) that records the live version. Production can be read-only. Enterprise only. | https://docs.retool.com/source-control/guides/multi-instance-releases/apps |
| D-10 | 2 | Every workflow run is logged with per-block status, filters and JSON export. Datadog and Sentry are supported. Agent logs show reasoning and tool calls. | https://docs.retool.com/workflows/concepts/logs |
| D-11 | 1 | The agent monitor tracks runtime, tokens, estimated cost and run success. No business-value or hours-saved metric is documented. | https://docs.retool.com/agents/guides/monitor |
| D-12 | 2 | Agent templates (for example Meeting Prep), quickstarts and prompt-based app generation through Assist. | https://docs.retool.com/agents/concepts/overview |
| D-13 | 2 | Permission groups, audit logs (Business+), SSO role mapping and self-hosting. Source control and protected triggers need an Enterprise license. | https://docs.retool.com/permissions/reference/admin-permissions |
| D-14 | 1 | Code blocks (JS/Python) and Git-backed definitions exist. No code-first SDK for authoring agents or workflows is documented. | https://docs.retool.com/source-control/guides/protect/workflows |

**Patterns worth borrowing**
- A global error-handler block for workflows, with the run ID in context, so failures notify the right person ([Error handlers](https://docs.retool.com/workflows/guides/error-handlers)).
- Releases as git tags plus a per-environment manifest, so promotion only moves a pointer and nothing is rebuilt ([Multi-instance releases](https://docs.retool.com/source-control/guides/multi-instance-releases/apps)).
- Protected triggers: changes to a workflow's trigger go through review, with trigger configuration kept in a dedicated file ([Protect workflows](https://docs.retool.com/source-control/guides/protect/workflows)).
- Billable agent time excludes time spent waiting for human approval or auth, so HITL does not penalise cost ([Agents billing](https://docs.retool.com/support/billing-usage/agents)).

**Limitations / anti-patterns**
- AI generation covers apps only. Workflows and agents are still assembled by hand ([Assist](https://docs.retool.com/apps/guides/assist/)).
- Workflows report "success" when no error is thrown, so they cannot tell whether the result was correct. Users must add their own checks ([Error reporting](https://docs.retool.com/workflows/guides/observability/workflow-error-reporting)).
- Shared OAuth credentials for unattended runs make every user act as one account ([Intro to resources](https://docs.retool.com/data-sources/concepts/intro/)).

---

## OpenAI Agent Builder / AgentKit (DEPRECATED, shutdown 2026-11-30)

Positioning: AgentKit combined Agent Builder (a visual workflow canvas), ChatKit, connectors and Evals. On 2026-06-03 OpenAI deprecated Agent Builder, Evals and prompt objects. The successors are the Agents SDK (code) and ChatGPT Workspace Agents (natural language). Accessed 2026-11-12.

**Interaction model**: Canvas. It is "a visual canvas for building multi-step agent workflows," with typed nodes and edges ([Agent Builder](https://developers.openai.com/api/docs/guides/agent-builder)).

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | UNKNOWN | No primary evidence of business-discovery tooling. | — |
| D-02 | 1 | Users start from a template or a blank canvas. OpenAI points natural-language building to Workspace Agents, not Agent Builder. | https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder |
| D-03 | 2 | Nodes have typed inputs and outputs, and connections become typed edges. Start, Agent, Logic, Data and Tool node families. | https://developers.openai.com/api/docs/guides/node-reference |
| D-04 | 2 | MCP nodes connect through OpenAI connectors, third-party MCP servers or the user's own server (for example Gmail or Zapier). | https://developers.openai.com/api/docs/guides/node-reference |
| D-05 | 2 | A Human approval logic node defers to end users before a step, for example confirming before an email is sent. | https://developers.openai.com/api/docs/guides/node-reference |
| D-06 | UNKNOWN | No primary evidence of secrets handling inside Agent Builder workflows. | — |
| D-07 | 2 | Preview runs show each node's execution. Evaluate runs trace graders over selected traces. Evals are being shut down. | https://developers.openai.com/api/docs/guides/agent-builder |
| D-08 | 2 | Work autosaves. Publishing creates a major-version snapshot, and API calls can target older versions. No diff is documented. | https://developers.openai.com/api/docs/guides/agent-builder |
| D-09 | 1 | Deployment is by workflow ID in ChatKit or by exported SDK code. No environment model is documented. | https://developers.openai.com/api/docs/guides/agent-builder |
| D-10 | 1 | Node-level execution view in Preview. Traces are mentioned only as input to graders. | https://developers.openai.com/api/docs/guides/agent-builder |
| D-11 | UNKNOWN | No primary evidence of ROI or business reporting. | — |
| D-12 | 2 | Templates, a visual canvas and an interactive Preview with sample files support a fast first run. | https://developers.openai.com/api/docs/guides/agent-builder |
| D-13 | 0 | The product is deprecated and scheduled to shut down on 2026-11-30, so it is not production-viable. | https://developers.openai.com/api/docs/deprecations |
| D-14 | 2 | Code export gives Agents SDK TypeScript or Python. The export does not convert the graph or guarantee identical behavior. | https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder |

**Patterns worth borrowing**
- A small, typed node vocabulary (Core, Tool, Logic, Data) that includes guardrail and human-approval nodes ([Node reference](https://developers.openai.com/api/docs/guides/node-reference)).
- Guardrails as pass/fail nodes on the previous node's output, with a documented fallback (end, or retry with a reminder) ([Node reference](https://developers.openai.com/api/docs/guides/node-reference)).
- Publishing produces an immutable versioned workflow object that callers can pin by version ([Agent Builder](https://developers.openai.com/api/docs/guides/agent-builder)).

**Limitations / anti-patterns**
- Lock-in risk: a hosted-only canvas runtime was retired about 13 months after launch. Workflows hosted there lose their backend on 2026-11-30 ([Deprecations](https://developers.openai.com/api/docs/deprecations)).
- Export to code is lossy. Graph, connectors, auth, publishing and permissions must be rebuilt by hand ([Migrate](https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder)).

---

## Google Gemini Enterprise Agent Platform (formerly Vertex AI Agent Builder; Agent Engine → Agent Runtime; ADK)

Positioning: a unified platform to build, scale, govern and optimize enterprise agents, described as the evolution of Vertex AI. Its parts are ADK (open-source framework), Agent Studio (low-code canvas), Agent Runtime and governance services. Accessed 2026-11-12.

**Interaction model**: Hybrid. ADK is code-first (Python, TypeScript, Go, Java). Agent Studio is a low-code canvas with a code pane and "Convert to agent" from a prompt ([Overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview), [Agent Studio](https://docs.cloud.google.com/gemini-enterprise-agent-platform/agent-studio/agent-studio-capabilities)).

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | UNKNOWN | No primary evidence of business-discovery tooling. | — |
| D-02 | 2 | "Convert to agent" (preview) turns a prompt into an agent. Intent detection offers a canvas with generated instructions and tools attached. | https://docs.cloud.google.com/gemini-enterprise-agent-platform/agent-studio/agent-studio-capabilities |
| D-03 | 2 | ADK code defines agents. Agent Studio offers a canvas with a code view, and the Managed Agents API handles configuration. | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview |
| D-04 | 2 | The ADK Application Integration toolset exposes 100+ Integration Connectors (Salesforce, SAP, ServiceNow) with OAuth2. Agent Registry catalogues tools and MCP servers. | https://adk.dev/integrations/application-integration/ |
| D-05 | 2 | ADK tool confirmation (`require_confirmation`, or a predicate per call) pauses before a tool runs. Policy-based confirmation is also supported. | https://adk.dev/tools-custom/confirmation/ |
| D-06 | 2 | Agent Runtime injects Secret Manager secrets as environment variables. Access needs the secretAccessor role, and secrets must be in the same project. | https://docs.cloud.google.com/gemini-enterprise-agent-platform/troubleshooting/agent-deployment |
| D-07 | 3 | ADK test files and evalsets run via the `adk eval` CLI, pytest or the web UI. Scoring covers tool trajectory and LLM-judged rubrics, with user simulation. Multi-turn AutoRaters and online evaluation are also available. | https://adk.dev/evaluate/ |
| D-08 | UNKNOWN | No primary evidence of agent version history or diff in the platform. | — |
| D-09 | 2 | Deploy to Agent Runtime, Cloud Run or GKE, with VPC-SC perimeters and private networking. No staged environment model is documented. | https://docs.cloud.google.com/gemini-enterprise-agent-platform/scale/runtime/deploy-an-agent |
| D-10 | 2 | The Observability pillar includes a Unified Trace Viewer. The ADK web UI adds a Trace tab for each session. | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview |
| D-11 | UNKNOWN | No primary evidence of ROI or business reporting. | — |
| D-12 | 2 | Agent Garden supplies prebuilt agents and templates. Agent Studio has quickstarts and one-click conversion from a prompt to an agent. | https://docs.cloud.google.com/gemini-enterprise-agent-platform/agent-studio |
| D-13 | 3 | Per-agent managed Agent Identity, an Agent Gateway policy point, Model Armor, governance policies and VPC Service Controls. | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview |
| D-14 | 3 | Open-source ADK in Python, TypeScript, Go and Java runs locally and deploys to Runtime, Cloud Run or GKE. | https://cloud.google.com/vertex-ai/generative-ai/docs/agent-development-kit/overview |

**Patterns worth borrowing**
- Eval cases are captured from real sessions in the dev web UI ("save session as eval case"), then run from the CLI or pytest in CI ([ADK evaluate](https://adk.dev/evaluate/)).
- Two eval tiers: unit-style test files for fast checks, and multi-turn evalsets for integration, with user simulation for unscripted flows ([ADK evaluate](https://adk.dev/evaluate/)).
- Background intent detection that turns an ad-hoc prompt into a scaffolded agent with one click ([Agent Studio](https://docs.cloud.google.com/gemini-enterprise-agent-platform/agent-studio/agent-studio-capabilities)).
- A governance plane split into Registry, Identity and Gateway, which enforces policy on every agent tool call ([Overview](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview)).

**Limitations / anti-patterns**
- Frequent renames (Vertex AI Agent Builder, then Agent Platform; Agent Engine, then Agent Runtime, with the API resource still called ReasoningEngine) create doc and SDK churn ([Name changes](https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes)).
- Agent Gateway is not supported under VPC Service Controls. An agent deployed before its project joins a perimeter stays unprotected ([VPC-SC](https://docs.cloud.google.com/gemini-enterprise-agent-platform/machine-learning/general/vpc-service-controls)).
- User simulation cannot be combined with expected-tool or expected-response criteria ([ADK evaluate](https://adk.dev/evaluate/)).

---

> Correction R-W1-05: The cited rename page has no "Vertex AI Agent Builder" -> "Gemini Enterprise Agent Platform" row (the nearest row is "Vertex AI Platform" -> "Gemini Enterprise Agent Platform"). The "formerly Vertex AI Agent Builder" naming in this section and in the status note near the top is UNCONFIRMED. The Agent Engine -> Agent Runtime rename is on the page.

## AWS Amazon Bedrock Agents and Amazon Bedrock AgentCore

Positioning: Bedrock Agents is a managed, console-configured agent service with action groups. AgentCore is a modular platform for any framework or model: Harness, Runtime, Gateway, Identity, Observability, Evaluations, Optimization, Policy and Registry. Accessed 2026-11-12.

**Interaction model**: Schema-driven (harness.yaml and CLI, OpenAPI action groups), with a conversational builder for Bedrock Agents ([Harness](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html), [Conversational builder](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create-cb.html)).

| Dim | Score 0–3 | Evidence (≤25 words) | Source URL |
|---|---|---|---|
| D-01 | UNKNOWN | No primary evidence of business-discovery tooling. | — |
| D-02 | 2 | The Bedrock Agents conversational builder drafts instructions, action groups and knowledge-base links from chat. Its set of actions is limited. | https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create-cb.html |
| D-03 | 2 | The Harness declares model, prompt, tools, memory and limits as configuration (harness.yaml). Changing a model or tool is a config change, with no redeploy. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html |
| D-04 | 3 | Gateway converts APIs, Lambda and services into MCP tools (Salesforce, Jira, Slack). Identity works with Cognito, Okta, Entra and Auth0. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html |
| D-05 | 2 | Action-group functions support `requireConfirmation`. RETURN_CONTROL hands execution back to the calling application. | https://docs.aws.amazon.com/bedrock/latest/userguide/agents-returncontrol.html |
| D-06 | 3 | The Identity Token Vault stores OAuth tokens and client credentials, encrypted with KMS (customer-managed keys supported), with per-agent access control. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-overview.html |
| D-07 | 3 | AgentCore Evaluations scores sessions, traces and spans before and after deployment. Optimization adds A/B testing of configuration bundles. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html |
| D-08 | 2 | Each Runtime update creates an immutable, self-contained version. Bedrock Agents provides version snapshots and aliases. No diff is documented. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-runtime-versioning.html |
| D-09 | 2 | The DEFAULT endpoint tracks the latest version, while pinned endpoints (for example PROD) stay put until someone moves them. Bedrock aliases work the same way. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-runtime-versioning.html |
| D-10 | 3 | Observability gives step-level OTEL traces in CloudWatch for inspecting execution paths and intermediate outputs. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-view.html |
| D-11 | UNKNOWN | No primary evidence of ROI or business reporting. | — |
| D-12 | 2 | AWS says Harness GA lets users go "from idea to production-grade agent in minutes" with a few console clicks. | https://aws.amazon.com/blogs/machine-learning/amazon-bedrock-agentcore-harness-is-now-generally-available-go-from-idea-to-production-grade-agent-in-minutes/ |
| D-13 | 3 | microVM session isolation, IAM, CloudTrail data and management events, and Policy rules (Cedar-compatible) enforced on every Gateway tool call. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-security-best-practices.html |
| D-14 | 3 | Framework-agnostic: works with Strands, LangGraph, ADK and the OpenAI Agents SDK. Provides an AgentCore CLI (GA), CDK constructs and APIs. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/develop-agents.html |

**Patterns worth borrowing**
- Immutable versions plus movable named endpoints (DEFAULT follows the latest version, PROD is pinned). Promotion is one pointer move ([Runtime versioning](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-runtime-versioning.html)).
- A deterministic policy layer that intercepts every tool call, with rules written in natural language or a Cedar-compatible language ([Overview](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)).
- A per-agent credential vault for OAuth and API keys, so secrets never sit in agent configuration ([Identity](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-overview.html)).
- Trace-driven optimization: AI-generated configuration recommendations validated by A/B traffic splitting ([Overview](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)).

**Limitations / anti-patterns**
- Two overlapping AWS agent products (Bedrock Agents and AgentCore Harness/Runtime) split the concepts and the docs ([Harness vs Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html), [Bedrock Agents](https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create.html)).
- The CLI-created execution role lacks Token Vault permissions by default, so users must add the policy by hand ([Identity auth](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/identity-authentication.html)).
- CloudTrail data events for session endpoints are off by default and do not appear in Event history ([Session endpoints](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-session-endpoints.html)).

---

> Correction R-W1-05: The Identity Token Vault / KMS / per-agent access statement (D-06 row, patterns, CI-B-12) is on the `key-features-and-benefits` page (https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/key-features-and-benefits.html), section "Secure credential storage", not on the cited `identity-overview` page. Score unchanged.

## Lane B design-implication candidates

- CI-B-01: Goal-first natural-language intake that proposes the *set* of assets (agents plus workflows) for one business problem → maps directly to SFX "business problem → scenario"; avoids asking users to pick a designer first → https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio
- CI-B-02: An owner-entered baseline (time and cost per run or per tool) multiplied by successful runs gives a live ROI tile that recomputes after the fact → covers D-11, which only Microsoft scores here; a differentiator for SFX business reporting → https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-cost-savings
- CI-B-03: Value definition before build (metrics chosen up front) as guidance → SFX can turn it into an in-product discovery step (D-01 is UNKNOWN for 4 of 5 vendors) → https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/agent-business-value-define-value
- CI-B-04: Immutable scenario versions plus movable named endpoints/aliases (DEFAULT, PROD) → gives SFX promote and rollback as pointer moves, with no rebuild → https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-runtime-versioning.html
- CI-B-05: Releases stored as git tags, with a per-environment manifest of the live version → a Git-native environment model shared by the Studio and the IDE toolkit → https://docs.retool.com/source-control/guides/multi-instance-releases/apps
- CI-B-06: "Save session as eval case" in the dev UI, with the same cases runnable from CLI or pytest in CI → one test asset across Studio and IDE (D-07 and D-14 parity) → https://adk.dev/evaluate/
- CI-B-07: AI-generated test cases from the scenario's own design and knowledge, plus an eval API for CI → lowers the cost of a first test set and supports TTFS → https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-agent-evaluation-create
- CI-B-08: Studio ⇄ IDE round-trip on a YAML definition (clone, edit with validation, sync back) → the D-14 parity mechanism SFX needs, so neither surface is a dead end → https://learn.microsoft.com/en-us/microsoft-copilot-studio/visual-studio-code-extension-overview
- CI-B-09: Avoid hosted-only, lossy-export designs. The scenario definition must be the portable source of truth, not a graph tied to a vendor runtime → the OpenAI Agent Builder shutdown shows the risk → https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder
- CI-B-10: HITL and guardrails as first-class typed steps (approval node, pass/fail guardrail with defined fallback) → configurable HITL without custom code → https://developers.openai.com/api/docs/guides/node-reference
- CI-B-11: A deterministic policy layer that intercepts every tool call, separate from the agent's prompt → enforceable business rules and audit for production readiness (D-13) → https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html
- CI-B-12: A per-scenario credential vault with explicit per-agent allow-listing (Key Vault AllowedBots tags, AgentCore Token Vault) → secrets stay out of scenario configuration (D-06) → https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/kit-configure-secrets
- CI-B-13: An explicit choice between end-user and maker credentials, governed by admins → prevents privilege leaks in shared scenarios and surfaces unattended-run failures early → https://learn.microsoft.com/en-us/microsoft-copilot-studio/configure-no-maker-authentication
- CI-B-14: A global error handler plus result validation (success ≠ correct) → SFX run results should separate "executed" from "achieved business outcome" (D-10 and D-11) → https://docs.retool.com/workflows/guides/observability/workflow-error-reporting

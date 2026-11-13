# SFX competitive evidence matrix (SFX-P1-01)

Ticket: `governance/tickets/T-IGX-037-B.md` · WBS unit: SFX-P1-01 · Status: research record (no product claims about Intergrax).

## 1. Header

**Purpose.** Score 15 workflow and agent builders on the 14 SFX research dimensions (D-01..D-14, R0 §4). The scores and patterns turn into a register of design implications (CI-01..CI-32). Each CI says what SCENARIO-FACTORY-X (SFX) should adopt, adapt or reject, and which SFX requirements and WBS units it affects.

**Access date.** Every source was accessed on **2026-11-12**.

**Method.**
- Sources: vendor primary sources only. These are docs and help centres, developer docs, official changelogs and release pages, and vendor-owned pages. Evidence cells in the lane records are paraphrases. **Verification depth:** every scored cell carries a source URL in the lane records under `research_lanes/` (or is `UNKNOWN`). The lane authors did not re-fetch every cited page, and no web spot-checks were made across lanes (see §9). The URL is the claimed source, not proof that the full page was re-read. Corrections from the independent review are listed in "Review corrections" at the end of this document.
- Research ran in three lanes:
  - Lane A: n8n, Zapier, Make.
  - Lane B: Copilot Studio, Retool, OpenAI, Google, AWS.
  - Lane C: Dify, Flowise, Langflow, Relevance AI, Vellum, Gumloop, StackAI.
- This matrix consolidates the lane records. It adds no new research and no claim that is not in a lane record.
- Scoring rubric:
  - **0**: documented absence.
  - **1**: basic or manual.
  - **2**: solid documented feature.
  - **3**: leading or automated.
  - **UNKNOWN (U)**: no primary evidence found.

**Evidence-quality caveats (from the lanes).**

| # | Caveat | Affected cells |
|---|---|---|
| Q-1 | **Partial full-page reads in lane A.** Some lane A source pages were not read in full. Lane A also records two other limits: n8n plan-gating statements conflict across pages, and Make's app count is unconfirmed in its docs. | Lane A rows, mainly n8n D-06/D-08/D-09/D-11 (plan gating) and Make D-04 |
| Q-2 | **Inferred zeros in lane C.** Lane C gives 0 when the vendor's own docs index or onboarding path shows no such step. This is an inferred absence, not a vendor statement. Lanes A and B use `UNKNOWN` for the same situation, so the two conventions are not directly comparable (see §9). | Dify D-01, Flowise D-01, Flowise D-08, Langflow D-01 (marked `0*`) |
| Q-3 | **Beta, preview and early-access features.** These cells score features that the vendor labels beta, preview or early access. They may change or be withdrawn. | Zapier D-02 (Copilot, labelled "Beta"), D-03/D-10/D-14 (Next Gen Zaps, Agentic Management early access); Make D-01/D-02/D-12 (Maia public beta); Copilot Studio D-02 (goal-based build, preview); Google D-02 ("Convert to agent", preview); Gumloop D-11 (Organization Insights under a `previews/` docs path) (marked `β`) |
| Q-4 | **Blog-sourced cells.** These cells cite a vendor blog post, not product documentation. In lane B this is AWS D-12. In lane C it is Vellum, where **two** cells (D-01 and D-12) cite the same "Vellum for Agents" blog post. | AWS D-12, Vellum D-01, Vellum D-12 (marked `†`) |
| Q-5 | **Release-notes-only cell.** The score rests on GitHub release notes. The official docs do not describe the feature. | Flowise D-02 (marked `‡`) |
| Q-6 | **No SLA and no measured TTFS.** None of the lanes confirms a vendor SLA. Lane A leaves SLA out of D-13. Lane C cites Dify's Sev-1 1-hour support response under D-13. D-12 scores reflect documented onboarding aids, not a measured time to first success. | All D-12 cells; D-13 cells |

**Mapping assumption.** R0 §9 lists REQ-SFX-001..012 (the subsystems) and REQ-SFX-013..023 (the lifecycle steps) as ranges. This document maps them in list order:
- 001 Studio, 002 Discovery, 003 Specification, 004 Capability Resolution, 005 Blueprint, 006 Profile Composition, 007 Integration Setup, 008 SEF-X orchestration, 009 Testing/Evaluation, 010 Canonical execution, 011 Reporting/ROI, 012 Registry/Catalog/lifecycle/reuse;
- 013 Describe, 014 Discover, 015 Understand, 016 Design, 017 Configure, 018 Approve, 019 Build, 020 Test, 021 Run, 022 Present, 023 Improve.

Lane records (per-cell source record, committed per review R-W1-03): [`research_lanes/lane-a.md`](research_lanes/lane-a.md), [`lane-b.md`](research_lanes/lane-b.md), [`lane-c.md`](research_lanes/lane-c.md). Each scored cell, with its source URL, is in the lane record for the product's lane. Where the lane records and this matrix differ, the "Review corrections" section at the end of this document and the `> Correction` notes in the lane records apply.

## 2. Product status

| Product | Current name (as scored) | Status on 2026-11-12 | Source |
|---|---|---|---|
| n8n | n8n (AI Workflow Builder, n8n Assistant, MCP server) | Active. The n8n Assistant has only "limited" Enterprise readiness. | https://docs.n8n.io/build/ways-of-building-workflows/ai-assistant |
| Zapier | Zapier: Zaps, AI by Zapier, Canvas, Copilot, Next Gen Zaps | Active. Agents are being migrated into AI by Zapier, with feature gaps. Copilot is labelled "(Beta)". Next Gen Zaps are in early access. | https://help.zapier.com/hc/en-us/articles/47402591569805-Migrating-from-Agents-to-AI-by-Zapier |
| Make | Make: Scenario Builder, Maia, AI Agents, Grid, MCP server | Active. Maia is in public beta, AI Agents in open beta and Grid in public beta. | https://help.make.com/introduction-to-maia-by-make |
| Microsoft Copilot Studio | Copilot Studio | Active. The name is current; the Overview page is dated 2026-10-27. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio |
| Retool | Retool Workflows, Retool Agents, Assist | Active. Names are current. | https://docs.retool.com/agents |
| OpenAI | Agent Builder / AgentKit | **DEPRECATED on 2026-06-03; shuts down on 2026-11-30.** Evals became read-only on 2026-10-31 and shut down on 2026-11-30. ChatKit remains. OpenAI names the Agents SDK and ChatGPT Workspace Agents as successors. | https://developers.openai.com/api/docs/deprecations |
| Google | **Gemini Enterprise Agent Platform** (a "Vertex AI Agent Builder" -> "Gemini Enterprise Agent Platform" rename is **UNCONFIRMED**: the cited rename page has no row for Agent Builder; its nearest row is "Vertex AI Platform" -> "Gemini Enterprise Agent Platform") | Described as the "evolution of Vertex AI". Agent Engine became **Agent Runtime** (release notes, 2026-04-22). The API resource is still called ReasoningEngine. ADK keeps its name. | https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes |
| AWS | Amazon Bedrock Agents and **Amazon Bedrock AgentCore** | Bedrock Agents is still documented. AgentCore is the current modular platform. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html |
| Dify | Dify | Active. Docs are current at application version 1.19.x (this is the Dify app version, not a package-format version). | https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md |
| Flowise | Flowise | Active. Latest release is flowise@3.1.4. **UNCONFIRMED:** the claim that Flowise is owned by Workday (acquired Aug 2025). The cited blog URL could not be loaded in the review (TLS failure from the fetch tool), so the claim has no verified source. | https://github.com/FlowiseAI/Flowise/releases (release); blog URL `https://blog.flowiseai.com/workday-flowise` unverified |
| Langflow | Langflow | Active. Docs are at 1.13.x, with 1.14.x as "Next". | https://docs.langflow.org/release-notes |
| Relevance AI | Relevance AI (Invent, Workforces) | Active. | https://relevanceai.com/docs/build/invent/invent.md |
| Vellum | Vellum workflow platform (docs.vellum.ai) | **Vellum introduced "Personal Intelligence" on 2026-05-07** (the cited page says only this; the earlier "relaunch" wording is withdrawn). The platform docs are still online, and the last changelog entry is from Feb 2026. Status for new buyers is uncertain. | https://www.vellum.ai/blog/introducing-vellum , https://docs.vellum.ai/changelog |
| Gumloop | Gumloop agent platform | Active but **repositioned** as "an AI agent platform". The current docs do not describe the earlier node canvas. | https://docs.gumloop.com/ |
| StackAI | StackAI | Active. Docs moved to docs.stackai.com. | https://docs.stackai.com/llms.txt |

## 3. Interaction model

R0 §4 defines five interaction models: conversational, wizard, canvas, hybrid and schema-driven.

| Product | Model | Detail | Source |
|---|---|---|---|
| n8n | Hybrid (canvas + chat), with a code path | The AI Workflow Builder and Assistant build and edit node graphs from chat. The MCP server accepts TypeScript Workflow SDK code. | https://docs.n8n.io/build/ways-of-building-workflows/ai-assistant ; https://docs.n8n.io/connect/connect-to-n8n-mcp-server/mcp-server-tools-reference |
| Zapier | Hybrid (chat + linear editor or canvas), plus a separate diagram canvas | Copilot builds Zaps from chat. Canvas maps whole systems. Next Gen Zaps add a canvas, a TypeScript view and a prompt box. | https://help.zapier.com/hc/en-us/articles/23503999825421-Build-Zap-workflows-faster-using-AI-powered-Copilot-Beta ; https://help.zapier.com/hc/en-us/articles/49005047515021-Use-the-Workflow-Manager-to-build-and-view-your-Next-Gen-Zaps |
| Make | Hybrid (canvas + chat) | Maia builds scenarios from chat. Plan mode previews the scenario before building it. | https://help.make.com/introduction-to-maia-by-make |
| Copilot Studio | Hybrid | Natural-language goal intake (preview), a drag-and-drop canvas and VS Code YAML editing. | https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio |
| Retool | Hybrid | Workflows use a block canvas. Agents are configured through forms. Assist generates apps from prompts. | https://docs.retool.com/workflows ; https://docs.retool.com/agents/concepts/overview |
| OpenAI Agent Builder | Canvas | A visual canvas with typed nodes and edges. | https://developers.openai.com/api/docs/guides/agent-builder |
| Google Gemini Enterprise Agent Platform | Hybrid | ADK is code-first. Agent Studio is a low-code canvas with a code pane and "Convert to agent". | https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview ; https://docs.cloud.google.com/gemini-enterprise-agent-platform/agent-studio/agent-studio-capabilities |
| AWS Bedrock Agents / AgentCore | Schema-driven, plus a conversational builder for Bedrock Agents | harness.yaml and CLI, OpenAPI action groups. | https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html ; https://docs.aws.amazon.com/bedrock/latest/userguide/agents-create-cb.html |
| Dify | Canvas; hybrid for agents | Node workflows and chatflows. A conversational "Build mode" exists for agents only. | https://docs.dify.ai/en/quick-start.md ; https://docs.dify.ai/en/cloud/use-dify/build/new-agent/build.md |
| Flowise | Canvas, plus a form-style Assistant builder | Agentflow V2 node graph. | https://docs.flowiseai.com/using-flowise/agentflowv2.md |
| Langflow | Hybrid | Canvas editor plus a graph-aware Assistant chat pane. | https://docs.langflow.org/langflow-assistant |
| Relevance AI | Hybrid | Conversational Invent builder plus a node-based Workforce canvas. | https://relevanceai.com/docs/build/invent/invent.md |
| Vellum | Hybrid, plus schema/code-first | Visual editor, conversational Agent Builder, and a Python SDK with two-way sync. | https://docs.vellum.ai/developers/workflows-sdk/introduction.md |
| Gumloop | Conversational | Guided agent setup. The Gumball agent creates and updates agents. | https://docs.gumloop.com/core-concepts/agents.md |
| StackAI | Hybrid | Canvas workflow builder plus Prompt-to-Flow and Ask AI editing. | https://docs.stackai.com/guides-and-quickstarts/workflow-builder/ai-assisted-workflow-editing.md |

Ten of the 15 products are hybrid: n8n, Zapier, Make, Copilot Studio, Retool, Google, Langflow, Relevance AI, Vellum and StackAI. The other five are:
- Canvas-first: OpenAI, Flowise, and Dify (which is hybrid for agents only).
- Conversational: Gumloop.
- Schema-driven: AWS.

None of the products is a pure wizard.

## 4. Scoring matrix

**Legend.**
- Scores: 0–3 per the rubric in §1. `U` = UNKNOWN.
- `0*` = inferred zero (Q-2). `β` = beta, preview or early access (Q-3). `†` = blog-sourced (Q-4). `‡` = release-notes-only (Q-5).
- The OpenAI row describes a deprecated product. Its D-13 = 0 reflects the shutdown on 2026-11-30.
- Column keys:

  | Key | Dimension | Key | Dimension |
  |---|---|---|---|
  | D-01 | Business discovery | D-08 | Versioning and diff |
  | D-02 | Workflow generation from natural language | D-09 | Deployment and environments |
  | D-03 | Configuration model | D-10 | Run results and observability |
  | D-04 | Integration catalogue and auth | D-11 | ROI and business reporting |
  | D-05 | Human in the loop (HITL) | D-12 | Usability and time to first success (TTFS) |
  | D-06 | Secrets handling | D-13 | Production readiness |
  | D-07 | Testing and evaluation | D-14 | IDE and code-first parity |

<div style="overflow-x:auto">

| Product | D-01 | D-02 | D-03 | D-04 | D-05 | D-06 | D-07 | D-08 | D-09 | D-10 | D-11 | D-12 | D-13 | D-14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| n8n | 1 | 3 | 2 | 2 | 3 | 3 | 3 | 3 | 2 | 3 | 3 | 2 | 3 | 3 |
| Zapier | 2 | 3β | 2β | 3 | 3 | 2 | 1 | 2 | 1 | 3β | 2 | 2 | 3 | 2β |
| Make | 1β | 3β | 2 | 2 | 2 | 2 | 1 | 2 | 1 | 3 | 1 | 2β | 2 | 2 |
| Copilot Studio | 2 | 3β | 2 | 3 | 3 | 2 | 3 | 1 | 3 | 2 | 3 | 2 | 3 | 2 |
| Retool | U | 1 | 2 | 2 | 2 | 2 | 2 | 3 | 2 | 2 | 1 | 2 | 2 | 1 |
| OpenAI Agent Builder (deprecated) | U | 1 | 2 | 2 | 2 | U | 2 | 2 | 1 | 1 | U | 2 | 0 | 2 |
| Google Gemini Ent. Agent Platform | U | 2β | 2 | 2 | 2 | 2 | 3 | U | 2 | 2 | U | 2 | 3 | 3 |
| AWS Bedrock Agents / AgentCore | U | 2 | 2 | 3 | 2 | 3 | 3 | 2 | 2 | 3 | U | 2† | 3 | 3 |
| Dify | 0* | 1 | 2 | 2 | 3 | 2 | 1 | 2 | 2 | 3 | 1 | 2 | 3 | 2 |
| Flowise | 0* | 2‡ | 2 | 2 | 2 | 2 | 2 | 0* | 2 | 3 | 1 | 2 | 2 | 1 |
| Langflow | 0* | 3 | 2 | 2 | 2 | 2 | 1 | 2 | 2 | 3 | 1 | 2 | 2 | 2 |
| Relevance AI | 3 | 3 | 2 | 2 | 3 | 2 | 3 | 2 | 1 | 2 | 2 | 3 | 3 | 1 |
| Vellum | 2† | 3 | 3 | 2 | 1 | 2 | 3 | 3 | 3 | 3 | 1 | 2† | 2 | 3 |
| Gumloop | 1 | 2 | 2 | 3 | 3 | 2 | 2 | 1 | 1 | 2 | 2β | 2 | 3 | 2 |
| StackAI | 1 | 3 | 2 | 3 | 2 | 2 | 2 | 3 | 3 | 2 | 1 | 2 | 3 | 1 |

</div>

**Count of 3s per dimension:**

| D-01 | D-02 | D-03 | D-04 | D-05 | D-06 | D-07 | D-08 | D-09 | D-10 | D-11 | D-12 | D-13 | D-14 |
|---|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | 8 | 1 | 5 | 6 | 2 | 6 | 4 | 3 | 8 | 2 | 1 | 9 | 4 |

D-06 changed from 3 to 2 after review R-W1-04 (Gumloop re-scored 3 -> 2).

D-01, D-03, D-11 and D-12 each have at most two products scoring 3. D-01 also carries the most `U` and `0*` cells (seven), followed by D-11 (three `U`). See §7.

## 5. Per-dimension findings

### D-01 Business discovery
- **Best in class:** Relevance AI (3). Invent works through inputs, systems, judgment calls, exceptions and success criteria, then reads the result back and flags gaps. https://relevanceai.com/docs/build/invent/invent.md
- **Next (2):**
  - Zapier: Canvas proposes a system that mixes manual and automated steps. https://help.zapier.com/hc/en-us/articles/19880280846221-Create-a-canvas-to-visualize-your-automated-system
  - Copilot Studio: value-definition guidance, not an in-product step. https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/agent-business-value-define-value
  - Vellum†: goal, then follow-up questions, then plan confirmation. https://www.vellum.ai/blog/introducing-vellum-for-agents
- **Field:** four of the five lane B products are `U`. Three lane C products are `0*`.
- **SFX implication:** a structured, adaptive interview with playback, run before any configuration, is the clearest differentiator. It covers REQ-SFX-002, 014 and 015 and units SFX-P6-01 and P6-02. See CI-01, CI-03, CI-04 and CI-05.

### D-02 Workflow generation from natural language
- **Best in class (3):**
  - n8n: https://docs.n8n.io/build/ways-of-building-workflows/ai-workflow-builder
  - Zapier β: https://help.zapier.com/hc/en-us/articles/23503999825421-Build-Zap-workflows-faster-using-AI-powered-Copilot-Beta
  - Make β: https://help.make.com/introduction-to-maia-by-make
  - Copilot Studio β: https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio
  - Langflow: https://docs.langflow.org/langflow-assistant
  - Relevance AI: https://relevanceai.com/docs/build/invent/invent.md
  - Vellum: https://docs.vellum.ai/product/agent-builder/agent-builder-sme.md
  - StackAI: https://docs.stackai.com/guides-and-quickstarts/workflow-builder/ai-assisted-workflow-editing.md
- **Common traits of the leaders:**
  - Generated output lands in draft or behind a review: Make Plan mode, the Gumloop approval card, StackAI review-and-apply, Dify Apply/Discard.
  - n8n generates without seeing credentials.
- **SFX implication:** natural-language generation is table stakes. SFX must stand out on gated generation: plan, then approve, then build. Generation should also be credential-blind, and the generator should emit its own acceptance tests. This covers REQ-SFX-003, 005, 018 and 035. See CI-02, CI-07 and CI-12.

### D-03 Configuration model
- **Best in class:** Vellum (3). A strictly typed graph is defined in code or the UI, with two-way sync. https://docs.vellum.ai/developers/workflows-sdk/introduction.md
- **Typical (2):**
  - JSON or YAML serialisation: n8n JSON/TypeScript SDK, Make blueprints, the Dify YAML DSL, Copilot Studio YAML, AWS harness.yaml.
  - Intent declared as configuration: AWS Harness lets you change a model or tool as a configuration change, with no redeploy. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html
- **SFX implication:** one typed, canonical schema should be the source of truth. YAML and JSON are presentations of it (PC-1, PC-5). This covers REQ-SFX-003, 005 and 045 and units SFX-P2-02 and P5-02..04. See CI-21 and CI-22.

### D-04 Integration catalogue and auth
- **Best in class (3):**
  - Zapier, thousands of apps (the earlier "9,000+" figure is not supported by the cited page): https://help.zapier.com/hc/en-us/articles/24393442652557-Build-an-agent-in-Zapier-Agents
  - Copilot Studio connectors with end-user or maker credentials: https://learn.microsoft.com/en-us/microsoft-copilot-studio/advanced-connectors
  - AWS Gateway (APIs to MCP tools) plus Identity: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/gateway.html
  - Gumloop, with personal, team and agent-owned connectors: https://docs.gumloop.com/help/connectors/personal-vs-team-vs-agent-owned.md
  - StackAI, 514 catalogue entries: https://docs.stackai.com/integrations/catalog.md
- **SFX implication:**
  - SFX does not compete on catalogue size. C10/C15 means no new vendor integrations, and L2 runs use fixture adapters only.
  - The patterns to take are about whose identity a scenario acts as. Every integration plan should state its identity mode.
  - This covers REQ-SFX-007, 017 and 030 and unit SFX-P7-03. See CI-14 and CI-28.

### D-05 HITL
- **Best in class (3):**
  - n8n per-tool review: https://docs.n8n.io/advanced-ai/human-in-the-loop-tools/
  - Zapier typed Decision output with timeout and decline policies: https://help.zapier.com/hc/en-us/articles/38731463206029-Request-approval-to-keep-your-workflow-running-with-Human-in-the-Loop
  - Copilot Studio: https://learn.microsoft.com/en-us/microsoft-copilot-studio/flows-overview
  - Dify timeout branch and email delivery to non-users: https://docs.dify.ai/en/cloud/use-dify/nodes/human-input.md
  - Relevance per-edge modes: https://relevanceai.com/docs/build/workforces/workforce-features/approvals-and-escalations.md
  - Gumloop CEL rules on tool arguments: https://docs.gumloop.com/core-concepts/human_in_the_loop.md
- **Weak:**
  - Vellum (1) has no native pause node.
  - StackAI and Retool HITL are unavailable in unattended or workflow-invoked runs.
- **SFX implication:** model HITL as a typed step with three parts:
  - a Decision output;
  - a first-class timeout branch;
  - stable decision IDs.

  Approval rules should depend on tool arguments. Approvals should work in unattended runs through the approval inbox. This covers REQ-SFX-018, 028, 038 and 051. See CI-09, CI-10 and CI-30.

### D-06 Secrets handling
- **Best in class (3):**
  - n8n external vaults, resolved only in credential fields: https://docs.n8n.io/administer/manage-credentials/use-external-secret-stores
  - AWS Identity Token Vault with KMS and per-agent access: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/key-features-and-benefits.html (corrected per R-W1-05; the earlier cited page, `identity-overview`, does not carry this statement).
- **Good (2):**
  - Gumloop (re-scored 3 -> 2 per R-W1-04): Gumloop gives the agent secret names, and supplies the values to the agent's code at runtime as environment variables. The page states that this "is not a guarantee that code cannot expose a credential". So values do reach agent code and leakage is not excluded. Rationale for 2: an encrypted vault and name binding in model context are a solid documented feature, but not leading, because there is no leakage guarantee. https://docs.gumloop.com/core-concepts/agent_sandbox_and_secrets.md
- **Anti-patterns:**
  - Dify export can embed Secret-type values in plain text when the optional export toggle is enabled. https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md
  - StackAI masking is screen-only and can be revealed in the UI. https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/create-environment-variables.md
- **SFX implication:** secrets enter by reference only. The model sees names only. Export strips values unconditionally. The target is 0 secret values in artifacts (R0 §8). This covers REQ-SFX-030 and units SFX-P7-03 and P9-05. See CI-12, CI-13, CI-15 and CI-29.

### D-07 Testing and evaluation
- **Best in class (3):**
  - n8n dataset and metric evaluations: https://docs.n8n.io/advanced-ai/evaluations/metric-based-evaluations/
  - Copilot Studio AI-generated test sets plus a REST API for CI: https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-agent-evaluation-rest-api
  - Google ADK `adk eval` from CLI, pytest or web UI: https://adk.dev/evaluate/
  - AWS AgentCore Evaluations: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html
  - Relevance minimum pass-rate publish gate: https://relevanceai.com/docs/build/agents/evals/introduction.md
  - Vellum Test Suites plus online evaluations: https://docs.vellum.ai/product/evaluation/quantitative-evaluation.md
- **Weak (1):**
  - Zapier: tests are live and change real apps.
  - Make, Dify and Langflow: manual testing only.
- **SFX implication:**
  - Tests are generated with the scenario, run in an L2 sandbox and gate the next lifecycle state.
  - A Studio session can be saved as an eval case that also runs from the CLI.
  - Existing EvalRunner is reused.
  - This covers REQ-SFX-009, 020 and 045 and units SFX-P9-01 and P10-03. See CI-02, CI-19, CI-20 and CI-23.

### D-08 Versioning and diff
- **Best in class (3):**
  - n8n visual diff: https://docs.n8n.io/workflows/history/
  - Retool git source control: https://docs.retool.com/source-control
  - Vellum Release Reviews: https://docs.vellum.ai/product/deployments/release-reviews.md
  - StackAI structural diff by Nodes, Edges and Interface: https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/project-version-history-and-diffs.md
- **Common gap:** most products have version history but "no diff described". This applies to Zapier, Make, Dify, Langflow, Relevance, Gumloop, AWS and OpenAI.
- **SFX implication:** use immutable versions with a semantic diff, and make the Improve step open a new version with a diff. This covers REQ-SFX-023 and 043 and units SFX-P7-04 and P5-05. See CI-16, CI-17 and CI-25.

### D-09 Deployment and environments
- **Best in class (3):**
  - Copilot Studio dev/test/prod with pipelines: https://learn.microsoft.com/en-us/microsoft-copilot-studio/authoring-solutions-import-export
  - Vellum promotes an identical release with per-environment keys: https://docs.vellum.ai/product/deployments/environments.md
  - StackAI pull-request deploy with per-stage approvals: https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/deployments-and-approval-requirements.md
- **Also notable:** AWS pinned endpoints (DEFAULT/PROD) with immutable versions. https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-runtime-versioning.html
- **Weak (1):**
  - Zapier and Make: moves drop connections.
  - Relevance and Gumloop: no environment stages.
  - OpenAI: no environment model.
- **SFX implication:**
  - SFX stops at ADMISSION_DOSSIER_READY (R1 §7.9), and L3 is never granted.
  - The environment patterns apply only to LAB, the sandbox and the dossier. Production promotion is out of scope.
  - This covers REQ-SFX-006, 043 and 052 and unit SFX-P11-04. See CI-13, CI-16 and CI-18.

### D-10 Run results and observability
- **Best in class (3):**
  - n8n "Debug in editor": https://docs.n8n.io/build/understand-workflows/understand-executions/debug-executions
  - Zapier β Agentic Management: https://help.zapier.com/hc/en-us/articles/49007548829069-Agentic-Management-in-Next-Gen-Zaps
  - Make incomplete executions: https://help.make.com/manage-incomplete-executions
  - AWS OTEL traces: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/observability-view.html
  - Dify: https://docs.dify.ai/en/cloud/use-dify/debug/history-and-logs.md
  - Flowise: https://docs.flowiseai.com/using-flowise/analytics.md
  - Langflow: https://docs.langflow.org/traces
  - Vellum: https://docs.vellum.ai/product/deployments/observability.md
- **Anti-pattern:** Retool workflows report "success" when no error is thrown, whether or not the result is correct. https://docs.retool.com/workflows/guides/observability/workflow-error-reporting
- **SFX implication:**
  - Run reports should separate "executed" from "achieved the business outcome".
  - Failed runs become resumable objects with safe retry.
  - Facts come from the existing evidence plane.
  - This covers REQ-SFX-021, 032 and 040 and units SFX-P11-01 and P8-02. See CI-06 and CI-24.

### D-11 ROI and business reporting
- **Best in class (3):**
  - n8n Insights with Time Saved nodes, counted only on successful production runs: https://docs.n8n.io/administer/observe-and-log/track-usage-with-insights ; https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.timesaved
  - Copilot Studio savings calculator with owner-entered baselines: https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-cost-savings
- **Next (2):**
  - Zapier ROI calculator with a default of 2 min per task: https://help.zapier.com/hc/en-us/articles/25444544607373-Review-your-account-usage-in-the-analytics-dashboard
  - Relevance credits per task: https://relevanceai.com/docs/enterprise/analytics.md
  - Gumloop β Organization Insights: https://docs.gumloop.com/previews/enterprise-features/organization_insights.md
- **Field:**
  - Lane C has no ROI or time-saved measurement.
  - Google, AWS and OpenAI are `U`.
- **SFX implication:** declared value metrics are captured during discovery. A baseline-based ROI is computed only from declared inputs, and every number has a source reference. This covers REQ-SFX-011 and 022 and unit SFX-P11-02. See CI-05 and CI-06.

### D-12 Usability and TTFS
- **Best in class:** Relevance AI (3). Invent builds the full asset set in draft from a process description. https://relevanceai.com/docs/build/invent/invent.md
- **Others (2):** all other products score 2, based on templates, quickstarts or guided setup. For example, Dify documents a 30-minute quick start (https://docs.dify.ai/en/quick-start.md), and AWS† claims "minutes".
- **No vendor publishes a measured TTFS.**
- **SFX implication:** SFX can publish a measured TTFS for both the business user and the engineer (M-1), with run artifacts. This covers REQ-SFX-025 and 039 and units SFX-P3-01 and P12-05. See §7 G-01.

### D-13 Production readiness
- **Best in class (3):**
  - n8n: https://docs.n8n.io/enterprise-hub
  - Zapier: https://help.zapier.com/hc/en-us/articles/39698983334797-User-roles-and-permissions-in-Team-and-Enterprise-accounts
  - Copilot Studio: https://learn.microsoft.com/en-us/microsoft-copilot-studio/security-and-governance
  - Google: https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview
  - AWS (policy enforced on every Gateway tool call): https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/runtime-security-best-practices.html
  - Dify: https://dify.ai/dify-enterprise
  - Relevance: https://relevanceai.com/docs/enterprise/rbac.md
  - Gumloop: https://docs.gumloop.com/enterprise-features/overview.md
  - StackAI: https://docs.stackai.com/guides-and-quickstarts/organization-administration/audit-logs.md
- **Field:**
  - No SLA is confirmed for any vendor.
  - Governance features are often plan-gated (n8n, Flowise, Relevance).
- **SFX implication:**
  - Reuse the existing PolicyEngine, tenant and audit paths. Do not build new ones (REQ-SFX-040).
  - Safety controls are never tier-gated.
  - This covers REQ-SFX-027 and 028 and unit SFX-P9-05. See CI-11, CI-26 and CI-27.

### D-14 IDE and code-first parity
- **Best in class (3):**
  - n8n: one MCP tool surface to validate, create from code, test with pin data, diff and publish. https://docs.n8n.io/connect/connect-to-n8n-mcp-server/mcp-server-tools-reference
  - Google ADK in four languages: https://cloud.google.com/vertex-ai/generative-ai/docs/agent-development-kit/overview
  - AWS framework-agnostic CLI and CDK: https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/develop-agents.html
  - Vellum two-way sync: https://docs.vellum.ai/developers/workflows-sdk/introduction.md
- **Weak (1):**
  - Relevance: the SDK can only run agents.
  - StackAI: the API covers runs only.
  - Retool and Flowise: no authoring SDK.
- **SFX implication:** IDE parity should come from construction:
  - one schema;
  - one operation set across Studio, CLI and MCP;
  - a headless executor over the same artifact.

  This covers REQ-SFX-026 and 045 (PC-1..6) and units SFX-P5-07, P10-02, P10-03 and P10-08. See CI-20 and CI-21.

## 6. Consolidated design-implications register

Decision column: **Adopt** = take the pattern as is. **Adapt** = take it with an SFX-specific constraint. **Reject** = do not take it (anti-pattern or out of scope).

| CI | From | Pattern | Products (sources) | SFX requirement(s) | SFX task(s) | Adopt / Adapt / Reject | Rationale |
|---|---|---|---|---|---|---|---|
| CI-01 | CI-C-01 | Structured discovery interview: inputs, systems, judgment calls, exceptions and success criteria, followed by playback and gap detection before generation. | Relevance AI ([Invent](https://relevanceai.com/docs/build/invent/invent.md)) | REQ-SFX-002, 014, 015 | SFX-P6-01, P6-02, P4-A01 (`sfx-discover`) | **Adopt** | Only one vendor scores 3 on D-01. It matches the R0 §8 adaptive interview and the "Understanding" summary, and it feeds M-2 (discovery efficiency). |
| CI-02 | CI-C-02, CI-C-03; Flowise eval-staleness pattern | The generator emits acceptance evals alongside the scenario. An eval pass rate acts as a publish gate. Evals are flagged stale when the flow changed since the last run. | Relevance AI ([Invent](https://relevanceai.com/docs/build/invent/invent.md), [Evals](https://relevanceai.com/docs/build/agents/evals/introduction.md)); Flowise ([Evaluations](https://docs.flowiseai.com/using-flowise/evaluations.md)) | REQ-SFX-009, 020, 035, 036..038 | SFX-P9-01, P4-A01 (`sfx-test-plan`), lifecycle gate BUILT→SANDBOX_TESTED (§7.9) | **Adopt** | Makes the scenario testable by construction. Staleness is checked by `canonical_digest` (PC-5), so a gate result applies only to the version it tested. |
| CI-03 | CI-B-01 | Goal-first intake that proposes the whole set of assets for one business problem, without asking the user to pick a designer first. | Copilot Studio ([Overview](https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio)) | REQ-SFX-013, 005 | SFX-P6-01, P7-01, P7-02 | **Adapt** | SFX goes from Describe to a specification and then to a blueprint. It does not emit assets directly. The asset set is resolved deterministically from the specification (P7-01). |
| CI-04 | CI-A-02 | Discovery canvas that mixes manual and automated steps and suggests extra steps, kept separate from the executable artifact. | Zapier ([Canvas](https://help.zapier.com/hc/en-us/articles/19880280846221-Create-a-canvas-to-visualize-your-automated-system)) | REQ-SFX-015, 016 | SFX-P6-02, P7-02, P10-05 | **Adapt** | Manual steps belong in the specification and appear on the read-only blueprint canvas. They are edited through forms, not free wiring (R0 §8). |
| CI-05 | CI-A-05, CI-B-02, CI-B-03 | Declared value metrics and a baseline ROI. Owner-entered baselines (time and cost per run or tool) are multiplied by successful runs. Value is defined before the build. | n8n ([Time Saved](https://docs.n8n.io/integrations/builtin/core-nodes/n8n-nodes-base.timesaved), [Insights](https://docs.n8n.io/administer/observe-and-log/track-usage-with-insights)); Copilot Studio ([Savings](https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-cost-savings), [Define value](https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/agent-business-value-define-value)) | REQ-SFX-011, 014, 022 | **SFX-P11-02**, P6-01, P11-01 | **Adopt** | Matches P11-02: "inputs declared, no fabricated baselines; every number has a source ref". Value metrics are captured in Discover. SFX reports sandbox-run estimates only, because production runs are outside SFX. |
| CI-06 | CI-B-14 | A global error handler plus result validation. "Executed" is not the same as "achieved the business outcome". | Retool ([Error handlers](https://docs.retool.com/workflows/guides/error-handlers), [Error reporting](https://docs.retool.com/workflows/guides/observability/workflow-error-reporting)) | REQ-SFX-011, 021 | SFX-P11-01, P9-01 | **Adopt** | Run reports need an outcome assertion tied to the scenario's success criteria (CI-01). Without it, ROI counts successful but wrong runs. |
| CI-07 | CI-A-03, CI-C-04; Dify Build mode; Gumloop builder | Plan-then-build gate (Reject, Revise, or Accept and build). All AI-generated changes land in draft as one review/apply card. The builder agent cannot disable approvals. | Make ([Maia Plan mode](https://help.make.com/introduction-to-maia-by-make)); Gumloop ([Gumball](https://docs.gumloop.com/core-concepts/gumball.md)); Dify ([Build mode](https://docs.dify.ai/en/cloud/use-dify/build/new-agent/build.md)) | REQ-SFX-018, 005, 035, 051 | Approval FSM BLUEPRINTED→APPROVED (§7.9); SFX-P5-03 (`approval.py`), P5-05 (`decisions.py`), P9-02, P10-06 | **Adopt** | Fits the decision-binding transition and the ticket/plan approval discipline. Generation never changes an approved version. "Cannot disable approvals" is a no-self-escalation rule (P9-02). |
| CI-08 | CI-A-04 | Intent-guarded self-healing. A declared "workflow intent" validates every proposed fix. Modes are Off, Suggest and Auto. Risky fix classes always need approval. | Zapier ([Agentic Management](https://help.zapier.com/hc/en-us/articles/49007548829069-Agentic-Management-in-Next-Gen-Zaps)) β | REQ-SFX-003, 023, 051 | SFX-P6-02 (intent in the specification), P7-04, P10-06 (Improve) | **Adopt** | The intent field belongs in the specification contract. In SFX, "Auto" can only open a new blueprint version (SANDBOX_TESTED→BLUEPRINTED). It never moves a version past APPROVED without a decision binding. The source feature is early access. |
| CI-09 | CI-A-07, CI-B-10, CI-C-09, CI-C-15 | Typed HITL step. It has a Decision output, a **timeout branch** and a decline policy. It delivers on several channels, including to non-users, and uses stable action IDs so that resume is deterministic. | Zapier ([HITL](https://help.zapier.com/hc/en-us/articles/38731463206029-Request-approval-to-keep-your-workflow-running-with-Human-in-the-Loop)); Dify ([Human Input](https://docs.dify.ai/en/cloud/use-dify/nodes/human-input.md)); Langflow ([Human Input](https://docs.langflow.org/human-input)); OpenAI ([Node reference](https://developers.openai.com/api/docs/guides/node-reference)) | REQ-SFX-018, 032, 038 | SFX-P5-03 (`approval.py`), P5-05, P9-05, P12-03 (REF-3) | **Adopt** | Timeout as a first-class path makes HITL testable and branchable. Stable decision IDs support idempotent resume (REQ-SFX-032). Runtime pauses go through the existing `HumanDecisionPersistence` (§7.5). |
| CI-10 | CI-C-10; n8n per-tool review; ADK predicate confirmation | Argument-dependent approval rules (for example CEL on tool arguments: approve only external-domain email) and graduated autonomy per tool or edge. | Gumloop ([HITL](https://docs.gumloop.com/core-concepts/human_in_the_loop.md)); Relevance AI ([Approvals](https://relevanceai.com/docs/build/workforces/workforce-features/approvals-and-escalations.md)); n8n ([HITL tools](https://docs.n8n.io/advanced-ai/human-in-the-loop-tools/)); Google ADK ([Confirmation](https://adk.dev/tools-custom/confirmation/)) | REQ-SFX-028, 051, 038 | SFX-P7-02 (profile), P9-02, P9-05 | **Adopt** | Tool policies become "approve when X" instead of on/off. Rules are evaluated by the existing PolicyEngine (CI-11). Relevance's "let agent decide" mode is excluded for effectful tools, because REF-3 requires 0 actions without approval. |
| CI-11 | CI-B-11 | A deterministic policy layer that intercepts every tool call, separate from the agent prompt. | AWS AgentCore ([Overview](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)); Google ([Agent Gateway](https://docs.cloud.google.com/vertex-ai/generative-ai/docs/agent-builder/overview)) | REQ-SFX-028, 040 | SFX-P2-01 (ADR-SFX-001/006), P9-05 | **Adapt** | The pattern is right, but SFX must reuse the existing single authorization path (PC-3: harness principal → tenant → PolicyEngine). It must not add a new policy mechanism (REQ-SFX-040). |
| CI-12 | CI-A-01, CI-C-11 | **The model sees secret names only.** Generation is credential-blind: the LLM sees schema and workflow, never credentials, and credentials are reviewed after generation. Secrets are bound by name. | n8n ([AI Workflow Builder](https://docs.n8n.io/build/ways-of-building-workflows/ai-workflow-builder)); Gumloop ([Secrets](https://docs.gumloop.com/core-concepts/agent_sandbox_and_secrets.md); cited for name binding in the model's context only: values reach sandbox code at runtime and leakage is not excluded, R-W1-04); Dify (Secret-type variables, [Variables](https://enterprise-docs.dify.ai/en/3.8.x/use/workflow/variables.md)) | REQ-SFX-030, 007, 017, 029 | SFX-P7-03 (secret refs only), P6-01, P9-05, P10-05 (Configure inspector) | **Adopt** | Directly enforces "Secret values in artifacts = 0" (R0 §8) and the config inspector that shows placeholders only. Gumloop does not guarantee that sandbox code cannot leak a bound secret, so SFX adds an adversarial check (P9-05). |
| CI-13 | CI-C-07, CI-B-12 | **Per-environment secrets and promotion** of an identical release. One key has separate values per stage, and per-agent credential allow-listing applies. | Vellum ([Environments](https://docs.vellum.ai/product/deployments/environments.md)); StackAI ([Env variables](https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/create-environment-variables.md)); Copilot Studio ([Secrets](https://learn.microsoft.com/en-us/microsoft-copilot-studio/guidance/kit-configure-secrets)); AWS ([Identity: key features](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/key-features-and-benefits.html)) | REQ-SFX-030, 006, 052 | SFX-P7-02, P7-03, P11-04 | **Adopt** | Secret references are bound per profile (LAB in SFX). The same canonical version carries into the admission dossier with unresolved references for the target environment. Promotion past ADMISSION_DOSSIER_READY belongs to the admission owner (OD-SFX-13), not SFX. |
| CI-14 | CI-C-13, CI-B-13, CI-A-12 | Explicit credential identity mode: personal, team or agent-owned; end-user or maker. Credential requests let the owner authorize a connection that the builder can use but never read. | Gumloop ([Connectors](https://docs.gumloop.com/help/connectors/personal-vs-team-vs-agent-owned.md)); Copilot Studio ([Maker credentials](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configure-no-maker-authentication)); Make ([Credential requests](https://help.make.com/credential-requests)) | REQ-SFX-007, 027, 030 | SFX-P7-03, P5-02 (`integration.py`) | **Adapt** | Every integration plan must state which identity a scenario acts as. At L2 only fixture adapters exist (C10/C15), so delegated real-credential flows are recorded as dossier obligations rather than built. |
| CI-15 | CI-C-12 | Anti-pattern: an optional (opt-in toggle) export setting that embeds secret values in plain text. | Dify ([Export/import](https://docs.dify.ai/en/cloud/use-dify/workspace/export-import-apps.md)) | REQ-SFX-030, 033 | SFX-P5-04 (canonical export), P11-04, P9-05 | **Reject** | SFX export and the dossier strip secret values unconditionally, with no option. The canonical form holds references only. |
| CI-16 | CI-B-04, CI-B-05, CI-A-10 (immutability) | **Immutable versions plus movable named endpoints.** DEFAULT follows the latest version and PROD is pinned. Promotion and rollback are pointer moves. Publishing locks the configuration, and edits go through a draft. Releases are tags plus a manifest. | AWS AgentCore ([Versioning](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/agent-runtime-versioning.html)); Retool ([Multi-instance](https://docs.retool.com/source-control/guides/multi-instance-releases/apps)); Zapier ([Agent versions](https://help.zapier.com/hc/en-us/articles/42070243063053-Publish-and-manage-agent-versions)); OpenAI ([Agent Builder](https://developers.openai.com/api/docs/guides/agent-builder)) | REQ-SFX-043, 012 | SFX-P5-05 (lifecycle CAS), P11-03, SFX-P2-01 (ADR-SFX-002) | **Adopt** | Maps one-to-one onto §7.5: `WorkArtifactVersion` is immutable and the current pointer moves by CAS. No rebuild on promotion or rollback. |
| CI-17 | CI-C-05, CI-A-10 (diff) | Semantic or structural diff (by Nodes, Edges and Interface, or a visual canvas diff) before publish. | StackAI ([Diffs](https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/project-version-history-and-diffs.md)); n8n ([History](https://docs.n8n.io/workflows/history/)) | REQ-SFX-023, 045 | SFX-P7-04, P10-06 (version history) | **Adopt** | R0 §8 already requires a semantic diff. The diff is computed over the canonical form so that Studio and IDE show the same diff. |
| CI-18 | CI-C-08; Retool protected triggers | PR-style release review with per-stage required approvals, where the previous version stays live until approval. Changes to triggers go through review. | StackAI ([Approvals](https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/deployments-and-approval-requirements.md)); Vellum ([Release Reviews](https://docs.vellum.ai/product/deployments/release-reviews.md)); Retool ([Protect workflows](https://docs.retool.com/source-control/guides/protect/workflows)) | REQ-SFX-052, 018, 035 | SFX-P11-04, §7.9 SANDBOX_TESTED→SANDBOX_CERTIFIED | **Adapt** | Inside SFX, review gates are the APPROVED decision binding and the independent check before SANDBOX_CERTIFIED. A production release review belongs to admission. Vellum's reviewer feedback loop is not closed in the product; SFX keeps the loop inside the lifecycle record. |
| CI-19 | CI-A-06, CI-B-07; ADK two-tier evals | Evaluation as a first-class scenario mode (dataset, metrics, "is evaluating" gating). There are two tiers: fast unit-style cases and multi-turn evalsets. Test cases can be AI-generated from the design. | n8n ([Evaluations](https://docs.n8n.io/advanced-ai/evaluations/metric-based-evaluations/)); Copilot Studio ([Eval create](https://learn.microsoft.com/en-us/microsoft-copilot-studio/analytics-agent-evaluation-create)); Google ADK ([Evaluate](https://adk.dev/evaluate/)) | REQ-SFX-009, 020, 040 | SFX-P9-01 | **Adapt** | Reuse the existing EvalRunner and proof runner (REQ-SFX-040). Generated test cases are proposals (L1) and must be reviewed before they gate. |
| CI-20 | CI-B-06 | **A Studio session saved as an eval case** that also runs from the CLI or pytest in CI. | Google ADK ([Evaluate](https://adk.dev/evaluate/)) | REQ-SFX-045, 026, 009 | **Parity PC-1..6**, SFX-P10-03, P9-01, P10-08 | **Adopt** | One test asset across Studio and IDE. It extends the PC-4 op-equivalence and PC-5 replay rules to test assets. |
| CI-21 | CI-A-09, CI-B-08, CI-C-06, CI-C-14 | One tool surface for UI and code (validate, create, test, diff, publish). Studio and IDE round-trip on one definition, with two-way sync. A stateless headless executor runs the exported definition. | n8n ([MCP tools](https://docs.n8n.io/connect/connect-to-n8n-mcp-server/mcp-server-tools-reference)); Copilot Studio ([VS Code ext](https://learn.microsoft.com/en-us/microsoft-copilot-studio/visual-studio-code-extension-overview)); Vellum ([SDK](https://docs.vellum.ai/developers/workflows-sdk/introduction.md)); Langflow ([LFX](https://docs.langflow.org/lfx-overview)) | REQ-SFX-026, 045 | PC-1..6; SFX-P5-07, P10-02, P10-03, P10-08 | **Adopt** | This is the R1 §7.7 parity contract. Doors hold no logic (PC-2), and the operation set is identical (PC-6). |
| CI-22 | CI-B-09, CI-C-17 | Anti-pattern: a hosted-only canvas runtime with lossy export, and dependence on a product whose vendor pivots, is acquired or removes features. | OpenAI ([Migrate](https://developers.openai.com/api/docs/guides/agent-builder/migrate-from-agent-builder), [Deprecations](https://developers.openai.com/api/docs/deprecations)); Vellum ([Personal Intelligence, 2026-05-07](https://www.vellum.ai/blog/introducing-vellum)); Gumloop ([Docs](https://docs.gumloop.com/)); Flowise (Workday acquisition: **UNCONFIRMED**, the cited blog URL could not be loaded in review) | REQ-SFX-045, 040, 012 | SFX-P2-01 (ADR-SFX-002), P5-04 | **Reject** | OpenAI retired its hosted runtime about 13 months after launch. The SFX scenario definition must be the portable, canonical source of truth. No external builder product is used as a backend. |
| CI-23 | CI-A-11 | Anti-pattern: live, unsandboxed test runs, and environment moves that drop connections or change webhook URLs. | Zapier ([Next Gen](https://help.zapier.com/hc/en-us/articles/49005047515021-Use-the-Workflow-Manager-to-build-and-view-your-Next-Gen-Zaps), [Move Zaps](https://help.zapier.com/hc/en-us/articles/8496275081357-Move-Zaps-between-accounts)); Make ([Blueprints](https://help.make.com/blueprints), [Replace connections](https://help.make.com/replace-connections-across-multiple-modules)) | REQ-SFX-031, 049, 050 | SFX-P7-02 (L2 profile: LAB, fixture adapters, egress deny), P9-04 | **Reject** | SFX tests run only at L2 with fixture adapters and egress deny. Connections are bound through profile references (CI-13), never embedded IDs. |
| CI-24 | CI-A-08 | Failed runs as resumable "incomplete executions", with automatic retry only for transient error classes. | Make ([Incomplete executions](https://help.make.com/manage-incomplete-executions)); n8n ([Debug executions](https://docs.n8n.io/build/understand-workflows/understand-executions/debug-executions)) | REQ-SFX-032, 021 | SFX-P8-02, P12-02 (REF-2) | **Adapt** | Reuse the long-running checkpoint and DLQ (§7.5). Retry policy is bounded and per error class (REQ-SFX-032). REF-2 requires 0 duplicates under retry. |
| CI-25 | Lane A/B limitations (n8n, Make, Copilot Studio) | Anti-pattern: a Git pull that overwrites local work without conflict detection, a restored version that is not saved, versions that expire after 60 days, and solution export that misses components. | n8n ([Push/pull](https://docs.n8n.io/administer/use-source-control-and-environments/push-and-pull-changes)); Make ([Restore](https://help.make.com/restore-and-recover-scenario)); Copilot Studio ([Import/export](https://learn.microsoft.com/en-us/microsoft-copilot-studio/authoring-solutions-import-export)) | REQ-SFX-043, 033 | SFX-P5-05, P11-04 | **Reject** | §7.5 requires append-only versions, CAS on the current pointer and "never deleted while referenced". A restore creates a new version. Exports are complete and digest-checked. |
| CI-26 | Lane A/C limitations (n8n, Flowise, Relevance) | Anti-pattern: governance and safety features gated by plan (external secrets, environments, evaluations, RBAC). | n8n ([Insights](https://docs.n8n.io/administer/observe-and-log/track-usage-with-insights)); Flowise ([Evaluations](https://docs.flowiseai.com/using-flowise/evaluations.md)); Relevance AI ([RBAC](https://relevanceai.com/docs/enterprise/rbac.md)) | REQ-SFX-027..035 | SFX-P9-05 | **Reject** | Safety controls REQ-SFX-027..035 are always on in every SFX door and profile. |
| CI-27 | Lane A/B limitations (Copilot Studio, AWS, Zapier, Google) | Anti-pattern: several overlapping harnesses, flow formats or products, and frequent renames that fragment the mental model. | Copilot Studio ([Overview](https://learn.microsoft.com/en-us/microsoft-copilot-studio/fundamentals-what-is-copilot-studio)); AWS ([Harness vs Runtime](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/harness-vs-runtime.html)); Zapier ([Migration](https://help.zapier.com/hc/en-us/articles/47402591569805-Migrating-from-Agents-to-AI-by-Zapier)); Google ([Name changes](https://docs.cloud.google.com/gemini-enterprise-agent-platform/vertex-ai-name-changes)) | REQ-SFX-045, 040 | SFX-P2-02, P5-04 | **Reject** | SFX keeps one scenario schema set and one versioned operation map (PC-1, PC-6) across all doors. |
| CI-28 | Lane B limitations (Retool, Copilot Studio) | Anti-pattern: shared maker or OAuth credentials for unattended runs, so every user acts as one account and may see more than they should. | Retool ([Resources](https://docs.retool.com/data-sources/concepts/intro/)); Copilot Studio ([Maker credentials](https://learn.microsoft.com/en-us/microsoft-copilot-studio/configure-no-maker-authentication)) | REQ-SFX-027, 030 | SFX-P7-03, P9-05 | **Reject** | Identity must be explicit per integration (CI-14) and tenant-scoped. A shared maker identity is not a default. |
| CI-29 | Lane C limitation (StackAI) | Anti-pattern: masking that hides values on screen only and can be revealed in the UI. | StackAI ([Env variables](https://docs.stackai.com/guides-and-quickstarts/releases-and-environments/create-environment-variables.md)) | REQ-SFX-030 | SFX-P10-05 (Configure inspector), P9-05 | **Reject** | The Studio never receives secret values. The inspector shows placeholders only (R0 §8). |
| CI-30 | Lane B/C limitations (StackAI, Retool) | Anti-pattern: HITL that is unavailable in unattended schedules or when an agent is called from a workflow. | StackAI ([Ask Human](https://docs.stackai.com/stackai-tools/communication-and-human-input/ask-human-in-chat-interface.md)); Retool ([Glossary](https://docs.retool.com/workflows/reference/glossary)) | REQ-SFX-018, 038 | SFX-P5-03, P10-06 (approval inbox) | **Reject** | SFX approvals are asynchronous, reached through the approval inbox and decision bindings, so they work for unattended runs. |
| CI-31 | Lane B pattern (Retool billing) | Time spent waiting for human approval or auth is not counted as billable or runtime cost. | Retool ([Agents billing](https://docs.retool.com/support/billing-usage/agents)) | REQ-SFX-034, 039 | SFX-P12-06, P11-02 | **Adapt** | Cost-budget and latency metrics (M-6, M-7) should report HITL wait separately, so that approvals do not trip budget caps or skew p95. |
| CI-32 | Lane B/C patterns (AWS Optimization; Make Grid; Dify `build_note.md`) | Three further patterns, each with its own decision:<br>(a) Trace-driven configuration optimisation validated by A/B traffic splitting (AWS).<br>(b) An auto-generated estate map of which scenarios read or write shared stores (Make Grid β).<br>(c) Builder cross-session memory notes (Dify). | AWS ([Overview](https://docs.aws.amazon.com/bedrock-agentcore/latest/devguide/what-is-bedrock-agentcore.html)); Make ([Grid](https://help.make.com/make-grid-public-beta-is-now-live)); Dify ([Build mode](https://docs.dify.ai/en/cloud/use-dify/build/new-agent/build.md)) | (a) REQ-SFX-023; (b) REQ-SFX-012; (c) REQ-SFX-043 | (a) none; (b) SFX-P11-03; (c) SFX-P6-03 | **(a) Reject; (b) Adapt; (c) Adapt** | (a) A/B splitting needs production traffic, which is outside SFX authority (L3 is never granted).<br>(b) The catalog sidecar can show capability reuse and shared dependencies.<br>(c) Session memory belongs in the versioned `SfxSessionState` artifact, not a free-form note. |

**Totals:** 32 CI items (CI-01..CI-32). By decision, counting CI-32's three parts separately:
- **Adopt: 14.** CI-01, 02, 05, 06, 07, 08, 09, 10, 12, 13, 16, 17, 20, 21.
- **Adapt: 10.** CI-03, 04, 11, 14, 18, 19, 24, 31, 32b, 32c.
- **Reject: 10.** CI-15, 22, 23, 25, 26, 27, 28, 29, 30, 32a.

## 7. Gaps no vendor covers

| Gap | Lane evidence | SFX differentiation opportunity |
|---|---|---|
| G-01 **No measured TTFS** | No vendor in any lane publishes a measured time to first success. D-12 scores rest on templates, quickstarts or claims: Dify "30-minute quick start", AWS† "minutes". | Publish a measured TTFS for both the business user and the engineer (M-1, R0 §8 targets ≤ 45 min and ≤ 20 min, PROPOSED until OD-SFX-07). Each value comes with a run artifact under `.tmp/proof/sfx/<run-id>/` (SFX-P12-05, P12-06). |
| G-02 **ROI is absent in lane C and not evidence-backed anywhere** | Lane C has no ROI or time-saved metric. The best it offers is credits per task (Relevance). In lane B only Copilot Studio documents ROI; Google, AWS and OpenAI are `U`. Where ROI exists (n8n, Copilot Studio, Zapier), it multiplies owner-entered or default assumptions (Zapier: 2 min per task) by run counts. | Build an ROI model in which every number carries a source reference. Value metrics are declared during discovery. ROI separates executed runs from runs that achieved the business outcome (CI-05, CI-06; SFX-P11-02). |
| G-03 **Discovery is rare and weak** | Only Relevance AI scores 3 on D-01. Four of five lane B products are `U`, and three lane C products are `0*`. Copilot Studio's value definition is guidance, not a product step. | Make an adaptive interview with playback the entry point of the lifecycle. Measure it with M-2: questions needed to reach a valid spec, and the share of fields inferred (CI-01; SFX-P6-01). |
| G-04 **Discovery and code-first parity never come together** (derived from §4) | No product scores 3 on both D-01 and D-14. Relevance (D-01 = 3) has a run-only SDK (D-14 = 1). On D-01 the D-14 leaders score n8n 1, Google `U`, AWS `U` and Vellum 2 (blog-sourced). | Offer the same discovery-to-blueprint lifecycle in the Studio and the IDE (`/sfx-session`, CLI and MCP), backed by one contract (CI-20, CI-21; REQ-SFX-026, 045). |
| G-05 **Discovery and ROI never come together** (derived from §4) | No product scores 3 on both D-01 and D-11. The ROI leaders (n8n, Copilot Studio) score at most 2 on D-01. The discovery leader (Relevance) scores 2 on D-11. | Connect success criteria captured during discovery to the value model and the run report, so that the business report answers the question asked in Describe (CI-01, CI-05). |
| G-06 **Only one vendor gates publishing on evals** | Only Relevance AI's lane record documents an eval pass-rate publish gate. Where StackAI and Vellum gate deployment, the gate is human approval. Zapier tests are live, and only n8n documents mocked test data (pin data). | Make a passing L2 sandbox test (fixture adapters, egress deny) a required lifecycle transition, with staleness bound to `canonical_digest` (CI-02, CI-23). |
| G-07 **No testable guarantee of zero secret leakage** | Gumloop supplies secret values to sandbox code at runtime and its docs state this "is not a guarantee that code cannot expose a credential" (R-W1-04). Dify export can embed secrets when its optional toggle is enabled. StackAI masking can be revealed. None of the lane records shows a vendor-stated, testable "no secret values in artifacts" guarantee. | Make the R0 §8 metric "Secret values in artifacts = 0" a qualification check, using a no-secret-value scan (SFX-P7-03) and an adversarial leakage suite (SFX-P9-05) (CI-12, CI-15). |
| G-08 **No SLA confirmed** | Lane A confirms no SLA for any of its vendors. Lane C records only Dify's Sev-1 support response time. | No SFX action. SLAs and production obligations belong to admission and PROD-Q, outside SFX (R1 §7.9, §14). Recorded so that it is not mistaken for an SFX gap. |

## 8. Traceability

### 8.1 CI to ADR-SFX-001..006 (R1 §7.11)

| ADR | Topic | CI items it should cite |
|---|---|---|
| ADR-SFX-001 | Placement and ownership (Option B); no duplicate mechanism | CI-11 (reuse PolicyEngine), CI-19 (reuse EvalRunner), CI-24 (reuse long-running checkpoint and DLQ), CI-22 (no external builder as backend), CI-26 |
| ADR-SFX-002 | Contracts, canonicalization, workspace, persistence mapping §7.5 | CI-03, CI-08 (intent field in the specification), CI-12, CI-15, CI-16, CI-17, CI-22, CI-25, CI-27, CI-32c |
| ADR-SFX-003 | Studio stack, auth, hosting | CI-14 and CI-28 (identity per integration, no shared maker identity), CI-29 (Studio never receives secret values) |
| ADR-SFX-004 | IDE toolkit, CLI and MCP parity | CI-20, CI-21, CI-27 |
| ADR-SFX-005 | Catalog sidecar | CI-16 (immutable versioned sidecar), CI-32b (estate and reuse view) |
| ADR-SFX-006 | Authority levels, engineering modes, admission separation | CI-07, CI-08 (Auto limited to new versions), CI-10, CI-13 (promotion stops at the dossier), CI-18, CI-23, CI-30, CI-32a (no production A/B) |

### 8.2 CI to SFX_UX_DESIGN screens (SCR-01..17)

`SFX_UX_DESIGN.md` (SFX-P3-01) is not yet authored in this worktree. As a result, the SCR-01..17 numbering is not available to this record. The table below maps each CI to the R0 §8 lifecycle step or interaction pattern. **SFX-P3-01 must add the SCR id to each row when it assigns screen numbers.**

| Lifecycle step or pattern (R0 §8) | SCR id | CI items |
|---|---|---|
| Describe (free text plus upload) | assign in P3-01 | CI-03 |
| Discover (adaptive interview) | assign in P3-01 | CI-01, CI-05 (value metrics) |
| Understand (editable summary) | assign in P3-01 | CI-01 (playback, gap flags) |
| Design (read-only blueprint canvas, form-edited) | assign in P3-01 | CI-04, CI-07 (plan preview) |
| Configure (profile and integration inspector) | assign in P3-01 | CI-12, CI-13, CI-14, CI-29 |
| Approve (HITL request) | assign in P3-01 | CI-07, CI-10 |
| Build (progress) | assign in P3-01 | CI-24 (resumable failures) |
| Test (sandbox run with dataset) | assign in P3-01 | CI-02, CI-19, CI-20, CI-23 |
| Run results | assign in P3-01 | CI-06, CI-24, CI-31 |
| Present (business report and ROI) | assign in P3-01 | CI-05, CI-06 |
| Improve (new version with diff) | assign in P3-01 | CI-08, CI-17 |
| Approval inbox | assign in P3-01 | CI-09, CI-30 |
| Version history and semantic diff | assign in P3-01 | CI-16, CI-17, CI-25 |
| Interruption recovery | assign in P3-01 | CI-32c |
| Authority badge (L0–L3) | assign in P3-01 | CI-08, CI-10 |
| Catalog and reuse | assign in P3-01 | CI-32b |

### 8.3 CI to IDE toolkit skills and personas (SFX-P4-A01/A02, P10-08)

| Skill or persona | CI items |
|---|---|
| `sfx-session` | CI-07 (plan-then-build in conversation), CI-21 (same operations as the Studio), CI-32c (resume state) |
| `sfx-discover` / persona `sfx-discovery-analyst` | CI-01, CI-03, CI-05 |
| `sfx-specify` | CI-01 (playback), CI-08 (intent field) |
| `sfx-resolve` / persona `sfx-capability-resolver` | CI-03, CI-11 (reuse existing owners) |
| `sfx-blueprint` / persona `sfx-scenario-architect` | CI-04, CI-07, CI-16, CI-17 |
| `sfx-integration-setup` / persona `sfx-integration-advisor` | CI-12, CI-13, CI-14, CI-28 |
| `sfx-test-plan` / persona `sfx-test-designer` | CI-02, CI-19, CI-20, CI-23 |
| `sfx-present` | CI-05, CI-06 |
| persona `sfx-critic` | CI-15, CI-22, CI-25, CI-26 (anti-pattern review checklist) |

## 9. Lane consistency notes

**Verification depth (per R-W1-03).** The per-cell source record is in [`research_lanes/`](research_lanes/): each scored cell carries a source URL there, or is `UNKNOWN`. The lane authors did not re-fetch every cited page; lane A states partial full-page reads (Q-1), and the lane records are paraphrases of the sources. The independent review re-checked a sample of pages on 2026-11-12 and found the drift listed in "Review corrections". The unsampled cells are not verified beyond the lane author's claim.

The lanes cover disjoint products, so no two lanes score the same product. For that reason no cross-lane web spot-checks were made. The following inconsistencies are recorded instead of being resolved:

1. **Absence convention.** Lane C scores inferred absence as 0. Lanes A and B use `UNKNOWN`. As a result, D-01 zeros (Dify, Flowise, Langflow) and D-01 `U` (Retool, OpenAI, Google, AWS) should be read as the same strength of evidence. A future re-score should pick one convention.
2. **Lane C promotion target.** CI-C-07 names promotion to "PRODUCTION_ATTACHED". This conflicts with R1 §7.9: SFX stops at ADMISSION_DOSSIER_READY and never grants L3. CI-13 is scoped accordingly.
3. **Blog-sourced count.** The brief expected one blog-sourced cell in lane C. Vellum has two (D-01 and D-12). Flowise D-02 is sourced from release notes only.
4. **SLA under D-13.** Lane A leaves SLA out of D-13. Lane C cites Dify's Sev-1 support response inside D-13. D-13 scores are not adjusted for this.
5. **Lane B "only Microsoft" ROI statement.** CI-B-02 is true within lane B. Across lanes, n8n also scores 3 on D-11.

## Review corrections

Applied after the independent review of PR #486 (findings R-W1-03, R-W1-04 and R-W1-05). R-W1-01, 02 and 06..11 are handled outside this document.

| Finding | What changed |
|---|---|
| R-W1-03 | The three lane records are committed under `research_lanes/` (`lane-a.md`, `lane-b.md`, `lane-c.md`) as the per-cell source record. Each starts with an evidence-record line and is otherwise unchanged except for `> Correction` notes. The header and §9 now say that every scored cell carries a source URL in the lane records, that verification depth is stated in §9, and that the lane authors did not re-fetch every cited page. |
| R-W1-04 | Gumloop D-06 was overstated. The page says Gumloop provides secret names to the agent and supplies the values to code at runtime as environment variables, and that this "is not a guarantee that code cannot expose a credential". §5 D-06, G-07 and CI-12 are reworded. **Score change: Gumloop D-06 3 -> 2** (§4 row and count of 3s: D-06 goes from 3 to 2). Added a `> Correction R-W1-04:` note in `research_lanes/lane-c.md`. The SFX rule "the model sees secret names only" stands as an SFX design rule, but Gumloop no longer counts as evidence that a vendor delivers it end to end. |
| R-W1-05 | Zapier app count changed from "9,000+" to "thousands of apps" (§5 D-04). Zapier Copilot is labelled "(Beta)", not "open beta" (§1 Q-3, §2). Google "Vertex AI Agent Builder" -> "Gemini Enterprise Agent Platform" rename is marked **UNCONFIRMED** (§2): the rename page has no such row (nearest: "Vertex AI Platform"). Vellum "relaunched" changed to "introduced Personal Intelligence, 2026-05-07" (§2). Dify: the secret export is an optional toggle, and 1.19.x is the application version, not a package-format version (§2, §5, CI-15). AWS token vault and KMS claim is cited to the `key-features-and-benefits` page (§5, CI-13). Flowise/Workday acquisition marked **UNCONFIRMED** (§2, CI-22). `> Correction R-W1-05:` notes added in the lane records. |

**Claims marked UNCONFIRMED:** Google rename of "Vertex AI Agent Builder"; Flowise acquisition by Workday. Three vendor pages were re-fetched while correcting (Gumloop secrets, AWS key-features, Google rename page); the other corrections follow the review's checks of the live pages on 2026-11-12.

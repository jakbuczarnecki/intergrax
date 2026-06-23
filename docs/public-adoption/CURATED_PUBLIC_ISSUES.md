<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Curated Public Issue Drafts

## 1. Purpose

This document defines a small, intentionally selected set of public GitHub issue drafts for Intergrax.

The purpose is to help Intergrax move from a public repository that can be inspected to a public repository where external evaluators know how to engage in a useful, controlled, and strategically aligned way.

Curated public issues are used to invite structured feedback in areas such as:

- proof-path evaluation,
- documentation clarity,
- evidence and trace inspection,
- attestation and boundary-event feedback,
- selected integration discussion,
- qualified design-partner interest.

This document is not a backlog, not a support promise, not an open-source contribution roadmap, and not an automation script.

It defines which public conversations should be opened first, why they matter, which risks they carry, and how they should be framed to remain aligned with the Intergrax collaboration model.

## 2. Public Collaboration Boundaries

Intergrax is a source-available proprietary project. The repository is public for evaluation, technical review, proof-path feedback, selected integration discussion, and technical partner discovery.

Intergrax is not currently presented as:

- a classical open-source framework,
- a finished SaaS product,
- a public production-support channel,
- a production-certification product,
- a compliance-attestation product,
- a general feature-request backlog.

Opening or responding to a public issue does not grant any rights to:

- production use,
- commercial use,
- redistribution,
- derivative works,
- incorporation into products or services,
- substantial implementation work based on Intergrax,
- support,
- SLA,
- automatic acceptance of proposed work.

Commercial, production, redistribution, derivative-work, or substantial implementation use requires explicit permission from the maintainer under the repository license and collaboration model.

Public issues should remain focused on structured evaluation, concrete feedback, and maintainer-reviewed collaboration discovery.

## 3. Issue Selection Principles

Curated public issues should be small, intentional, feedback-oriented, and aligned with the source-available/proprietary collaboration model.

### 3.1 Prefer feedback over feature requests

Good public issues ask external evaluators to inspect, run, review, or validate an existing path.

Preferred framing:

```text
Run this proof path and tell us where it breaks.
```

Avoid framing such as:

```text
Build this feature.
```

### 3.2 Prefer proof paths over abstract discussion

Each issue should point to a concrete artifact or evaluation surface, such as:

- README quick start,
- lab host,
- trace output,
- evidence export,
- attestation demo,
- BoundaryAttest case study,
- collaboration model,
- public roadmap.

Avoid broad prompts such as:

```text
What do you think about Intergrax?
```

### 3.3 Each issue must have a clear audience

A curated issue should clearly identify who is expected to respond.

Useful audience examples:

- first-time evaluator,
- AI systems architect,
- platform engineer,
- governance or observability builder,
- attestation or receipt integrator,
- potential design partner,
- technical reviewer.

If an issue is for everyone, it is probably too broad.

### 3.4 Each issue must define out-of-scope boundaries

Each public issue should clearly say what it is not for.

Common out-of-scope areas:

- production support,
- SLA expectations,
- commercial use requests,
- license debates,
- broad feature requests,
- security vulnerabilities,
- free implementation work,
- open-source contribution assumptions.

### 3.5 Do not imply an open-source contribution model

Avoid issue language such as:

```text
Good first issue
Help wanted
Contributors wanted
Pick this up
Implement this integration
```

Prefer language such as:

```text
Feedback requested
Evaluation path
Design partner interest
Integration feedback
Maintainer-reviewed proposal
```

### 3.6 Do not imply production readiness

Public issues should not imply production guarantees, certification, compliance approval, enterprise support, or hosted-product availability.

Avoid language such as:

```text
production-ready
certified
enterprise support
SLA
official compliance
```

Prefer language such as:

```text
evaluation path
technical validation
proof path
case study
integration feedback
not production certification
not compliance attestation
```

### 3.7 Do not open broad "what should we build?" discussions

Avoid issues such as:

```text
What should Intergrax build next?
What integrations should we add?
What agent applications should Intergrax support?
Suggest product ideas.
```

These discussions create broad feature funnels and can shift the repository away from controlled technical validation.

### 3.8 Treat integrations as feedback or scoped PoC discussion

Integration issues should ask whether a surface is valuable, understandable, or worth validating.

They should not become public implementation commitments for specific tools, vendors, dashboards, exporters, or products.

### 3.9 Design partner issues must qualify the conversation

Design partner issues should ask what real problem a person or team is trying to validate.

They should not invite unlimited product wishlists or imply automatic partnership.

### 3.10 Wave 1 should remain small

The first public issue wave should contain only a small number of carefully chosen issues.

Recommended Wave 1 size: five issues.

A small set signals intentional public adoption. A large initial issue set may look like an uncontrolled backlog or an artificial community-building tactic.

## 4. Issue Categories

### 4.1 Proof path feedback

Collects feedback from evaluators who run an existing documented path.

Examples:

- README quick start,
- lab host run,
- trace export,
- evidence inspection,
- attestation demo run.

Goal: verify whether an external evaluator can run and inspect Intergrax without internal project knowledge.

### 4.2 Documentation clarity

Collects feedback about whether Intergrax is understandable to a first-time reader.

Examples:

- what Intergrax is,
- what Intergrax is not,
- agent / harness / Nexus / application boundaries,
- source-available/proprietary collaboration model,
- first-time evaluator path.

Goal: reduce conceptual and navigation friction.

### 4.3 Evidence / attestation / boundary events feedback

Collects feedback on trace, evidence, boundary events, host attestation, external verification, receipt mapping, and the BoundaryAttest case study.

Goal: validate whether Intergrax's evidence surfaces are understandable and useful to external attestation, auditability, and governance-oriented reviewers.

### 4.4 Design partner interest

Collects qualified interest from people or teams building governed agent applications.

Goal: identify real product or platform problems where Intergrax's Harness AI model may be relevant.

### 4.5 Integration feedback

Collects feedback on selected external integration surfaces.

Examples:

- trace and evidence export surfaces,
- MCP as a controlled task surface,
- governance or policy evaluation surfaces,
- attestation consumers,
- external evidence stores.

Goal: understand which integration surfaces are worth validating without turning GitHub Issues into an implementation backlog.

### 4.6 Product-validation track

Collects feedback on concrete product-validation directions built on Intergrax.

Primary candidate: Local Knowledge Workspace alpha.

Goal: validate whether a product direction built on Intergrax solves a real problem while exercising the harness on practical workloads.

### 4.7 Deferred / later issues

Contains valuable issues that should not be opened immediately because they depend on earlier feedback, stronger public narrative, or reduced ambiguity.

### 4.8 Not recommended now

Contains issue types that should not be opened because they create chaos, false expectations, licensing confusion, support expectations, or strategic drift.

## 5. Wave 1 — Recommended First Issues

Wave 1 contains exactly five curated public issues.

These issues are selected because they maximize proof-path feedback, documentation clarity, external technical validation, and design-partner discovery while avoiding broad feature requests, support expectations, open-source assumptions, or uncontrolled product-roadmap discussions.

### 5.1 Proof path feedback: README quick start

**Category:** Proof path feedback  
**Recommended template:** `proof_path_feedback.yml`  
**Recommended labels:** `feedback`, `triage`, `proof-path`  
**Primary audience:** first-time evaluator, platform engineer, technical reviewer  
**Open now:** Yes — Wave 1, priority 1

**Goal:** Validate whether an external evaluator can follow the README quick start path without internal project knowledge.

**Why it matters:** The README quick start is the primary public proof path. It should allow an evaluator to clone, install, verify, run, execute a sample request, and inspect trace output.

**Risk:** Low. This issue asks for concrete evaluation feedback and does not imply support, production readiness, or feature delivery.

**Draft body:**

```md
## Purpose

We are looking for first-run feedback from external evaluators following the README quick start path.

The goal is not to request new features, but to identify friction in the public evaluation path:

- clone the repository
- install dependencies
- run verification commands
- start the lab host
- execute a sample request
- inspect the resulting trace

## Suggested evaluation path

Start from the README quick start section and follow the documented local evaluation flow.

Please note where the path was clear, where it was unclear, and where a command or expectation did not match your environment.

## Useful feedback

Please share:

- which steps worked as expected
- where the instructions became unclear
- which command failed, if any
- relevant command output, with secrets redacted
- your environment: OS, Python version, uv version, branch or commit
- what would make the first-run path easier to evaluate

## Out of scope

This issue is not for:

- production support
- commercial use requests
- broad feature requests
- license debates
- security vulnerabilities
- requests for free implementation work

Intergrax is source-available/proprietary. Evaluation feedback does not grant production, redistribution, derivative-work, commercial-use, or support rights.
```

### 5.2 Documentation clarity: first-time evaluator path

**Category:** Documentation clarity  
**Recommended template:** `documentation_feedback.yml`  
**Recommended labels:** `documentation`, `triage`, `feedback`  
**Primary audience:** first-time evaluator, AI systems architect, technical reviewer, potential design partner  
**Open now:** Yes — Wave 1, priority 2

**Goal:** Validate whether the public repository clearly explains what Intergrax is, what it is not, who it is for, and where an evaluator should start.

**Why it matters:** Intergrax has a non-trivial mental model. External readers must understand the distinction between agent, harness, Nexus, application, policy, trace, and evidence before they can evaluate the project correctly.

**Risk:** Low. This issue is focused on documentation clarity and navigation.

**Draft body:**

```md
## Purpose

We are looking for documentation feedback from first-time evaluators.

The goal is to understand whether the public repository clearly explains:

- what Intergrax is
- what Intergrax is not
- who it is for
- where an evaluator should start
- what collaboration tracks are currently appropriate
- what requires maintainer permission

## Useful feedback

Please share:

- the first document you opened
- where the project became clear
- where the project became confusing
- which term, diagram, or boundary was hardest to understand
- whether the source-available/proprietary collaboration model was clear
- whether the difference between agent, harness, Nexus, and application was understandable
- what link, section, diagram, or wording would improve the first-time evaluator path

## Reader profile

It is helpful to mention your perspective, for example:

- first-time evaluator
- agent developer
- platform engineer
- integration partner
- AI systems architect
- product team
- other

## Out of scope

This issue is not for:

- broad feature requests
- production support
- commercial use requests
- license debates
- implementation tasks
- general product brainstorming

Intergrax is source-available/proprietary. Documentation feedback does not create partnership, support, production-use, commercial-use, redistribution, or derivative-work rights.
```

### 5.3 Proof path feedback: evidence and trace inspection

**Category:** Proof path feedback / evidence feedback  
**Recommended template:** `proof_path_feedback.yml`  
**Recommended labels:** `feedback`, `triage`, `proof-path`, `evidence`  
**Primary audience:** platform engineer, observability builder, governance engineer, technical reviewer  
**Open now:** Yes — Wave 1, priority 3

**Goal:** Validate whether trace and evidence outputs are understandable to external reviewers.

**Why it matters:** Intergrax's core value is not only agent execution. It is policy-bound execution with trace, evidence, and reviewable runtime boundaries.

**Risk:** Medium. The issue must not imply production audit, compliance attestation, or certification.

**Draft body:**

```md
## Purpose

We are looking for feedback on the public trace and evidence inspection path.

The goal is to understand whether an external evaluator can inspect Intergrax outputs and understand:

- what happened during a run
- which events or records were produced
- what information is clear
- what information is missing
- which fields require better explanation
- whether the difference between runtime trace and evidence output is understandable

## Suggested evaluation path

Start from the README proof path and inspect available trace or evidence outputs, such as:

- trace rendering commands
- trace export commands
- lab host debug or trace endpoints, where applicable
- generated evidence files, where applicable

## Useful feedback

Please share:

- which trace or evidence output you inspected
- whether the output was understandable without internal project knowledge
- which fields were clear
- which fields were unclear
- whether the ordering, lineage, run identifiers, or step identifiers were understandable
- what additional explanation, diagram, or example would make inspection easier

## Important boundary

This issue is about evaluation-path clarity.

It is not a production audit request, not a compliance attestation request, and not a production certification claim.

## Out of scope

This issue is not for:

- production support
- compliance certification
- legal or security approval
- commercial use requests
- broad observability feature requests
- requests to implement a specific exporter or dashboard without prior maintainer discussion

Intergrax is source-available/proprietary. Feedback does not grant production, redistribution, derivative-work, commercial-use, or support rights.
```

### 5.4 Attestation integration feedback: BoundaryAttest case study

**Category:** Evidence / attestation / boundary events feedback  
**Recommended template:** `integration_proposal.yml`, framed as feedback rather than implementation request  
**Recommended labels:** `integration`, `feedback`, `triage`, `attestation`, `evidence`  
**Primary audience:** attestation integrator, auditability builder, governance engineer, security-oriented platform engineer  
**Open now:** Yes — Wave 1, priority 4

**Goal:** Collect technical feedback on the BoundaryAttest case study and Intergrax Execution Boundary Export pattern.

**Why it matters:** This is currently one of the strongest public technical proofs for Intergrax. It demonstrates how external systems can consume host-signed execution boundary events without collapsing Intergrax host/runtime claims and external client-observed receipt claims.

**Risk:** Medium to high. The issue must avoid implying production certification, compliance attestation, formal partnership, bundled support, or security/legal approval.

**Draft body:**

```md
## Purpose

We are looking for technical feedback on the BoundaryAttest attestation case study and the Intergrax Execution Boundary Export pattern.

The goal is to validate whether the public case study clearly explains:

- host-signed execution boundary events
- separate tool and harness-step boundary claims
- how an external verifier can consume Intergrax evidence
- the distinction between Intergrax host/runtime claims and external client-observed receipts
- what is clear or missing for external attestation integrators

## Suggested review path

Please review the BoundaryAttest case study and related attestation demo documentation.

Useful feedback may focus on:

- trust model clarity
- boundary event shape
- receipt mapping clarity
- signature / verification explanation
- separation between Intergrax host claims and external partner claims
- whether the case study gives enough information for an external PoC review

## Useful feedback

Please share:

- whether the event/receipt boundary is clear
- whether the host-attestation role is clear
- whether the case study avoids overclaiming
- which terms or fields need better explanation
- what diagram, example, field table, or verification note would help
- whether this pattern fits external attestation, audit, or receipt systems you are familiar with

## Important non-claims

This issue does not imply:

- production certification
- compliance attestation
- legal approval
- security approval
- formal partnership
- bundled BoundaryAttest support
- commercial or production use rights
- guaranteed acceptance of integration work

BoundaryAttest remains an external project. Intergrax remains independently maintained.

## Out of scope

This issue is not for:

- security vulnerability disclosure
- production deployment requests
- commercial licensing requests
- compliance certification requests
- broad feature requests
- requests to bundle or maintain external projects without maintainer approval

Intergrax is source-available/proprietary. Feedback does not grant production, redistribution, derivative-work, commercial-use, or support rights.
```

### 5.5 Design partner interest: governed agent applications

**Category:** Design partner interest  
**Recommended template:** `design_partner_interest.yml`  
**Recommended labels:** `design-partner`, `triage`, `feedback`  
**Primary audience:** AI product teams, platform teams, agent application builders, governance-focused teams, potential design partners  
**Open now:** Yes — Wave 1, priority 5

**Goal:** Create a controlled public entry point for people or teams evaluating governed agent applications where Intergrax may be relevant.

**Why it matters:** Intergrax needs qualified design-partner discovery, not generic user acquisition or broad feature requests. This issue focuses the conversation on real governance, trace, evidence, HITL, controlled tool use, and runtime-boundary problems.

**Risk:** Medium. The issue must not create expectations of support, consulting, production permission, partnership, or implementation commitment.

**Draft body:**

```md
## Purpose

We are looking for design-partner interest from people or teams building governed agent applications.

This track is for evaluating whether Intergrax's Harness AI model fits real product needs where agents require:

- policy-bound execution
- trace and evidence
- clear agent/application/runtime boundaries
- human-in-the-loop checkpoints
- controlled tool use
- repeatable proof paths
- maintainable multi-agent orchestration

## Good fit

This may be relevant if you are building or evaluating:

- internal AI workspaces
- governed agent tools
- agent-backed business applications
- compliance-sensitive automation
- multi-agent systems that need runtime discipline
- products where agents must be observable, constrained, and reviewable

## What to share

Please describe:

- what kind of agent application you are building or evaluating
- which governance, runtime, evidence, or integration problem matters most
- what would make a design-partner conversation useful
- what kind of proof path or evaluation would validate fit
- preferred contact method

Do not include confidential information, private credentials, customer data, or secrets in this public issue.

## Not a commitment

Submitting interest does not create:

- a partnership
- an implementation commitment
- an SLA
- a support obligation
- production-use permission
- commercial-use permission
- redistribution rights
- derivative-work rights
- guaranteed acceptance of proposed work

Intergrax is source-available/proprietary. Commercial, production, redistribution, derivative-work, or substantial implementation use requires explicit maintainer permission.
```

## 6. Wave 2 — Deferred Candidates

Wave 2 contains valuable but intentionally deferred public issue candidates.

These issues should not be opened together with Wave 1 because they either require stronger public narrative, depend on first feedback from evaluators, overlap with Wave 1 topics, or may attract broad implementation requests if opened too early.

### 6.1 Documentation clarity: Harness AI mental model

**Category:** Documentation clarity  
**Recommended status:** Wave 2 or conditional

**Goal:** Validate whether external readers understand the central Intergrax mental model:

```text
agents decide,
harness executes,
Nexus orchestrates,
applications host.
```

**Why not Wave 1:** This overlaps with `Documentation clarity: first-time evaluator path`. It should be opened only if first-time evaluator feedback shows confusion around the agent / harness / Nexus / application model.

**Opening condition:** Open if Wave 1 feedback shows repeated confusion about Intergrax's architectural boundaries.

**Risk:** Low. This is a documentation clarity issue, but it should not duplicate an active Wave 1 issue.

### 6.2 Integration feedback: trace and evidence export surfaces

**Category:** Integration feedback / evidence feedback  
**Recommended status:** Wave 2

**Goal:** Collect feedback from observability, governance, auditability, and platform-engineering reviewers on which external systems could meaningfully consume Intergrax trace and evidence outputs.

**Why not Wave 1:** First validate that current trace and evidence outputs are understandable. Only then ask which external export surfaces are worth exploring.

**Opening condition:** Open after receiving useful feedback from `Proof path feedback: evidence and trace inspection`.

**Risk:** Medium. This can turn into a wishlist for specific exporters, dashboards, or vendor integrations unless framed as feedback on surfaces rather than implementation commitments.

### 6.3 Design partner interest: Local Knowledge Workspace alpha

**Category:** Product-validation track / design partner interest  
**Recommended status:** Wave 2

**Goal:** Collect feedback from people with real local knowledge, document, RAG, and controlled-workspace problems that may fit the Local Knowledge Workspace alpha direction.

**Why not Wave 1:** Local Knowledge Workspace needs a lighter public narrative before it becomes a public design-partner issue. The architecture exists, but external evaluators need a clearer product-validation story.

**Opening condition:** Open after adding or improving a public Local Knowledge Workspace alpha narrative.

**Risk:** Medium. It may create expectations of a ready desktop app, SaaS product, production use, custom integrations, or commercial permission.

### 6.4 Integration feedback: MCP as a controlled Intergrax task surface

**Category:** Integration feedback  
**Recommended status:** Wave 2

**Goal:** Collect feedback on MCP as a controlled task intake or tool surface for Intergrax, mapped to Task, policy, HITL, trace, and controlled execution.

**Why not Wave 1:** MCP can attract broad and noisy feature requests. Intergrax should first validate its public proof path and architectural boundaries before opening MCP-specific discussion.

**Opening condition:** Open after first feedback confirms that external readers understand the agent / harness / Nexus / application boundaries and the role of trace/evidence.

**Risk:** Medium to high. This can become an MCP wishlist unless strongly framed around Intergrax boundaries and controlled execution.

## 7. Not Recommended Now

The following issue types should not be opened as public curated issues at this stage.

### 7.1 Generic feature request issues

Do not open issues such as:

```text
What should Intergrax build next?
Suggest new Intergrax features.
What integrations should we add?
What agents should Intergrax support?
```

**Risk:** Creates broad, low-signal feature funnels.

**Safer alternative:** Use proof-path feedback, selected integration feedback, or qualified design-partner interest.

### 7.2 Help wanted / good first issue implementation tasks

Do not open issues such as:

```text
Good first issue: add X
Help wanted: implement Y
Contributors wanted
Pick this up
```

**Risk:** Implies a classical open-source contribution model.

**Safer alternative:** Use feedback-oriented issues and maintainer-reviewed proposals.

### 7.3 Broad small reference application proposals

Do not open issues such as:

```text
What demo app should Intergrax build?
Small reference app ideas
Agent/application proposal: small reference application
```

**Risk:** Attracts random chatbot, dashboard, productivity, CRM, email, or document-agent ideas that may not validate the harness.

**Safer alternative:** Later, consider a narrower issue such as `Reference application feedback: smallest app that proves harness value` with strict criteria.

### 7.4 Public support requests

Do not open issues such as:

```text
Support: help me run Intergrax
How do I deploy this?
Can you help me integrate this into my product?
```

**Risk:** Creates support and SLA expectations.

**Safer alternative:** Accept reproducible proof-path feedback or concrete bug reports with environment and output.

### 7.5 Production readiness / certification / compliance issues

Do not open issues such as:

```text
Production readiness checklist
Certify Intergrax for production
Compliance roadmap
Security certification
Enterprise deployment support
```

**Risk:** Implies production guarantees, certification, compliance attestation, or enterprise support.

**Safer alternative:** Discuss evidence clarity, trace inspection, and attestation case study feedback without making production claims.

### 7.6 Public licensing or commercial-use discussions

Do not open issues such as:

```text
Can I use Intergrax commercially?
Change license to open source.
Open-core roadmap discussion.
Can I redistribute a modified version?
```

**Risk:** Turns GitHub Issues into a public licensing negotiation channel.

**Safer alternative:** Redirect commercial, production, redistribution, derivative-work, and permission requests to direct maintainer contact.

### 7.7 Public security vulnerability reports

Do not open issues for vulnerability disclosure.

**Risk:** Publicly exposes security-sensitive information.

**Safer alternative:** Redirect to the repository security policy or private maintainer contact.

### 7.8 Tool-specific integration build requests

Do not open issues such as:

```text
Add OpenTelemetry exporter.
Add LangSmith integration.
Add Grafana dashboard.
Add MCP support for X.
Add Slack integration.
Add Notion integration.
```

**Risk:** Looks like a public implementation backlog and can imply maintainer commitment.

**Safer alternative:** Ask for feedback on integration surfaces, not requests to build a specific vendor integration.

### 7.9 SaaS / hosted product / pricing / onboarding issues

Do not open issues such as:

```text
Intergrax Cloud roadmap
Hosted Intergrax beta
SaaS onboarding
Pricing feedback
Production deployment pilot
```

**Risk:** Suggests a hosted product or SaaS offer that is not currently part of the public repository promise.

**Safer alternative:** Use design-partner interest issues without implying hosted-product onboarding.

### 7.10 Historical cleanup issues without public-facing value

Do not open issues whose only purpose is to rewrite historical implementation journals or ADRs because they contain old names or old context.

**Risk:** Low ROI and loss of historical context.

**Safer alternative:** Update only current public-facing documents when old context causes real confusion for current evaluators.

## 8. Recommended Opening Order

Recommended Wave 1 opening order:

1. `Proof path feedback: README quick start`
2. `Documentation clarity: first-time evaluator path`
3. `Proof path feedback: evidence and trace inspection`
4. `Attestation integration feedback: BoundaryAttest case study`
5. `Design partner interest: governed agent applications`

Rationale:

1. First validate that the repository can be run.
2. Then validate that the repository can be understood.
3. Then validate that trace and evidence can be inspected.
4. Then validate the strongest public attestation proof.
5. Then open a controlled design-partner discovery path.

This order moves from basic technical evaluation to strategic collaboration without creating a broad public roadmap or support funnel.

## 9. Maintainer Notes

Public issue handling should remain consistent with the collaboration model.

Recommended maintainer behavior:

- thank contributors for concrete feedback,
- ask for reproduction details when needed,
- keep discussions scoped to the issue purpose,
- avoid promising implementation,
- avoid promising support timelines,
- avoid suggesting production readiness,
- redirect commercial or licensing questions to direct contact,
- redirect security disclosures away from public issues,
- close off-topic issues politely,
- keep labels consistent,
- treat public issues as evaluation signals, not automatic roadmap commitments.

Suggested response pattern for useful feedback:

```md
Thank you for running this path and sharing concrete feedback.

This is useful for improving the public evaluation flow. I will treat this as evaluation feedback rather than an implementation commitment.

If more detail is needed, I may ask for a specific command, output snippet, environment detail, or documentation location.
```

Suggested response pattern for out-of-scope feature requests:

```md
Thank you for the suggestion.

This repository uses curated public issues for focused evaluation feedback, documentation clarity, selected integration feedback, and design-partner discovery. This request is broader than the current public-adoption scope, so I am not treating it as a roadmap commitment.
```

Suggested response pattern for commercial or production-use questions:

```md
Thank you for your interest.

Commercial use, production use, redistribution, derivative works, incorporation into products or services, or substantial implementation work require explicit maintainer permission under the repository license and collaboration model. Please contact the maintainer directly for that discussion.
```

Suggested response pattern for security-sensitive reports:

```md
Please do not disclose security-sensitive details in a public issue.

Use the repository security policy or contact the maintainer privately so the report can be handled safely.
```

## 10. Future Automation Notes

This document is intentionally not an automation script.

Future automation may be added later, after the curated issue set is reviewed and accepted.

A possible future structure:

```text
docs/public-adoption/curated_public_issues.yml
scripts/public_adoption/create_curated_issues.py
```

Recommended automation constraints:

- dry-run by default,
- explicit `--apply` required to create issues,
- idempotency check by issue title,
- option to open only selected waves,
- option to open only a single issue by ID,
- no automatic issue creation from Markdown without review,
- no scheduled or background issue creation,
- no automatic creation of production, support, licensing, or broad feature-request issues.

Automation should reduce manual copy-paste work. It should not replace maintainer judgment.

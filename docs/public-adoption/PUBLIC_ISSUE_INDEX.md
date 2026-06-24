<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Public Issue Index

This document lists the currently active maintainer-curated public issues for Intergrax.

The purpose of this index is to make public feedback entry points easy to discover and easy to use without turning the repository into a broad feature-request backlog, production-support channel, or open-source contribution board.

Intergrax is source-available/proprietary. Public issues are for structured evaluation feedback, documentation clarity, selected integration feedback, qualified design-partner discovery, architecture discussion, product-validation discussion, and deep technical review. They do not grant production, commercial, redistribution, derivative-work, implementation, support, or SLA rights.

## Active curated issues

| Issue | Purpose | Best for | Status |
|-------|---------|----------|--------|
| [#186 Proof path feedback: README quick start](https://github.com/jakbuczarnecki/intergrax/issues/186) | First-run feedback on the README quick start path | First-time evaluators, platform engineers, technical reviewers | Open |
| [#187 Documentation clarity: first-time evaluator path](https://github.com/jakbuczarnecki/intergrax/issues/187) | Feedback on whether the public repository explains what Intergrax is, is not, and where to start | First-time evaluators, AI systems architects, technical reviewers | Open |
| [#188 Proof path feedback: evidence and trace inspection](https://github.com/jakbuczarnecki/intergrax/issues/188) | Feedback on whether trace and evidence outputs are understandable | Platform engineers, governance engineers, observability builders | Open |
| [#189 Attestation integration feedback: BoundaryAttest case study](https://github.com/jakbuczarnecki/intergrax/issues/189) | Feedback on the BoundaryAttest case study and Execution Boundary Export pattern | Attestation integrators, auditability builders, governance engineers | Open |
| [#190 Design partner interest: governed agent applications](https://github.com/jakbuczarnecki/intergrax/issues/190) | Qualified design-partner interest for governed agent applications | AI product teams, platform teams, agent application builders | Open |
| [#191 Documentation clarity: Harness AI mental model](https://github.com/jakbuczarnecki/intergrax/issues/191) | Feedback on the Agent / Harness / Nexus / Application mental model | Architecture reviewers, technical evaluators, documentation reviewers | Open |
| [#192 Integration feedback: trace and evidence export surfaces](https://github.com/jakbuczarnecki/intergrax/issues/192) | Feedback on possible trace and evidence export surfaces | Observability builders, governance engineers, integration reviewers | Open |
| [#193 Design partner interest: Local Knowledge Workspace alpha](https://github.com/jakbuczarnecki/intergrax/issues/193) | Design-partner feedback on the Local Knowledge Workspace alpha direction | Product teams, local knowledge workflow evaluators, controlled-RAG users | Open |
| [#194 Integration feedback: MCP as a controlled Intergrax task surface](https://github.com/jakbuczarnecki/intergrax/issues/194) | Feedback on MCP as a controlled task or tool surface | MCP reviewers, integration builders, agent tool-surface evaluators | Open |

## Recommended evaluation order

For a time-boxed 5/15/30/60-minute review flow, start with [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md).

For first-time evaluators, the recommended order is:

1. [#186 README quick start feedback](https://github.com/jakbuczarnecki/intergrax/issues/186)
2. [#187 First-time evaluator documentation clarity](https://github.com/jakbuczarnecki/intergrax/issues/187)
3. [#188 Evidence and trace inspection feedback](https://github.com/jakbuczarnecki/intergrax/issues/188)
4. [#189 BoundaryAttest case study feedback](https://github.com/jakbuczarnecki/intergrax/issues/189)
5. [#190 Governed agent applications design-partner interest](https://github.com/jakbuczarnecki/intergrax/issues/190)

The remaining issues are useful for more specific reviewers:

- use [#191](https://github.com/jakbuczarnecki/intergrax/issues/191) if the main question is whether the Harness AI mental model is clear,
- use [#192](https://github.com/jakbuczarnecki/intergrax/issues/192) if the main question is trace/evidence export integration,
- use [#193](https://github.com/jakbuczarnecki/intergrax/issues/193) if the main question is Local Knowledge Workspace product fit,
- use [#194](https://github.com/jakbuczarnecki/intergrax/issues/194) if the main question is MCP as a controlled Intergrax surface.

## Which issue should I use?

| If you want to... | Use |
|-------------------|-----|
| Run Intergrax for the first time and report friction | [#186](https://github.com/jakbuczarnecki/intergrax/issues/186) |
| Report confusing public documentation or navigation | [#187](https://github.com/jakbuczarnecki/intergrax/issues/187) |
| Inspect trace or evidence outputs | [#188](https://github.com/jakbuczarnecki/intergrax/issues/188) |
| Review the BoundaryAttest attestation case study | [#189](https://github.com/jakbuczarnecki/intergrax/issues/189) |
| Discuss governed agent application fit | [#190](https://github.com/jakbuczarnecki/intergrax/issues/190) |
| Review the Harness AI mental model | [#191](https://github.com/jakbuczarnecki/intergrax/issues/191) |
| Discuss trace/evidence export surfaces | [#192](https://github.com/jakbuczarnecki/intergrax/issues/192) |
| Discuss Local Knowledge Workspace alpha fit | [#193](https://github.com/jakbuczarnecki/intergrax/issues/193) |
| Discuss MCP as a controlled task/tool surface | [#194](https://github.com/jakbuczarnecki/intergrax/issues/194) |

## Expanded discussion waves

Additional architecture, product-validation, and deep technical discussion issues are prepared in [Public Discussion Issue Expansion](PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) and [curated_public_discussion_issues.yml](curated_public_discussion_issues.yml).

| Wave | Purpose | Source | GitHub state |
|------|---------|--------|--------------|
| Wave 3 | Architecture discussion issues | [curated_public_discussion_issues.yml](curated_public_discussion_issues.yml) | Prepared; create explicitly with `--apply` |
| Wave 4 | Product / application validation issues | [curated_public_discussion_issues.yml](curated_public_discussion_issues.yml) | Prepared; create explicitly with `--apply` |
| Wave 5 | Deep technical discussion issues | [curated_public_discussion_issues.yml](curated_public_discussion_issues.yml) | Prepared; create explicitly with `--apply` |

These expanded waves are intentionally separated from the active issue list until they are created on GitHub and assigned issue numbers.

## Maintainer handling rules

Maintainer responses should keep public issues focused and scoped. See [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) for the full public-adoption triage policy and response templates.

Recommended handling rules:

- thank people for concrete feedback,
- ask for reproducible details when needed,
- avoid promising implementation,
- avoid promising support timelines,
- avoid suggesting production readiness,
- redirect commercial or licensing questions to direct maintainer contact,
- redirect security disclosures away from public issues,
- close off-topic issues politely,
- treat public issues as evaluation signals, not automatic roadmap commitments.

## Out of scope

The curated public issues are not for:

- production support,
- commercial use requests,
- redistribution or derivative-work permission requests,
- broad feature requests detached from architecture or product validation,
- license debates,
- public security vulnerability disclosure,
- hosted SaaS or pricing discussions,
- requests for free implementation work,
- requests that imply an open-source contribution model.

For commercial licensing, production use, partnerships, or permission requests, contact the maintainer directly.

## Source documents

- [Maintainer Triage Playbook](MAINTAINER_TRIAGE_PLAYBOOK.md) — maintainer handling rules, close/keep-open criteria, escalation rules, and response templates.
- [Public Discussion Issue Expansion](PUBLIC_DISCUSSION_ISSUE_EXPANSION.md) — expanded architecture, product-validation, and deep technical discussion wave plan.
- [Curated Public Issue Drafts](CURATED_PUBLIC_ISSUES.md) — strategy and draft bodies for curated public issues.
- [curated_public_issues.yml](curated_public_issues.yml) — structured source data for active issue automation.
- [curated_public_discussion_issues.yml](curated_public_discussion_issues.yml) — structured source data for expanded public discussion issue automation.
- [create_curated_issues.py](../../scripts/public_adoption/create_curated_issues.py) — maintainer utility for dry-run and explicit issue creation.
- [ROADMAP.md](../../ROADMAP.md) — public adoption roadmap and collaboration priorities.
- [COLLABORATION.md](../../COLLABORATION.md) — collaboration model and permission boundaries.
- [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md) — time-boxed evaluation guide for external reviewers.

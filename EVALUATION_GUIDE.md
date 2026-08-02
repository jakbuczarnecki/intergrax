<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Evaluation Guide

Intergrax is a source-available Harness AI / Agent OS for building governed agent applications.

This guide helps technical reviewers, design partners, and integration builders evaluate Intergrax in a focused 5, 15, 30, or 60 minute pass. All evaluation steps described here — including clone, local install, Quick Start, tests, benchmarks, proof paths, private evaluation modifications, and evaluation-only integrations — are permitted under the [Intergrax Evaluation and Collaboration License 1.0](LICENSE).

It is not a product brochure, SaaS offer, open-source license grant, production-readiness claim, certification, compliance statement, support commitment, or partnership term.

Production use, commercial use, hosted services, redistribution as an independent product, incorporation into products or services, and commercial derivative works require explicit written permission. See [LICENSE](LICENSE), [COLLABORATION.md](COLLABORATION.md), and [PARTNERS.md](PARTNERS.md).

---

## Who this guide is for

Use this guide if you are:

- evaluating whether Intergrax solves a real governed-agent problem;
- reviewing the Harness AI / Agent OS architecture for the first time;
- checking whether the proof path is understandable;
- assessing trace/evidence, policy, HITL, RAG, memory, tools, or orchestration surfaces;
- considering design-partner or selected integration feedback;
- deciding which public feedback path to use.

If you are looking for a finished SaaS, production support, an open-source contribution model, or unrestricted commercial use, start with [FAQ.md](FAQ.md), [COLLABORATION.md](COLLABORATION.md), and [LICENSE](LICENSE).

---

## Evaluation paths by available time

| Time | Goal | Start here | Outcome |
|------|------|------------|---------|
| 5 minutes | Understand what Intergrax is and is not | [README.md](README.md), [FAQ.md](FAQ.md) | Know whether the repo is relevant to you |
| 15 minutes | Understand the architecture and use cases | [README.md#start-here](README.md#start-here), [USE_CASES.md](USE_CASES.md) | Know which use case or validation path fits |
| 30 minutes | Run or inspect a proof path | [README.md#quick-start](README.md#quick-start), [README.md#proof-of-platform](README.md#proof-of-platform) | Report first-run friction or proof-path gaps |
| 45-60 minutes | Review deeper validation surfaces | [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md), [BoundaryAttest PoC](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md), [LKW alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) | Provide structured feedback or design-partner signal |

---

## 5-minute orientation

Goal: decide whether Intergrax is relevant before reading deeply.

Read:

1. [README.md](README.md) — repository overview and Start here table.
2. [FAQ.md](FAQ.md) — common questions about license, status, collaboration, and public use.
3. [COLLABORATION.md](COLLABORATION.md) — what is allowed and what requires permission.

Check whether the following statement is clear:

> Intergrax is a source-available Harness AI / Agent OS for governed agent applications, not a finished SaaS, not an open-source framework, and not a production certification claim.

Useful feedback after this pass:

- Is the repository positioning clear?
- Is the license/collaboration boundary clear?
- Do you understand where to start?
- What phrase or section is confusing?

Recommended feedback path: [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md), especially documentation clarity issues.

---

## 15-minute architecture and use-case scan

Goal: understand what problem Intergrax is trying to solve.

Read:

1. [README.md#the-agent-model--why-architects-choose-intergrax](README.md#the-agent-model--why-architects-choose-intergrax)
2. [USE_CASES.md](USE_CASES.md)
3. [PARTNERS.md](PARTNERS.md)
4. Optional — cross-layer capabilities: [docs/features/README.md](docs/features/README.md) (`docs/features/architecture/<FEATURE>.md` ↔ `docs/features/plan/<FEATURE>.md`)

Evaluate whether the model is understandable:

- agents decide domain steps;
- the harness executes under policy, trace, state, and budgets;
- Nexus orchestrates graphs and multi-agent flow;
- Tier-3 applications own environment, identity, profile, and product boundaries;
- tools, skills, RAG, memory, HITL, and evidence are harness-managed surfaces.

Questions to answer:

- Does the Harness AI / Agent OS model make sense?
- Does the separation between agent, harness, Nexus, and application host feel useful?
- Which use case is most relevant to your work?
- What would you need to see next to justify deeper evaluation?

Recommended feedback path: [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md), especially Harness AI mental model and governed agent application issues.

---

## 30-minute proof-path evaluation

Goal: test whether the local evaluation path is understandable and reproducible.

Start from:

- [README.md#quick-start](README.md#quick-start)
- [README.md#proof-of-platform](README.md#proof-of-platform)

Typical local flow:

```bash
uv sync --extra dev
uv run intergrax doctor
uv run pytest -m gate -q
```

Optional proof/evidence paths are documented in README. Follow only the commands that match your evaluation intent and environment.

Evaluate:

- Are prerequisites clear?
- Does install work as written?
- Does `intergrax doctor` provide useful feedback?
- Are gate tests discoverable and understandable?
- Is the proof-of-platform path visible enough?
- Are trace/evidence outputs understandable when inspected?

Do not assume that passing local checks grants production permission, commercial use, support, certification, or compliance status.

Recommended feedback path: [#186 README quick start feedback](https://github.com/jakbuczarnecki/intergrax/issues/186) and [#188 evidence and trace inspection feedback](https://github.com/jakbuczarnecki/intergrax/issues/188).

---

## 45-60 minute validation review

Goal: evaluate whether Intergrax is worth deeper technical feedback, selected integration discussion, or design-partner discovery.

Choose the path that matches your interest:

| Interest | Read | Feedback path |
|----------|------|---------------|
| Boundary events, receipts, attestation, auditability | [BoundaryAttest Attestation PoC](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) | [#189](https://github.com/jakbuczarnecki/intergrax/issues/189) |
| Trace/evidence export and observability | [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md), README trace/evidence sections | [#188](https://github.com/jakbuczarnecki/intergrax/issues/188), [#192](https://github.com/jakbuczarnecki/intergrax/issues/192) |
| Local/private knowledge workflows and controlled RAG | [Local Knowledge Workspace alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) | [#193](https://github.com/jakbuczarnecki/intergrax/issues/193) |
| Governed agent applications beyond demos | [USE_CASES.md](USE_CASES.md), [PARTNERS.md](PARTNERS.md) | [#190](https://github.com/jakbuczarnecki/intergrax/issues/190) |
| MCP or controlled tool/task surfaces | [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) | [#194](https://github.com/jakbuczarnecki/intergrax/issues/194) |

Useful review questions:

- Which current agent/platform problem does Intergrax address clearly?
- Which runtime boundary is most valuable: policy, trace, HITL, tools, RAG, memory, Nexus, or application host?
- What is missing for a serious technical evaluation?
- Which proof path should be easier?
- Which claims are unclear or too strong?
- What should not happen without explicit approval?

---

## What to report

High-value feedback is concrete and scoped.

Good feedback includes:

- the exact document, section, command, or issue you evaluated;
- what you expected to happen;
- what actually happened;
- what was unclear;
- your environment when reporting local proof-path friction;
- which use case or validation track you were evaluating;
- whether you are reporting documentation clarity, proof-path friction, integration feedback, or design-partner interest.

Avoid broad feature requests unless they are tied to a specific use case and current validation track.

Use [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) to choose the right curated issue.

---

## What not to assume

Do not infer that this repository offers:

- hosted SaaS;
- open-source license rights;
- production support or SLA;
- certification, compliance, legal attestation, or security approval;
- free commercial use;
- permission to redistribute, derive from, incorporate, or productize Intergrax;
- acceptance of every proposed integration or partnership;
- a broad public feature-request backlog.

For commercial licensing, production use, partnership, or permission requests, contact the maintainer directly. See [PARTNERS.md](PARTNERS.md) and [COLLABORATION.md](COLLABORATION.md).

---

## Where to go next

| Need | Next document |
|------|---------------|
| Repository overview | [README.md](README.md) |
| Common questions | [FAQ.md](FAQ.md) |
| Use-case fit | [USE_CASES.md](USE_CASES.md) |
| Partner or design-partner fit | [PARTNERS.md](PARTNERS.md) |
| License and collaboration boundaries | [COLLABORATION.md](COLLABORATION.md), [LICENSE](LICENSE) |
| Public feedback routing | [Public Issue Index](docs/public-adoption/PUBLIC_ISSUE_INDEX.md) |
| Attestation validation | [BoundaryAttest Attestation PoC](docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) |
| Local/private knowledge validation | [Local Knowledge Workspace alpha](docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) |


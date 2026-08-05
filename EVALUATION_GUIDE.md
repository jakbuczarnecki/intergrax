<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Evaluation Guide

Intergrax is **source-available** and under **active R&D**. **LKW** is the **Primary product proof** — classified as **Backend Product Alpha / MVP** and **PARTIAL**. **Real-user validation** and **commercial validation** are incomplete. The [LICENSE](LICENSE) is legally authoritative for permitted use.

This guide provides time-boxed, reader-facing evaluation paths for technical reviewers, design partners, and integration builders; it does not claim that external validation has occurred.

Production use, commercial use, hosted services, redistribution, and commercial derivative works require explicit written permission. See [LICENSE](LICENSE), [COLLABORATION.md](COLLABORATION.md), and [PARTNERS.md](PARTNERS.md).

---

## Choose the route

| Goal | Start here |
|------|------------|
| Begin building or extending an application | [Builder Quick Start](BUILDER_QUICKSTART.md) |
| Run a bounded evaluation | This Evaluation Guide |
| Try the LKW product | [LKW Quick Start](applications/local_workspace_application/docs/QUICKSTART.md) |

Builder onboarding ≠ product trial ≠ broader platform evaluation.

---

## At a glance

| Time | Start here | Goal |
|------|------------|------|
| 5 minutes | [README.md](README.md) + [FAQ.md](FAQ.md) | Understand what Intergrax is and is not |
| 15 minutes | [WHY_INTERGRAX.md](WHY_INTERGRAX.md) + [USE_CASES.md](USE_CASES.md) + [ROADMAP.md](ROADMAP.md) | Understand problem, fit and direction |
| 30 minutes | [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md) + [PROOFS.md](PROOFS.md) | Choose a bounded builder or technical evaluation path |
| 45–60 minutes | Choose LKW Platform Proof, Token Optimization guide, architecture review or partner/pilot review | Deep evaluation of one surface |

---

## Who this guide is for

Use this guide if you are:

- evaluating whether Intergrax addresses a governed-agent or knowledge-workflow problem;
- reviewing the public architecture and responsibility boundaries;
- checking whether a documented proof path is understandable;
- assessing trace, evidence, policy, human approval, retrieval and grounding,
  memory, tools, or orchestration;
- considering design-partner or selected integration feedback;
- deciding which evaluation path fits your available time.

If you need a finished SaaS, production support, an open-source contribution model, or unrestricted commercial use, start with [FAQ.md](FAQ.md), [COLLABORATION.md](COLLABORATION.md), and [LICENSE](LICENSE).

---

## 5-minute orientation

Goal: decide whether Intergrax is relevant before reading deeply.

Read:

1. [README.md](README.md) — repository overview, LKW product proof, and maturity boundaries.
2. [FAQ.md](FAQ.md) — common questions about license, status, collaboration, and public use.

Check whether the following statement is clear:

> Intergrax is source-available for evaluation and collaboration, not open source, not a finished SaaS, and not a production-readiness claim.

Useful feedback after this pass:

- Is the repository positioning clear?
- Is the license/collaboration boundary clear?
- Do you understand where to start?

---

## 15-minute problem, fit and direction

Goal: understand what problem Intergrax addresses and whether it fits your context.

Read:

1. [WHY_INTERGRAX.md](WHY_INTERGRAX.md) — problem, value, and audience.
2. [USE_CASES.md](USE_CASES.md) — use-case fit and applicability.
3. [ROADMAP.md](ROADMAP.md) — product-validation direction and outcome gates.

Questions to answer:

- Does the problem statement resonate with your work?
- Which use case is most relevant?
- What would you need to see next to justify deeper evaluation?

---

## 30-minute bounded technical evaluation

Goal: test whether the local evaluation path is understandable and reproducible.

Start from:

- [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md) — route selection and prerequisites.
- [README.md](README.md) — product overview and LKW workflow.
- [PROOFS.md](PROOFS.md) — current proof status and claim boundaries.

A simplified public LKW trial is available through the supported product quickstart: [applications/local_workspace_application/docs/QUICKSTART.md](applications/local_workspace_application/docs/QUICKSTART.md).

This path is indexed-only LKW. First-run duration varies; timing is not yet externally validated. Deeper technical proof remains separate — see [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) and [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md).

Typical local flow:

```bash
uv sync --extra dev
uv run intergrax doctor
uv run pytest -m gate -q
```

Evaluate:

- Are prerequisites clear?
- Does install work as written?
- Does `intergrax doctor` provide useful feedback?
- Are gate tests discoverable and understandable?
- Is the proof status visible enough?
- Are trace/evidence outputs understandable when inspected?

Do not assume that passing local checks grants production permission, commercial use, support, certification, or compliance status.

---

## 45–60 minute deep evaluation

Goal: evaluate whether Intergrax is worth deeper technical feedback, integration discussion, or pilot discovery.

Choose the path that matches your interest:

| Interest | Read |
|----------|------|
| LKW product proof and workflow | [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| Token Optimization capability | [Token Optimization guide](docs/features/token_optimization/README.md) |
| High-level architecture | [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) |
| Partner or pilot fit | [PARTNERS.md](PARTNERS.md) |
| Permission and collaboration boundaries | [COLLABORATION.md](COLLABORATION.md) |

Useful review questions:

- Which current problem does Intergrax address clearly?
- Which product or execution boundary is most valuable?
- What is missing for a serious technical evaluation?
- Which proof path should be easier?
- Which claims are unclear or too strong?

---

## What to report

High-value feedback is concrete and scoped.

A useful feedback record must include:

- exact commit;
- environment;
- path followed;
- expected result;
- observed result;
- evidence;
- blocker or confusion.

Route ordinary evaluation feedback through [COLLABORATION.md](COLLABORATION.md).

Good feedback also includes:

- the exact document, section, or command you evaluated;
- which use case or evaluation track you were following;
- whether you are reporting documentation clarity, evaluation-path friction, integration feedback, or design-partner interest.

Avoid broad feature requests unless they are tied to a specific use case and current evaluation track.

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
- completed real-user or commercial validation.

For commercial licensing, production use, partnership, or permission requests, contact the maintainer directly. See [PARTNERS.md](PARTNERS.md) and [COLLABORATION.md](COLLABORATION.md).

---

## Where to go next

| Need | Next document |
|------|---------------|
| Repository overview | [README.md](README.md) |
| Common questions | [FAQ.md](FAQ.md) |
| Problem and value | [WHY_INTERGRAX.md](WHY_INTERGRAX.md) |
| Use-case fit | [USE_CASES.md](USE_CASES.md) |
| Product-validation direction | [ROADMAP.md](ROADMAP.md) |
| Evaluation and building routes | [BUILD_WITH_INTERGRAX.md](BUILD_WITH_INTERGRAX.md) |
| Proof status | [PROOFS.md](PROOFS.md) |
| Architecture overview | [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) |
| LKW product proof | [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) |
| Token Optimization | [Token Optimization guide](docs/features/token_optimization/README.md) |
| Partner or pilot fit | [PARTNERS.md](PARTNERS.md) |
| License and collaboration boundaries | [COLLABORATION.md](COLLABORATION.md), [LICENSE](LICENSE) |
| Public reader navigation | [Public Documentation Map](docs/PUBLIC_DOCUMENTATION_MAP.md) |
| Technical/developer navigation | [Documentation Map](docs/DOCUMENTATION_MAP.md) |

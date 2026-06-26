<!--
© Artur Czarnecki. All rights reserved.
Intergrax framework – proprietary and confidential.
Use, modification, or distribution without written permission is prohibited.
-->

# Intergrax Outreach Kit

This document contains maintainer-facing outreach drafts for introducing Intergrax to technical reviewers, design partners, and integration builders.

These drafts are intentionally conservative. They do not create commitments, license grants, support obligations, certification claims, or partnership terms. Adapt wording as needed, but keep the guardrails below.

Always link readers back to [README.md](../../README.md), [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md), [USE_CASES.md](../../USE_CASES.md), [PARTNERS.md](../../PARTNERS.md), [FAQ.md](../../FAQ.md), and [COLLABORATION.md](../../COLLABORATION.md) for authoritative repository context and permission boundaries.
For near-term public adoption priorities, point to [ROADMAP.md](../../ROADMAP.md).

---

## Positioning guardrails

- Say **source-available**, not **open source**.
- Say **technical integration validation**, not **certification**.
- Say **alpha/product-validation direction**, not **finished product**.
- Say **design-partner discovery**, not **guaranteed partnership**.
- Say **proof-path feedback**, not **production support**.
- Route time-boxed technical reviewers to [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md).
- Route use-case fit questions to [USE_CASES.md](../../USE_CASES.md).
- Route design-partner or partner-fit discussions to [PARTNERS.md](../../PARTNERS.md).
- For commercial or production use, point to maintainer contact and [COLLABORATION.md](../../COLLABORATION.md).

---

## Short repository intro

### One sentence

Intergrax is a source-available Harness AI / Agent OS for governed agent applications: agents decide, the harness executes under policy, Nexus orchestrates, and trace/evidence surfaces make runs inspectable.

### Short paragraph

Intergrax is built for governed agent applications, not just demo agents. It focuses on runtime boundaries, policy-controlled tools, orchestration, RAG/memory, trace/evidence, and application hosts. Current public goals are technical review, proof-path feedback, selected integration feedback, and design-partner discovery — not a finished SaaS or open-source framework. [README.md](../../README.md) is the repository overview; [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md) gives a time-boxed reviewer path; [USE_CASES.md](../../USE_CASES.md) maps problem/use-case fit; [PARTNERS.md](../../PARTNERS.md) explains design-partner and partner-fit discussion boundaries.

---

## LinkedIn / public post draft

We published a source-available Intergrax repository for technical review of a Harness AI / Agent OS for governed agent applications — where agents decide, the harness executes under policy, Nexus orchestrates, and trace/evidence surfaces make runs inspectable.

This is not open source and not a production product claim. It is a structured evaluation surface for architects and platform engineers who care about runtime boundaries, policy-controlled tools, and external verification hooks.

The README includes a proof path, FAQ, and collaboration boundaries. We also published a BoundaryAttest attestation case study and a Local Knowledge Workspace alpha narrative as product-validation directions.

We are looking for technical feedback and qualified design-partner discovery: Does the Harness AI model make sense? Is the first-run path clear? What is confusing about license or collaboration boundaries?

Links: [README](https://github.com/jakbuczarnecki/intergrax#start-here) · [Evaluation Guide](https://github.com/jakbuczarnecki/intergrax/blob/main/EVALUATION_GUIDE.md) · [Use Cases](https://github.com/jakbuczarnecki/intergrax/blob/main/USE_CASES.md) · [Partners](https://github.com/jakbuczarnecki/intergrax/blob/main/PARTNERS.md) · [FAQ](https://github.com/jakbuczarnecki/intergrax/blob/main/FAQ.md) · [Collaboration](https://github.com/jakbuczarnecki/intergrax/blob/main/COLLABORATION.md) · [BoundaryAttest case study](https://github.com/jakbuczarnecki/intergrax/blob/main/docs/case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) · [Local Knowledge Workspace alpha](https://github.com/jakbuczarnecki/intergrax/blob/main/docs/product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md)

Commercial or production use requires explicit permission — see [COLLABORATION.md](https://github.com/jakbuczarnecki/intergrax/blob/main/COLLABORATION.md).

---

## Technical reviewer DM / email

Subject: Intergrax — source-available Harness AI repo — technical review request

Hi [Name],

I maintain Intergrax, a source-available Harness AI / Agent OS for governed agent applications. The public repo is for evaluation and proof-path feedback, not open-source redistribution or production support.

Would you skim the README **Start here** section and follow the time-boxed review flow in [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md)? I am especially interested in whether the Harness AI separation (agent decides / harness executes / Nexus orchestrates) reads clearly, and whether trace/evidence surfaces are understandable.

If attestation is relevant, see the [BoundaryAttest case study](../case-studies/BOUNDARYATTEST_ATTESTATION_POC.md).

Questions:

- Does the Harness AI model make sense?
- Is the first-run path clear?
- Are license and collaboration boundaries clear?
- What is confusing?

Context: [README](../../README.md) · [EVALUATION_GUIDE](../../EVALUATION_GUIDE.md) · [USE_CASES](../../USE_CASES.md) · [FAQ](../../FAQ.md) · [COLLABORATION](../../COLLABORATION.md).

Thanks — even brief notes help.

[Your name]

---

## Attestation / observability builder outreach

Subject: Intergrax boundary attestation POC — integration feedback request

Hi [Name],

I am sharing Intergrax as a source-available Harness AI runtime with an attestation-oriented proof path. It is for technical integration validation and feedback, not certification or compliance claims.

Two entry points:

- [BoundaryAttest attestation case study](../case-studies/BOUNDARYATTEST_ATTESTATION_POC.md)
- [attestation_demo application README](../../applications/attestation_demo/README.md)

The POC separates Intergrax runtime claims from an external client-observed wrapper around boundary events. I would value your read on whether that separation is clear and what would be needed to integrate with a real boundary event stream in your environment.

Questions:

- Is the separation between Intergrax runtime claim and external client-observed wrapper clear?
- What would you need to integrate with a boundary event stream?
- What should be diagrammed or documented better?

License and collaboration boundaries: [COLLABORATION.md](../../COLLABORATION.md). Repository overview: [README](../../README.md).

Thanks for any structured feedback.

[Your name]

---

## Local Knowledge Workspace design-partner outreach

Subject: Local Knowledge Workspace alpha — design-partner discovery

Hi [Name],

Intergrax includes a Local Knowledge Workspace alpha direction for product-validation feedback — not a finished local knowledge product or SaaS offer.

Entry points:

- [Local Knowledge Workspace alpha narrative](../product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md)
- [local_workspace_application architecture](../../applications/local_workspace_application/docs/ARCHITECTURE.md)
- [USE_CASES.md](../../USE_CASES.md)
- [PARTNERS.md](../../PARTNERS.md)
- [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md) (optional time-boxed review path)

We are in design-partner discovery: qualified teams willing to describe a local document workflow worth validating and report friction against the alpha narrative. This does not imply partnership terms, support, or production permission.

Questions:

- What local document workflow would be worth validating in your context?
- Which file types matter most?
- What outputs would be valuable?
- What should never happen without explicit approval?

See [COLLABORATION.md](../../COLLABORATION.md) for permission boundaries and [ROADMAP.md](../../ROADMAP.md) for public adoption priorities.

Happy to share more context if this matches your evaluation interests.

[Your name]

---

## Governed agent application builder outreach

Subject: Intergrax — governed agent applications — builder feedback request

Hi [Name],

Intergrax is a source-available Harness AI / Agent OS aimed at governed agent applications: policy-controlled tools, orchestration, trace/evidence, RAG/memory, and Tier-3 application hosts. The public repo supports proof-path evaluation and selected integration feedback — not a drop-in replacement for other agent frameworks or a production-ready claim.

If you are building agent-backed applications beyond demos, I would appreciate your read on where prototypes typically break down and whether a governed harness is worth testing in your stack.

Questions:

- Where do your current agent prototypes break down?
- Do you need policy, HITL, trace, evidence, or evaluation surfaces?
- What would make a governed harness worth testing?

Start paths: [README](../../README.md) · [EVALUATION_GUIDE](../../EVALUATION_GUIDE.md) · [USE_CASES](../../USE_CASES.md) · [PARTNERS](../../PARTNERS.md) · [FAQ](../../FAQ.md) · [Public Issue Index](PUBLIC_ISSUE_INDEX.md). Commercial or production use: [COLLABORATION](../../COLLABORATION.md).

Thanks for any concrete feedback.

[Your name]

---

## What not to say

| Avoid | Safer alternative |
|-------|-------------------|
| open-source agent framework | source-available Harness AI runtime for evaluation |
| production-ready agent OS | available for proof-path evaluation and design-partner feedback |
| certified attestation | technical integration validation via BoundaryAttest POC |
| compliance-ready | governance-oriented trace/evidence surfaces for review — not compliance certification |
| secure by default for all deployments | policy-controlled execution model — deployment security remains operator responsibility |
| free to use commercially | source-available for evaluation; commercial/production use requires explicit permission per [COLLABORATION.md](../../COLLABORATION.md) |
| drop-in replacement for LangChain/CrewAI/etc. | complementary Harness AI / Agent OS with explicit runtime boundaries |
| finished local knowledge product | Local Knowledge Workspace alpha / product-validation direction |
| guaranteed partnership | design-partner discovery under [COLLABORATION.md](../../COLLABORATION.md) |
| we support all use cases | proof-path feedback and selected integration feedback via curated public issues |
| we are looking for partners | we are doing design-partner discovery under [PARTNERS.md](../../PARTNERS.md) and [COLLABORATION.md](../../COLLABORATION.md) |
| this solves all agent use cases | [USE_CASES.md](../../USE_CASES.md) maps current validation areas and partner-fit discussions |
| review the repo whenever you have time | [EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md) provides a time-boxed 5/15/30/60-minute review path |

---

## Related documents

| Document | Purpose |
|----------|---------|
| [../../README.md](../../README.md) | Repository overview and start paths |
| [../../EVALUATION_GUIDE.md](../../EVALUATION_GUIDE.md) | Time-boxed evaluation guide for reviewers |
| [../../USE_CASES.md](../../USE_CASES.md) | Use-case map and validation paths |
| [../../PARTNERS.md](../../PARTNERS.md) | Partner and design-partner brief |
| [../../FAQ.md](../../FAQ.md) | Common external-reader questions |
| [../../COLLABORATION.md](../../COLLABORATION.md) | Collaboration and permission model |
| [../../ROADMAP.md](../../ROADMAP.md) | Public adoption roadmap |
| [PUBLIC_ISSUE_INDEX.md](PUBLIC_ISSUE_INDEX.md) | Active curated public issues |
| [../case-studies/BOUNDARYATTEST_ATTESTATION_POC.md](../case-studies/BOUNDARYATTEST_ATTESTATION_POC.md) | Attestation case study |
| [../product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md](../product-validation/LOCAL_KNOWLEDGE_WORKSPACE_ALPHA.md) | LKW alpha narrative |

<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Outreach Kit

This document contains **maintainer-facing** templates for recruiting independent validation participants and requesting structured feedback.

**External reader validation is not complete.** These invitations do not create partnership, support, production permission, or commercial rights. Participants should receive no detailed project explanation before Track A first-contact tasks.

Canonical public positioning is defined in [INTERGRAX_PUBLIC_POSITIONING.md](../../overview/INTERGRAX_PUBLIC_POSITIONING.md). All outreach drafts must remain consistent with it.

---

## Positioning guardrails

- Say **source-available**, not **open source**.
- Say **active R&D**, not **production-ready**.
- Say **Active reference product** for LKW, not **finished product**.
- Say **Backend Product Alpha / MVP** and **PARTIAL** for LKW maturity.
- Say **Featured platform-capability proof** and **PARTIAL** for Token Optimization.
- Say **real-user validation incomplete** and **commercial validation incomplete**.
- Say **design-partner discovery**, not **guaranteed partnership**.
- Say **proof-path feedback**, not **production support**.
- Route time-boxed technical reviewers to [docs/project/builders/EVALUATION_GUIDE.md](../../builders/EVALUATION_GUIDE.md).
- Route use-case fit questions to [docs/project/overview/USE_CASES.md](../../overview/USE_CASES.md).
- Route pilot or partner-fit discussions to [docs/project/community/PARTNERS.md](../../community/PARTNERS.md).
- For commercial or production use, point to maintainer contact and [docs/project/community/COLLABORATION.md](../../community/COLLABORATION.md).

---

## Participant eligibility

Three primary cohorts (one per participant):

| Cohort | Minimum per wave | Profile |
|--------|------------------|---------|
| Unfamiliar technical readers | 2 | Engineers or architects who have not authored or reviewed the public-documentation rewrite |
| Potential LKW or governed-knowledge workflow users | 2 | People evaluating local or governed knowledge workflows |
| Architecture, platform, governance or observability reviewers | 1 | People reviewing architecture, governance, or evidence surfaces |

Independence requirements:

- maintainers and public-documentation authors do not count;
- participants should not receive a detailed project explanation before Track A;
- prior familiarity must be recorded;
- cohort is qualitative, not statistically representative.

---

## Blind first-contact invitation

Subject: 15-minute documentation review — independent feedback request

Hi,

I maintain a source-available repository called Intergrax and am running a structured documentation comprehension review. I would value 15 minutes of your time.

Repository:
<pinned-repository-root-url>

Pinned revision:
<pinned-ref>

Use only the public repository documentation. Think aloud as you navigate. I will not explain the project until you complete the tasks.

This is not a request for endorsement, partnership, or production permission. Participation is voluntary.

If you are willing, I will send the task list when you confirm availability.

Thanks,
[Your name]

---

## Technical evaluation invitation

Subject: 30–60 minute technical evaluation — documentation navigation review

Hi,

Thank you for completing the first-contact documentation review. If you are willing to continue, I would appreciate a 30–60 minute technical evaluation pass.

Evaluation guide:
<pinned-evaluation-guide-url>

Pinned revision:
<pinned-ref>

You may inspect or run documented evaluation and proof paths at your own pace.

When reporting, please include:

- exact commit or tag;
- environment (OS, Python version, tools);
- path followed;
- expected result;
- observed result;
- evidence (screenshots or logs kept privately unless you consent to sharing).

If your environment is unsuitable for execution, record `NOT_RUN` with the reason.

This does not grant production permission, commercial rights, or partnership terms.

Thanks,
[Your name]

---

## LKW workflow-fit invitation

Subject: LKW workflow-fit review — governed knowledge workflow evaluation

Hi,

You indicated interest in local or governed knowledge workflows. I would value a structured review of whether the public documentation helps you evaluate a concrete workflow fit.

Local Knowledge Workspace (LKW) is Intergrax's **active reference product** — a **Backend Product Alpha / MVP** with **PARTIAL** public status. Real-user and commercial validation are incomplete. LKW is not a finished product or commercially validated offering. For bounded evidence, see the LKW Platform Proof and proof dashboard below.

LKW Platform Proof:
<pinned-lkw-proof-url>

Proof dashboard:
<pinned-proofs-url>

Use Cases:
<pinned-use-cases-url>

Partners and Pilots:
<pinned-partners-url>

Pinned revision:
<pinned-ref>

I would like you to describe:

- one concrete knowledge workflow you would evaluate;
- users and data sources involved;
- whether it can stay evaluation-only;
- when written permission would be required;
- what evidence would build trust.

Positive interest does not equal a partnership, production permission, or commercial validation.

Thanks,
[Your name]

---

## Architecture and governance review invitation

Subject: Architecture and governance review — platform boundary evaluation

Hi,

I would value your review of Intergrax's public architecture and evidence surfaces. This is a documentation comprehension and navigation review, not a security audit or compliance certification.

Architecture overview:
<pinned-architecture-url>

Proof dashboard:
<pinned-proofs-url>

Build and evaluation routes:
<pinned-build-url>

Pinned revision:
<pinned-ref>

Questions:

- Are responsibility boundaries clear?
- Is the proof-status model understandable?
- What governance or observability surfaces need clearer documentation?
- Which claims feel too strong or unclear?

License and collaboration boundaries are described in the public documentation at the pinned revision.

This does not grant production permission or imply partnership.

Thanks,
[Your name]

---

## Moderator opening

Use this exact neutral opening at the start of every Track A session:

```text
Start at the repository README. Please use only the public repository
documentation. Explain what you think as you navigate. I will not explain
the project until the tasks are complete.
```

---

## Post-session questions

After task completion, ask:

1. What did you think Intergrax was?
2. What was hardest to understand?
3. Where did you expect to click?
4. Which claim felt too strong or unclear?
5. What would justify deeper evaluation?
6. What would stop you from continuing?

---

## Privacy and quotation

- Participation is voluntary.
- Raw notes containing personal data are not committed to the repository.
- Direct quotes require explicit participant permission.
- Participants should not share confidential workflow data or production data.
- Anonymized aggregate findings may be published after review.
- Security issues remain routed through [SECURITY.md](../../../../SECURITY.md), not validation notes.

---

## What not to say

| Avoid | Safer alternative |
|-------|-------------------|
| open source | source-available for evaluation and collaboration |
| production ready / enterprise ready | active R&D; available for bounded evaluation |
| commercially validated | commercial validation incomplete |
| finished LKW product | LKW Backend Product Alpha / MVP — PARTIAL |
| validated by users | real-user validation incomplete |
| guaranteed partnership | design-partner discovery under [docs/project/community/PARTNERS.md](../../community/PARTNERS.md) |
| certification / certified | technical integration validation — not certification |
| compliance-ready | governance-oriented surfaces for review — not compliance approval |
| universal usability | qualitative documentation comprehension review |
| universal Token Optimization savings | Featured platform-capability proof — PARTIAL; no universal savings claim |
| free to use commercially | evaluation permitted under [LICENSE](../../../../LICENSE); commercial use requires permission |
| we support all use cases | [docs/project/overview/USE_CASES.md](../../overview/USE_CASES.md) maps current validation areas |

---

## Related documents

| Document | Purpose |
|----------|---------|
| [EXTERNAL_READER_VALIDATION_PROTOCOL.md](EXTERNAL_READER_VALIDATION_PROTOCOL.md) | Validation methodology and completion gates |
| [PUBLIC_LAUNCH_CHECKLIST.md](PUBLIC_LAUNCH_CHECKLIST.md) | Pre-session readiness checklist |
| [INTERGRAX_PUBLIC_POSITIONING.md](../../overview/INTERGRAX_PUBLIC_POSITIONING.md) | Canonical public positioning contract |
| [../../EVALUATION_GUIDE.md](../../builders/EVALUATION_GUIDE.md) | Reader-facing evaluation paths |
| [../../README.md](../../../../README.md) | Repository overview |
| [../../COLLABORATION.md](../../community/COLLABORATION.md) | Collaboration and permission model |

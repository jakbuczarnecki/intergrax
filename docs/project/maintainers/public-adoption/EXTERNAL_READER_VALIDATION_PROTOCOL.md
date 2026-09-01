<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# External Reader Validation Protocol

This protocol defines a repeatable, privacy-conscious method for testing public-documentation comprehension and navigation with independent external readers.

**Creating this protocol does not mean external validation is complete.**

Real-user product validation remains incomplete. Commercial validation remains incomplete.

This protocol is not usability certification, market validation, product validation, commercial validation, legal review, security review, or proof of operational production-readiness.

---

## At a glance

| Item | Status |
|------|--------|
| Protocol status | READY_TO_RUN |
| External reader validation status | NOT_STARTED |
| Minimum completed sessions | 5 |
| Primary surface | root README and linked public reader documents |
| Raw personal data | must not be committed |
| Results | may be claimed only after real sessions and aggregate review |

---

## What this validates

This protocol validates:

- first-contact positioning comprehension;
- LKW role and maturity comprehension;
- navigation to use cases, proof, evaluation and architecture;
- source-available and permission-boundary comprehension;
- pilot and collaboration-route discoverability;
- reader ability to choose a next action;
- documented technical evaluation path discoverability.

---

## What this does not validate

This protocol does not validate:

- product usefulness in real work;
- real-user product adoption;
- market demand;
- commercial willingness to pay;
- operational production-readiness;
- security or compliance;
- technical correctness beyond documented proof;
- universal usability;
- statistical representativeness;
- legal advice.

---

## Reviewer cohorts

A validation wave requires a minimum of **five completed independent sessions**:

| Cohort | Minimum |
|--------|---------|
| Technical readers unfamiliar with the repository | at least 2 |
| Potential LKW or governed-knowledge workflow users | at least 2 |
| Architecture, platform, governance or observability reviewer | at least 1 |

Rules:

- one participant is counted in one primary cohort only;
- maintainers and people who authored or reviewed the public-documentation rewrite do not count;
- participants should not receive a detailed project explanation before the first-contact tasks;
- prior familiarity must be recorded;
- the cohort is qualitative and not statistically representative.

---

## Validation tracks

### Track A - first-contact comprehension

Mandatory for every participant.

| Property | Value |
|----------|-------|
| Time box | 15 minutes |
| Starting point | root README at one pinned commit or tag |

### Track B - technical evaluation navigation

Required for at least two completed sessions before describing the technical evaluation path as externally validated.

| Property | Value |
|----------|-------|
| Time box | 30–60 minutes |
| Scope | participant may inspect or run documented evaluation/proof paths |

### Track C - LKW workflow and pilot fit

Used for potential LKW or governed-knowledge users.

This validates whether the reader can describe a concrete workflow and locate the evaluation-versus-operational pilot boundary. It does not validate product usefulness.

---

## Standardized first-contact tasks

Use this exact neutral setup:

```text
Start at the repository README. Please use only the public repository
documentation. Explain what you think as you navigate. I will not explain
the project until the tasks are complete.
```

Mandatory tasks (do not show expected answers to the participant):

1. Explain Intergrax in one sentence.
2. Identify the strongest current product proof.
3. State the maturity of that product proof.
4. Find where to decide whether a use case fits.
5. Find where to begin technical evaluation.
6. Determine whether production or commercial use is automatically permitted.
7. Find how to discuss a pilot or partnership.
8. State the next action you would take.

---

## Technical evaluation tasks

For Track B, the participant should:

- locate prerequisites;
- locate the current proof-status source;
- locate the bounded evaluation path;
- identify what passing the path does not prove;
- optionally execute the documented path;
- capture the exact commit, environment, expected result and observed result;
- identify the appropriate feedback route.

Do not require execution when the environment is unsuitable. Record `NOT_RUN` with reason.

---

## LKW workflow-fit tasks

For Track C, require the participant to:

- describe one concrete knowledge workflow;
- identify users and data sources;
- decide whether it can stay evaluation-only;
- identify when written permission is required;
- locate required pilot-brief fields;
- identify useful success criteria;
- state what evidence would build trust.

Do not treat positive interest as product or commercial validation.

---

## Facilitation rules

Every participant-facing repository and document URL in one validation wave must resolve to the same pinned commit or immutable tag. A moving default-branch URL is not sufficient evidence of the tested revision.

Pinned-revision rules (frozen):

- the pinned ref is selected before recruitment;
- all Track A, B and C participant-facing URLs resolve to that ref;
- changing the pinned ref creates a new validation wave unless explicitly recorded as a rerun;
- the ref written in the session record must match the content actually shown;
- a moving `main`, `development` or repository-root URL is insufficient.

Participant-facing URL placeholders for a validation wave:

```text
<pinned-ref>
<pinned-repository-root-url>
<pinned-evaluation-guide-url>
<pinned-lkw-proof-url>
<pinned-use-cases-url>
<pinned-partners-url>
<pinned-architecture-url>
<pinned-proofs-url>
<pinned-build-url>
```

Session facilitation:

- one pinned repository revision for a validation wave;
- the same mandatory prompts for all participants;
- no coaching before task completion;
- no expected answers distributed to participants;
- no explanation of Intergrax before the participant answers;
- no navigation hints unless an intervention is recorded;
- no correction of wrong conclusions until the end;
- record first route and dead ends;
- allow think-aloud commentary;
- keep optional technical execution separate from comprehension scoring;
- do not pressure participants to endorse the project;
- do not recruit only contributors or existing supporters.

---

## Task scoring

| Score | Meaning |
|-------|---------|
| PASS | completed without moderator intervention and with materially correct understanding |
| FRICTION | completed but with avoidable confusion, dead end or minor intervention |
| FAIL | not completed or materially incorrect conclusion |
| NOT_RUN | intentionally not attempted with a recorded reason |

---

## Finding severity

| Severity | Meaning |
|----------|---------|
| CRITICAL | legal, permission, maturity or claim misunderstanding that could materially mislead a reader |
| MAJOR | failure to understand positioning, LKW role or a core navigation route |
| MINOR | local clarity, wording or navigation friction |
| OBSERVATION | useful preference or idea without a demonstrated failure |

---

## Evidence capture

Require for every session:

- anonymized session ID;
- date;
- pinned commit or tag;
- participant cohort;
- prior familiarity;
- validation tracks;
- environment for technical execution;
- task result for every mandatory task;
- participant's own one-sentence description;
- identified product proof and maturity;
- wrong or uncertain conclusions;
- first navigation route;
- dead ends;
- moderator interventions;
- broken links;
- technical errors;
- finding severity;
- follow-up notes;
- consent status for any direct quotation.

Privacy boundaries:

- do not commit names, email addresses, employer-confidential data or raw recordings;
- keep raw notes private when they contain personal data;
- only anonymized and sanitized aggregate findings may be committed;
- quotes require participant permission and removal of identifying details.

---

## Completion gates

External reader validation may move from `IN_PROGRESS` to `VALIDATED_FOR_BOUNDED_OUTREACH` only when:

1. at least five completed sessions exist;
2. all required cohorts are represented;
3. every participant attempted all Track A tasks;
4. at least 80% of completed sessions pass each mandatory Track A task;
5. no unresolved `CRITICAL` finding remains;
6. all `MAJOR` findings are fixed or explicitly accepted with rationale;
7. core links used in the tasks are not broken;
8. at least two Track B sessions are complete before claiming the technical evaluation route is externally validated;
9. the pinned revision is recorded;
10. an anonymized aggregate summary has been reviewed;
11. failed or materially changed tasks are rerun after corrections.

`VALIDATED_FOR_BOUNDED_OUTREACH` does not mean: product validated, real-user validation complete, commercially validated, production-ready, certified, or universally usable.

---

## Validation states

| State | Meaning |
|-------|---------|
| NOT_STARTED | no real sessions completed |
| READY_TO_RUN | protocol frozen; sessions may begin |
| IN_PROGRESS | at least one session completed |
| CHANGES_REQUIRED | findings require documentation correction |
| VALIDATED_FOR_BOUNDED_OUTREACH | completion gates met |
| BLOCKED | cannot proceed (e.g. remote drift, missing prerequisites) |

Current state after protocol creation:

| Item | State |
|------|-------|
| Protocol | READY_TO_RUN |
| External validation | NOT_STARTED |

---

## Session record template

Copy and complete for each session. Do not populate with fictional results.

```markdown
# Session record - <session-id>

| Field | Value |
|-------|-------|
| Session ID | <session-id> |
| Validation wave | <wave-id> |
| Date | <date> |
| Pinned commit or immutable tag | <pinned-ref> |
| Participant-facing root URL | <pinned-repository-root-url> |
| Primary cohort | <cohort> |
| Prior familiarity | <none / brief / detailed> |
| Tracks attempted | <A / A+B / A+C / A+B+C> |
| Track B environment | <OS, Python, tools - or N/A> |
| Consent for quotation | <yes / no / partial> |

## Track A tasks

| # | Task | Result | Notes |
|---|------|--------|-------|
| 1 | Explain Intergrax in one sentence | PASS/FRICTION/FAIL/NOT_RUN | |
| 2 | Identify strongest product proof | PASS/FRICTION/FAIL/NOT_RUN | |
| 3 | State proof maturity | PASS/FRICTION/FAIL/NOT_RUN | |
| 4 | Find use-case fit route | PASS/FRICTION/FAIL/NOT_RUN | |
| 5 | Find technical evaluation start | PASS/FRICTION/FAIL/NOT_RUN | |
| 6 | Production/commercial permission | PASS/FRICTION/FAIL/NOT_RUN | |
| 7 | Find pilot or partnership route | PASS/FRICTION/FAIL/NOT_RUN | |
| 8 | State next action | PASS/FRICTION/FAIL/NOT_RUN | |

## Evidence

| Field | Value |
|-------|-------|
| Participant one-sentence description | |
| Identified strongest product proof | |
| Stated product-proof maturity | |
| Wrong or uncertain conclusions | |
| First navigation route | |
| Dead ends | |
| Moderator interventions | |
| Broken links | |
| Technical errors | |
| Follow-up notes | |

## Findings

| Severity | Finding | Evidence | Resolution | Rerun required |
|----------|---------|----------|------------|----------------|
| CRITICAL/MAJOR/MINOR/OBSERVATION | | | | yes / no |
```

---

## Aggregate summary template

Copy after a validation wave. Do not populate with fictional results.

```markdown
# Aggregate summary - wave <wave-id>

Pinned ref: <pinned-ref> · Dates: <start>–<end> · Sessions: <count>

Cohort counts: unfamiliar technical <n> · LKW/governed-knowledge <n> · architecture/governance <n>

Gates: ≥5 sessions · cohorts covered · all Track A attempted · ≥80% pass/task ·
no unresolved CRITICAL · MAJOR fixed/accepted · links OK · ≥2 Track B · ref recorded ·
aggregate reviewed · rerun after corrections

Task counts (PASS/FRICTION/FAIL/NOT_RUN): tasks 1–8 - fill per task

Findings: critical unresolved · major unresolved · resolved · rerun results

Decision: NOT_STARTED / IN_PROGRESS / CHANGES_REQUIRED / VALIDATED_FOR_BOUNDED_OUTREACH / BLOCKED

Claim boundary: documentation comprehension only - not product, real-user, commercial,
security, legal or production-readiness validation.
```

---

## Next step

The next roadmap step is to run real external-reader sessions, capture anonymized evidence and create an aggregate summary only from actual reviewed results. Do not create a results document until sessions are completed and reviewed.

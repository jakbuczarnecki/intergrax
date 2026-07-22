# Product-First MVP Development Brief

## Status

This document defines the default product-development rule for every Intergrax application and agent.

It is the reference to use:

- before starting a new application or agent,
- when preparing its roadmap,
- when deciding whether a platform change is justified,
- when implementation expands beyond the product goal,
- when a team is unsure what to build next,
- when work risks becoming proof-first, framework-first, or infrastructure-first.

The governing principle is:

```text
Deliver the smallest real product experience that demonstrates user value.
Use implementation of that product to discover and improve the platform.
Do not build the platform first and hope a useful product appears later.
```

---

## 1. Why this rule exists

Intergrax is a platform for building applications and agents. That creates a recurring risk: implementation may drift toward abstractions, providers, proofs, portability matrices, frameworks, infrastructure, or generic mechanisms before a user can perform a valuable task.

Those mechanisms may be technically correct and reusable, but they are not the primary measure of progress.

The primary measure is:

```text
What useful thing can a real user do now that they could not do before?
```

A product or agent is not successful because it contains many platform capabilities. It is successful because it solves a concrete problem in a way a user can understand, use, and value.

The product therefore leads development. Platform improvement follows product pressure.

---

## 2. The governing development loop

The required loop is:

```text
real user problem
→ product hypothesis
→ smallest valuable workflow
→ working MVP
→ real use or realistic validation
→ concrete blocker or repeated pattern
→ reusable platform improvement
→ product consumes the improvement
→ end-to-end validation
```

The following loop is explicitly rejected:

```text
generic platform idea
→ abstraction
→ provider
→ proof
→ documentation
→ portability matrix
→ search for a product that might use it
```

Platform work is justified when it enables, protects, simplifies, or scales an active product workflow.

---

## 3. Product comes before platform proof

Every Intergrax application or agent may also serve as:

- a platform proof,
- a source of reusable platform requirements,
- a platform problem detector,
- a reference implementation.

These are secondary roles.

They exist because a real product is being built.

They must not define the implementation order independently of user value.

The correct relationship is:

```text
real product capability
→ exposes a platform gap
→ gap is classified
→ blocking gap is solved in a reusable way
→ product uses the shared solution
→ capability is validated end to end
```

A platform proof that does not unlock or protect a real product outcome is not an automatic next task.

---

## 4. Required product brief for every application or agent

Before implementation begins, every application or agent must have a concise product brief.

The brief must answer the following questions.

### 4.1 Product purpose

- What are we building?
- Why should it exist?
- What problem does it solve?
- Why is this problem worth solving now?

### 4.2 Target user

- Who is the first user?
- What work are they trying to complete?
- What tools and habits do they already have?
- What level of technical knowledge can be assumed?

The first user must be concrete enough that product decisions can be made for that user.

Avoid descriptions such as:

- everyone,
- all companies,
- any developer,
- any knowledge worker.

### 4.3 Current pain

Describe the existing process:

- What does the user do today?
- Where is time lost?
- What is difficult, repetitive, risky, or expensive?
- What information or action is hard to obtain?

### 4.4 Value proposition

Complete this statement:

```text
For [specific user],
who needs to [specific job],
this product enables [valuable outcome]
by [core mechanism],
unlike [current alternative].
```

### 4.5 Primary user workflow

Define one primary workflow in observable steps.

Example shape:

```text
user provides input
→ product performs bounded work
→ user receives a useful result
→ user can verify, accept, or use the result
```

The primary workflow must be demonstrable end to end.

### 4.6 MVP

Define the smallest version that proves the value proposition.

The MVP must answer:

- What can the user do?
- What result do they receive?
- How do they access it?
- What must be reliable?
- What can remain manual?
- What is explicitly excluded?

### 4.7 Value measurement

Define observable evidence that the MVP is valuable.

Possible measures include:

- time saved,
- fewer manual steps,
- higher answer accuracy,
- faster retrieval,
- successful task completion,
- fewer errors,
- user reuse,
- user willingness to continue testing,
- successful completion without developer assistance.

A proof receipt is not a substitute for user-value evidence.

### 4.8 MVP demonstration

Describe the exact demo:

```text
starting state
→ user action
→ product behavior
→ visible result
→ evidence that the result is useful and trustworthy
```

A person unfamiliar with the implementation should understand the value from this demonstration.

---

## 5. MVP definition rules

An MVP is not the smallest amount of code.

It is the smallest complete user experience that demonstrates value.

A valid MVP contains:

- a specific user,
- a specific problem,
- a complete workflow,
- a usable result,
- a way to access the workflow,
- enough reliability to repeat the demonstration,
- enough evidence to judge whether the result is correct or useful.

An MVP may use:

- manual setup,
- a limited file type set,
- one supported provider,
- one operating system,
- one user,
- one tenant,
- a simple interface,
- a controlled design-partner environment.

An MVP should not be delayed by requirements that do not affect the first valuable workflow.

Examples of common non-MVP requirements:

- support for every provider,
- multi-region deployment,
- enterprise RBAC,
- complete UI polish,
- multiple interchangeable adapters,
- full observability stack,
- broad portability matrices,
- generalized scaffolds,
- optimization before a baseline exists.

---

## 6. The fastest-value rule

When multiple tasks are possible, choose the task that most directly reduces the distance between the current state and the first valuable user workflow.

Use this decision order:

1. Does it complete a missing step in the primary user workflow?
2. Does it make the result useful, trustworthy, or repeatable?
3. Does it remove a blocker preventing a user from trying the MVP?
4. Does it enable real validation with a design partner?
5. Is it a reusable platform improvement required by one of the above?

If the answer to all five is no, the task is not an active MVP priority.

---

## 7. Task qualification rule

Every implementation task must answer:

```text
What new valuable thing can the user do after this task?
```

Acceptable answers include:

- ask a question and receive an answer with sources,
- upload a file and receive a validated analysis,
- approve an action and receive a completed artifact,
- use the capability through a familiar tool,
- recover a failed operation that otherwise blocks use.

Insufficient answers include:

- a new abstraction exists,
- a provider was added,
- a proof document was created,
- a contract was generalized,
- another scaffold was introduced,
- another backend was made portable.

Such work may still be necessary, but only when it directly blocks or protects the active product slice.

---

## 8. Product vertical slices

Implementation should proceed through vertical slices.

A vertical slice crosses the boundaries needed to produce a user-visible result:

```text
user surface
→ application contract
→ agent or workflow
→ platform execution
→ tools or data
→ result
→ user-visible response
```

Prefer one narrow end-to-end flow over many disconnected horizontal layers.

A vertical slice should include only the platform changes required for that flow.

It should not absorb unrelated refactors discovered along the way.

---

## 9. Platform-gap handling

Every discovered platform issue must be classified.

### 9.1 Product-blocking

The current MVP workflow cannot be completed safely or correctly without it.

Action:

- solve it now,
- solve it in shared platform code when reusable,
- make the active product consume the shared solution,
- verify the complete product flow again.

### 9.2 Platform-reusable but non-blocking

The pattern could help future products, but the current MVP works without it.

Action:

- record it,
- do not automatically create the next implementation task,
- revisit when another real product need confirms it.

### 9.3 Production hardening

The issue matters for broader release readiness but does not block MVP validation.

Action:

- place it after the MVP gate,
- pull it forward only if risk is unacceptable for the intended MVP user.

### 9.4 Product-specific

The behavior belongs only to the application or agent.

Action:

- keep it in the product,
- do not generalize it prematurely.

The default rule is:

```text
Not every discovered pattern becomes a platform task.
```

---

## 10. Stop conditions against implementation drift

Stop and reassess when any of the following occurs:

- the task no longer produces a user-visible capability,
- a second major platform gap appears,
- implementation adds providers or abstractions not required by the MVP,
- repeated fixes target symptoms rather than a frozen contract,
- a live proof becomes the primary debugging mechanism,
- the product roadmap starts following platform backlog order,
- no real user can try the result after several completed tasks,
- implementation cost grows without reducing the distance to MVP.

Use one of the following explicit outcomes:

```text
BLOCKED_BY_PLATFORM_GAP
SCOPE_DRIFT_DETECTED
TOKEN_BUDGET_EXCEEDED
MVP_VALUE_UNCLEAR
```

The correct response is to stop, summarize the blocker, and restore the product boundary.

---

## 11. Token and iteration discipline

Large token use is a signal of uncertainty, scope expansion, or repeated diagnosis.

Before a substantial task:

1. inspect the current implementation path,
2. freeze the intended contract,
3. identify the expected product/platform boundary,
4. define focused boundary tests,
5. define the one permitted major platform gap.

Suggested limits:

### Ordinary task

```text
soft limit: 2M tokens
stop and review: 4M tokens
```

### Large vertical slice

```text
soft limit: 4M tokens
stop and review: 8M tokens
```

After the review threshold, implementation must stop rather than continue an open-ended fix loop.

---

## 12. Test and validation order

Use the narrowest test that can reveal a contract failure.

Required order:

```text
contract test
→ focused unit test
→ boundary integration test
→ application API test
→ one end-to-end live run
→ real-user or design-partner validation
```

Do not use repeated full live runs to discover errors that should have been caught at a typed boundary.

Technical proof confirms that the system works.

Product validation determines whether the result is valuable.

Both are required, but they answer different questions.

---

## 13. MVP gate and post-MVP roadmap

Every product roadmap must distinguish between:

### MVP gate

The earliest point at which a real user can experience and judge the core value.

### Production or 1.0 gate

The point at which the product is installable, operationally safe, supportable, secure enough for its declared use, and ready for broader release.

Do not make the complete 1.0 roadmap a prerequisite for MVP validation.

The required sequence is:

```text
product brief
→ MVP vertical slice
→ minimal packaging or access
→ real validation
→ feedback-driven next priority
→ production hardening
→ broader release
```

After MVP validation, the next task is selected by:

- observed user need,
- measurable value,
- a concrete blocker,
- unacceptable operational or security risk.

It is not automatically selected by an old platform backlog.

---

## 14. Familiar tools as user surfaces

Where valuable, applications and agents should meet users in tools they already know.

Examples may include:

- Slack,
- Microsoft Teams,
- MCP-compatible tools,
- e-mail,
- existing business systems,
- a minimal local companion.

The rule remains:

```text
The user surface translates interaction.
The user surface does not own the product capability.
```

Product logic, policy, orchestration, persistence, and evidence must remain independent of a specific user surface.

However, surface portability must not delay the first MVP merely to prove abstraction quality.

Before MVP, neutrality may be protected through:

- a canonical product API,
- typed surface-neutral results,
- adapter boundary tests,
- absence of vendor-specific fields in product domain contracts.

A second large surface implementation is justified when it has product, partner, or market value.

---

## 15. Required application or agent planning structure

Every new application or agent plan should use the following structure.

```text
1. Product purpose
2. Target user
3. Current pain
4. Value proposition
5. Primary workflow
6. MVP scope
7. Explicit non-goals
8. User access and UX
9. Trust, evidence, and safety needs
10. MVP success measures
11. MVP demonstration
12. Current product stage
13. Next vertical slice
14. Known blockers
15. Platform gaps discovered by implementation
16. Post-MVP possibilities
```

The plan must make the current MVP path visible near the top.

---

## 16. Review questions before approving a roadmap

Before accepting a product roadmap, ask:

1. Is the first user concrete?
2. Is the problem real and understandable?
3. Is the primary workflow complete?
4. Can the MVP be demonstrated without explaining platform internals?
5. Does the user receive a result they can use?
6. Can value be measured or observed?
7. Are non-goals explicit?
8. Does the roadmap distinguish MVP from 1.0?
9. Are platform tasks pulled by the product?
10. Can a real user test the product early?
11. Is the shortest route to value obvious?
12. Does every active task reduce the distance to MVP?

A roadmap that cannot answer these questions should not drive implementation.

---

## 17. Review questions before approving a task

Before approving an implementation task, ask:

1. Which MVP workflow step does this complete?
2. What new capability will be visible to the user?
3. Why is this required now?
4. What is the narrowest acceptable implementation?
5. What is explicitly out of scope?
6. Which contract must be frozen first?
7. What focused test catches failure earliest?
8. Is there more than one major platform gap?
9. Can the result be validated end to end?
10. Can the task stop safely if the token budget is exceeded?

---

## 18. Recovery procedure when a product becomes buried in implementation

When implementation feels fragmented, expensive, or directionless, perform this reset.

### Step 1 — Restate the user and problem

Write one sentence for each.

### Step 2 — Restate the MVP outcome

Complete:

```text
The MVP is successful when the user can...
```

### Step 3 — Draw the shortest end-to-end flow

Use no more than ten steps.

### Step 4 — Mark the current missing step

Only one step should be the immediate priority.

### Step 5 — Classify all other open work

Use:

- required now,
- blocks later,
- post-MVP,
- platform backlog,
- discard.

### Step 6 — Stop unrelated implementation

Do not continue work merely because it has already consumed time.

### Step 7 — Resume with one vertical slice

The next task must restore visible product progress.

---

## 19. LKW as the current reference example

LKW demonstrates this rule.

Its product value is not:

- a queue provider,
- a portability matrix,
- a proof document,
- an interaction abstraction.

Its product value is:

```text
A user points LKW at local documents,
asks a natural-language question,
and receives a useful answer with verifiable sources.
```

The intended MVP path is:

```text
managed workspace
→ local folder source
→ synchronization
→ Ask Workspace
→ answer with citations
→ familiar user surface
→ minimal setup
→ real user validation
```

LKW may expose reusable platform gaps during this implementation. Those gaps improve Intergrax because the product needs them, not because the platform backlog says they should be built first.

---

## 20. Final decision rule

When uncertain what to build next, use this rule:

```text
Choose the smallest next step that allows a real user
to experience, verify, or receive more of the product's core value.
```

If a proposed task does not satisfy that rule, it must be justified as a direct blocker, safety requirement, or reliability requirement of the active MVP.

The default destination is not a more complete platform.

The default destination is a working product that users can try, understand, and value.

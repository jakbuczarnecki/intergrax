# Intergrax Proof Library — Authoring Guide

**Status:** Canonical  
**Audience:** Independent Scenario Proof and Conformance Proof author sessions

This is the **single canonical practical instruction** for independent proof-author sessions. Future sessions receive only:

1. this guide; and
2. a description of the real-world scenario or mechanism under test.

The guide supplies the rest of the context and rules. Do not create a second overlapping authoring guide.

**Normative companion:** [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) — classification, falsification philosophy, evidence contracts. This guide is the operational workflow; the protocol is the governance layer.

**Never:** classify a product (`applications/`) as a platform domain or migrate product proofs into `platform_proofs/`.

---

## Proof Library public model

The Intergrax Proof Library has two public classes. Both are **executable falsification attempts** — not demos. **Platform proof ≠ product proof.** Product proofs stay under `applications/`.

### SCENARIO

The **primary public Proof Library unit**. Starts from:

```text
REAL PROBLEM
→ RISK
→ REQUIRED GUARANTEES
→ INTERGRAX MECHANISMS
→ EXECUTABLE FALSIFICATION
→ EVIDENCE
→ VERDICT
→ REPORT
→ REPRODUCTION
```

A Scenario Proof may exercise **multiple** Intergrax mechanisms and domains. The scenario must **never** be selected merely to demonstrate a feature.

### CONFORMANCE

Mechanism-level proof used for:

- CI
- regression
- contract verification
- architecture confidence
- platform development

Conformance proofs are **secondary** in the public Proof Library. They are mechanism-first; Scenario proofs are problem-first.

Existing domain-oriented paths may remain for Conformance proofs under `platform_proofs/<domain>/…`. Scenario proofs use `platform_proofs/scenarios/<scenario_slug>/`. Do not mandate mass folder migration for symmetry.

---

## Proof Library philosophy

> **The Intergrax Proof Library is not a catalog of Intergrax features. It is a catalog of difficult real-world AI system problems that Intergrax can solve and falsifiably demonstrate.**

> **Scenario selection precedes mechanism selection.**

> **A missing platform capability does not automatically invalidate a strong scenario. If the capability is reusable and architecturally justified, implement it in Intergrax first and then resume the Scenario Proof.**

> **One independent author session owns one Scenario Proof from qualification through accepted evidence.**

Every Scenario Proof session is a **falsification attempt** against a real problem — not a feature demo, marketing slide, or architecture tour.

---

## One session owns one Scenario Proof

**Normative rule:**

```text
One independent Scenario Proof author session owns exactly one Scenario Proof
from scenario qualification through accepted evidence/publication.
```

A session **may** inspect:

- relevant architecture;
- reusable platform components;
- existing proofs;
- tests and contracts;

—but its implementation and publication **ownership** remains **one scenario**.

A single session **MUST NOT**:

- design multiple Scenario Proofs in parallel;
- opportunistically modify unrelated Scenario Proofs;
- broaden into general Proof Library redesign.

Conformance proof work follows the same shared-development rules but is outside this one-session-one-scenario contract unless the operator explicitly scopes otherwise.

---

## Two-layer working model

Every Scenario Proof session operates in two conceptual layers. **Do not collapse them.**

### LAYER 1 — SCENARIO / PROBLEM

Before discussing Intergrax implementation, determine whether the real-world scenario is strong enough to deserve entry in the Proof Library.

Focus on:

- real problem;
- real user / operator / business pain;
- failure consequences;
- why the problem is difficult for AI systems;
- adversarial conditions;
- skeptical-engineer challenge;
- public “wow” value.

**Do not begin mechanism selection before the scenario itself is understood and accepted.**

### LAYER 2 — SOLUTION / IMPLEMENTATION

Only after scenario quality is accepted:

1. design how the system should solve the problem;
2. define desired behavior and guarantees;
3. map required Intergrax mechanisms;
4. perform capability-fit / gap analysis;
5. implement reusable missing capability if justified;
6. package / run / evaluate / report under the canonical Proof Library contract (see § Technical Proof Library lifecycle).

---

## Five-stage Scenario Proof session model

Every Scenario Proof session **MUST** explicitly work through these stages **in order**. Do not skip stages. Do not implement before earlier stages are honest.

```text
STAGE 1 — Qualify and strengthen the scenario
STAGE 2 — Design the scenario solution and proof semantics
STAGE 3 — Map required Intergrax mechanisms
STAGE 4 — Missing capability decision
STAGE 5 — Apply canonical Proof Library engine
```

Conformance proofs follow the same engineering discipline in Stages 3–5 but may omit problem-first public framing where mechanism verification is the sole purpose.

### Stage 1 — Qualify and strengthen the scenario

Discuss the scenario in **plain problem language** before any implementation.

**Required questions:**

- What is the concrete real-world problem?
- Who suffers from it?
- What happens if the AI / system gets it wrong?
- Why is this not a toy problem?
- What makes it difficult?
- What is the naive / simple solution likely to get wrong?
- What adversarial situation exposes that weakness?
- What makes the result interesting even to someone who does not know Intergrax?

The session **must actively strengthen** weak scenario proposals rather than immediately implement them.

#### Scenario Quality Gate

A Scenario Proof should satisfy **most** of these:

- real operational / business / engineering pain exists;
- failure has meaningful consequences;
- problem was not invented merely to exercise an Intergrax feature;
- scenario contains uncertainty, conflict, failure, recovery, temporal complexity, governance, side-effect risk, adversarial input, or another meaningful systems challenge;
- negative / adversarial falsification is possible;
- outcome can be evaluated;
- story is understandable without reading Intergrax internals;
- result demonstrates more than basic LLM orchestration;
- public demonstration has a credible “this is a system, not a chatbot” effect.

**Weak scenario examples that require strengthening** (capabilities may appear inside a strong scenario, but are not strong scenarios by themselves):

```text
AI answers a question from RAG.
AI uses memory.
AI calls several tools.
Two agents discuss an answer.
AI summarizes documents.
```

#### Skeptic Challenge

Every Scenario Proof **must** explicitly answer:

> Could a skeptical Staff / Principal Engineer reasonably say:
>
> “I can build the same thing with an LLM + memory + RAG + a few LangGraph nodes in 10 minutes”?

**If YES** — the scenario is currently too weak. **Do not** automatically discard the underlying real problem. First attempt to strengthen it by introducing the actual difficult guarantees naturally required by the real problem, such as:

- conflicting evidence;
- missing evidence;
- stale evidence;
- temporal reconstruction;
- independently verified decisions;
- bounded recovery;
- durable state;
- exactly-once / duplicate side-effect protection;
- policy enforcement;
- human approval;
- adversarial input;
- auditability;
- evidence provenance;
- multi-step evidence dependency;
- interruption / resume;
- disagreement requiring further evidence;
- explicit unresolved outcome when certainty is impossible.

Do **not** add complexity arbitrarily.

**If NO** — document why the scenario requires guarantees beyond trivial orchestration.

#### WOW criterion

```text
WOW does not mean “more agents” or “more infrastructure”.
WOW means a difficult, credible guarantee is visibly achieved and evidenced.
```

Reject feature stuffing such as adding many agents, many databases, Kafka, RAG, Memory, Critic, HITL, or Governance **solely** to make a proof look impressive.

The scenario should remain the **smallest credible system** that demonstrates the difficult guarantee.

**Strong WOW example:**

```text
An external refund succeeds.
The process crashes before local completion is recorded.
Execution resumes.
The system proves that a duplicate refund did not occur and preserves an auditable recovery history.
```

This can be more impressive than a seven-agent demo.

#### Scenario rejection criteria

Reject or redesign a scenario if, after strengthening attempts:

- there is no meaningful real-world pain;
- failure is inconsequential;
- PASS is subjective prose only;
- no credible falsification exists;
- the model is directly given the expected answer / ground truth;
- the proof is only a feature demo;
- simple orchestration genuinely provides equivalent guarantees;
- complexity exists only for visual impression;
- the scenario cannot be reproduced honestly;
- the claim would require overstating evidence.

Also reject proofs whose primary motivation is:

- “show that tool calling works”
- “show that RAG works”
- marketing demos
- hello-world agents
- arbitrary multi-hop requirements with no real failure risk

### Stage 2 — Design the scenario solution and proof semantics

Once the Scenario Quality Gate passes, design the solution in **plain language** before discussing APIs, classes, or package layout.

**Required story structure:**

```text
WHAT HAPPENS
→ WHAT THE SYSTEM RECEIVES
→ WHAT IT MUST DETERMINE / ACHIEVE
→ WHAT CAN GO WRONG
→ WHAT THE SYSTEM DOES
→ HOW IT REACTS TO FAILURE / CONFLICT
→ WHAT FINAL RESULT IS ACCEPTABLE
```

Where useful, use:

```text
INPUT
→ ACTION
→ OBSERVATION
→ DECISION
→ CHALLENGE / FAILURE
→ RECOVERY / VERIFICATION
→ FINAL RESULT
```

Do not force phases that do not naturally apply to the scenario.

#### Proof semantics

Before mechanism selection, define:

| Item | Required |
|------|----------|
| **Problem** | The real pain |
| **Risk** | What harmful / incorrect system behavior is being tested |
| **Desired outcome** | What correct system behavior looks like |
| **Claim** | One bounded falsifiable claim |
| **PASS** | Explicit conditions required for success — prefer machine-checkable invariants |
| **FAIL** | Explicit conditions that invalidate the claim |
| **Adversarial cases** | Concrete attempts to break the claim |
| **Excluded claims** | What the scenario does not prove |
| **Limitations** | Known boundaries |

### Stage 3 — Map required Intergrax mechanisms

Only after the problem and solution story are clear, determine technical realization.

Create a mapping:

```text
REQUIRED GUARANTEE
→ REQUIRED CAPABILITY
→ EXISTING INTERGRAX COMPONENT
→ STATUS
```

Statuses:

- **AVAILABLE**
- **AVAILABLE BUT NEEDS WIRING**
- **MISSING**

Examples of mechanisms — **only when naturally required**:

- bounded execution
- tool runtime
- durable state
- persistence
- recovery
- idempotency
- side-effect coordination
- Critic / verification
- HITL
- governance
- policy enforcement
- bitemporal knowledge
- RAG
- Memory
- observability
- Unified Run Journal
- evidence identity / dependencies
- multi-agent coordination
- budget enforcement
- security / adversarial input handling

```text
Problem chooses mechanisms.
Mechanisms do not choose the problem.
```

**Do not** start from platform mechanisms and invent a scenario around them. Never add components solely to make the proof appear sophisticated.

Verify reuse at current repository HEAD before claiming **AVAILABLE**.

### Stage 4 — Missing capability decision

A **MISSING** capability does **not** automatically invalidate a strong scenario.

When a required capability is **MISSING**, determine:

1. Is the requirement genuinely necessary for the real scenario?
2. Is the missing capability reusable beyond this one proof?
3. Does it belong naturally in Intergrax architecture?
4. Can it be implemented cleanly as a typed reusable platform mechanism?
5. Would implementing it materially strengthen Intergrax?

**If YES to the above:**

- **STOP** Scenario implementation temporarily;
- define the architecture gap;
- prepare / approve a bounded platform implementation task;
- implement and independently verify the reusable capability;
- return to the Scenario Proof.

**If NO:**

- do **not** create proof-local fake infrastructure;
- redesign the scenario realization, narrow the claim, or reject the scenario if the guarantee cannot honestly be demonstrated.

```text
The Proof Library is also a discovery mechanism for meaningful platform gaps.
```

**Proof-local clones of missing platform capabilities are forbidden.**

#### STOP ≠ abandon scenario

**STOP** means:

> do not improvise or create proof-local platform substitutes.

It does **not** necessarily mean:

> abandon the Scenario Proof.

For strong scenarios, STOP may trigger:

```text
scenario
→ architecture gap
→ reusable platform enhancement
→ verification
→ resume scenario
```

This distinction is mandatory. Report the gap; do not silently weaken the scenario or fork a proof-local substitute.

### Stage 5 — Apply canonical Proof Library engine

Only after:

- scenario accepted;
- solution designed;
- claim defined;
- mechanisms mapped;
- gaps resolved;

proceed to the **technical pipeline** defined in § Technical Proof Library lifecycle and the sections that follow (package structure, descriptor, execution, evidence, report, acceptance).

```text
proof package
→ proof.json
→ .env.example
→ optional proof-owned Compose / fixtures
→ run_proof.py
→ targeted tests
→ real execution
→ typed evidence
→ proof-result.json
→ report.html
→ manual report / evidence audit
→ Proof Library acceptance
```

Do not duplicate those technical contracts here — they remain canonical in their existing sections.

---

## Mandatory session conversation format

Structure user-facing discussion in this order:

### A. SCENARIO

Discuss:

- REAL PROBLEM
- WHY IT MATTERS
- FAILURE CONSEQUENCES
- WOW FACTOR
- SKEPTIC CHALLENGE
- ADVERSARIAL CONDITIONS

**No implementation yet.**

### B. SOLUTION

Discuss:

- DESIRED BEHAVIOR
- STEP-BY-STEP STORY
- GUARANTEES
- CLAIM
- PASS
- FAIL
- ADVERSARIAL ATTACKS
- EXCLUDED CLAIMS

### C. INTERGRAX FIT

Present a clear matrix:

```text
required guarantee
→ Intergrax mechanism
→ exact existing ownership / component
→ AVAILABLE / AVAILABLE BUT NEEDS WIRING / MISSING
```

### D. GAP DECISION

For every **MISSING** item:

- is it necessary?
- is it reusable?
- should Intergrax implement it?
- what architectural owner should own it?

**Stop for approval** before implementation when a new reusable platform capability is required.

### E. PROOF BUILD

Only then proceed:

- implementation;
- tests;
- real run;
- evidence;
- report;
- publication.

---

## User-facing explanation requirement

Explain each stage first in **plain user / problem language**. Technical detail comes afterward.

A session **must always** make clear:

- where it currently is in the scenario roadmap;
- what is being decided now;
- why that decision matters;
- whether the scenario is becoming stronger or weaker;
- whether the final public Proof Library objective is still preserved.

Avoid unnecessary implementation jargon during scenario design (Stages 1–4).

---

## Technical Proof Library lifecycle

After Stages 1–4 are complete, every Scenario Proof session **MUST** follow this strict technical sequence. Do not skip phases.

```text
4. PROOF DESIGN
5. IMPLEMENTATION
6. TARGETED TESTING
7. REAL PROOF EXECUTION
8. EVIDENCE / REPORT VERIFICATION
9. PUBLICATION / LIBRARY ACCEPTANCE
10. CLOSE AND MOVE TO NEXT SCENARIO
```

| Phase | Action |
|-------|--------|
| **4. Proof design** | Positive and adversarial paths; real boundaries; evaluator contract; artifact plan |
| **5. Implementation** | Smallest package that exercises the claim; reuse platform mechanisms |
| **6. Targeted testing** | Deterministic unit/integration gates for proof-owned logic |
| **7. Real proof execution** | Actual run with real boundaries (not dry-run substitute) |
| **8. Evidence / report verification** | Typed evidence validates; report manually audited |
| **9. Publication / library acceptance** | All acceptance gates pass (see § Public Library acceptance gate) |
| **10. Close** | Update coverage map; move to next scenario |

---

## Self-contained executable proof package

> A Proof Library entry is a **self-contained executable proof package** inside the cloned Intergrax repository.

### Self-contained does NOT mean

- a separate repository
- a copied Intergrax runtime
- copied Critic / RAG / Observability / etc.
- an independent application framework

### Self-contained DOES mean

> The proof can be configured, started, validated, executed, evaluated, reported, and cleaned up **independently** of other proofs and undocumented repository state.

**Mandatory acceptance invariant:**

```text
A proof is not accepted into the Proof Library unless it can be executed
independently from its own package using documented configuration, without
relying on another proof's infrastructure or undocumented local state.
```

The **cloned Intergrax repository** is the supported execution boundary. Do **not** require that copying only the proof directory outside the repository must work.

---

## Package structure

### Scenario proofs (canonical shape)

```text
platform_proofs/scenarios/<scenario_slug>/

├── README.md
├── proof.json
├── .env.example               # when configuration is required
├── run_proof.py
├── evaluator.py               # when needed
├── evidence_builder.py        # when needed
├── scenario.py / scenarios.py # when needed
├── docker-compose.yml         # only if proof owns containerized infra
├── fixtures/                  # only if needed
├── sql/                       # only if needed
└── output/                    # canonical published artifacts
```

### Conformance proofs

Existing domain-oriented paths may remain (e.g. `platform_proofs/<domain>/<proof_slug>/`). Scenario proofs use `platform_proofs/scenarios/<scenario_slug>/`. The same component rules apply.

### Rules

- Do **not** create empty placeholder files or directories.
- Create only components the proof actually needs.
- `run_proof.py` is the **canonical proof-owned entrypoint**.
- `proof.json` is the **machine-readable package contract**.
- Source and runtime components remain inside the package unless genuine reusable platform code belongs in Intergrax.
- Platform mechanisms are **imported / reused** — never copied.

---

## Descriptor v2

Every proof declares a static `proof.json` with schema:

```text
intergrax.platform_proof_descriptor.v2
```

**Canonical schema source:** `scripts/proof/intergrax_platform_proof_descriptor.py` and [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) § D2. Do not duplicate full schema here.

### Required for all proofs

| Field | Role |
|-------|------|
| `library_class` | `SCENARIO` or `CONFORMANCE` |
| `domain` | Technical ownership / runner grouping |
| `proof_kind` | Proof-specific kind slug |
| `mechanisms_exercised` | Mechanisms the proof exercises |

### Additional for SCENARIO

| Field | Role |
|-------|------|
| `problem_category` | Problem taxonomy |
| `problem_summary` | Short problem statement |
| `failure_mode_summary` | Failure being tested |

**No descriptor v1.** No loose metadata bags. No compatibility shims. No aliases. No fallback. No legacy proof execution path.

Discovery is automatic from `proof.json` under `platform_proofs/` — no central manifest registration for descriptor-backed proofs.

---

## Standalone execution and suite execution

Every proof must support **two execution paths** using the **same underlying implementation**.

### A. Standalone

For a person exploring one proof.

**Canonical invocation context:** repository root. Scenario README commands must be valid from repository root unless an exceptional proof explicitly documents otherwise and has architectural justification.

Canonical style:

```bash
uv run python platform_proofs/scenarios/<scenario_slug>/run_proof.py
```

For existing Conformance paths, substitute the actual package path (e.g. `platform_proofs/<domain>/<proof_slug>/run_proof.py`).

Standalone execution must **not** require understanding proof-suite internals.

### B. Suite

For CI, regression, and multiple proofs. Reuse existing canonical `scripts/proof/` discovery and runner infrastructure.

```bash
uv run python scripts/proof/run-intergrax-proof-suite.py --profile full
```

**Never build a second runner.**

**Invariant:**

```text
Standalone run and suite run execute the same proof logic.
```

No separate “demo implementation.”

---

## Standard lifecycle commands

Every proof README must document only the phases relevant to it.

**Canonical lifecycle:**

```text
CONFIGURE
→ PREFLIGHT
→ INFRA UP
→ DATA / FIXTURE SETUP
→ VALIDATE
→ RUN
→ EVALUATE
→ REPORT
→ CLEANUP
```

For proofs with infrastructure, expected UX should be simple and explicit. All commands below assume **repository root** as the working directory. For containerized proof infrastructure, use an explicit Compose path — do not document ambiguous bare `docker compose up -d` as the canonical example.

```bash
cp platform_proofs/scenarios/<scenario_slug>/.env.example \
  platform_proofs/scenarios/<scenario_slug>/.env
docker compose \
  -f platform_proofs/scenarios/<scenario_slug>/docker-compose.yml \
  up -d
uv run python \
  platform_proofs/scenarios/<scenario_slug>/run_proof.py \
  --validate-only
uv run python \
  platform_proofs/scenarios/<scenario_slug>/run_proof.py
docker compose \
  -f platform_proofs/scenarios/<scenario_slug>/docker-compose.yml \
  down -v
```

Do **not** require Docker for proofs that do not need it.

---

## `--validate-only`

For proofs with meaningful environmental or data prerequisites, provide a **non-LLM** validation path:

```bash
uv run python .../run_proof.py --validate-only
```

Validate what can be validated **without** consuming provider inference:

- infrastructure reachable
- schema / data materialized
- fixtures valid
- deterministic invariants
- configuration completeness where possible

Failure must be **explicit and actionable**. Prefer:

```text
BLOCKED_ENVIRONMENT
PostgreSQL unavailable.
Run: docker compose -f platform_proofs/scenarios/<scenario_slug>/docker-compose.yml up -d
```

over raw connection stack traces as the primary UX.

Not every proof needs identical validation internals — but proofs with infra/data prerequisites should expose this path.

---

## Configuration contract

### Reuse platform configuration

Standard Intergrax configuration and provider names **MUST** be reused. Do not invent proof-specific aliases for existing platform settings.

| Bad | Good |
|-----|------|
| `MY_PROOF_OPENAI_KEY` | `OPENAI_API_KEY` (per provider) |
| `SQL_PROOF_MODEL` | `INTERGRAX_LLM_MODEL` |

See [PLATFORM_CONFIGURATION.md](../docs/project/technical/guides/PLATFORM_CONFIGURATION.md) for canonical names.

### Proof-local configuration

Proof-specific infrastructure may use proof-local variables, e.g. `PROOF_POSTGRES_PORT` or the existing `INTERGRAX_PP_*` convention where established. Names must be scoped sufficiently to avoid collisions when multiple proofs run concurrently.

---

## `.env` contract

Every proof requiring configuration **SHOULD** provide a committed `.env.example` in the proof package:

- contains **no secret values**
- documents every proof-specific setting
- safe / defaultable values receive defaults in documentation
- required secrets remain blank
- `.env` itself is **never** committed (local / gitignored)

### Precedence

Intergrax reads configuration from the **process environment**. The platform library does **not** load `.env` by itself — see [PLATFORM_CONFIGURATION.md](../docs/project/technical/guides/PLATFORM_CONFIGURATION.md).

The Proof Library defines **one deterministic** configuration contract for proof packages:

```text
process environment
> nearest .env found by walking upward from the proof package directory
> safe defaults
```

Where:

- process environment has highest priority
- `.env` must **never** overwrite an already-set process variable
- only **one** `.env` file is loaded — the nearest file on the path from the proof package up to (and including) repository root
- search never continues above repository root; `$HOME` and filesystem paths outside the repository are never scanned
- `.env.example` is committed when configuration is required
- `.env` is local / gitignored
- secrets never enter CLI args, `proof.json`, evidence, report, logs, or unsafe observability

Every configured Scenario Proof has `.env.example`. At runtime the canonical proof environment loader (`scripts/proof/intergrax_proof_environment.py`):

1. starts at the proof package directory;
2. searches upward for `.env`;
3. stops at repository root;
4. loads only the nearest `.env`;
5. never overwrites process environment.

Repository-root `.env` can therefore act as shared configuration for every proof that does not provide a nearer `.env`.

Proof implementations must call the shared loader — do **not** implement a proof-local dotenv / config loader.

---

## Secrets

Strict rules — secrets must **never**:

- appear in `proof.json`
- be committed in `.env`
- be passed through CLI arguments
- be persisted in evidence
- be rendered in report
- be emitted unredacted through observability
- be put in Docker image layers

Safe provenance is acceptable:

```text
provider=openai
model=<model>
credential_source=environment
```

Never record credential values.

---

## Proof-owned infrastructure

A proof **MAY** own Docker Compose infrastructure (PostgreSQL, Redis, Qdrant, MinIO, mock external service, etc.).

| Rule | Requirement |
|------|-------------|
| Ownership | Infrastructure belongs to the proof package |
| Isolation | Must not depend on another proof's Compose project |
| Volumes | Do not reuse another proof's volumes |
| Dev stack | Do not depend on general Intergrax dev stack unless scenario explicitly proves that integration and it is documented |
| Images | Pin important external image versions |
| Health | Add healthchecks where meaningful |
| Lifecycle | Document startup and teardown; cleanup must be explicit |
| Data | Setup must be deterministic where the claim depends on data |

---

## Docker Compose concurrency

Proofs must be safe for a growing library and parallel author sessions.

### No fixed global container names

Do **not** use `container_name:` unless a proven external integration absolutely requires one. Let Compose namespace resources.

### Ports

Avoid undocumented hardcoded globally scarce ports. Preferred model:

```yaml
ports:
  - "${PROOF_POSTGRES_PORT:-5435}:5432"
```

The proof README and `.env.example` must expose that variable. If a future canonical dynamic-port allocator is implemented, this guide may evolve — do not invent one now.

### Volumes and networks

Use proof-local Compose resources. No external shared volume by default.

---

## Parallel execution requirement

A proof author must consider whether two different proofs can run concurrently.

**Acceptance posture:**

- no global fixed container names
- configurable host ports
- proof-scoped volumes
- proof-scoped environment variables where local infra requires them
- no shared mutable fixture state
- no assumption that another proof is stopped

If true parallel execution is impossible because of an external provider or service constraint, document the limitation explicitly in the proof README.

---

## Dataset and fixtures

When a proof relies on controlled data:

- deterministic source / generator preferred
- fixed version / seed where applicable
- verify materialized data before model execution
- model must **not** receive hidden expected answers or ground-truth implementation
- **expected-answer prompting is forbidden**
- fixture may simulate environment only when the boundary under proof remains real

Ground truth belongs to evaluator / evidence logic — **not** the model prompt.

---

## Falsification requirements

Every **SCENARIO** must define:

- positive path where relevant
- meaningful negative / adversarial paths
- exact conditions that cause FAIL
- bounded execution
- model-independent invariants wherever possible

**PASS cannot mean:** “The answer looks reasonable.”

**Prefer:**

- structural checks
- deterministic invariant checks
- typed critic / evaluator verdicts
- evidence dependency validation
- explicit unsupported-claim detection

LLM-as-judge may augment but must **not** become the sole source of truth when deterministic evidence exists.

---

## Critic / second agent

A second agent / Critic **MUST NOT** be introduced merely to make a scenario appear multi-agent.

Use Critic when **independent verification is part of the required guarantee**. It verifies observable artifacts:

- candidate output
- evidence
- tool traces
- trajectory
- `InvestigationProof`
- structured claims

Do **not** collect private chain-of-thought as proof evidence. Preferred concept: **observable reasoning trajectory** — not hidden internal reasoning.

Reuse Intergrax Critic & Verification Layer when applicable. Do not create proof-local critic architecture when a platform mechanism exists.

---

## Observability

Every **SCENARIO** must explicitly decide what execution evidence a skeptical reader needs to reconstruct what happened.

Reuse Intergrax observability / HOS where applicable. The proof report should expose useful operational evidence such as:

- execution identity
- attempts
- tool invocations
- evidence dependencies
- critic verdicts
- revisions / retries
- terminal reason
- relevant timings
- provenance

Do **not** create a private competing observability bus or store. Logs are **not** canonical execution evidence. Never expose chain-of-thought or secrets.

---

## Evidence contract

Reuse existing contracts — no competing model:

- `PlatformProofEvidence`
- `SuiteReceipt`
- artifact verification (PP-SUITE-4)

Evidence must distinguish:

- what the model could see
- hidden ground truth
- real vs fixture boundaries
- execution evidence
- evaluator verdict
- limitations
- excluded claims

**Exit code alone is never sufficient** when `evidence_required=true`.

For descriptor-backed proofs with `evidence_required=true`, write `evidence.json` to the runner-provided `INTERGRAX_PROOF_ARTIFACT_DIR` when executed via suite.

---

## Report contract

Every accepted public **SCENARIO** must generate a rich human-readable `report.html`. Reuse the canonical renderer / report standard ([PLATFORM_PROOF_REPORT_STANDARD.md](../docs/project/proofs/PLATFORM_PROOF_REPORT_STANDARD.md)).

The report must let a skeptical engineer understand, **without reading source first**:

1. the real problem
2. why it matters
3. failure being tested
4. claim
5. architecture / mechanisms used
6. real vs simulated boundaries
7. what information the model had
8. adversarial conditions
9. execution story / timeline
10. evidence graph / dependencies where relevant
11. critic / verification results where relevant
12. final verdict
13. why PASS / FAIL occurred
14. limitations
15. excluded claims
16. exact reproduction instructions
17. provenance / environment

The report derives from typed evidence — it is **not** a second source of truth. Do not make the report a marketing page.

---

## Artifacts

| Location | Role |
|----------|------|
| `.artifacts/` | Transient run state (gitignored) |
| `<proof-package>/output/` | Canonical publishable output (Git-trackable) |

Stable filenames where declared:

```text
evidence.json
proof-result.json
report.html
```

Canonical output must be safe to commit and link from public documentation. Never commit provider secrets or raw unsafe traces.

`.artifacts/` remains transient suite state; canonical `output/` is commit-ready public proof evidence.

---

## README contract for every SCENARIO

Each Scenario package README must be **problem-first**. Required order:

```text
1. Real problem
2. Who has this problem / why it matters
3. Failure mode
4. Claim under test
5. What would falsify the claim
6. Intergrax mechanisms exercised
7. Real / fixture boundaries
8. Adversarial cases
9. How the proof works
10. Quick start / configuration
11. Validate
12. Run
13. View report / artifacts
14. Cleanup
15. Limitations
16. What this does NOT prove
```

Do **not** lead with platform-domain internals. A visitor should understand the problem before learning which Intergrax layers solve it.

---

## Public Library acceptance gate

A Scenario Proof is **not accepted** as a Proof Library entry until **all** are true:

- [ ] real problem clearly defined
- [ ] meaningful failure risk
- [ ] falsifiable claim
- [ ] Intergrax mechanisms mapped
- [ ] capability gaps resolved honestly
- [ ] package independently runnable within cloned repo
- [ ] configuration documented
- [ ] infrastructure isolated
- [ ] deterministic / preflight validation where relevant
- [ ] real proof executed
- [ ] required evidence validates
- [ ] evaluator passes
- [ ] rich report generated and manually audited
- [ ] reproduction tested
- [ ] limitations / excluded claims present
- [ ] no secrets
- [ ] no undocumented dependencies
- [ ] canonical output ready for publication

If any fail:

```text
NOT ACCEPTED INTO PROOF LIBRARY
```

even if code exists.

Update [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md) coverage (`NO_PROOF` → `DESIGNED` → `EXECUTABLE` → `QUALIFIED`). Update [PROOFS.md](../docs/project/proofs/PROOFS.md) **only** when accepted public evidence / claim boundaries change.

---

## Workflow for parallel proof-author sessions

Every new independent session receives:

```text
Read and follow:
platform_proofs/PLATFORM_PROOF_AUTHORING_GUIDE.md

Scenario:
<USER-SUPPLIED REAL PROBLEM DESCRIPTION>
```

The session **MUST** then:

1. resolve current HEAD (`git fetch origin development`)
2. read this guide
3. confirm **one session = one Scenario Proof** ownership
4. work through Stages 1–4 using the mandatory conversation format (§ Mandatory session conversation format)
5. inspect only architecture / components relevant to the scenario
6. pass Scenario Quality Gate and Skeptic Challenge before implementation
7. **STOP** for architecture decision if a reusable mechanism is **MISSING** (STOP ≠ abandon scenario)
8. only after approval — Stage 5 / technical lifecycle (implement → test → execute → evidence → report)
9. publish only after acceptance gates

Do not assume decisions from another scenario unless encoded in canonical repo documentation. **Current repo wins.**

---

## Shared-development rules

Mandatory rules for proof sessions:

| Rule | Requirement |
|------|-------------|
| Branch | `development` |
| HEAD | resolve before every material task |
| Concurrency | parallel sessions expected |
| Branching | no branches / worktrees unless operator changes policy |
| Destructive git | no reset, rebase, stash, clean, amend, force |
| Unrelated work | never overwrite concurrent changes |
| Commits | only scoped changes; return SHA |
| Conflicts | if scoped concurrent conflict occurs — **STOP** |

---

## Engineering restrictions

- reuse before create
- typed contracts
- `extra="forbid"` where descriptor / evidence contracts use it
- no `dict[str, Any]` in new proof contracts
- no `getattr` / `setattr` / loose reflection
- no speculative abstraction
- no second runner
- no second observability stack
- no second receipt
- no expected-answer prompt leakage
- no fake proof-local replacement for platform mechanism
- no descriptor v1, compatibility shims, aliases, fallback, or legacy proof execution path
- no mass migrations for symmetry

---

## Session STOP conditions

A proof-author session **MUST STOP** and report instead of improvising when:

- required reusable mechanism is **MISSING** (see § Stage 4 — Missing capability decision; STOP ≠ abandon scenario)
- existing architecture ownership is ambiguous
- proof would need to copy product logic
- real boundary cannot be exercised
- ground truth would need to be leaked to the model
- evaluator cannot define meaningful falsification
- package cannot be made independently reproducible
- scope begins requiring platform redesign not approved by user
- concurrent edits conflict in scoped files
- canonical proof-package configuration / environment loading required by the scenario is not available as reusable proof infrastructure — **STOP** and report the gap; do not implement a proof-local dotenv / config loader
- Scenario Quality Gate or Skeptic Challenge cannot be satisfied after honest strengthening attempts — reject or redesign; do not proceed to proof build

---

### Design-stage Scenario scaffold

Before implementation, create the canonical design-stage package with:

```bash
uv run python scripts/proof/create_scenario_proof.py --slug <scenario_slug> --title "<title>"
```

This produces `platform_proofs/scenarios/<scenario_slug>/README.md` only — no fake `proof.json`, runtime entrypoint, or evidence artifacts.

Workflow:

```text
canonical scaffold
→ design-stage package
→ human Scenario Quality Gate
→ implementation only after acceptance
```

Do not manually invent scenario directory shapes or skip the quality gate.

---

## Current reference proof

No executable Scenario or Conformance platform proof is designated as the canonical reference example yet. The first Scenario Proof (`ai_incident_investigation`) is in **design qualification** under `platform_proofs/scenarios/`.

See [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md).

---

## Anti-patterns

| Anti-pattern | Why it fails |
|--------------|--------------|
| Feature chosen before problem | Demo, not falsification |
| Multiple scenarios in one session | Violates one-session ownership; fragments quality gates |
| Feature stuffing for WOW | Complexity without credible guarantee |
| LangGraph-equivalent scenario without strengthening | Fails Skeptic Challenge |
| Fake replacing mechanism under proof | Cannot claim that boundary proved |
| Product proof masquerading as platform proof | Violates ownership rule |
| Proof buried only in `applications/` | Product-owned; not platform proof |
| Duplicate proof runner or manifest | Fragments execution truth |
| Undocumented environment assumptions | BLOCKED or false PASS |
| Claim broader than evidence | Public governance violation |
| No negative scenario | Not falsification — demo only |
| PASS based only on prose | Not machine-checkable |
| PASS based only on exit code when `evidence_required=true` | Suite verifies typed `evidence.json` |
| Chain-of-thought collection as evidence | Not platform invariant |
| Reimplementing platform components inside proof | Proves clone, not platform |
| Proof-local critic / observability / receipt stack | Fragments platform truth |
| Expected-answer prompt leakage | Invalidates falsification |

---

## Related documents

| Document | Role |
|----------|------|
| [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) | Governance — classification, falsification, evidence |
| [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md) | Coverage map |
| [README.md](README.md) | Proof Library gateway |
| [PLATFORM_CONFIGURATION.md](../docs/project/technical/guides/PLATFORM_CONFIGURATION.md) | Canonical env / provider names |
| [PROOFS.md](../docs/project/proofs/PROOFS.md) | Public proof dashboard |
| [PUBLIC_PROOF_AND_CLAIMS_MODEL.md](../docs/project/maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md) | Public wording governance |

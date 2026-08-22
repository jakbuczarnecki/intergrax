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

Existing domain-oriented paths (e.g. `platform_proofs/tools/…`) may remain for Conformance proofs. Do not mandate mass folder migration for symmetry.

---

## Mandatory scenario development lifecycle

Every **Scenario Proof** session **MUST** follow this strict sequence:

```text
1. PROBLEM DEFINITION
2. MECHANISM SELECTION
3. CAPABILITY FIT / GAP ANALYSIS
4. PROOF DESIGN
5. IMPLEMENTATION
6. TARGETED TESTING
7. REAL PROOF EXECUTION
8. EVIDENCE / REPORT VERIFICATION
9. PUBLICATION / LIBRARY ACCEPTANCE
10. CLOSE AND MOVE TO NEXT SCENARIO
```

Do not skip phases. Do not implement before gap analysis is honest.

### 1. Problem definition

Before any implementation, define:

| Item | Required |
|------|----------|
| Concrete real-world problem | Yes |
| Who has it | Yes |
| Why failure matters | Yes |
| Naive / simple failure mode | Yes |
| Adversarial conditions | Yes |
| Claim under test | Yes — single, bounded, falsifiable |
| Explicit PASS | Yes — machine-checkable where possible |
| Explicit FAIL | Yes |
| Excluded claims | Yes |
| Limitations | Yes |

**Reject** proofs whose primary motivation is:

- “show that tool calling works”
- “show that RAG works”
- marketing demos
- hello-world agents
- arbitrary multi-hop requirements with no real failure risk

### 2. Mechanism selection

Select only mechanisms **naturally required** by the problem.

```text
Problem chooses mechanisms.
Mechanisms do not choose the problem.
```

Never add components solely to make the proof appear sophisticated.

### 3. Capability fit / gap analysis

Before implementing:

1. Map required guarantees to existing Intergrax mechanisms.
2. Classify each as **AVAILABLE**, **AVAILABLE BUT NEEDS WIRING**, or **MISSING**.
3. Verify reuse at current repository HEAD.
4. If a required **reusable** mechanism is **MISSING**:
   - **STOP** implementation;
   - report the architecture gap;
   - decide whether platform implementation is justified.

**Proof-local clones of missing platform capabilities are forbidden.**

### 4–10. Design through acceptance

| Phase | Action |
|-------|--------|
| **4. Proof design** | Positive and adversarial paths; real boundaries; evaluator contract; artifact plan |
| **5. Implementation** | Smallest package that exercises the claim; reuse platform mechanisms |
| **6. Targeted testing** | Deterministic unit/integration gates for proof-owned logic |
| **7. Real proof execution** | Actual run with real boundaries (not dry-run substitute) |
| **8. Evidence / report verification** | Typed evidence validates; report manually audited |
| **9. Publication / library acceptance** | All acceptance gates pass (see § Public Library acceptance gate) |
| **10. Close** | Update coverage map; move to next scenario |

Conformance proofs follow the same engineering discipline but may omit problem-first public framing where mechanism verification is the sole purpose.

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

Existing domain-oriented paths may remain (e.g. `platform_proofs/tools/<proof_slug>/`). The same component rules apply.

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

For existing Conformance paths, substitute the actual package path (e.g. `platform_proofs/tools/iterative_sql_investigation/run_proof.py`).

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

The Proof Library defines **one deterministic target** configuration contract for proof packages:

```text
process environment
> explicit <proof-package>/.env
> safe defaults
```

Where:

- process environment has highest priority
- `.env` must **never** overwrite an already-set process variable
- `.env` location is exactly the proof package — no parent / root / home recursive discovery
- `.env.example` is committed when configuration is required
- `.env` is local / gitignored
- secrets never enter CLI args, `proof.json`, evidence, report, logs, or unsafe observability

**Do NOT** introduce magic recursive `.env` discovery. **Do NOT** silently scan `../.env`, `../../.env`, or `$HOME/.env`.

> Proof implementations must use the canonical shared proof environment loader once available. Until its existence is verified, authors must **STOP** during capability-fit rather than inventing per-proof dotenv loading.

Do **not** implement a proof-local dotenv / config loader. Do **not** claim a shared proof dotenv loader already exists in the repository.

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
3. inspect only architecture / components relevant to the scenario
4. produce a user-facing roadmap
5. define problem / risk / claim / failure conditions
6. map mechanisms
7. perform capability-fit / gap analysis
8. **STOP** for architecture decision if reusable mechanism missing
9. only after approval — implement
10. test
11. execute actual proof
12. inspect evidence / report
13. publish only after acceptance gates

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

- required reusable mechanism is missing
- existing architecture ownership is ambiguous
- proof would need to copy product logic
- real boundary cannot be exercised
- ground truth would need to be leaked to the model
- evaluator cannot define meaningful falsification
- package cannot be made independently reproducible
- scope begins requiring platform redesign not approved by user
- concurrent edits conflict in scoped files
- canonical proof-package configuration / environment loading required by the scenario is not available as reusable proof infrastructure — **STOP** and report the gap; do not implement a proof-local dotenv / config loader

---

## Current reference proof

**`TOOLS-ITERATIVE-SQL-INVESTIGATION`** (`platform_proofs/tools/iterative_sql_investigation/`) is an **existing executable Conformance proof** being evolved under the new Proof Library strategy. It exercises bounded iterative SQL investigation with real PostgreSQL, real model provider, and real tool runtime.

It is **not** yet reframed as a public Scenario proof. Do not claim future Investigator / Critic / Observability scenarios already exist.

See [tools/iterative_sql_investigation/README.md](tools/iterative_sql_investigation/README.md) and [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md).

---

## Anti-patterns

| Anti-pattern | Why it fails |
|--------------|--------------|
| Feature chosen before problem | Demo, not falsification |
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

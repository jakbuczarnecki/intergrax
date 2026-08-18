# Intergrax Platform Audit Protocol v2.1

**Status:** Canonical
**Version:** 2.1
**Audience:** Human operators and model executors (any harness, any IDE)
**Scope:** Full Intergrax platform — Tier-0 `intergrax/`, Tier-1 `intergrax/runtime/`, Tier-2 `agents/`, Tier-3 `applications/`

---

## Core mindset

> **Your job is not to confirm that Intergrax is well designed. Your job is to try to prove that it is not.**

**Falsification principles (brutal and neutral):**

- Assume every material claim is **UNVERIFIED** until independently examined.
- Actively search for counterexamples and failure modes.
- There is no reward for PASS and no reward for FAIL.
- Never manufacture, inflate, or preserve a finding merely to satisfy the adversarial posture.
- If a hypothesis cannot be supported with evidence, downgrade it to an open question or discard it.
- A clean **PASS** backed by a serious falsification attempt is a valid audit outcome.

A prior PASS, a green CI run, or polished architecture prose does not reduce skepticism.

This protocol is **model-executable**: an agent following it must produce the same artifacts, evidence standards, and verdict discipline regardless of tooling. It is **not** tied to Cursor, any specific orchestrator, `progress.json`, or deleted harness machinery.

---

## A. Audit purpose

An Intergrax platform audit is:

1. **Adversarial** — actively search for ways the system fails, lies, drifts, or only works by accident.
2. **Independent** — do not inherit optimism from authors, prior chats, or roadmap status (`Done`, `PASS`, milestone claims).
3. **Unincentivized toward PASS** — no quota, no desire to close quickly, no credit for a clean narrative. A FAIL with strong evidence is a successful audit.
4. **Defect- and assumption-focused** — primary output is falsified claims, hidden coupling, and false assumptions about behavior under production stress.
5. **Evidence-bound** — opinions without inspectable proof are hypotheses, not findings.

Audits do **not** implement fixes, refactor for style, or "helpfully" patch issues discovered mid-run (see section L).

---

## B. Source-of-truth hierarchy

When sources conflict, resolve in this order (highest wins):

| Rank | Source | Role |
|------|--------|------|
| **1** | **Implementation at exact SHA** | What the system actually does at audit time. Pin `git rev-parse HEAD` (or named tag/commit) in every campaign artifact. |
| **2** | **Tests and CI** | **Evidence, not proof.** Tests show what was exercised; they do not prove completeness, production safety, or architectural compliance. Missing or weak tests are themselves findings. |
| **3** | **Architecture docs** (`docs/project/architecture/`, capability architecture hubs) | **Target state** — design intent and invariants. Deviations are findings unless explicitly documented as accepted gaps. |
| **4** | **Plan docs** (`docs/project/maintainers/plans/`, capability plan hubs) | **Work claims** — what maintainers assert is built, deferred, or verified. Treat `Done` as a claim to falsify, not a fact. |
| **5** | **Prior audit artifacts** (`docs/audit_results/`) | **Historical only** — context for regression and fixed defects. Never authoritative over current implementation. |

**Rule:** If architecture says X and code does Y, record a finding unless a dated, explicit exception exists in architecture or plan with operator acceptance.

---


## B1. Architecture as auditable claim

Architecture (`docs/project/architecture/`, capability architecture hubs) is a **target-state source of truth for implementation**, but architecture itself is **also an auditable claim**.

The audit must not stop at "code matches architecture." Separately ask:

- Is the target architecture production-grade?
- Are the abstractions correct?
- Are responsibilities in the correct layer?
- Are dependency directions appropriate?
- Are stated guarantees realistically enforceable?
- Is the design safe under failure, retry, restart, concurrency, and multi-host operation?
- Does it support reusable platform mechanisms rather than current-product special cases?
- Would a second independent application reuse the mechanism cleanly?
- Does the design create unnecessary parallel universal mechanisms?
- Does another platform component already solve substantially the same concern?
- Are two registries, event systems, retry systems, policy evaluators, context assemblers, memory abstractions, tool gateways, identity mechanisms, or config systems competing for the same ownership?
- Is one implementation merely a product-local clone?
- Are shared invariants implemented multiple times and drifting?
- Could the second mechanism be a thin adapter over the canonical one?
- If two mechanisms are intentionally distinct, is their ownership boundary explicit and non-overlapping?
- Are important production invariants absent from the architecture?
- Is a documented abstraction only paper architecture?
- Are industry-standard mature production patterns materially stronger for this problem?

**Finding kinds (architecture vs implementation):**

| Kind | Meaning |
|------|---------|
| **IMPLEMENTATION DEFECT** | Code contradicts a valid architecture target or invariant. |
| **ARCHITECTURE DEFECT** | Target design is wrong, incomplete, or not production-grade. |
| **IMPLEMENTATION/ARCHITECTURE DRIFT** | Code and architecture disagree; ownership of the gap must be explicit. |

Do not treat canonical architecture prose as automatically correct. External comparison is evidence-driven when materially useful; do not perform internet research merely for decoration.


## C. Campaign model

An audit **campaign** is a **longitudinal audit sequence** across one or more layers. Each **layer audit** is an **immutable snapshot** pinned to an exact repository SHA.

After each accepted layer the canonical workflow is:

```
audit → persist result → sync architecture → sync implementation plan → commit → next layer
```

Therefore `development` may advance between layer audits. Do **not** model a multi-layer campaign as audited against a single pinned SHA.

### Campaign identity

- **Campaign date:** `YYYY-MM-DD` (UTC calendar date when the campaign **starts**).
- **Campaign directory (`<CAMPAIGN_DIR>`):** `YYYY-MM-DD` \| `YYYY-MM-DD_run-2` \| `YYYY-MM-DD_run-3` (see same-day naming below).
- **Campaign root:** `docs/audit_results/<CAMPAIGN_DIR>/`
- **Global registry:** `docs/audit_results/README.md` — protocol entry point, global campaign registry, latest-campaign discovery, legacy pointer.

### Campaign directory layout

```text
docs/audit_results/
  README.md
  AUDIT_PROTOCOL.md
  AUDIT_REMEDIATION_PROTOCOL.md
  <CAMPAIGN_DIR>/
    README.md              # REQUIRED — campaign master register AND rollup
    <LAYER_CODE>.md        # immutable per-layer audit snapshot
    ...
  legacy/
    ...
```

A campaign has **one** coordination/rollup document: `docs/audit_results/<CAMPAIGN_DIR>/README.md`. It holds campaign metadata, layer register, finding register, cross-layer rollup, and remediation/verification trace. There is **no** `CAMPAIGN_SUMMARY.md`. Do **not** create or publish a second campaign-summary artifact.

### Same-day campaigns

Use **one** convention only:

- `YYYY-MM-DD`
- `YYYY-MM-DD_run-2`
- `YYYY-MM-DD_run-3`

Do **not** use `YYYY-MM-DD-a` / `-b` suffixes.

### Campaign `README.md` — master register contract

`docs/audit_results/<CAMPAIGN_DIR>/README.md` is the **mutable master register** for one campaign. It is the single coordination/rollup document. Do **not** create `CAMPAIGN_SUMMARY.md` or any other parallel campaign-status artifact.

#### A. Campaign metadata (required section)

| Field | Description |
|-------|-------------|
| `campaign_id` | Dated directory name (e.g. `2026-08-18` or `2026-08-18_run-2`) |
| `campaign_token` | Immutable token derived from directory name (section H) |
| `started_at` | UTC timestamp when campaign started |
| `completed_at` | UTC timestamp when the **scoped audit** closed (or `—` while in progress) |
| `status` | `IN_PROGRESS` \| `COMPLETE` \| `ABORTED` — tracks **audit lifecycle only** (section C2) |
| `campaign_start_sha` | Repository SHA at campaign start |
| `campaign_end_sha` | Final repository state after campaign-owned documentation synchronization (or `—` while in progress) |
| `scope` | Layers/domains in scope |
| `overall_verdict` | Campaign rollup verdict (section J; or `—` while in progress) |

Legacy material is identified by location under `legacy/`, not by a competing active campaign status.

#### B. Layer register (required table)

| Column | Description |
|--------|-------------|
| `layer` | Layer code (e.g. `MEMORY`, `ORCHESTRATION`) |
| `status` | Layer audit status |
| `audited_sha` | **Exact** SHA this layer was audited against (immutable once published) |
| `verdict` | Layer verdict (section J) |
| `critical` | Count of CRITICAL findings |
| `high` | Count of HIGH findings |
| `medium` | Count of MEDIUM findings |
| `low` | Count of LOW findings |
| `architecture_sync` | Whether arch sync completed for accepted findings |
| `plan_sync` | Whether plan sync completed for accepted findings |
| `post_sync_sha` | Commit SHA after arch/plan sync (when synchronization created a later commit; or `—`) |
| `report` | Path to immutable per-layer snapshot (e.g. `<LAYER_CODE>.md`) |

#### C. Finding register — current authoritative state (required table)

The finding register in the campaign `README.md` is the **authoritative current state** for finding lifecycle and remediation. Initially unknown fields use `—`.

| Column | Description |
|--------|-------------|
| `finding_id` | Immutable ID (section H) |
| `layer` | Layer code (or `CROSS`) |
| `severity` | CRITICAL \| HIGH \| MEDIUM \| LOW |
| `category` | Primary category (section G) |
| `status` | Current lifecycle status (section G1) |
| `remediation_block` | Named remediation block (or `—` until accepted findings are grouped) |
| `dependencies` | Canonical finding IDs and/or explicitly named remediation blocks (or `—`) |
| `arch_ref` | Architecture doc path + section anchor (or `—`) |
| `plan_ref` | Plan doc path + block/phase id (or `—`) |
| `implementation_commit` | One or more full commit SHAs (or `—` until they exist) |
| `verification_evidence` | Tests, CI run, reviewer id, date (or `—` until they exist) |
| `notes` | Operator/executor notes (or `—`) |

`dependencies` reference canonical finding IDs and/or explicitly named remediation blocks. `remediation_block`, `implementation_commit`, and `verification_evidence` remain `—` until they exist.

#### D. Campaign rollup (required sections — one document, two semantic rollups)

The campaign `README.md` holds **two** rollup sections with distinct semantics. Do **not** create another file (no `CAMPAIGN_SUMMARY.md`).

##### D.1 Audit rollup (AUDIT ROLLUP)

Produced when campaign status becomes `COMPLETE`. Frozen as historical audit conclusion except via an explicit correction note (section C2).

At minimum:

- systemic themes
- CROSS findings
- audit finding counts by severity
- overall audit verdict (`overall_verdict`)
- recommended remediation order

Do **not** overwrite this section during post-audit remediation. Remediation progress belongs in section D.2.

##### D.2 Remediation rollup (REMEDIATION ROLLUP)

Updated **after** campaign status is `COMPLETE`, as remediation proceeds. Mutable post-audit state only.

At minimum:

- finding counts by **current remediation status**
- implementation commits
- verification progress
- deferred/rejected residual items
- remediation completion statement (when applicable)

The remediation rollup does **not** overwrite the audit rollup. Reaching remediation completion does **not** transition campaign status — the campaign was already `COMPLETE` as an audit campaign.

Every per-layer result file MUST record the same `audited_sha`. Evidence in that result refers to **that** SHA only.

**Do not** retroactively change `audited_sha` when architecture/plan sync advances `development`.

At **audit** completion, `campaign_end_sha` records the audit-closeout repository SHA after campaign-owned documentation synchronization. Remediation MUST NOT replace `campaign_end_sha` with a later remediation commit.

### Concurrent changes between layer audits

If runtime/product code changes concurrently between layer audits:

- Audit the next layer against the **then-current** exact SHA; record it explicitly.
- If concurrent changes invalidate already-audited findings, flag the affected layer for **revalidation** rather than pretending the whole campaign used one immutable source tree.

### C2. Campaign status `COMPLETE` — audit completion, not remediation completion

Campaign status tracks the **audit lifecycle**, not the remediation lifecycle.

**`COMPLETE` means:** the scoped **audit itself** is complete. It means:

- scoped layer audits finished,
- operator decisions were recorded,
- immutable layer snapshots were published,
- architecture/plan synchronization required by accepted findings was completed for audit closeout,
- campaign audit rollup (section D.1) was completed,
- audit baseline metadata was frozen (below).

**`COMPLETE` does NOT mean:**

- accepted findings were implemented,
- remediation finished,
- findings are `CLOSED`.

Remediation normally begins **against** an already `COMPLETE` audit campaign. Campaign status remains `COMPLETE` throughout post-audit remediation. `ABORTED` is the only other terminal campaign status.

**Canonical sequence:**

```text
campaign IN_PROGRESS
    ↓
perform audit
    ↓
campaign COMPLETE          ← audit baseline frozen
    ↓
remediation may begin
    ↓
finding ACCEPTED → IMPLEMENTING → IMPLEMENTED → VERIFIED → CLOSED
```

#### Audit baseline freeze at `COMPLETE`

Once campaign status becomes `COMPLETE`, the audit baseline is **historical evidence** and MUST NOT be silently rewritten by remediation.

**Frozen audit-baseline metadata:**

- `campaign_id`
- `campaign_token`
- `started_at`
- `completed_at`
- `campaign_start_sha`
- `campaign_end_sha` — audit-closeout repository SHA; not a later remediation commit
- `scope`
- `overall_verdict`

**Frozen historical audit facts:**

- per-layer `audited_sha`
- per-layer audit verdict
- original finding ID
- original severity/category at audit publication
- immutable per-layer finding evidence
- original audit rollup conclusions (section D.1)

Remediation MUST NOT change `overall_verdict` from `FAIL` / `PASS WITH GAPS` to `PASS` merely because fixes were implemented. A later audit campaign determines whether the remediated platform now deserves a different verdict.

#### Mutable post-audit remediation state

After campaign `COMPLETE`, the campaign `README.md` remains mutable **only** for post-audit remediation/verification state. The following finding-register fields may evolve:

- `status`
- `remediation_block`
- `dependencies` (where remediation planning legitimately changes)
- `arch_ref` / `plan_ref` (when current remediation trace needs updated target references)
- `implementation_commit`
- `verification_evidence`
- `notes`

Do **not** mutate immutable per-layer snapshot text. If audit-baseline metadata itself was factually wrong due to a clerical error, correction requires an explicit correction note preserving the old value/provenance — never silent rewrite.

#### Periodic audit semantics

Fixing and closing every finding does **not** retroactively change the original campaign verdict.

Example:

- `2026-08-18`: `overall_verdict` = `FAIL`; all accepted findings later `CLOSED`.
- `2026-11-18`: a fresh audit determines current platform verdict independently.

This is the mechanism by which audit history shows real architectural progress and regression.

### Lifecycle

**On campaign initialization** (both steps required; no `IN_PROGRESS` campaign without a registry row):

1. Create `docs/audit_results/<CAMPAIGN_DIR>/` and campaign `README.md` with metadata section A populated: `status` = `IN_PROGRESS`, `campaign_start_sha` set, `completed_at` = `—`, `campaign_end_sha` = `—`, `overall_verdict` = `—`.
2. Immediately add **one** row to the global campaign registry in `docs/audit_results/README.md` with the same values. Never append a duplicate registry row for the same `campaign_id`.

**During campaign:**

3. Execute layers per operator roadmap (section P); persist each layer at its `audited_sha` to `docs/audit_results/<CAMPAIGN_DIR>/<LAYER_CODE>.md`.
4. After each accepted layer: arch/plan sync → commit → record `post_sync_sha` in the campaign layer register; update the campaign finding register for lifecycle transitions.
5. Maintain the campaign finding register as the authoritative current state for remediation (section G1); do **not** mutate immutable per-layer snapshots to advance remediation status.

**On audit campaign completion** (update **both** campaign `README.md` and the **same** root-registry row):

6. Finish/update campaign `README.md` **audit rollup** (section D.1); freeze audit baseline (section C2).
7. Set campaign `status` = `COMPLETE` in campaign `README.md`; populate `completed_at`, `campaign_end_sha`, and `overall_verdict`. This closes the **audit**, not remediation.
8. Update the corresponding row in `docs/audit_results/README.md` with the same completion values. Do **not** create a second campaign-summary artifact. Root registry row remains unchanged while remediation proceeds afterward.

**On abort** (update **both** campaign `README.md` and the **same** root-registry row):

9. Set campaign `status` = `ABORTED` with completion/abort timestamp and reason; preserve the campaign directory and evidence.
10. Update the corresponding root-registry row to `ABORTED`. Do **not** append a duplicate registry row.

Registry rows are **newest-first**. The root registry is sufficient for discovering the latest campaign, the latest `COMPLETE` campaign, and any active (`IN_PROGRESS`) campaign.

**No dependency** on `progress.json`, orchestrator scripts, or IDE-specific state files.


## C1. Context and read discipline

- Audit **one layer** as the normal atomic unit.
- Begin from its architecture/plan ownership pair and known code entrypoints.
- Use path-filtered textual search before broad exploration.
- Follow concrete call paths and evidence.
- Do **not** run repo-wide semantic searches by default.
- Do **not** load all architecture/plan docs for all layers.
- Do **not** run full-suite tests when targeted tests or gates are sufficient.
- Expand context only when a concrete hypothesis or cross-layer dependency requires it.
- When expanding materially, record **why** in the layer file.

Accuracy outranks token savings, but targeted evidence outranks bulk reading.


## D. Layer-by-layer discipline

Intergrax is audited **one layer (domain) at a time** unless the operator explicitly scopes a cross-layer slice.

Per layer:

1. **Declare scope** — layer code, tier, paths, architecture/plan doc pair, in-scope/out-of-scope boundaries.
2. **Pin SHA** — all evidence references this commit.
3. **Load hierarchy** — read implementation first; use arch/plan as falsification checklists, not as ground truth.
4. **Execute falsification pass** — systematically attack section E targets within scope.
5. **Record findings** — each with ID, severity, category, evidence (sections H–I).
6. **Assign layer verdict** (section J).
7. **Present to operator** — no silent persistence of findings (section P).
8. **After acceptance** — persist artifacts, then arch/plan sync if required (section M).

Do not begin layer N+1 until layer N is `COMPLETE` or explicitly deferred in the campaign README with operator sign-off.

**Tier discipline (import boundaries):**

- Tier-0 `intergrax/` MUST NOT import from `agents/` or `applications/`.
- Tier-2 `agents/` MUST NOT import from `applications/`.
- Explicit prohibited cross-tier import violations are always at least **HIGH** unless proven unreachable dead code with evidence.
- Material bypass of a mandatory security/reliability/governance boundary should normally be **HIGH** or **CRITICAL** according to section G severity definitions.
- Do not make every harmless internal helper import automatically **HIGH**.

**Import direction is necessary but not sufficient.**

A boundary audit MUST evaluate all of the following ownership dimensions, not only import graphs:

| Dimension | Audit requirement |
|-----------|-------------------|
| **A. IMPORT OWNERSHIP** | Lower/core tiers do not depend on higher/product tiers; existing Tier-0/Tier-2 import rules remain mandatory. |
| **B. CONTRACT OWNERSHIP** | A higher layer / consumer must use the canonical public contract of the lower platform layer rather than reaching into private implementation details when a public contract exists. Audit for direct access to private/internal modules, implementation-only helpers, internal stores/repositories, internal event buses, private singleton/global state, hidden metadata channels, internal registries, and implementation classes where a public Protocol/ABC/API exists. |
| **C. DATA OWNERSHIP** | A layer must not directly read/write another layer's owned persistence/state when a canonical service/repository/contract owns that state. |
| **D. BEHAVIOR OWNERSHIP** | A higher layer must not reimplement or bypass guarantees owned by a lower layer, including policy, identity, retry, observability, tool execution, persistence, etc. |
| **E. PRODUCT LEAKAGE** | Generic/core tiers must not acquire product/application-specific vocabulary, keys, routing, state, or business semantics. |
| **F. BYPASS EQUIVALENCE** | If an alternate path does not call the canonical component literally, it must prove equivalent mandatory guarantees. "Different call chain" is acceptable only when guarantees remain structurally equivalent. |

**Normative boundary rule:**

> A boundary is violated not only by an illegal import, but also when ownership, state, behavior, or mandatory guarantees cross layers through backdoors, duplicated mechanisms, hidden metadata, direct storage access, or private API use.

### D1. Platform mechanism coverage (layer and consumer audits)

An auditor must not conclude **PASS** merely because no local defect is visible.

For each material concern owned by the audited scope:

```text
identify canonical owner
    ↓
identify canonical mechanism
    ↓
trace actual use
    ↓
verify guarantees preserved
    ↓
check for duplicate/bypass paths
```

If ownership or canonical mechanism cannot be determined, record an open question or architecture finding depending on evidence. This prevents "looks fine locally" audits.

### D2. Platform reuse and canonical mechanism discipline

**Core rule: REUSE BEFORE INVENT.**

Before accepting any new local mechanism that solves a platform-level concern, the auditor MUST determine whether Intergrax already owns a canonical mechanism for that concern.

Explicitly inspect applicable mechanisms such as (illustrative, not exhaustive):

- execution runtime / HarnessKernel / UAEP
- identity
- policy / governance
- approvals / HITL
- tool invocation / ToolGateway
- integrations / connectors
- capability registration / discovery
- registries
- memory
- RAG
- context engineering
- prompt management
- LLM/model routing
- retry / resilience / recovery
- idempotency
- observability / events / tracing
- workspace
- sandbox
- configuration
- secrets/trust boundaries
- budgets / cost controls
- state / persistence contracts

For every local competing mechanism, classify it conceptually as one of:

| Classification | Meaning |
|----------------|---------|
| **CANONICAL REUSE** | Directly uses the platform mechanism. |
| **THIN ADAPTER** | Translates product/domain input into the canonical mechanism without recreating its semantics. |
| **JUSTIFIED SPECIALIZATION** | Platform mechanism exists but domain-specific behavior legitimately sits above it while preserving the platform guarantee. |
| **DUPLICATED PLATFORM MECHANISM** | Recreates a platform-owned concern locally. |
| **BYPASSED PLATFORM MECHANISM** | Avoids a canonical mandatory mechanism and loses/changes guarantees. |
| **MISSING PLATFORM CAPABILITY** | Platform truly lacks the required reusable capability. |

**Normative rules:**

- **DUPLICATED PLATFORM MECHANISM** without explicit architectural justification is a finding.
- **BYPASSED PLATFORM MECHANISM** is a finding whenever required guarantees differ or disappear.
- If the platform genuinely lacks the capability, the consumer/layer must not silently establish a second universal mechanism as an accidental local standard. Record the architectural gap and make ownership explicit.
- Thin adapters are desirable when they preserve canonical semantics.
- Do not force reuse of a platform component that demonstrably does not satisfy the required contract; in that case record an architecture gap rather than pretending reuse is correct.

**Parallel universal mechanism** — a mechanism that independently tries to own a concern already intended to be owned globally by another Intergrax platform component. Unjustified parallel universal mechanisms are architecture findings even if both implementations individually work.

Reuse is about shared **ownership / guarantees / semantics**, not superficial code deduplication. Do not create premature shared abstractions merely because two functions look similar.

### D3. Platform consumer audit scope

A **PLATFORM CONSUMER AUDIT** is an explicit audit scope type. A platform consumer may be:

- application
- agent
- subagent
- plugin
- integration adapter
- reference product
- other higher-level component consuming Intergrax capabilities

A consumer audit is **not** primarily asking "Does this product feature work?" It asks **"Is this component a correct Intergrax platform consumer?"** It may inspect product correctness when necessary to determine platform conformance, but its primary concern is platform usage.

Consumer audits MUST use the same campaign/finding machinery as domain audits. Use a stable `LAYER_CODE`, for example:

- `APPLICATION_<STABLE_NAME>`
- `AGENT_<STABLE_NAME>`
- `PLUGIN_<STABLE_NAME>`

Do not require these exact prefixes if a better stable domain code already exists, but the code must be explicit and stable. Do **not** change finding ID syntax (section H).

**Mandatory consumer falsification questions:**

- Does the consumer call platform public contracts or internal implementations?
- Does it bypass HarnessKernel/runtime/governance/tool gateways?
- Does it create its own execution identity?
- Does it maintain its own retry loop where platform reliability should own it?
- Does it directly call external SDKs where Intergrax integration/tool contracts should own access?
- Does it create product-local memory/RAG/context/prompt registries duplicating platform facilities?
- Does it emit its own incompatible observability/event model?
- Does it manage state/persistence owned by another platform layer?
- Does it carry untyped `dict[str, Any]`/metadata payloads across boundaries instead of typed contracts?
- Does it use reflection/dynamic dispatch to emulate capability contracts?
- Does it embed product concerns down into reusable tiers?
- Does it correctly declare capabilities and dependencies?
- Does it preserve policy, identity, auditability, retry, cancellation, and HITL semantics supplied by the platform?
- Could a second application reuse the same mechanism, or is platform logic trapped inside this consumer?

**Consumer conformance matrix (required for platform consumer audits):**

For each applicable platform concern, record one of:

- `REUSED`
- `THIN ADAPTER`
- `JUSTIFIED SPECIALIZATION`
- `DUPLICATED`
- `BYPASSED`
- `MISSING PLATFORM CAPABILITY`
- `NOT APPLICABLE`
- `INSUFFICIENT EVIDENCE`

At minimum evaluate applicability of:

1. canonical execution/runtime
2. identity / Task / Run / Attempt
3. policy / governance
4. approval / HITL
5. tools / ToolGateway
6. integrations/connectors
7. memory
8. RAG
9. context engineering
10. prompt/model routing where relevant
11. retries / recovery / idempotency
12. observability / events / tracing
13. workspace / sandbox
14. configuration / secrets
15. budgets / cost controls
16. registries / capability discovery
17. state/persistence ownership
18. typed request/result contracts

The matrix must **not** force irrelevant mechanisms onto every consumer. `NOT APPLICABLE` is valid but requires a concise reason for material platform concerns. `DUPLICATED`, `BYPASSED`, `MISSING PLATFORM CAPABILITY`, or `INSUFFICIENT EVIDENCE` should drive findings/open questions as warranted by severity and evidence.

---

## E. Falsification targets

For each in-scope component, attempt to disprove stated invariants. At minimum, probe:

### Architecture and structure

- **Architecture bypasses** — code paths that skip documented gates, policies, or lifecycle stages.
- **Alternate paths** — feature flags, env toggles, legacy branches, "admin" or debug entrypoints that change behavior.
- **Dependency violations** — cross-tier imports, circular deps, runtime plugin loading that breaks boundaries; contract/state/behavior ownership violations beyond import graphs (section D).
- **Duplicated / bypassed platform mechanisms** — local retry loops, registries, policy evaluators, tool gateways, memory/RAG/context stacks, or observability models that recreate or bypass canonical platform ownership (section D2).
- **Hidden global state** — module-level singletons, process-wide caches, class variables, thread-locals used as implicit channels.
- **Product leakage** — application-specific logic in runtime/agents tiers; domain logic in wrong tier.
- **Host-specific behavior** — paths, shells, OS assumptions, hardcoded developer machine layout.
- **False portability** — claims of cloud-agnostic or multi-host operation contradicted by local-only locks, paths, or APIs.
- **Legacy paths** — deprecated modules still reachable; dual implementations where only one is documented.
- **Doc drift** — architecture/plan describes behavior or APIs that code no longer implements.

### Reliability and correctness

- **Fail-open** — errors swallowed, defaults that grant access, silent fallbacks, "best effort" that hides failure.
- **Restart / resume** — state after crash, partial writes, checkpoint replay, stale checkpoints.
- **Retries** — unbounded retry, retry without idempotency, thundering herd, duplicate side effects.
- **Idempotency** — duplicate delivery, at-least-once semantics without dedup keys.
- **Concurrency** — races, TOCTOU, missing locks, async boundaries, shared mutable state.
- **Multi-host** — split brain, non-distributed locks, file-based coordination on shared FS assumptions.
- **Durability** — fsync, atomic rename, WAL gaps, "write succeeded" before durable.
- **Failure / recovery** — partial failure leaves system inconsistent; no reconciliation path.

### Security and identity

- **Identity** — conflation of user, tenant, session, job, agent instance; missing propagation across layers.
- **Authorization** — checks only at edge; missing downstream enforcement; trust of caller-supplied IDs.
- **Security by convention** — "internal only", "not exposed", "trusted network" without enforcement.
- **Secrets** — logging, env dumps, error messages, test fixtures, committed placeholders treated as real.

### Operations and scale

- **Observability gaps** — failures without correlation IDs, missing metrics on critical paths, undebuggable async.
- **Scaling limits** — O(n) scans, unbounded queues, single-process bottlenecks presented as scalable.
- **Unbounded state** — caches, logs, queues, DB tables, in-memory maps without TTL or cap.
- **Cost / budget** — unbounded LLM/tool calls, missing limits, retry storms amplifying cost.

### Contracts and typing

**Default invariant:**

> Critical Intergrax boundaries are statically explicit and strongly typed. Dynamic structures are exceptions requiring evidence-based justification, not the default integration mechanism.

Preferred contract forms should include, where appropriate:

- typed domain models
- dataclasses / validated models
- `Protocol`
- ABC / explicit base classes
- enums
- `NewType` / value objects / typed identifiers
- explicit request/result models
- typed capability interfaces
- typed registries
- discriminated/versioned payload models where external variation requires them

Strong typing may use Protocols, composition, models, value objects, ABCs, or base classes as appropriate. Do **not** force one giant base class.

**Audit triggers** (do **not** ban these constructs universally):

- `dict[str, Any]`
- `Any`
- generic `object`
- arbitrary nested dictionaries/lists used as contracts
- magic string metadata keys controlling behavior
- `getattr`
- `setattr`
- `hasattr` used for capability/behavior discovery
- string-based method dispatch
- reflection used to select critical behavior
- `importlib` / dynamic imports
- monkey patching outside intentionally isolated tests
- runtime mutation of interfaces/classes
- unvalidated JSON crossing trust or ownership boundaries
- dynamically shaped payloads crossing platform boundaries

Risk rises when dynamic structures cross trust boundaries, ownership boundaries, persistence boundaries, execution boundaries, or public author/consumer contracts. Do **not** treat every `dict` as a finding.

**Acceptable exception model** — dynamic structures MAY be appropriate for:

- raw external input before validation
- opaque extension payloads deliberately preserved as opaque
- narrowly scoped serialization/deserialization
- validated plugin discovery
- local non-contract scratch state
- test doubles / test instrumentation

When used:

1. keep the dynamic surface narrow,
2. validate/normalize at the boundary,
3. convert into typed contracts as early as practical,
4. do not use dynamic reflection as a hidden substitute for a canonical platform contract,
5. document why static typing is not appropriate.

**Falsification question:**

> Could this runtime decision be expressed through an existing or new typed Protocol/model/enum/value object instead of reflection or `Any`-shaped state?

If **YES** and no credible reason for dynamism exists, record an appropriate finding.

- **Weak contracts** — undocumented JSON shapes, optional fields that change semantics, version skew.
- **Hidden dependencies** — import side effects, registry mutation at import time, implicit plugin discovery.

### Verification quality

- **"Done" without behavior** — plan marks complete but no implementation or no observable effect.
- **Tests missing invariants** — critical properties never asserted (authz, idempotency, failure modes).
- **Paper abstractions** — interfaces with single no-op implementation; layers that delegate without adding enforcement.
- **Missing negative tests** — only happy path; no tests for denial, timeout, malformed input, crash mid-flight.
- **Cross-layer mismatches** — tier A assumes tier B provides guarantee B does not implement.

Record **negative results** (target examined, not falsified) briefly in the layer file — they bound confidence but are not findings.

---

## F. Production perspective

Evaluate as if deployed tomorrow under adversarial conditions. Ask:

| Stress | Questions |
|--------|-----------|
| **Failures** | What breaks first? Is failure contained? Is it visible? |
| **Restart** | What is lost, duplicated, or corrupted after process/host restart? |
| **Duplicate delivery** | Messages, webhooks, tool calls, file events — what happens twice? |
| **Retries** | Who retries, with what backoff, and what if the first attempt partially succeeded? |
| **Parallel execution** | Same resource mutated concurrently — last write wins? corruption? |
| **Partial outage** | DB up, queue down; API up, auth down — degrade safe or fail catastrophically? |
| **Multi-tenant** | Can tenant A read, trigger, or exhaust tenant B's resources? |
| **Scale** | 10x jobs, 100x files, long-running campaigns — what exhausts memory, FDs, API quotas? |
| **Malicious input** | Oversized payloads, path traversal, injection, prompt injection at tool boundaries. |
| **Policy denial** | When policy says no, is denial enforced at every layer or only logged? |
| **Timeouts** | Hung calls, partial results, cancel propagation. |
| **Drift** | Config, schema, feature flags, dependency versions — runtime vs documented. |

Lack of evidence for production-safe behavior under these stresses is a finding when the architecture claims that property.

---

## G. Finding classification

### Severity

| Level | Definition |
|-------|------------|
| **CRITICAL** | Exploitable security issue, data loss/corruption, cross-tenant breach, or production outage likely under normal operation. |
| **HIGH** | Serious correctness/reliability flaw, boundary violation, or fail-open with material impact; workaround difficult. |
| **MEDIUM** | Real defect or risk with limited blast radius, missing enforcement on non-critical path, or significant test/doc gap. |
| **LOW** | Minor inconsistency, maintainability hazard, weak typing on internal-only path, cosmetic doc drift. |

### Categories (use one primary)

- **IMPLEMENTATION DEFECT** — implementation contradicts specified behavior or invariants.
- **ARCHITECTURE DEFECT** — target design is wrong, incomplete, or not production-grade.
- **IMPLEMENTATION/ARCHITECTURE DRIFT** — code and architecture disagree; gap ownership must be explicit.
- **BOUNDARY VIOLATION** — tier/import/product-surface rule broken.
- **SECURITY** — authn/authz/secrets/trust-boundary issue.
- **RELIABILITY** — crash, duplication, lost work, non-idempotent side effects.
- **OPERABILITY** — observability, deployability, config, runbooks inadequate for claimed ops model.
- **SCALABILITY / COST** — unbounded resource use or economic risk.
- **TEST GAP** — missing or misleading tests; CI gives false confidence.
- **DOC DRIFT** — documentation or plan claims falsified by code.
- **PROCESS / CLAIM** — `Done` or `PASS` unsupported by evidence.

---

## G1. Finding lifecycle status

| Status | Meaning |
|--------|---------|
| `PROPOSED` | Produced by audit; not yet operator-accepted. |
| `ACCEPTED` | Operator accepted; queued for remediation. |
| `IMPLEMENTING` | Remediation in progress. |
| `IMPLEMENTED` | Fix applied; awaiting independent verification. |
| `VERIFIED` | Independent verification passed. |
| `CLOSED` | Independent verification passed and remediation for this finding is finalized in the campaign remediation rollup (section D.2). Does **not** close the audit campaign. |
| `DISPUTED` | Operator disputes; finding and evidence preserved without acceptance. |
| `DEFERRED` | Postponed; requires rationale and revisit trigger. |
| `REJECTED` | Invalid or out of scope; requires rationale. |
| `WITHDRAWN` | Withdrawn; ID is not reused. |

Rules: audit produces `PROPOSED`; operator acceptance → `ACCEPTED`; remediation starts → `IMPLEMENTING`; implementer may reach `IMPLEMENTED`; implementer **must not** self-certify `VERIFIED` or `CLOSED`; independent verifier → `VERIFIED`; `CLOSED` follows `VERIFIED` and recording the finding's final remediation disposition in the remediation rollup (section D.2). Closing a finding does **not** transition campaign status — the campaign was already `COMPLETE` as an audit campaign.

**State ownership:**

- **Campaign `README.md` finding register** — authoritative **current** lifecycle, remediation, and verification state (`PROPOSED` through `CLOSED`, including `IMPLEMENTING`, `IMPLEMENTED`, `VERIFIED`).
- **Per-layer snapshot** (`docs/audit_results/<CAMPAIGN_DIR>/<LAYER_CODE>.md`) — immutable historical observation plus **publication-time operator decision** only (section O). Do **not** use per-layer files as the current location for `IMPLEMENTING`, `IMPLEMENTED`, `VERIFIED`, or `CLOSED`.


## H. Finding IDs

Format (immutable once published):

```
AUDIT-<CAMPAIGN_TOKEN>-<LAYER_CODE>-NN
```

**Campaign token mapping** (deterministic, from campaign directory name):

| Campaign directory | `CAMPAIGN_TOKEN` |
|--------------------|------------------|
| `YYYY-MM-DD` | `YYYYMMDD` (e.g. `20260818`) |
| `YYYY-MM-DD_run-2` | `YYYYMMDD-R2` (e.g. `20260818-R2`) |
| `YYYY-MM-DD_run-3` | `YYYYMMDD-R3` (e.g. `20260818-R3`) |

- `CAMPAIGN_TOKEN` — immutable after publication; never reused across campaigns.
- `<LAYER_CODE>` — uppercase domain/layer identifier (e.g. `ORCHESTRATION`, `MEMORY`, `PLATFORM_FOUNDATION`).
- `NN` — two-digit sequence per layer per campaign (`01`, `02`, ...).

**Examples:**

- First campaign (`2026-08-18`): `AUDIT-20260818-MEMORY-01`
- Same-day run 2 (`2026-08-18_run-2`): `AUDIT-20260818-R2-MEMORY-01`, `AUDIT-20260818-R2-CROSS-01`
- Same-day run 3 (`2026-08-18_run-3`): `AUDIT-20260818-R3-MEMORY-01`

**Rules:**

- IDs are never reused or renumbered.
- If a finding is withdrawn, mark status `WITHDRAWN` with reason; do not delete.
- If duplicate root cause, cross-reference IDs; keep one primary.
- `WITHDRAWN` IDs remain reserved.

---

## I. Required evidence (per finding)

Every finding MUST include:

1. **Claim falsified** — quote or paraphrase the architecture, plan, or implied invariant.
2. **Observation** — what the implementation actually does (behavior, not intent).
3. **Location** — file path(s) and line range(s) at audited SHA, or runtime trace/log snippet.
4. **Reproduction or inspection steps** — commands, call sequence, or static analysis path another auditor can repeat.
5. **Impact** — who/what is affected under section F stresses.
6. **Confidence** — `CONFIRMED` (reproduced or directly read) | `PROBABLE` (strong static evidence) | `HYPOTHESIS` (needs operator follow-up; not for CRITICAL without escalation).

Findings without item 3 and 4 are **insufficient** — downgrade to notes or block verdict (section J).

---

## J. Verdicts

Per-layer and campaign rollup:

| Verdict | Criteria |
|---------|----------|
| **PASS** | No CRITICAL/HIGH findings; MEDIUM findings acknowledged or accepted by operator; scope fully executed. |
| **PASS WITH GAPS** | No CRITICAL; HIGH items have documented mitigation or time-bound acceptance; known MEDIUM/LOW listed. |
| **FAIL** | One or more CRITICAL findings, or multiple unmitigated HIGH, or core scope claims falsified. |
| **BLOCKED / INSUFFICIENT EVIDENCE** | Cannot access code, SHA mismatch, scope ambiguity, or missing reproduction — audit cannot complete honestly. |

**Campaign verdict** is the worst layer verdict unless operator documents explicit acceptance of scoped exclusions.

---

## K. Prior-audit comparison (without anchoring)

When a prior campaign exists for the same layer:

1. List **resolved** findings (with fix SHA or evidence of removal).
2. List **regressions** (prior fixed, now present again).
3. List **still open** findings (by old ID if still valid, or new ID if changed manifestation).
4. List **new** findings.

**Anti-anchoring rules:**

- Do not assume prior PASS implies current PASS.
- Do not downgrade severity because a prior audit "already knew" — re-verify.
- Do not inherit prior verdict; compute fresh from current SHA.
- Prior audits are section B rank 5 only.

---

## L. Audit vs implementation

**During an audit:**

- **No runtime fixes** — do not patch production code, tests, or docs to "clear" findings mid-run.
- **No scope creep into implementation** — note defects; implementation is a separate change with separate review.

**Required sequence:**

1. **Inspect** — read code, config, infra as needed at pinned SHA.
2. **Falsify** — attack section E targets; run tests as evidence gathering, not as goal.
3. **Evidence** — attach section I proof to each finding.
4. **Classify** — section G severity and category.
5. **Present** — deliver findings to operator before treating them as accepted (section P).
6. **Accept** — operator confirms, disputes, or defers each finding.
7. **Persist** — write accepted artifacts to `docs/audit_results/<CAMPAIGN_DIR>/`.
8. **Sync arch/plan** — after acceptance, update docs per section M.

---

## M. Architecture / plan synchronization (after acceptance)

After operator accepts findings:

1. **Architecture** — update target docs when an accepted finding requires target-design change, especially for **ARCHITECTURE DEFECT** or **IMPLEMENTATION/ARCHITECTURE DRIFT**. Implementation-only defects (**IMPLEMENTATION DEFECT** and similar) may require plan updates without changing architecture when the existing target is already correct.
2. **Plan** — adjust task status, add remediation items, link finding IDs; never mark `Done` without behavior evidence.
3. **Cross-links** — layer audit file links to arch/plan sections; plan cites `AUDIT-...` IDs; campaign finding register holds current remediation trace.
4. **No silent rewrite** — doc changes that reverse a finding require a new audit or explicit operator waiver recorded in the campaign `README.md`.

Synchronization is **post-acceptance** only — audits do not edit arch/plan to match broken code without recording the gap as a finding or accepted debt. Never update immutable per-layer report text merely to advance remediation status; update the campaign `README.md` finding register instead.

---

## N. Cross-layer review (rollup)

When multiple layers complete in one campaign, update the campaign `README.md` **audit rollup** (section D.1):

1. **Scope table** — layer, SHA, verdict, finding counts by severity.
2. **Cross-layer findings** — mismatches spanning tiers (e.g. orchestration assumes memory guarantee memory does not provide).
3. **Systemic themes** — explicitly look for repeated:
   - `dict[str, Any]` / `Any` boundary usage
   - reflection / magic-string dispatch
   - duplicate registries
   - duplicate policy systems
   - duplicate retry systems
   - duplicate event/observability systems
   - duplicate context/memory/RAG mechanisms
   - direct storage ownership violations
   - consumer bypasses
   - repeated thin adapters that may indicate a missing shared platform abstraction

   Repeated adapters may indicate healthy reuse **or** a missing reusable abstraction. The auditor must distinguish them rather than automatically flag duplication. Also include fail-open patterns and test gaps.
4. **Campaign verdict** — per section J.
5. **Recommended remediation order** — operator-facing, not implementation.

Cross-layer issues get their own IDs using layer code `CROSS` (e.g. `AUDIT-20260818-CROSS-01`, `AUDIT-20260818-R2-CROSS-01`) in the campaign `README.md` rollup.

---

## O. Required per-layer result format

Each `docs/audit_results/<CAMPAIGN_DIR>/<LAYER_CODE>.md` is an **immutable historical snapshot** after publication. It records the original finding, evidence, reproduction, impact, and publication-time operator decision. It does **not** store future live remediation state.

Each per-layer file MUST contain:

```markdown
# <LAYER_CODE> — Platform Audit

## Metadata
- Campaign date:
- Layer code:
- Tier(s):
- audited_sha:
- Status: IN_PROGRESS | COMPLETE | ABORTED
- Auditor:
- Architecture doc(s):
- Plan doc(s):
- Scope in:
- Scope out:
- Prior audit reference(s):

## Executive summary
(<=1 paragraph: verdict, headline risks)

## Verdict
(PASS | PASS WITH GAPS | FAIL | BLOCKED / INSUFFICIENT EVIDENCE)

## Findings
(enumerated AUDIT-... entries, each with section I evidence)

### AUDIT-<CAMPAIGN_TOKEN>-<LAYER_CODE>-01
- Severity:
- Category:
- Status at publication: PROPOSED | ACCEPTED | DEFERRED | DISPUTED | REJECTED | WITHDRAWN
- Claim falsified:
- Observation:
- Location:
- Reproduction:
- Impact:
- Confidence:

## Falsification log (negative results)
(targets examined, not falsified — brief)

## Prior-audit comparison
(section K, or N/A)

## Consumer conformance matrix
(REQUIRED for PLATFORM CONSUMER AUDITS only — omit for DOMAIN / CONCEPTUAL audits)

| concern | canonical platform owner/mechanism | observed consumer mechanism | classification | evidence | finding_id / note |
|---------|-----------------------------------|-----------------------------|----------------|----------|-------------------|

## Open questions / blocked items

## Operator acceptance
- Date:
- Accepted findings:
- Deferred findings:
- Disputed findings:
```

**Publication-time status values** represent audit/operator decision at persist time. Normally: `ACCEPTED`, `DEFERRED`, `DISPUTED`, `REJECTED`, `WITHDRAWN`. If a finding is persisted before operator decision for an explicitly authorized reason, `PROPOSED` may be used.

Do **not** record `IMPLEMENTING`, `IMPLEMENTED`, `VERIFIED`, or `CLOSED` in per-layer snapshots. Those lifecycle transitions belong **only** in the campaign `README.md` finding register.

---

## P. Operator workflow

1. **Roadmap before layer** — operator defines campaign date, SHA (or branch to pin), layer order, and tier scope before the auditor starts.
2. **Scope confirmation** — auditor restates scope; operator confirms or corrects in writing (chat log or campaign README).
3. **Findings before persist** — present all findings and draft verdict; **no** writing to `docs/audit_results/` until operator reviews (except optional `IN_PROGRESS` stub without findings).
4. **Acceptance gates:**
   - Operator accepts, defers, or disputes each finding.
   - CRITICAL/HIGH deferrals require explicit rationale and target date in campaign README.
   - Only then persist final layer files and update register.
5. **Arch/plan sync** — separate step after acceptance (section M); may be same or follow-up change set.
6. **Audit campaign close** —
   1. Finish/update campaign `README.md` audit rollup (section D.1); freeze audit baseline (section C2).
   2. Set campaign status `COMPLETE` or `ABORTED` in campaign `README.md`.
   3. Update the corresponding row in root `docs/audit_results/README.md`.
   4. Do **not** create a second campaign-summary artifact (no `CAMPAIGN_SUMMARY.md`).
   5. Remediation (if any) begins afterward per [AUDIT_REMEDIATION_PROTOCOL.md](AUDIT_REMEDIATION_PROTOCOL.md); campaign status remains `COMPLETE`.

---

## Appendix: Audit scope shapes and layer codes (non-exhaustive)

Protocol v2.1 supports three audit shapes:

1. **DOMAIN / LAYER AUDIT** — one architecture/domain pair (section D).
2. **CONCEPTUAL / CROSS-DOMAIN AUDIT** — explicit invariant slice spanning domains (below).
3. **PLATFORM CONSUMER AUDIT** — application, agent, plugin, or integration adapter as platform consumer (section D3).

A **domain / layer audit** normally maps to one architecture/domain pair aligned with `docs/project/architecture/<DOMAIN>.md`, e.g.:

`PLATFORM_FOUNDATION`, `ORCHESTRATION`, `NEXUS_EXECUTION_FLOW`, `UNIFIED_EXECUTION_RUNTIME`, `MEMORY`, `RAG`, `CONTEXT_ENGINEERING`, `INTEGRATIONS`, `MODALITY`, `RELIABILITY_FAILURE_AND_HITL`, `ADAPTIVE_HARNESS_INTELLIGENCE`, `PLATFORM_PLUGINS`

Capability-scoped audits may use capability hub names from `docs/project/capabilities/`.

### Conceptual / cross-domain slices

A campaign may explicitly define a **conceptual invariant slice** spanning multiple domains when the operator declares that scope. Example: `STRATEGIC_HARNESS_MODEL` may legitimately span agent execution, UER, Nexus, identity, and policy boundaries.

Requirements for a conceptual slice:

- stable `LAYER_CODE`
- explicit in-scope architecture docs
- explicit code entrypoints
- explicit scope out
- exact `audited_sha`
- no double-counting hidden by the cross-layer scope

This preserves the Intergrax audit roadmap without forcing every audit layer to equal one Markdown domain file.

---

*End of Intergrax Platform Audit Protocol v2.1*

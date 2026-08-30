# Intergrax Platform Proof Protocol

**Status:** Canonical  
**Version:** 1.0 (PP-2)  
**Audience:** Maintainers, architects, proof authors  
**Scope:** Executable falsification proofs under `platform_proofs/` — **SCENARIO** and **CONFORMANCE** classes

---

## Core mindset

> **The Intergrax Proof Library holds executable falsification attempts for bounded real-world Scenario claims and reusable platform Conformance claims.**

It is **not** a happy-path demo. The author must try to prove the claim false under named conditions. A PASS backed by explicit invariants and at least one meaningful negative path is valid. A FAIL with clear evidence is also valid.

This protocol governs **SCENARIO** and **CONFORMANCE** proofs under `platform_proofs/`. **Product proofs** remain with owning products under `applications/`.

**Normative ownership rule (non-negotiable):**

> A Product Proof may demonstrate that a product successfully consumes platform mechanisms, but product-specific execution **MUST NOT** substitute for an independently owned Platform Proof of the reusable platform capability.

**Frozen distinctions:**

```text
implementation          ≠ platform proof
platform proof          ≠ product proof
product proof           ≠ real-user validation
real-user validation    ≠ commercial validation
any of the above        ≠ production readiness
```

Reuse the public proof/claims philosophy in [`PUBLIC_PROOF_AND_CLAIMS_MODEL.md`](../docs/project/maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md) and [`PROOFS.md`](../docs/project/proofs/PROOFS.md). Do not introduce a conflicting maturity model.

---

## A. Purpose

| Activity | What it checks |
|----------|----------------|
| **Unit test** | Bounded local contract |
| **Integration test** | Cooperating components under test scope |
| **Scenario proof** | Whether a **bounded real-world system claim** holds under adversarial conditions via a production-capable application component |
| **Conformance proof** | Whether a **reusable platform mechanism** exhibits the claimed behavior across realistic boundaries |
| **Product proof** | A product workflow end-to-end |
| **Real-user validation** | User outcomes in realistic use |
| **Commercial validation** | Market or business value |

Scenario proofs are **problem-first** and may exercise multiple domains and mechanisms. Conformance proofs are **mechanism-first** and serve CI, regression, and contract verification. Neither substitutes for product proofs.

**Examples:**

- AI incident investigation — Scenario proof (`platform_proofs/scenarios/`)
- TOOLS iterative investigation — Conformance proof (`platform_proofs/`)
- LKW Product Quick Start — product proof (`applications/local_workspace_application/`)
- LKW exercising tool runtime during a product workflow — product consumption, **not** independent platform proof

**LKW is a product.** LKW must not appear as a platform domain or inside `platform_proofs/`.

**Scenario application ≠ Product.** A Scenario may possess real business workflow and production-capable application core. That does **not** make it a product-owned proof. Scenario proofs belong under `platform_proofs/scenarios/`; product proofs under `applications/`. Do not reclassify a Scenario as Product Proof because it has business workflow.

---

## B. Proof Library classes

The Intergrax Proof Library distinguishes two public proof classes via `library_class` in `proof.json` (`intergrax.platform_proof_descriptor.v3`):

| Class | Role | Entry framing |
|-------|------|---------------|
| **CONFORMANCE PROOF** | Executable evidence for a **specific platform mechanism** — CI, regression, development confidence, architectural assurance | Mechanism-first |
| **SCENARIO PROOF** | **Production-capable autonomous application component** that solves a concrete real-world problem; Proof Library layer falsifies and evidences that application — it does not replace it | Problem-first |

Both classes remain **executable falsification attempts** — not demos. **Platform proof ≠ product proof.** Product proofs stay under `applications/`.

**SCENARIO normative definition:**

> A Scenario Proof is a production-capable autonomous application component that solves a concrete real-world problem through normal Intergrax runtime and platform contracts. The Proof Library layer executes adversarial cases, falsifies claims, captures evidence, evaluates outcomes, and renders reports; it does not replace the application itself.

**SCENARIO observability (normative):** a Scenario Proof **MUST NOT** be a black box. Canonical Scenario execution **MUST** be observable and reconstructable from structured production-path artifacts. Every material autonomous decision, external action, evidence acquisition, claim transition, challenge, recovery action, and terminal outcome **MUST** be traceable through application/platform observability (`TraceEvent`, `ToolCallTrace`, typed diagnostics, and evidence projection) — not invented by the Proof layer after execution. Explainability **MUST** use explicit decision summaries, objectives, evidence references, selected actions, and bounded rationales; hidden chain-of-thought is neither required nor accepted. Operational detail: [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md) § Mandatory observability, explainability, and diagnostics.

Use **production-capable** (architecture suitable for real deployment; no test-only application shortcuts) — not "production-proven" or "production-validated". Scenario Proof acceptance does **not** automatically establish production validation in live user/environment terms. Operational workflow: [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md) § Scenario Proof — production-capable application contract.

Every proof declares **`domains_exercised`** (non-empty; no owning or primary domain) and **`mechanisms_exercised`**. A proof does not belong to one domain — it exercises one or more domains. Library metadata (`library_class`, `domains_exercised`, `mechanisms_exercised`, SCENARIO problem fields) is descriptor-owned and does not appear in runner-facing `ProofManifestEntry` unless execution genuinely requires it (currently: domain metadata is not execution authority).

**Scenario documentation:** Scenario design is **problem-owned and multi-domain by default** — participating domains are discovered during Intergrax Fit, then declared truthfully in `domains_exercised` when the proof package ships. See [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md).

Every descriptor declares `library_class`, `domains_exercised`, and `mechanisms_exercised`. SCENARIO additionally requires `problem_category`, `problem_summary`, and `failure_mode_summary`. CONFORMANCE forbids those problem fields.

---

## C. Source of truth

For a platform proof, resolve conflicts in this order:

| Rank | Source | Role |
|------|--------|------|
| **1** | **Implementation at exact SHA** | What the system actually does at proof time |
| **2** | **Architecture owner** | Claim boundary and mechanism under proof |
| **3** | **Tests / integration evidence** | Supporting evidence — not substitute for proof |
| **4** | **Proof execution evidence** | `SuiteReceipt` and proof artifacts from `scripts/proof/` |
| **5** | **Published proof documentation** | Scenario, invariants, limitations under `platform_proofs/` |

Architecture defines intended claim boundaries but is **not** runtime evidence.

---

## D. Proof identity

Every executable platform proof must have a stable **`proof_id`**.

**SCENARIO proofs:** `SCENARIO-<SCENARIO-ID>` — uppercase, hyphenated (e.g. `SCENARIO-AI-INCIDENT-INVESTIGATION`).

**CONFORMANCE proofs:** `<DOMAIN>-<CAPABILITY>` or mechanism-appropriate identifier — uppercase, hyphenated, consistent with existing manifest style.

Do **not** use domain-capability naming as the universal pattern for Scenario proofs. Scenario identity is class-appropriate and problem-owned.

Canonical executable identity is declared in two layers during migration:

| Layer | Role |
|-------|------|
| **`proof.json`** (package-owned) | Static descriptor — discovery source (PP-SUITE-1) |
| **`ProofManifestEntry`** (runner-facing) | Normalized manifest contract consumed by `scripts/proof/` |

The central manifest in `scripts/proof/intergrax_proof_manifest.py` remains authoritative for legacy and product entries. Descriptor-backed Platform Proofs under `platform_proofs/` are discovered automatically from package `proof.json` files (PP-SUITE-2). Do **not** duplicate conflicting metadata in ad-hoc local config files — a descriptor/static migration twin with non-equivalent execution metadata fails manifest loading.

---

## D2. Platform Proof Package Contract (PP-SUITE-1)

Platform proofs under `platform_proofs/` are **self-describing packages**:

**Conformance proofs** (domain-oriented path):

```text
platform_proofs/<domain>/<proof_slug>/
    proof.json          # static descriptor (canonical filename)
    run_proof.py        # executable entrypoint
    ...                 # proof-owned implementation
```

**Scenario proofs** — design stage (no executable artifacts yet):

```text
platform_proofs/scenarios/<scenario_slug>/
    README.md           # public gateway
    SCENARIO_SPEC.md    # deep canonical contract (A/B/C/D/E)
    assets/             # optional — after Scenario Quality Gate
```

After implementation, Scenario packages add `proof.json`, `run_proof.py`, and other runtime artifacts per [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md). Scenario proofs are **problem-first** and may exercise **multiple domains**; Conformance proofs remain **mechanism-first** and declare every domain they actually exercise in `domains_exercised`.

**Scenario lifecycle governance (normative):**

- Scenario implementation artifacts **MUST NOT** be initialized before lifecycle acceptance (`ACCEPTED FOR IMPLEMENTATION` in `SCENARIO_SPEC.md` frontmatter).
- The canonical implementation skeleton **MUST** be created through `scripts/proof/init_scenario_implementation.py` — not by manual directory layout.
- `scripts/proof/create_scenario_proof.py` creates the design package only (`README.md` + `SCENARIO_SPEC.md`); it does **not** authorize implementation or generate runtime artifacts.

Operational procedure: [PLATFORM_PROOF_AUTHORING_GUIDE.md § Canonical Scenario Lifecycle](PLATFORM_PROOF_AUTHORING_GUIDE.md#canonical-scenario-lifecycle).

**Why static JSON (`proof.json`):** language-neutral, human-readable, machine-validated, deterministic, inspectable by CI, and free of Python import side effects during discovery. Discovery must **not** import proof modules or execute `run_proof.py` to read metadata.

**Descriptor schema:** `intergrax.platform_proof_descriptor.v3` — implemented in `scripts/proof/intergrax_platform_proof_descriptor.py` and loaded by `scripts/proof/intergrax_platform_proof_descriptor_loader.py`. Only the current schema version is accepted; v2 and v1 are rejected with no fallback.

**Command contract:** structured `argv` only (`shell=False`). No shell strings.

**Path safety:** descriptor location defines package root; entrypoints must resolve inside the repository and package; `..` traversal and repo-escaping absolute paths are rejected.

**Discovery (PP-SUITE-2):** recursively scan `platform_proofs/` for `proof.json`, validate, ensure unique `proof_id`, normalize to `ProofManifestEntry`, merge with static legacy entries, fail closed on any invalid package. Descriptor-backed entries replace semantically equivalent static migration twins exactly once; conflicting duplicates fail manifest loading. Static central registration remains only for not-yet-migrated legacy and product proofs.

**Evidence and report:** descriptors may declare `evidence_schema`, `expected_artifacts`, and `report_required`. Machine evidence validation (PP-SUITE-3) and report verification (PP-SUITE-5) are follow-on tasks; `report_required=false` is allowed during renderer migration (PP-REPORT-3/4).

**Artifact contract (PP-SUITE-4):** a descriptor-backed Platform Proof is successful only when (1) subprocess contract passes, (2) required machine evidence passes validation when declared, and (3) every required declared artifact in `expected_artifacts` satisfies the generic artifact contract. Optional artifacts may be absent; if present they must still be safe regular non-symlink files. Artifact paths resolve only under the runner-owned proof artifact directory.

**Roadmap:** PP-SUITE-1 package contract · PP-SUITE-2 dynamic discovery · PP-SUITE-3 evidence validation · PP-SUITE-4 artifact verification · PP-REPORT-3 generic HTML renderer · PP-REPORT-4 TOOLS report integration · PP-SUITE-5 report contract verification · PP-SUITE-6 CI regression profiles.

**Transition:** Phase 1 — descriptor-backed packages ship `proof.json`. Phase 2 — dynamic discovery (current). Phase 3 — static manifest coexists for unmigrated proofs. Phase 4 — migrate remaining platform proofs. Phase 5 — remove static platform registrations when complete. Duplicate `proof_id` across static manifest and discovery fails unless entries are semantically equivalent migration twins (descriptor wins once).

---

## E. Required proof claim

Every proof must state **one bounded falsifiable claim**.

| Bad | Good (SCENARIO) | Good (CONFORMANCE) |
|-----|-----------------|---------------------|
| "Tools works." | "Under adversarial incident evidence, the bounded investigation application reaches a defensible root-cause or explicit UNRESOLVED without unsafe autonomous action." | "The bounded iterative tool runtime can use real SQL observations to drive subsequent evidence-dependent tool calls and reach a bounded conclusion while preserving explicit proof of the investigation chain." |

### SCENARIO claim semantics

SCENARIO documentation and descriptors **MUST** include:

- bounded real-world system claim
- real problem and failure consequences
- application responsibility (production-capable component)
- PASS / FAIL / adversarial conditions
- excluded claims and limitations
- `domains_exercised` and `mechanisms_exercised` (metadata — may be multiple)

SCENARIO **MUST NOT** require a single `mechanism under proof` field — a Scenario may legitimately exercise multiple mechanisms.

### CONFORMANCE claim semantics

CONFORMANCE documentation and descriptors **MAY** require:

- architecture owner
- mechanism under proof
- bounded mechanism claim
- invariant
- PASS / FAIL
- excluded claims

Each proof documentation artifact must include at minimum:

- exact claim
- user relevance (why the claim matters)
- excluded claims (what is out of scope)

---

## F. Real boundary rules

A proof must exercise the **real boundary** relevant to its claim.

| Claim type | Real boundary |
|------------|---------------|
| Database behavior | Real database when database behavior matters |
| Model provider | Real provider/model when provider behavior matters |
| Filesystem | Real filesystem |
| External integration | Real external boundary when claimed |

Do not use a fake to replace the mechanism being claimed.

**SCENARIO-specific:** canonical Scenario execution **MUST** exercise the same production-capable application path intended for real deployment. Controlled/synthetic provider implementations **MAY** supply data through normal application contracts when the external system itself is not the claim. See [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md) § Scenario Proof — production-capable application contract.

**Deterministic fixtures** are allowed when they provide controlled input — not when they substitute the capability or fake application behavior.

| Allowed | Not allowed |
|---------|-------------|
| Synthetic deterministic logistics dataset in real PostgreSQL | Fake SQL tool returning pre-written answers when proving tool/database investigation |
| Synthetic telemetry through typed provider contract in Scenario about investigation behavior | Fake model output replacing AI behavior when AI behavior is material to the Scenario claim |

---

## G. Mock / fixture policy

Class-specific rules supersede any generic "dependency not under proof" allowance for **SCENARIO canonical application execution**.

| Context | Fake/mock policy |
|---------|------------------|
| **Scenario canonical application path** | **PROHIBITED** for material app/runtime behavior (including FakeLLM, scripted model, `Mock`/`MagicMock`, `testing_support` in application core) |
| **Scenario unit/integration tests** | Allowed |
| **Conformance proof** | Allowed only **outside** claimed boundary |
| **Mechanism under proof** | **MUST NOT** be mocked |
| **Fixtures (controlled data)** | Allowed when they provide deterministic input through normal contracts — not when they decide outcomes or replace AI behavior material to the claim |

If a proof uses a fake at a material claimed boundary, it **cannot** claim that boundary as proved.

**Legacy summary (Conformance and non-canonical tests):** mocks/fakes remain usable outside the claimed mechanism boundary. Conformance **MAY** legitimately use controlled harnesses/test doubles that do not invalidate the mechanism claim. SCENARIO canonical execution has stricter production-capable application requirements — see Authoring Guide § Fake / mock policy (Scenario canonical path).

---

## G2. SCENARIO observability, explainability, and diagnostics

This section applies to **SCENARIO** proofs only. Conformance observability depth may differ by mechanism under proof.

**Application Observability Test (mandatory):**

> If the proof evaluator, evidence packaging, and HTML report are removed, does the application/runtime still produce enough structured execution information to reconstruct its material decisions, actions, observations, challenges, recoveries, diagnostics, and terminal result?

Required answer: **YES**. **NO** means the Scenario is **not acceptable**.

| Owner | Responsibility |
|-------|----------------|
| **Application / platform** | Emit canonical structured execution trace, decision provenance, diagnostics, claim/challenge lifecycle |
| **Proof** | Validate, project, package, render — **MUST NOT** invent post-hoc execution explanations |

**PASS cannot rely solely on final answer or narrative prose.** Acceptance requires traceability of material decisions, actions, evidence, challenges, recoveries, diagnostics, and terminal outcome. Reports **MUST** derive execution claims from machine-readable structured evidence — not from expected proof truth or renderer-invented story.

Machine-readable provenance for accepted executable SCENARIO proofs **MUST** be available via existing contracts where possible (`PlatformProofEvidence` v3 `scenarios[].steps`, evidence graph, runtime trace export). Do not require a parallel Proof-only logging system.

---

## H. Required scenario content

Every Platform Proof documentation artifact must include:

1. Claim  
2. Why it matters  
3. Architecture under proof  
4. Real boundaries  
5. Scenario  
6. Setup  
7. Execution command  
8. Expected flow  
9. PASS invariants  
10. FAIL conditions  
11. Negative scenario  
12. Evidence produced  
13. Limitations  
14. What this proof explicitly does NOT prove  
15. Educational explanation  

**SCENARIO-specific additions (design stage):** Application Survival Test result; Application Observability Test result; observability / explainability / diagnostics contract (material decisions, coverage, redaction, expected machine-readable artifact, report projection plan). See Authoring Guide scaffold § Observability / Explainability / Diagnostics Contract.

---

## I. Falsification requirement

Every **primary** platform proof must contain at least one meaningful **negative or counterexample path**.

Examples:

- unsupported action blocked
- missing evidence produces bounded limitation
- malformed input fails closed
- process restart preserves durability
- contradiction prevents unsupported conclusion

Do not allow a primary proof to be only a polished happy path.

---

## J. PASS / FAIL

**PASS** must be based on explicit machine-checkable invariants where possible.

**SCENARIO-specific:** PASS **MUST NOT** rely solely on final answer text or report narrative. Material autonomous actions **MUST** have provenance in structured production-path observability; model-selected actions **MUST** correlate to runtime execution; failure paths **MUST** include structured diagnostics; operational failure **MUST** be distinguished from epistemic unresolved outcome.

| Avoid | Prefer |
|-------|--------|
| "output looked correct" | exact runtime stop reason |
| | persisted receipt exists |
| | required evidence count |
| | required state transition |
| | forbidden action did not happen |
| | correct durable reconstruction |
| | explicit boundary result |

Semantic LLM expectations may exist, but deterministic platform invariants must remain separately identifiable.

| Result | Meaning |
|--------|---------|
| **PASS** | Claimed capability demonstrated under named proof conditions |
| **FAIL** | Claimed capability not demonstrated under named proof conditions |
| **BLOCKED** | Environment/configuration prevented execution |

Reuse existing `ProofStatus` and runner semantics in `scripts/proof/intergrax_proof_contracts.py` and `scripts/proof/intergrax_proof_runner.py`. Do not redefine unless required.

---

## K. Evidence

Execution evidence must identify:

- `proof_id`
- exact git SHA
- dirty/clean state
- environment/profile where relevant
- result
- duration
- limitations / diagnostics where appropriate

Reuse **`SuiteReceipt`** from `scripts/proof/` — do not invent a competing suite receipt.

Do **not** merge `SuiteReceipt` with runtime/domain `ProofReceipt`. Preserve the current explicit separation documented in `intergrax_proof_contracts.py`.

Every Platform Proof execution must also produce a human-readable **Proof Report**
(self-contained HTML) per
[`PLATFORM_PROOF_REPORT_STANDARD.md`](../docs/project/proofs/PLATFORM_PROOF_REPORT_STANDARD.md)
(PP-REPORT-1). The report presents typed proof evidence; it is not an independent
source of truth.

**SCENARIO reports:** execution narrative in the report **MUST** be derivable from structured evidence (`PlatformProofEvidence`, trace steps, evidence graph) — the renderer **MUST NOT** invent tool calls, decisions, rationales, evidence, or challenges absent from canonical artifacts.

---

## L. Limitations

Every proof **MUST** state: **"What this proof does not prove."**

No implied universalization. A proof against one model, provider, OS, or workload does not imply all providers, models, platforms, or workloads.

---

## M. Versioning

Protocol v1.0 (PP-2) applies **prospectively**. Existing product proofs and historical proof artifacts are **not** migrated merely because this methodology exists.

**Important:** This prospective rule is **not** permission to reclassify LKW as platform proof. LKW remains a product; its proofs remain product proofs.

Historical platform-capability evidence may later be assessed for conformance without moving product proofs into `platform_proofs/`.

---

## N. Relation to public claim governance

| Artifact | Responsibility |
|----------|----------------|
| **PLATFORM_PROOF_PROTOCOL** (this document) | How platform proofs are designed and executed |
| **PUBLIC_PROOF_AND_CLAIMS_MODEL** | What public wording evidence permits |
| **PROOFS.md** | Public proof/evidence dashboard |
| **scripts/proof/** | Generic execution infrastructure |
| **applications/** | Real products and their product proofs |

Update `PROOFS.md` only when accepted public evidence or claim boundaries change — not merely because a platform proof was designed or executed internally.

---

## Execution infrastructure (reuse only)

Platform proofs **must** reuse:

- `scripts/proof/intergrax_proof_manifest.py` — canonical manifest and `proof_id` registry
- `scripts/proof/intergrax_proof_runner.py` — suite runner
- `scripts/proof/intergrax_proof_contracts.py` — profiles, safety classes, `SuiteReceipt`, `ProofStatus`

Do **not** create a second manifest, runner, or receipt contract.

# Intergrax Proof Library

**Status:** Canonical gateway  
**Audience:** Maintainers, architects, proof authors

---

## What is the Proof Library?

The **Intergrax Proof Library** (`platform_proofs/`) holds executable falsification attempts against bounded claims about **reusable Intergrax platform mechanisms** — not product workflow demos and not a substitute for unit or integration tests.

The library has two public classes:

| Class | Role | Entry framing |
|-------|------|---------------|
| **SCENARIO** | **Production-capable autonomous mini application** that solves a concrete real-world problem — plus falsification, evidence, evaluation, and report | Problem-first — primary public layer |
| **CONFORMANCE** | **Mechanism-level executable proof** — CI, regression, contract verification, architecture confidence | Mechanism-first — secondary in public library |

**SCENARIO in one line:** production-capable application component + adversarial proof layer that falsifies and evidences it (the proof layer does **not** substitute for the application).

**CONFORMANCE in one line:** platform mechanism → controlled harness → contract/invariant → evidence.

Normative detail: [Authoring Guide § Scenario Proof — production-capable application contract](PLATFORM_PROOF_AUTHORING_GUIDE.md#scenario-proof--production-capable-application-contract) · [Authoring Guide § Application Survival Test](PLATFORM_PROOF_AUTHORING_GUIDE.md#application-survival-test) · [Protocol § B2](PLATFORM_PROOF_PROTOCOL.md#b2-proof-library-classes) · [Protocol § G Mock/fixture policy](PLATFORM_PROOF_PROTOCOL.md#g-mock--fixture-policy)

Products under `applications/` may **consume** platform mechanisms, but product execution is **not** independent proof of those mechanisms.

```mermaid
flowchart LR
    A[Real problem / mechanism claim] --> B[Implementation]
    B --> C[Tests / integration evidence]
    C --> D[Proof Library<br/>platform_proofs/]
    D --> E[Product consumption<br/>applications/]
    E --> F[Public evidence dashboard<br/>docs/project/proofs/]

    subgraph execution [Canonical execution]
        S[scripts/proof/<br/>discovery · runner · SuiteReceipt]
    end

    D --> S
```

**Normative rule (non-negotiable):**

> A Product Proof may demonstrate that a product successfully consumes platform mechanisms, but product-specific execution **MUST NOT** substitute for an independently owned Platform Proof of the reusable platform capability.

---

## Folder responsibilities

| Path | Owns |
|------|------|
| [`platform_proofs/`](.) | Proof Library — methodology, coverage map, proof packages |
| [`scripts/proof/`](../scripts/proof/) | Canonical **execution infrastructure** — discovery, profiles, runner, `SuiteReceipt` |
| [`docs/project/proofs/`](../docs/project/proofs/) | **Public** proof and evidence dashboard — what wording accepted evidence permits |
| [`applications/`](../applications/) | **Products** and their **product proofs** — not platform domains |

**LKW** (`applications/local_workspace_application/`) is a **product**. LKW proofs remain product-owned. They do not qualify platform domains in the [Platform Proof Map](PLATFORM_PROOF_MAP.md).

---

## Canonical documents

| Document | Role |
|----------|------|
| [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md) | **Canonical practical workflow** for independent Scenario and Conformance proof sessions |
| [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) | How proofs are designed, classified, executed, and evidenced |
| [PLATFORM_PROOF_MAP.md](PLATFORM_PROOF_MAP.md) | Coverage map for canonical domains + feature proofs |

**Related (outside this folder):**

- [Public proof dashboard](../docs/project/proofs/PROOFS.md)
- [Public proof and claims model](../docs/project/maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md)
- [Runtime architecture hub](../docs/project/architecture/intergrax_runtime_architecture.md) — canonical domain topology

---

## Scenario proofs (design stage)

Design-stage Scenario packages live under `platform_proofs/scenarios/<scenario_slug>/`. The first scenario in qualification is [`scenarios/ai_incident_investigation/README.md`](scenarios/ai_incident_investigation/README.md) — not yet accepted for public Proof Library catalog.

Create new design-stage packages with:

```bash
uv run python scripts/proof/create_scenario_proof.py --slug <slug> --title "<title>"
```

See [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md) for the Scenario Quality Gate before implementation.

---

## What this is not

- Not a product directory (`applications/`)
- Not a generic test suite (`tests/`)
- Not a public evidence archive (`docs/project/proofs/`)
- Not an alternative proof runner (use `scripts/proof/`)

# Intergrax Platform Proof Protocol

**Status:** Canonical  
**Version:** 1.0 (PP-2)  
**Audience:** Maintainers, architects, proof authors  
**Scope:** Reusable Intergrax platform mechanism proofs under `platform_proofs/`

---

## Core mindset

> **A Platform Proof is an executable falsification attempt against a specific claim about a reusable Intergrax platform mechanism.**

It is **not** a happy-path demo. The author must try to prove the claim false under named conditions. A PASS backed by explicit invariants and at least one meaningful negative path is valid. A FAIL with clear evidence is also valid.

This protocol governs **DOMAIN_PROOF** and **FEATURE_PROOF** only. **PRODUCT_PROOF** remains with owning products under `applications/`.

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
| **Platform proof** | Whether a **reusable platform capability** exhibits the claimed behavior across realistic boundaries |
| **Product proof** | A product workflow end-to-end |
| **Real-user validation** | User outcomes in realistic use |
| **Commercial validation** | Market or business value |

Platform proofs sit between integration tests and product proofs. They prove the **platform mechanism**, not a single product's business logic.

**Examples:**

- TOOLS iterative investigation — platform proof (`platform_proofs/`)
- LKW Product Quick Start — product proof (`applications/local_workspace_application/`)
- LKW exercising tool runtime during a product workflow — product consumption, **not** independent TOOLS domain proof

**LKW is a product.** LKW must not appear as a platform domain, in `PLATFORM_PROOF_MAP` as a layer, or inside `platform_proofs/`.

---

## B. Proof classification

| Class | Owner | Governed by this protocol |
|-------|-------|---------------------------|
| **DOMAIN_PROOF** | Platform domain (`docs/project/architecture/<DOMAIN>.md`) | Yes |
| **FEATURE_PROOF** | Cross-layer feature (`docs/project/capabilities/`) | Yes |
| **PRODUCT_PROOF** | Product (`applications/<product>/`) | **No** — product-owned |

`platform_proofs/` owns DOMAIN_PROOF and FEATURE_PROOF artifacts only.

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

**Recommended naming:** `<DOMAIN>-<CAPABILITY>` — uppercase, hyphenated, consistent with existing manifest style.

**Reference example:** `TOOLS-ITERATIVE-SQL-INVESTIGATION`

Canonical executable identity and execution metadata remain owned by:

```text
scripts/proof/intergrax_proof_manifest.py
```

Do **not** duplicate manifest metadata in arbitrary local config files. Register proofs in the canonical manifest; document scenario content under `platform_proofs/`.

---

## E. Required proof claim

Every proof must state **one bounded falsifiable claim**.

| Bad | Good |
|-----|------|
| "Tools works." | "The bounded iterative tool runtime can use real SQL observations to drive subsequent evidence-dependent tool calls and reach a bounded conclusion while preserving explicit proof of the investigation chain." |

Each proof documentation artifact must include:

- exact claim
- user relevance (why the claim matters)
- architecture owner (domain or feature)
- mechanism under proof
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

**Deterministic fixtures** are allowed when they provide controlled input — not when they substitute the capability.

| Allowed | Not allowed |
|---------|-------------|
| Synthetic deterministic logistics dataset in real PostgreSQL | Fake SQL tool returning pre-written answers when proving tool/database investigation |

---

## G. Mock / fixture policy

| Technique | Policy |
|-----------|--------|
| **Fixtures** | Allowed for controlled deterministic data |
| **Mocks / fakes** | Allowed for dependencies that are **not** the mechanism under proof |
| **Mechanism under proof** | **MUST NOT** be mocked |

If a proof uses a fake at a material claimed boundary, it **cannot** claim that boundary as proved.

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

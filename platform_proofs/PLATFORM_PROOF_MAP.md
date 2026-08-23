# Intergrax Platform Proof Map

**Status:** Canonical coverage map  
**Scope:** Proof-development coverage for 28 canonical platform domains + cross-layer feature proofs

**Important:** Coverage labels describe **proof-design and execution readiness** — not public claim status, not `ProofStatus`, and not production maturity.

| Coverage | Meaning |
|----------|---------|
| **NO_PROOF** | No canonical platform proof designed |
| **DESIGNED** | Claim/scenario defined; executable proof absent |
| **EXECUTABLE** | Registered/runnable proof exists in `scripts/proof/intergrax_proof_manifest.py` |
| **QUALIFIED** | Successfully executed with accepted evidence under a named bounded environment |

**Ownership rule:** Product proofs under `applications/` (including **LKW**) do **not** qualify platform domains here. Product execution ≠ platform-layer proof.

Canonical domain topology: [intergrax_runtime_architecture.md](../docs/project/architecture/intergrax_runtime_architecture.md).

---

## Domain coverage (28 canonical domains)

| Domain | Primary falsifiable claim | Primary proof | Coverage | Real boundary | Notes |
|--------|---------------------------|---------------|----------|---------------|-------|
| `PLATFORM_FOUNDATION` | — | — | NO_PROOF | — | Platform-wide foundation; proof scoped per mechanism |
| `UNIFIED_EXECUTION_RUNTIME` | — | — | NO_PROOF | — | |
| `ORCHESTRATION` | — | — | NO_PROOF | — | LKW orchestration paths are product evidence only |
| `NEXUS_EXECUTION_FLOW` | — | — | NO_PROOF | — | |
| `REASONING_AND_COGNITION` | — | — | NO_PROOF | — | |
| `AGENT_CONTRACTS_AND_ASSEMBLY` | — | — | NO_PROOF | — | |
| `LLM_ADAPTERS` | — | — | NO_PROOF | — | `LKW-MODEL-RUNTIME` is product-scoped, not LLM-adapters domain proof |
| `TOOLS` | Bounded iterative tool runtime uses real observations to drive evidence-dependent follow-on calls and terminate with an explicit investigation chain under configured limits | — | **NO_PROOF** | — | Legacy TOOLS-first conformance proof removed; scenario-first Proof Library bootstrap in progress |
| `CODE_CRAFT` | — | — | NO_PROOF | — | |
| `SKILLS` | — | — | NO_PROOF | — | |
| `INTEGRATIONS` | — | — | NO_PROOF | — | |
| `RAG` | — | — | NO_PROOF | — | |
| `MEMORY` | — | — | NO_PROOF | — | |
| `CONTEXT_ENGINEERING` | — | — | NO_PROOF | — | |
| `MODALITY` | — | — | NO_PROOF | — | |
| `OBSERVABILITY` | — | — | NO_PROOF | — | |
| `RELIABILITY_FAILURE_AND_HITL` | — | — | NO_PROOF | — | |
| `CRITIC_VERIFICATION` | — | — | NO_PROOF | — | |
| `ADAPTIVE_HARNESS_INTELLIGENCE` | — | — | NO_PROOF | — | |
| `ELASTIC_CAPACITY_AND_SCALING` | — | — | NO_PROOF | — | |
| `EXPERIMENTATION_AND_DEVELOPER_EXPERIENCE` | — | — | NO_PROOF | — | |
| `TIER3_APPLICATION_ENVIRONMENT` | — | — | NO_PROOF | — | LKW is a reference product, not this domain's platform proof |
| `APPLICATION_HOSTING` | — | — | NO_PROOF | — | |
| `UNIFIED_CONTEXT_LIFECYCLE` | — | — | NO_PROOF | — | |
| `GOVERNED_EXECUTION` | — | — | NO_PROOF | — | LKW governed-evidence paths are product proofs |
| `AGENT_DISTRIBUTION` | — | — | NO_PROOF | — | |
| `PLATFORM_PLUGINS` | — | — | NO_PROOF | — | |
| `PROOF_RECEIPTS` | — | — | NO_PROOF | — | Receipt contracts exist; dedicated platform proof not yet designed |

---

## Feature proof coverage (cross-layer)

Feature proofs are **not** rows in the 28-domain table. Feature ownership remains distinct from domain ownership.

| Feature | Primary falsifiable claim | Primary proof | Coverage | Real boundary | Notes |
|---------|---------------------------|---------------|----------|---------------|-------|
| `TOKEN_OPTIMIZATION` | Deterministic token optimization pipeline executes with bounded offline smoke invariants | `RUNTIME-TOKEN-OPTIMIZATION-OFFLINE` | EXECUTABLE | Offline deterministic fixture path via canonical runner | Pre-PP-2 manifest entry; bounded offline smoke — not live vLLM/provider qualification |
| `LANGCHAIN_INDEPENDENCE` | — | — | NO_PROOF | — | Independence mechanisms may exist in code; no canonical feature proof designed |

---

## Explicit non-entries

The following are **not** platform domains and must **not** appear in the domain table:

| Name | Classification | Proof location |
|------|----------------|----------------|
| **LKW** (`local_workspace_application`) | **Product** | `applications/local_workspace_application/docs/proof/` |
| Any other `applications/<product>/` | **Product** | Product-owned proof docs + manifest entries |

---

## Maintenance

When adding or advancing a proof:

1. Update this map (coverage column)
2. Ensure scenario doc meets [PLATFORM_PROOF_PROTOCOL.md](PLATFORM_PROOF_PROTOCOL.md) § H
3. Register executable proofs only in `scripts/proof/intergrax_proof_manifest.py`
4. Update [PROOFS.md](../docs/project/proofs/PROOFS.md) only when public claim boundaries change

See [PLATFORM_PROOF_AUTHORING_GUIDE.md](PLATFORM_PROOF_AUTHORING_GUIDE.md).

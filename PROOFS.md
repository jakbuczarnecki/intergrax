<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Intergrax Proofs

**Evidence before promises.**

Intergrax separates **implemented mechanisms**, **bounded verification**, **partial product capability**, **planned work**, and **claims that are not currently supported**. This page is the public proof dashboard — no internal task IDs required.

> [!NOTE]
> Intergrax is **source-available** and under **active R&D**. Technical proof does not imply finished SaaS, production readiness, real-user validation, or commercial validation.

> [!NOTE]
> This page reports public claims and accepted evidence.
> It does not mirror implementation roadmaps.
> For current tasks, dependencies and next implementation steps,
> follow the detailed roadmap linked for each product or capability.

## Status legend

| Label | Meaning |
|-------|---------|
| ✅ **IMPLEMENTED** | A concrete mechanism exists with bounded tests or accepted implementation evidence; no live, product, or production validation is implied. |
| 🧪 **BOUNDED PROOF** | Proof was executed in a named environment with available evidence and stated limitations; wording must not generalize beyond that scope. |
| 🟡 **PARTIAL** | Some slices are implemented; end-to-end capability, dependency, or validation gate remains open. |
| 🗓️ **PLANNED** | Architecture or scheduling exists; runtime behavior, proof, or integration is not yet available. |
| ⛔ **NOT CLAIMABLE** | No evidence supports the claim, or guardrails explicitly block public wording. |

Text labels are authoritative; symbols are visual support only.

## Proof landscape

```mermaid
flowchart LR
    LR[LKW implementation roadmap] --> LI[LKW implementation]
    LI --> LE[Accepted LKW evidence]

    TR[Token Optimization roadmap] --> TI[Token implementation]
    TI --> TE[Accepted Token evidence]

    LE --> P[PROOFS.md public claims]
    TE --> P

    P --> O[Public overview documents]
```

Roadmaps own changing implementation progress.
Accepted evidence determines public claims.
Overview documents summarize those claims
without copying either roadmap.

## At a glance

| Proof path | Public classification | Current public status | What it demonstrates | Verify | Detailed roadmap |
|------------|------------------------|----------------------|----------------------|--------|------------------|
| **LKW** | Primary product proof | 🟡 **PARTIAL** (Backend Product Alpha / MVP) | Bounded end-to-end application and platform behavior, indexed knowledge, background ingest, hosting, observability | [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) | [LKW implementation plan](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |
| **Token Optimization** | Featured platform-capability proof | 🟡 **PARTIAL** | Deterministic optimization pipeline, cache-aware execution, bounded vLLM prefix-cache proof | [Token Optimization guide](docs/features/token_optimization/README.md) | [Token Optimization plan](docs/features/plan/TOKEN_OPTIMIZATION.md) |
| **Shared platform foundations** | Supporting evidence | ✅ **IMPLEMENTED** (bounded) | Retrieval and grounding, observability, persisted execution evidence, application hosting contracts exercised by LKW | [Public documentation map](docs/PUBLIC_DOCUMENTATION_MAP.md) | — |

`Verify` answers **what has been demonstrated?**
`Detailed roadmap` answers **what is being implemented and what comes next?**

---

## LKW — Primary product proof

**Role:** Primary product proof · **Product status:** Backend Product Alpha / MVP

| Capability | Status | What it demonstrates | Limitation |
|------------|--------|----------------------|------------|
| Core application and platform proof | 🧪 **BOUNDED PROOF** | Real application startup, observability, ingest, hosting, persisted execution evidence | Bounded to documented certification profiles; not full live platform proof |
| Web URL knowledge intake | 🧪 **BOUNDED PROOF** | End-to-end WEB_URL intake, indexing, grounded Ask over indexed content | Not live external-website certification |
| Ollama / vLLM model runtime portability | 🧪 **BOUNDED PROOF** | Same workspace workflows on Ollama and vLLM without reindexing | Not complete product parity across all features |

### Not established by the accepted public proof

- Complete Hybrid Ask product behavior is not established by the accepted public proof.
- A complete Slack connected-knowledge product proof is not established.
- A Google Workspace LKW product proof is not established.
- The complete multi-source live platform proof is not established.
- Real-user validation is not established.
- Commercial validation is not established.

Detailed implementation roadmap:
[applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md)

Accepted technical proof:
[LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md)

---

## Token Optimization — Featured platform-capability proof

**Role:** Featured platform-capability proof

| Capability | Status | What it demonstrates | Limitation |
|------------|--------|----------------------|------------|
| Deterministic optimization pipeline | ✅ **IMPLEMENTED** | Layer registry, pipeline runner, built-in catalog, plugin contract | Not globally auto-enabled |
| Approved-configuration LLM routing | ✅ **IMPLEMENTED** | Policy-governed router with approved catalog only | Router output is advisory until application applies it |
| Protected-region validation | ✅ **IMPLEMENTED** | Central validation after content-changing layers | Does not guarantee semantic equivalence for all lossy paths |
| Receipts and fallback | ✅ **IMPLEMENTED** | Attribution, savings metadata, safe fallback on validation failure | Receipts exclude raw content |
| Cache-stable prompt assembly | ✅ **IMPLEMENTED** | Stable prefix, dynamic tail, append-only rules | Does not prove provider cache behavior alone |
| Exact-send integrity | ✅ **IMPLEMENTED** | Message and tool-schema integrity before adapter send | Provider-specific cache behavior varies |
| Cache-aware execution gate | ✅ **IMPLEMENTED** | Only `RUN` executes pipeline; conflicting evidence rejected | Does not perform in-cache compaction |
| Durable in-cache compaction mechanism | ✅ **IMPLEMENTED (BOUNDED)** | Durable SQLite repository, validation and CAS activation exist | Live provider proof, rollback execution and production rollout remain incomplete |
| Bounded vLLM prefix-cache proof | 🧪 **BOUNDED PROOF** | Cold/warm/changed-prefix reuse in documented vLLM environment | Named version, model, and workload only |
| Universal token reduction | ⛔ **NOT CLAIMABLE** | — | No universal savings evidence |
| Production-proven savings | ⛔ **NOT CLAIMABLE** | — | Required proof and promotion gates are incomplete. |

### Not established by the accepted public proof

- Durable in-cache compaction is implemented as a bounded mechanism; an accepted public live proof of complete provider-wide behavior is not established.
- Cross-provider behavior is not established by the named vLLM proof.
- Provider-independent cache behavior is not established.
- Universal token reduction is not established.
- Production-proven savings are not established.

Detailed implementation roadmap:
[docs/features/plan/TOKEN_OPTIMIZATION.md](docs/features/plan/TOKEN_OPTIMIZATION.md)

Verification and claim routes:
[Token Optimization guide](docs/features/token_optimization/README.md) · [Claim guardrails](docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md)

---

## Claim-to-proof lifecycle

```mermaid
flowchart LR
    M[Mechanism exists] --> T[Bounded tests pass]
    T --> E[Bounded evidence recorded]
    E --> L[Limitations explicit]
    L --> W[Approved public wording]
    T -->|missing evidence| NC[NOT CLAIMABLE]
    E -->|excessive generalization| NC
    L -->|limitations omitted| NC
```

**Remember:** implemented code ≠ live proof ≠ product validation ≠ commercial validation ≠ production readiness.

---

## What Intergrax does not currently claim

> [!WARNING]
> The following are **not** current public claims:
>
> - finished SaaS product
> - completed commercial validation
> - universal production readiness
> - universal token or cost savings
> - completed Hybrid Ask
> - complete vendor integration catalog
> - production-proven durable compaction rollout
> - real-user validation at scale

Real-user and commercial validation remain **incomplete**.

---

## Verification paths

| Document | Purpose |
|----------|---------|
| [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) | Guided LKW reviewer proof path |
| [LKW Implementation Plan](applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) | Detailed LKW implementation roadmap |
| [Token Optimization guide](docs/features/token_optimization/README.md) | Engine overview and proof catalog |
| [Token Optimization plan](docs/features/plan/TOKEN_OPTIMIZATION.md) | Detailed Token Optimization implementation roadmap |
| [Token Optimization claim guardrails](docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md) | Safe public wording boundaries |
| [Public documentation map](docs/PUBLIC_DOCUMENTATION_MAP.md) | Reader-intent navigation |
| [Technical documentation map](docs/DOCUMENTATION_MAP.md) | Deep technical review entry |

Maintainer status and wording rules: [Public Proof and Claims Model](docs/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md)

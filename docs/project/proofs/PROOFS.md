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
| **LKW** | Primary product proof | 🟡 **PARTIAL** (Backend Product Alpha / MVP) | Bounded end-to-end application and platform behavior, indexed knowledge, background ingest, hosting, observability | [Product Quick Start](../../../applications/local_workspace_application/docs/product/QUICKSTART.md) · [LKW Platform Proof](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) · [Core Platform Proof](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md#core-platform-proof) | [LKW implementation plan](../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) |
| **Token Optimization** | Featured platform-capability proof | 🟡 **PARTIAL** | Deterministic optimization pipeline, cache-aware execution, bounded offline smoke proof | [Token Optimization guide](../capabilities/token_optimization/README.md) | [Token Optimization plan](../capabilities/plan/TOKEN_OPTIMIZATION.md) |

`Verify` answers **what has been demonstrated?**
`Detailed roadmap` answers **what is being implemented and what comes next?**

Supporting-foundation evidence is capability-specific. The LKW proof exercises
only the reusable mechanisms named in its accepted evidence; other reusable
foundations do not inherit a blanket platform-wide **IMPLEMENTED** status.

---

## LKW — Primary product proof

**Role:** Primary product proof · **Product status:** Backend Product Alpha / MVP

**Primary executable path:** [Product Quick Start](../../../applications/local_workspace_application/docs/product/QUICKSTART.md) —
indexed **Ask V1** product evaluation with one command per supported OS, managed
sample upload, indexing, grounded Ask, source citation, and persisted Ask-run
verification.

**Separate technical proof:** the bounded indexed Hybrid Ask branch is
demonstrated through the production Hybrid Ask `indexed_only` path. Some
indexed-path proofs validate retrieval and boundary behavior even when answer
assembly returns insufficient evidence. This is not the Product Quick Start path.

The Trusted Ask durability proof separately verifies completed grounded Ask
outcomes before and after restart. Core Platform Proof separately verifies bounded
infrastructure behavior. Mixed indexed + authorized live Hybrid Ask remains
incomplete, and complete live-provider access remains incomplete.

| Capability | Status | What it demonstrates | Limitation |
|------------|--------|----------------------|------------|
| **Product Quick Start / indexed Ask V1** | 🧪 **BOUNDED PROOF** · supported product-evaluation path | One-command local execution of the canonical LKW application path: managed sample upload → indexing → grounded indexed Ask → source citation → persisted Ask-run verification — [run it](../../../applications/local_workspace_application/docs/product/QUICKSTART.md) | Indexed knowledge only; script-driven; no polished end-user UI; not Hybrid Ask certification; not production readiness; no real-user or commercial validation |
| **Indexed Hybrid Ask** | 🧪 **BOUNDED PROOF** | Bounded indexed branch through production Hybrid Ask `indexed_only`; some proofs validate retrieval and boundary behavior even when answer assembly returns insufficient evidence | Not Product Quick Start; not mixed indexed + authorized-live Hybrid Ask; not production readiness or commercial validation |
| **Trusted Ask / durable indexed workspace Ask** | 🧪 **BOUNDED PROOF** | Current runtime image through Ollama generation and embeddings, workspace source sync, Qdrant indexed retrieval, completed first Ask with evidence, non-destructive restart, completed second Ask without resync/reindex, and unchanged persisted first Ask run | Indexed knowledge only; bounded local runtime; no mixed indexed + authorized-live Hybrid Ask; no universal provider, production, or commercial claims |
| **Core Platform Proof** | 🧪 **BOUNDED PROOF** | Real application startup/readiness, durable knowledge and execution, background processing, persisted reviewable evidence, hosting/recovery, and watched-folder indexing (Elasticsearch/Kibana observability, Sentry problem signals, MongoDB ProofReceipts) | Bounded to documented certification profiles; not production readiness, commercial validation, or all-provider certification |

**Product Quick Start proofs:** `LKW-PRODUCT-QUICKSTART-WINDOWS`, `LKW-PRODUCT-QUICKSTART-LINUX`, `LKW-PRODUCT-QUICKSTART-MACOS`

**Indexed Hybrid Ask proof:** `LKW-HYBRID-ASK-INDEXED`

**Trusted Ask proof:** `LKW-ASK-WORKSPACE-LIVE` — [authoritative live proof section](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md#trusted-ask-workspace-mvp-2)

**Core platform proofs:** `LKW-CORE-PLATFORM-WINDOWS`, `LKW-CORE-PLATFORM-LINUX`, `LKW-CORE-PLATFORM-MACOS`, `LKW-BACKGROUND-TASK`, `LKW-HOSTING`, `LKW-FILE-WATCHER`

| Web URL knowledge intake | 🧪 **BOUNDED PROOF** | Real WEB_URL capture and RAG indexing into the exact tenant/workspace Qdrant scope, verified indexed retrieval, and execution through Hybrid Ask `indexed_only`; a suite **PASS** always requires those indexed-path steps — grounded answer and indexed citation/evidence are verified only when the run completes with `status=completed`; `status=insufficient_evidence` may still **PASS** after successful indexed retrieval without proving completed answer assembly | Controlled `example.com` origin only; not arbitrary external-site certification; mixed indexed + authorized live Hybrid Ask remains incomplete; complete live-provider access remains incomplete |
| Ollama / vLLM model runtime portability | 🧪 **BOUNDED PROOF** | Same workspace workflows on Ollama and vLLM without reindexing | Not complete product parity across all features |

**Proof:** `LKW-WEB-URL-INDEXED-ASK`

**Proof:** `LKW-MODEL-RUNTIME`

### Not established by the accepted public proof

- Mixed indexed + authorized live Hybrid Ask remains incomplete.
- Complete live-provider access remains incomplete.
- Complete multi-source live capability remains incomplete.
- Real-user validation is not established.
- Commercial validation is not established.

Detailed implementation roadmap:
[applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md](../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md)

Accepted technical proof:
[LKW Platform Proof](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md)

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
| Durable in-cache compaction mechanism | ✅ **IMPLEMENTED (BOUNDED)** | Durable SQLite repository, validation and CAS activation exist | Live provider proof, rollback execution and production rollout remain incomplete; numeric savings are not claimed |
| Bounded offline smoke proof | 🧪 **BOUNDED PROOF** | Offline_smoke universal proof path via local adapter; exercises deterministic pipeline contracts | Does not execute live vLLM prefix-cache verification |
| vLLM prefix-cache reuse | 🧪 **BOUNDED PROOF** (manual) | Cold/warm/changed-prefix reuse when operator runs the documented manual live path | Named version, model, and workload only; no accepted public manifest proof exists for this path today |

**Proof:** `RUNTIME-TOKEN-OPTIMIZATION-OFFLINE`

| Universal token reduction | ⛔ **NOT CLAIMABLE** | — | No universal savings evidence |
| Production-proven savings | ⛔ **NOT CLAIMABLE** | — | Required proof and promotion gates are incomplete. |

### Not established by the accepted public proof

- Durable in-cache compaction is implemented as a bounded mechanism; an accepted public live proof of complete provider-wide behavior is not established.
- Cross-provider behavior is not established by the named vLLM proof.
- Provider-independent cache behavior is not established.
- Universal token reduction is not established.
- Production-proven savings are not established.

Detailed implementation roadmap:
[docs/project/capabilities/plan/TOKEN_OPTIMIZATION.md](../capabilities/plan/TOKEN_OPTIMIZATION.md)

Verification and claim routes:
[Token Optimization guide](../capabilities/token_optimization/README.md) · [Claim guardrails](../capabilities/TOKEN_OPTIMIZATION_CLAIMS.md)

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
> - completed Hybrid Ask combining indexed and authorized live evidence
> - complete vendor integration catalog
> - production-proven durable compaction rollout
> - real-user validation at scale

Real-user and commercial validation remain **incomplete**.

---

## Verification paths

| Document | Purpose |
|----------|---------|
| [LKW Platform Proof](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) | Guided LKW reviewer proof path |
| [Core Platform Proof](../../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md#core-platform-proof) | Bounded platform-level verification (startup, sentry, elasticsearch, persistence, background task, application hosting, file watcher) |
| [LKW Implementation Plan](../../../applications/local_workspace_application/docs/IMPLEMENTATION_PLAN.md) | Detailed LKW implementation roadmap |
| [Token Optimization guide](../capabilities/token_optimization/README.md) | Engine overview and proof catalog |
| [Token Optimization plan](../capabilities/plan/TOKEN_OPTIMIZATION.md) | Detailed Token Optimization implementation roadmap |
| [Token Optimization claim guardrails](../capabilities/TOKEN_OPTIMIZATION_CLAIMS.md) | Safe public wording boundaries |
| [Public documentation map](../community/PUBLIC_DOCUMENTATION_MAP.md) | Reader-intent navigation |
| [Technical documentation map](../technical/DOCUMENTATION_MAP.md) | Deep technical review entry |

Maintainer status and wording rules: [Public Proof and Claims Model](../maintainers/public-adoption/PUBLIC_PROOF_AND_CLAIMS_MODEL.md)

---

## Executable proof suite

Public documentation references `proof_id` values only. Canonical membership,
commands, profiles, and environment requirements live in
`scripts/proof/intergrax_proof_manifest.py`.

Run the repository-wide suite from the repository root:

```bash
uv run python scripts/proof/run-intergrax-proof-suite.py --profile quick
uv run python scripts/proof/run-intergrax-proof-suite.py --profile full
uv run python scripts/proof/run-intergrax-proof-suite.py --profile live
```

Validate explicit public proof references:

```bash
uv run python scripts/proof/check-public-proof-references.py
```

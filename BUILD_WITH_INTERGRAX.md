<!--
© Artur Czarnecki. All rights reserved.
Intergrax is source-available under the Intergrax Evaluation and Collaboration License 1.0.
See LICENSE for permitted evaluation, collaboration, and contribution use.
-->

# Build and Evaluate with Intergrax

Choose the right path to evaluate Intergrax, inspect its proof, or begin building a specialized application.

> [!WARNING]
> **Evaluation does not grant production permission, commercial permission, hosting, or redistribution permission.** [LICENSE](LICENSE) is authoritative.

> [!NOTE]
> Intergrax is under **active R&D**. Proof paths validate **bounded mechanisms**, not universal production readiness.

---

## At a glance

| Question | Answer |
| -------- | ------ |
| **Current stage** | Source-available, active R&D, bounded proof paths |
| **Best first step** | [README Quick start](README.md#quick-start) |
| **Primary product proof** | LKW |
| **Featured platform capability** | Token Optimization |
| **Permission boundary** | Production/commercial use requires written permission |

---

## Choose your path

| Goal | Recommended path | What you will learn |
| ---- | ---------------- | ------------------- |
| First technical contact | [README Quick start](README.md#quick-start) | Install, lab host, one bounded execution, trace inspection |
| See the primary product proof | [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md) | Bounded application and platform behavior via LKW |
| Explore Token Optimization | [Token Optimization guide](docs/features/token_optimization/README.md) | Deterministic mechanisms and bounded vLLM proof scope |
| Inspect current evidence | [PROOFS.md](PROOFS.md) | Public proof status and claim boundaries |
| Build a specialized application | [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md) and [application usage docs](applications/USAGE.md) | How to compose product workflow on shared foundations |
| Perform deep technical review | [Technical Documentation Map](docs/DOCUMENTATION_MAP.md) | Architecture canon, plans, and implementation navigation |

```mermaid
flowchart TD
    S[What do you want to evaluate?]
    S -->|First contact| Q[README Quick start]
    S -->|Real product workflow| L[LKW Platform Proof]
    S -->|Platform capability| T[Token Optimization]
    S -->|Build an application| B[Agent Creation Guide]
    S -->|Deep review| D[Technical Documentation Map]
```

---

## Path 1 — First technical contact

Start at [README Quick start](README.md#quick-start).

You verify:

- installation and repository health;
- lab host startup;
- one bounded execution;
- trace inspection.

The README owns the command sequence. This document routes you there without duplicating it.

---

## Path 2 — LKW product proof

**Primary product proof:** [LKW Platform Proof](docs/public-adoption/LKW_PLATFORM_PROOF.md)

This path is more involved than quick start. It validates bounded application and platform behavior through Local Knowledge Workspace — ingest, indexed knowledge, hosting, observability, and ProofReceipt — in a guided reviewer flow.

It does **not** prove finished SaaS, completed Hybrid Ask, or commercial readiness. LKW remains **PARTIAL** at **Backend Product Alpha / MVP**.

---

## Path 3 — Token Optimization

**Featured platform-capability proof:** [Token Optimization guide](docs/features/token_optimization/README.md)

Covers implemented deterministic mechanisms, cache-aware execution, protected-region validation, receipts, and a **bounded vLLM** proof path. Claim guardrails: [TOKEN_OPTIMIZATION_CLAIMS.md](docs/public-adoption/TOKEN_OPTIMIZATION_CLAIMS.md).

No universal or production-proven savings claim is made. Status remains **PARTIAL**.

---

## Path 4 — Build a specialized application

| Resource | Role |
| -------- | ---- |
| [Agent Creation Guide](docs/guides/AGENT_CREATION_GUIDE.md) | Domain agent and harness integration |
| [applications/USAGE.md](applications/USAGE.md) | Application-layer usage patterns |
| [ARCHITECTURE_OVERVIEW.md](ARCHITECTURE_OVERVIEW.md) | Responsibility boundaries before you build |

**Build sequence:**

```text
define the product workflow
→ choose reusable platform capabilities
→ configure policy, knowledge and tools
→ implement domain decisions
→ run bounded tests and proof paths
→ record limitations
```

This sequence describes bounded evaluation and building — not unrestricted production deployment.

---

## What evidence should you capture?

When you run a proof or evaluation path, record:

- exact commit;
- environment;
- model or provider;
- configuration;
- proof path;
- observed result;
- limitation;
- failing or skipped step.

Full public proof dashboard: [PROOFS.md](PROOFS.md).

---

## Evaluation and permission boundaries

- Local **non-production evaluation** is permitted subject to [LICENSE](LICENSE).
- Collaboration routes are in [COLLABORATION.md](COLLABORATION.md).
- **Production use**, **commercial use**, hosting, and redistribution require **explicit written permission**.
- The license is authoritative; this section does not reproduce legal clauses.

---

## Detailed evaluation material

| Document | Role |
| -------- | ---- |
| [EVALUATION_GUIDE.md](EVALUATION_GUIDE.md) | Detailed bounded execution companion — timed evaluation passes |
| [README Quick start](README.md#quick-start) | First technical contact commands |
| [PROOFS.md](PROOFS.md) | Proof status and verification paths |
| [Public Documentation Map](docs/PUBLIC_DOCUMENTATION_MAP.md) | Reader-intent routing across public docs |

**BUILD_WITH_INTERGRAX.md** owns public route selection. **EVALUATION_GUIDE.md** remains the detailed bounded execution companion for step-by-step evaluation passes.

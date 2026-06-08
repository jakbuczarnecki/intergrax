# Intergrax Harness — Architecture Decision Records

**Domain:** Tier-0 platform + Tier-1 Nexus (`intergrax/`, `intergrax/runtime/`)

Canonical architecture: [`../intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
Implementation tracker: [`../intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)

---

## When to write an ADR

Create or update an ADR for **significant** Harness decisions, including:

- Nexus execution semantics, orchestration contracts, lifecycle, delegation
- Tool / skill / integration layer boundaries and catalog contracts
- LLM adapter envelopes, RAG retrieval policy, memory models
- Policy, HITL, observability, and cross-cutting platform behavior
- New universal Tier-0 mechanisms or changes that affect multiple agents

**Not required:** typo fixes, test-only changes, agent-specific business logic (use agent ADRs),
or product-host wiring that does not change platform contracts.

If no ADR is needed, record **"no ADR needed"** with rationale in the PR or plan row.

## Naming

```text
ADR-{AREA}-{NNN}.md
```

Examples: `ADR-FLOW-001`, `ADR-LLM-001`, `ADR-ADAPT-001`.

## Process

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential id for your area tag.
2. Fill **Context**, **Decision**, **Consequences**, and **Compliance**.
3. Link from canon (`intergrax_runtime_architecture.md`) and/or `intergrax_runtime_architecture.md`.
4. Set **Status** to `Accepted` when implemented; `Superseded` when replaced.

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-FLOW-001](ADR-FLOW-001.md) | Declarative delegation (`DELEGATES_TO`) expansion | Accepted · implemented |
| [ADR-FLOW-002](ADR-FLOW-002.md) | Reserved lifecycle states | Accepted |
| [ADR-FLOW-003](ADR-FLOW-003.md) | `MODIFY_PLAN` decision semantics | Accepted |
| [ADR-ADAPT-001](ADR-ADAPT-001.md) | Adaptive Harness Intelligence over classical RL | Accepted |
| [ADR-LLM-001](ADR-LLM-001.md) | Typed LLM adapter response envelope | Accepted |
| [ADR-CRITIC-001](ADR-CRITIC-001.md) | Critic & Verification Layer — tier-separated PEV verify stack | Accepted |
| [ADR-OBS-001](ADR-OBS-001.md) | Harness Observability Spine — unified bus for all tiers | Accepted |
| [ADR-MEM-001](ADR-MEM-001.md) | Context Compiler — global budget allocator and degradation ladder | Accepted |
| [ADR-SCALE-001](ADR-SCALE-001.md) | Harness Elastic Capacity Plane — complement K8s HPA | Accepted |

---

*Scaffold baseline: 2026-06-07*

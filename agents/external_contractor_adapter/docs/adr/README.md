# external_contractor_adapter agent — Architecture Decision Records

**Domain:** Tier-2 adapter agent (`agents/external_contractor_adapter`)

Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)  
Implementation tracker: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)  
Platform ADRs: [`../../../../docs/project/technical/adr/README.md`](../../../../docs/project/technical/adr/README.md)

---

## When to write an ADR

Create an ADR for agent-level decisions that affect **domain behavior, contracts, or integration choices**, for example:

- Capability model, I/O schemas, or lifecycle mapping structure
- Cognitive pattern changes away from scaffold reflex
- External integration consumption choices
- Idempotency / correlation rules specific to this adapter

**Not required:** harness platform changes (use `docs/project/technical/adr`), Tier-3 host wiring (use application ADRs),
or trivial refactors with no behavioral impact.

## Naming

```text
ADR-EXTERNAL_CONTRACTOR_ADAPTER-{NNN}.md
```

## Process

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential number.
2. Link from [`ARCHITECTURE.md`](../ARCHITECTURE.md) when the decision affects runtime layout.
3. Track implementation in [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-EXTERNAL_CONTRACTOR_ADAPTER-001](ADR-EXTERNAL_CONTRACTOR_ADAPTER-001.md) | Tier-2 domain adapter (not orchestrator) | Accepted |
| [ADR-EXTERNAL_CONTRACTOR_ADAPTER-002](ADR-EXTERNAL_CONTRACTOR_ADAPTER-002.md) | Mapping ownership and fake-provider proof (GEC-3) | Accepted |

Platform ADRs (composition owned by platform — no agent-local ADR):

- [`ADR-GOVERNED-CONTINUATION-001`](../../../../docs/project/technical/adr/entries/2026-07-20/ADR-GOVERNED-CONTINUATION-001.md)
- [`ADR-POLICY-SIDE-EFFECT-001`](../../../../docs/project/technical/adr/entries/2026-07-20/ADR-POLICY-SIDE-EFFECT-001.md)
- [`ADR-GOVERNED-PROOF-001`](../../../../docs/project/technical/adr/entries/2026-07-20/ADR-GOVERNED-PROOF-001.md)

**Consolidation:** [`docs/project/technical/platform/governed_external_execution.md`](../../../../docs/project/technical/platform/governed_external_execution.md)

---

*GEC-6 baseline: 2026-07-20*

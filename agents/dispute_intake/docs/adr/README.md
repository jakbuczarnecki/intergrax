# dispute_intake agent — Architecture Decision Records

**Domain:** Tier-2 business agent (`agents/dispute_intake/`)

Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)  
Implementation tracker: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)  
Platform ADRs: [`../../docs/adr/README.md`](../../../../docs/adr/README.md)

---

## When to write an ADR

Create an ADR for agent-level decisions that affect **domain behavior, contracts, or integration choices**, for example:

- Capability model, I/O schemas, or multi-step pipeline structure
- Tool/skill selection policy specific to this agent
- Prompt strategy, evaluation hooks, or risk classification changes
- External data sources or vendor choices consumed through Harness tools

**Not required:** harness platform changes (use `docs/adr/`), Tier-3 host wiring (use application ADRs),
or trivial refactors with no behavioral impact.

## Naming

```text
ADR-DISPUTE_INTAKE-{NNN}.md
```

## Process

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential number.
2. Link from [`ARCHITECTURE.md`](../ARCHITECTURE.md) when the decision affects runtime layout.
3. Track implementation in [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

## Index

| ADR | Title | Status |
|-----|-------|--------|
| — | *No agent ADRs yet* | — |

---

*Scaffold baseline: 2026-06-07*

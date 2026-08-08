# Governed Contractor — Architecture Decision Records

**Domain:** Tier-3 application host (`applications/governed_contractor_application/`)

Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)  
Implementation tracker: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)  
Platform ADRs: [`../../../../docs/project/technical/adr/README.md`](../../../adr/README.md)

---

## When to write an ADR

Create an ADR for **product-environment** decisions, for example:

- Manifest roster, agent bindings, or execution graph topology
- Environment profile, tool/skill/integration profiles for this host
- Serving API shape, auth model, deployment topology, or MCP exposure
- Partner handoff contracts for this host (not platform-core partner identity)

**Not required:** Nexus platform contract changes (use `docs/project/technical/adr/`), single-agent domain logic (use agent ADRs),
or configuration-only tweaks with no architectural impact.

## Naming

```text
ADR-GOVERNED_CONTRACTOR-{NNN}.md
```

## Process

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential number.
2. Link from [`ARCHITECTURE.md`](../ARCHITECTURE.md) when the decision affects host layout.
3. Track implementation in [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-GOVERNED_CONTRACTOR-001](ADR-GOVERNED_CONTRACTOR-001.md) | GEC vertical — product host + external adapter split | Accepted |

---

*Scaffold baseline: 2026-07-20*

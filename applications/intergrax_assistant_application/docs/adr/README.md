# Intergrax Assistant - Architecture Decision Records

**Domain:** Tier-3 application host (`applications/intergrax_assistant_application`)

Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)  
Implementation tracker: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)  
Platform ADRs: [`../project/technical/adr/README.md`](../../../../docs/project/technical/adr/README.md)

---

## When to write an ADR

Create an ADR for **product-environment** decisions, for example:

- Manifest roster, agent bindings, or execution graph topology
- Environment profile, tool/skill/integration profiles for this host
- Serving API shape, auth model, deployment topology, or MCP exposure
- Cross-agent orchestration declared in this application (not Nexus core semantics)

**Not required:** Nexus platform contract changes (use `docs/project/technical/adr`), single-agent domain logic (use agent ADRs),
or configuration-only tweaks with no architectural impact.

## Naming

```text
ADR-INTERGRAX_ASSISTANT-{NNN}.md
```

## Process

1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential number.
2. Link from [`ARCHITECTURE.md`](../ARCHITECTURE.md) when the decision affects host layout.
3. Track implementation in [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

## Index

| ADR | Title | Status |
|-----|-------|--------|
| [ADR-INTERGRAX_ASSISTANT-001](ADR-INTERGRAX_ASSISTANT-001.md) | Hub-and-spoke harness chat environment | Accepted |

---

*Scaffold baseline: 2026-06-08*

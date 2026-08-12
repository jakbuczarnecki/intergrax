# Harness Guides & Strategy

Authoring guides, strategic context, and audit methodology.  
**Navigation hub:** [`DOCUMENTATION_MAP.md`](../DOCUMENTATION_MAP.md) — what to read, when, and doc roles.  
**Platform canon** (architecture + implementation pairs) lives in [`intergrax_runtime_architecture.md`](../../architecture/intergrax_runtime_architecture.md).

| Document | Purpose |
|----------|---------|
| [SYSTEM_INVARIANTS.md](SYSTEM_INVARIANTS.md) | **Cross-layer MUST/MUST NOT rules** + `SYS-INV-*` index (P2-ARCH-01) — read before every implementation session |
| [MATURITY_TAXONOMY.md](MATURITY_TAXONOMY.md) | **Four-axis maturity vocabulary** (A/I/P/E) — required before using *production-ready*, *L4*, *done*, etc. (P2-ARCH-02) |
| [AGENT_AUTHOR_MINIMAL_PATH.md](AGENT_AUTHOR_MINIMAL_PATH.md) | **Minimal safe path for Tier-2 agent authors** — contracts, flow, MUST NOT, Cursor checklist |
| [TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md](TIER3_PRODUCT_HYPOTHESIS_CONTRACT.md) | **Tier-3 product hypothesis contract** — required before new application hosts |
| [LAYER_COMPLETION_MODE.md](LAYER_COMPLETION_MODE.md) | **Deep domain layer closeout** — extended workflow beyond default iteration |
| [INTERGRAX_DEVELOPMENT_STRATEGY.md](INTERGRAX_DEVELOPMENT_STRATEGY.md) | Strategic goal, decision hierarchy, work cycle |
| [IDEAL_HARNESS_AI_ARCHITECTURE.md](IDEAL_HARNESS_AI_ARCHITECTURE.md) | Target Harness AI reference model |
| [INTEGRAX_HARNESS_AUDIT_MAP.md](INTEGRAX_HARNESS_AUDIT_MAP.md) | 32 auditable layers, evidence, audit procedure |
| [AGENT_CREATION_GUIDE.md](AGENT_CREATION_GUIDE.md) | Scaffold → register → run → evaluate |
| [EXTENSION_AUTHOR_GUIDE.md](EXTENSION_AUTHOR_GUIDE.md) | **Platform Plugin / extension development** — start here: surface decision tree, delivery model, 12-surface matrix → domain guides |
| [HARNESS_ENVIRONMENT.md](HARNESS_ENVIRONMENT.md) | Lab stack, OTLP, presets |
| [HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md](HARNESS_IMPLEMENTATION_AUDIT_PROMPT.md) | Multi-layer / full-platform audit prompt |
| [audit/](../../maintainers/audit) | **Architecture audit orchestration** — 22 per-domain prompts (Modes A/B/C/I) |
| [bootstrap/](../../maintainers/bootstrap) | **Cursor session paste files** — copy into new agent chat |
| [implementation-journal/README.md](../../maintainers/implementation-journal/README.md) | **Implementation journal** — chronological episode log (Tier-0–3) |

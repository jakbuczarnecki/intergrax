# Intergrax Documentation

This is the canonical documentation home for Intergrax. Choose the next document by
what you want to understand, try, inspect, build, or review.

## Start here

- **Understand Intergrax** - [Why Intergrax](overview/WHY_INTERGRAX.md) explains the
  problem, value, and fit.
- **Explore Virtual Workforce / Virtual Workers** - [Virtual Workforce](overview/VIRTUAL_WORKFORCE.md)
  explains the product-facing concept; [Autonomous Work](architecture/AUTONOMOUS_WORK.md)
  is the canonical technical domain and [its implementation plan](maintainers/plans/AUTONOMOUS_WORK.md)
  tracks delivery. This is a strategic architecture direction, **not a shipped production claim**.
- **Try LKW** - [LKW Product Tour](../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md) →
  [LKW Quick Start](../../applications/local_workspace_application/docs/product/QUICKSTART.md). This is the primary product action.
- **Review proof** - [Proofs](proofs/PROOFS.md) shows evidence status; [LKW Platform
  Proof](../../applications/local_workspace_application/docs/proof/LKW_PLATFORM_PROOF.md) provides the deeper product evidence route.
- **Evaluate** - use the [Evaluation Guide](builders/EVALUATION_GUIDE.md) to test one
  selected claim or workflow.
- **Build with Intergrax** - [Builder Quick Start](builders/BUILDER_QUICKSTART.md) →
  [Build With Intergrax](builders/BUILD_WITH_INTERGRAX.md).
- **Review architecture** - [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md)
  gives the project-level architecture route; [Autonomous Work](architecture/AUTONOMOUS_WORK.md)
  defines the canonical persistent-worker layer.
- **Review cross-domain architecture evolution** - [Harness Architecture Evolution Roadmap](overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md)
  defines the canonical implementation sequence, cross-domain invariants, migration gates, and proof requirements for the next architecture stage.

## Choose your path

- **Public or product-oriented reader** - use the
  [Public Documentation Map](community/PUBLIC_DOCUMENTATION_MAP.md) for detailed
  intent routing; for the Virtual Worker direction start with [Virtual Workforce](overview/VIRTUAL_WORKFORCE.md).
- **Builder** - start with the [Builder Quick Start](builders/BUILDER_QUICKSTART.md),
  then continue to [Build With Intergrax](builders/BUILD_WITH_INTERGRAX.md).
- **Evaluator** - use the [Evaluation Guide](builders/EVALUATION_GUIDE.md) as the
  separate bounded evaluation route.
- **Architect** - read the [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md),
  then [Autonomous Work](architecture/AUTONOMOUS_WORK.md) for persistent governed workers when relevant;
  use the [Harness Architecture Evolution Roadmap](overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md)
  for cross-domain sequencing; continue to the [Technical Documentation Map](technical/DOCUMENTATION_MAP.md)
  for deep engineering material.
- **Deep technical reviewer** - go directly to the
  [Technical Documentation Map](technical/DOCUMENTATION_MAP.md).
- **Maintainer** - use [Maintainer Documentation](maintainers/public-adoption/README.md);
  maintainer controls are not a normal reader route.

## Explore the platform

Grouped platform areas and canonical domain links:
[Platform Map in root README](../../README.md#explore-the-intergrax-platform).

Persistent autonomous work and Virtual Worker architecture:
[Autonomous Work](architecture/AUTONOMOUS_WORK.md) → [implementation plan](maintainers/plans/AUTONOMOUS_WORK.md).

Product-facing Virtual Workforce direction:
[Virtual Workforce](overview/VIRTUAL_WORKFORCE.md).

Strategic multi-layer features and cross-layer coordination:
[Capabilities README](capabilities/README.md).

Primary product route:
[LKW Product Tour](../../applications/local_workspace_application/docs/product/LKW_PRODUCT_TOUR.md).

Integration documentation index:
[Integrations](integrations/README.md).

## Documentation depth

| Layer | Role |
| --- | --- |
| [Root README](../../README.md) | First contact, Platform Map, maturity snapshot |
| [Public Documentation Map](community/PUBLIC_DOCUMENTATION_MAP.md) | Intent routing by what you want to do |
| [Architecture Overview](architecture/ARCHITECTURE_OVERVIEW.md) | Project-level architecture mental model |
| [Autonomous Work](architecture/AUTONOMOUS_WORK.md) | Canonical Virtual Worker / persistent autonomous responsibility domain |
| [Virtual Workforce](overview/VIRTUAL_WORKFORCE.md) | Product-facing explanation and strategic positioning; not runtime canon |
| [Harness Architecture Evolution Roadmap](overview/HARNESS_ARCHITECTURE_EVOLUTION_ROADMAP.md) | Cross-domain architecture migration, sequencing, invariants, and proof gates |
| [Technical Documentation Map](technical/DOCUMENTATION_MAP.md) | Engineering routing, domain pairs, guides |
| [Intergrax Proofs](proofs/PROOFS.md) | Bounded evidence and proof status |
| [Maintainer Documentation](maintainers/public-adoption/README.md) | Maintainer controls - not a normal reader route |

## Current maturity

Intergrax is **source-available** and under **active R&D**. LKW is a **Backend
Product Alpha / MVP**. Indexed proof is bounded; mixed indexed + authorized live
Hybrid Ask remains incomplete. **Real-user validation** and **commercial validation**
remain incomplete. The Autonomous Work / Virtual Workforce direction has accepted canonical
architecture and a registered implementation plan, but its dedicated worker runtime,
control plane, reference application, and end-to-end proof are **not implemented yet**.
Detailed evidence and claim boundaries belong in [Intergrax Proofs](proofs/PROOFS.md).

## Browse by documentation area

- `overview/` - project context, use cases, FAQ, roadmaps, positioning, and strategic product concepts (including Virtual Workforce and Agent Marketplace). LKW product docs live under `applications/local_workspace_application/docs/product/`.
- `builders/` - build and evaluation routes.
- `architecture/` - project architecture canon, including the `AUTONOMOUS_WORK` domain.
- `proofs/` - bounded evidence and proof status.
- `capabilities/` and `integrations/` - cross-layer feature coordination, platform capabilities, and integration documentation; they do not replace canonical domain ownership.
- `community/` - public navigation, partner, and collaboration guidance.
- `technical/` - deep engineering and operator material.
- `maintainers/` - documentation governance, implementation plans, and maintainer controls.

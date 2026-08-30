# DECISION_VERIFICATION — pipeline and stage architecture

**Parent hub:** [`DECISION_VERIFICATION.md`](../DECISION_VERIFICATION.md)

## Stage catalog

| Stage kind | Purpose |
| ---------- | ------- |
| Structural / schema | Typed artifact shape and contract validity |
| Deterministic | Rule-based validation, guardrails, L0-class checks |
| Evidence | Evidence refs, claim admissibility, provenance |
| Semantic | Rubric-backed LLM judge (independent profile) |
| Trajectory | Multi-step reasoning path evaluation |
| Independent / custom | Domain or third-party verifier plugins |

## Composition

Pipelines are **declarative compositions** of registered stages — not a monolithic orchestrator class.

## Stage registration

Stages register through typed contracts with explicit required/optional posture and ordering constraints.

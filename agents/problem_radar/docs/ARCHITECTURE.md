# Problem radar agent — architecture

Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

## Purpose

**Band 3 placeholder** — problem discovery capability (`problem_radar.scan`). No product feature work until K.1 reprioritized.

## Capabilities

- `problem_radar.scan`

## I/O

- Domain schemas under `schemas` (Pydantic)
- Notebook: `notebooks/01_problem_radar_experiment.ipynb`

## Runtime

- `HarnessReferenceAgent` + UAEP stub step
- Tier-2 only — imports `intergrax.agents.reference_harness`, not `applications`

## Status

Frozen for business logic; conformance and tier hygiene maintained under Phase AA.

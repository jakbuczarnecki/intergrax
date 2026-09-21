# ADR-GR-10-003 — GR-10 vs GR-13 Governance Evidence certification scope

| Field | Value |
| ----- | ----- |
| **Status** | Accepted |
| **Date** | 2026-09-21 |
| **Related** | [ADR-GR-8-001](../2026-09-17/ADR-GR-8-001.md) · [GOVERNED_EXECUTION.md](../../../../architecture/GOVERNED_EXECUTION.md) §10 |

## Context

GR-8 freezes the **Governance Evidence public contract** and default persistence boundary. ADR-GR-8-001 explicitly excludes **runtime evaluation-point adoption** from GR-8 closure; adoption is owned by **GR-10** and **GR-13**.

ORCHESTRATION SSOT previously used `gr8_evidence_applicability = NOT_APPLICABLE` to mean deferred, which conflated conceptual applicability with certification phase.

## Decision

Split orthogonal semantics in `tests/qualification/governance/strategy/catalog.py`:

| Field | Meaning |
| ----- | ------- |
| `gr8_evidence_applicability` | Conceptual GR-8 fact applicability for the GEP |
| `gr10_evidence_requirement` | GR-10 enterprise qualification obligation |
| `gr13_evidence_requirement` | GR-13 proof-matrix obligation |
| `gr8_evidence_coverage` | Current orchestration GR-8 fact implementation status |

`Gr10EvidenceCertificationRequirement`: `NOT_APPLICABLE`, `REQUIRED_IN_GR10`, `DEFERRED_TO_GR13`, `REQUIRED_IN_GR13`.

**Forbidden:** `NOT_APPLICABLE` on `gr8_evidence_applicability` for applicable Governance GEPs only because adoption is postponed.

## GR-10 qualified Governance Evidence scope

`GR10_ORCHESTRATION_GOVERNANCE_EVIDENCE_INVENTORY` rows with status `QUALIFIED`: root admission, MSE, inner-guard DENY, decision-bound, post-HITL correlation.

Per-GEP adoption for PRE_MODEL, TOOL_PLAN_OR_ACCESS, TOOL_INVOCATION_POLICY, PRE_OUTPUT, POST_RUN is `DEFERRED_TO_GR13` (`GR13_ORCHESTRATION_GOVERNANCE_EVIDENCE_DEFERRED`).

## GR-13 scope

Full per-GEP Governance Evidence proof matrix for deferred rows; enterprise sign-off not claimed here.

## Consequences

- ORCHESTRATION may close under GR-10 with explicit GR-13 deferred obligations.
- GR-8 frozen semantics unchanged; evidence remains non-authoritative.


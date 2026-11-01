# EVOL-X — Contract spec freeze (W1 draft)

**Status:** FROZEN DRAFT for W2-01 implementation in `testing_support/change_safety/contracts/`.  
**Rules:** Pydantic `extra=forbid`; every artifact has `schema_version: str`; **no** `Any`; **no** dict pseudo-contracts; **no** fields named `approve`, `authorize`, `execute`.

## Core enums

- `ImpactStatus`: `COMPLETE | PARTIAL | INTEGRITY_FAILURE | REFUSED`
- `SelectionStatus`: `SELECTED | CONSERVATIVE_FULL | REFUSED_INTEGRITY`
- `CompatibilityAtChangeResult`: `COMPATIBLE | INCOMPATIBLE | MIGRATION_REQUIRED | UNKNOWN`
- `WorkflowEvolutionDecision`: `SAFE_IN_PLACE | VERSIONED_DEPLOY | MIGRATE | REJECT | EXPLICITLY_UNSUPPORTED`
- `DeploySafetyVerdict`: `ELIGIBLE | ELIGIBLE_WITH_DRAIN | REFUSE`
- `DeployRefusalReason`: `PERSISTED_STATE_INCOMPATIBLE | DURABLE_STATE_FORWARD_ONLY | INFLIGHT_PINS | CONTRACT_INCOMPATIBLE | INSUFFICIENT_EVIDENCE | ...`

## Artifacts (minimum fields)

| Artifact | schema_version | Key fields |
|----------|----------------|------------|
| `ImpactAssessmentResult` | evol_impact_report.v1 | `source_commit_sha`, `graph_digest`, `inventory_digest`, `affected[]`, `status` |
| `SelectionDecision` | evol_test_selection.v1 | `decision_id`, `status`, `selected[]`, `mandatory[]`, `reasons[]`, provenance digests |
| `ConsumerClosureReport` | evol_consumer_closure.v1 | `contract_id`, `consumers[]`, `orphan_count` |
| `WorkflowEvolutionVerdict` | evol_workflow_verdict.v1 | `matrix_id` WM-*, `decision`, `witness` |
| `DeploySafetyAssessment` | evol_deploy_assessment.v1 | `verdict`, `reason`, `dimensions[]` (read-only facts) |
| `ChangeDossier` | evol_change_dossier.v1 | sections Q1–Q6, `evidence_refs[]` |
| `EvolMetricsBundle` | evol_metrics.v1 | per-metric numerator/denominator/threshold/status |
| `EvolXAcceptedBaselineManifest` | evol_baseline_manifest.v1 | `inventory_digest`, `triggers_version`, `accepted_at_sha` |

## Ports (`ports.py`)

- `SurfaceSourcePort`, `ConsumerSourcePort`, `SuiteCatalogPort`, `ChangeSetPort`, `RevisionStateFactsPort`, `ClockPort`

## Alignment checks (W1)

- Capability slice: reuse `CapabilityNodeType` / edge types from `capability_graph.py` where adapted — do not fork SSOT.
- COMPAT-X: direction/classes imported via adapter only.

**W2 gate:** `gates/test_evol_x_contracts_*` round-trip + AST forbid `typing.Any` in contracts package.

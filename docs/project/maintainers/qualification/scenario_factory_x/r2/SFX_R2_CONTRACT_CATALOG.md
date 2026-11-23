# SFX R2 — Proposed Contract Catalog

**Status:** PROPOSED — CON-18..23 numbering is draft pending ADR/COMPAT review.

| Draft ID | Schema file | Extends / new | Purpose |
| --- | --- | --- | --- |
| CON-R2-01 | `sfx.mission_record.v1` | new | Mission root record |
| CON-R2-02 | `sfx.autonomy_envelope.v1` | new | Grant scope, budgets, stop rules |
| CON-R2-03 | `sfx.source_intake_manifest.v1` | new | Source provenance |
| CON-R2-04 | `sfx.candidate_iteration.v1` | new | Iteration lineage |
| CON-R2-05 | `sfx.scenario_evaluation_verdict.v1` | new | Business evaluation view |
| CON-R2-06 | `sfx.candidate_comparison.v1` | new | Champion selection input |

All use RFC 8785 digests per `SFX_CANONICALIZATION.md`. Persistence: CW `WorkArtifactVersion` only.

**Count:** six schemas under design/schemas/proposed/ (validated IGX-SFX-R2-CORR-001).

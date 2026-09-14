# INTEGRAx-NPSC-5F-V1-V2-COMPATIBILITY-AND-MIGRATION-IMPLEMENTATION

## Metadata

| Field | Value |
| ----- | ----- |
| **Task** | `INTEGRAx-NPSC-5F-V1-V2-COMPATIBILITY-AND-MIGRATION-IMPLEMENTATION` |
| **Branch** | `development` |
| **Global frozen baseline** | `a185403d0c7524c29bea2fe09212f9508e6bccd8` (unchanged) |
| **Scoped reopen** | `fe2b1510ddd87ebd809de14f9ed9d0a459bfb5b8` |
| **Protected drift source** | `962bf1ade25b220873cb724523ffeff1abf0fbc7` |

## Reopen Reference

Implementation follows [`INTEGRAX_NPSC_5F_SCOPED_ARCHITECTURE_REOPEN.md`](INTEGRAX_NPSC_5F_SCOPED_ARCHITECTURE_REOPEN.md): read v1 + v2, write v2 only, no observability identity mint, provider-neutral persistence.

## Changed Production Surfaces

- `intergrax/contracts/npsc5f_compatibility.py` — typed compatibility errors
- `intergrax/runtime/observability/causal_evidence_legacy.py`, `platform_causal_evidence_codec.py`, `causal_evidence_enrichment.py`
- `intergrax/runtime/observability/causal_evidence_record_codec.py`, `causal_evidence_export.py`, `export_boundary.py`
- `intergrax/runtime/observability/document_store_causal_evidence_persistence.py` — dual-read index decode
- `intergrax/runtime/background_execution/identity_types.py`, `identity_record_codec.py`, `identity_dual_read.py`, `identity_persistence.py`

## Compatibility Architecture

Explicit schema dispatch (`platform_causal_evidence.v1` / `.v2`, `causal_evidence_export_source.v1` / `.v2`, KV 3- vs 4-field identity, document partitions `intergrax.bg_exec_identity.v1` / `.v2`). Shared resolver `reconcile_dual_read_records` + optional `CanonicalExecutionIdLookupPort`.

## v1 Read Semantics

Legacy causal evidence and 3-field background identity decode as incomplete; enrichment only via injected canonical lookup (exactly one ExecutionId).

## v2 Write Semantics

All new platform causal evidence persistence and background identity stores write v2-only shapes; v1 platform write guarded by `ForbiddenPlatformCausalEvidenceV1WriteError`.

## Migration / Enrichment Rules

Deterministic, fail-closed; no mint/guess/heuristic resolution. Document/KV dual-read with conflict detection (`BackgroundExecutionIdentityConflictError`).

## Typed Errors

See `intergrax/contracts/npsc5f_compatibility.py`.

## Persistence Provider Design

Provider-specific load/store; shared codec + dual-read reconciliation in core (no vendor branches in enrichment).

## Export Versioning

`CausalEvidenceExportSource` (v2, required `target_execution_id`); `LegacyCausalEvidenceExportSource` (v1); explicit `CausalEvidenceExportVersion` selection.

## Consumer Changes

Persistence decode returns `DecodedPlatformCausalEvidence`; strict v2 paths use `decode_causal_evidence_record_v2`. Background persistence accepts optional `legacy_lookup` on wire.

## Architecture Invariants

Single identity authority unchanged; evidence ≠ control; observability enrichment modules contain no identity mint imports.

## Tests

`tests/unit/runtime/observability/test_npsc5f_v1_v2_compatibility.py` — causal, export, background identity, tenant isolation, no-mint guard.

## Static Quality

`ruff check/format` + `pyright` on changed scope (pre-existing `DocumentQueryCursorCodec` annotations in document store unchanged).

## Findings

None blocking implementation scope.

## Remaining Requalification Work

**NPSC-5F R3 + Final Requalification** — resign protected drift fingerprints (expected RED until next task).

## Commit

(Set at commit — `INTEGRAx-NPSC-5F-V1-V2-COMPATIBILITY-AND-MIGRATION-IMPLEMENTATION`)

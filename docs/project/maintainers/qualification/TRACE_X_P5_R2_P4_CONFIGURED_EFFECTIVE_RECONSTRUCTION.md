# TRACE-X-P5-R2-P4 — Configured / Effective Reconstruction Projection

**Status:** **BLOCKED** (P4 implementation; architecture locks through P4-R1-R1)  
**Rejected implementation baseline:** `055ed448cb890026c8c34e336e7baf7422daa2f6`  
**R1 reconciliation:** [`TRACE_X_P5_R2_P4_R1_RECONSTRUCTION_REQUIREMENT_AUTHORITY_AS_OF_RECONCILIATION.md`](TRACE_X_P5_R2_P4_R1_RECONSTRUCTION_REQUIREMENT_AUTHORITY_AS_OF_RECONCILIATION.md)  
**R1-R1 reconciliation:** [`TRACE_X_P5_R2_P4_R1_R1_REQUIREMENT_EVIDENCE_EMISSION_BOUNDARY_DUAL_WRITE_RECONCILIATION.md`](TRACE_X_P5_R2_P4_R1_R1_REQUIREMENT_EVIDENCE_EMISSION_BOUNDARY_DUAL_WRITE_RECONCILIATION.md)  
**Parent:** **TRACE-X-P5-R2** = **CURRENT**  
**P5-GAP-04** = **IMPLEMENTATION IN PROGRESS** (P4 wave blocked)  
**FRZ-TRC-11** = **OPEN**  
**P5 / CERT:** **NOT ENTERED**

## Blockers (independent audit)

| ID | Summary |
|---|---|
| `R2-P4-PROVENANCE-REQUIREMENT-AUTHORITY-24` | Magic runtime payload requirement — no production emitter; fail-closed not mechanical |
| `R2-P4-AS-OF-CONFIG-PROVENANCE-FUTURE-LEAK-25` | `read_all` against non-temporal P2 store under `execution_as_of` |
| `R2-P4-REQUIREMENT-EVIDENCE-EMITTER-BOUNDARY-26` | Integrations must not own runtime spine emit — P4-R1-R1 |
| `R2-P4-PIN-REQUIREMENT-EVIDENCE-DUAL-WRITE-27` | Pin vs spine dual-write protocol — P4-R1-R1 |

**Do not** treat this document as audit PASS until **P4-R1-R1** is independently accepted and corrected P4 lands per R1-R1 + R1 §9.

## Inventory @ rejected baseline (lineage only)

| Artifact | Path |
|---|---|
| Neutral reader Protocol | `intergrax/contracts/execution_integration_configuration_provenance.py` |
| Pinning-store reader adapter | `intergrax/applications/_shared/integrations/integration_configuration_provenance_reader.py` |
| Reconstruction projection | `intergrax/runtime/observability/reconstruction/integration_configuration_provenance_projection.py` |
| Reconstructor injection | `intergrax/runtime/observability/reconstruction/execution_reconstruction.py` |
| Reconstruction DTO fields | `intergrax/contracts/execution_reconstruction_models.py` |
| Diagnostic composition | `intergrax/applications/_shared/diagnostic_composition.py`, `diagnostic_read_wiring.py` |
| Host wiring | `harness_host_runtime.py`, `scenario_runtime_baseline.py` |

## Superseded claims (rejected baseline)

- Runtime payload `execution_integration_configuration_provenance_required` as requirement authority — **rejected** in P4-R1.
- “Unresolved / ADR: None” — **incorrect**; blockers 24–25 require R1 reconciliation.

## Tests (rejected baseline — not audit evidence)

- Unit: `tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py`
- Gates: `tests/qualification/trace_x/test_trace_x_p5_r2_p4_reconstruction_gates.py`

Corrected P4 must add Docker durable restart E2E per P4-R1 §10.

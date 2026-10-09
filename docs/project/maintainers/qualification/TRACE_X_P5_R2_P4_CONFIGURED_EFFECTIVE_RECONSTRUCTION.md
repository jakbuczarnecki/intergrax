# TRACE-X-P5-R2-P4 — Configured / Effective Reconstruction Projection

**Status:** **READY FOR AUDIT**  
**START_HEAD:** `5c47154066915820cd150d4fd370bc8b1ae8f85c`  
**Parent:** **TRACE-X-P5-R2** = **CURRENT**  
**P5-GAP-04** = **IMPLEMENTATION IN PROGRESS**  
**FRZ-TRC-11** = **OPEN**  
**P5 / CERT:** **NOT ENTERED**

## Inventory @ implementation

| Artifact | Path |
|---|---|
| Neutral reader Protocol | `intergrax/contracts/execution_integration_configuration_provenance.py` |
| Pinning-store reader adapter | `intergrax/applications/_shared/integrations/integration_configuration_provenance_reader.py` |
| Reconstruction projection | `intergrax/runtime/observability/reconstruction/integration_configuration_provenance_projection.py` |
| Reconstructor injection | `intergrax/runtime/observability/reconstruction/execution_reconstruction.py` |
| Reconstruction DTO fields | `intergrax/contracts/execution_reconstruction_models.py` |
| Diagnostic composition | `intergrax/applications/_shared/diagnostic_composition.py`, `diagnostic_read_wiring.py` |
| Host wiring | `harness_host_runtime.py`, `scenario_runtime_baseline.py` |

## Reader adapter

`PinningStoreExecutionIntegrationConfigurationProvenanceReader` delegates to `ExecutionIntegrationConfigurationPinningStore.read_all` only; validates tenant + `ExecutionId` per record (fail closed). `resolve_pinning_store_integration_configuration_provenance_reader` binds the same KV/DocumentStore backend as P2/P3 writes.

## Projection semantics

- Execution IDs from `discover_execution_ids_in_positioned_history` (positioned runtime evidence only).
- Optional requirement classification: runtime payload `execution_integration_configuration_provenance_required: true` (no heuristic join).
- Multiplicity: all subjects per execution preserved; stable ordering from store + sorted execution IDs.
- Child executions: no parent provenance inheritance.
- As-of: integration provenance is **not** temporally versioned in P2 store; `execution_as_of` limits **which execution IDs appear** via truncated positioned history only — **no timestamp filtering of pins**.

## Duplicate audit

| Check | Result |
|---|---|
| `ExecutionReconstructor` count | 1 |
| Integration provenance reader semantic owner | 1 (adapter) |
| Pinning store semantic owner | 1 (unchanged P2) |
| Reconstruction projection core | 1 |
| Diagnostics truth owner | 0 |
| Provider resolver in reconstruction | 0 |

## Tests

- Unit: `tests/unit/runtime/observability/reconstruction/test_trace_x_p5_r2_p4_integration_configuration_provenance.py`
- Gates: `tests/qualification/trace_x/test_trace_x_p5_r2_p4_reconstruction_gates.py`
- Regression replay: P2 persistence, P4 gates, P5-R1 profile provenance reconstruction, diagnostic composition (see pytest log in session).

## Pyright

Targeted surfaces: contracts reconstruction models, projection, reconstructor, reader adapter, diagnostic composition — **0 errors** expected.

## Unresolved / ADR

None — no STOP conditions triggered.

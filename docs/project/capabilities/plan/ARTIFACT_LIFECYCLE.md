# Artifact Lifecycle — delivery plan

**Architecture pair:** [architecture/ARTIFACT_LIFECYCLE.md](../architecture/ARTIFACT_LIFECYCLE.md)  
**Ticket and plan:** [`governance/tickets/T-IGX-036-Q.md`](../../../../governance/tickets/T-IGX-036-Q.md), [`PLAN_T-IGX-036-Q.md`](../../../../governance/tickets/plans/PLAN_T-IGX-036-Q.md) (PDC-X batch B-D8)  
**Decision:** ADR-PDC-08; OD-PDC-SCOPE-04 = A

## Current wave

| Wave | Scope | Ticket | State |
| --- | --- | --- | --- |
| W3b | Taxonomy, mandatory digest, governed access and download port, durable store, retention, spill, dedupe | T-IGX-036-Q | implemented, draft PR, not merged (OD-PDC-SEQ-01: merge after MCP-SERVER-X P8) |
| W3c | Hosting (B-D4) follows this batch | T-IGX-036-R | not started |
| P6 | Cross-domain XD-09 and XD-10 re-pointed by PDC-P6-01 | T-IGX-036-S | not started |

## Requirement coverage

| Requirement | Mechanism | Test |
| --- | --- | --- |
| PDC-D6-R022 | TTL sweep deletes expired artifacts | `tests/qualification/pdc_x/d8/test_xd10_tenant_and_ttl.py` |
| PDC-D8-R001 | taxonomy, owner table, store refuses foreign classes | `test_d8_taxonomy_gate.py` |
| PDC-D8-R002 | `DurableArtifactRef` mandatory digest, mismatch rejected | `test_d8_artifact_ref_contract.py` |
| PDC-D8-R003 | append-only versions | `test_d8_versioning_and_durable_store.py` |
| PDC-D8-R004 | producer, lineage, `reconstruct_artifact_lineage`, bundle projection | `test_d8_artifact_ref_contract.py`, `test_d8_collaborative_work_and_bundle.py` |
| PDC-D8-R005 | tenant check in the governed port | `test_xd10_tenant_and_ttl.py` |
| PDC-D8-R006 | authorization through the governance owner | `test_d8_governed_port.py` |
| PDC-D8-R007 | `SQLiteArtifactStore` | `test_d8_versioning_and_durable_store.py` |
| PDC-D8-R008 | retention class, TTL, `ArtifactRetentionPolicy` consumer, sweep | `test_d8_retention.py` |
| PDC-D8-R009 | evidence and legal hold never deleted; no hard-delete API in Collaborative Work | `test_d8_retention.py`, `test_d8_collaborative_work_and_bundle.py` |
| PDC-D8-R010 | governed download (port plus short-lived URL) | `test_d8_governed_port.py` |
| PDC-D8-R012 | tool output spill round trip | `test_xd09_spill_and_context_reference.py` |
| PDC-D8-R013 | context reference resolves after compaction | `test_xd09_spill_and_context_reference.py` |
| PDC-D8-R014 | restart recovery on a real SQLite store | `test_d8_versioning_and_durable_store.py` |
| PDC-D8-R016 | one audit event per artifact operation | `test_d8_governed_port.py`, `test_d8_retention.py` |
| PDC-D8-R017 | single definition and COMPAT-X shim | `test_d8_dedupe_and_aliases.py` |
| PDC-D8-R018 | one `ArtifactRef` contract, trace alias renamed | `test_d8_dedupe_and_aliases.py` |

## Open items outside the B-D8 write set

Reported to the orchestrator, not edited here (hub registration, wiring, alias callers); the rest is closed:

- **Hub registration:** `docs/project/capabilities/README.md` (feature index) and the runtime hub do not list this pair yet.
- **Wiring:** the tool runtime must call `ToolOutputSpiller` for oversized tool results; the composition root must build the governed port, bind the audit recorder and register `register_retention_sweep` on the worker registry; `sqlite_opens.py` and `sqlite_composition.py` need an artifact store opener.
- **Callers of the renamed trace alias** (still working through the COMPAT-X shim; they should import `TraceArtifactRef` from `intergrax.contracts.tracing`): `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/runtime/execution/trace_persistence_models_bridge.py`, `intergrax/runtime/nexus/context/context_manager.py`, `intergrax/runtime/nexus/context/shared_task_context.py`, `intergrax/runtime/nexus/engine/runtime_state.py`, `intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py`, `intergrax/runtime/nexus/tracing/persistence_models.py`.
- **Closed by Amendment 1:** the STATE-X classification entry for `sqlite_artifact_store.py`; `storage.presigned_url` now fails closed (`GovernedPresignRequiredError`) so presigned URLs come only from the governed port; the replay type is `ReplayArtifactRef` and `intergrax.runtime.replay.models.ArtifactRef` still resolves through a COMPAT-X shim.

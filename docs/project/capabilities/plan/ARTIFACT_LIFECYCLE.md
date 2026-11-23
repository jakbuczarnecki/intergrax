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

## PDC-P7 repair after the independent audit (T-IGX-041-H, 2026-11-15)

Ticket [`governance/tickets/T-IGX-041-H.md`](../../../../governance/tickets/T-IGX-041-H.md), plan [`PLAN_T-IGX-041-H.md`](../../../../governance/tickets/plans/PLAN_T-IGX-041-H.md) (batch B-P7-D8). Tests: `tests/qualification/pdc_x/d8_repair/`.

| Area | Delivered | Requirement rows |
| --- | --- | --- |
| Runtime gateway | `RuntimeContext.create_artifact` stores tenant, producer and lineage | PDC-D8-R004, R007 |
| Retention sweep | delete conditional on the observed version; refused without an audit recorder unless explicitly allowed; positive limit | PDC-D8-R008, R009, R016, PDC-D6-R022 |
| Expiry at read | an expired, unprotected version is not found | PDC-D8-R008 |
| Workspace scope | a stored workspace is required on read, hold, release and delete | PDC-D8-R005, R006 |
| Governed port | a returned DENY or approval-required decision denies; backend failures leave an outcome event | PDC-D8-R006, R016 |
| Raw storage tools | refuse the Collaborative Work content key space | PDC-D8-R005, R006, R009 |

**Declared exceptions (not repaired, reopen conditions in the architecture doc):** host composition of the artifact store, governance port, audit recorder and sweep trigger; an `AUDIT_EVIDENCE` producer; the spill resolver; tenant binding of the raw storage tools; principal or run binding of the capabilities; the LOW findings F-D8-09 (remainder), F-D8-10, F-D8-11, F-D8-13, F-D8-14, F-D8-15.

## Open items outside the B-D8 write set

Reported to the orchestrator, not edited here (hub registration, wiring, alias callers); the rest is closed:

- **Hub registration:** `docs/project/capabilities/README.md` (feature index) and the runtime hub do not list this pair yet.
- **Wiring:** the tool runtime must call `ToolOutputSpiller` for oversized tool results; the composition root must build the governed port, bind the audit recorder and register `register_retention_sweep` on the worker registry; `sqlite_opens.py` and `sqlite_composition.py` need an artifact store opener.
- **Callers of the renamed trace alias** (still working through the COMPAT-X shim; they should import `TraceArtifactRef` from `intergrax.contracts.tracing`): `intergrax/runtime/execution/agent_runtime_context.py`, `intergrax/runtime/execution/trace_persistence_models_bridge.py`, `intergrax/runtime/nexus/context/context_manager.py`, `intergrax/runtime/nexus/context/shared_task_context.py`, `intergrax/runtime/nexus/engine/runtime_state.py`, `intergrax/runtime/nexus/engine/contracts/runtime_state_contract.py`, `intergrax/runtime/nexus/tracing/persistence_models.py`.
- **Closed by Amendment 1:** the STATE-X classification entry for `sqlite_artifact_store.py`; `storage.presigned_url` now fails closed (`GovernedPresignRequiredError`) so presigned URLs come only from the governed port; the replay type is `ReplayArtifactRef` and `intergrax.runtime.replay.models.ArtifactRef` still resolves through a COMPAT-X shim.

**Note (EQO-A remediation, as of 2026-11-16; Operator decision DP-57: the exceptions stay PDC-D8 reopen conditions).** Two bullets above are point-in-time. Hub registration: `docs/project/capabilities/README.md` L79 and the runtime hub (`docs/project/architecture/intergrax_runtime_architecture.md` L39) list this pair at `2ab57e36d94eae98c962b3d7b09e5a540d8c70d2`. Wiring: `ToolOutputSpiller` has one guarded call site in `intergrax/runtime/nexus/tools/plan_context_invocation.py` L68, inert while `artifact_store` is `None`, and an artifact store opener exists (`open_artifact_store_at`, `intergrax/runtime/persistence/sqlite_opens.py` L87, used at `sqlite_composition.py` L127). The remaining items of the wiring bullet were not re-verified here; the exceptions table of the architecture document states the open reopen conditions. This note closes none of them and claims no certification.

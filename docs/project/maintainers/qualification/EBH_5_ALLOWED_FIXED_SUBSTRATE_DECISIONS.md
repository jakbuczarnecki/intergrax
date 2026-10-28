# EBH-5 — ALLOWED_FIXED_SUBSTRATE_DECISIONS

| Field | Value |
| --- | --- |
| **Status** | Ratification records for EBH-5 — **not** global FRZ PASS |
| **Evidence SHA** | `5c9b66db632976ada039d406474530d7c6cb5f86` |

## Decisions

| substrate_id | technology | scope_paths | contract | multi_instance_constraints | adr_ref |
| --- | --- | --- | --- | --- | --- |
| `runtime-sqlite-persistence-bundle` | SQLite file stores | `intergrax/runtime/**/sqlite*.py`, `intergrax/agents/persistence/*sqlite*` | Port per store (TaskCheckpointPersistence, etc.) | Single-node / lab / profile-scoped; not claimed horizontally scalable | STATE-X inventory SX-F* |
| `single-node-profile-v1-sqlite-triad` | SQLite (GCA, PIS, AD) | `intergrax/applications/_shared/single_node_production_profile_v1.py` | `SingleNodeProductionProfileV1Config` | Strict single-node topology; advisory cross-process lock | PROD-Q ADR chain |
| `lab-adaptive-sqlite` | SQLite | `applications/lab_application/host/factory.py` | SignalStore / ProposalStore | Lab harness only; paths under lab trace/checkpoint anchor | EBH-5 GAP-01 remediation |

## Operator review (OD-EBH5-04)

Runtime SQLite ports remain **FIXED SUBSTRATE** unless operator promotes specific ports to `extensibility_promised=yes` with full RPL L3 proof.

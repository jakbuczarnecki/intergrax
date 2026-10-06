# TRACE-X-P3-R1 — Governed boundary v2 certification

**Parent:** TRACE-X-P3 — Tool, Provider & Side-Effect Authorization Attribution

**Accepted evidence/code baseline:** `3799b2d974369e6002ac5326e62c8c8b381e7944`

Qualification entrypoint: `tests/qualification/trace_x/test_trace_x_p3_r1_governed_boundary_v2.py`

Provenance (child lineage):

| Stage | START_HEAD / acceptance SHA |
|---|---|
| TRACE-X-P3-R1 (initial) | `7bc5a7ec0fbacaf52558fdaf53384c20fa359e1b` |
| TRACE-X-P3-R1-R1 | `22a0725fe397ffaf0d82a955a6bec4f0cf644a1b` |
| TRACE-X-P3-R1-R1-Q1 | `8b448c5d99c02cc695b73fba2a5c1fa3830ac526` |
| TRACE-X-P3-R1-R1-Q2 | `fa8a4656efb131a18a2c5f8d98c941bf53aa6572` |
| TRACE-X-P3-R1-R1-Q3 / P3-R1 / P3 final | `3799b2d974369e6002ac5326e62c8c8b381e7944` |

`TRACE_X_P3_R1_START_HEAD = 72c1aca53663fe002faf2bb9bc4d5a14d2be92b1` (ancestry anchor)

Scope: **FRZ-TRC-04**, **FRZ-TRC-06** — independently accepted @ `3799b2d974369e6002ac5326e62c8c8b381e7944`.

## Status

| Item | State |
|---|---|
| **TRACE-X-P3-R1** | **CLOSED / INDEPENDENTLY ACCEPTED** @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |
| **TRACE-X-P3-R1-R1-Q3** | **CLOSED / INDEPENDENTLY ACCEPTED** @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |
| **TRACE-X-P3** (parent) | **CLOSED / INDEPENDENTLY ACCEPTED** @ `3799b2d974369e6002ac5326e62c8c8b381e7944` |

## Blockers (final)

| ID | State |
|---|---|
| P3-B04-01 | **RESOLVED** |
| P3-B06-01 | **RESOLVED** |
| P3-R1-BLK-TASK-IDENTITY-01 | **RESOLVED** |
| P3-R1-BLK-TENANT-01 | **RESOLVED** |
| P3-R1-R1-Q-BLK-01 | **RESOLVED** / independently accepted |
| P3-R1-R1-Q-BLK-02 | **RESOLVED** / independently accepted (exact authorization→effect binding; Branch B propagation via optional GR-8 persistence at external-work composition) |
| P3-R1-R1-Q2-BLK-EVIDENCE-PERSIST-01 | **RESOLVED** / independently accepted |

## Certified properties

- `governed_execution_boundary_event.v2` + **ProofReceiptV2** with exact provider invocation binding and exact authorization binding.
- Canonical **TaskId** / **RunId** / **AttemptId** / **ExecutionId** continuity from **Execution** only (**Governance ≠ Execution**).
- **FRZ-TRC-04:** `TaskId`, `RunId`, `AttemptId`, `ExecutionId`, `ProviderInvocation.invocation_id` preserved atomically in versioned execution evidence; no heuristic join; no host-minted canonical **ExecutionId**.
- **FRZ-TRC-06:** authorization → Task/Run/Attempt/Execution → provider invocation → provider outcome → governed side-effect evidence → signed **ProofReceiptV2**; adversarial cross-execution / cross-attempt / cross-tenant rejection; DENY cannot produce successful receipt.
- **GovernanceEvidenceRef** only when `GovernanceEvidencePersistenceOutcome.persisted == True` for the exact fact; `persisted=False` → no **GovernanceEvidenceRef** → no dangling signed evidence reference.
- **GR-8** evidence persistence failure does **not** convert ALLOW into DENY.
- `governed_execution_boundary_event.v1` retained for legacy/read compatibility; v2 = canonical writer; ProofReceipt v1 ↔ v1, v2 ↔ v2; cross-version pairings fail closed.
- Reliability/Observability projections do **not** promote to execution truth.

## P3-R1-R1 mechanical qualification

- **Gate SSOT:** `tests/qualification/trace_x/_trace_x_p3_r1_support.py` (`P3R1GateEvidence`, `P3_R1_R1_GATE_REGISTRY`)
- **Pass 1:** set `TRACE_X_P3_R1_R1_PASS1=1` and run qualification entry + `test_p3_r1_r1_governed_identity_tenant_gate.py` + `test_p3_r1_r1_exact_authorization_effect_binding.py` (session hook validates `PASS1_MECHANICAL_NODEIDS`)
- **Pass 2:** `test_governed_execution_result.py`, `test_host_attestation_and_receipt.py`, `test_canonical_serialization.py`, `test_fresh_side_effect_authorization.py`, `test_mse_governance_evidence_projection.py`, `test_governed_proof.py`

### Mechanical gate → evidence (TXP3R1R1-Q03..Q39)

See `P3_R1_R1_GATE_REGISTRY` in `_trace_x_p3_r1_support.py` for authoritative `gate_id → pytest nodeid → FRZ` mapping.

### Tracked freeze debt (not P3-R1-R1 closure)

- **Nodeid:** `applications/governed_contractor_application/tests/host/test_gr6_wire_production_decision_governance.py::test_strict_host_composition_wires_agent_boundary_and_integration`
- **Exception:** `ToolDependencyAttemptBoundaryMaterializationError`
- **Classification:** **TRACKED FREEZE DEBT** (Reliability / Composition / PROD-Q / QUAL-X)

ADR: `docs/project/maintainers/architecture/ADR/ADR-TRACE-X-P3-R1-GOVERNED-BOUNDARY-V2.md`

**Next mandatory TRACE-X stage:** **TRACE-X-P4 — Model Call & Context Decision Attribution** — **NOT ENTERED**.

---

**Wprowadzone zmiany muszą zostać niezależnie zaudytowane na podstawie kodu z commitu znajdującego się na GitHubie. Raport Cursor AI nie jest podstawą do finalnego zamknięcia zadania.**

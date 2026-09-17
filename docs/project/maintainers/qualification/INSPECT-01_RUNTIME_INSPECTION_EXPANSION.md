# INSPECT-01 — Runtime inspection expansion

## INSPECT-01-A — Canonical Runtime Read Contract & Federation Foundation

**Task SHA (baseline):** `63608aae899b69d4c9fa5f8613054a2d25de6931` (origin/development at task intake)  
**Implementation HEAD:** recorded at INSPECT-01-A closeout commit.

### As-built before A

- Tier-3 `RuntimeInspectionService` and P1.4 `RuntimeInspectionProvider` seam for profile/revision/execution pinning/capability extension evidence.
- No canonical cross-domain `ExecutionId → RuntimeInspectionSnapshot` contract in `intergrax/contracts/`.

### Canonical ownership

| Concern | Owner |
| --- | --- |
| Execution facts / timeline | `ExecutionReconstructionReader` |
| Diagnostic interpretation | `DiagnosticReadService` |
| Evidence references | `FunctionalEvidencePersistence` (IDs only in snapshot) |
| Federation aggregation | `FederatedRuntimeInspectionReadService` (read-only) |
| Profile/capability P1.4 inspection | Tier-3 application host (`RuntimeInspectionService`) |

### New canonical runtime read boundary

- Contracts: `intergrax/contracts/runtime_inspection/`
- Federation: `intergrax/runtime/runtime_inspection/`
- Tier-3 bridge: `inspect_execution_runtime_snapshot` in `applications/_shared/runtime_inspection/`

Dependency direction: Tier-3 → canonical contracts; contracts do not import application implementations.

### Completeness, redaction, side effects

- `RuntimeInspectionCompleteness` scoped to configured federation sources.
- Timeline/diagnostic summaries sanitized before snapshot assembly (`runtime_inspection/redaction.py`).
- Federation holds no command ports; qualification gates assert zero writes via read-only adapters.

### Follow-ups

| Item | Track |
| --- | --- |
| Tool/Governance/Session/Memory full sections | INSPECT-01-B |
| Search/RQ surfaces | INSPECT-01-RQ |
| Memory persistence gaps | MEM-01 / INSPECT-01-C |

## INSPECT-01-B — Core Domain Read Adoption

**Baseline parent (INSPECT-01-A):** `6b3744f356c6e725fc124843f1da4a8fd5417120`  
**Task intake `origin/development`:** `995c66f42e54e41c1a62495dc87ace0ccb0fd469`  
**Closeout HEAD:** recorded at INSPECT-01-B commit on `development`.

### Domain read ownership

| Domain | Canonical owner | Read contract | Inspection section |
| --- | --- | --- | --- |
| Tool invocations | ToolRuntime spine / reconstruction | `ToolRuntimeInvocationReadPort` (`intergrax/contracts/tool_runtime_read.py`) | `RuntimeInspectionToolSection` |
| Governance decisions | Agent runtime governance audit | `GovernanceAuditReadPort` (`intergrax/contracts/governance_audit_read.py`) | `RuntimeInspectionGovernanceSection` |
| Continuation lifecycle | `ExecutionContinuationStateStore` | `ExecutionContinuationSnapshotReadPort` (`intergrax/contracts/execution_continuation_read.py`) | `RuntimeInspectionContinuationSection` |

### Federation

`FederatedRuntimeInspectionReadService` accepts optional `tool_reader`, `governance_reader`, and `continuation_reader` (explicit injection only). Optional source failure domains: `tool_runtime`, `governance`, `continuation`.

### Partial / empty semantics

- **Empty + `source_available=True`:** reader succeeded, zero records.
- **Unavailable:** typed `RuntimeInspectionSourceFailure` on reader failure; snapshot section omitted.
- **Integrity:** wrong `ExecutionId` / tenant on facts → `INTEGRITY` failure (fail closed).

### Security

Tool/governance/continuation sections expose safe summaries, digests, and evidence refs only (no raw args, policy secrets, or human response bodies).

### Qualification gates

`tests/qualification/inspect_01/` — **B-Q1..B-Q20** (see `catalog.py` `INSPECT_01_B_Q_CATALOG`). **A-Q1..A-Q15** remain required regressions.

## INSPECT-01-B-C1 — Governance Tenant & Identity Integrity Closure

**Task intake `origin/development`:** `856683946abbaec313892cf61726f33e7c31eeac`  
**Closeout HEAD:** recorded at INSPECT-01-B-C1 commit on `development`.

### Root cause

`GovernanceAuditReadPort.list_audit_events_for_execution` accepted `tenant_id` but `InMemoryGovernanceAuditReadAdapter` filtered only by `ExecutionId`. `GovernanceAuditInspectionAdapter` validated execution id only, not the full identity spine.

### Tenant provenance

Canonical `tenant_id` on immutable `GovernanceAuditEvent`, recorded at audit write time from `ToolAuthorizationRequest.agent.tenant_id` (`GovernanceAuditRecorder`). No inference from agent naming or global registries.

### Identity validation

Governance facts enter inspection only when `tenant_id`, `task_id`, `run_id`, `attempt_id`, and `execution_id` match `RuntimeInspectionExecutionScope`. Tenant collision on matching `ExecutionId` raises tenant boundary (fail closed). Identity mismatch raises `SOURCE_INTEGRITY`. Federation re-raises governance integrity/tenant errors; availability failures remain `PARTIAL`.

### Legacy semantics

Events without recorded tenant provenance are unsupported for tenant-safe inspection (constructor requires `tenant_id`; no default/unknown placeholders).

### Qualification gates

**C1-Q1..C1-Q15** in `catalog.py` `INSPECT_01_C1_Q_CATALOG` (`test_inspect_01b_c1_governance_integrity.py`).

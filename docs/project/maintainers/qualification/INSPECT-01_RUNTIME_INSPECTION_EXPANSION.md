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

## INSPECT-01-C — Extended Domain Read Adoption

**Status:** **INSPECT-01-C: CLOSED** (INSPECT-01-C-FINAL qualification on `development`).

**Qualified HEAD:** `e6f005f328b8ab9f7d822a614855ed9be8936168` (`origin/development` at closeout).

**Remediation lineage:** INSPECT-01-C-R1 / INSPECT-01-C-R1A — identity provenance, truncation/completeness, cross-tenant integrity (last audited fix `ebcabb28a38d1d51d55fa2568ba8e2a93490f6ba`).

**Next recommended task:** **INSPECT-01-RQ** — enterprise requalification of the full Runtime Inspection surface (not a scope expansion of C).

### Domain read ownership

| Domain | Canonical owner | Read contract | Inspection section |
| --- | --- | --- | --- |
| Memory usage facts | Task memory / runtime spine | `MemoryRuntimeOperationReadPort` (`intergrax/contracts/memory_runtime_read.py`) | `RuntimeInspectionMemorySection` |
| Model invocation facts | LLM runtime spine | `ModelRuntimeInvocationReadPort` (`intergrax/contracts/model_runtime_read.py`) | `RuntimeInspectionModelSection` |
| External work boundary facts | External operations spine | `ExternalWorkRuntimeFactReadPort` (`intergrax/contracts/external_work_runtime_read.py`) | `RuntimeInspectionExternalWorkSection` |
| Artifact metadata | Harness artifact refs on spine | `ExecutionArtifactMetadataReadPort` (`intergrax/contracts/execution_artifact_read.py`) | `RuntimeInspectionArtifactSection` |

Default federation adapters project spine events via `ExecutionReconstruction` (`Reconstruction*Reader` under `intergrax/runtime/runtime_inspection/adapters/`). Custom implementations plug in through the same read ports and `RuntimeInspection*ReadPort` facades.

### Tenant / identity / integrity

Spine events must match the full `RuntimeInspectionExecutionScope` identity spine (`validate_spine_event_scope`). Cross-tenant spine facts raise `TENANT_BOUNDARY`. Missing spine `tenant_id` provenance raises `SOURCE_INTEGRITY` (fail closed — no default/unknown tenant placeholders).

Canonical domain read records (`MemoryRuntimeOperationRecord`, `ModelRuntimeInvocationRecord`, `ExternalWorkRuntimeFactRecord`, `ExecutionArtifactMetadataRecord`) carry full execution provenance (`tenant_id`, `task_id`, `run_id`, `attempt_id`, `execution_id`). Adapters validate each record with `validate_domain_record_scope` against the shared `ExecutionScopeIdentity` contract. Cross-tenant records raise `TENANT_BOUNDARY`; any other identity mismatch raises `SOURCE_INTEGRITY` (never downgraded to `PARTIAL` / `UNAVAILABLE` / empty). Malformed external-operation typed payloads raise `SOURCE_INTEGRITY`.

Custom implementations qualify at the canonical read ports (`MemoryRuntimeOperationReadPort`, `ModelRuntimeInvocationReadPort`, `ExternalWorkRuntimeFactReadPort`, `ExecutionArtifactMetadataReadPort`) wired through the inspection adapters.

### Truncation / completeness

`is_truncated=True` maps to section `PARTIAL`; `is_truncated=False` maps to `COMPLETE` (`completeness_for_read_result`) — independent of whether any records were returned. **C-Q15** exercises truncation/completeness through `MemoryOperationInspectionAdapter`, not projection-only ordering checks.

`scope.attempt_id is None` remains valid for execution-level inspection; when scope specifies an `attempt_id`, records with a different attempt raise `SOURCE_INTEGRITY`.

### Availability / empty

Empty sections with `source_available=True` when the reader succeeds with zero records. Reader failures emit `RuntimeInspectionSourceFailure` and optional-domain `PARTIAL` completeness.

### Security

Sections expose safe summaries, refs, and token counts only — no memory payloads, prompts, artifact URIs, or vendor error bodies.

### Qualification gates

**C-Q1..C-Q18** in `catalog.py` `INSPECT_01_C_Q_CATALOG` (`test_inspect_01c_extended_domains.py`). Additional **C-R1** gates (`test_c_r1_*` in the same module) cover post-audit provenance and custom canonical port wiring. **A-Q***, **B-Q***, and **C1-Q*** catalogs remain required regressions.

### Conscious limits

- External work inspection reflects spine-recorded external **operation** failures, not live vendor polling or full `ExternalWorkSnapshot` lifecycle stores (see BG-01 / integrations).
- Artifact metadata is derived from spine `artifact_refs` / `artifact_id` registrations, not binary artifact store reads (see ART-01).

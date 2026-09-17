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

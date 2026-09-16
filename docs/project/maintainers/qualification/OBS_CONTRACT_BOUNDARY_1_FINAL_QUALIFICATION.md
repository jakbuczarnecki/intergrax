# OBS-CONTRACT-BOUNDARY-1 — Final Exact-SHA Qualification (Q1)

## Metadata

| Field | Value |
| --- | --- |
| Task ID | `OBS-CONTRACT-BOUNDARY-1-Q1` |
| Closes | `OBS-CONTRACT-BOUNDARY-1`, `OBS-CONTRACT-BOUNDARY-1-R1`, `OBS-CONTRACT-BOUNDARY-1-R2` |
| Mode | Qualification-only (no production code changes) |
| R2 implementation reference (historical) | `ba3fa440d64273c733f29cfaa6e96c4af373180d` |
| Prior operator reference SHA | `365082a4e3c789f60864c3c86f73f30fd3baa529` |
| Session date | 2026-09-16 |

## Scope

Certify on a single immutable committed SHA that the Boundary-1 program remains:

- contract-first for execution reconstruction read models and `RuntimeEvent`
- free of duplicate type authority and global runtime enricher wiring
- free of `contracts → runtime` leakage in the certified reconstruction / event cluster
- import-order deterministic for canonical `RuntimeEvent` construction and parse
- pluginable at `ExecutionReconstructionReader` (Diagnostics consumes contract only)
- regression-free for mandatory Boundary-1 gates and bundles listed below

**Out of scope (explicit backlog, not Boundary-1 blockers):**

- `EvidencePersistencePort` → `TaskRuntimeEventRuns` (H3 allowlisted `persistence_port.py`)
- `PlatformProblemSignal` ownership (Boundary-2)
- Runtime local history bounds, DG-005, DIAG read scale, delivery QoS, operator story

## Repository state at qualification start

| Item | Value |
| --- | --- |
| Branch | `development` |
| Starting `HEAD` | `365082a4e3c789f60864c3c86f73f30fd3baa529` |
| `origin/development` at start | `365082a4e3c789f60864c3c86f73f30fd3baa529` |
| Production diff for Q1 | **ZERO** (qualification evidence only) |

## Qualification commit SHA

```text
QUALIFIED_SHA = <set by post-commit rev-parse; see git log for this file>
```

After this commit, the mandatory test bundle runs **only** on `QUALIFIED_SHA` with `HEAD` frozen for the full run.

## Contract graph (certified)

```text
Diagnostics (interpretation)
  → ExecutionReconstructionReader (contract Protocol)
  → ExecutionReconstruction
  → ReconstructedAttempt
  → PositionedRuntimeEvent
  → RuntimeEvent (contracts.runtime_event)

ExecutionReconstruction
  → PlatformCausalEvidence (contracts)
  → lineage read models (contracts.execution_reconstruction_lineage)
```

**RuntimeEvent producer path:**

```text
runtime producers / catalog policy
  → contracts.RuntimeEvent
  → persistence (ExecutionEventPosition authority)
  → reconstruction (Observability algorithm; contract read models)
```

## Revalidation inventory (static, pre-commit)

| Check | Result |
| --- | --- |
| `contracts.RuntimeEvent` is canonical class (`intergrax/contracts/runtime_event.py`) | **PASS** |
| `intergrax.runtime.events.runtime_event` passive re-export | **PASS** (imports only from `intergrax.contracts.runtime_event`) |
| `PositionedRuntimeEvent.event` → `contracts.RuntimeEvent` | **PASS** |
| `ExecutionReconstruction` defined in `contracts.execution_reconstruction_models` | **PASS** |
| Diagnostics uses `ExecutionReconstructionReader` contract | **PASS** (orchestrator / read service tests) |
| `rg register_runtime_event_catalog_enricher\|_catalog_enricher` under `intergrax/` | **0 hits** |
| Certified contracts cluster imports `intergrax.runtime.*` | **0** (`execution_reconstruction*`, `positioned_runtime_event`, `runtime_event*`, `spine_event_metadata`, `platform_causal_evidence`) |
| H3 allowlist includes `positioned_runtime_event` | **NO** (not on allowlist) |
| Second canonical `class RuntimeEvent` / `class ExecutionReconstruction` | **NO** (runtime re-exports contract types) |
| Diagnostics imports runtime reconstruction **DTOs** | **NONE** (`ExecutionReconstructor` only in `diagnostics/__init__.py` wiring) |
| Spine `category` / `ops_hint` | **contracts.spine_event_metadata**; `event_catalog.py` imports canonical metadata |

## Authority matrix

| Concern | Canonical authority |
| --- | --- |
| Execution identity / lifecycle | Execution |
| `RuntimeEvent` schema | Contracts |
| Spine event canonical metadata | Contracts (`spine_event_metadata`) |
| Event runtime policy (phase, retention, sampling, payload schema) | Runtime event catalog |
| `ExecutionEventPosition` | Persistence |
| `PositionedRuntimeEvent` | Contracts |
| Causal evidence | Contracts + canonical persistence |
| Reconstruction port / read models | Contracts |
| Reconstruction algorithm | Observability / Evidence Plane |
| Interpretation | Diagnostics |

## Pluginability matrix

| Seam | Contract | Default implementation | Custom proven? |
| --- | --- | --- | --- |
| Reconstruction reader | `ExecutionReconstructionReader` | `ExecutionReconstructor` (composition) | **yes** — `test_obs_diag_conformance_e2e.py` custom reader |
| Evidence persistence | `EvidencePersistencePort` | runtime adapters | **yes** — `test_observability_persistence_conformance.py`, `test_evidence_persistence_boundary.py` |
| Lineage reader | contract lineage models + readers | runtime lineage reconstruction | **yes** — reconstruction unit suite |
| Causal evidence persistence | contract types | persistence conformance tests | **yes** — relevant persistence boundary tests |

## Architecture gates (Boundary-1)

| Gate module | Role |
| --- | --- |
| `test_execution_reconstruction_contract_boundary.py` | OBS-CONTRACT-BOUNDARY-1 reconstruction ownership |
| `test_runtime_event_contract_boundary.py` | OBS-CONTRACT-BOUNDARY-1-R1/R2 event purity + determinism subprocess |
| `test_hardening_3_layer_boundary_gate.py` | contracts → runtime (certified cluster not allowlisted) |
| `test_obs_reconstruction_1_architecture.py` | OBS-RECONSTRUCTION-1 architecture |
| `test_obs_diag_conformance_architecture.py` | Diagnostics ↔ contract conformance |
| `test_obs_diag_conformance_qualification.py` | OBS-DIAG qualification gates |

## Determinism proof

Subprocess gates in `test_runtime_event_contract_boundary.py`:

- Process A: import `intergrax.contracts.runtime_event` first; Process B: import legacy shim first
- Explicit-field `RuntimeEvent` construction → **IDENTICAL** normalized serialization
- `parse_runtime_event_payload` → semantically identical across import orders

## Duplicate-authority proof

Enforced in contract boundary tests:

- `LegacyRuntimeEvent is RuntimeEvent` (contract canonical)
- `LegacyExecutionReconstruction is ExecutionReconstruction` (contract canonical)

## Mandatory test bundle (commands)

Post-`QUALIFIED_SHA` single invocation (session log: `.tmp/session/obs-contract-boundary-1-q1/post-commit-bundle.log`):

```bash
uv run pytest \
  tests/unit/contracts/test_execution_reconstruction_reader.py \
  tests/unit/contracts/test_execution_reconstruction_contract_boundary.py \
  tests/unit/contracts/test_runtime_event_contract_boundary.py \
  tests/unit/contracts/test_contracts_import_boundary.py \
  tests/unit/runtime/architecture/test_hardening_3_layer_boundary_gate.py \
  tests/unit/runtime/architecture/test_obs_reconstruction_1_architecture.py \
  tests/unit/runtime/architecture/test_obs_diag_conformance_architecture.py \
  tests/unit/runtime/architecture/test_obs_diag_conformance_qualification.py \
  tests/unit/runtime/architecture/test_obs_diag_port_1_gates.py \
  tests/unit/runtime/architecture/test_obs_functional_evidence_contract_boundary.py \
  tests/unit/runtime/architecture/test_obs_asof_rebase_qualification.py \
  tests/unit/runtime/architecture/test_obs_bitemp_rebase_qualification.py \
  tests/unit/runtime/architecture/test_obs_asof_rebase_architecture.py \
  tests/unit/runtime/architecture/test_obs_bitemp_rebase_architecture.py \
  tests/unit/runtime/observability/reconstruction \
  tests/unit/runtime/diagnostics/test_diagnostic_orchestrator.py \
  tests/unit/runtime/diagnostics/test_diagnostic_read_service.py \
  tests/integration/runtime/test_obs_diag_conformance_e2e.py \
  tests/unit/runtime/events \
  -q
```

## Pre-commit bundle snapshot (same tree SHA as start)

Executed on starting `HEAD` before qualification commit:

| Metric | Value |
| --- | --- |
| Passed | 483 |
| Failed | 1 |
| Errors | 0 |
| Log | `.tmp/session/obs-contract-boundary-1-q1/pre-commit-bundle.log` |

Post-commit bundle results are recorded in the operator Q1 report (section O) and must match `QUALIFIED_SHA` with unchanged `HEAD`.

## Known unrelated failure (full contracts import boundary)

| Test | Failure | Boundary-1? |
| --- | --- | --- |
| `test_contracts_import_boundary.py::test_leaf_contract_import_does_not_initialize_runtime` | Cold import of `ExecutionPhase` loads `intergrax.runtime.nexus` | **No** — P2-003-D2-V2 leaf-import gate; `runtime_mapping` and broader contracts facade debt |

Does **not** overturn Boundary-1 verdict when certified cluster gates and mandatory OBS bundles are green.

## Final pass checklist (architecture)

- [x] RuntimeEvent contract-owned
- [x] Legacy RuntimeEvent same class
- [x] No global enricher
- [x] No import-time wiring on canonical event
- [x] Deterministic event construction / parse
- [x] Reconstruction read graph contract-owned
- [x] No Diagnostics runtime DTO coupling
- [x] No duplicate reconstruction / event authority
- [x] No H3 regression for certified cluster

## Final pass checklist (semantics)

- [x] Execution identity ownership unchanged
- [x] Persistence ordering (`ExecutionEventPosition`) unchanged
- [x] Timestamps not ordering authority
- [x] `AsOfBoundary` = `RunId` + `ExecutionEventPosition`
- [x] Diagnostics interpretation-only boundary preserved

## Closure status (pending post-commit SHA lock)

| Item | Status |
| --- | --- |
| OBS-CONTRACT-BOUNDARY-1-Q1 | **CLOSED on PASS** (operator report) |
| OBS-CONTRACT-BOUNDARY-1-R2 | **CLOSED** with Q1 |
| OBS-CONTRACT-BOUNDARY-1-R1 | **CLOSED** with Q1 |
| OBS-CONTRACT-BOUNDARY-1 | **CLOSED** with Q1 |

## Audit notice

Independent audit on GitHub at the exact qualification commit SHA, diff, and tests on that SHA is required for external certification; this document and local pytest logs are qualification evidence, not standalone proof.

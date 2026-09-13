# ADR-RAG-001: Generic Multi-Channel Retrieval Coordination

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-09-13 |
| **Deciders** | Platform / RAG architecture |
| **Related** | [`multichannel_retrieval_coordination.md`](../../platform/multichannel_retrieval_coordination.md) · VPI `PLATFORM_CAPABILITY_MAPPING.md` · P1A `dc9ed5cc8558f08cb3db331fac23418d74563063` · P1A-R1 `08bb4c6edc7a547aaff1cb7758269b9d6bf1edf9` |

## Context

The **Verified Product Identification (VPI)** enterprise scenario must identify the exact product among near matches, incomplete or conflicting input, and abstain when evidence is insufficient. That requires **independent retrieval channels**—exact identifier lookup, lexical search, structured catalog search, and vector similarity—as **peer inputs** with per-channel success, skip, and controlled failure semantics. A single unified retrieval path cannot express this model.

Platform `RetrievalService` and `RetrieverRegistry` (`intergrax.rag.retrieval`) model **document-oriented RAG retrieval** (registry of retrievers, unified trace). They do **not** model ordered execution of scenario-defined peer catalog channels with scenario-owned `TResult` and fusion policy downstream.

Implementing another VPI-local orchestration framework would duplicate a **reusable platform gap**: generic **ordered N-channel coordination** with typed envelopes and deterministic semantics. Platform-first architecture requires the Harness to own that coordination contract; scenarios own catalog semantics, channel policy, and fusion.

## Decision

### Platform owns

- Channel execution envelope (`RetrievalChannelOperation` protocol)
- Typed channel identity (`RetrievalChannelKey` value object—not a platform enum of channel kinds)
- Typed status (`RetrievalChannelStatus`)
- Typed controlled failure (`RetrievalChannelFailure` with canonical `failure_code`)
- Outcome envelope (`RetrievalChannelOutcome[TResult]`) and aggregate (`MultiChannelRetrievalResult[TResult]`)
- Deterministic orchestration (default: `SequentialMultiChannelRetrievalCoordinator`)
- Execution plan validation (duplicate declared channel keys rejected before execution)
- Public coordinator contract (`MultiChannelRetrievalCoordinator[TResult]` protocol)

Module: `intergrax.rag.retrieval.multichannel`.

### Scenario owns

- Product / catalog semantics and channel-selection policy
- Query and result DTOs (`TResult` and request shapes)
- GTIN, MPN, product meaning, offer grain
- Provider-neutral catalog port adapters
- Fusion policy (e.g. RRF at offer level)
- Fatal-vs-tolerant business policy after aggregation

### Pluginability

Scenarios depend on `MultiChannelRetrievalCoordinator[TResult]` (protocol) and inject an implementation (default sequential coordinator). Scenario behavior remains pluggable via operation implementations and policy. **No VPI-specific types** appear in platform contracts.

### Invariants

1. Duplicate declared channel keys in the execution plan fail **before** any channel runs.
2. `operation.channel_key` is **authoritative** for plan identity.
3. `outcome.channel_key` **must equal** `operation.channel_key`.
4. Identity mismatch is a **contract violation** (`MultiChannelRetrievalContractError`), **not** a controlled channel `FAILED` outcome.
5. A controlled `FAILED` channel outcome does **not** automatically abort later channels.
6. An **empty** execution plan (`operations=()`) is valid and yields an empty aggregate result.
7. `failure_code` is the canonical machine-readable identity: non-empty and already trimmed.
8. `message` remains human-readable diagnostic text (separate from `failure_code`).

### Non-goals (this ADR / P1A scope)

- Rank fusion (RRF) or consolidation of fusion math
- Retries, concurrency, or timeout policy inside the coordinator
- Product or catalog DTOs in platform contracts
- Provider SDK execution inside the coordinator
- Diagnostic spine / `RetrievalTrace` projection (**planned P1B**)
- Decision System mapping for retrieval outcomes
- VPI runtime migration to the platform coordinator (**adoption pending**)
- Physical plugin registry registration unless independently justified

### Platform evolution consequence

This capability exists because a **real enterprise scenario** exposed a **reusable platform gap**. That is intentional Harness evolution: scenario requirements drive platform design, not the reverse.

### Integration positioning (truthful as of P1A-R1)

```text
Execution Engine / runtime composition  (future wiring)
              |
              v
     scenario execution (e.g. VPI pipeline)
              |
              v
platform retrieval capability
  MultiChannelRetrievalCoordinator + envelopes
              |
              v
scenario retrieval policy / plugins / MultiChannelRetrievalPort
              |
              v
provider-neutral catalog adapters
```

| Concern | State |
| --- | --- |
| Platform capability (contract + default coordinator) | **IMPLEMENTED** |
| VPI adoption (delegate scenario orchestrator) | **PENDING** |
| Execution Engine integration | **PENDING** (later scenario integration) |
| Diagnostic projection | **PENDING** (P1B) |
| Governance integration | Evaluate when VPI E2E execution boundary is wired |
| Plugin system | Logical contract/DI now; physical plugin registration only where justified |

P1A does **not** claim completed Execution Engine wiring, governance hooks, or observability projection.

## Consequences

### Positive

- One auditable platform contract for multi-channel retrieval orchestration reusable beyond VPI.
- Clear scenario/platform ownership; no shadow coordinator in scenario core.
- Deterministic, testable semantics without domain leakage.

### Negative

- VPI still uses scenario-local `MultiChannelRetrievalService` until migration (adoption task).
- Follow-up ADR/work (P1B diagnostics, P1C RRF consolidation) remains on the roadmap.

## Compliance

- Tier boundaries preserved: `intergrax.rag.retrieval.multichannel` has no imports from `platform_proofs` or applications.
- Platform contracts use frozen dataclasses, protocols, and explicit errors—no untyped dict envelopes.
- Companion doc: [`multichannel_retrieval_coordination.md`](../../platform/multichannel_retrieval_coordination.md).

## Implementation notes

- Code: `intergrax/rag/retrieval/multichannel/contracts.py`, `coordinator.py`, `errors.py`.
- Tests: `tests/unit/rag/retrieval/test_multichannel_retrieval_coordinator.py`, `test_multichannel_retrieval_architecture_gate.py`.
- Verification: `python scripts/maintenance/check_harness_adr.py`; `uv run pytest tests/unit/rag/retrieval/test_multichannel_retrieval_coordinator.py`.

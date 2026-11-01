# ADR-EBH-7 — INTERNAL: PROD-Q colocated runtime adapter isinstance gate

| Field | Value |
| --- | --- |
| **Status** | Ratified @ operator EBH-7 terminal authorization (`fd3bc0104`) |
| **Concern** | `prodq.colocated_runtime_adapter_isinstance` |
| **Classification** | `INTERNAL IMPLEMENTATION — EXTENSIBILITY NOT CLAIMED` |

## Context

PROD-Q acceptance uses an `isinstance` gate on the colocated runtime deployment adapter. This is a qualification acceptance mechanism, not a published vendor interchangeability contract.

## Decision

- RPL-H for `prodq.colocated_runtime_adapter_isinstance` is **INTERNAL**.
- Evidence: adapter module existence + ADR + terminal INTERNAL RPL gates.
- Replacing the adapter for true vendor neutrality is out of scope for this concern id.

## Consequences

- B6 closure uses `INTERNAL_RATIFIED` normative pathway, not dual-provider RPL-G2 fiction.

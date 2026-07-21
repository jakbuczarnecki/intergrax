# ADR-EXECUTION-BOUNDARY-EVENT-001: Governed execution boundary event (host-owned)

| Field | Value |
|-------|-------|
| **Status** | Accepted |
| **Date** | 2026-07-20 |
| **Deciders** | Platform / Execution Evidence |
| **Related** | ADR-GOVERNED-PROOF-001 · harness EBE-9 · docs/platform/execution_evidence_and_host_attestation.md |

## Context

Governed External Execution ends at `GovernedProofProfile`. Partner validation needs a host-owned event that joins policy decision, provider invocation, execution identity, and proof after a successful side effect.

An existing harness export `ExecutionBoundaryEventV1` (`execution_boundary_event.v1`) already serves **tool/step** BoundaryAttest export under `intergrax.runtime.attestation`. Its required shape (`boundary_type` tool/harness_step, `tool_id`, `agent_id`, `step_id`, free-form input/output) is **not** semantically equivalent to a governed side-effect boundary.

## Decision

1. Introduce a **new** host-owned contract `ExecutionBoundaryEvent` with schema id `governed_execution_boundary_event.v1` under `intergrax.contracts.execution_evidence`.
2. Do **not** overload or break harness `ExecutionBoundaryEventV1` / BoundaryAttest golden vectors.
3. Host composes the event only after successful provider execution **and** successful `GovernedProofProfile` composition.
4. Event never authorizes or resumes execution; it describes a completed boundary.
5. Reuse existing task/run/correlation/idempotency identities — never mint a replacement `run_id`.
6. Reuse `canonical_json` + host attestation machinery for signing (see ADR-HOST-ATTESTATION-001).

Rejected: stuffing governed facts into harness EBE `input`/`output` dicts (loses typed invariants; conflates observability planes).

## Consequences

### Positive

- Clear ownership: Tier-2 proof → host event → attestation
- BoundaryAttest harness path remains stable

### Negative

- Two boundary-event schemas exist; partner docs must cite the governed schema id

## Compliance

- No secrets/credentials/transport responses in the event
- Tier-2 never composes or signs the event

## Implementation notes

- `intergrax/contracts/execution_evidence/boundary_event.py`
- Host composer in `applications/governed_contractor_application/host/`

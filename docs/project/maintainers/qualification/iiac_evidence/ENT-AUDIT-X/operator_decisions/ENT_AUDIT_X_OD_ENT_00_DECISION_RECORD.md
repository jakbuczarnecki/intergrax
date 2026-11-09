# ENT-AUDIT-X OD-ENT-00 — accept EVOL-X formal predecessor status

**Decision ID:** OD-ENT-00  
**Stage:** ENT-AUDIT-X  
**Policy version:** IIAC-2026-11-05.1  
**Record origin:** operator-delegate (designated signing session `ent-sign-w3-2026-11-09`)

## Decision

Accept EVOL-X formal closure and ENT handoff as the satisfied predecessor for ENT-AUDIT-X formal closure chain (**BLK-ROADMAP-ENT-DEP** / cross-stage matrix). This does not close ENT-AUDIT-X, does not promote FRZ rows, and does not waive any IIAC gate.

## Evidence basis (informational)

| Pin | SHA |
|-----|-----|
| EVOL formal closure | `be77b3a5fb97f10a3bd2df6ee038cb92aad77eca` |
| EVOL → ENT handoff | `7a4f279fd8c0986431941c77283d872ea5b0c1ba` |
| EVOL IIAC B6 acceptance | `1dd769331506149a1ee2f411c2bc62e85e5bbf25` |
| D10 attestation | PASS dated 2026-11-08 (process_enforced) |
| Canonical matrix | `docs/project/maintainers/qualification/ENT_EVOL_CROSS_STAGE_DECISION_MATRIX.md` |

FRZ-EVOL-01..10 already promoted in checklist; limitations LIM-01..07 carry into ENT handoff per W0-03.

## Signature (S-2)

Approved OD-ENT-00: operator-delegate @ 2026-11-09T16:20:00+01:00

Delegation: docs/project/maintainers/qualification/iiac_evidence/ENT-AUDIT-X/operator_decisions/ENT_AUDIT_X_O1_SIGNING_DELEGATION_RECORD.md@c7d18b61bb837a2708955da62adbe538b65e82bd

## Assurance

`process_enforced` — operator chat commission + designated session dedication; no cryptographic proof of operator identity.

# ADR-PDC-08 — Artifact taxonomy with per-class owners, mandatory digest, governed access and enforced retention

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-08 |
| **Status** | **Accepted** (OD-PDC-ADR-01, 2026-11-12) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D8 Artifact Lifecycle |
| **Requirements** | PDC-D8-R001..R009, R012, R013, R016..R018 (and PDC-D6-R022) — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-OWN-01, FRZ-OWN-02, FRZ-OWN-05, FRZ-SEC-09, FRZ-AUD-05, FRZ-REC-10, FRZ-TEN-04 |

## Context

- Artifacts were never certified (ART-01 handoff to STATE-X dropped). Execution `ArtifactStore` is in-memory only and explicitly has no security; retention policy fields have zero consumers; no spill mechanism; `task_artifacts.py` and `application_artifacts.py` are identical; three types are named `ArtifactRef`.
- Principles §14/§15: artifact classes have different owners, so this is a **feature** coordinating domain owners, not a new domain (PDC-P1-F03). GAP_AUDIT §16: evidence plane PRESERVE/EXTEND.

## Decision (proposed)

- Four classes, four owners, no global storage engine: business artifacts → Collaborative Work; execution/audit evidence → evidence plane; context artifacts → UCL; temporary tool output → tool runtime with spill-to-reference.
- A shared minimal reference contract: content digest **mandatory**, producer identity and execution lineage, tenant/workspace scope, retention class.
- Governed read port in front of every store (authorization + tenant check + audit event); stores stay storage-only.
- Durable execution artifact store through the canonical persistence plane; retention/TTL enforced by a sweep admitted as background work; audit evidence and legal hold are never deletable by compaction or TTL.
- Remove the duplicate `applications/contracts/application_artifacts.py` under COMPAT-X R5 shim policy; rename trace aliases so only one `ArtifactRef` remains.
- Feature doc pair `docs/project/capabilities/{architecture,plan}/ARTIFACT_LIFECYCLE.md`, registered in the runtime hub and `capabilities/README.md`, created **after** OD-PDC-ADR-01 acceptance (PDC-P3-06).
- Governed secure download is decided by OD-PDC-SCOPE-04.

- **OD-PDC-SCOPE-04 = A (implement):** governed secure download = governed read port + short-lived provider URL; no presigned URL is issued outside the governed port.

## Alternatives rejected

- One global artifact storage engine (merges semantics for elegance; forbidden by the campaign mandate).
- Security logic inside store implementations.

## Verification

`tests/qualification/pdc_x/d8/` plus `xd_09`, `xd_10`.

## Status note

Accepted by OD-PDC-ADR-01 (Operator, 2026-11-12, recorded under T-IGX-036-E). This authorizes PDC-P4 design. Product merges still follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8), which is pending.

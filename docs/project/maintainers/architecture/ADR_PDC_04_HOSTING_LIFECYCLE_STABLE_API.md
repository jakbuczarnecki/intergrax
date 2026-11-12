# ADR-PDC-04 — Hosting lifecycle qualification, metrics and stable public API

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-04 |
| **Status** | **Accepted** (OD-PDC-ADR-01, 2026-11-12) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D4 Host Runtime |
| **Requirements** | PDC-D4-R003..R007, PDC-D4-R009, PDC-D4-R010, PDC-D4-R014 (APP-HOST-3D, 5D, 5E, 9F) — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-REL-04, FRZ-REL-10, FRZ-OBS-01, FRZ-OBS-05, FRZ-REC-01, FRZ-REC-02, FRZ-CTR-04, FRZ-DOC-05 |

## Context

- `tests/qualification/host_01` qualifies the host execution boundary, not the APP-HOST lifecycle; hosting emits no metrics; restart digest preservation is proven only on the LKW path; APP-HOST-9F (stable API) is Planned.
- GAP_AUDIT §16: ApplicationEnvironmentProfile authority PRESERVE (PDC-P1-F05).

## Decision (proposed)

- Add a generic APP-HOST lifecycle qualification suite: startup ordering + reverse rollback, readiness/degraded, drain, instance guard, and an OS-subprocess crash/restart proof (APP-HOST-5E) using the existing `HostedApplicationSupervisor`.
- Hosting lifecycle metrics (start, ready, degraded, drain duration, restart count, exit class) are emitted through the canonical observability/event port; hosting does not own a metrics backend (APP-HOST-3D).
- Restart preserves and verifies the configuration/profile digest for every hosted application, not only LKW (APP-HOST-5D).
- Declare the stable public hosting API as a manifest covered by COMPAT-X version policy (APP-HOST-9F).
- APP-HOST-6 (InteractionProfile), APP-HOST-7 (OS service adapters) and APP-HOST-9E (host plugins) are proposed as authorized post-freeze exclusions (OD-PDC-SCOPE-02): none is required for the declared supported lifecycle, and adding InteractionProfile now would compete with ApplicationEnvironmentProfile authority.

- **OD-PDC-SCOPE-02 = A (authorized exclusion):** APP-HOST-6 (InteractionProfile), APP-HOST-7 (OS service adapters) and APP-HOST-9E (host plugins) are outside the freeze scope and recorded as FRZ-DEBT-06 evidence-backed exclusions.

## Alternatives rejected

- A hosting-owned metrics store or a second composition root.
- Product (LKW) vocabulary in the generic `HostDeploymentProfile` (TL-FIX-C is corrected instead).

## Verification

`tests/qualification/pdc_x/d4/` plus `xd_04`/`xd_05` cross-domain scenarios.

## Status note

Accepted by OD-PDC-ADR-01 (Operator, 2026-11-12, recorded under T-IGX-036-E). This authorizes PDC-P4 design. Product merges still follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8), which is pending.

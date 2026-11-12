# ADR-PDC-06 — Fail-closed security defaults, single tool-input defense path and durable security audit

| Field | Value |
| --- | --- |
| **Document ID** | ADR-PDC-06 |
| **Status** | **Proposed** (pending OD-PDC-ADR-01) |
| **Campaign** | PDC-X P3 (T-IGX-036-D) |
| **Domain** | PDC-D6 Runtime Security |
| **Requirements** | PDC-D6-R007, R008, R013, R014, R020, R021, R023 (SEC-BND-04/05/06) — see [`PDC_DOMAIN_REQUIREMENTS_MATRIX.md`](../qualification/platform_domain_completeness/PDC_DOMAIN_REQUIREMENTS_MATRIX.md) |
| **FRZ** | FRZ-SEC-02, FRZ-SEC-03, FRZ-SEC-07, FRZ-SEC-08, FRZ-SEC-10, FRZ-CFG-02, FRZ-AUD-05 |

## Context

- `critical_action_signing_wiring.py:37` falls back to the literal `"harness-dev-signing-key"`.
- The RAG service reads `retrieval_poisoning_defense_enabled` with a fallback of False (fail-open when the profile is absent).
- `assert_tool_input_safe` has zero consumers; `SecurityAuditTrail` is in-memory only; no static raw-secret gate exists; no security architecture doc pair exists (broken `IDENTITY_TRUST.md` link).

## Decision (proposed)

- Missing signing secret fails closed outside lab profiles (typed configuration error); the literal fallback is removed.
- Retrieval-poisoning defense defaults to enabled when the profile is absent; disabling requires an explicit profile value.
- One tool-input defense path: the guard is either wired into the canonical tool invocation middleware or removed with evidence; no dead security claims.
- The security audit trail persists through the canonical evidence/persistence plane (append-only, tenant-scoped); no security-owned store.
- Add a static raw-secret gate plus a caplog/trace secret-leak test.
- Publish a consolidated security architecture satellite and responsibility matrix (encryption/residency) referencing GOVERNED_EXECUTION, INTEGRATIONS credentials, sandbox and the EE-B3 threat model — a satellite, not a new policy authority.

## Alternatives rejected

- A new policy engine or identity authority.
- Keeping the dev fallback behind an environment check inside product code.

## Verification

`tests/qualification/pdc_x/d6/` plus `xd_06`, `xd_10`.

## Status note

Proposed only. No implementation before OD-PDC-ADR-01 accepts this ADR; product changes then follow OD-PDC-SEQ-01 (merge after MCP-SERVER-X P8).

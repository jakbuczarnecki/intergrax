# EVOL-X W1 — Consolidated operator decision register

**Record status:** **Operator authorization package 2026-11-01** — see [`EVOL_X_OPERATOR_AUTHORIZATION_2026-11-01.md`](EVOL_X_OPERATOR_AUTHORIZATION_2026-11-01.md).

## W0 signature status

| ID | Status | Signed date | Notes |
|----|--------|-------------|-------|
| **OD-EVOLX-01** | SIGNED / APPROVED | 2026-11-01 | Batch B–F execution entry |
| **OD-EVOL-01** | SIGNED / APPROVED | 2026-11-01 | Option C @ baseline `884885774…` |
| **OD-EVOL-02** | SIGNED / APPROVED | 2026-11-01 | Policy (i) |
| **OD-EVOL-05a** | SIGNED / APPROVED | 2026-11-01 | Full assessor scope |
| **OD-EVOL-05b** | SIGNED / APPROVED | 2026-11-01 | **Option 1** qual-only; **no** W4-04 port |
| **OD-EVOL-05c** | SIGNED / APPROVED | 2026-11-01 | NA-01..04 |
| **OD-EVOL-06** | SIGNED / APPROVED | 2026-11-01 | Sidecar revision facts |
| **OD-EVOL-09** | SIGNED / APPROVED | 2026-11-01 | X-01,02,04,05; X-03,06,07 proofs in W2 |
| **OD-EVOL-10** | SIGNED / APPROVED | 2026-11-01 | Before W2-09 / W3-05 |
| **N-04 (OD) / N4-A** | SIGNED / APPROVED | 2026-11-01 | Migration-doc semver only |
| **OD-EVOL-03** | **PENDING** | — | W0-07 → W6-01 |
| **OD-EVOL-07** | **PENDING** | — | W0-07 → P-H |
| **OD-EVOL-08** | **PENDING** | — | W0-07 → M-09 SLA |
| **OD-EVOL-11** | **PENDING** | — | W0-08 → W7-01 (Batch G not in 2026-11-01 scope) |
| **OD-EVOL-12** | **PENDING** | — | W7-03 (Batch G not in 2026-11-01 scope) |
| **OD-EVOLX-08** | **PENDING** | — | W7-04 (Batch G not in 2026-11-01 scope) |

## Decision text (ratified where signed)

| ID | Proposed decision text (operator signs or edits) | Technical basis | Alternatives | W2–W6 impact | Authorizes code? | Before W2? | Before later wave |
|----|---------------------------------------------------|-----------------|--------------|--------------|------------------|------------|-------------------|
| **OD-EVOLX-01** | «Authorize EVOL-X stage entry at START_HEAD `5a4e922…`.» | Roadmap §3.0.9 mandatory stage; EBH-7 closed | Defer stage | Enables all waves | No | **Yes** | — |
| **OD-EVOL-01** | «Adopt placement **Option C**: `testing_support/change_safety/` + `tests/qualification/evol_x/adapters/`.» | Packaging + import boundaries | A/B/D per ADR | W2 paths | W2+ in Q/T only | **Yes** | — |
| **OD-EVOL-02** | «Workflow policy **(i)**: VERSIONED_DEPLOY + pin/checkpoint; at-rest profile MIGRATE; in-flight definition change **EXPLICITLY_UNSUPPORTED**.» | Pinning verified; no in-flight executor | (ii) build in-flight migration (Class C) | W4-02 matrix | No (policy) | **Yes** (for W4) | W4-02 |
| **OD-EVOL-05a** | «Require full EVOL deploy **assessor** scope (DM-01..07) in qualification harness.» | N-03 gap documented | Narrow FRZ-07 scope | W4-03, S7 | No (Q tooling) | No | W4 |
| **OD-EVOL-05b** | «**Option 1:** qualification-only FRZ-07 scope — **no** W4-04 production precondition port.» (Option 2 text retained for reference only) | S7 assessor ≠ production protection | Option 2: AD port default off | W4-04 **not authorized** | No (2026-11-01) | No | W4-04 blocked |
| **OD-EVOL-05c** | «Accept NA-01..04 out-of-scope register with evidence N-06/N-10.» | PROD-Q/DIST-SCALE live OPEN | Expand scope | S7 NA arms | No | No | W4/W6 |
| **OD-EVOL-06** | «Revision state facts via sidecar `schema_snapshot@revision` manifest (not RuntimeRevision fields).» | Avoid Class B unless needed | Add RuntimeRevision fields | W4-03 facts | Optional data files | No | W4-03 |
| **OD-EVOL-09** | «Accept exclusions X-01,02,04,05; require W2 proofs for X-03,06,07.» | W1 exclusion register | Shrink/expand universe | W2 CW-04 | No | **Yes** (proofs W2) | W2-05 |
| **OD-EVOL-10** | «Authorize: (1) `evol_x_p3_l3` profile in `unit-tests.yml`; (2) CONSERVATIVE_FULL = CI `dev-ci` manifest node list, not default local `-m gate` collect; (3) push workflow to `main` before GHA dispatch; (4) optional collect-only reruns under `dev-ci`.» | N-18 seven root causes; RC-7 baseline defect | Use broken local marker set | W2-09, W6-04 | Workflow YAML only | **Yes** (W2-09) | W6 |
| **OD-EVOL-12** | «At W7 promote FRZ-EVOL-08/10 to PASS with scoped footnotes (Option A).» | FRZ shared owners | Leave OPEN to ENT-AUDIT-X | W7-05 checklist | Docs only | No | W7 (**PENDING**) |
| **N-04 (OD)** | «Adopt **N4-A**: graph semver in migration docs only until optional N4-B approved.» | [EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md](EVOL_X_N04_GRAPH_VERSION_RECONCILIATION_W1.md) | N4-B Class B field | W4 WM rows | N4-B only | **Yes** (wording) | W4 if N4-B |

**OD-EVOL-11** (auditor appointment): **PENDING** — required before W7 independent B6; recommended before W6 evidence pack.

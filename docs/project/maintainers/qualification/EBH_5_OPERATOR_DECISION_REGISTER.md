# EBH-5 — Operator Decision Register

| ID | Decision | Status | Blocks |
| --- | --- | --- | --- |
| OD-EBH5-01 | Authorize EBH-5 implementation wave (`production delta > 0`) | **WAVE COMPLETE** @ publication commit | Formal roadmap **READY FOR AUDIT** |
| OD-EBH5-02 | Approve SAC-EBH-5-v1 freeze | **APPROVED 2026-10-28** — `OPERATOR APPROVED — EBH-5 SAC FREEZE` | — (SAC **FROZEN**) |
| OD-EBH5-03 | Allow L4 physical/live vendor tests | **OPEN** (default: deferred) | Optional QUALIFIED rows needing live brokers |
| OD-EBH5-04 | Runtime SQLite extensibility vs FIXED ratification | **APPROVED 2026-10-28** (see sub-rows) | — |
| OD-EBH5-04-A | Runtime SQLite persistence bundle (scoped FIXED) | **APPROVED 2026-10-28** — `OPERATOR APPROVED — OD-EBH5-04 RUNTIME SQLITE FIXED SUBSTRATE (SCOPED)` | Scope: `runtime-sqlite-persistence-bundle`; single-node/profile constraints preserved |
| OD-EBH5-04-B | Lab adaptive SQLite (lab-only FIXED) | **APPROVED 2026-10-28** — `OPERATOR APPROVED — OD-EBH5-04 LAB ADAPTIVE SQLITE FIXED SUBSTRATE (LAB ONLY)` | Scope: `lab-adaptive-sqlite`; H_REFERENCE / not production-effective default |
| OD-EBH5-04-C | Single-node profile v1 SQLite triad | **RATIFIED — SCOPED PROD-Q** (unchanged; prior qualification) | `platform.single_node_profile_v1.sqlite_triad` |
| OD-EBH5-05 | Accept DIST-SCALE historical vendor proofs @ scoped SHA | **RECOMMENDED** | message_bus L3 deferral |
| OD-EBH5-06 | PROD-Q F02-14 isinstance debt disposition | **TRACKED DEBT** | FRZ-RPL-04 global only |
| OD-EBH5-07 | Independent auditor + Protocol 03 | **OPEN** | CLOSED — INDEPENDENTLY ACCEPTED |
| OD-EBH5-08 | Roadmap §5 EBH-5 closure publication | **OPEN** | Post-audit only |

## Operator approval record (2026-10-28)

```text
OPERATOR APPROVED — EBH-5 SAC FREEZE
OPERATOR APPROVED — OD-EBH5-04 RUNTIME SQLITE FIXED SUBSTRATE (SCOPED)
OPERATOR APPROVED — OD-EBH5-04 LAB ADAPTIVE SQLITE FIXED SUBSTRATE (LAB ONLY)
```

Recorded by publication batch on `development` — resolve **OPERATOR_DECISIONS_PUBLICATION_SHA** in [`EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md`](EBH_5_CERTIFICATION_COMPLETION_RECONCILIATION.md).

Cursor/agent **STOP** before OD-EBH5-07/08 closure publication.

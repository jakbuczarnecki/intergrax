# ARCH-FREEZE ← HIST campaign handoff (predecessor readiness)

| Field | Value |
|---|---|
| From | Historical certification campaign |
| To | ARCH-FREEZE maintainers / `T-IGX-010-C` wave |
| Date | 2026-11-09 |
| Evidence SHA | POST_ACTIVATION `bf03550aa95ce239a63542916331c6a648617e33` |
| Predecessor verdict | **NOT READY** for inherited PASS (unchanged intent) |

## Successor to earlier handoff

This document **supersedes the status narrative only** in [`ARCH_FREEZE_HISTORICAL_PREDECESSOR_READINESS_HANDOFF.md`](ARCH_FREEZE_HISTORICAL_PREDECESSOR_READINESS_HANDOFF.md) (2026-11-08 pin `ba07ebaa3`). The 2026-11-08 structural inventory remains valid; **facts below are additive @ POST_ACTIVATION**.

## What improved since 2026-11-08 (HIST deliverables)

| Area | Before | After (HIST) |
|---|---|---|
| EBH-4 R1 / F-013 (`T-IGX-030-B`) | HELD red | **Implemented** in C1 @ integration SHA; P4 **PASS** |
| Invoker F-015 (`T-IGX-030-C`) | HELD | **Implemented** C2; P4 **PASS** |
| P2-C pin precedence F-015 (`032-A`) | Red / owner gap | **A2 resolver** C3; P4 **PASS**; durability residual tracked |
| Tier-0 CLI BND-03 primary edge (`033-A`) | Tier-0 stubs | **(b2)** launcher; P4 **PASS_SCOPED**; **HIST-F-BND-03B** still OPEN |
| `test_sec_ent` / W5 isolation (`031-A`) | Red | Tests-only merged; PASS_SCOPED audits |
| Dimension A replay | Not on final SHA | **595 passed** POST_ACTIVATION @ `bf03550aa` |
| IIAC on product merge | ACT-N11 red window | **IN_FORCE** after P5 activation |

## What ARCH-FREEZE still cannot inherit

1. **RPL-01..04** — independent VBN acceptance not run (external WBS).
2. **FRZ-BND-03 full PASS** — manifest registry edge **HIST-F-BND-03B** OPEN.
3. **Path B live rows** — PRD-01/02 etc. per Path B register.
4. **Population E REG rollup** — ENT-AUDIT-X owned.
5. **19 + 10 AUDIT_REQUIRED rows** — still need per-row independent audit / evidence gaps closed (see predecessor handoff §2).

## RPL / VBN

HIST **did not** execute `T-B7-VBN-INDEPENDENT-ACCEPT` or ARCH-FREEZE R1–R6. No FRZ row promoted to checklist PASS by this campaign.

## Recommended ARCH-FREEZE entry steps

1. Read [`HISTORICAL_CAMPAIGN_BLOCKER_DOSSIER.md`](HISTORICAL_CAMPAIGN_BLOCKER_DOSSIER.md).
2. Reconcile BND-03 disposition with **PASS_SCOPED** CLI portion + open manifest edge.
3. Schedule independent acceptance for RPL family when WBS unblocks.
4. Use POST_ACTIVATION SHA for replay comparisons, not `ba07ebaa3` alone.

## HIST claim boundary

HIST claims **scoped product corrective evidence** and **honest OPEN rows** only. ARCH-FREEZE **CLOSED** is **not** asserted by HIST.

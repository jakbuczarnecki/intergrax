# ASF-SEF-P1 — Operator acceptance record

> **Label:** **Operator acceptance record — not independent audit signature.**  
> **Signed by:** operator @ 2026-11-05T11:09:00+01:00 (Session E formal P1 closure).  
> **Independent auditor verdict:** cite separate session when available; this file does **not** substitute Protocol 03 or G6 certification.

## Identity

| Field | Value |
|-------|-------|
| **Campaign** | ASF-SEF-P1 (P1-B1 + corrective) |
| **Ticket** | [`governance/tickets/T-IGX-SEF-A.md`](../../../../../governance/tickets/T-IGX-SEF-A.md) |
| **START_HEAD** | `b342e5e7f56fe67aa4dcc7b7f828da5fd4f15bbe` |
| **P1-B1 publication SHA** | `e2f88cdf4f25d0f77642503731361f669b3f3200` |
| **AUDITED_SHA** | `7de6b41f637df1d6949985d9dc567602a074e230` |
| **FORMAL_CLOSURE_SHA** | `55fb0341672e37ae6da9b750874f1e27eb677090` |
| **Branch** | `development` |

## Scope accepted (documentation only)

P1-B1 + corrective batch: OD register OD-SEF-01/02/03, ADR-SEF-001 **`Proposed`**, architecture satellites, roadmap Option 3 coupling (no Option 4), ticket lifecycle formatting. **Excluded:** product CAP-049..052, ADR **`Accepted`**, FRZ/B6/ARCH-FREEZE CLOSED, P2–P13 engineering.

## Independent audit / dossier reference

| Artifact | Path |
|----------|------|
| Corrective implementer dossier | `.tmp/session/ASF-SEF-P1-CORRECTIVE/DOSSIER.md` |
| Closure prep dossier (ADR reconciliation) | `.tmp/session/ASF-SEF-P1-CLOSURE/DOSSIER.md` |
| ADR validator log @ AUDITED_SHA | `.tmp/session/ASF-SEF-P1-CLOSURE/check_harness_adr_audited.log` |
| ADR validator log @ Session E closure | `.tmp/session/ASF-SEF-P1-CLOSURE/check_harness_adr_closure.log` |

## Mechanical verification @ acceptance baseline

| Check | Result | Notes |
|-------|--------|-------|
| Relative links (7 P1 files + adr README) | **108 OK / 0 broken** | Corrective wave [`.tmp/session/ASF-SEF-P1-CORRECTIVE/check_links_after.log`](../../../../../../.tmp/session/ASF-SEF-P1-CORRECTIVE/check_links_after.log) |
| `verify_plan_approval.py` | **OK** | Session E — `.tmp/session/ASF-SEF-P1-CLOSURE/` |
| `verify_ticket_lifecycle.py` | **OK** | Session E — `.tmp/session/ASF-SEF-P1-CLOSURE/` |
| `igx_gov ticket` | **PASS** | Session E — `.tmp/session/ASF-SEF-P1-CLOSURE/igx_gov_ticket_closure.log` |
| `check_harness_adr.py` | **FAIL (24)** | **Disclosed** — global drift; **OD-SEF-P1-EXCEPTION**; **not** P1-introduced; **T-IGX-014-A** open |

## Findings disposition

| Area | Disposition |
|------|-------------|
| P1 governance content (OD-SEF-01/02/03, satellites, links) | **PASS WITH FINDINGS** — findings limited to pre-existing global ADR index validator drift |
| ADR-SEF-001 | **`Proposed`** only (OD-SEF-03) |
| Global ADR README / parser | **OUT OF SCOPE for P1** — **OD-SEF-P1-EXCEPTION** approved; hygiene **T-IGX-014-A** |

## Operator signature

**Status:** **ACCEPTED (PASS WITH FINDINGS — documentation scope)**

```text
Accepted: operator @ 2026-11-05T11:09:00+01:00 (ASF-SEF-P1 PASS WITH FINDINGS — documentation scope accepted @ AUDITED_SHA 7de6b41f637df1d6949985d9dc567602a074e230; OD-SEF-P1-EXCEPTION acknowledged; ADR-SEF-001 remains Proposed; not formal B6/FRZ/ARCH-FREEZE closure; P2–P13 not started)
```

# HIST-03 - Architecture vs Live scope matrix (40 rows)

Baseline `ba07ebaa34d04633bd60510f68c18346b1ef160d` - generated 2026-11-08T07:53:24Z. Basis: document-level inventory; **no test executed yet unless stated in STATUS.json**.

## Path B rule text (canonical, checklist L57-68)

> **Live qualification-scoped (Dimension B)** - Remains OPEN (or BLOCKED) until PROD-Q live slice / per-integration gate closes with exact-SHA evidence; mandatory before production execution or external side effects for that integration.
> **Live-scoped criteria (remain OPEN at architecture freeze until PROD-Q live gate):** FRZ-PRD-01, FRZ-PRD-02, and the live-environment / operator-production-profile obligations within FRZ-PRD-03, FRZ-PRD-08, and FRZ-SEC-04.
> `OPEN` = 0 and `BLOCKED` = 0 apply to architecture-scoped criteria only. This amendment does not change any criterion row to PASS.

Anything not in that list is **architecture-scoped (Dimension A)** and must reach PASS or N/A-WITH-EVIDENCE; a live-provider failure is never converted to an architecture PASS.

| FRZ | Scope | Primary | State | Architecture obligation | Live obligation (Path B, owner PROD-Q) |
|---|---|---|---|---|---|
| FRZ-OBS-01 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Architecture-scoped; Observability non-authoritative invariants. | None. |
| FRZ-OBS-02 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Architecture-scoped; Observability non-authoritative invariants. | None. |
| FRZ-OBS-03 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Architecture-scoped; Observability non-authoritative invariants. | None. |
| FRZ-OBS-04 | A | SSOT_DRIFT | AUDIT_REQUIRED | Architecture-scoped; completeness previously deferred to TRACE-X/CTRL-X. | None. |
| FRZ-OBS-05 | A | SSOT_DRIFT | AUDIT_REQUIRED | Architecture-scoped; completeness previously deferred to TRACE-X/CTRL-X. | None. |
| FRZ-OBS-06 | A | SSOT_DRIFT | AUDIT_REQUIRED | Architecture-scoped; completeness previously deferred to TRACE-X/CTRL-X. | None. |
| FRZ-OBS-07 | A | SSOT_DRIFT | AUDIT_REQUIRED | Architecture-scoped; completeness previously deferred to TRACE-X/CTRL-X. | None. |
| FRZ-SEC-01 | A+B | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Strict HTTP caller/tenant chain, cross-tenant strict (P3-A, P3-A2), TENANT-X scoped; platform-wide tenant isolation criterion. | Multi-instance (Kubernetes) trust anchor/global first-writer PIS (Decision-03; D03-A Spanner+Azure witness) - infra, not ratified/live. |
| FRZ-SEC-02 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Secret references only; resolution at the sanctioned late-resolution point. | None (live secret stores are vendor concern). |
| FRZ-SEC-03 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | No raw secret payload across layer boundaries (HOST-Q5, SEC contracts). | None. |
| FRZ-SEC-04 | A+B | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Architecture auth-bypass gates: D4 governance deny/legacy entry blocked, P2-C admission, P3-A2 strict HTTP chain, DIM16 F02. | Operator production profiles (DIM6/DIM17) - Path B live-scoped (checklist L66). |
| FRZ-SEC-05 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Provider-neutral isolation/egress contract + reference enforcement. | Real provider sandboxes (see SEC-06). |
| FRZ-SEC-06 | A (live dependency recorded as secondary; MASTER ruling: not in L66, Path B not widened) | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Plugin-trust architecture obligation: plugin/provider admission contract and trust gates (PLUG-*, PLG rows owned by other workstream) - must reach PASS or N/A-WITH-EVIDENCE at ARCH-FREEZE. | E2B/Modal/Daytona production provider qualification (PROD-Q-P1) - row note says OPEN under PROD-Q. |
| FRZ-SEC-07 | A | EVIDENCE_GAP | QUALIFICATION_REQUIRED | Explicit platform threat-model boundary statement. | None. |
| FRZ-SEC-08 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Explicit data-flow/propagation boundaries (hook ctx, CE fragment guard, tool/provider payload gateway). | None. |
| FRZ-SEC-09 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Explicit responsibility record for platform-owned durable data (STATE-X checkpoints/evidence/traces). | Operator/infra retention config (not platform). |
| FRZ-SEC-10 | A | EVIDENCE_GAP | QUALIFICATION_REQUIRED | Responsibility statement; N/A-WITH-EVIDENCE only where platform is not owner. | None. |
| FRZ-REL-01 | A | EVIDENCE_GAP | QUALIFICATION_REQUIRED | Explicit owner per timeout (root deadline, attempt boundary, provider). | None. |
| FRZ-REL-02 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Explicit retry owner (RetryEngine, tool retry-safety, provider retry budget). | None. |
| FRZ-REL-03 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | PIS + GovernedProviderInvocationDispatchGate ownership; stable CREATE identity; F02 durable follow-on. | Multi-instance global first-writer PIS (Decision-03) - infra/live-dependent. |
| FRZ-REL-04 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Cooperative cancellation, permit cleanup, retry/fan-out cancellation. | None. |
| FRZ-REL-05 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Intent != physical termination, stable operation identity, CAS, recovery ambiguity; Reflex external-work port (ADR-01). | Provider native termination live behaviour. |
| FRZ-REL-06 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Bounded retry/fallback; typed provider failure. | None. |
| FRZ-REL-07 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Bounded retry/fallback; typed provider failure. | None. |
| FRZ-REL-08 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Strict ToolRuntime typed admission/REJECT/WAIT; retry budget + rate limit + circuit. | Production capacity (see PRD-11). |
| FRZ-REL-09 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Strict ToolRuntime typed admission/REJECT/WAIT; retry budget + rate limit + circuit. | Production capacity (see PRD-11). |
| FRZ-REL-10 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Strict ToolRuntime typed admission/REJECT/WAIT; retry budget + rate limit + circuit. | Production capacity (see PRD-11). |
| FRZ-REL-11 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Strict ToolRuntime typed admission/REJECT/WAIT; retry budget + rate limit + circuit. | Production capacity (see PRD-11). |
| FRZ-PRD-01 | B | PATH_B_LIVE_DEFERRED | PATH_B_DEFERRED | None (explicitly live-scoped). | Independent production qualification evidence; acceptance owner PROD-Q live gate / SCENARIO-GATE. |
| FRZ-PRD-02 | B | PATH_B_LIVE_DEFERRED | PATH_B_DEFERRED | Structural replaceability (EBH-5/EBH-6) - other workstream. | Per-provider live qualification (PROD-Q-P1, operator profiles). |
| FRZ-PRD-03 | A+B | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Architecture fail-closed mechanisms (D1/D1b validate_colocated_serving_admission_topology, D4-03 legacy entry, P2 ENV-1/2, IG-E01). | Unsupported-config proof for operator production profiles (DIM6/DIM17) - live (checklist L66). |
| FRZ-PRD-04 | A+B? | QUALIFICATION_GAP | AUTHORITY_REQUIRED | D2/D4 local launcher ordering, D4-R2 slot release after failed strict session, P2 integrated D2->D4. | Live HTTP/GCA reproduction (R1-SQLITE-ENV-01 / PROD-Q-R1 mandatory child, roadmap places PROD-Q-R1 in live slice). |
| FRZ-PRD-05 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | ASGI lifespan + session close; D4-R2 startup-failure cleanup; D1 drain contracts. | None. |
| FRZ-PRD-06 | A | EVIDENCE_GAP | AUDIT_REQUIRED | Hosted MARK_DEGRADED semantics and projection; non-hosted -> explicit 'unknown' (degraded=null). | None. |
| FRZ-PRD-07 | A | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Strict external ASGI path vs reference main.run(); no production lab_compat on strict integrated path. | None. |
| FRZ-PRD-08 | A+B | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Architecture bypass census (EBH-5/6 other workstream, DUP-X), D1/D1b grant registry absent, P2-C admission, P3 negatives. | Production-graph bypass proof for unqualified operator profiles (live). |
| FRZ-PRD-09 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Readiness struct on strict serving session, unready guard, stale-READY/failed-prepare boundaries. | None. |
| FRZ-PRD-10 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Fail-closed stderr on legacy entry (D4-03), dual-failure ExceptionGroup on cleanup failure (D4-R2-b). | None. |
| FRZ-PRD-11 | A | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Cross-reference CTRL-X/HARNESS-W4 admission/overload evidence. | Load-level capacity of live deployment. |
| FRZ-PRD-12 | A | EVIDENCE_GAP | QUALIFICATION_REQUIRED | Explicit recovery responsibility/procedure (restart reconciliation D3, launcher handoff D4, STATE-X recovery). | Operator runbook for live environments. |

## Counts

| Primary classification | n |
|---|---|
| EVIDENCE_GAP | 5 |
| PATH_B_LIVE_DEFERRED | 2 |
| QUALIFICATION_GAP | 18 |
| SSOT_DRIFT | 4 |
| VALID_SCOPED_PASS_CANDIDATE | 11 |

## Rows whose scope is a RULING REQUEST (not decided by this worker)

- **FRZ-SEC-06** - RULED by MASTER: not in checklist L66, Path B is not widened; primary = architecture-side plugin-trust gap (QUALIFICATION_GAP), provider live dependency (E2B/Modal/Daytona, PROD-Q-P1) = secondary cause; row stays OPEN. Proposed L66 note/split = Operator-level SSOT proposal (candidate patch P-03), not applied.
- **FRZ-PRD-04** - blocker owner PROD-Q-R1 is in the roadmap live slice (L478) but PRD-04 is not in checklist L66; mechanical record pins BLOCKER. Ruling needed.
- **FRZ-SEC-05** - evidence is Linux reference substrate only; scope meaning at architecture level needs certifier/operator ruling.
- **FRZ-SEC-01 / REL-03 / REL-05 / PRD-08** - ratified Kubernetes multi-instance trust-anchor/global-PIS (Decision-03, D03-A Spanner+Azure) obligations are infrastructure-dependent; ruling needed on which parts are Dimension A.

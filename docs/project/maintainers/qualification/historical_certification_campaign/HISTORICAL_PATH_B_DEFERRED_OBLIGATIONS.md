# Historical Path B deferred obligations
Generated 2026-11-08T10:56:32Z. Source rule: `PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md` section "Path B - architecture-scoped vs live qualification acceptance" (the live-scoped list is defined only there; HIST03 finding F-001 records that `ROADMAP_REPLAY_X_FRZ_225_RECONCILIATION.md` must not be used for it). Path B defers LIVE qualification only; it never waives the architecture obligation and never grants production readiness.

## 1. Rows whose primary classification is PATH_B_LIVE_DEFERRED (3)
| FRZ id | requirement | live obligation | residual |
|---|---|---|---|
| FRZ-REG-08 | environment failures cannot become false PASS | (see matrix) | build_r1_sqlite_disposition is NOT_EXECUTED for live HTTP / GCA: owned by PROD-Q Dimension B (live), which this campaign must not take over. Related observation |
| FRZ-PRD-01 | production qualification independent of lab/harness | Independent production qualification evidence; acceptance owner PROD-Q live gate / SCENARIO-GATE. | Live slice NOT QUALIFIED by design. |
| FRZ-PRD-02 | provider/plugin qualification enforced | Per-provider live qualification (PROD-Q-P1, operator profiles). | Live slice NOT QUALIFIED by design. |

## 2. Rows that keep their ARCHITECTURE obligation and carry a LIVE obligation as a secondary cause (17)
These rows are NOT deferred as a whole: the architecture part still has to reach its acceptance rule.
| FRZ id | primary classification | state | live obligation (deferred) |
|---|---|---|---|
| FRZ-SEC-01 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Multi-instance (Kubernetes) trust anchor/global first-writer PIS (Decision-03; D03-A Spanner+Azure witness) - infra, not ratified/live. |
| FRZ-SEC-02 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | None (live secret stores are vendor concern). |
| FRZ-SEC-04 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Operator production profiles (DIM6/DIM17) - Path B live-scoped (checklist L66). |
| FRZ-SEC-05 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Real provider sandboxes (see SEC-06). |
| FRZ-SEC-06 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | E2B/Modal/Daytona production provider qualification (PROD-Q-P1) - row note says OPEN under PROD-Q. |
| FRZ-SEC-09 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Operator/infra retention config (not platform). |
| FRZ-REL-03 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Multi-instance global first-writer PIS (Decision-03) - infra/live-dependent. |
| FRZ-REL-05 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Provider native termination live behaviour. |
| FRZ-REL-08 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Production capacity (see PRD-11). |
| FRZ-REL-09 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Production capacity (see PRD-11). |
| FRZ-REL-10 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Production capacity (see PRD-11). |
| FRZ-REL-11 | VALID_SCOPED_PASS_CANDIDATE | AUDIT_REQUIRED | Production capacity (see PRD-11). |
| FRZ-PRD-03 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Unsupported-config proof for operator production profiles (DIM6/DIM17) - live (checklist L66). |
| FRZ-PRD-04 | QUALIFICATION_GAP | AUTHORITY_REQUIRED | Live HTTP/GCA reproduction (R1-SQLITE-ENV-01 / PROD-Q-R1 mandatory child, roadmap places PROD-Q-R1 in live slice). |
| FRZ-PRD-08 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Production-graph bypass proof for unqualified operator profiles (live). |
| FRZ-PRD-11 | QUALIFICATION_GAP | QUALIFICATION_REQUIRED | Load-level capacity of live deployment. |
| FRZ-PRD-12 | EVIDENCE_GAP | QUALIFICATION_REQUIRED | Operator runbook for live environments. |

## 3. Pending scope rulings (Operator decision DR-9, unanswered)
FRZ-SEC-06 is NOT on the checklist live-scoped list; it is classified with the architecture side as primary and the provider-live dependency as secondary (MASTER ruling; Path B was not widened). FRZ-PRD-04 (architecture vs live scope), FRZ-SEC-05 (meaning of scope), Decision-03 / D03-A (Kubernetes multi-instance trust anchor, Spanner + Azure witness) and the waivers for AUTH-08 HTTP / G-08 / IG-F01 / DIM16 S2b await the Operator.

## 4. Ownership
Live qualification stays with PROD-Q Dimension B (PROD-Q-P1 and the live gate). The campaign did not start, simulate or close any live integration, and made no production configuration change.

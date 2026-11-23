# SFX R2 — Implementation readiness traceability matrix

**Status:** PROPOSED documentation · **NOT** `IMPLEMENTATION_AUTHORIZED`
**Campaign:** `IGX-SFX-R2-DOC-FINALIZATION-001` · parent corrective `IGX-SFX-R2-CORR-001`

Binding architecture remains **R1** until operator `CANONICAL_R2_APPROVED`.

## Operator gates (programme — not WBS exit)

| Gate | Role | Blocks until |
| --- | --- | --- |
| ARCH-FREEZE | Programme ENTRY | W3 implementation |
| GATE-SFX-ENTRY | Factory ENTRY | Tier-2 `agents/scenario_factory/` work |
| GATE-SEF-CONSUME | SEF handoff ENTRY | R2-W4-06+ |
| HO-2 / OD-SEF-R2 | Delegated autonomy ENTRY | R2-W4-07, R2-W5-03 auto path |
| ECAP §16 (`C-ECAP-01`) | Certification INPUT | R2-W8-03 rebind |
| CANONICAL_R2_APPROVED | Architecture adoption | Master plan R2 merge |
| GATE-SFX-FULL | Factory CERTIFICATION | W8 closure |
| First sandbox POC | Demo scope only | R2-W6-03..04 internal demo gate — **not** PROD-Q |

## REQ-R2 → execution chain

| REQUIREMENT | TASK(S) | CONTRACT | DEPENDENCY | ENTRY_GATE | CERT_GATE | TEST | EVIDENCE |
| --- | --- | --- | --- | --- | --- | --- | --- |
| REQ-R2-001 | R2-D02, R2-W5-03 | schema/CON per catalog | WBS deps per task row | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | mode policy matrix tests | `.tmp/proof/sfx/` |
| REQ-R2-002 | R2-W3-04, SFX-P5-05 | schema/CON per catalog | WBS deps per task row | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | CAS + resume tests | `.tmp/proof/sfx/` |
| REQ-R2-003 | R2-W3-03 | schema/CON per catalog | WBS deps per task row | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | grant expiry/revoke tests | `.tmp/proof/sfx/` |
| REQ-R2-004 | R2-W3-01, R2-W4-01 | schema/CON per catalog | WBS deps per task row | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | validator golden tests | `.tmp/proof/sfx/` |
| REQ-R2-005 | R2-W3-02 | schema/CON per catalog | WBS deps per task row | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | provenance + tenant tests | `.tmp/proof/sfx/` |
| REQ-R2-006 | R2-W3-04, R2-W4-05 | schema/CON per catalog | WBS deps per task row | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | append-only replay | `.tmp/proof/sfx/` |
| REQ-R2-007 | R2-W5-01 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | oracle injection tests | `.tmp/proof/sfx/` |
| REQ-R2-008 | R2-W5-03 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | adversarial/budget_parallel | `.tmp/proof/sfx/` |
| REQ-R2-009 | R2-W5-02 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | regression suite | `.tmp/proof/sfx/` |
| REQ-R2-010 | R2-W5-04, SFX-P10-03 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | PC-1..6 mission ops | `.tmp/proof/sfx/` |
| REQ-R2-011 | R2-W4-07, R2-W8-02 | schema/CON per catalog | WBS deps per task row | GATE-SFX-ENTRY + GATE-SEF-CONSUME | HO bundle / idempotency log | HO+grant integration tests | `.tmp/proof/sfx/` |
| REQ-R2-012 | R2-W4-06 | schema/CON per catalog | WBS deps per task row | GATE-SFX-ENTRY + GATE-SEF-CONSUME | HO bundle / idempotency log | handoff/grant_required | `.tmp/proof/sfx/` |
| REQ-R2-013 | R2-W7-01, SFX-P12-01 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | REF1-O-* bundle | `.tmp/proof/sfx/` |
| REQ-R2-014 | R2-W7-07, SFX-P12-06 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | .tmp/proof/sfx/ | `.tmp/proof/sfx/` |
| REQ-R2-015 | R2-W6-05 | schema/CON per catalog | WBS deps per task row | parent task outputs | checklist doc only | handoff checklist | `.tmp/proof/sfx/` |
| REQ-R2-016 | SFX-P11-04, SFX-P13-03 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | dossier ≠ admission | `.tmp/proof/sfx/` |
| REQ-R2-017 | R2-W5-06, SFX-P9-05 | schema/CON per catalog | WBS deps per task row | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | T-R2-* tests | `.tmp/proof/sfx/` |
| REQ-R2-018 | R2-D05 | schema/CON per catalog | WBS deps per task row | post-W8 programme | post-W8 programme | roadmap doc | `.tmp/proof/sfx/` |

## WBS unit chain (37 tasks)

| TASK | WAVE | INPUTS (deps) | OUTPUTS | ENTRY_GATE | CERT_GATE | TEST+ | EVIDENCE | ACCEPTANCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2-W3-01 | W3 | GATE-SFX-ENTRY satisfied | agents/scenario_factory/contracts/ (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | contract round-trip | .tmp/proof/sfx/R2-W3-01/ | round-trip green |
| R2-W3-02 | W3 | R2-W3-01 | contracts + schema artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | provenance tests | log | schema valid |
| R2-W3-03 | W3 | R2-W3-01 | mission/envelope modules (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | grant deny/expire | log | fail-closed budgets |
| R2-W3-04 | W3 | R2-W3-01 | persistence views (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | CAS/replay | log | append-only |
| R2-W3-05 | W3 | R2-W3-03 | Tier-3 API (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | PC drift | parity subset | digests stable |
| R2-W4-01 | W4-CORE | R2-W3-01 | discovery/spec (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | replay fixtures | eval | blocking validation |
| R2-W4-02 | W4-CORE | R2-W4-01 | testing/ (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | oracle unit | golden | independent negatives |
| R2-W4-03 | W4-CORE | R2-W3-01 | blueprint/ (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | golden digest | log | deterministic |
| R2-W4-04 | W4-CORE | R2-W4-03 | handoff/scaffold (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | scaffold determinism | tree digest | ≠ POC label |
| R2-W4-05 | W4-CORE | R2-W3-04 | mission/planner (PROPOSED) artifact | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY checklist | per-task unit tests recorded | planner unit | log | no side effects |
| R2-W4-06 | W4-SEF | R2-W4-04,R2-W4-01,R2-W4-02,R2-W4-03 | handoff/sef_adapter (PROPOSED) artifact | GATE-SFX-ENTRY + GATE-SEF-CONSUME | HO bundle / idempotency log | HO tests | HO bundle | HO-1..7 green |
| R2-W4-08 | W4-SEF | R2-W4-06 | handoff/jobs (PROPOSED) artifact | GATE-SFX-ENTRY + GATE-SEF-CONSUME | HO bundle / idempotency log | idempotency | log | no duplicate effects |
| R2-W4-07 | W4-DELEGATED | R2-W4-06,OD-SEF-R2 | sef_adapter grant mode (PROPOSED) artifact | GATE-SEF-CONSUME + OD-SEF-R2 signed | grant integration log | grant integration | cross-track log | fail without OD |
| R2-W5-01 | W5 | R2-W4-02 | testing/eval_view (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | oracle injection | verdict digest | code≠business pass |
| R2-W5-02 | W5 | R2-W5-01 | compare (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | regression suite | compare record | security non-regress |
| R2-W5-03 | W5 | R2-W3-03,R2-W4-07 | mission/driver (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | adversarial driver | driver log | STOP rules |
| R2-W5-04 | W5 | R2-W3-05 | web mission (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | e2e subset | screenshots | pause/stop |
| R2-W5-05 | W5 | R2-W5-04 | web trace (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | e2e | log | provenance visible |
| R2-W5-06 | W5 | R2-W5-03 | P9+P10 (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | adversarial T-R2 | qual log | fail closed |
| R2-W5-07 | W5 | R2-W5-04,R2-W3-05 | parity/ (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | PC-1..6 | parity | 100% mission ops |
| R2-W6-01 | W6 | R2-W5-02 | reporting/compare (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | golden | report | predeclared metrics |
| R2-W6-02 | W6 | R2-W6-01 | value_model (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | source ref scan | value report | every number sourced |
| R2-W6-03 | W6 | R2-W5-01 | packaging (PROPOSED) artifact | R2-W6-03 portable POC bundle | demo manifest with NOT PRODUCTION labels | repro checklist | bundle | independent replay |
| R2-W6-04 | W6 | R2-W6-03 | demo pack (PROPOSED) artifact | R2-W6-03 portable POC bundle | demo manifest with NOT PRODUCTION labels | label tests | demo manifest | tenant-bound |
| R2-W6-05 | W6 | R2-W6-01 | commercial (PROPOSED) artifact | parent task outputs | checklist doc only | checklist | handoff doc | PROD separate |
| R2-W7-01 | W7 | R2-W5-01,R2-W4-06 | fixtures+qual (PROPOSED) artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | REF1-O-* | ref-1 bundle | assisted or bounded per mode |
| R2-W7-02 | W7 | R2-W5-01,R2-W4-06 | REF-2 qual artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | REF-2 suite | bundle | OD-SFX-08 |
| R2-W7-03 | W7 | R2-W5-01,R2-W4-06 | REF-3 qual artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | REF-3 suite | bundle | OD-SFX-08 |
| R2-W7-04 | W7 | R2-W7-01 | GEN qual artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | GEN suite | GEN report | seal verified |
| R2-W7-05 | W7 | R2-W5-04 | usability artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | protocol | report | no safety ambiguity |
| R2-W7-06 | W7 | R2-W4-06 | qual report artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | backend matrix | report | honest non-equiv |
| R2-W7-07 | W7 | R2-W7-01 | metrics job artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | metrics @ SHA | .tmp/proof/ | MS rules |
| R2-W8-01 | W8 | W7 | qual/trace artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | trace matrix | matrix | 0 unmapped critical |
| R2-W8-02 | W8 | R2-W4-07 | integration qual artifact | GATE-SFX-ENTRY + GATE-SEF-CONSUME | HO bundle / idempotency log | HO+grant tests | log | no HO bypass |
| R2-W8-03 | W8 | R2-W8-01,R2-W8-02,C-ECAP-01-OPEN | governance artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | ACT-N11 | rebind log | OD-SFX-06+09 |
| R2-W8-04 | W8 | R2-W8-01,R2-W8-02,R2-W8-03 | P13 artifact | W3–W4 deps satisfied + GATE-SFX-FULL checklist | qualification artifact @ SHA | P13 suite | closure text | ≠ PROD-Q |
| R2-W8-05 | W8 | R2-W6-05 | doc only artifact | parent task outputs | checklist doc only | checklist | pilot gate | explicit limits |

## Authority invariant

SFX **does not** assume engineering execution authority from SEF-X. SEF owns jobs, workspaces, bounded repair, and engineering evidence; SFX owns discovery, business evaluation views, and handoff **requests** only (`SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md`).

## Doc-only REQ rows (no WBS impl id)

| REQ | Doc deliverable |
| --- | --- |
| REQ-R2-001 (partial) | `R2-D02` → `SFX_R2_AUTONOMOUS_MISSION_SPEC.md` mode matrix |
| REQ-R2-018 | `R2-D05` → `SFX_R2_COMMERCIALIZATION_ROADMAP.md` §5 |

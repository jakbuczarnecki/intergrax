# SFX R2 — Implementation readiness traceability matrix

**Status:** PROPOSED documentation · **NOT** `IMPLEMENTATION_AUTHORIZED`
**Campaign:** `IGX-SFX-R2-FINAL-PUBLICATION-002`

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
| REQ-R2-001 | R2-D02, R2-W5-03 | CON-R2-02; mission spec | WBS deps per task | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY | **GATE-SFX-ENTRY** | mode policy matrix tests | mode policy matrix tests |
| REQ-R2-002 | R2-W3-04, SFX-P5-05 | CON-R2-01, CON-R2-04 | WBS deps per task | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY | **GATE-SFX-ENTRY** | CAS + resume tests | CAS + resume tests |
| REQ-R2-003 | R2-W3-03 | CON-R2-02 | WBS deps per task | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY | **GATE-SFX-ENTRY** | grant expiry/revoke tests | grant expiry/revoke tests |
| REQ-R2-004 | R2-W3-01, R2-W4-01 | CON-01/03 | WBS deps per task | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY | **GATE-SFX-ENTRY** | validator golden tests | validator golden tests |
| REQ-R2-005 | R2-W3-02 | CON-R2-03 | WBS deps per task | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY | **GATE-SFX-ENTRY** | provenance + tenant tests | provenance + tenant tests |
| REQ-R2-006 | R2-W3-04, R2-W4-05 | CON-R2-04 | WBS deps per task | ARCH-FREEZE CLOSED + GATE-SFX-ENTRY | **GATE-SFX-ENTRY** | append-only replay | append-only replay |
| REQ-R2-007 | R2-W5-01 | CON-R2-05 | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | oracle injection tests | oracle injection tests |
| REQ-R2-008 | R2-W5-03 | CON-R2-02 | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | adversarial/budget_parallel | adversarial/budget_parallel |
| REQ-R2-009 | R2-W5-02 | CON-R2-06 | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | regression suite | regression suite |
| REQ-R2-010 | R2-W5-04, SFX-P10-03 | CON-R2-01; Studio UX | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | PC-1..6 mission ops | PC-1..6 mission ops |
| REQ-R2-011 | R2-W4-07, R2-W8-02 | CON-08; grant | WBS deps per task | GATE-SFX-ENTRY + GATE-SEF-CONSUME | **GATE-SEF-CONSUME** | HO+grant integration tests | HO+grant integration tests |
| REQ-R2-012 | R2-W4-06 | CON-08; HO-2 | WBS deps per task | GATE-SFX-ENTRY + GATE-SEF-CONSUME | **GATE-SEF-CONSUME** | handoff/grant_required | handoff/grant_required |
| REQ-R2-013 | R2-W7-01, SFX-P12-01 | CON-R2-05; REF1-O-* | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | REF1-O-* bundle | REF1-O-* bundle |
| REQ-R2-014 | R2-W7-07, SFX-P12-06 | SFX_METRICS | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | .tmp/proof/sfx/ | metrics @ SHA (R2-W7-07) |
| REQ-R2-015 | R2-W6-05 | commercial checklist | WBS deps per task | parent task outputs | **—** | handoff checklist | handoff checklist |
| REQ-R2-016 | SFX-P11-04, SFX-P13-03 | admission dossier; INV-01 | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | dossier ≠ admission | dossier ≠ admission |
| REQ-R2-017 | R2-W5-06, SFX-P9-05 | mission security | WBS deps per task | W3–W4 outputs + programme prerequisites | **GATE-SFX-FULL** | T-R2-* tests | T-R2-* tests |
| REQ-R2-018 | R2-D05 | commercial roadmap §5 | WBS deps per task | W8 factory closure | **post-W8 programme** | roadmap doc | roadmap doc |

## WBS unit chain (37 tasks)

| TASK | WAVE | INPUTS (deps) | OUTPUTS | ENTRY_GATE | CERT_GATE | TEST+ | EVIDENCE | ACCEPTANCE |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| R2-W3-01 | W3 | GATE-SFX-ENTRY satisfied | agents/scenario_factory/contracts/ (PROPOSED) artifact | GATE-SFX-ENTRY | **GATE-SFX-ENTRY** | contract round-trip | .tmp/proof/sfx/R2-W3-01/ | round-trip green |
| R2-W3-02 | W3 | R2-W3-01 | contracts + schema artifact | prior outputs (R2-W3-01) | **GATE-SFX-ENTRY** | provenance tests | log | schema valid |
| R2-W3-03 | W3 | R2-W3-01 | mission/envelope modules (PROPOSED) artifact | prior outputs (R2-W3-01) | **GATE-SFX-ENTRY** | grant deny/expire | log | fail-closed budgets |
| R2-W3-04 | W3 | R2-W3-01 | persistence views (PROPOSED) artifact | prior outputs (R2-W3-01) | **GATE-SFX-ENTRY** | CAS/replay | log | append-only |
| R2-W3-05 | W3 | R2-W3-03 | Tier-3 API (PROPOSED) artifact | prior outputs (R2-W3-03) | **GATE-SFX-ENTRY** | PC drift | parity subset | digests stable |
| R2-W4-01 | W4-CORE | R2-W3-01 | discovery/spec (PROPOSED) artifact | prior outputs (R2-W3-01) | **GATE-SFX-ENTRY** | replay fixtures | eval | blocking validation |
| R2-W4-02 | W4-CORE | R2-W4-01 | testing/ (PROPOSED) artifact | prior outputs (R2-W4-01) | **GATE-SFX-ENTRY** | oracle unit | golden | independent negatives |
| R2-W4-03 | W4-CORE | R2-W3-01 | blueprint/ (PROPOSED) artifact | prior outputs (R2-W3-01) | **GATE-SFX-ENTRY** | golden digest | log | deterministic |
| R2-W4-04 | W4-CORE | R2-W4-03 | handoff/scaffold (PROPOSED) artifact | prior outputs (R2-W4-03) | **GATE-SFX-ENTRY** | scaffold determinism | tree digest | ≠ POC label |
| R2-W4-05 | W4-CORE | R2-W3-04 | mission/planner (PROPOSED) artifact | prior outputs (R2-W3-04) | **GATE-SFX-ENTRY** | planner unit | log | no side effects |
| R2-W4-06 | W4-SEF | R2-W4-04,R2-W4-01,R2-W4-02,R2-W4-03 | handoff/sef_adapter (PROPOSED) artifact | prior outputs (R2-W4-04,R2-W4-01,R2-W4-02,R2-W4-03) | **GATE-SEF-CONSUME** | HO tests | HO bundle | HO-1..7 green |
| R2-W4-08 | W4-SEF | R2-W4-06 | handoff/jobs (PROPOSED) artifact | prior outputs (R2-W4-06) | **GATE-SEF-CONSUME** | idempotency | log | no duplicate effects |
| R2-W4-07 | W4-DELEGATED | R2-W4-06,OD-SEF-R2 | sef_adapter grant mode (PROPOSED) artifact | R2-W4-06,OD-SEF-R2 | **GATE-SEF-CONSUME+OD** | grant integration | cross-track log | fail without OD |
| R2-W5-01 | W5 | R2-W4-02 | testing/eval_view (PROPOSED) artifact | prior outputs (R2-W4-02) | **GATE-SFX-FULL** | oracle injection | verdict digest | code≠business pass |
| R2-W5-02 | W5 | R2-W5-01 | compare (PROPOSED) artifact | prior outputs (R2-W5-01) | **GATE-SFX-FULL** | regression suite | compare record | security non-regress |
| R2-W5-03 | W5 | R2-W3-03,R2-W4-07 | mission/driver (PROPOSED) artifact | prior outputs (R2-W3-03,R2-W4-07) | **GATE-SFX-FULL** | adversarial driver | driver log | STOP rules |
| R2-W5-04 | W5 | R2-W3-05 | web mission (PROPOSED) artifact | prior outputs (R2-W3-05) | **GATE-SFX-FULL** | e2e subset | screenshots | pause/stop |
| R2-W5-05 | W5 | R2-W5-04 | web trace (PROPOSED) artifact | prior outputs (R2-W5-04) | **GATE-SFX-FULL** | e2e | log | provenance visible |
| R2-W5-06 | W5 | R2-W5-03 | P9+P10 (PROPOSED) artifact | prior outputs (R2-W5-03) | **GATE-SFX-FULL** | adversarial T-R2 | qual log | fail closed |
| R2-W5-07 | W5 | R2-W5-04,R2-W3-05 | parity/ (PROPOSED) artifact | prior outputs (R2-W5-04,R2-W3-05) | **GATE-SFX-FULL** | PC-1..6 | parity | 100% mission ops |
| R2-W6-01 | W6 | R2-W5-02 | reporting/compare (PROPOSED) artifact | prior outputs (R2-W5-02) | **GATE-SFX-FULL** | golden | report | predeclared metrics |
| R2-W6-02 | W6 | R2-W6-01 | value_model (PROPOSED) artifact | prior outputs (R2-W6-01) | **GATE-SFX-FULL** | source ref scan | value report | every number sourced |
| R2-W6-03 | W6 | R2-W5-01 | packaging (PROPOSED) artifact | prior outputs (R2-W5-01) | **internal demo gate** | repro checklist | bundle | independent replay |
| R2-W6-04 | W6 | R2-W6-03 | demo pack (PROPOSED) artifact | prior outputs (R2-W6-03) | **internal demo gate** | label tests | demo manifest | tenant-bound |
| R2-W6-05 | W6 | R2-W6-01 | commercial (PROPOSED) artifact | prior outputs (R2-W6-01) | **—** | checklist | handoff doc | PROD separate |
| R2-W7-01 | W7 | R2-W5-01,R2-W4-06 | fixtures+qual (PROPOSED) artifact | prior outputs (R2-W5-01,R2-W4-06) | **GATE-SFX-FULL** | REF1-O-* | ref-1 bundle | assisted or bounded per mode |
| R2-W7-02 | W7 | R2-W5-01,R2-W4-06 | REF-2 qual artifact | prior outputs (R2-W5-01,R2-W4-06) | **GATE-SFX-FULL** | REF-2 suite | bundle | OD-SFX-08 |
| R2-W7-03 | W7 | R2-W5-01,R2-W4-06 | REF-3 qual artifact | prior outputs (R2-W5-01,R2-W4-06) | **GATE-SFX-FULL** | REF-3 suite | bundle | OD-SFX-08 |
| R2-W7-04 | W7 | R2-W7-01 | GEN qual artifact | prior outputs (R2-W7-01) | **GATE-SFX-FULL** | GEN suite | GEN report | seal verified |
| R2-W7-05 | W7 | R2-W5-04 | usability artifact | prior outputs (R2-W5-04) | **GATE-SFX-FULL** | protocol | report | no safety ambiguity |
| R2-W7-06 | W7 | R2-W4-06 | qual report artifact | prior outputs (R2-W4-06) | **GATE-SFX-FULL** | backend matrix | report | honest non-equiv |
| R2-W7-07 | W7 | R2-W7-01 | metrics job artifact | prior outputs (R2-W7-01) | **GATE-SFX-FULL** | metrics @ SHA | .tmp/proof/ | MS rules |
| R2-W8-01 | W8 | W7 | qual/trace artifact | W7 qualification slice | **GATE-SFX-FULL** | trace matrix | matrix | 0 unmapped critical |
| R2-W8-02 | W8 | R2-W4-07 | integration qual artifact | prior outputs (R2-W4-07) | **GATE-SEF-CONSUME** | HO+grant tests | log | no HO bypass |
| R2-W8-03 | W8 | R2-W8-01,R2-W8-02,C-ECAP-01-OPEN | governance artifact | prior outputs (R2-W8-01,R2-W8-02,C-ECAP-01-OPEN) | **GATE-SFX-FULL** | ACT-N11 | rebind log | OD-SFX-06+09 |
| R2-W8-04 | W8 | R2-W8-01,R2-W8-02,R2-W8-03 | P13 artifact | prior outputs (R2-W8-01,R2-W8-02,R2-W8-03) | **GATE-SFX-FULL** | P13 suite | closure text | ≠ PROD-Q |
| R2-W8-05 | W8 | R2-W6-05 | doc only artifact | prior outputs (R2-W6-05) | **—** | checklist | pilot gate | explicit limits |

## Authority invariant

SFX **does not** assume engineering execution authority from SEF-X. SEF owns jobs, workspaces, bounded repair, and engineering evidence; SFX owns discovery, business evaluation views, and handoff **requests** only (`SFX_R2_SEF_CROSS_PROGRAM_AMENDMENT.md`).

## Doc-only REQ rows (no WBS impl id)

| REQ | Doc deliverable |
| --- | --- |
| REQ-R2-001 (partial) | `R2-D02` → `SFX_R2_AUTONOMOUS_MISSION_SPEC.md` mode matrix |
| REQ-R2-018 | `R2-D05` → `SFX_R2_COMMERCIALIZATION_ROADMAP.md` §5 |

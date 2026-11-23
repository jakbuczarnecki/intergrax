# SFX R2 — Full Scenario Lifecycle Stage Specification (PROPOSED)

Stages span customer problem → commercial handoff. Authority: L0/L1 design; L2 sandbox execution only with approvals/grant.

| Stage | Inputs | Outputs | Contract / schema | Owner | Auth | Persistence | On error | Tests | Evidence | Operator rule | Gate |
| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |
| Problem intake | text, uploads (synthetic) | `BusinessProblemStatement` | CON-01 | SFX | L0 | CW version | reject untrusted exec content | schema round-trip | digest | no PII in LAB default | — |
| Discovery | problem ref | `DiscoveryRecord` | CON-02 | SFX discovery | L0/L1 | CW | `NEEDS_INPUT` | replay fixtures | eval report | human clarifies blockers | — |
| Source intake | files/API samples | `SourceIntakeManifest` | **PROPOSED** | SFX | L1 | CW | quarantine malware fail | provenance tests | scan log | classify data | OD-SFX-19 |
| Specification | discovery | `BusinessScenarioSpecification` + goals | CON-03 | SFX | L1 | CW | validator reject | golden specs | validator log | edit before approve | — |
| Capability resolve | spec | `CapabilityResolutionReport` | CON-04 | SFX | L1 | CW | MISSING caps flagged | golden | report digest | no runtime invoke | — |
| Blueprint | resolution | `ScenarioBlueprint` | CON-05 | SFX | L1 | CW | fail closed | golden digest | blueprint_digest | — | — |
| Profile / integration | blueprint | profile + setup plan | CON-06/07 | SFX | L1 | CW | secret scan fail | no-secret tests | refs only | — | — |
| Mission envelope | blueprint+acceptance | `MissionRecord`+`AutonomyEnvelope` | **PROPOSED** | SFX | L2 approve | CW+decision | deny if incomplete | grant tests | approval_ref | human signs grant | OD-SFX-R2-01 |
| Baseline scaffold | blueprint | tree digest | HO + scaffold | SEF/SFX handoff | L2 | evidence | determinism fail | scaffold test | tree digest | not labeled POC | GATE-SFX-ENTRY |
| Candidate build | hypothesis | `EngineeringJobRecord` | CON-16 | SEF | L2 | SEF store | DLQ | HO-1..7 | job evidence | HO-2 human until grant | GATE-SEF-CONSUME |
| Supply-chain qualify | candidate pkg | gate verdict | P9 modules | SFX qual | L2 | proof dir | fail closed | SAST/deps | scan artifacts | OD-SFX-14 | — |
| Sandbox test run | profile+pkg | runtime results | runtime evidence | **Runtime** | L2 | evidence store | policy block | REF oracles | run_id | no production creds | — |
| Business evaluate | runs+acceptance | `ScenarioEvaluationVerdict` | **PROPOSED** | SFX view | L2 | CW | `INSUFFICIENT_EVIDENCE` | oracle suite | verdict digest | independent criteria | — |
| Improve / iterate | verdict | new `CandidateIteration` | **PROPOSED** | SFX planner | L2 grant | CW | STOP rules | driver tests | iteration chain | no scope creep | grant |
| Champion freeze | comparisons | champion ref | lifecycle | SFX | L2 | CW | security regression deny | compare tests | champion digest | independent check | SANDBOX_CERTIFIED path |
| Present / dossier | champion | reports + admission dossier | CON-11/17 | SFX | L1 | CW | incomplete dossier | completeness | export bundle | not production admission | SCENARIO-GATE ext |
| Commercial package | dossier | `CommercialDeliveryPackage` | **PROPOSED** view | SFX | L1 | CW | — | checklist | handoff doc | sales boundary | — |
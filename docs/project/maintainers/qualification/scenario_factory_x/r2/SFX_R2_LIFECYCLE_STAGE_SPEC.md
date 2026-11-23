# SFX R2 — Full Scenario Lifecycle Stage Specification (PROPOSED)

**Contract SSOT:** R1 master plan §7.4. R2 additive schemas: `design/schemas/proposed/`.

| Stage | Outputs | R1 CON | Auth to persist | Owner | Gate |
| --- | --- | --- | --- | --- | --- |
| Problem intake | BusinessProblemStatement | CON-01 | L1 door (offline CLI validate-only = L0) | SFX | — |
| Discovery | DiscoveryRecord | CON-02 | L1 | SFX | — |
| Source intake | SourceIntakeManifest | CON-R2-03 | L1 | SFX | OD-SFX-19 |
| Specification | BusinessScenarioSpecification | CON-03 | L1 | SFX | — |
| Capability resolve | CapabilityResolutionReport | CON-04 | L1 | SFX | — |
| Blueprint | ScenarioBlueprint | CON-05 | L1 | SFX | — |
| Integration setup | IntegrationSetupPlan | CON-06 | L1 | SFX | — |
| Test plan | ScenarioTestPlan | CON-07 | L1 | SFX | — |
| Mission envelope | MissionRecord + AutonomyEnvelope | CON-R2-01/02 + CON-13 approval | **Human CON-13** unlocks listed L2 ops | SFX | OD-SFX-R2-01 |
| Handoff | EngineeringHandoffRequest | CON-08 | L2; HO-2 submit | SFX draft / SEF exec | GATE-SEF-CONSUME |
| Engineering job | EngineeringJobRecord | **CON-15** | L2 | SEF-X | GATE-SEF-CONSUME |
| Qualification record | ScenarioQualificationRecord | CON-16 | L2 | SFX | — |
| Sandbox run | evidence | runtime | L2 | Runtime | — |
| Business evaluate | ScenarioEvaluationVerdict | CON-R2-05 view | L2 | SFX view | — |
| Iterate | CandidateIteration | CON-R2-04 | L2 in grant | SFX | grant |
| Present | RunReport + ValueReport | CON-09, CON-10 | L1 | SFX | — |
| Dossier | ScenarioAdmissionDossier | **CON-17** | Human export decision | SFX | OD-SFX-18 PROPOSED |
| Catalog | ScenarioCatalogEntry | **CON-11** | L1 | SFX | — |

**Corrections applied (IGX-SFX-R2-CORR-001):** CON-15 for jobs; CON-17 for dossier; CON-11 catalog only; CON-06/07 split; no L0 persistence for intake; envelope uses CON-13 not “L2 approve”.
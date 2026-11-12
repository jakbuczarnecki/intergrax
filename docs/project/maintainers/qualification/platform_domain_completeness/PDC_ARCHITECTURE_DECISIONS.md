# PDC-X — architecture decisions (PDC-P3)

**Ticket:** T-IGX-036-D · **Lock gate:** [OD-PDC-ADR-01](./operator_decisions/OD-PDC-ADR-01.md) (pending). Every ADR below has status **Proposed**. None is accepted, and no implementation starts before acceptance.

| ADR | Domain | Decision summary | Scope OD |
| --- | --- | --- | --- |
| [ADR-PDC-01](../../architecture/ADR_PDC_01_AGENT_UPGRADE_RETIRE.md) | D1 | Explicit `upgrade_agent` / `retire_agent` on the existing admin service, with a compatibility check | OD-PDC-SCOPE-01 (skills) |
| [ADR-PDC-02](../../architecture/ADR_PDC_02_CONTEXT_TRIM_FIDELITY_AND_MEMORY_RETENTION.md) | D2 | Hard trim keeps tool-call linkage; CE-02 closure through PDC; memory retention sweep through `MemoryControlPlane` | OD-PDC-SCOPE-06 (freshness) |
| [ADR-PDC-03](../../architecture/ADR_PDC_03_MODEL_STREAMING_CAPABILITY_CONTRACT.md) | D3 | Declared streaming, tool-streaming and cancel capabilities; conformance gate; fail-closed mismatch; ModelCatalog owns context windows | OD-PDC-SCOPE-03 |
| [ADR-PDC-04](../../architecture/ADR_PDC_04_HOSTING_LIFECYCLE_STABLE_API.md) | D4 | Generic APP-HOST lifecycle suite, metrics through observability, digest-preserving restart, stable hosting API | OD-PDC-SCOPE-02 |
| [ADR-PDC-05](../../architecture/ADR_PDC_05_EVALUATION_CONTRACTS.md) | D5 | Typed EvalDataset / EvalRun / EvalComparison; evaluator kinds incl. SAFETY; evidence persistence; advisory only | — |
| [ADR-PDC-06](../../architecture/ADR_PDC_06_RUNTIME_SECURITY_HARDENING.md) | D6 | Fail-closed signing and poisoning defaults; single tool-input defense; durable audit trail; static secret gate; security satellite | OD-PDC-SCOPE-07 |
| [ADR-PDC-07](../../architecture/ADR_PDC_07_BACKGROUND_RECURRENCE_AND_DLQ.md) | D7 | Recurrence as one-shot occurrences on SchedulerLedger; terminal dead-letter; identity on enqueue | OD-PDC-SCOPE-05 |
| [ADR-PDC-08](../../architecture/ADR_PDC_08_ARTIFACT_LIFECYCLE_TAXONOMY.md) | D8 | Four artifact classes, four owners; mandatory digest; governed read port; enforced retention; dedupe | OD-PDC-SCOPE-04 |

**Scope note.** ADR-PDC-02 was not in the original P3 list. Master plan §F made it conditional ("No ADR is planned for D2 unless P2 proves a contract gap in R005/R011/R012/R015"). P2 proved gaps in R011, R012 and R015, so the condition triggered.

## Decisions recorded in P3 (no ADR needed)

| ID | Decision | Basis |
| --- | --- | --- |
| PDC-P3-D01 | Dual-owner findings resolved in [`PDC_OWNERSHIP_MATRIX.md`](./PDC_OWNERSHIP_MATRIX.md): D1-R020, D1-R021, D2-R017, D3-R023, D7-R022, D8-R017, D8-R018 | P2 evidence; DUP-X has no conflicting disposition |
| PDC-P3-D02 | **D8 artifact lifecycle is a FEATURE pair**, not a new domain (Principles §14 vs §15). The doc pair is created after OD-PDC-ADR-01 (PDC-P3-06) | PDC-P1-F03 |
| PDC-P3-D03 | **D6 security docs** become a satellite plus a responsibility matrix under existing domain pairs, not a new domain or policy authority | PDC-P1-F02 |
| PDC-P3-D04 | **ECAP §16 impact (PDC-P3-07):** ECAP needs no new domain sections. Evaluation (D5) is a behavioral mechanism already covered by the FRZ families it maps to. Artifacts (D8) are a feature coordinating existing owners. If OD-PDC-ADR-01 accepts the feature pair, a one-line ECAP traceability note for D8 is added in P4 | ECAP §16 triggers only for a new platform domain or a new FRZ criterion; PDC adds neither |
| PDC-P3-D05 | Runtime hub registration of `BACKGROUND_TASKS` and `COLLABORATIVE_WORK` (PDC-P1-F01) goes to the PDC-P5-DOC lane | Hub rule "classified exactly once" |

## Open Operator Decisions at this gate

OD-PDC-SCOPE-01..07, OD-PDC-PATHB-01, OD-PDC-ADR-01, OD-PDC-IIAC-SCOPE, OD-PDC-FRZ-01, OD-PDC-ARCHF-01 and OD-PDC-SEQ-01: see [`PDC_OPERATOR_DECISIONS.md`](./PDC_OPERATOR_DECISIONS.md).

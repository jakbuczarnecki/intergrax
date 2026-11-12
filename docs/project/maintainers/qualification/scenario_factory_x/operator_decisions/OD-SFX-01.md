# OD-SFX-01 — SCENARIO-FACTORY-X placement and stage registration

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-01 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **APPROVED** |
| **Provenance** | Transcribed by the SFX orchestrator session as a labelled English rendering of the operator's Polish chat. Not operator-signed. |
| **Ticket** | `governance/tickets/T-IGX-037-A.md` |

## Question

Where is SCENARIO-FACTORY-X registered, and how does it relate to SEF-X, SCENARIO-READINESS-X, SCENARIO-GATE and PROD-Q?

## Decision (English rendering of the operator's chat)

SCENARIO-FACTORY-X (SFX) is a new, standalone roadmap stage and the first strategic product priority after ARCH-FREEZE. SFX is the product layer covering Scenario Studio, Business Discovery, Scenario Specification, Blueprint, Capability Resolution, Profile Composition, Integration Setup, Scenario Registry, Evaluation, Reporting and ROI. SEF-X provides the engineering engine for generating, implementing, testing and delivering scenarios; SFX uses SEF-X and does not duplicate its responsibilities. SCENARIO-READINESS-X provides the Scenario Delivery Kit mechanisms and patterns; SFX uses them and does not take over their responsibility. Intergrax Runtime remains the only authorized scenario execution environment. Place SFX directly after ARCH-FREEZE as the highest product priority. SFX implementation requires formal closure of ARCH-FREEZE. SFX does not block the start or closure of SCENARIO-READINESS-X. Preserve the approved SEF-X and SCENARIO-READINESS-X parallelism (OD-SEF-02). SFX may be developed in parallel with SEF-X, but functionality using SEF-X requires ready and accepted corresponding contracts. SFX does not replace SCENARIO-GATE or PROD-Q qualification. Goal: build a complete, production-grade Scenario Factory that turns a customer's business problem into a working, tested and demonstrable solution, using existing Intergrax mechanisms to the maximum. Do not create a second engineering factory, runtime engine or parallel governance mechanisms.

## Operator line

**Provenance:** signed by an operator-designated signing session under `docs/project/maintainers/qualification/scenario_factory_x/operator_decisions/DELEGATION-SFX-OD-SIGNING.md@7dab0e14c71c0a6f42a0c7bed5f4cb9d52355ad1` (IIAC Amendment 01, commission given in chat 2026-11-12; operator answer "Sesja podpisująca"). The line below is in the plain form because it is written by that session; it is not claimed to be operator-written. **Transcription check:** compared with the operator's own free-text answer to the placement question in the SFX planning chat transcript (session "Scenario Factory-X campaign planning", 2026-11-12), read by this session through the local session transcript (answer at 2026-11-12T11:53Z). Every point of the rendering (product-layer scope, SEF-X, SCENARIO-READINESS-X and Runtime relations, roadmap dependencies incl. OD-SEF-02, goal, no second factory/runtime/governance) is present in the Polish original; judged a faithful rendering with no widening.

```text
Approved: OD-SFX-01 @ 2026-11-12T14:10:14+01:00
```

**Explicit non-actions:** does not authorize SFX-P5 or later, does not accept any ADR, does not change FRZ, does not satisfy GATE-SFX-ENTRY, and does not bypass SCENARIO-GATE or PROD-Q.

# OD-SFX-01 — SCENARIO-FACTORY-X placement and stage registration

| Field | Value |
| --- | --- |
| **Decision ID** | OD-SFX-01 |
| **Stage** | SCENARIO-FACTORY-X |
| **Status** | **RECORDED — SIGNATURE PENDING** (decision given by the operator in the planning chat, 2026-11-12) |
| **Provenance** | Transcribed by the SFX orchestrator session as a labelled English rendering of the operator's Polish chat. Not operator-signed. |
| **Ticket** | `governance/tickets/T-IGX-037-A.md` |

## Question

Where is SCENARIO-FACTORY-X registered, and how does it relate to SEF-X, SCENARIO-READINESS-X, SCENARIO-GATE and PROD-Q?

## Decision (English rendering of the operator's chat)

SCENARIO-FACTORY-X (SFX) is a new, standalone roadmap stage and the first strategic product priority after ARCH-FREEZE. SFX is the product layer covering Scenario Studio, Business Discovery, Scenario Specification, Blueprint, Capability Resolution, Profile Composition, Integration Setup, Scenario Registry, Evaluation, Reporting and ROI. SEF-X provides the engineering engine for generating, implementing, testing and delivering scenarios; SFX uses SEF-X and does not duplicate its responsibilities. SCENARIO-READINESS-X provides the Scenario Delivery Kit mechanisms and patterns; SFX uses them and does not take over their responsibility. Intergrax Runtime remains the only authorized scenario execution environment. Place SFX directly after ARCH-FREEZE as the highest product priority. SFX implementation requires formal closure of ARCH-FREEZE. SFX does not block the start or closure of SCENARIO-READINESS-X. Preserve the approved SEF-X and SCENARIO-READINESS-X parallelism (OD-SEF-02). SFX may be developed in parallel with SEF-X, but functionality using SEF-X requires ready and accepted corresponding contracts. SFX does not replace SCENARIO-GATE or PROD-Q qualification. Goal: build a complete, production-grade Scenario Factory that turns a customer's business problem into a working, tested and demonstrable solution, using existing Intergrax mechanisms to the maximum. Do not create a second engineering factory, runtime engine or parallel governance mechanisms.

## Operator line

_Pending. To be written by the operator, or by a dedicated signing session the operator commissions (IIAC Amendment 01). The orchestrator, its implementer subagents and reviewers never write this line._

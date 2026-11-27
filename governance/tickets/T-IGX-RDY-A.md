# T-IGX-RDY-A — `SCENARIO-READINESS-X` Scenario Delivery Velocity & Boundary Integration Certification

## Scope

Execute `SCENARIO-READINESS-X` (`P0–P4`) per [`governance/tickets/plans/PLAN_T-IGX-RDY-A.md`](plans/PLAN_T-IGX-RDY-A.md) and [`docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md`](../../docs/project/maintainers/plans/PLATFORM_ENTERPRISE_COMPLETION_ROADMAP.md) §3.0.7:

- **P0 (Pre-Freeze Prep, `production delta = 0`):** Canonical execution plan (`PLAN_T-IGX-RDY-A.md`) and draft bootstrap ticket (`T-IGX-RDY-A.md`) scaffolded pre-freeze under `T-IGX-090-D` (`OP-EC-01` / `OD-W2-SFX-PREP=A`) to eliminate Wave 6 cold-start delay to `SCENARIO-GATE`.
- **P1–P4 (Post-Freeze Execution — BLOCKED until `ARCH-FREEZE = CLOSED` and Operator approval):**
  - Build the synthetic reference business settlement/reconciliation boundary scenario as a thin, separately deployable Tier-3 component (`core platform changes = 0`, `new platform mechanisms = 0`).
  - Prove boundary adapter swap without changing core scenario logic (`RDY-EC-06`) and representative alternate infrastructure/provider configuration (`RDY-EC-23..25`, Windows C10/C15 compliant).
  - Measure all 11 mandatory delivery velocity metrics (`Time to First Running Scenario`, `Time to Reproducible Local Deployment`, etc.), inventory and classify every friction point against the 6-class taxonomy, and publish the `Intergrax Scenario Delivery Kit` standardization analysis (implementation deferred).
  - Hand off exact-SHA evidence bundle to a separate independent audit conversation for `SCENARIO-READINESS-X` closure prior to `SCENARIO-GATE`.

Hard dependency: **`ARCH-FREEZE = CLOSED`** (`FREEZE_CERTIFICATION_SHA` pinned) ONLY — independent of `SEF-X` and `SCENARIO-FACTORY-X` (OD-SEF-02 Option 3).

## Plan digest

Plan digest: ce549cb742897bf84476c7a1573976fa4fb9366ce67046f51791b8a1215a040b

## Approved

_Pending post-freeze Operator approval after `ARCH-FREEZE = CLOSED` (`FREEZE_CERTIFICATION_SHA` pinned) and signing of `OD-RDY-01..04`. Do not start P1 implementation until the Operator records an `Approved:` line on this ticket._

## Test contract

Post-freeze verification gates (serial execution only; zero `pytest` runs during P0 pre-freeze scaffold):

- `git diff <FREEZE_CERTIFICATION_SHA> -- intergrax/` returns empty (`RDY-EC-01`, frozen core delta = 0).
- Serial `uv run pytest` covering typed input/output contracts, all 7 deterministic settlement classifications, boundary adapter swap, optional HITL, evidence persistence, cross-tenant negative isolation, dual alternate-infrastructure configuration parity, and full automated E2E (`RDY-EC-03..16`, `RDY-EC-22..25`).
- `EVOL-X` / `ENT-AUDIT-X` drift & S8 conformance verification.
- `uv run python -m scripts.governance.igx_gov ticket` before push / independent audit handoff.

## Out of scope

- Any modification to frozen Tier-0 / Tier-1 core (`intergrax/`) — if required, halt immediately with `STOP — FROZEN ARCHITECTURE VIOLATION`.
- Any production code changes prior to `ARCH-FREEZE = CLOSED` and Operator `Approved:` on this ticket.
- Customer production rollout or opening scenario-driven platform development (remains gated behind `SCENARIO-GATE`).
- Implementation of `Intergrax Scenario Delivery Kit` templates (analysis only in this stage).
- Mandatory Docker migration or new vendor integrations (C10/C15 Windows environment freeze preserved).
- Declaring `CLOSED`, `SCENARIO-GATE PASS`, `IIAC PASS`, `FRZ PASS`, or `G6 PASS` from an implementer, tester, or reviewer session.

## Lifecycle state

`plan_pending`

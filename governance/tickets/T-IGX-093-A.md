# T-IGX-093-A — AHE W1-B01 canonical ADAS contracts and candidate archive

## Scope

Implement roadmap package **AHE-W1-01**: canonical ADAS typed contracts (§13.0–§13.7 identities) and append-only candidate archive foundation for the configuration-only LAB profile. Obligations **AHI-ADAS-10.1** through **AHI-ADAS-10.14**. Total engineering scope: **eleven files** published as proposed paired candidates **W1-B01-P** (product, seven files) and **W1-B01-D** (docs, four files) under finding **F-AHE-08** — approved under **OD-AHE-03** for this ticket only.

Canonical plan: [`plans/PLAN_T-IGX-093-A.md`](plans/PLAN_T-IGX-093-A.md).

## Plan digest

Previous plan digest (historical; unapproved): eb98e85bc95600ee626570ab249cfac73d7b9ac3012aff9d08f68b25f630ce16

Plan digest: b4231abab9cf00ae44e72319d22a2d8f7ba0e3b5f1e62ceb86dfb3c07048f980

## Approved

Approved: operator @ 2026-11-27T15:44:43+01:00

**OD-AHE-03 - APPROVED:** Operator approves the complete presented scope and the exact plan digest b4231abab9cf00ae44e72319d22a2d8f7ba0e3b5f1e62ceb86dfb3c07048f980, including the limited paired-publication exception. This timestamp records the decision in the repository; it is not asserted to be the original chat submission time. No production promotion, W1-B02 implementation or governance bypass is authorized. The product candidate is limited to the seven approved paths; the four documentation paths require a separate later publication record after product integration. The architect's scoped AHE-G1 acceptance becomes effective for W1-B01-P only after canonical publication authorization and successful publisher preflight. All independent review and integration requirements remain mandatory.

Neither OD-AHE-01 nor OD-AHE-02 constitutes approval of this exact implementation plan.

## Test contract

Sequential execution (preserve exit codes and evidence):

- `uv run --no-sync pytest -q tests/unit/runtime/adaptive/agent_design_search/`
- `uv run --no-sync pytest -q tests/unit/runtime/adaptive/ --ignore=tests/unit/runtime/adaptive/agent_design_search/`
- `uv run --no-sync ruff check intergrax/runtime/adaptive/agent_design_search/ tests/unit/runtime/adaptive/agent_design_search/`
- `uv run --no-sync pyright intergrax/runtime/adaptive/agent_design_search/`
- `uv run --no-sync python -m compileall -q intergrax/runtime/adaptive/agent_design_search/`
- `uv run --no-sync python -m scripts.governance.igx_gov ticket`

## Out of scope

Execution engine, strategy implementation, static gate, evaluation runner, search controller, experiment bridge, production activation or promotion; modifications to parent adaptive runtime package exports or existing profile stores; ARCH-FREEZE, Scenario Factory runtime, Intelligence master plan; governance publication authorization file changes; G6 or IIAC certification claims by implementer; approving the paired publication exception; registering product or documentation publication authorization; product implementation in this preparation batch; single mixed-kind candidate publication.

## Governance execution evidence

Historical `igx_gov ticket` **FileNotFoundError** from prior preparation execution evidence is preserved as historical diagnostic only (not relabeled PASS).

## Preparation checkpoint

**AHE-W1-B01-PREP-C01:** publication compatibility recorded as **F-AHE-08**; amended plan proposes paired candidates; lifecycle remains `plan_pending`; no `Approved:` line.

## Lifecycle state

plan_approved

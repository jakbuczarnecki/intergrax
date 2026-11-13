# PLAN T-IGX-037-F — SCENARIO-FACTORY-X pre-freeze documentation closeout

Parent: SFX master plan R1 (`governance/tickets/plans/SFX-MASTER-PLAN.md`). Wave W2 closeout, documentation only.

Target status of this campaign: `SFX_PRE_FREEZE_DOCS_CLOSED`. The status `SFX_PRE_IMPLEMENTATION_READY` is a separate, later operator decision. This campaign does not grant it, does not sign it and does not set `PRE_IMPLEMENTATION_READY_SHA`.

Authority:
- **OD-SFX-04** and **OD-SFX-15** (pre-freeze scope: documentation, research, design, proposal-only IDE toolkit).
- The operator's scope correction of 2026-11-13, transcribed in **OD-SFX-21** (recorded, not signed by the agent). The closeout plan called this record `OD-SFX-17`. That number is already used by a PROPOSED decision in `SFX_GAP_ANALYSIS.md`, so the record takes the next free number.

Starting state: `origin/development` at `364b3892544535e0969c28cfdcc0d308d565af3f` (merge of PR #502). The plan was written at `dbc6102be`; the only change since is a PDC-X documentation merge. MCP-SERVER-X, PDC-X and ARCH-FREEZE states are unchanged.

## Scope

1. Reconcile the SFX index, gap register and SHA ladder text with the published merges of PR #482 to #500. Rungs are set only in a pin commit after the merge. `PRE_IMPLEMENTATION_READY_SHA`, `FREEZE_*`, `ADR_ACCEPTANCE_SHA` and `GATE_SFX_ENTRY_SHA` stay `_unset_`.
2. Record OD-SFX-21 and annotate the pre-freeze criterion of unit SFX-P3-01 (human round 0 and TTFS leave the documentation closure). `NOT_RUN` results stay `NOT_RUN`. SFX-P12 and the OD-SFX-07 requirements are not changed.
3. Public documentation: new `docs/project/overview/SCENARIO_FACTORY.md`, a README subsection and strategic-directions row, and one-line routes in the documentation map, project README, use cases, Autonomous Organization overview, proofs, positioning contract and the capability architecture page. The first-screen action stays Try LKW.
4. Integrate the six operator-supplied PNG files (`scenario-factory-{hero,journey,ownership}-{light,dark}.png`) with `<picture>` blocks and full-size wrapper pages. No image is generated, redrawn or edited by an agent.
5. One independent review of the candidate before merge.

## Out of scope

- Any SFX code, Studio, services, persistence, handoff, MCP product endpoints, `agent-env/`, `scripts/`, `sync_agent_env.py`, `role_registry.yaml`, freeze contracts.
- ADR acceptance, FRZ or IIAC promotion, `GATE-SFX-ENTRY`, SFX-P4-B and SFX-P5+, merges.
- Human usability sessions, recruitment, TTFS with people. The operator walkthrough of IDE toolkit v0, GEN content confidentiality and the five known `roadmap_replay_x` group C failures stay open and are not touched.
- Editing the pixels of any image. A failed visual check blocks visual publication and returns a concrete correction list to the operator's image author.

## Units

| Unit | Content |
| --- | --- |
| C0 | Re-check remote HEAD, ancestry of the SFX merges, stage states, asset availability. Fresh worktree from `origin/development`. |
| C1 | Index, gap register (rows whose evidence is already on `development`), SHA ladder text. |
| C2 | OD-SFX-21 and criteria annotations. |
| C4 | Public documentation, assets, tests, ticket F and this plan. |
| C5 | Independent reviewer session on the candidate, before merge. The implementer does not issue PASS. |
| C6 | Pin commit after merge: SHA ladder rungs and the closure record. |
| C7 | STOP brief. The next product unit waits for ARCH-FREEZE, ADR acceptance, OD-SFX-05/09/14, inventory re-pin and `GATE-SFX-ENTRY`. |

## Test contract

```text
uv run pytest <public docs contract tests touched> tests/unit/governance tests/qualification/roadmap_replay_x -q
uv run python -m scripts.governance.igx_gov ticket
```

`roadmap_replay_x` is expected to show the 5 known group C failures and no new ones. `verify_plan_approval` for this ticket runs only after an operator `Approved:` line exists.

## Stop conditions

The diff enters SFX code, `agent-env/`, the generator or the role registry; an ADR becomes Accepted; FRZ or IIAC is promoted; `PRE_IMPLEMENTATION_READY` or its SHA appears; an agent writes an operator signature or an `Approved:` line; a `NOT_RUN` becomes `PASS`; SFX-P12 or OD-SFX-07 wording changes; an agent draws or edits an image; the session works on the local `development` checkout at `7172f294` or on a PDC, MCP or ARCH-FREEZE branch.

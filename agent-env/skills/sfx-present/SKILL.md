---
name: sfx-present
description: SFX Present step - reviews the full draft set with the sfx-critic persona and writes presentation.md under .tmp/sfx/<slug>/, a plain summary of the proposal, its gaps and the next human decision; claims no approval, run result or value (proposal-only, L1).
---

# SFX present (Present)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

Give the user and the business owner one readable summary of the proposed scenario, with an independent review attached, so they can decide what happens next. The presentation is a proposal summary. It is not a run report or a value report: those need sandbox runs (SFX-P9+), which v0 cannot do.

## When to use

- After `sfx-test-plan`; `sfx-session` at stage `PRESENT`.
- The user asks "show me where this scenario stands".

## Inputs

- Every draft in `.tmp/sfx/<slug>/`: the seven contract drafts (`business_problem_statement`, `discovery_record`, `business_scenario_specification`, `capability_resolution_report`, `scenario_blueprint`, `integration_setup_plan`, `scenario_test_plan`, each `.v1.json`), the companion notes, `open_questions.md`, `blueprint_diff.md` if present, and `sfx_session_state.v1.json`.
- The schemas in `docs/project/capabilities/plan/satellites/scenario_factory/design/schemas/` (for the critic).

## Persona brief

`agent-env/agents/sfx-critic.md`. Run it as a **separate subagent** that did not write the drafts. If the tool has no subagents, do not review the drafts yourself; write "independent review not done" in the presentation.

## Procedure

1. List the drafts found and the drafts missing. A missing draft is shown as missing, never summarized from memory.
2. Spawn `sfx-critic` on the full draft set. It returns findings rated `BLOCKER`, `MAJOR`, `MINOR`, `INFO` and may write `critic_findings.md`.
3. Write `presentation.md` with these sections, each short and in plain business language:
   - **Status:** "Proposal (L1 draft). Not approved, not built, not tested, not certified." Lifecycle `DRAFT`.
   - **Problem and objective:** from the problem statement and the specification.
   - **What the scenario does:** actors, main flow, human approval points, deterministic rules versus LLM steps.
   - **Reuse:** the proposed G-1 reuse ratio, resolved capabilities with their owner paths, unresolved requirements.
   - **Integrations:** each integration with its sandbox adapter kind, its `secret_ref` names, and its open PROD-Q obligations.
   - **Test plan:** case counts by kind, adversarial coverage, uncovered criteria. No results exist.
   - **Review:** critic finding counts by severity and every `BLOCKER` in one line each.
   - **Open questions:** the blocking ones first, with who can answer them.
   - **Next decision:** a human approval of an exact blueprint version, outside SFX v0 (planned: API / MCP approval request, SFX-P10). Build, sandbox run, catalog and timeline need SFX-P5+ (STOP).
4. With any `BLOCKER`, say so first and offer `sfx-blueprint` (Improve) or the owning step skill. Never present a blocked proposal as ready.
5. Update `sfx_session_state.v1.json` (stage `PRESENT`, snapshot `DRAFT`) when `sfx-session` is not running.

## Outputs

- `.tmp/sfx/<slug>/presentation.md` (companion summary, not a contract)
- `.tmp/sfx/<slug>/critic_findings.md` (written by the critic, optional)

## Validation

Every number in the presentation comes from a draft (reuse ratio from `resolution_notes.md`, counts from the test plan). Every capability named carries its owner path. The text contains no approval, pass, certification or admission wording. Nothing is hashed; digests remain placeholders until SFX-P5 tooling computes them.

## Handoff

- `BLOCKER` findings: the owning step skill, or `sfx-blueprint` for Improve.
- Otherwise: the human decision outside v0. `sfx-session` can resume later from the session state.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any product or tracked path.
- Run any runtime, scenario, MCP tool or external call.
- Write `scenario_run_report`, `scenario_value_report`, `scenario_approval_ref`, `scenario_qualification_record`, `scenario_admission_dossier` or `scenario_catalog_entry` drafts.
- Claim approval, certification, sandbox qualification, a test result, a measured value or production admission.
- Review the drafts in the same session that wrote them and call it independent.
- Quote a secret value, or name a capability without its owner path.

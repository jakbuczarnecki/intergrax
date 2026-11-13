---
name: sfx-session
description: SFX guided session - runs the Scenario Factory lifecycle step by step with the user, one question at a time, resuming from .tmp/sfx/<slug>/sfx_session_state.v1.json; writes proposal-only L1 drafts and explains which steps are not available before SFX-P5.
---

# SFX session (full lifecycle, IDE toolkit v0)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply to every step below.

## Purpose

Guide one user through the SFX lifecycle (Describe → Discover → Understand → Design → Configure → Approve → Build → Test → Run → Present → Improve) for one scenario. Each step that has a v0 skill produces L1 drafts under `.tmp/sfx/<slug>/`. Each step without one is explained and skipped.

## When to use

- The user wants to design a new business scenario with SFX, or to continue one.
- The user asks "where was I?" for an SFX scenario.
- Use a single-step skill (`sfx-discover` ... `sfx-present`) instead when the user names one step only.

## Inputs

- `.tmp/sfx/` folder listing (scenario home: one folder per slug).
- `.tmp/sfx/<slug>/sfx_session_state.v1.json` if it exists, plus every draft it references.
- Schema: `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/sfx.sfx_session_state.v1.schema.json`.

## Persona briefs

This skill loads no persona itself. Each step skill names its brief in `agent-env/agents/`. After every drafting step, and always before Present, run `sfx-critic` (`agent-env/agents/sfx-critic.md`) as a separate subagent when the tool supports subagents. If it does not, say that no independent review was done.

## Procedure

1. **Open.** List `.tmp/sfx/`. Ask one question: resume an existing slug, or start a new scenario. For a new scenario, propose a slug (`^[a-z][a-z0-9_]*$`, max 64) from the user's words and ask the user to confirm it.
2. **Resume.** If `sfx_session_state.v1.json` exists, read it and the drafts in `active_artifact_refs`. Tell the user in three lines or fewer: current stage, drafts present, open questions (`pending_question_ids`). Continue at `resume_cursor` (a pending question id or the next step id). Never restart a step whose draft exists; the step skill revises it.
3. **Run the next step** with the matching skill and follow that skill's procedure in this session:

   | `current_stage` | Skill | Drafts after the step |
   |---|---|---|
   | `DESCRIBE`, `DISCOVER` | `sfx-discover` | `business_problem_statement.v1.json`, `discovery_record.v1.json`, `open_questions.md` |
   | `UNDERSTAND` | `sfx-specify` | `business_scenario_specification.v1.json` |
   | `DESIGN` | `sfx-resolve`, then `sfx-blueprint` | `capability_resolution_report.v1.json`, `resolution_notes.md`, `scenario_blueprint.v1.json`, `blueprint_notes.md` |
   | `CONFIGURE` | `sfx-integration-setup` | `integration_setup_plan.v1.json`, `integration_notes.md` |
   | `APPROVE`, `BUILD` | none in v0 (step 5) | none |
   | `TEST` | `sfx-test-plan` (plan only, nothing runs) | `scenario_test_plan.v1.json`, `test_plan_notes.md` |
   | `RUN` | none in v0 (step 5) | none |
   | `PRESENT` | `sfx-present` (runs `sfx-critic`) | `presentation.md`, `critic_findings.md` |
   | `IMPROVE` | `sfx-blueprint` (new version) | `scenario_blueprint.v1.json`, `versions/...`, `blueprint_diff.md` |

4. **Between steps.** Ask one question at a time. After each step, show a short summary of what was drafted and the open questions with `blocking_for` `draft` or `build` (rule, Blocking), then ask whether to continue to the next step. Stop and ask when a question with `blocking_for: draft` remains; never design around it. A `build` blocker (for example any `UNRESOLVED` requirement, listed as an `IQ` id) does not stop drafting, but repeat it in every summary and in Present; the drafts are not ready for Approve while it is open.
5. **Steps without a v0 skill.** Explain the step in two or three lines, name the planned operation, and move on to the next available step. Operation names are not final; they come from the PC-6 operation map (ADR-SFX-004, UX design §7.1).

   | Step | Why not in v0 | Planned door | Plan unit |
   |---|---|---|---|
   | Approve | A human decision binding on an exact artifact digest (`intergrax/contracts/collaborative_decision_binding.py`, wrapped by `sfx.scenario_approval_ref.v1`); no offline path (PC-3) | API / MCP: submit approval request; Studio approval inbox | SFX-P10-01, SFX-P10-02 |
   | Build | Engineering handoff of an approved blueprint digest (`sfx.engineering_handoff_request.v1`) | CLI `sfx handoff` (local preview only); MCP handoff and job status | SFX-P8-01..03 |
   | Run | L2 sandbox run (`sfx.scenario_run_report.v1`) | API / MCP sandbox run (L2) | SFX-P9-01, SFX-P10-02 |
   | Catalog | Catalog sidecar (`sfx.scenario_catalog_entry.v1`) | MCP catalog read | SFX-P11-03 |
   | Timeline | Lifecycle record stream (`sfx.scenario_lifecycle_record.v1`) | MCP lifecycle record read | SFX-P5-05, SFX-P6-03 |

   All of these need the SFX-P5+ service, CLI and MCP, after ARCH-FREEZE CLOSED and GATE-SFX-ENTRY. **STOP** there: do not emulate them, do not write their contracts, and do not hand-write an approval, run report, qualification record or dossier.
6. **Save state** once after every completed step and once when the user stops (not after every question; answered questions are already in the drafts and in `open_questions.md`). Rewrite `sfx_session_state.v1.json` in place with `version` + 1. `version` counts saves (no history is kept) and `created_at` is the time of the latest save:
   - `schema_version` `sfx.sfx_session_state.v1`; `door` `IDE`; `chat_session_id` `ide-local` unless the tool exposes a session id;
   - `current_stage` the stage just reached; `lifecycle_state_snapshot` always `DRAFT`;
   - `active_artifact_refs` one `artifact_version_ref` per current contract draft (`schema_id`, `artifact_id`, `version`, placeholder digest), sorted by `schema_id`;
   - `pending_question_ids` every open `Q-<nn>` and `IQ-<nn>` id from `open_questions.md` (any `blocking_for`), sorted ascending;
   - `resume_cursor`, by this precedence: (1) the first open id with `blocking_for: draft`; else (2) the next step id that has a v0 skill (for example `step:CONFIGURE`); else (3) the first open id with `blocking_for: build`; else `null` when the session is complete.
7. **Close.** At the end, or when the user stops, give the draft folder path, the file list, the open questions, and the next available step. State plainly that the drafts are proposals and that nothing was approved, built, run or certified.

## Outputs

- `.tmp/sfx/<slug>/sfx_session_state.v1.json` (`sfx.sfx_session_state.v1`), plus the drafts written by the step skills.

## Validation

Read `sfx.sfx_session_state.v1.schema.json` and check the state file: every `required` field present, no extra field, `door` and `current_stage` from their enums, ids match `$defs/id`, timestamp and digest patterns. Ask each step skill to validate its own drafts the same way. Record deviations in `open_questions.md` under "Schema deviations".

## Handoff

Next step skill per the table in step 3. After Present, the next real step is a human approval outside v0. Offer `sfx-blueprint` (Improve) if the user wants changes.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any product or tracked path.
- Run `uv`, `python`, `pytest`, a scenario, the scaffold, an executing MCP tool, or call an external system.
- Ask for or store a secret value; use `secret_ref` only.
- Ask several unrelated questions at once, or ask again about something already answered.
- Set `lifecycle_state_snapshot` past `DRAFT`, or claim approval, certification, qualification, a passed test or production admission.
- Emulate Approve, Build, Run, Catalog or Timeline, or write their contract files.
- Name a platform capability without its owner path from the capability inventory.

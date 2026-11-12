---
name: sfx-specify
description: SFX Understand step - turns the discovery record into business_scenario_specification.v1.json under .tmp/sfx/<slug>/, deterministic rules first and every field traced to discovery (proposal-only, L1).
---

# SFX specify (Understand)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

State what the scenario must do: actors, inputs, outputs, business rules, decision points, required integrations, non-functional targets and acceptance criteria. Every field traces to the discovery record.

## When to use

- After `sfx-discover`, with no blocking open question left.
- `sfx-session` at stage `UNDERSTAND`.
- The user corrected the specification (write a new version).

## Inputs

- `.tmp/sfx/<slug>/business_problem_statement.v1.json`, `discovery_record.v1.json`, `open_questions.md`.
- An existing `business_scenario_specification.v1.json`, if any.
- Schema `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/sfx.business_scenario_specification.v1.schema.json`.

## Persona brief

`agent-env/agents/sfx-scenario-architect.md`, specification phase. Run it as a subagent with that file as its brief when the tool supports subagents; otherwise follow the brief in this session.

## Procedure

1. If `open_questions.md` has a `blocking: yes` question, stop and return it to the user.
2. Fill the envelope per the rule's draft conventions. `discovery_record_ref` references the current discovery record draft; `scenario_slug` is the slug.
3. Write `title` and `objective` from the problem statement.
4. `actors` (`actor_id`, `role`, `kind` `HUMAN` | `SYSTEM` | `AGENT`); `inputs` (`input_id`, `description`, `data_classification`); `outputs` (`output_id`, `description`).
5. `business_rules` (`rule_id`, `statement`, `deterministic`). Default to `deterministic: true` (validation, lookup, threshold, routing, state transition). Use `false` only when the rule needs language understanding or generation, and write the one-line justification in the statement.
6. `decision_points` (`decision_id`, `description`, `requires_human_approval`). Mark every human approval explicitly.
7. `integrations_required` (`integration_id`, `purpose`, `direction` `INBOUND` | `OUTBOUND` | `BIDIRECTIONAL`). Names and purposes only; no endpoints, no credentials.
8. `non_functional`: `latency_p95_target_ms` and `volume_per_day` as integers, or `null` when discovery did not settle them (and add an open question).
9. `acceptance_criteria` (`criterion_id`, `statement`, `measured_by`). Each one must be checkable later by a test case.
10. `field_provenance`: one entry per top-level field, `ANSWERED` | `INFERRED` | `DEFAULT` | `MANUAL_EDIT`. Ask the user to confirm the `INFERRED` ones.
11. `authority_level_requested`: `L1`, or `L2` when the scenario is meant for a later sandbox run. Never anything else; this is a request, not a grant.
12. Show the user a short summary (rules split into deterministic and LLM, approval points, open questions) and record corrections as a new version.

## Outputs

- `.tmp/sfx/<slug>/business_scenario_specification.v1.json` (`sfx.business_scenario_specification.v1`)
- New open questions appended to `open_questions.md`.

## Validation

Read the schema and check: `required` fields (24 at the top level), no extra field, enums, `$defs/id` patterns, `scenario_slug` pattern, integers only, `field_provenance` keys matching `^[a-z][a-z0-9_]*$`. List deviations under "Schema deviations" in `open_questions.md`.

## Handoff

- Next: `sfx-resolve` with the specification.
- Optional review: `sfx-critic` as a separate subagent.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any product or tracked path.
- Run any runtime, scenario, MCP tool or external call.
- Design around a blocking open question, or invent an answer the discovery record lacks.
- Bind platform capabilities (that is `sfx-resolve`), or name one without its owner path.
- Put a secret, endpoint or credential into the specification.
- Request `L3`, or claim approval or any state past `DRAFT`.

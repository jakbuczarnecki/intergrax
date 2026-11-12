---
name: sfx-blueprint
description: SFX Design step, part 2, and Improve - composes scenario_blueprint.v1.json and blueprint_notes.md under .tmp/sfx/<slug>/ from resolved platform capabilities only, LAB profile, deterministic steps first; on Improve writes a new version and blueprint_diff.md (proposal-only, L1).
---

# SFX blueprint (Design, part 2 · Improve)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

State how the scenario is assembled from existing platform capabilities: capability bindings, agents, tool allowlist, authority level and LAB profile. The blueprint is a design document; nothing is generated or built.

## When to use

- After `sfx-resolve`; `sfx-session` at stage `DESIGN` (second half).
- Improve: the user wants a change after review or presentation (`sfx-session` stage `IMPROVE`).

## Inputs

- `.tmp/sfx/<slug>/business_scenario_specification.v1.json` and `capability_resolution_report.v1.json` (plus `resolution_notes.md`).
- On Improve: the current `scenario_blueprint.v1.json`, the user's change request, and `critic_findings.md` if present.
- Schema `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/sfx.scenario_blueprint.v1.schema.json`.

## Persona brief

`agent-env/agents/sfx-scenario-architect.md`, blueprint phase. Run it as a subagent with that file as its brief when the tool supports subagents; otherwise follow the brief in this session.

## Procedure

1. Read the specification and the resolution report. Bind only capabilities the report resolved. A requirement that is `UNRESOLVED` gets no binding; list it in the notes and send new needs back to `sfx-resolve`.
2. Envelope per the rule's draft conventions. `specification_ref` and `resolution_report_ref` reference the current drafts. `scenario_slug` is the slug.
3. `workspace_path`: `applications/<slug>_scenario/` (pattern in the schema; the final suffix is open decision OQ-002-1). It names where a later scaffold would write. Never create or touch that path.
4. `capability_bindings`, one per resolved requirement, sorted by `binding_id`: `binding_id`, `requirement_id`, `capability_id` and `resolution` copied from the report (`REUSE_AS_IS` | `REUSE_WITH_CONFIGURATION` | `EXTEND` | `NEW_REQUIRED`), `configuration_ref` (an id of the configuration described in the notes, or `null`).
5. `agents`: `agent_id`, `role`, `lifecycle_state` (`experimental` | `development` | `candidate` | `staging`; never production). Add an agent only where a step genuinely needs language understanding or generation.
6. `tool_allowlist`: explicit, sorted by `tool_id`, each with `side_effect` `NONE` | `READ` | `WRITE`. Every `WRITE` tool needs a matching approval point or a stated reason in the notes.
7. `authority_level`: `L1`, or `L2` as a proposal for a later sandbox run. `environment_profile`: `LAB` (the only allowed value).
8. `integration_plan_ref` and `test_plan_ref`: `null` in v0. The integration and test plans reference the blueprint, not the reverse; linking both ways is left to SFX-P7 tooling.
9. Write `blueprint_notes.md` (companion note): step sequence with the binding of each step, deterministic steps first; each LLM step with its one-line justification, prompt purpose, guardrails and budget (tokens, cost, time, steps); approval steps; data flow and tenant scope; integration placeholders (`secret_ref` names only); the no-duplicate check (no new engine, registry, store, state machine, approval store or governance authority; no `ApplicationManifest` change); "Schema deviations".
10. **Improve:** write the current blueprint to `versions/scenario_blueprint.v1.version-<n>.json`, write the new draft with `version` + 1, and write `blueprint_diff.md` (semantic diff: bindings, agents, tools, authority and notes that changed, with the reason for each).

## Outputs

- `.tmp/sfx/<slug>/scenario_blueprint.v1.json` (`sfx.scenario_blueprint.v1`)
- `.tmp/sfx/<slug>/blueprint_notes.md` (companion note)
- On Improve: `versions/scenario_blueprint.v1.version-<n>.json` and `blueprint_diff.md`

## Validation

Read the schema and check: `required` fields, no extra field in the blueprint or its items, `environment_profile` const `LAB`, enums, `workspace_path` and slug patterns, set-array sort order (`capability_bindings` by `binding_id`, `tool_allowlist` by `tool_id`) and no duplicate ids. Check that every binding's `capability_id` and `resolution` match the report.

## Handoff

- Next: `sfx-integration-setup`, then `sfx-test-plan`.
- `sfx-critic` reviews before `sfx-present`. Approval of the blueprint is a human decision outside v0.

## Must never

- Write outside `.tmp/sfx/<slug>/`; never create `applications/<slug>_scenario/` or any product path.
- Generate code, run the scaffold, or run any runtime, scenario, MCP tool or external call.
- Bind a capability the resolution report does not contain, or name one without its owner path.
- Propose a second execution engine, registry, store, approval store or governance authority, or an `ApplicationManifest` change.
- Propose `L3`, a production profile, live endpoints, or a secret value.
- Overwrite a previous blueprint version, or claim `BLUEPRINTED`, `APPROVED` or any state past `DRAFT`.

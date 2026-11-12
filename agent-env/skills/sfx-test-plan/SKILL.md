---
name: sfx-test-plan
description: SFX Test step (plan only) - drafts scenario_test_plan.v1.json and test_plan_notes.md under .tmp/sfx/<slug>/ with positive, negative, security and adversarial cases and dataset needs; never runs a test (proposal-only, L1).
---

# SFX test plan (Test, plan only)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

Design how the scenario will be proven in a later L2 sandbox: intended behaviour, failure behaviour and deliberate attacks, with the datasets each case needs. Running the plan is not available in v0 (SFX-P9-01, after SFX-P5).

## When to use

- After `sfx-integration-setup`; `sfx-session` at stage `TEST`.
- The specification, blueprint or integration plan changed (write a new version).

## Inputs

- `.tmp/sfx/<slug>/business_scenario_specification.v1.json` (rules, decision points, acceptance criteria, non-functional targets).
- `.tmp/sfx/<slug>/scenario_blueprint.v1.json` and `blueprint_notes.md` (steps, LLM steps, budgets).
- `.tmp/sfx/<slug>/integration_setup_plan.v1.json` and `integration_notes.md` (stubs, emulated failure modes, `secret_ref` names).
- `.tmp/sfx/<slug>/capability_resolution_report.v1.json` (evaluation and conformance capabilities with owner paths).
- Schema `docs/project/capabilities/plan/satellites/scenario_factory/design/schemas/sfx.scenario_test_plan.v1.schema.json`.

## Persona brief

`agent-env/agents/sfx-test-designer.md`. Run it as a subagent with that file as its brief when the tool supports subagents; otherwise follow the brief in this session.

## Procedure

1. Envelope per the rule's draft conventions. `blueprint_ref` references the current blueprint draft. `authority_level`: `L2` (the level the cases will need later) or `L1`; never higher.
2. Use the specification's `criterion_id` values as requirement ids. Every criterion gets at least one `POSITIVE` and one `NEGATIVE` case, or is listed as uncovered in the notes.
3. Write `test_cases`, each with `test_id`, `kind`, `description`, `fixture_ref` (a dataset id from the notes, or `null`), `expected_outcome` and `acceptance_criterion_ids`:
   - `POSITIVE`: the main flow and each meaningful variant;
   - `NEGATIVE`: invalid or missing input, rule violations, rejected approvals, stub errors (timeouts, rate limits, schema drift, partial failure), each with its fail-closed outcome;
   - `SECURITY`: at least one case per adversarial class, with the class first in `description`: prompt injection (direct and indirect), tenant crossover, secret leakage (canary `secret_ref` names, never real values), budget exhaustion (tokens, cost, time, steps on every LLM step), retry (idempotent writes, bounded retries, dead-lettering, no duplicate side effects);
   - `PERFORMANCE`: the latency and volume targets from `non_functional`;
   - `EVAL`: LLM-judged quality, only with a rubric in the notes.
4. `eval_suites`: `suite_id`, `metric` (an id), `target_ref` matching `^OD-SFX-07(:...)?$`. Targets are not fixed here; they reference OD-SFX-07.
5. Write `test_plan_notes.md` (companion note): per case the oracle (deterministic where possible; LLM-judged only with rubric and reason) and the runner capability with its owner path from the resolution report, or `runner: unresolved`; dataset needs per `fixture_ref` (shape, size, synthetic generation method, provenance, labels; no real personal data, no production extracts); coverage (criteria without a case, adversarial classes without a case); "Schema deviations".
6. Tell the user the coverage and the gaps. State that no test has run and no result exists.

## Outputs

- `.tmp/sfx/<slug>/scenario_test_plan.v1.json` (`sfx.scenario_test_plan.v1`)
- `.tmp/sfx/<slug>/test_plan_notes.md` (companion note)

## Validation

Read the schema and check: `required` fields, no extra field in the plan or its items, `kind` and `authority_level` enums, `target_ref` pattern, `$defs/id` patterns, every `acceptance_criterion_ids` entry present in the specification. The plan contains no pass or fail value.

## Handoff

- Next: `sfx-critic` on the full draft set (separate subagent), then `sfx-present`.
- Uncovered criteria go back to `sfx-specify` or `sfx-blueprint`.
- Running the plan needs the SFX-P9 test runner and an L2 sandbox; STOP before it.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any product or tracked path.
- Run a test, the evaluation runner, a scenario, an MCP tool or any runtime.
- Claim that a test passed or failed, or record any result.
- Use a real secret, real personal data or a production extract in a case or dataset.
- Require `L3` or a live endpoint for any case.
- Name a runner or evaluator without its owner path, or claim qualification or any state past `DRAFT`.

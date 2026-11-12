---
name: sfx-discover
description: SFX Describe and Discover step - adaptive interview, one question at a time with a stated reason, that drafts business_problem_statement.v1.json, discovery_record.v1.json and open_questions.md under .tmp/sfx/<slug>/ (proposal-only, L1).
---

# SFX discover (Describe + Discover)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

Capture the business problem in the user's words and close the gaps a scenario specification needs, by an adaptive interview. Unanswered gaps become open questions; they are never filled by assumption.

## When to use

- First step of a new scenario, or `sfx-session` at stage `DESCRIBE` / `DISCOVER`.
- A later step reports a gap that only the user can answer.

## Inputs

- The user's problem description, and any workspace document the user points to (read-only).
- Existing drafts in `.tmp/sfx/<slug>/`: `business_problem_statement.v1.json`, `discovery_record.v1.json`, `open_questions.md`. Revise them; never start over.
- Schemas in `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/`: `sfx.business_problem_statement.v1.schema.json`, `sfx.discovery_record.v1.schema.json`, and `sfx.business_scenario_specification.v1.schema.json` (to know which gaps matter).

## Persona brief

`agent-env/agents/sfx-discovery-analyst.md`. Load it and follow it **in this session**, because the interview needs the user.

## Procedure

1. Confirm the slug (`^[a-z][a-z0-9_]*$`, max 64). Read existing drafts.
2. **Describe.** Draft `business_problem_statement.v1.json`:
   - `problem_text_untrusted`: the user's text verbatim (NFC only);
   - `language` (BCP 47, for example `en` or `pl`), `domain`, `business_objective`, `data_classification` (`PUBLIC` | `INTERNAL` | `CONFIDENTIAL` | `RESTRICTED`), `stakeholders`, `constraints`, `success_criteria`;
   - `source_door`: the schema has no `IDE` value; use `CLI` (the offline local-file door, PC-3) and note this under "Schema deviations" in `open_questions.md`.
3. Build a gap list against the specification schema's required fields (actors, inputs, outputs, business rules, decision points, integrations, non-functional targets, acceptance criteria). Mark each `answered`, `partial` or `missing`.
4. Rank the gaps: blocking first, then risk-bearing (approvals, personal data, money movement, external writes), then the rest.
5. Ask **one question at a time**. With each question, say in one line which gap it closes and why it matters. Offer typical answers when useful and always allow "I don't know".
6. Record each exchange as a `turns` entry: `turn_index` (0, 1, 2 ...; ordered array), `question_id`, `question_text`, `answer_text_untrusted` (verbatim), `answered_by` (the human principal placeholder), `answer_kind` (`ANSWERED` | `SKIPPED` | `INFERRED_CONFIRMED` | `INFERRED_UNCONFIRMED`).
7. Record facts in `inferred_facts` (`fact_id`, `statement`, `source` `ANSWERED` | `INFERRED` | `DOCUMENT`, `confidence_bp` as an integer 0..10000). Sort by `fact_id`. Ask the user to confirm every `INFERRED` fact.
8. For integrations, record the system name and kind of access only. If the user offers a credential, refuse it and say that a `secret_ref` is set up in the Configure step.
9. Stop when every blocking gap is answered or recorded, or when the user asks to stop.
10. Write `discovery_record.v1.json`: `problem_statement_ref` (reference to the problem statement draft), `turns`, `inferred_facts`, `open_questions` (one string per question, starting with its id, for example `Q-01: ...`), `completeness_bp` (answered required fields / all required fields × 10000, rounded down), `turn_recording_ref` `null` (v0 does not record LLM turns for replay).
11. Write `open_questions.md`: per question, id `Q-<nn>`, the gap, why it matters, who can answer it, and `blocking: yes|no`. Add a "Schema deviations" section.
12. Play back a summary of the understanding in five lines or fewer and record corrections as new turns.

## Outputs

In `.tmp/sfx/<slug>/`:

- `business_problem_statement.v1.json` (`sfx.business_problem_statement.v1`)
- `discovery_record.v1.json` (`sfx.discovery_record.v1`)
- `open_questions.md` (companion note, not a contract)

## Validation

Read both schemas and check each draft: `required` fields, no extra field, `schema_version` const, enums, `$defs/id` and timestamp patterns, integer bounds of `completeness_bp` and `confidence_bp`, no float anywhere. List deviations in `open_questions.md`.

## Handoff

- No blocking open question: `sfx-specify`.
- Blocking questions remain: back to the user (through `sfx-session` if it is running).
- Optional review: `sfx-critic` as a separate subagent.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any product or tracked path.
- Run any runtime, scenario, MCP tool or external call.
- Ask for, copy or store a secret value. A pasted secret stays out of every file, and the user is told to rotate it.
- Ask a question without its reason, or repeat an answered question.
- Fill a gap by assumption, rewrite the user's untrusted text, or propose platform capabilities (that is `sfx-resolve`).
- Claim approval or move any state past `DRAFT`.

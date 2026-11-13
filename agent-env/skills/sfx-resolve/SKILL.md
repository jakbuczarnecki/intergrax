---
name: sfx-resolve
description: SFX Design step, part 1 - closed-world mapping of every specification requirement to an existing platform capability with its owner path, drafting capability_resolution_report.v1.json and resolution_notes.md under .tmp/sfx/<slug>/ (proposal-only, L1).
---

# SFX resolve (Design, part 1)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

For each requirement in the specification, find the existing platform capability that serves it, cite its owner path, or record the gap. Closed world: a capability exists only if the capability inventory names it. Scenario-specific business logic is not a platform capability (step 2): the report must not count it as reuse.

## When to use

- After `sfx-specify`; `sfx-session` at stage `DESIGN` (before `sfx-blueprint`).
- `sfx-blueprint` reports a need the report does not cover.

## Inputs

- `.tmp/sfx/<slug>/business_scenario_specification.v1.json` and `discovery_record.v1.json`.
- Capability inventory `docs/project/maintainers/qualification/scenario_factory_x/SFX_CAPABILITY_INVENTORY.json` (read-only; row fields: `capability_id`, `owner`, `contract`, `code_path`, `configuration`, `maturity`, `qualification`, `reuse_class`, `limitations`, `evidence_sha`). Fallback only if it is absent: the master plan §5 seed, labelled "seed, not re-pinned".
- UCA gap vocabulary: `DiscoveryCompletionOutcome` in `intergrax/contracts/capability_catalog/discovery_completion.py` (read-only).
- Schema `docs/project/maintainers/qualification/scenario_factory_x/design/schemas/sfx.capability_resolution_report.v1.schema.json`.

## Persona brief

`agent-env/agents/sfx-capability-resolver.md`. Run it as a subagent with that file as its brief when the tool supports subagents; otherwise follow the brief in this session. This skill holds the exact rules (mapping, owner path, inventory reference, G-1); the brief points here.

## Procedure

1. List the requirements: every `rule_id`, `decision_id`, `integration_id`, `input_id`, `output_id` and `criterion_id` of the specification, plus `nfr:latency` and `nfr:volume`, each **only when its target is not `null`** (a null target is an open question, not a requirement). These ids are the `requirement_id` values. The model-provider integration (`INT-LLM`, see `sfx-specify`) is an ordinary `integration_id` and is resolved like any other.
2. **Classify each requirement before searching.** Test: if the scenario's own rules, thresholds, tables and field names were removed, would the capability still do the job? Yes: the requirement is *platform-served*. No, because the content is the requirement: it is *scenario-specific*.

   | Requirement | Default class |
   |---|---|
   | `rule_id` with `deterministic: true` (validation, lookup, threshold, match, routing); `decision_id` without human approval | scenario-specific logic |
   | `rule_id` with `deterministic: false` (an LLM step); `decision_id` with human approval; `integration_id`; `criterion_id`; `nfr:*` | platform-served |
   | `input_id` / `output_id` delivered by an integration, a store, a queue or a report capability | platform-served |
   | `input_id` that is reference data the scenario must supply (for example a rate table) | scenario-specific data |

   Scenario-specific logic or data is **never** `REUSE_AS_IS` or `REUSE_WITH_CONFIGURATION`, even when a generic row (tool runtime, workflow host, store) can host it. Resolve it as `NEW_REQUIRED`: `capability_id` is the hosting row (the blueprint needs a non-null id) and `evidence_ref` its owner path; the rationale starts `scenario-specific logic; hosted by <row id>` and says what the host does and does not provide. If no row can host it, resolve it as `UNRESOLVED` with `capability_id` `null`. If the hosting row itself maps to `UNRESOLVED` in step 3 (`REQUALIFICATION_REQUIRED`, `BLOCKED`, `DEFERRED_WITH_RATIONALE`), the requirement is `UNRESOLVED` and keeps that row and its owner path. A scenario-specific rule that needs a platform *function* the inventory does not list (for example PDF text extraction) is `UNRESOLVED` with that function as its `IQ` gap: hosting alone does not make it buildable. A dependency on another requirement that is tracked on its own (for example a rule that reads a reference table that is itself a requirement) stays in the notes and does not change the rule's own resolution.
3. For each requirement, search the inventory by capability, owner and SFX use. Do not search the code base for new candidates. A plausible owner missing from the inventory becomes an inventory question (`IQ-<nn>`). Write one `resolutions` entry per requirement, sorted by `requirement_id`:
   - `capability_id`: the inventory row id, or `null` when no row matches;
   - `resolution`: map the inventory `reuse_class` of a platform-served requirement to the schema enum:

     | Inventory class | `resolution` |
     |---|---|
     | `REUSE_AS_IS` | `REUSE_AS_IS` |
     | `REUSE_WITH_CONFIGURATION` | `REUSE_WITH_CONFIGURATION` |
     | `EXTEND_CONTRACT` | `EXTEND` |
     | `NEW_SCENARIO_FACTORY_COMPONENT` | `NEW_REQUIRED` |
     | `REQUALIFICATION_REQUIRED` | `UNRESOLVED` |
     | `DEFERRED_WITH_RATIONALE`, `BLOCKED`, no row | `UNRESOLVED` |

     `REQUALIFICATION_REQUIRED` rule: the row exists but has no pinned qualification, so it is never used as reuse and never upgraded. Keep `capability_id` and `evidence_ref` of the row, set `UNRESOLVED`, and begin the rationale with `qualification_state=REQUALIFICATION_REQUIRED; target_class=<class>.` where `<class>` is the class the row's `limitations` give after "Target class after requalification:" (or `unknown`). Repeat `qualification_state` and `target_class` per requirement in `resolution_notes.md`. Other inventory qualifiers (`LIMITATION`, `MISSING_EVIDENCE`) are carried in the rationale as they stand.
   - `evidence_ref` (owner path): from the matched row's `code_path`, take the first word that begins with `intergrax/`, `agents/`, `applications/`, `scripts/` or `testing_support/` (words are separated by whitespace; ignore a leading `(` or quote and a trailing `,` `;` `)` or quote), keep its `:<line>` suffix if present, and use that string. A row whose `code_path` starts with `No code`, `Verified absent` or `None` has no owner path. `evidence_ref` is `null` only when `capability_id` is `null` or the row has no owner path; a matched but `UNRESOLVED` row keeps its owner path. When another path named in the same `code_path` is the one that serves the requirement, still put the first word in `evidence_ref` and name the other path in the rationale;
   - `rationale` (at most 2000 characters): the inventory class verbatim, its qualifiers, the UCA gap class, one sentence on why the row fits, and for every `UNRESOLVED` entry the gap id (`Gap IQ-<nn>.`).
4. Assign each requirement one UCA gap class (a v0 convention; the vocabulary belongs to `DiscoveryCompletionOutcome`):

   | Situation | UCA gap class |
   |---|---|
   | `REUSE_AS_IS`, platform-served | `direct_reuse` |
   | `REUSE_WITH_CONFIGURATION`, `EXTEND`, `NEW_REQUIRED` (including hosted scenario logic), `REQUALIFICATION_REQUIRED` | `realization_required` |
   | `BLOCKED` row | `blocked` |
   | `DEFERRED_WITH_RATIONALE` row | `scope_unavailable` |
   | Row states the capability is absent at the base commit | `unavailable` |
   | Two rows could serve with different semantics (list both) | `conflict` |
   | No row, and the inventory states the family is complete | `missing_capability` |
   | No row, family not proven complete | `incomplete` |

5. **Track every gap.** Give each distinct gap behind an `UNRESOLVED` entry an inventory-question id `IQ-<nn>` (several requirements may share one). Append one row per `IQ` to `open_questions.md` with `blocking_for` `build` (use `draft` only when `sfx-blueprint` cannot proceed at all), who can answer it, and the requirements it blocks. List the same ids in `resolution_notes.md`. The next session save puts them in `pending_question_ids` (rule, Blocking).
6. `unresolved_count`: the number of `UNRESOLVED` entries.
7. `specification_ref`: reference to the current specification draft. `inventory_ref.inventory_version`: the top-level `inventory_version` string of the inventory JSON, copied verbatim (for example `sfx-capability-inventory.v1+r1`). If the file has no such string, use `<schema_version>:<base_sha>` and say so in the notes. `inventory_ref.inventory_digest`: `sha256:` followed by 64 zeros ("not computed", like `canonical_digest`; the SFX-P5 tooling computes it); say so in the notes.
8. Write `resolution_notes.md` (companion note): per requirement the class (platform-served or scenario-specific), UCA gap class, `qualification_state` and `target_class` where the rule applies, and the proposed disposition (configure, governed acquisition, change request to the owner, requalification, deferral); a no-duplicate justification for each `NEW_REQUIRED` (for hosted scenario logic: that no inventory row implements it, and the hosting row); the `IQ` list; the inventory source (file, `inventory_version`) and any fallback; the proposed G-1 reuse ratio; a "Schema deviations" section.
   - **G-1:** `reuse_ratio_bp = floor(10000 * (n_REUSE_AS_IS + n_REUSE_WITH_CONFIGURATION) / n_requirements)` (integer division, rounded down), where `n_requirements` is the number of `resolutions` entries. Also print the counts of all five `resolution` values, with `n_NEW_REQUIRED` on its own line, so reuse is not read as "share of logic reused".
9. Tell the user the reuse ratio, the unresolved requirements and any blocking gap. Ask about one blocking gap at a time.

## Outputs

- `.tmp/sfx/<slug>/capability_resolution_report.v1.json` (`sfx.capability_resolution_report.v1`)
- `.tmp/sfx/<slug>/resolution_notes.md` (companion note)
- Appended `IQ` rows in `.tmp/sfx/<slug>/open_questions.md`

## Validation

Read the schema and check: `required` fields, no extra field in the report or in any `resolutions` item, `resolution` enum, `$defs/id` pattern for ids and `evidence_ref` (a path with `:<line>` matches), `inventory_version` pattern, digest pattern, `unresolved_count` equal to the `UNRESOLVED` count, sort order by `requirement_id`, no duplicate `requirement_id`, every `UNRESOLVED` rationale names an `IQ` id that exists in `open_questions.md`, and no `REUSE_*` entry whose requirement is scenario-specific.

## Handoff

- Next: `sfx-blueprint` with the report.
- Integration requirements (including `INT-LLM`) are also input to `sfx-integration-setup`.
- Blocking gaps go back to the user. `sfx-critic` reviews before `sfx-present`.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any tracked file, including the capability inventory.
- Run any runtime, discovery service, MCP tool or external call.
- Name a capability, module, class or API that the inventory does not list with an owner path.
- Upgrade an inventory REUSE class, drop a qualifier the row states, or count scenario-specific logic as reuse.
- Claim `missing_capability` from an incomplete lookup.
- Claim approval, qualification or any state past `DRAFT`.

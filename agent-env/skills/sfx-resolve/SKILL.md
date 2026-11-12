---
name: sfx-resolve
description: SFX Design step, part 1 - closed-world mapping of every specification requirement to an existing platform capability with its owner path, drafting capability_resolution_report.v1.json and resolution_notes.md under .tmp/sfx/<slug>/ (proposal-only, L1).
---

# SFX resolve (Design, part 1)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

For each requirement in the specification, find the existing platform capability that serves it, cite its owner path, or record the gap. Closed world: a capability exists only if the capability inventory names it.

## When to use

- After `sfx-specify`; `sfx-session` at stage `DESIGN` (before `sfx-blueprint`).
- `sfx-blueprint` reports a need the report does not cover.

## Inputs

- `.tmp/sfx/<slug>/business_scenario_specification.v1.json` and `discovery_record.v1.json`.
- Capability inventory `docs/project/capabilities/plan/satellites/scenario_factory/SFX_CAPABILITY_INVENTORY.json` (read-only). Fallback only if it is absent: the master plan §5 seed, labelled "seed, not re-pinned".
- UCA gap vocabulary: `DiscoveryCompletionOutcome` in `intergrax/contracts/capability_catalog/discovery_completion.py` (read-only).
- Schema `docs/project/capabilities/plan/satellites/scenario_factory/design/schemas/sfx.capability_resolution_report.v1.schema.json`.

## Persona brief

`agent-env/agents/sfx-capability-resolver.md`. Run it as a subagent with that file as its brief when the tool supports subagents; otherwise follow the brief in this session.

## Procedure

1. List the requirements: every `rule_id`, `decision_id`, `integration_id`, `input_id`, `output_id` and `criterion_id` of the specification, plus the non-functional targets (`nfr:latency`, `nfr:volume`). These ids are the `requirement_id` values.
2. For each requirement, search the inventory by capability, owner and SFX use. Do not search the code base for new candidates. A plausible owner missing from the inventory becomes an inventory question in the notes.
3. Write one `resolutions` entry per requirement, sorted by `requirement_id`:
   - `capability_id`: the inventory row id, or `null` when no row matches;
   - `resolution`: map the inventory REUSE class to the schema enum:

     | Inventory class | `resolution` |
     |---|---|
     | `REUSE_AS_IS` | `REUSE_AS_IS` |
     | `REUSE_WITH_CONFIGURATION` | `REUSE_WITH_CONFIGURATION` |
     | `EXTEND_CONTRACT` | `EXTEND` |
     | `NEW_SCENARIO_FACTORY_COMPONENT` | `NEW_REQUIRED` |
     | `DEFERRED_WITH_RATIONALE`, `BLOCKED`, no row | `UNRESOLVED` |

   - `evidence_ref`: the owner path from the inventory row (for example `intergrax/contracts/...py`), or `null` when unresolved;
   - `rationale` (at most 2000 characters): the inventory class verbatim, its qualifiers (`REQUALIFICATION_REQUIRED`, `LIMITATION`, `MISSING_EVIDENCE`), the UCA gap class, and one sentence on why the row fits.
4. Assign each requirement one UCA gap class from `DiscoveryCompletionOutcome` (`direct_reuse`, `realization_required`, `missing_capability`, `blocked`, `scope_unavailable`, `unavailable`, `conflict`, `incomplete`). Use `missing_capability` only when the inventory family is complete; otherwise `incomplete`.
5. `unresolved_count`: the number of `UNRESOLVED` entries.
6. `specification_ref`: reference to the current specification draft. `inventory_ref`: `inventory_version` and `inventory_digest` as stated in the inventory file; if it states no digest, use the zero placeholder and say so in the notes.
7. Write `resolution_notes.md` (companion note): per requirement the UCA gap class and proposed disposition (configure, governed acquisition, change request to the owner, deferral); a no-duplicate justification for each `NEW_REQUIRED` naming the rejected rows; the proposed G-1 reuse ratio in basis points (`REUSE_AS_IS` + `REUSE_WITH_CONFIGURATION` over all requirements); inventory questions; the inventory source and any fallback; a "Schema deviations" section.
8. Tell the user the reuse ratio, the unresolved requirements and any blocking gap. Ask about one blocking gap at a time.

## Outputs

- `.tmp/sfx/<slug>/capability_resolution_report.v1.json` (`sfx.capability_resolution_report.v1`)
- `.tmp/sfx/<slug>/resolution_notes.md` (companion note)

## Validation

Read the schema and check: `required` fields, no extra field in the report or in any `resolutions` item, `resolution` enum, `$defs/id` pattern for ids and `evidence_ref`, digest pattern, `unresolved_count` equal to the `UNRESOLVED` count, sort order by `requirement_id`, no duplicate `requirement_id`.

## Handoff

- Next: `sfx-blueprint` with the report.
- Integration requirements are also input to `sfx-integration-setup`.
- Blocking gaps go back to the user. `sfx-critic` reviews before `sfx-present`.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any tracked file, including the capability inventory.
- Run any runtime, discovery service, MCP tool or external call.
- Name a capability, module, class or API that the inventory does not list with an owner path.
- Upgrade an inventory REUSE class or drop a qualifier the row states.
- Claim `missing_capability` from an incomplete lookup.
- Claim approval, qualification or any state past `DRAFT`.

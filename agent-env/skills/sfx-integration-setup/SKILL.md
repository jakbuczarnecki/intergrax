---
name: sfx-integration-setup
description: SFX Configure step - drafts integration_setup_plan.v1.json and integration_notes.md under .tmp/sfx/<slug>/ with secret_ref names only, fixture or stub adapters, egress DENY_ALL and the PROD-Q obligations (proposal-only, L1; touches no credential or endpoint).
---

# SFX integration setup (Configure)

Load the rule `agent-env/rules/intergrax-scenario-factory.md` first. Its invariants and draft conventions apply.

## Purpose

Plan how every external system in the blueprint will be configured for a later L2 sandbox, without touching a credential, an endpoint or any configuration file.

## When to use

- After `sfx-blueprint`; `sfx-session` at stage `CONFIGURE`.
- The blueprint gained or changed an integration (write a new version).

## Inputs

- `.tmp/sfx/<slug>/scenario_blueprint.v1.json` and `blueprint_notes.md` (integration placeholders).
- `.tmp/sfx/<slug>/capability_resolution_report.v1.json` (integration rows and owner paths).
- `.tmp/sfx/<slug>/business_scenario_specification.v1.json` (`integrations_required`, data classes, volumes).
- Integration, adapter and secret-store rows of the capability inventory `docs/project/capabilities/plan/satellites/scenario_factory/SFX_CAPABILITY_INVENTORY.json` (read-only).
- Schema `docs/project/capabilities/plan/satellites/scenario_factory/design/schemas/sfx.integration_setup_plan.v1.schema.json`.

## Persona brief

`agent-env/agents/sfx-integration-advisor.md`. Run it as a subagent with that file as its brief when the tool supports subagents; otherwise follow the brief in this session.

## Procedure

1. List every integration in the specification and the blueprint notes. Each one needs a resolved capability row with an owner path; otherwise stop and send it back to `sfx-resolve`.
2. Envelope per the rule's draft conventions. `blueprint_ref` references the current blueprint draft.
3. One `integrations` entry per integration, in the order of `integrations_required`:
   - `integration_id`: the specification's id;
   - `kind`: `FIXTURE_ADAPTER`, `STUB_HTTP` or `LOCAL_VECTOR_STORE` for the L2 sandbox. `VENDOR_SANDBOX` only as a proposal flagged for the operator. Never `LIVE_VENDOR`;
   - `adapter_ref`: the id of the adapter capability from the resolution report (owner path in the notes);
   - `settings`: non-secret settings only. Keys match `^[a-z][a-z0-9_]*$` and contain none of `password`, `secret`, `token`, `apikey`, `api_key`, `credential`, `private_key`. Values are strings, integers or booleans. No URL with credentials, no live host;
   - `secret_refs`: `{"name": ..., "secret_ref": ...}` entries. The `secret_ref` is `<scheme>:<path>` (pattern `^[a-z][a-z0-9+.-]*:[^\s]+$`); take the scheme from the inventory's secret-store row. If the inventory names none, use `secretref:<slug>/<integration_id>/<name>` and record that in the notes. Never a value or an example value;
   - `network_egress`: `{"allowlist": [], "mode": "DENY_ALL"}` (L2 requires deny-all with an empty allowlist);
   - `prod_q_required`: `true` for every integration that will touch a real system later.
4. Write `integration_notes.md` (companion note): per integration the capability row and owner path; the secret-store owner path and the scope and least privilege of each `secret_ref`; the synthetic fixture dataset (no real personal data); the failure modes the stub must emulate (timeouts, rate limits, schema drift, partial failure); the PROD-Q obligations (live qualification slice, credential provisioning by the owner, data processing approval, residency check) with the owner of each, none marked done; items out of SFX scope (new vendor integration, production configuration) flagged for the operator; "Schema deviations".
5. Tell the user which integrations are planned, which need the operator, and that credentials are provisioned later by their owners, never in this session.

## Outputs

- `.tmp/sfx/<slug>/integration_setup_plan.v1.json` (`sfx.integration_setup_plan.v1`)
- `.tmp/sfx/<slug>/integration_notes.md` (companion note)

## Validation

Read the schema and check: `required` fields, no extra field in the plan or its items, `kind` and `network_egress.mode` enums, settings key pattern and forbidden key words, `secret_ref` pattern, egress `DENY_ALL` with an empty allowlist. Scan every file you wrote for credential-shaped strings; if one appears, remove it and tell the user.

## Handoff

- Next: `sfx-test-plan`, with the emulated failure modes.
- `sfx-critic` reviews before `sfx-present`.

## Must never

- Write outside `.tmp/sfx/<slug>/`, or edit any tracked path, environment file or local configuration.
- Run any runtime, connectivity probe, MCP tool or external call.
- Ask for, accept, copy or store a secret value. A pasted secret stays out of every file, and the user is told to rotate it.
- Propose live endpoints, production credentials, `LIVE_VENDOR`, or egress allowances for the L2 sandbox.
- Name an adapter, profile or secret store without its owner path from the resolution report or the inventory.
- Mark a PROD-Q obligation as done, or claim qualification, approval or any state past `DRAFT`.

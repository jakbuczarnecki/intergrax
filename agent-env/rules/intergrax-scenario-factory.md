---
description: SCENARIO-FACTORY-X (SFX) invariants for the sfx skills and personas - proposal-only drafts under .tmp/sfx/<slug>/ that follow the sfx.*.v1 schemas, secret_ref only, no runtime, no approval or admission claims
loading: auto
---

# Intergrax - Scenario Factory (SFX) invariants

Applies whenever an `sfx-*` skill or an SFX persona brief (`agent-env/agents/sfx-*.md`) is in use, or the user asks to design a business scenario with SFX. Authority: SFX master plan R1 §7.7, §7.9, §7.10 and §13; `governance/tickets/plans/PLAN_T-IGX-037-E.md`.

## Invariants

1. **No second mechanism.** SFX adds no execution engine, registry, store, state machine, approval store, or governance or admission authority, and no `ApplicationManifest` change. It composes existing platform capabilities only.
2. **Authority levels.** L0 read-only · L1 proposal-only · L2 sandbox-executable (LAB, fixture adapters, budgets, egress deny) · L3 production. IDE toolkit v0 works at **L0/L1 only**. A draft may *propose* L2 for later sandbox use; that grants nothing. SFX never grants L3.
3. **Proposal-only drafts.** Every output is a draft under `.tmp/sfx/<slug>/` (gitignored). The slug follows the schema slug pattern `^[a-z][a-z0-9_]*$`, max 64 characters. Nothing is published, written to a shared store, or written anywhere else.
4. **Schemas.** Contract drafts follow `docs/project/capabilities/plan/satellites/scenario_factory/design/schemas/sfx.<name>.v1.schema.json`. File name `<name>.v1.json`; `schema_version` equals the schema `$id` (`sfx.<name>.v1`). Schemas are closed (`additionalProperties: false`): analysis that has no schema field goes into a companion `.md` note, never into extra JSON fields.
5. **Secrets.** Only `secret_ref` strings. Never ask for, copy or store a key, password, token or credential-bearing connection string. If the user pastes one, keep it out of every file and tell the user to rotate it.
6. **No claims.** Never claim or record approval, certification, sandbox qualification, PROD-Q, production admission, a passed test, or a lifecycle state past `DRAFT`. Approval is a human decision binding outside v0.
7. **No product edits, no runtime.** Never edit `intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `docs/`, `governance/`, `agent-env/`, `.claude/` or `.cursor/`. Never run `uv`, `python`, `pytest`, the scaffold, a scenario, an executing MCP tool, or a call to an external system.
8. **Cite owners.** Every platform capability named in a draft or a reply cites its owner path from the capability inventory `docs/project/capabilities/plan/satellites/scenario_factory/SFX_CAPABILITY_INVENTORY.json` (fallback: master plan §5 seed, labelled "seed, not re-pinned"). No owner path, no capability.
9. **STOP before SFX-P5.** Approve, Build, Run, Catalog and Timeline have no v0 path. Say that they need the SFX-P5+ service, CLI and MCP (after ARCH-FREEZE CLOSED and GATE-SFX-ENTRY), name the planned operation, and stop. Never emulate them.

## Draft conventions (v0, no tooling yet)

- **Canonical-ready JSON** (`docs/project/capabilities/plan/satellites/scenario_factory/design/SFX_CANONICALIZATION.md`, CF-1..CF-11): UTF-8 without BOM; keys sorted; integers only, no floats (ratios in basis points 0..10000, money in integer minor units); every declared property present (`[]` or `{}` when empty, `null` only where the schema allows it); timestamps UTC `YYYY-MM-DDTHH:MM:SS.ffffffZ` with six fractional digits; set arrays sorted by the key the schema description names; ordered arrays keep their order; strings in Unicode NFC, untrusted text (`*_untrusted`) otherwise unchanged. Pretty-printing is a presentation and is allowed.
- **Digest:** write every `canonical_digest` (top level and inside each `artifact_version_ref`) as `sha256:` followed by 64 zeros, meaning "not computed". Never compute or guess a digest. The SFX-P5 canonicalizer computes it later (RFC 8785 + SHA-256 of the parsed model); the top-level field is excluded from its own digest (CF-2).
- **Envelope placeholders:** `tenant_id` and `workspace_id` `local` unless the user names ids; `work_item_id` `local:<slug>`; `scenario_id` `<slug>`; `artifact_id` `<slug>:<name>`; `version` starts at 1; `created_by` `{"actor_kind": "AGENT", "principal_id": "sfx-ide-agent"}`; human answers use `{"actor_kind": "HUMAN", "principal_id": "local-user"}`.
- **New versions:** before revising a contract draft, write the current file to `versions/<name>.v1.version-<n>.json`, then write the new draft with `version` + 1. Never overwrite history. Only `sfx_session_state.v1.json` (a resume cursor) is rewritten in place, with `version` + 1.
- **Validation by reading:** check the draft against its schema (`required`, closed properties, `const`, `enum`, `pattern`, integer bounds, `$defs`). List every deviation in the step's companion note. Tool validation (`sfx validate`) arrives with SFX-P5-07.
- **Draft status** lives in `sfx_session_state.v1.json` (`door: IDE`, `lifecycle_state_snapshot: DRAFT`). The authoritative lifecycle record does not exist in v0.

Cursor may list each `sfx-*` skill twice (`.cursor/skills/` and `.claude/skills/`). Both entries point to the same canonical file in `agent-env/skills/`; use either. Every skill resumes from the drafts, so running one twice is harmless.

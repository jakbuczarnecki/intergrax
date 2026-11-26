---
trigger: model_decision
description: SCENARIO-FACTORY-X (SFX) invariants whenever sfx-* skills, agent-env/agents/sfx-*.md personas, or business scenario drafting under .tmp/sfx/<slug>/ are in use.
---

# Scenario Factory (SFX) Invariants (Antigravity adapter)

Canonical rule (SSOT): `agent-env/rules/intergrax-scenario-factory.md`

- L0/L1 proposal-only drafts under `.tmp/sfx/<slug>/` only; strictly follow closed `sfx.*.v1.schema.json` schemas and `secret_ref` only (never real credentials).
- Never edit product/governance/tool trees (`intergrax/`, `agents/`, `applications/`, `scripts/`, `tests/`, `docs/`, `governance/`, `agent-env/`, `.claude/`, `.cursor/`, `.agents/`) and never execute `uv`/`python`/`pytest` during SFX drafting.
- Read `agent-env/rules/intergrax-scenario-factory.md` in full whenever any `sfx-*` skill or scenario drafting session runs.

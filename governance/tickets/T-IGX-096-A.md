# T-IGX-096-A — Codex project environment setup

## Scope

Provide Codex-native guidance, typed agent profiles, all shared skills and
on-demand rules, invocation policy, lifecycle guards, and a repeatable sync/check.
Keep existing Claude Code, Cursor, Antigravity, canonical rules, and role authority
unchanged. No product runtime, remote delivery, secrets, or unrelated user settings.
Install only the new local Codex plugin; keep it enabled in this project only.

## Plan digest

Operator-requested environment setup in this Codex chat, 2026-11-28.

## Approved

Approved: operator request in the current chat, 2026-11-28 (Europe/Warsaw).

Delivery authorization: subsequent explicit operator instruction in this chat,
2026-11-28: commit this setup and push the development branch. This authorization
applies to this delivery only; permanent role and harness rules are unchanged.

## Write set

```text
.codex/**
.agents/plugins/marketplace.json (new Codex discovery catalog only)
AGENTS.override.md (generated Codex-only projection; root AGENTS.md unchanged)
scripts/governance/sync_agent_env.py
tests/unit/governance/test_codex_agent_env.py
governance/tickets/T-IGX-096-A.md
```

## Test contract

```text
uv run python scripts/governance/sync_agent_env.py --check
uv run pytest tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_codex_agent_env.py -q
uv run ruff check scripts/governance/sync_agent_env.py .codex/hooks .codex/setup.py tests/unit/governance/test_codex_agent_env.py
git diff --check -- scripts/governance/sync_agent_env.py .codex AGENTS.override.md .agents/plugins/marketplace.json tests/unit/governance/test_codex_agent_env.py governance/tickets/T-IGX-096-A.md
Codex strict config parsing, native skill/profile discovery without a model run.
Compare before/after SHA256 of existing harness adapters and canonical authority.
```

## Out of scope

Existing uncommitted qualification work; PR creation/merge; global Codex
defaults or existing plugin settings; trust bypass; production/cloud environment migration; G6/IIAC.

## Checkpoint

- 2026-11-28: Implemented 31 Codex workflow adapters, five native agent profiles,
  five lifecycle registrations, Codex guidance projection, portable project
  configuration, local plugin packaging, repeatable installer, and setup guide.
- Native `codex-cli 0.171.0-alpha.13.2` accepted configuration through a strict
  app-server probe. Effective project configuration loaded; discovery with its
  startup overrides found all 31 plugin skills and disabled Antigravity adapters.
  No model turn or delegated work was started. Ephemeral probe thread only.
- Verification: 45/45 focused tests; Ruff clean; SSOT sync check clean (142
  adapters); scoped `git diff --check` clean. Existing 200 harness/SSOT/control-plane
  files remained SHA256-identical. Main checkout remains attached to development.
- Local plugin installed/refreshed; installer restored user config byte for byte.
- Existing unrelated qualification changes make workspace-wide whitespace checks
  fail; they were left untouched. The scoped setup check passed.
- Manual client activation remains: reopen Codex and review/trust the five hooks
  in Settings > Hooks or /hooks. Native discovery reports them as untrusted;
  no trust bypass was used. This is not G6/IIAC certification.
- Proofs and native compatibility probe: `.tmp/codex-setup/`. Setup details:
  `.codex/README.md`. No commit, push, PR, branch integration, or global default change.

## Lifecycle state

in_progress

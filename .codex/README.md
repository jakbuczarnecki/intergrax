# Intergrax in Codex

Repository-local Codex setup, based on the existing Claude Code configuration
and the canonical `agent-env/` sources. Existing `AGENTS.md`, `CLAUDE.md`,
`GEMINI.md`, `.claude/`, `.cursor/`, Antigravity adapters, and control-plane
authority are unchanged. There is no `cloud.md` in this checkout; `CLAUDE.md`
is the Claude Code entry point used as the reference.

## Start or refresh

From the repository root on Windows:

```powershell
uv run python .codex/setup.py
```

This regenerates adapters and installs/refreshes only `intergrax-codex` in the
local Codex plugin cache. Installation uses a command-scoped local marketplace
source; it does not persist a machine-specific marketplace path. The installer
restores the original user config after verifying that the CLI changed only
this plugin's enablement. No secret-bearing config backup is written. Project
enablement lives in `.codex/config.toml`; other repositories keep their defaults.

Reopen the project in Codex after installation. The project must be trusted for
project configuration to load. Review the five hook definitions using **Settings
> Hooks** / **Review hooks**, or `/hooks` in the CLI, and trust them there.
Hook definitions were detected by the native client but remain untrusted until
that review. The installer never bypasses hook trust. A changed hook definition
requires review again. This is a client security step, not an approval for each
ordinary task.

After editing canonical workflows, run setup again to refresh the installed
plugin cache as well as generated source files. Running the generator alone
does not refresh an already installed copy. Config and agent definitions may
require a new chat or app restart; an existing chat retains its loaded context.

## Layout and ownership

| Path | Purpose / source |
| --- | --- |
| `AGENTS.override.md` | Generated Codex entry point: existing root stub + always-on rule extracts + Codex policy |
| `.codex/AGENTS.md` | Authored Codex tool, role, lifecycle, and authority mapping |
| `.codex/config.toml` | Portable project defaults; typed agents, hook support, plugin enabled, Antigravity skill collisions disabled |
| `.codex/agents/*.toml` | Five generated native profiles; source `agent-env/agents/igx-*.md` and existing tester role map |
| `.codex/plugin/skills/*` | 31 generated workflow adapters, each with `agents/openai.yaml` invocation policy |
| `.codex/plugin/plugin.json` | Portable local plugin manifest |
| `.codex/plugin/.codex-plugin/plugin.json` | Compatibility manifest explicitly declaring the skill directory |
| `.agents/plugins/marketplace.json` | New Codex repository discovery catalog; existing Antigravity files unchanged |
| `.codex/hooks.json` / `.codex/hooks/` | Codex lifecycle registration and implementation |
| `.tmp/session/codex/<session-hash>/plan.json` | Ignored, session-owned plan state; never another harness's state |
| `.codex/setup.py` | Repeatable generation and local installation |

Do not edit generated files. Edit canonical workflows in `agent-env/` and
regenerate; edit Codex-specific mapping in `.codex/AGENTS.md`. Root override
generation includes the entire current `AGENTS.md`, because Codex chooses the
override instead of that file. Other clients continue loading their existing
entry points. No model/provider, sandbox, approvals, credentials, MCP server,
production configuration, global skill installation, or cloud execution is set.

## Roles

| Codex profile | Existing DevOS role | Limit | Authority |
| --- | --- | --- | --- |
| Parent chat | orchestrator | 1 | Coordinates; cannot commit |
| `igx-researcher` | architect-planner | 3 | Read-only; no shell |
| `igx-architect` | architect-planner | 1 | Named plans/tickets only |
| `igx-implementer` | implementer | 2 | Approved isolated worktree; own branch commits with `Ref:` |
| `ci-investigator` | tester | 1 | Serial Test contract; no product edits or commits |
| `igx-reviewer` | reviewer | 1 | Read-only; no shell; cannot certify |

The Codex total is three children, additionally constrained by each existing
role limit and client capacity. Test/`uv` processes remain serial across all
workers. Worker models inherit the parent; Claude/Composer model names are not
copied. Delegation needs an operator request or applicable workflow. Workers do
not create workers. File isolation is not implied by spawning a child: the
parent must arrange an approved worktree first. Native custom profiles inherit
permissions; their write restrictions are behavioral, not separate OS sandboxes.
No-shell roles need a file-read tool or parent-provided excerpts/diffs.

If a client exposes only generic collaboration tools, the parent supplies the
entire typed brief explicitly. A hook may reject generic spawns; do not bypass
it. Report unavailable profile/tool support instead. The current setup session
does not spawn workers merely to test delegation.

SFX's six canonical personas remain declarative briefs. Their eight skills
retain explicit-only invocation. Independent G6/IIAC audit requires a separate
operator chat. Cursor's integrator and direct-execution exceptions are not
transferred to Codex. No role registry, routing registry, or trust pin is changed.

## Skills and plans

Native names include the plugin namespace:

```text
$intergrax-codex:codex-igx-devos-ticket-driven-work
$intergrax-codex:codex-intergrax-hep-step
$intergrax-codex:codex-igx-run-plan agent-env/plans/<approved-plan>.md
$intergrax-codex:codex-igx-run-plan status
$intergrax-codex:codex-igx-run-plan stop
```

Every active shared skill and every `auto`/`explicit` rule has an adapter.
`always` rules are inlined in the root override. Retired and `cursor-always`
rules have no Codex skills. Existing explicit-only flags become the supported
`policy.allow_implicit_invocation: false`, including explicit rule modes.
Canonical workflow bodies remain in the repository and load only on demand.
Slash commands in canonical text map to the names above, and all canonical
paths resolve from the Git worktree root, not the installed plugin cache.

The plan hook recognizes the explicit command in `UserPromptSubmit`, binds the
approved checklist to that session, and continues open work on `Stop`. It
checks actual checkboxes before accepting completion, allows `ESCALATION:`,
isolates simultaneous chats, and stops
automatic continuation after three stalls. Interrupt pauses continuation;
invoke the plan command again to resume. `stop` clears only this session's state.
Without a trusted/running hook, guidance still applies but continuation is not
mechanically enforced. The shared Claude/Cursor stop guard is never invoked.

Tool hooks deny common forbidden Git mutations, governance bypasses, secret
file access through supported paths, large unbounded dumps, and generic/audit
child profiles. They are guardrails, not a complete shell parser or security
sandbox. `IGX_HOOKS=off` is the existing shared diagnostic kill switch, not normal
operation. Codex cloud orchestration does not establish local command-hook
support; this setup preserves the project's Windows execution environment.

## Verification and references

```powershell
uv run python scripts/governance/sync_agent_env.py --check
uv run pytest tests/unit/governance/test_agent_env_sync.py tests/unit/governance/test_codex_agent_env.py -q
uv run ruff check scripts/governance/sync_agent_env.py .codex/hooks .codex/setup.py tests/unit/governance/test_codex_agent_env.py
git diff --check
```

Native compatibility was checked against `codex-cli 0.171.0-alpha.13.2` using
effective configuration, skill discovery, and hook discovery without a model
turn. The app-server's base `skills/list` uses startup configuration; verification
must supply the project's enablement to that probe as session overrides.
`config/read` with `cwd` separately verifies the project layer. Name selectors
work; do not replace them with machine-specific paths based on an unconfigured
base-process probe.

Official mechanisms: [project configuration](https://learn.chatgpt.com/docs/config-file/config-basic),
[AGENTS discovery](https://learn.chatgpt.com/docs/agent-configuration/agents-md),
[custom agents](https://learn.chatgpt.com/docs/agent-configuration/subagents),
[skills and invocation policy](https://learn.chatgpt.com/docs/build-skills),
[hooks and trust](https://learn.chatgpt.com/docs/hooks), and
[local plugin packaging](https://developers.openai.com/plugins/build/plugins).

# Agent token economy

How to keep the first turn of an agent session small and how to measure it. Applies to Claude Code; Cursor uses the same `agent-env/` sources. Campaign: AGENT-OPS-X, wave W1 (ticket `T-IGX-053-B`).

## What loads on turn one

| Layer | Source | Control |
|-------|--------|---------|
| System prompt + built-in tools | Claude Code | fixed, not controllable from the repository |
| L0 | `CLAUDE.md`, `AGENTS.md`, `agent-env/rules/intergrax-core.md` | gated at 8 KB by `sync_agent_env --check` |
| Skill listing | project and account skills | `skillOverrides` in `.claude/settings.json` |
| Agent listing | `.claude/agents/*.md` | four typed `igx-*` agents; `general-purpose` is denied |
| Connector tool names | claude.ai connectors, `.mcp.json` | deferred (names only); see checklist |
| Memory index | `~/.claude/projects/<project>/memory/MEMORY.md` | Operator profile; keep it short |

## Settings that matter (`.claude/settings.json`)

- `skillOverrides`: `user-invocable-only` hides a skill from the model (all `sfx-*`); `name-only` lists the name without its description (on-demand rules, roles not used by packets, account and bundled skills). The skills stay invocable by name or `/<name>`.
- `effortLevel: medium`; `CLAUDE_CODE_AUTO_COMPACT_WINDOW=200000` (plain integer, minimum 100000); `bashOutputMaxChars: 12000` (overrides `BASH_MAX_OUTPUT_LENGTH`).
- Hooks `scripts/hooks/igx_read_guard.py` and `igx_bash_guard.py` deny un-ranged reads and dumps of files over 80 KB or 1500 lines, repeated identical reads in one context epoch, and `sed -n` ranges over 600 lines. A denial states a heading outline; an identical retry right after a denial is allowed. `IGX_HOOKS=off` disables only these two hooks; `IGX_ROLE=auditor` exempts the repeat rule.
- `disabledMcpjsonServers` only covers servers of a project `.mcp.json` (the repository has none). It does not affect claude.ai connectors.

## Per-session connector checklist

1. Run `/context` first. Note Skills, Agents, MCP tools and Memory files.
2. Account connectors (Atlassian, Microsoft 365, Docs, browser and similar) cost about 3.9k tokens per session in tool names alone. Disable them for agent and campaign sessions: add `"disableClaudeAiConnectors": true` to `.claude/settings.local.json` (untracked, per machine) or use `/mcp`. It is not set in the tracked project settings because interactive Operator sessions may need Jira or Docs.
3. Keep MCP tools deferred; fetch a schema with ToolSearch only when a call needs it.
4. If a `.mcp.json` is ever added, list every server the packets do not use in `disabledMcpjsonServers`.
5. Compare with the numbers below; if Skills or MCP exceed a few hundred tokens each, an override was missed.

## Memory policy

- The memory index is part of every first turn. Keep each `MEMORY.md` index to 12 lines and 1.5 KB, pointers only (one line per fact, no status, no paths a command can answer, no secrets).
- Move detail into the linked memory file; never into the index.

## Measuring

Method: headless `claude -p "Reply with the single word: ok" --output-format json --model sonnet` from the project root, no tools. First-turn context = `input_tokens + cache_creation_input_tokens + cache_read_input_tokens` of that single API call. Repeat twice and compare only runs with the same connector state (a connector that has not finished connecting changes the number by about 3.9k). Interactive sessions add tool schemas and reminders that headless runs do not, so compare like with like.

## Measured results (W1, 2026-11-20, headless, model sonnet)

| State | First-turn tokens | Delta |
|-------|-------------------|-------|
| Baseline (`8cee3c5a9`) | 58,944 | - |
| L0 diet (core rule, CLAUDE.md 71 to 10 lines, L0 21.1 KB to 8.0 KB) | 53,040 | -5,904 |
| Project skills overrides | 51,861 | -1,179 |
| Typed agents, deny `general-purpose` | 51,885 | +24 |
| Runtime settings and hooks | 51,872 | -13 |
| Account skills `name-only` | 46,201 | -5,671 |
| Optional: `disableClaudeAiConnectors` (not tracked) | about 42.3k | about -3.9k |

Total tracked: 58.9k to 46.2k (-12.7k, -21.6 %). An empty directory with no project at all measures 45.1k, so the repository now adds about 1k above the Claude Code floor. A model swap is not counted anywhere in this table.

## Old rule to new section (nothing lost)

| Old always-on rule | Now |
|--------------------|-----|
| `git-branch-safety` | `intergrax-core.md` Branch safety |
| `no-ai-attribution` | `intergrax-core.md` Attribution |
| `session-artifacts` | `intergrax-core.md` Session artifacts |
| `test-process-budget` | `intergrax-core.md` Test budget |
| `token-budget` | `intergrax-core.md` Token use (preflight, I1/O1); Full O1 template stays in `AGENT_INSTRUCTIONS.md` |
| `env-vendor-qualification` | on-demand skill; core keeps the C10/C15 freeze note |
| `CLAUDE.md` worktree section | skill `/igx-worktree` |
| `CLAUDE.md` vocabulary table | one line in `CLAUDE.md` (agents, `isolation: worktree`) |

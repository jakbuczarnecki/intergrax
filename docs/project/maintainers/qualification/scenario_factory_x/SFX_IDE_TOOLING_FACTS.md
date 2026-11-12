# SFX IDE tooling facts (SFX-P1-02)

Ticket: T-IGX-037-B (plan: `governance/tickets/plans/PLAN_T-IGX-037-B.md:17-22`). Documentation only.

**Access date for every URL below: 2026-11-12.** Vendor pages are living documents without per-page revision dates; facts reflect the pages as served on that date. Repository citations are `path:line` in the `sfx-w1-research` worktree.

Source key:

| Key | URL |
|-----|-----|
| [CC-SKILLS] | https://code.claude.com/docs/en/skills |
| [CC-AGENTS] | https://code.claude.com/docs/en/sub-agents |
| [CC-MCP] | https://code.claude.com/docs/en/mcp |
| [CUR-RULES] | https://cursor.com/docs/context/rules |
| [CUR-SKILLS] | https://cursor.com/docs/context/skills |
| [CUR-AGENTS] | https://cursor.com/docs/context/subagents |
| [CUR-MCP] | https://cursor.com/docs/context/mcp |
| [CUR-2.4] | https://cursor.com/changelog/2-4 |
| [MCP-T-0326] | https://modelcontextprotocol.io/specification/2025-03-26/basic/transports |
| [MCP-VER] | https://modelcontextprotocol.io/specification/versioning |

---

## 1. Claude Code skills

**Location** [CC-SKILLS]:

| Scope | Path |
|-------|------|
| Enterprise | `.claude/skills/<skill-name>/SKILL.md` in the managed settings directory |
| Personal | `~/.claude/skills/<skill-name>/SKILL.md` |
| Project | `.claude/skills/<skill-name>/SKILL.md` |
| Nested | `<subdir>/.claude/skills/<skill-name>/SKILL.md` (loaded on demand) |
| Plugin | `<plugin>/skills/<skill-name>/SKILL.md`, invoked as `/plugin-name:skill-name` |

Name collisions: enterprise > personal > project; plugin skills are namespaced [CC-SKILLS].

**Frontmatter** [CC-SKILLS]. All fields optional; `description` recommended; unknown fields are silently ignored.

| Field | Meaning |
|-------|---------|
| `name` | `/` command name; defaults to the directory name |
| `description` | What the skill does and when to use it; drives automatic loading |
| `when_to_use` | Extra trigger text appended to `description` (listing truncated at 1,536 chars combined) |
| `argument-hint`, `arguments` | Autocomplete hint; named positional arguments for `$name` substitution |
| `disable-model-invocation` | `true` = only the user can invoke; also blocks preloading into subagents |
| `user-invocable` | `false` = hidden from `/` menu, model-only |
| `allowed-tools` / `disallowed-tools` | Tools usable without prompts during the invoking turn / tools removed while active |
| `model`, `effort` | Per-turn overrides |
| `context: fork` + `agent` (+ `background`) | Run the skill in a forked subagent of the given type (built-in or custom agent) |
| `hooks`, `paths`, `shell`, `metadata`, `license`, `compatibility` | Hooks; glob limit for auto-loading; `bash`/`powershell` for `` !`cmd` `` injection; free-form map; Agent Skills spec fields (accepted, not acted on) |

Body substitutions include `$ARGUMENTS`, `$N`, `${CLAUDE_SKILL_DIR}`, `${CLAUDE_PROJECT_DIR}` [CC-SKILLS].

**Invocation** [CC-SKILLS]: manual `/skill-name` (at message start runs it); automatic when the request matches `description`/`when_to_use` unless `disable-model-invocation: true`. `.claude/commands/<name>.md` still works and is merged into skills; a skill wins on a name clash.

**Standard** [CC-SKILLS]: Claude Code follows the open Agent Skills standard (agentskills.io), whose spec defines six fields (`name`, `description`, `license`, `compatibility`, `metadata`, `allowed-tools`); outside Claude Code (claude.ai upload, Skills API) other keys are a hard error.

**Repository fit:** the generator emits `name`, `description` and, only for `loading: explicit` rules, `disable-model-invocation: true` (`scripts/governance/sync_agent_env.py:111-121`, `:155-161`). Skill adapters are always emitted with `explicit_only=False` (`sync_agent_env.py:164-167`), so a canonical skill cannot currently be made operator-only through the generator.

## 2. Claude Code subagents

**Location and scope** [CC-AGENTS]:

| Location | Scope | Priority |
|----------|-------|----------|
| Managed settings | Organization | 1 |
| `--agents` CLI flag (JSON) | Session | 2 |
| `.claude/agents/` | Project | 3 |
| `~/.claude/agents/` | User (all projects) | 4 |
| Plugin `agents/` | Where enabled | 5 |

Project and user directories are scanned recursively; identity comes only from the `name` field, not the filename [CC-AGENTS].

**Format** [CC-AGENTS]: Markdown with YAML frontmatter; the body is the subagent's system prompt (the subagent does not receive the Claude Code system prompt).

**Frontmatter** [CC-AGENTS]. Required: `name` (max 256 chars, no `:`) and `description`. Case-sensitive camelCase; unknown fields ignored.

| Field | Meaning |
|-------|---------|
| `tools` / `disallowedTools` | Allowlist (inherits all if omitted) / denylist |
| `model` | `sonnet`, `opus`, `haiku`, `fable`, full model ID, or `inherit` |
| `permissionMode` | `default`, `acceptEdits`, `auto`, `dontAsk`, `bypassPermissions`, `plan` |
| `skills` | Skills preloaded in full at startup (max 32); skills with `disable-model-invocation: true` cannot be preloaded |
| `mcpServers` | Named or inline MCP servers for the subagent |
| `hooks`, `memory`, `color`, `effort`, `isolation: worktree`, `background`, `maxTurns`, `omitClaudeMd`, `initialPrompt`, `autoCompactWindow`, `experimental` | Lifecycle and runtime controls |

**Invocation** [CC-AGENTS]: automatic delegation by `description`; natural language ("use the X subagent"); `@agent-<name>` mention; session-wide `claude --agent <name>` or `"agent"` in `.claude/settings.json`. Subagents may spawn subagents up to three levels by default (`CLAUDE_CODE_MAX_SUBAGENT_SPAWN_DEPTH`).

**Repository fit:** the generator has no agents input or output; it reads only `agent-env/rules` and `agent-env/skills` (`sync_agent_env.py:31-33`, `:148-168`) and its drift scan covers only `.cursor/rules/*.mdc`, `.cursor/skills/*/SKILL.md`, `.claude/skills/*/SKILL.md` (`sync_agent_env.py:171-178`). Confirmed by the module docstring (`sync_agent_env.py:9-12`).

## 3. Cursor

### 3.1 Rules [CUR-RULES]

- Project rules live in `.cursor/rules/` (subfolders allowed) and **must** use `.mdc`; a plain `.md` file there is ignored. `AGENTS.md` (root or nested) is a metadata-free alternative. User rules and Team rules are configured in the app/dashboard.
- `.mdc` frontmatter: `description`, `globs`, `alwaysApply`.

| `alwaysApply` | `description` | `globs` | Rule type |
|---|---|---|---|
| `true` | - | - | Always Apply |
| `false` | - | set | Apply to Specific Files |
| `false` | set | omitted | Apply Intelligently (agent decides) |
| `false` | omitted | omitted | Apply Manually (`@rule-name` only) |

- No page date shown. `.cursorrules` is not mentioned on the page: deprecation status `UNKNOWN`.

**Repository fit (finding F-01):** the generator writes every rule as `description` + `alwaysApply` with no `globs` (`sync_agent_env.py:100-108`). Per the table above, `loading: explicit` rules therefore become **Apply Intelligently** in Cursor, not "use `@`" as `agent-env/README.md:26` states. Manual-only behaviour would need the `description` omitted from the `.mdc`. Not SFX-blocking (the SFX rule is `loading: auto`, `PLAN_T-IGX-037-E.md:16-18`), but it is a semantic drift between the two tools.

### 3.2 Skills [CUR-SKILLS], [CUR-2.4]

- Introduced in Cursor **2.4, dated Jan 22, 2026** (editor and CLI) [CUR-2.4].
- Directories: project `.agents/skills/`, `.cursor/skills/`; user `~/.agents/skills/`, `~/.cursor/skills/`; compatibility `.claude/skills/`, `.codex/skills/`, `~/.claude/skills/`, `~/.codex/skills/`. Scanned recursively.
- `SKILL.md` frontmatter: `name` (required; lowercase letters, digits, hyphens; **must match the parent folder**), `description` (required), `paths` (legacy `globs` accepted), `disable-model-invocation`, `icon`, `color`, `metadata`. `license`/`compatibility` not listed.
- Invocation: automatic by relevance; `/skill-name` for one message; or as a **Custom Mode** (Alt+Enter on Windows) for the whole session.
- The generator already enforces `name == directory` (`sync_agent_env.py:90-93`), satisfying Cursor's constraint.
- **Finding F-02:** Cursor also scans `.claude/skills/`, so each generated skill is discoverable twice in Cursor (`.cursor/skills/<id>` and `.claude/skills/<id>`, `sync_agent_env.py:164-167`). Cursor's de-duplication behaviour for identical names across these roots: `UNKNOWN`.

### 3.3 Subagents / custom agents [CUR-AGENTS], [CUR-2.4]

- Subagents with user-defined custom subagents introduced in Cursor **2.4 (Jan 22, 2026)** [CUR-2.4]; nested subagents mentioned for 2.5 in the FAQ (2.5 date `UNKNOWN`) [CUR-AGENTS].
- Locations: project `.cursor/agents/`, `.claude/agents/`, `.codex/agents/`; user `~/.cursor/agents/`, `~/.claude/agents/`, `~/.codex/agents/`. Project overrides user; within one scope `.cursor/` wins over `.claude/`/`.codex/`.
- Format: Markdown + YAML frontmatter, body is the prompt. Fields (all optional): `name` (defaults to filename; lowercase + hyphens), `description`, `model` (default `inherit`), `readonly` (default `false`), `is_background` (default `false`). No `tools` field is documented.
- Invocation: automatic delegation by description; explicit `/name` or natural language; parallel via multiple Task calls. Built-ins: Explore, Browser.
- Custom modes: the subagents page does not mention them; skills can be used as a Custom Mode (`icon`, `color` fields) [CUR-SKILLS]. A separate custom-modes file format: `UNKNOWN`.
- How Cursor treats Claude-only frontmatter (`tools`, `permissionMode`, `skills`) in a shared `.claude/agents/*.md`: `UNKNOWN`.

## 4. MCP clients

### 4.1 Claude Code [CC-MCP]

- Transports: **HTTP (streamable HTTP) recommended for remote servers**; `"type": "streamable-http"` is an accepted alias of `"http"`. SSE is deprecated (v2.1.265+ tries HTTP first, then SSE). stdio for local processes; `ws` via JSON only.
- CLI: `claude mcp add --transport http <name> <url> --header "Authorization: Bearer <token>"` (`-t`, `-H` short forms).
- `.mcp.json` (project scope, committed, requires interactive approval):

```json
{
  "mcpServers": {
    "api-server": {
      "type": "http",
      "url": "${API_BASE_URL:-https://api.example.com}/mcp",
      "headers": { "Authorization": "Bearer ${API_KEY}" }
    }
  }
}
```

- `${VAR}` / `${VAR:-default}` expansion in `command`, `args`, `env`, `url`, `headers`. Credential-named variables (examples given: `ANTHROPIC_API_KEY`, `ANTHROPIC_AUTH_TOKEN`, `AWS_BEARER_TOKEN_BEDROCK`, `HTTPS_PROXY`, `NPM_TOKEN`) expand to empty in remote `url`/`headers`; other names expand normally. Whether the list is exhaustive: `UNKNOWN`.
- Scopes: `local` (default, `~/.claude.json` per project), `project` (`.mcp.json`), `user` (`~/.claude.json`).
- Other auth: OAuth 2.0 (`/mcp`, `claude mcp login`, `oauth` object); `headersHelper` command that prints a JSON header map per connection (10 s limit; re-run after 401/403; an `Authorization` from it replaces OAuth).

### 4.2 Cursor [CUR-MCP]

- Transports: `stdio`, `SSE`, `Streamable HTTP`.
- Files: project `.cursor/mcp.json`, global `~/.cursor/mcp.json`.
- Remote entry under `mcpServers`: `url` + `headers` (the documented remote example carries no `type`).

```json
"url": "http://localhost:3000/mcp",
"headers": { "Authorization": "Bearer ${env:MY_SERVICE_TOKEN}" }
```

- Interpolation in `command`, `args`, `env`, `url`, `headers`: `${env:NAME}`, `${userHome}`, `${workspaceFolder}`, `${workspaceFolderBasename}`, `${pathSeparator}`/`${/}`.
- OAuth: authorization code + PKCE, RFC 9728 protected resource metadata, RFC 8707 resource indicators, optional static `auth` object.
- MCP protocol revision supported by Cursor: not stated - `UNKNOWN`.

### 4.3 Protocol revision

- Streamable HTTP is defined by revision **2025-03-26**, where it "replaces the HTTP+SSE transport from protocol version 2024-11-05" [MCP-T-0326]. It requires a single MCP endpoint supporting POST and GET, optional `Mcp-Session-Id`, and documents client fallback to the old HTTP+SSE transport.
- The **current** revision is **2026-07-28** [MCP-VER]: per-request version in `_meta` (`io.modelcontextprotocol/protocolVersion`) and the `MCP-Protocol-Version` header on Streamable HTTP, an optional `server/discover` RPC, and a backward-compatibility path for handshake-based revisions (`2025-11-25` and earlier).
- Claude Code supports 2026-07-28 via its v2 client runtime and falls back to the earlier handshake when the server does not support it; `MCP_PROTOCOL_NEGOTIATION=auto|legacy` [CC-MCP].
- Intergrax pins `fastmcp>=3.3.1,<4` (`pyproject.toml:227`; `docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_CONTRACT_AND_PROTOCOL_MATRIX.md:12`). The protocol revision(s) FastMCP 3.3.1 negotiates are not recorded in the repository: `UNKNOWN`.
- Conclusion: the version is a factor only for interoperability proof, not for configuration. Claude Code has a documented fallback; Cursor's behaviour is `UNKNOWN` until a live check.

## 5. Implications for Intergrax

### 5.1 Intergrax MCP server surface (repo)

- Streamable HTTP only; stdio and SSE unsupported by design (`docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_P6_DEVELOPER_GUIDE.md:54-55`, `:83-84`; OD-MCP-03).
- Endpoint: FastMCP `http_app(path="/")` mounted at `/mcp` (default) on a wrapper app (`intergrax/applications/_shared/fastapi_mcp.py:41-45`, `:61-65`, `:86`).
- Auth: harness middleware is applied to the wrapper (covers the mounted MCP app) when the host profile requires auth (`fastapi_mcp.py:88-91`; `intergrax/applications/_shared/harness_auth.py:355-365`). It accepts `X-Api-Key: <key>` or `Authorization: Bearer <key>`; a Bearer value takes precedence over `X-Api-Key` (`harness_auth.py:60-68`, `:326-343`). It also accepts identity-provider bearer tokens (`harness_auth.py:338-343`). Failure returns `401 {"detail": "Invalid or missing harness credentials"}` (`harness_auth.py:349-352`). Legacy default key env var: `INTERGRAX_HARNESS_API_KEY` (`harness_auth.py:40-43`).

### 5.2 Does the sync generator need extending for agent definitions (SFX-P4-B01)?

**Yes, if SFX wants native subagents; no, for v0.** Native discovery requires files in `.claude/agents/` (Claude Code) or `.cursor/agents/` / `.claude/agents/` (Cursor) [CC-AGENTS], [CUR-AGENTS]; the generator emits neither (`sync_agent_env.py:9-12`, `:148-178`). Native definitions are what enable automatic delegation by `description`, `@agent-<name>` / `/name`, `claude --agent`, a skill's `context: fork` + `agent: <custom>`, and per-persona `tools`/`model`/`readonly`. SFX-P4-B01 scope for W3:

1. Add an `agent-env/agents/<id>.md` loader with required `name` (no `:`, lowercase-hyphen to satisfy both tools) and `description` (Claude requires both [CC-AGENTS]).
2. Emit `.claude/agents/<id>.md`; decide whether also to emit `.cursor/agents/<id>.md`. Cursor reads `.claude/agents/` already, but `.cursor/` wins in-scope and field sets differ (`tools` vs `readonly`) [CUR-AGENTS]; Cursor handling of Claude-only fields is `UNKNOWN`.
3. Decide the body strategy: a subagent body *is* its system prompt [CC-AGENTS], so a thin pointer body ("read agent-env/agents/<id>.md") costs one Read turn; inlining the canonical body is the alternative.
4. Extend `_generated_on_disk` and `--check` to the agents directories (`sync_agent_env.py:171-178`).
5. Optionally carry a skill-level `disable-model-invocation` through to skill adapters (`sync_agent_env.py:166` hard-codes `False`), noting such skills then cannot be preloaded via a subagent's `skills` field [CC-AGENTS].

### 5.3 Can skills reference `agent-env/agents/*.md` without sync (v0)?

**Yes.** The generated adapter body only tells the agent to read the canonical file (`sync_agent_env.py:115-120`); the canonical skill can in turn instruct it to read `agent-env/agents/sfx-<persona>.md` and pass that text as the prompt of a generic subagent (Claude Code `general-purpose`, Cursor Task). This is plain file reading and needs no vendor feature; it matches the v0 plan (`governance/tickets/plans/PLAN_T-IGX-037-E.md:23-25`). Limits in v0: `agent-env/agents/` is outside every documented discovery path [CC-AGENTS], [CUR-AGENTS], so personas get no automatic delegation, no `@`/`/` mention, no `context: fork` with a custom `agent`, and no per-persona tool/model restriction (they inherit the generic subagent's tools).

### 5.4 Does harness API-key auth fit the clients' header configuration?

**Yes, at configuration level.** Both clients send arbitrary static headers on remote servers and expand environment variables in `headers` [CC-MCP], [CUR-MCP]; the harness accepts either `X-Api-Key` or `Authorization: Bearer` (`harness_auth.py:60-68`). Examples (key never committed, per the env/vendor rule):

- Claude Code `.mcp.json`: `"type": "http", "url": "http://127.0.0.1:<port>/mcp", "headers": {"X-Api-Key": "${INTERGRAX_HARNESS_API_KEY}"}`; or `claude mcp add --transport http intergrax <url> --header "X-Api-Key: ..."` (local scope, stays out of the repo).
- Cursor `.cursor/mcp.json`: `"url": "...", "headers": {"X-Api-Key": "${env:INTERGRAX_HARNESS_API_KEY}"}`.

Residual risks: (a) the 401 response carries no `WWW-Authenticate`/RFC 9728 metadata (`harness_auth.py:349-352`); whether either client then attempts OAuth discovery or simply reports failure is `UNKNOWN`. (b) Whether `INTERGRAX_HARNESS_API_KEY` falls under Claude Code's credential-variable blank-out is `UNKNOWN` (not among the named examples [CC-MCP]); `headersHelper` or local-scope `--header` avoid the question. (c) No live client-to-`/mcp` interop proof exists in this research.

### 5.5 Gap SFX-GAP-07

The gap text lives in the SFX master plan, which is not present in this worktree; scope is taken from `PLAN_T-IGX-037-B.md:17-22` (skill/agent formats, Cursor formats, streamable HTTP client support, decision input for SFX-P4-B01).

**Verdict: RESOLVED** (documentation-level facts).

- Claude Code skill and subagent formats: VERIFIED [CC-SKILLS], [CC-AGENTS].
- Cursor rules, skills, subagents formats and dates (2.4, Jan 22, 2026): VERIFIED [CUR-RULES], [CUR-SKILLS], [CUR-AGENTS], [CUR-2.4].
- Streamable HTTP client support with header auth in both tools: VERIFIED [CC-MCP], [CUR-MCP]; matches Intergrax's only transport (`MCP_SERVER_X_P6_DEVELOPER_GUIDE.md:54-55`).
- ME for SFX-P4-B01 answered in 5.2 (generator extension required for native agents; not required for v0).

Carried to SFX-P10 (live-mode qualification), not blocking SFX-P4: R-01 Cursor MCP protocol revision; R-02 FastMCP 3.3.1 negotiated revision versus the current 2026-07-28; R-03 client behaviour on harness 401; R-04 live interop proof from each client to `/mcp`. Findings F-01 (explicit rules become Apply Intelligently in Cursor) and F-02 (duplicate skill discovery in Cursor) are recorded for SFX-P4-B01.

---

## Summary table

| Fact | Value | Source | Status |
|------|-------|--------|--------|
| CC skill project path | `.claude/skills/<name>/SKILL.md` | [CC-SKILLS] | VERIFIED |
| CC skill invocation control | `disable-model-invocation`, `user-invocable` | [CC-SKILLS] | VERIFIED |
| CC skill fork to subagent | `context: fork` + `agent` | [CC-SKILLS] | VERIFIED |
| CC skill required fields | none (description recommended) | [CC-SKILLS] | VERIFIED |
| CC subagent path / scope | `.claude/agents/` project, `~/.claude/agents/` user | [CC-AGENTS] | VERIFIED |
| CC subagent required fields | `name`, `description` | [CC-AGENTS] | VERIFIED |
| CC subagent key fields | `tools`, `disallowedTools`, `model`, `permissionMode`, `skills`, `mcpServers` | [CC-AGENTS] | VERIFIED |
| CC subagent invocation | auto by description, `@agent-<name>`, `claude --agent` | [CC-AGENTS] | VERIFIED |
| Cursor rule format | `.cursor/rules/*.mdc`; `description`, `globs`, `alwaysApply` | [CUR-RULES] | VERIFIED |
| Cursor rule with description, alwaysApply false, no globs | Apply Intelligently | [CUR-RULES] | VERIFIED |
| Cursor `.cursorrules` status | not stated | [CUR-RULES] | UNKNOWN |
| Cursor skills + subagents release | 2.4, Jan 22, 2026 | [CUR-2.4] | VERIFIED |
| Cursor skill dirs | `.cursor/skills`, `.agents/skills`, compat `.claude/skills`, `.codex/skills` | [CUR-SKILLS] | VERIFIED |
| Cursor skill `name` rule | lowercase-hyphen, equals folder | [CUR-SKILLS] | VERIFIED |
| Cursor duplicate-skill handling | not stated | [CUR-SKILLS] | UNKNOWN |
| Cursor subagent dirs | `.cursor/agents`, `.claude/agents`, `.codex/agents` (+ user) | [CUR-AGENTS] | VERIFIED |
| Cursor subagent fields | `name`, `description`, `model`, `readonly`, `is_background` | [CUR-AGENTS] | VERIFIED |
| Cursor handling of Claude-only agent fields | not stated | [CUR-AGENTS] | UNKNOWN |
| Cursor custom-modes file format | not documented on cited pages | [CUR-SKILLS], [CUR-AGENTS] | UNKNOWN |
| CC streamable HTTP | supported, recommended (`type: http`) | [CC-MCP] | VERIFIED |
| CC header auth | `--header`, `.mcp.json` `headers`, `${VAR}` expansion, `headersHelper` | [CC-MCP] | VERIFIED |
| Cursor streamable HTTP | supported (`url` + `headers`) | [CUR-MCP] | VERIFIED |
| Cursor header auth | `headers` with `${env:NAME}` | [CUR-MCP] | VERIFIED |
| Spec revision defining streamable HTTP | 2025-03-26 | [MCP-T-0326] | VERIFIED |
| Current MCP revision | 2026-07-28 | [MCP-VER] | VERIFIED |
| CC protocol negotiation fallback | yes (`MCP_PROTOCOL_NEGOTIATION`) | [CC-MCP] | VERIFIED |
| Cursor supported MCP revision | not stated | [CUR-MCP] | UNKNOWN |
| FastMCP 3.3.1 negotiated revision | not recorded in repo | `pyproject.toml:227` | UNKNOWN |
| Intergrax MCP transport | streamable HTTP only, `/mcp` | `MCP_SERVER_X_P6_DEVELOPER_GUIDE.md:54-55`; `fastapi_mcp.py:61-65` | VERIFIED |
| Intergrax harness auth headers | `X-Api-Key` or `Authorization: Bearer` | `harness_auth.py:60-68`, `:109-115` | VERIFIED |
| Generator emits agent definitions | no | `sync_agent_env.py:9-12`, `:148-178` | VERIFIED |
| Generator skill-level operator-only flag | no (`explicit_only=False`) | `sync_agent_env.py:164-167` | VERIFIED |
| Client behaviour on harness 401 | not documented | - | UNKNOWN |
| SFX-GAP-07 | RESOLVED (doc level; R-01..R-04 to SFX-P10) | this file §5.5 | VERIFIED |

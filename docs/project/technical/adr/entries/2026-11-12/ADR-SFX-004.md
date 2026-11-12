# ADR-SFX-004: IDE toolkit, CLI and MCP surface; parity contract

| Field | Value |
|-------|-------|
| **Status** | **Proposed** (T-IGX-037-C, SFX-P2; **not Accepted** - acceptance is OD-SFX-06) |
| **Date** | 2026-11-12 |
| **Deciders** | Operator (OD-SFX-03 IDE model; OD-SFX-04 pre-freeze scope; OD-SFX-06 acceptance) |
| **Reference SHA** | `eadb1c2116fe` (`origin/development`); every repository `path:line` below is at this SHA unless marked as a facts-file line |
| **Related** | SFX master plan R1 §7.7, §7.10, §10 W2/W3/W5, §13 (`governance/tickets/plans/SFX-MASTER-PLAN.md`; session draft `.tmp/session/sfx/SFX_MASTER_PLAN_R1.md`) · [`SFX_IDE_TOOLING_FACTS.md`](../../../../capabilities/plan/satellites/scenario_factory/SFX_IDE_TOOLING_FACTS.md) (SFX-P1-02, T-IGX-037-B; cited as `FACTS §n` / `FACTS:line`) · [`ADR-SFX-003`](ADR-SFX-003.md) (Studio stack and auth) · [`ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md`](../../../../maintainers/architecture/ADR_MCP_03_TRANSPORT_STREAMABLE_HTTP.md) · [`T-IGX-037-C`](../../../../../../governance/tickets/T-IGX-037-C.md) · [`PLAN_T-IGX-037-E`](../../../../../../governance/tickets/plans/PLAN_T-IGX-037-E.md) |

## Context

OD-SFX-03 (operator, 2026-11-12) sets the IDE model: **one factory core**. agent-env skills, rules and agents use the CLI offline, with an optional MCP live mode over streamable HTTP. SFX therefore has three doors onto the same operations: the Scenario Studio API (ADR-SFX-003), a CLI and an MCP server. Without a normative parity contract they would drift in schemas, validation, permissions and identity.

Authority levels (§7.10): L0 read-only · L1 proposal-only · L2 sandbox-executable · L3 production. **SFX never grants L3.**

### Verified facts at `eadb1c2116fe`

**MCP authoring surface.**
- `intergrax/applications/_shared/mcp_developer_api.py` is the supported import surface for Tier-3 MCP authors (`:4-8`, `:50-75`).
- It exports `build_mcp_server_definition`, `McpExposurePolicy`, `resolve_effective_bindings`, `compose_mcp_fastmcp_for_definition`, `couple_mcp_fastmcp_on_harness_host` and `compute_mcp_exposure_schema_fingerprint` (`:17-22`, `:32-48`).

**Exposure policy.**
- `McpExposurePolicy` is frozen, forbids extra fields, and holds an explicit `allowed_binding_ids` tuple (`intergrax/contracts/mcp_server_definition.py:91-97`).
- `resolve_effective_bindings` exposes only allowlisted, non-retired, non-hidden bindings. An empty allowlist exposes nothing (`mcp_server_definition.py:212-225`, empty-set check at `:216-218`).

**SFX-GAP-17, allowlist widening.**
- `compose_application_mcp_server` starts from the legacy Nexus definition and adds the catalog surface whenever a `tool_registry` is passed (`intergrax/applications/_shared/mcp_application_server.py:66-78`).
- It then **unions** every `extra_bindings` id into the allowlist through `exposure_policy.model_copy(update={"allowed_binding_ids": allowed})` (`mcp_application_server.py:80-92`, union at `:83`, `model_copy` at `:91`).
- So the effective surface of a host built this way is "legacy + catalog + extras". It is not a stated SFX operation set.

**Transport.**
- Streamable HTTP is the only supported transport; stdio and SSE are unsupported by design under OD-MCP-03 (`docs/project/maintainers/qualification/mcp_server_x/MCP_SERVER_X_P6_DEVELOPER_GUIDE.md:52-55`).
- The endpoint is FastMCP `http_app(path="/")` mounted on the host wrapper (`intergrax/applications/_shared/fastapi_mcp.py:58-65`).

**MCP auth.**
- The harness middleware accepts `X-Api-Key` or `Authorization: Bearer`, with Bearer taking precedence (`intergrax/applications/_shared/harness_auth.py:60-68`, `:326-337`).
- It also accepts identity-provider bearer tokens (`harness_auth.py:338-343`).
- Failure returns 401 `{"detail": "Invalid or missing harness credentials"}` (`harness_auth.py:349-352`).

**agent-env generator.**
- The generator reads only rules and skills, emits no agent definitions, and its drift scan covers only `.cursor/rules/*.mdc`, `.cursor/skills/*/SKILL.md` and `.claude/skills/*/SKILL.md` (`scripts/governance/sync_agent_env.py:162-178`; FACTS §2, `FACTS:90`).

## Alternatives

| Variant | Summary | Disposition |
|---------|---------|-------------|
| **A - One Tier-2 service layer, three thin adapters (selected)** | `agents/scenario_factory/service/` owns every operation; Studio API, CLI and MCP only parse transport and map results | **Selected** (OD-SFX-03 "one factory core") |
| **B - MCP as the only core, CLI and Studio as MCP clients** | Route all doors through the MCP server | **Rejected.** The offline CLI would need a running host, which contradicts OD-SFX-03 ("CLI offline"). It couples Studio to an MCP transport that is optional live mode. |
| **C - Per-door implementations sharing only schemas** | Each door validates and persists on its own | **Rejected.** It violates PC-2 and PC-4 by construction and multiplies the authorization paths (PC-3). |
| **D - Reuse `compose_application_mcp_server` with `extra_bindings` as is** | Lowest-effort MCP surface | **Rejected** as the SFX composition path (SFX-GAP-17). Its allowlist is widened by union with the legacy and catalog surfaces, so the SFX MCP surface would not equal the SFX operation set. |

## Decision

### 1. One service layer, three adapters

- **Core.** All SFX operations are functions of one Tier-2 service layer, `agents/scenario_factory/service/` (SFX-P5-07). It is the only place that validates, authorizes (through the existing principal → tenant → PolicyEngine path), persists and computes identity.
- **Adapters (Tier-3, `applications/scenario_studio_application/`).** Three adapters sit on the core:
  - Studio API v1 (`serving/api_v1.py`, SFX-P10-01);
  - CLI (`cli/__main__.py`, v0 in SFX-P5-07);
  - MCP (`mcp/server.py`, SFX-P10-02).
- **What adapters do.** Adapters parse transport input, call the service, and render the result. They hold no domain logic.
- **Tier rules.** `agents/` never imports `applications/`, and `intergrax/` is unchanged by this ADR.

### 2. Parity contract (normative)

The doors MUST satisfy PC-1..PC-6. Each rule has a mandatory test in the parity suite `tests/qualification/scenario_factory_x/parity/` (SFX-P10-03).

| ID | Normative rule | Mandatory test |
|----|----------------|----------------|
| **PC-1 Schemas** | There MUST be exactly one schema set, generated from the Tier-2 pydantic contracts into `agents/scenario_factory/schemas/*.schema.json`. The Studio client types and the MCP tool input/output schemas MUST be generated from that set, never hand-written. | Schema-drift test: regenerated output is byte-equal to the committed schemas. The MCP exposure fingerprint (`compute_mcp_exposure_schema_fingerprint`) is pinned against the same set. |
| **PC-2 Validation** | Every door MUST call the same service function for the same operation. A door MUST NOT add validation beyond transport parsing. | Error-code parity: an identical invalid input yields an identical `sfx.error` code and JSON pointer in all three doors. |
| **PC-3 Permissions** | There MUST be one authorization path: harness principal → tenant reconciliation → PolicyEngine. The offline CLI is limited to **L0/L1 on local files**. Shared-store writes and L2 runs MUST go only through an authenticated door (API or MCP) that uses the same principal model. No door may grant L3. | Permission matrix per authority level × door; the CLI is denied shared writes and L2. |
| **PC-4 Semantics** | The same operation MUST produce the same `artifact_id`, version, `canonical_digest`, lifecycle transition and side effects through any door. | Op-equivalence test over the service operation catalog. |
| **PC-5 Canonicalization** | The canonical form MUST be RFC 8785 JSON Canonicalization of the model dump, and `canonical_digest = sha256(canonical form)`. YAML and pretty JSON are presentations and MUST NOT be hashed. The determinism scope is validate, resolve, compose, canonicalize and scaffold. LLM stages are compared by replaying recorded turns. | Byte-identity of the canonical form across doors; replay equivalence. |
| **PC-6 Versioning** | API v1, MCP tool set v1 and CLI v1 MUST share one version constant and a 1:1 operation map. Adding an operation to one door without the others is a test failure. | Operation-map completeness test. |

### 3. Offline CLI scope

- The CLI runs **without a host and without credentials**. It is limited to L0/L1 on local files, for example validate, show, canonicalize and draft generation under the caller's working tree or `.tmp/sfx/<slug>/`.
- It MUST NOT write to shared stores (Content Workspace, Session or Lifecycle Record), start L2 runs, or call a runtime.
- An operation that needs L2 or shared state fails in the CLI with a typed `sfx.error` that points the user at the API or MCP door. It never degrades silently.

### 4. MCP surface

- **Transport.** Streamable HTTP only (OD-MCP-03). stdio and SSE are not offered. The server is mounted on the Studio host through the existing coupling path (`couple_mcp_fastmcp_on_harness_host` / `couple_harness_host_fastapi_with_mcp`) and behind the harness auth middleware.
- **Explicit allowlist (SFX-GAP-17).**
  - The SFX MCP definition is built with `build_mcp_server_definition`, with bindings for the SFX operation set only and an explicit `McpExposurePolicy.allowed_binding_ids` equal to that set.
  - SFX MUST NOT obtain its definition through `compose_application_mcp_server(..., extra_bindings=...)`, nor pass a `tool_registry` that adds the catalog surface.
  - Any future recomposition replaces the allowlist explicitly; it never unions into it.
- **No-widening test (SFX-P10-02).** The test asserts that the ids from `resolve_effective_bindings(definition)` **equal** the PC-6 MCP operation set. Then:
  1. Adding a binding outside the set without updating the operation map fails the test.
  2. Composing on a host that also has legacy or catalog bindings leaves the SFX effective set unchanged.
  3. An empty allowlist exposes nothing (fail closed, `mcp_server_definition.py:216-218`).
- **Principal.** MCP requests are authenticated by the harness middleware (API key or identity-provider bearer). They then pass through the same PC-3 path as the Studio API.

### 5. Client configuration

These facts come from FACTS §4 and §5.4. They are documentation-level and not live-proven.

**Claude Code** (FACTS §4.1, `FACTS:130-150`):
- Remote transport `"type": "http"`; `streamable-http` is an alias.
- Project scope is `.mcp.json`, committed and approved interactively. Local scope is `claude mcp add --transport http <name> <url> --header "..."`.
- `${VAR}` and `${VAR:-default}` expand in `url` and `headers`.
- Some credential-named variables expand to empty in remote headers. Whether `INTERGRAX_HARNESS_API_KEY` is affected is `UNKNOWN` (`FACTS:148`, `:204`).
- `headersHelper` is available.

**Cursor** (FACTS §4.2, `FACTS:152-165`):
- `.cursor/mcp.json` or `~/.cursor/mcp.json`, with `url` plus `headers`.
- Interpolation uses `${env:NAME}`.
- The supported protocol revision is `UNKNOWN`.

**Rules for SFX:**
- No credential value is ever committed.
- The documented configuration uses local scope (Claude Code `--header`, or `headersHelper`) or an environment reference (Cursor `${env:INTERGRAX_HARNESS_API_KEY}`).
- A committed `.mcp.json` or `.cursor/mcp.json` is **not** part of this ADR. Adding one is persistent configuration and needs a separate operator decision.
- The residual risks FACTS R-01..R-04 (`FACTS:217`) are qualified live in SFX-P10-02 / SFX-P10-08:
  - R-01: the Cursor protocol revision;
  - R-02: the revision negotiated by FastMCP 3.3.1;
  - R-03: client behaviour on the harness 401, which carries no `WWW-Authenticate` header;
  - R-04: a live interop proof.

### 6. Toolkit path v0 → v1

**v0, pre-freeze (OD-SFX-04; SFX-P4-A01, T-IGX-037-E).**
- agent-env sources only: rule `intergrax-scenario-factory` (`loading: auto`), skills `sfx-*`, and persona briefs `agent-env/agents/sfx-*.md`.
- Outputs are L1 drafts under `.tmp/sfx/<slug>/` against the SFX-P2 design schemas.
- Personas are read as plain files and passed to a generic subagent (FACTS §5.3, `FACTS:193-195`), so there is no native agent discovery, no per-persona tool restriction, and no generator change.
- v0 calls no CLI or MCP, because neither exists before W3/W5.

**v1, post-freeze.**
- **SFX-P4-B01 (W3)** extends `sync_agent_env.py` to generate native agent definitions (`.claude/agents/`, plus a decision on `.cursor/agents/`) (FACTS §5.2, `FACTS:183-191`).
- **SFX-P4-B02 (W3)** adds the role-registry entries.
- **SFX-P10-08 (W5)** rewires the skills to call the CLI offline (L0/L1) and MCP in live mode, with parity tests showing that IDE artifacts equal Studio artifacts.

## Consequences

### Positive

- One implementation of every operation. Doors are cheap to add and to test, and parity is mechanically checked.
- The MCP surface is exactly the SFX operation set, and GAP-17 widening is structurally excluded.
- The offline CLI gives IDE agents a deterministic, credential-free door for L0/L1 work.

### Negative

- The parity suite (SFX-P10-03) is a permanent maintenance cost. Every new operation touches the service layer, the operation map and three adapters.
- The SFX MCP server cannot reuse the legacy Nexus and catalog surfaces on the same definition. A host that wants both runs them as separate definitions.

### Risks (findings from FACTS, recorded; generator unchanged by this ADR)

| ID | Risk | Evidence | Mitigation |
|----|------|----------|------------|
| **F-01** | Cursor rule mapping for `loading: explicit`. The generator writes every rule as `description` plus `alwaysApply` with no `globs`, so Cursor classifies explicit rules as **Apply Intelligently**, not manual `@`-only as `agent-env/README.md:26` states. Operator-only rules may be auto-applied in Cursor. | `sync_agent_env.py:100-108`; `agent-env/README.md:26`; FACTS §3.1 (`FACTS:99-108`) | SFX is not blocked, because its rule is `loading: auto`. SFX safety invariants live in rule and skill **content** (L0/L1 only, `.tmp/sfx/` only, no runtime), never in the loading mode. The candidate fix, omitting `description` from `.mdc` for explicit rules, is proposed for SFX-P4-B01 scope and needs operator approval. It is not implemented here. |
| **F-02** | Duplicate skill discovery. Cursor also scans `.claude/skills/`, so every generated skill is discoverable twice (`.cursor/skills/<id>` and `.claude/skills/<id>`). Cursor's de-duplication behaviour is `UNKNOWN`. | `sync_agent_env.py:164-167`; FACTS §3.2 (`FACTS:113`, `:117`) | Both adapters are thin pointers to the same canonical `agent-env/skills/<id>/SKILL.md` (`sync_agent_env.py:115-120`), so duplicates are semantically identical. SFX skill names are unique (`sfx-*`). A live check of Cursor duplicate handling is part of SFX-P10-08. Whether to emit one skill root per tool is an SFX-P4-B01 decision for the operator. The same question applies to `.claude/agents/` versus `.cursor/agents/` in v1 (`FACTS:122`). |
| R-004-03 | The MCP API-key principal differs from the Studio JWT principal. | `harness_auth.py:60-68`, `:338-343` | PC-3 requires one principal model. The mapping of API-key callers to a harness principal and their maximum authority level is fixed and tested in SFX-P10-02. Until then the MCP door is limited to L0/L1. |

## Related

- Compliance: tier boundaries unchanged; streamable HTTP only; no secrets in configuration, logs or evidence; SFX never grants L3 (§7.10, §13 "Safe tools").
- Revision triggers: any second validation path in a door; a new MCP transport; union-based allowlist composition; native agent generation landing (SFX-P4-B01) with a field mapping that differs from FACTS §5.2.
- Verification when implemented: parity suite PC-1..PC-6 (SFX-P10-03), the MCP no-widening test (SFX-P10-02), CLI tests (SFX-P5-07), and `sync_agent_env.py --check` with sync tests (SFX-P4-B01, SFX-P10-08).
- ADR status stays **Proposed** until OD-SFX-06.

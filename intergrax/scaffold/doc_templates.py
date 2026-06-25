# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ARCHITECTURE.md and IMPLEMENTATION_PLAN.md templates for Tier-2/Tier-3 scaffolds."""

from __future__ import annotations

from datetime import date
from textwrap import dedent

from intergrax.scaffold.agent_catalog import ScaffoldAgentSpec
from intergrax.scaffold.application_names import ScaffoldApplicationNames


def _today() -> str:
    return date.today().isoformat()


def _cap_prefix(slug: str) -> str:
    return slug.upper().replace("-", "_")


def render_agent_architecture_doc(
    *,
    slug: str,
    class_name: str,
    capabilities: list[str],
    reference: bool = False,
    pattern: str | None = None,
) -> str:
    caps = ", ".join(f"`{c}`" for c in capabilities)
    agent_base = f"`{class_name}`"
    if pattern is not None:
        runtime_line = (
            f"- Typed **{pattern}** cognitive pattern (`CognitiveAgent` / `on_next_step`)\n"
            "- Stub LLM adapter in `{slug}_agent.py` for offline smoke tests"
        )
        purpose = (
            f"Tier-2 typed **{pattern}** agent scaffold for `{slug}`. "
            "Implement domain logic in `perceive` / `reason` / `act` / `evaluate`."
        )
        layout_rows = (
            f"| `{slug}_agent.py` | {agent_base} — cognitive pattern hooks |\n"
            "| `contract.py` | `AgentContract` + `cognitive_pattern` |\n"
            "| `capabilities.py` | Capability ids |\n"
            "| `steps/domain_job.py` | Domain step entry — **Cursor implementation point** |\n"
            "| `prompts/system.md` | Prompt assets |\n"
            "| `schemas/` | I/O models |\n"
            "| `tests/` | Agent smoke tests |\n"
            "| `adr/` | Architecture decision records — [`adr/README.md`](adr/README.md) |"
        )
    elif reference:
        runtime_line = (
            "- `HarnessReferenceAgent` + `on_next_step` (ACP reference probe)\n"
            "- Optional `LabHarnessContext` injected by Tier-3 host builders"
        )
        purpose = f"Tier-2 harness reference agent scaffold for `{slug}`."
        layout_rows = (
            f"| `{slug}_agent.py` | {agent_base} — `on_next_step` entry |\n"
            "| `contract.py` | `AgentContract` |\n"
            "| `capabilities.py` | Capability ids |\n"
            "| `prompts/system.md` | Prompt assets |\n"
            "| `schemas/` | I/O models |\n"
            "| `tests/` | Agent smoke tests |\n"
            "| `adr/` | Architecture decision records — [`adr/README.md`](adr/README.md) |"
        )
    else:
        runtime_line = (
            "- Typed **reflex** cognitive pattern (`CognitiveAgent` / `on_next_step`)\n"
            "- Stub LLM adapter in `{slug}_agent.py` for offline smoke tests"
        )
        purpose = (
            f"Tier-2 typed agent scaffold for `{slug}`. "
            "Implement domain logic in `perceive` / `reason` / `act` / `evaluate`."
        )
        layout_rows = (
            f"| `{slug}_agent.py` | {agent_base} — cognitive pattern hooks |\n"
            "| `contract.py` | `AgentContract` + `cognitive_pattern` |\n"
            "| `capabilities.py` | Capability ids |\n"
            "| `steps/domain_job.py` | Domain step entry — **Cursor implementation point** |\n"
            "| `prompts/system.md` | Prompt assets |\n"
            "| `schemas/` | I/O models |\n"
            "| `tests/` | Agent smoke tests |\n"
            "| `adr/` | Architecture decision records — [`adr/README.md`](adr/README.md) |"
        )
    return dedent(
        f"""\
        # {slug} agent — architecture

        **Status:** Scaffold baseline ({_today()})

        Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

        ---

        ## Purpose

        {purpose}

        ## Capabilities

        {caps}

        ## Layout

        | Path | Role |
        |------|------|
        {layout_rows}

        ## Runtime

        {runtime_line}

        ## Pattern anchor (Cursor)

        - Tool invocation pattern: [`agents/lkw_shared/PATTERN.md`](../../agents/lkw_shared/PATTERN.md)
        - **Implementation point:** `steps/domain_job.py` — implement `run_domain_job(step_ctx)`; wire from `act()`.
        - Do **not** grep runtime/Nexus for `invoke_tool` when this section is in read scope.

        ## Tier hygiene

        - Imports only `intergrax.*` and `agents/{slug}` — **no** `applications/` imports
        - Tools resolved by Tier-3 host `ToolProfile` / `ApplicationEnvironmentProfile`

        ## Registration

        - Programmatic: `AgentRegistry.register({class_name}())`
        - Tier-3 host: `AgentBinding.mount({class_name}, ...)` in application `manifest.py`
        - Workflow: [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md) Step 4
        """
    )


def render_agent_implementation_plan(
    *,
    slug: str,
    class_name: str,
    capabilities: list[str],
    reference: bool = False,
    pattern: str | None = None,
) -> str:
    prefix = _cap_prefix(slug)
    primary = capabilities[0] if capabilities else f"{slug}.basic"
    reference_note = (
        "\n- Harness reference agent — keep gate smoke stable when editing shared lab hosts."
        if reference
        else ""
    )
    if pattern is not None:
        domain_task = (
            f"| {prefix}-1 | Implement domain hooks in `{slug}_agent.py` "
            f"(perceive/reason/act/evaluate) | Planned | High | {pattern} pattern |"
        )
    else:
        domain_task = (
            f"| {prefix}-1 | Implement domain hooks in `{slug}_agent.py` "
            f"(perceive/reason/act/evaluate) | Planned | High | ACP pattern |"
        )
    return dedent(
        f"""\
        # {slug} agent — Implementation Plan

        **The implementation map** for this Tier-2 agent — phases, status, gaps, and verification.

        Status: Working draft ({_today()}) — **Scaffold baseline**

        Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
        Platform plan: [`docs/intergrax_runtime_architecture.md`](../../docs/intergrax_runtime_architecture.md)  
        Agent workflow: [`docs/guides/AGENT_CREATION_GUIDE.md`](../../docs/guides/AGENT_CREATION_GUIDE.md)

        Principle: **evolve, not rewrite** · **reuse Tier-0** · **no Tier-3 imports in agent code**

        ---

        ## Documentation model

        Do not maintain separate status/readiness files under this agent. Use:

        | Topic | Where |
        |-------|--------|
        | Purpose, contracts, I/O, runtime layout | **ARCHITECTURE.md** (this directory) |
        | Task status, phases, next steps | **This file** |
        | Significant agent architecture decisions | **`adr/`** — [`adr/README.md`](adr/README.md) |
        | Platform harness work | `docs/intergrax_runtime_architecture.md` §6.1 |
        | UAEP / Nexus workflow | `docs/guides/AGENT_CREATION_GUIDE.md` |

        ---

        ## 0. Scope at a glance

        | Field | Value |
        |-------|-------|
        | Agent id | `{slug}` |
        | Class | `{class_name}` |
        | Primary capability | `{primary}` |
        | Tier | Tier-2 (`agents/{slug}/`) |
        | Host wiring | Tier-3 application manifest (when mounted) |

        ---

        ## 1. Implementation queue

        | ID | Task | Status | Priority | Notes |
        |----|------|--------|----------|-------|
        {domain_task}
        | {prefix}-2 | Extend `prompts/system.md` for domain | Planned | Medium | Keep prompts versioned here |
        | {prefix}-3 | Register skills/tools on `contract.py` | Planned | Medium | See `docs/architecture/SKILLS.md` |
        | {prefix}-4 | Agent smoke test green | Done | High | `tests/test_{slug}_agent.py` |
        | {prefix}-5 | Mount in Tier-3 host (optional) | Planned | Medium | `AgentBinding.mount({class_name}, ...)` |

        ---

        ## 2. Verification

        ```bash
        uv run pytest agents/{slug}/tests -q
        ```

        After host wiring:

        ```bash
        uv run pytest applications/<app>_application/<app>_application_tests -q
        ```
        {reference_note}

        ---

        ## 3. Platform alignment

        Business agents and product-only work remain **end of plan** unless explicitly reprioritized —
        see platform [`§6.3`](../../docs/intergrax_runtime_architecture.md#63-end-of-plan--deferred-product-work-only).
        """
    )


def render_application_architecture_doc(
    *,
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
    profile: str,
    minimal: bool = False,
) -> str:
    agents_list = ", ".join(s.class_name for s in specs)
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    deploy_section = (
        "| Deploy triad | `docker/`, `BUILD_AND_DEPLOY.md` |\n"
        if not minimal
        else "| Deploy triad | Deferred — run `python -m intergrax.scaffold expand` to promote |\n"
    )
    profile_label = "lab" if profile == "lab" else "product"
    return dedent(
        f"""\
        # {names.pkg} — architecture

        **Status:** Scaffold baseline ({_today()}) — **{profile_label} profile**

        Implementation tracker: [`IMPLEMENTATION_PLAN.md`](IMPLEMENTATION_PLAN.md)

        ---

        ## Purpose

        Tier-3 **{profile_label}** host for {agents_list}. HTTP entry via `POST {names.route_prefix}/run`.

        ## Manifest

        - `app_id="{names.short}"` in `manifest.py`
        - `ApplicationEnvironmentProfile` from `host/environment_profile.py`
        - Agent roster via `AgentBinding.mount(...)`

        ## Factory

        - `host/factory.py` → `build_harness_host_runtime`
        - Registry via `host/wiring.py` → `build_application_registry`

        ## Agents

        {agents_list}

        ## Example capability

        `{cap}`

        {deploy_section}
        ## Architecture decisions

        - `adr/` — [`adr/README.md`](adr/README.md) (decision index + `TEMPLATE.md`)

        ## Dependencies (pyproject.toml)

        - Core `Intergrax-ai` install from repository root (`uv sync`)
        - Optional: `[harness-author]` for external repos; `[dev-ci]` for gate tests
        - LLM provider env vars per `.env.example` / `BUILD_AND_DEPLOY.md`

        ## Docs

        - Engine: `intergrax/applications/USAGE.md`
        - Layout: `applications/USAGE.md`
        """
    )


def render_application_implementation_plan(
    *,
    names: ScaffoldApplicationNames,
    specs: list[ScaffoldAgentSpec],
    profile: str,
    minimal: bool = False,
) -> str:
    prefix = _cap_prefix(names.short)
    agents_csv = ", ".join(s.slug for s in specs)
    cap = specs[0].capabilities[0] if specs and specs[0].capabilities else "echo.basic"
    deploy_rows = (
        f"| {prefix}-4 | Deploy triad present (`docker/`, `BUILD_AND_DEPLOY.md`) | Done | High | Gate `test_application_deploy_triad` |\n"
        f"| {prefix}-5 | MCP coupled to FastAPI factory | Done | Medium | `mcp/server.py` |\n"
        if not minimal
        else f"| {prefix}-4 | Promote minimal host to standard deploy triad | Planned | High | `python -m intergrax.scaffold expand {names.short}` |\n"
        f"| {prefix}-5 | Add MCP + full factory | Planned | Medium | Via `expand` subcommand |\n"
    )
    profile_label = "lab" if profile == "lab" else "product"
    health_path = f"{names.route_prefix}/agents" if profile == "lab" else "/health"
    return dedent(
        f"""\
        # {names.display} — Implementation Plan

        **The implementation map** for this Tier-3 application — phases, status, gaps, and verification.

        Status: Working draft ({_today()}) — **{profile_label} profile scaffold**

        Architecture: [`ARCHITECTURE.md`](ARCHITECTURE.md)  
        Platform plan: [`docs/intergrax_runtime_architecture.md`](../../docs/intergrax_runtime_architecture.md)  
        Application engine: [`intergrax/applications/USAGE.md`](../../intergrax/applications/USAGE.md)

        Principle: **compose Tier-0** · **no business logic in Nexus** · **manifest-driven wiring**

        ---

        ## Documentation model

        Do not maintain separate status/readiness files under this application. Use:

        | Topic | Where |
        |-------|--------|
        | Host purpose, manifest, factory, dependencies | **ARCHITECTURE.md** (this directory) |
        | Task status, phases, next steps | **This file** |
        | Significant application architecture decisions | **`adr/`** — [`adr/README.md`](adr/README.md) |
        | Platform harness work | `docs/intergrax_runtime_architecture.md` §6.1 |
        | Scaffold / deploy recipes | `applications/TIER3_READINESS.md` |

        ---

        ## 0. Scope at a glance

        | Field | Value |
        |-------|-------|
        | Package | `{names.pkg}` |
        | Profile | `{profile_label}` |
        | Route prefix | `{names.route_prefix}` |
        | Default port | `{names.port}` |
        | Mounted agents | {agents_csv} |
        | Smoke capability | `{cap}` |

        ---

        ## 1. Implementation queue

        | ID | Task | Status | Priority | Notes |
        |----|------|--------|----------|-------|
        | {prefix}-1 | Customize `host/environment_profile.py` | Planned | High | Tool/skill/integration profiles |
        | {prefix}-2 | Tune `manifest.py` agent bindings | Done | High | Scaffold defaults |
        | {prefix}-3 | Host smoke tests green | Done | High | `{names.tests_pkg}/host/` |
        {deploy_rows}| {prefix}-6 | Product/domain serving routes | Planned | Medium | `serving/` extensions |

        ---

        ## 2. Verification

        ```bash
        uv run pytest applications/{names.pkg}/{names.tests_pkg} -q
        ```

        Local run:

        ```bash
        cp applications/{names.pkg}/.env.example applications/{names.pkg}/.env
        uv run uvicorn {names.pkg}.host.main:app --host 127.0.0.1 --port {names.port}
        curl -s http://127.0.0.1:{names.port}{health_path}
        ```

        ---

        ## 3. Platform alignment

        Tier-3 product environments follow explicit reprioritization — see platform
        [`§6.3`](../../docs/intergrax_runtime_architecture.md#63-end-of-plan--deferred-product-work-only)
        and [`§6.3a`](../../docs/intergrax_runtime_architecture.md#63a-business-backlog-register-consolidated).
        """
    )

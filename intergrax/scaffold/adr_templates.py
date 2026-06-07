# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""ADR README and TEMPLATE scaffolds for Harness, Tier-2 agents, and Tier-3 applications."""

from __future__ import annotations

from datetime import date
from pathlib import Path
from textwrap import dedent

ADR_README = "README.md"
ADR_TEMPLATE = "TEMPLATE.md"


def _today() -> str:
    return date.today().isoformat()


def adr_prefix(slug: str) -> str:
    """Uppercase ADR area prefix from a slug (e.g. local_workspace → LOCAL_WORKSPACE)."""
    return slug.upper().replace("-", "_")


def _write_adr_file(path: Path, content: str, *, force: bool) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def render_adr_template(*, prefix: str, title_hint: str, related_hint: str) -> str:
    return dedent(
        f"""\
        # ADR-{prefix}-NNN: {title_hint}

        | Field | Value |
        |-------|-------|
        | **Status** | Proposed |
        | **Date** | YYYY-MM-DD |
        | **Deciders** | Team / role |
        | **Related** | {related_hint} |

        ## Context

        What problem or constraint requires a decision? What alternatives exist?

        ## Decision

        State the chosen option clearly. List rejected options and why they were rejected.

        ## Consequences

        ### Positive

        - …

        ### Negative

        - …

        ## Compliance

        - Tier boundaries preserved
        - Policy / security constraints respected
        - Linked architecture and plan docs updated

        ## Implementation notes

        - Code paths, tests, and verification commands
        """
    )


def render_harness_adr_readme() -> str:
    return dedent(
        f"""\
        # Intergrax Harness — Architecture Decision Records

        **Domain:** Tier-0 platform + Tier-1 Nexus (`intergrax/`, `intergrax/runtime/`)

        Canonical architecture: [`../intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md)  
        Implementation tracker: [`../INTERGRAX_IMPLEMENTATION_PLAN.md`](../INTERGRAX_IMPLEMENTATION_PLAN.md)

        ---

        ## When to write an ADR

        Create or update an ADR for **significant** Harness decisions, including:

        - Nexus execution semantics, orchestration contracts, lifecycle, delegation
        - Tool / skill / integration layer boundaries and catalog contracts
        - LLM adapter envelopes, RAG retrieval policy, memory models
        - Policy, HITL, observability, and cross-cutting platform behavior
        - New universal Tier-0 mechanisms or changes that affect multiple agents

        **Not required:** typo fixes, test-only changes, agent-specific business logic (use agent ADRs),
        or product-host wiring that does not change platform contracts.

        If no ADR is needed, record **"no ADR needed"** with rationale in the PR or plan row.

        ## Naming

        ```text
        ADR-{{AREA}}-{{NNN}}.md
        ```

        Examples: `ADR-FLOW-001`, `ADR-LLM-001`, `ADR-ADAPT-001`.

        ## Process

        1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential id for your area tag.
        2. Fill **Context**, **Decision**, **Consequences**, and **Compliance**.
        3. Link from canon (`intergrax_runtime_architecture.md`) and/or `INTERGRAX_IMPLEMENTATION_PLAN.md`.
        4. Set **Status** to `Accepted` when implemented; `Superseded` when replaced.

        ## Index

        | ADR | Title | Status |
        |-----|-------|--------|
        | [ADR-FLOW-001](ADR-FLOW-001.md) | Declarative delegation (`DELEGATES_TO`) expansion | Accepted · implemented |
        | [ADR-FLOW-002](ADR-FLOW-002.md) | Reserved lifecycle states | Accepted |
        | [ADR-FLOW-003](ADR-FLOW-003.md) | `MODIFY_PLAN` decision semantics | Accepted |
        | [ADR-ADAPT-001](ADR-ADAPT-001.md) | Adaptive Harness Intelligence over classical RL | Accepted |
        | [ADR-LLM-001](ADR-LLM-001.md) | Typed LLM adapter response envelope | Accepted |

        ---

        *Scaffold baseline: {_today()}*
        """
    )


def render_agent_adr_readme(*, slug: str) -> str:
    prefix = adr_prefix(slug)
    return dedent(
        f"""\
        # {slug} agent — Architecture Decision Records

        **Domain:** Tier-2 business agent (`agents/{slug}/`)

        Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)  
        Implementation tracker: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)  
        Platform ADRs: [`../../docs/adr/README.md`](../../docs/adr/README.md)

        ---

        ## When to write an ADR

        Create an ADR for agent-level decisions that affect **domain behavior, contracts, or integration choices**, for example:

        - Capability model, I/O schemas, or multi-step pipeline structure
        - Tool/skill selection policy specific to this agent
        - Prompt strategy, evaluation hooks, or risk classification changes
        - External data sources or vendor choices consumed through Harness tools

        **Not required:** harness platform changes (use `docs/adr/`), Tier-3 host wiring (use application ADRs),
        or trivial refactors with no behavioral impact.

        ## Naming

        ```text
        ADR-{prefix}-{{NNN}}.md
        ```

        ## Process

        1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential number.
        2. Link from [`ARCHITECTURE.md`](../ARCHITECTURE.md) when the decision affects runtime layout.
        3. Track implementation in [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

        ## Index

        | ADR | Title | Status |
        |-----|-------|--------|
        | — | *No agent ADRs yet* | — |

        ---

        *Scaffold baseline: {_today()}*
        """
    )


def render_application_adr_readme(*, pkg: str, short: str, display: str) -> str:
    prefix = adr_prefix(short)
    return dedent(
        f"""\
        # {display} — Architecture Decision Records

        **Domain:** Tier-3 application host (`applications/{pkg}/`)

        Architecture: [`../ARCHITECTURE.md`](../ARCHITECTURE.md)  
        Implementation tracker: [`../IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)  
        Platform ADRs: [`../../docs/adr/README.md`](../../docs/adr/README.md)

        ---

        ## When to write an ADR

        Create an ADR for **product-environment** decisions, for example:

        - Manifest roster, agent bindings, or execution graph topology
        - Environment profile, tool/skill/integration profiles for this host
        - Serving API shape, auth model, deployment topology, or MCP exposure
        - Cross-agent orchestration declared in this application (not Nexus core semantics)

        **Not required:** Nexus platform contract changes (use `docs/adr/`), single-agent domain logic (use agent ADRs),
        or configuration-only tweaks with no architectural impact.

        ## Naming

        ```text
        ADR-{prefix}-{{NNN}}.md
        ```

        ## Process

        1. Copy [`TEMPLATE.md`](TEMPLATE.md) to the next sequential number.
        2. Link from [`ARCHITECTURE.md`](../ARCHITECTURE.md) when the decision affects host layout.
        3. Track implementation in [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md).

        ## Index

        | ADR | Title | Status |
        |-----|-------|--------|
        | — | *No application ADRs yet* | — |

        ---

        *Scaffold baseline: {_today()}*
        """
    )


def write_harness_adr_scaffold(*, root: Path, force: bool = False) -> Path:
    adr_dir = root / "docs" / "adr"
    _write_adr_file(
        adr_dir / ADR_README,
        render_harness_adr_readme(),
        force=force,
    )
    _write_adr_file(
        adr_dir / ADR_TEMPLATE,
        render_adr_template(
            prefix="AREA",
            title_hint="Short decision title",
            related_hint="[`intergrax_runtime_architecture.md`](../intergrax_runtime_architecture.md) · plan row",
        ),
        force=force,
    )
    return adr_dir


def write_agent_adr_scaffold(*, agent_dir: Path, slug: str, force: bool = False) -> Path:
    adr_dir = agent_dir / "adr"
    prefix = adr_prefix(slug)
    _write_adr_file(
        adr_dir / ADR_README,
        render_agent_adr_readme(slug=slug),
        force=force,
    )
    _write_adr_file(
        adr_dir / ADR_TEMPLATE,
        render_adr_template(
            prefix=prefix,
            title_hint="Short agent decision title",
            related_hint=f"[`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)",
        ),
        force=force,
    )
    return adr_dir


def write_application_adr_scaffold(
    *,
    app_dir: Path,
    pkg: str,
    short: str,
    display: str,
    force: bool = False,
) -> Path:
    adr_dir = app_dir / "adr"
    prefix = adr_prefix(short)
    _write_adr_file(
        adr_dir / ADR_README,
        render_application_adr_readme(pkg=pkg, short=short, display=display),
        force=force,
    )
    _write_adr_file(
        adr_dir / ADR_TEMPLATE,
        render_adr_template(
            prefix=prefix,
            title_hint="Short application decision title",
            related_hint=f"[`ARCHITECTURE.md`](../ARCHITECTURE.md) · [`IMPLEMENTATION_PLAN.md`](../IMPLEMENTATION_PLAN.md)",
        ),
        force=force,
    )
    return adr_dir

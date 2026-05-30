# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Known Tier-2 agents for ``scaffold new-application --agents``."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class ScaffoldAgentSpec:
    """One agent entry for generated manifest and builders."""

    slug: str
    module: str
    class_name: str
    capabilities: tuple[str, ...] = ()


# Built-in reference agents (must exist under ``agents/`` on pythonpath).
BUILTIN_AGENTS: dict[str, tuple[ScaffoldAgentSpec, ...]] = {
    "echo": (
        ScaffoldAgentSpec("echo", "echo.echo_agent", "EchoAgent", ("echo.basic",)),
    ),
    "signoff_probe": (
        ScaffoldAgentSpec("signoff_probe", "signoff_probe.signoff_probe_agent", "SignoffProbeAgent", ()),
    ),
    "research": (
        ScaffoldAgentSpec("research", "research.research_agent", "ResearchAgent", ()),
        ScaffoldAgentSpec("summary", "research.summary_agent", "SummaryAgent", ()),
    ),
}


def _class_name(slug: str) -> str:
    return "".join(part.capitalize() for part in slug.split("_")) + "Agent"


def resolve_agent_specs(slugs: list[str]) -> list[ScaffoldAgentSpec]:
    """Resolve ``--agents`` slugs to import paths and class names."""
    if not slugs:
        slugs = ["echo"]

    specs: list[ScaffoldAgentSpec] = []
    seen_classes: set[str] = set()

    for raw in slugs:
        key = raw.strip().lower().replace("-", "_")
        if not key:
            continue
        if key in BUILTIN_AGENTS:
            for item in BUILTIN_AGENTS[key]:
                if item.class_name in seen_classes:
                    continue
                seen_classes.add(item.class_name)
                specs.append(item)
            continue

        class_name = _class_name(key)
        if class_name in seen_classes:
            continue
        seen_classes.add(class_name)
        specs.append(
            ScaffoldAgentSpec(
                slug=key,
                module=f"{key}.{key}_agent",
                class_name=class_name,
                capabilities=(f"{key}.basic",),
            )
        )

    if not specs:
        raise ValueError("No agents resolved; use --agents echo or a scaffolded agent slug")
    return specs

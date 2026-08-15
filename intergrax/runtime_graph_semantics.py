# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Pure runtime-graph dependency semantics shared across consumers.

Tier-neutral helpers for distribution-name normalization, dependency parsing,
package taxonomy classification, and cycle reporting. No imports from
``applications/`` or ``agent_distribution/``.
"""

from __future__ import annotations

import re
from enum import Enum

_AGENT_DIST_RE = re.compile(r"^intergrax-(.+)-agent$")
_APPLICATION_DIST_RE = re.compile(r"^intergrax-.+-application$", re.IGNORECASE)
_EXTRA_RE = re.compile(r"Intergrax-ai(?:\[([^\]]*)\])?", re.IGNORECASE)
_ASSISTANT_AGENT_DIST = "intergrax-assistant-agent"


class GraphVisitState(Enum):
    """DFS visit markers for agent dependency traversal."""

    UNVISITED = 0
    VISITING = 1
    VISITED = 2


def normalize_distribution_name(name: str) -> str:
    """Canonical lowercase hyphenated distribution key."""
    return name.strip().lower().replace("_", "-")


def parse_dependency_name(dep: str) -> str:
    """Extract bare distribution name from a PEP 508 dependency string."""
    raw = dep.strip().strip("\"'")
    raw = raw.split(";")[0].strip()
    raw = re.split(r"[<>=!~\[]", raw, maxsplit=1)[0].strip()
    return raw


def parse_platform_extras(dep: str) -> tuple[str, ...]:
    """Parse optional extras from an ``Intergrax-ai[...]`` dependency string."""
    match = _EXTRA_RE.search(dep)
    if not match or not match.group(1):
        return ()
    return tuple(part.strip() for part in match.group(1).split(",") if part.strip())


def is_platform_dependency(name: str) -> bool:
    """Return True when ``name`` refers to the Tier-0 platform distribution."""
    return normalize_distribution_name(name) == "intergrax-ai"


def is_agent_distribution(name: str) -> bool:
    """Return True when ``name`` matches the Tier-2 agent naming convention."""
    cleaned = name.strip()
    if normalize_distribution_name(cleaned) == _ASSISTANT_AGENT_DIST:
        return True
    return _AGENT_DIST_RE.match(cleaned) is not None


def is_application_distribution(name: str) -> bool:
    """Return True when ``name`` matches the Tier-3 application naming convention."""
    return _APPLICATION_DIST_RE.match(name.strip()) is not None


def agent_distribution_name(agent_dir: str) -> str:
    """Map agent workspace directory name to canonical distribution name."""
    if agent_dir == "intergrax_assistant":
        return _ASSISTANT_AGENT_DIST
    return f"intergrax-{agent_dir.replace('_', '-')}-agent"


def agent_dir_from_distribution(dist: str) -> str | None:
    """Map agent distribution name back to workspace directory name."""
    cleaned = dist.strip()
    if normalize_distribution_name(cleaned) == _ASSISTANT_AGENT_DIST:
        return "intergrax_assistant"
    match = _AGENT_DIST_RE.match(cleaned)
    if not match:
        return None
    return match.group(1).replace("-", "_")


def format_agent_dependency_cycle(path: list[str], closing: str) -> str:
    """Format a detected agent dependency cycle for fail-closed errors."""
    chain = path + [closing]
    return "AGENT_DEPENDENCY_CYCLE:\n→ " + "\n→ ".join(chain)

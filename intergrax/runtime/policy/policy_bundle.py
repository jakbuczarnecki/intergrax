# © Artur Czarnecki. All rights reserved.

"""Composed runtime policy bundle (architecture §42.11.4, Phase R-Policy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
from intergrax.runtime.tools.scope_policy import ToolScopePolicy


@dataclass(frozen=True, slots=True)
class RuntimePolicyBundle:
    """
    Single Tier-3 composition object referencing live policy engines.

    Nexus and UAEP read from this bundle instead of ad-hoc policy construction.
    ``budget`` / ``plan_loop`` use Nexus config types at wiring time (avoid import cycles).
    """

    tool_access: ToolAccessPolicy | ToolScopePolicy | None = None
    budget: Any | None = None
    plan_loop: Any | None = None
    require_human_on_critical: bool = True
    domain_fragments: Dict[str, Any] = field(default_factory=dict)

    def fragment(self, key: str) -> Optional[Any]:
        return self.domain_fragments.get(key)

# © Artur Czarnecki. All rights reserved.

"""Composed runtime policy bundle (architecture §42.11.4, Phase R-Policy)."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from intergrax.core.plugins.admission import DomainPluginLoadReport
from intergrax.runtime.nexus.budget.budget_models import BudgetPolicy
from intergrax.runtime.nexus.planning.plan_loop_models import PlanLoopPolicy
from intergrax.runtime.nexus.tools.tool_access_policy import ToolAccessPolicy
from intergrax.runtime.policy.rules.evaluation import PolicyEnforcementMode
from intergrax.runtime.policy.rules.provenance import PolicyBundleProvenance
from intergrax.runtime.policy.rules.registry import PolicyRuleRegistry
from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule
from intergrax.runtime.tools.scope_policy import ToolScopePolicy


@dataclass(frozen=True, slots=True)
class DeclarativePolicyRuntime:
    """
    Immutable declarative policy composition for a single ``RuntimePolicyBundle``.

    ``registry`` is populated during host wiring only; consumers treat it as
  read-only after bundle publication. Enforcement is BLOCK B (CAND-007).
    """

    registry: PolicyRuleRegistry
    rules: tuple[DeclarativePolicyRule, ...]
    load_report: DomainPluginLoadReport
    provenance: PolicyBundleProvenance
    enforcement_mode: PolicyEnforcementMode = PolicyEnforcementMode.AUDIT_ONLY


@dataclass(frozen=True, slots=True)
class RuntimePolicyBundle:
    """
    Single Tier-3 composition object referencing live policy engines.

    Nexus and UAEP read from this bundle instead of ad-hoc policy construction.
    ``budget`` / ``plan_loop`` use Nexus config types at wiring time (avoid import cycles).
    """

    tool_access: ToolAccessPolicy | ToolScopePolicy | None = None
    budget: BudgetPolicy | None = None
    plan_loop: PlanLoopPolicy | None = None
    require_human_on_critical: bool = True
    domain_fragments: Dict[str, Any] = field(default_factory=dict)
    declarative_policy_runtime: DeclarativePolicyRuntime | None = None

    def fragment(self, key: str) -> Optional[Any]:
        return self.domain_fragments.get(key)

# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Qualification plan for cross-domain execution (DIAG-FUNCTIONAL-Q5)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.core.qualification.functional_qualification_identity import FunctionalQualificationPluginId
from intergrax.core.qualification.functional_qualification_registry import (
    QualificationPluginRegistry,
    QualificationPluginRegistryError,
)


@dataclass(frozen=True, slots=True)
class QualificationPlan:
    plugin_ids: tuple[FunctionalQualificationPluginId, ...]
    repeatability_required: bool = True
    continue_on_plugin_failure: bool = True


def resolve_plan_plugins(
    plan: QualificationPlan,
    registry: QualificationPluginRegistry,
) -> tuple[FunctionalQualificationPluginId, ...]:
    resolved: list[FunctionalQualificationPluginId] = []
    for plugin_id in plan.plugin_ids:
        try:
            registry.get(plugin_id)
        except QualificationPluginRegistryError as exc:
            raise QualificationPluginRegistryError(
                f"plan_references_unknown_plugin:{plugin_id.value}",
            ) from exc
        resolved.append(plugin_id)
    return tuple(resolved)


__all__ = [
    "QualificationPlan",
    "resolve_plan_plugins",
]

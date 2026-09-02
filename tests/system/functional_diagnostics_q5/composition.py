# © Artur Czarnecki. All rights reserved.

"""Q5 composition root for cross-domain functional qualification."""

from __future__ import annotations

from intergrax.core.qualification.functional_qualification_identity import (
    MODEL_ROUTING_PLUGIN_ID,
    RAG_PLUGIN_ID,
    TOOL_SELECTION_PLUGIN_ID,
    WEB_SEARCH_PLUGIN_ID,
)
from intergrax.core.qualification.functional_qualification_plan import QualificationPlan
from intergrax.core.qualification.functional_qualification_registry import QualificationPluginRegistry
from tests.system.functional_diagnostics_q5.plugins import (
    ModelRoutingQualificationPlugin,
    RagQualificationPlugin,
    ToolSelectionQualificationPlugin,
    WebSearchQualificationPlugin,
)


def build_q5_qualification_registry() -> QualificationPluginRegistry:
    registry = QualificationPluginRegistry()
    registry.register(RagQualificationPlugin())
    registry.register(ToolSelectionQualificationPlugin())
    registry.register(WebSearchQualificationPlugin())
    registry.register(ModelRoutingQualificationPlugin())
    return registry


def build_q5_qualification_plan() -> QualificationPlan:
    return QualificationPlan(
        plugin_ids=(
            RAG_PLUGIN_ID,
            TOOL_SELECTION_PLUGIN_ID,
            WEB_SEARCH_PLUGIN_ID,
            MODEL_ROUTING_PLUGIN_ID,
        ),
        repeatability_required=True,
        continue_on_plugin_failure=True,
    )

# © Artur Czarnecki. All rights reserved.

"""Unit tests for functional qualification plugin registry."""

from __future__ import annotations

import pytest

from intergrax.core.qualification.functional_qualification_identity import (
    FunctionalQualificationPluginId,
    RAG_PLUGIN_ID,
    WEB_SEARCH_PLUGIN_ID,
)
from intergrax.core.qualification.functional_qualification_plugin import QualificationPluginDescriptor
from intergrax.core.qualification.functional_qualification_registry import (
    QualificationPluginRegistry,
    QualificationPluginRegistryError,
)
from intergrax.core.qualification.functional_qualification_result import (
    QualificationPluginMetrics,
    QualificationPluginResult,
)
from intergrax.core.qualification.functional_qualification_verdict import QualificationVerdict

pytestmark = pytest.mark.unit


class _StubPlugin:
    def __init__(self, plugin_id: FunctionalQualificationPluginId) -> None:
        self._descriptor = QualificationPluginDescriptor(
            plugin_id=plugin_id,
            domain="stub",
            version="1",
            display_name="Stub",
            contract_version="v1",
            qualification_level="unit",
        )

    @property
    def descriptor(self) -> QualificationPluginDescriptor:
        return self._descriptor

    def execute(self) -> QualificationPluginResult:
        metrics = QualificationPluginMetrics(
            total_cases=0,
            matched_cases=0,
            mismatched_cases=0,
            false_positives=0,
            false_negatives=0,
            inconclusive_correct_cases=0,
            stage_matched_cases=0,
            stage_accuracy_percent=0.0,
            inconclusive_accuracy_percent=0.0,
            repeatability_pass=True,
            full_case_match_rate=0.0,
        )
        return QualificationPluginResult(
            plugin_id=self._descriptor.plugin_id,
            verdict=QualificationVerdict.PASS,
            metrics=metrics,
            gate_results=(),
            case_results=(),
            artifact_ref=None,
            blocked_reason=None,
            report_sections=(),
            analyzer_class="FunctionalDiagnosticAnalyzer",
            analyzer_module="intergrax.runtime.diagnostics.functional_diagnostic_analyzer",
        )


def test_registry_rejects_duplicate() -> None:
    registry = QualificationPluginRegistry()
    plugin = _StubPlugin(RAG_PLUGIN_ID)
    registry.register(plugin)
    with pytest.raises(QualificationPluginRegistryError, match="duplicate_plugin_id"):
        registry.register(plugin)


def test_registry_unknown_plugin() -> None:
    registry = QualificationPluginRegistry()
    with pytest.raises(QualificationPluginRegistryError, match="unknown_plugin_id"):
        registry.get(WEB_SEARCH_PLUGIN_ID)

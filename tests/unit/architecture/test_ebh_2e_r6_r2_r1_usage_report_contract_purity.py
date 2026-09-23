# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R2-R1 — usage aggregation contract purity & report DTO ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.llm_adapters.contracts import llm_usage_aggregation
from intergrax.llm_adapters.contracts import llm_usage_report as canonical_report
from intergrax.llm_adapters.contracts.llm_usage_aggregation import LLMUsageAggregator
from intergrax.llm_adapters.contracts.llm_usage_report import LLMUsageReport
from intergrax.llm_adapters.tracking import llm_usage_track as legacy_tracking

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FORBIDDEN_CONTRACT_PREFIXES = (
    "intergrax.llm_adapters.base",
    "intergrax.llm_adapters.tracking",
    "intergrax.runtime",
    "intergrax.applications",
    "intergrax.llm_adapters.registry",
)


def _import_from_modules(relative: str) -> list[str]:
    path = _REPO_ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_ebh_2e_r6_r2_r1_usage_aggregation_contract_does_not_import_tracking() -> None:
    for module in _import_from_modules("intergrax/llm_adapters/contracts/llm_usage_aggregation.py"):
        assert not module.startswith(_FORBIDDEN_CONTRACT_PREFIXES)


def test_ebh_2e_r6_r2_r1_usage_report_contract_does_not_import_implementation() -> None:
    for module in _import_from_modules("intergrax/llm_adapters/contracts/llm_usage_report.py"):
        assert not module.startswith(_FORBIDDEN_CONTRACT_PREFIXES)


def test_ebh_2e_r6_r2_r1_single_llm_usage_report_definition_in_tracking() -> None:
    path = _REPO_ROOT / "intergrax/llm_adapters/tracking/llm_usage_track.py"
    tree = ast.parse(path.read_text(encoding="utf-8"))
    report_classes = [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == "LLMUsageReport"
    ]
    assert report_classes == []


def test_ebh_2e_r6_r2_r1_legacy_tracking_reexports_canonical_report_dto() -> None:
    assert legacy_tracking.LLMUsageReport is canonical_report.LLMUsageReport
    assert legacy_tracking.LLMAdapterUsageEntry is canonical_report.LLMAdapterUsageEntry
    assert legacy_tracking.LLMAdapterMeta is canonical_report.LLMAdapterMeta


def test_ebh_2e_r6_r2_r1_external_aggregator_uses_only_canonical_contract_imports() -> None:
    class ExternalUsageAggregator:
        def register_adapter(self, trackable: object, label: str | None = None) -> None:
            del trackable, label

        def build_report(self) -> LLMUsageReport:
            from intergrax.llm_adapters.contracts.llm_usage_stats import LLMRunStats

            return LLMUsageReport(
                run_id="external",
                total=LLMRunStats(),
                entries=[],
                by_provider_model={},
                adapter_instance_ids={},
            )

    aggregator = ExternalUsageAggregator()
    assert isinstance(aggregator, LLMUsageAggregator)
    assert aggregator.build_report().run_id == "external"

    assert llm_usage_aggregation.__name__ == "intergrax.llm_adapters.contracts.llm_usage_aggregation"

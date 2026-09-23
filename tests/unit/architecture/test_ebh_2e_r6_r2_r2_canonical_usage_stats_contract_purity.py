# © Artur Czarnecki. All rights reserved.

"""EBH-2E-R6-R2-R2 — canonical usage stats & usage-source contract ownership gates."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.llm_adapters import base as llm_base
from intergrax.llm_adapters.base import usage_log as legacy_usage_log
from intergrax.llm_adapters.contracts import llm_usage_aggregation
from intergrax.llm_adapters.contracts import llm_usage_report as canonical_report
from intergrax.llm_adapters.contracts.llm_usage_aggregation import LLMUsageAggregator
from intergrax.llm_adapters.contracts.llm_usage_report import LLMUsageReport
from intergrax.llm_adapters.contracts.llm_usage_stats import (
    LLMRunStats,
    LLMRunStatsReader,
    LLMUsageTrackable,
)
from intergrax.llm_adapters.contracts import llm_usage_stats as canonical_stats_module
from intergrax.llm_adapters.contracts.llm_provider import LLMProvider

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_FORBIDDEN_CONTRACT_PREFIXES = (
    "intergrax.llm_adapters.base",
    "intergrax.llm_adapters.tracking",
    "intergrax.runtime",
    "intergrax.applications",
    "intergrax.llm_adapters.registry",
)

_CANONICAL_USAGE_CONTRACT_MODULES = (
    "intergrax/llm_adapters/contracts/llm_usage_stats.py",
    "intergrax/llm_adapters/contracts/llm_usage_report.py",
    "intergrax/llm_adapters/contracts/llm_usage_aggregation.py",
)

_CANONICAL_TYPE_NAMES = ("LLMRunStats", "LLMRunStatsReader", "LLMUsageTrackable")


def _import_from_modules(relative: str) -> list[str]:
    path = _REPO_ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _class_defs_named(relative: str, name: str) -> list[str]:
    path = _REPO_ROOT / relative
    tree = ast.parse(path.read_text(encoding="utf-8"))
    return [
        node.name
        for node in ast.walk(tree)
        if isinstance(node, ast.ClassDef) and node.name == name
    ]


@pytest.mark.parametrize("relative", _CANONICAL_USAGE_CONTRACT_MODULES)
def test_ebh_2e_r6_r2_r2_usage_contract_modules_forbid_implementation_imports(
    relative: str,
) -> None:
    for module in _import_from_modules(relative):
        assert not any(
            module == prefix or module.startswith(f"{prefix}.")
            for prefix in _FORBIDDEN_CONTRACT_PREFIXES
        )


def test_ebh_2e_r6_r2_r2_base_usage_log_does_not_define_canonical_types() -> None:
    relative = "intergrax/llm_adapters/base/usage_log.py"
    for type_name in _CANONICAL_TYPE_NAMES:
        assert _class_defs_named(relative, type_name) == []


def test_ebh_2e_r6_r2_r2_legacy_usage_log_reexports_canonical_types() -> None:
    assert legacy_usage_log.LLMRunStats is LLMRunStats
    assert legacy_usage_log.LLMRunStatsReader is LLMRunStatsReader
    assert legacy_usage_log.LLMUsageTrackable is LLMUsageTrackable
    assert llm_base.usage_log.LLMRunStats is LLMRunStats


class ExternalRunStatsReader:
    def get_run_stats(self, run_id: str | None = None) -> LLMRunStats | None:
        return LLMRunStats(calls=1, input_tokens=2, output_tokens=3, total_tokens=5)


class ExternalUsageSource:
    provider = LLMProvider.OPENAI
    model = "external-structural"
    usage = ExternalRunStatsReader()


def test_ebh_2e_r6_r2_r2_structural_external_usage_source() -> None:
    source = ExternalUsageSource()
    assert isinstance(source, LLMUsageTrackable)
    assert isinstance(source.usage, LLMRunStatsReader)
    stats = source.usage.get_run_stats()
    assert stats is not None
    assert stats.total_tokens == 5


def test_ebh_2e_r6_r2_r2_external_aggregator_uses_only_canonical_usage_contracts() -> None:
    class ExternalUsageAggregator:
        def register_adapter(
            self,
            trackable: LLMUsageTrackable,
            label: str | None = None,
        ) -> None:
            del trackable, label

        def build_report(self) -> LLMUsageReport:
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
    assert canonical_stats_module.__name__ == "intergrax.llm_adapters.contracts.llm_usage_stats"
    assert llm_usage_aggregation.__name__ == "intergrax.llm_adapters.contracts.llm_usage_aggregation"
    assert canonical_report.__name__ == "intergrax.llm_adapters.contracts.llm_usage_report"

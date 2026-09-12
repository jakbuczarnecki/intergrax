# © Artur Czarnecki. All rights reserved.

"""HARDENING-5 — self-healing core orchestration depends on plugin contracts, not in-memory adapters."""

from __future__ import annotations

import ast
from pathlib import Path

import pytest

from intergrax.contracts.self_healing.registry import SelfHealingStrategyRegistry
from intergrax.contracts.self_healing.strategy import SelfHealingStrategy
from intergrax.contracts.self_healing.quality_evaluation.assessor import StrategyQualityAssessor
from intergrax.runtime.self_healing.decision_engine import SelfHealingDecisionEngine
from intergrax.runtime.self_healing.quality_evaluation.service import StrategyQualityEvaluationService
from tests.unit.runtime.self_healing.test_autonomous_enterprise_self_healing_strategy_r1_q import (
    _sample_context,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_REPO_ROOT = Path(__file__).resolve().parents[4]

# See PLUGIN_ARCHITECTURE_HARDENING.md
_CORE_ORCHESTRATION_REL = (
    "intergrax/runtime/self_healing/decision_engine.py",
    "intergrax/runtime/self_healing/workflow/orchestrator.py",
    "intergrax/runtime/self_healing/lifecycle/rollback_coordinator.py",
    "intergrax/runtime/self_healing/lifecycle/validation_pipeline.py",
    "intergrax/runtime/self_healing/adaptive/engine.py",
    "intergrax/runtime/self_healing/knowledge_evolution/service.py",
    "intergrax/runtime/self_healing/strategy_recommendation/service.py",
    "intergrax/runtime/self_healing/autonomy/service.py",
    "intergrax/runtime/self_healing/autonomy/plugin_control_engine.py",
    "intergrax/runtime/self_healing/autonomy/plugin_decision_evaluator.py",
)

_FORBIDDEN_RUNTIME_IMPORT_FRAGMENTS = (
    "InMemorySelfHealing",
    "BasicStrategyQualityEvaluator",
    "WeightedStrategyQualityEvaluator",
    "QualityBasedStrategyRecommendationEngine",
    "BasicStrategyLearningEngine",
    "DefaultAutonomyPolicy",
    "DefaultAutonomyRiskEvaluator",
    "PlatformDefaultSelfHealingPlanBuilder",
)


def _collect_imported_modules(path: Path) -> list[str]:
    tree = ast.parse(path.read_text(encoding="utf-8"))
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def test_hardening_5_core_orchestration_avoids_concrete_plugin_imports() -> None:
    violations: list[str] = []
    for rel in _CORE_ORCHESTRATION_REL:
        path = _REPO_ROOT / rel
        for module in _collect_imported_modules(path):
            if not module.startswith("intergrax.runtime.self_healing"):
                continue
            for fragment in _FORBIDDEN_RUNTIME_IMPORT_FRAGMENTS:
                if fragment in module:
                    violations.append(f"{rel}: {module}")
    assert violations == [], "core → concrete plugin coupling:\n" + "\n".join(violations)


class _RecordingStrategyRegistry:
    def __init__(self, strategies: tuple[SelfHealingStrategy, ...]) -> None:
        self._strategies = strategies

    def register(self, strategy: SelfHealingStrategy) -> None:
        raise AssertionError("not used in this test")

    def resolve(self, strategy_id: str) -> SelfHealingStrategy | None:
        for strategy in self._strategies:
            if strategy.strategy_id == strategy_id:
                return strategy
        return None

    def list_available(self, *, tenant_id: str | None = None) -> tuple[SelfHealingStrategy, ...]:
        return self._strategies


def test_hardening_5_decision_engine_accepts_contract_registry() -> None:
    from intergrax.runtime.self_healing.defaults import platform_default_strategies

    strategies = platform_default_strategies()
    registry: SelfHealingStrategyRegistry = _RecordingStrategyRegistry(strategies)
    engine = SelfHealingDecisionEngine(registry)
    results = engine.select_and_evaluate(_sample_context())
    assert results


def test_hardening_5_quality_evaluation_service_satisfies_assessor_port() -> None:
    from intergrax.runtime.self_healing.quality_evaluation.basic_evaluator import BasicStrategyQualityEvaluator
    from intergrax.runtime.self_healing.performance_memory import InMemoryStrategyPerformanceMemoryRepository

    service = StrategyQualityEvaluationService(
        repository=InMemoryStrategyPerformanceMemoryRepository(),
        evaluator=BasicStrategyQualityEvaluator(),
    )
    assert isinstance(service, StrategyQualityAssessor)

# © Artur Czarnecki. All rights reserved.

"""ME-18 — final enterprise audit architecture and contract gates."""

from __future__ import annotations

import ast
import importlib
from pathlib import Path

import pytest

from intergrax.contracts.capability_catalog import (
    CapabilityCatalogSnapshotCacheObserver,
    CapabilityCatalogSource,
)
from intergrax.contracts.capability_catalog.governance import CapabilityGovernanceContext
from intergrax.contracts.marketplace import (
    MarketplaceDiagnosticObserver,
    MarketplaceListingProjection,
    MarketplaceMetadataSource,
)
from intergrax.contracts.marketplace.acquisition import (
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_REQUEST_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1,
    SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1,
    SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1,
)
from intergrax.contracts.marketplace.lifecycle_handoff_handler import (
    MarketplaceLifecycleHandoffHandler,
)

pytestmark = pytest.mark.unit

_FORBIDDEN_NEXUS = ("intergrax.nexus", "intergrax.runtime.nexus")
_FORBIDDEN_EXECUTION = ("intergrax.runtime.execution", "intergrax.runtime.nexus")
_FORBIDDEN_DOMAIN_REGISTRY = (
    "intergrax.tools.registry.runtime",
    "intergrax.skills.registry.runtime",
    "intergrax.agent_distribution",
)

_AUDIT_ROOTS = (
    "intergrax.marketplace",
    "intergrax.contracts.marketplace",
    "intergrax.capability_catalog",
)


def _package_py_files(module_name: str) -> list[Path]:
    package = importlib.import_module(module_name)
    assert package.__path__ is not None
    return sorted(path for path in Path(package.__path__[0]).rglob("*.py") if path.is_file())


def _is_handoff_adapter(path: Path, marketplace_root: Path) -> bool:
    adapter_root = marketplace_root / "handoff" / "adapters"
    return adapter_root in path.parents or path.parent == adapter_root


def _iter_imports(tree: ast.AST) -> list[str]:
    modules: list[str] = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            modules.extend(alias.name for alias in node.names)
        elif isinstance(node, ast.ImportFrom) and node.module:
            modules.append(node.module)
    return modules


def _assert_no_import_prefixes(
    *,
    module_name: str,
    forbidden: tuple[str, ...],
    skip_handoff_adapters: bool,
) -> None:
    marketplace_root = Path(importlib.import_module("intergrax.marketplace").__path__[0])
    for path in _package_py_files(module_name):
        if skip_handoff_adapters and module_name == "intergrax.marketplace":
            if _is_handoff_adapter(path, marketplace_root):
                continue
        tree = ast.parse(path.read_text(encoding="utf-8"))
        for imported in _iter_imports(tree):
            for prefix in forbidden:
                if imported == prefix or imported.startswith(f"{prefix}."):
                    raise AssertionError(f"{path.as_posix()} imports forbidden {imported}")


def test_me18_marketplace_has_no_nexus_dependency() -> None:
    _assert_no_import_prefixes(
        module_name="intergrax.marketplace",
        forbidden=_FORBIDDEN_NEXUS,
        skip_handoff_adapters=True,
    )


def test_me18_marketplace_has_no_execution_dependency() -> None:
    _assert_no_import_prefixes(
        module_name="intergrax.marketplace",
        forbidden=_FORBIDDEN_EXECUTION,
        skip_handoff_adapters=True,
    )


def test_me18_marketplace_has_no_domain_registry_implementation_dependency() -> None:
    _assert_no_import_prefixes(
        module_name="intergrax.marketplace",
        forbidden=_FORBIDDEN_DOMAIN_REGISTRY,
        skip_handoff_adapters=True,
    )


def test_me18_production_has_no_testing_support_imports() -> None:
    violations: list[str] = []
    for module_name in _AUDIT_ROOTS:
        for path in _package_py_files(module_name):
            tree = ast.parse(path.read_text(encoding="utf-8"))
            for imported in _iter_imports(tree):
                if imported == "testing_support" or imported.startswith("testing_support."):
                    violations.append(path.as_posix())
    assert violations == []


def test_me18_contract_spi_matrix_is_complete() -> None:
    """Public SPI families required for enterprise pluginability are importable contracts."""
    from intergrax.capability_catalog.governance import CapabilityGovernanceEvaluator
    from intergrax.capability_catalog.ranking import CapabilityRanker
    from intergrax.capability_catalog.recommendation import CapabilityRecommendationStrategy
    from intergrax.capability_catalog.search import CapabilitySearchStrategy
    from intergrax.capability_catalog.snapshot_cache_port import CapabilityCatalogSnapshotCache
    from intergrax.capability_catalog.snapshot_provider import CapabilityCatalogSnapshotProvider

    assert CapabilityCatalogSource is not None
    assert MarketplaceMetadataSource is not None
    assert MarketplaceListingProjection is not None
    assert CapabilitySearchStrategy is not None
    assert CapabilityRanker is not None
    assert CapabilityGovernanceEvaluator is not None
    assert CapabilityRecommendationStrategy is not None
    assert MarketplaceDiagnosticObserver is not None
    assert CapabilityCatalogSnapshotCache is not None
    assert CapabilityCatalogSnapshotCacheObserver is not None
    assert CapabilityCatalogSnapshotProvider is not None
    assert MarketplaceLifecycleHandoffHandler is not None
    assert CapabilityGovernanceContext is not None


def test_me18_public_schema_ids_are_unique() -> None:
    ids = (
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_REQUEST_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_RESPONSE_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_SELECTION_V1,
        SCHEMA_MACHINE_CAPABILITY_RECOMMENDATION_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_REQUEST_V1,
        SCHEMA_MACHINE_CAPABILITY_ACQUISITION_HANDOFF_RESPONSE_V1,
    )
    assert len(ids) == len(set(ids))


def test_me18_metadata_source_query_boundary_is_green() -> None:
    from tests.unit.marketplace import test_me17_c1_metadata_source_read_semantics as c1

    c1.test_me17_c1_marketplace_service_constructor_does_not_read_metadata_sources()
    c1.test_me17_c1_query_reads_each_metadata_source_once()
    c1.test_me17_c1_metadata_provider_failure_occurs_at_query_not_construction()


def test_me18_usage_commercial_billing_boundaries_hold() -> None:
    from tests.integration.marketplace import (
        test_me17_marketplace_production_qualification as me17,
    )

    me17.test_me17_usage_event_contains_no_pricing_or_settlement()
    me17.test_me17_skill_binding_is_not_usage()


def test_me18_skill_is_not_executable() -> None:
    from tests.integration.marketplace import (
        test_me15_marketplace_skill_composition_e2e as me15,
    )

    me15.test_me15_semantic_gate_no_skill_executor_symbols_in_me15_production_modules()

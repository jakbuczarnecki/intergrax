# © Artur Czarnecki. All rights reserved.

"""CONFIG-X-FINAL-R1-R1-R1 — named-constant semantic default resolution gates."""

from __future__ import annotations

import pytest

from tests.qualification.config_x._config_x_semantic_production_scan import (
    SemanticProductionFindingClass,
    classify_semantic_defaults_in_module_source,
    discover_named_constant_semantic_blind_spot_paths,
    discover_sanctioned_vector_store_localhost_transport_paths,
    frz_cfg_05_named_constant_blind_spot_count,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate, pytest.mark.qualification]

_QDRANT_CONFIG = "intergrax/integrations/providers/vector_store/qdrant/config.py"
_CHROMA_CONFIG = "intergrax/integrations/providers/vector_store/chroma/config.py"


def test_named_constant_localhost_resolves_to_sanctioned_on_vector_store_surface() -> None:
    source = '''
DEFAULT_HOST = "localhost"

class QdrantIntegrationConfig:
    host: str = DEFAULT_HOST
'''
    findings = classify_semantic_defaults_in_module_source(source, _QDRANT_CONFIG)
    classes = {f.finding_class for f in findings}
    assert SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT in classes
    assert SemanticProductionFindingClass.NAMED_CONSTANT_SEMANTIC_BLIND_SPOT not in classes
    assert (
        SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION not in classes
    )


def test_named_constant_localhost_unsanctioned_surface_is_hard_coded_selection() -> None:
    source = '''
DEFAULT_HOST = "localhost"

class ExampleConfig:
    host: str = DEFAULT_HOST
'''
    rel = "intergrax/integrations/providers/example/config.py"
    findings = classify_semantic_defaults_in_module_source(source, rel)
    classes = {f.finding_class for f in findings}
    assert SemanticProductionFindingClass.HARD_CODED_SEMANTIC_PRODUCTION_SELECTION in classes


def test_unresolved_named_constant_semantic_default_is_blind_spot() -> None:
    source = '''
class ExampleConfig:
    host: str = UNKNOWN_HOST_CONSTANT
'''
    rel = "intergrax/integrations/providers/example/config.py"
    findings = classify_semantic_defaults_in_module_source(source, rel)
    assert any(
        f.finding_class is SemanticProductionFindingClass.NAMED_CONSTANT_SEMANTIC_BLIND_SPOT
        for f in findings
    )


def test_from_env_get_fallback_resolves_module_constant() -> None:
    source = '''
import os

DEFAULT_HOST = "localhost"
ENV_HOST = "INTERGRAX_EXAMPLE_HOST"

class QdrantIntegrationConfig:
    @classmethod
    def from_env(cls):
        host = os.environ.get(ENV_HOST, DEFAULT_HOST).strip() or DEFAULT_HOST
        return host
'''
    findings = classify_semantic_defaults_in_module_source(source, _QDRANT_CONFIG)
    assert any(
        f.finding_class is SemanticProductionFindingClass.SANCTIONED_DEPLOYMENT_DEFAULT
        for f in findings
    )


def test_frz_cfg_05_repo_has_zero_named_constant_blind_spots() -> None:
    assert frz_cfg_05_named_constant_blind_spot_count() == 0
    assert discover_named_constant_semantic_blind_spot_paths() == frozenset()


def test_qdrant_and_chroma_localhost_transport_defaults_explicitly_sanctioned() -> None:
    sanctioned = discover_sanctioned_vector_store_localhost_transport_paths()
    assert _QDRANT_CONFIG in sanctioned
    assert _CHROMA_CONFIG in sanctioned

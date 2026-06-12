# © Artur Czarnecki. All rights reserved.

"""APP-OPS-3 — EnvironmentHealthScore wiring."""

from __future__ import annotations

from pathlib import Path

import pytest

from intergrax.applications._shared.health_score_wiring import (
    build_application_health_score,
    compute_environment_health_score,
)
from intergrax.applications._shared.product_manifest_registry import iter_strict_product_manifests
from intergrax.applications.contracts.environment_health_score import (
    PRODUCTION_READY_THRESHOLD,
    HealthDimension,
)

pytestmark = [pytest.mark.unit, pytest.mark.gate]

REPO_ROOT = Path(__file__).resolve().parents[3]


def test_strict_product_health_score_is_production_ready() -> None:
    product_id, manifest = next(iter(iter_strict_product_manifests()))
    score = compute_environment_health_score(product_id, manifest, repo_root=REPO_ROOT)
    assert score.overall >= PRODUCTION_READY_THRESHOLD
    assert score.snapshot_id
    assert {item.dimension for item in score.dimensions} == set(HealthDimension)
    assert not score.blockers, score.blockers


def test_application_health_rollup() -> None:
    product_id, manifest = next(iter(iter_strict_product_manifests()))
    rollup = build_application_health_score(product_id, manifest, repo_root=REPO_ROOT)
    assert rollup.production_ready
    assert rollup.environments[0].app_id == manifest.app_id

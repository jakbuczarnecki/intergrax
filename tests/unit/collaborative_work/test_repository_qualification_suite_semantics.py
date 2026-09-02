# © Artur Czarnecki. All rights reserved.

"""Collaborative Work repository qualification suite semantic vs infrastructure semantics."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest

from intergrax.collaborative_work.repository_qualification_suite import (
    collaborative_work_sqlite_repository_qualification_binding,
    collaborative_work_sqlite_repository_qualification_suite,
)
from intergrax.collaborative_work.persistence import CollaborativeWorkRepositories
from intergrax.core.qualification.execution import ProviderQualificationSuiteInfrastructureError
from intergrax.core.qualification.status import QualificationStatus
from intergrax.integrations.providers.relational_store.sqlite.register import register_sqlite_integration
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import SQLITE
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = pytest.mark.unit


@pytest.fixture(autouse=True)
def _sqlite_catalog() -> None:
    clear_catalog()
    register_sqlite_integration()
    yield
    clear_catalog()


def test_semantic_repository_failure_maps_to_rejected(tmp_path: Path) -> None:
    suite = collaborative_work_sqlite_repository_qualification_suite()
    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(tmp_path / "sqlite-data")}},
    )
    bundle = collaborative_work_sqlite_repository_qualification_binding().materialize(
        profile,
        resolved_provider_id="sqlite",
    )[0]
    assert isinstance(bundle, CollaborativeWorkRepositories)

    with patch(
        "intergrax.collaborative_work.repository_qualification_suite._run_repository_contract_checks",
        return_value=(4, 2),
    ):
        outcome = suite.execute(bundle)
    bundle.close()
    assert outcome.status is QualificationStatus.REJECTED


def test_suite_infrastructure_failure_is_not_semantic_rejection(tmp_path: Path) -> None:
    suite = collaborative_work_sqlite_repository_qualification_suite()
    with pytest.raises(ProviderQualificationSuiteInfrastructureError):
        suite.execute(object())

    profile = IntegrationProfile(
        relational_store=SQLITE,
        options={SQLITE: {"data_dir": str(tmp_path / "sqlite-data")}},
    )
    bundle = collaborative_work_sqlite_repository_qualification_binding().materialize(
        profile,
        resolved_provider_id="sqlite",
    )[0]
    assert isinstance(bundle, CollaborativeWorkRepositories)

    with patch(
        "intergrax.collaborative_work.repository_qualification_suite._run_repository_contract_checks",
        side_effect=RuntimeError("database host unavailable"),
    ):
        with pytest.raises(RuntimeError, match="database host unavailable"):
            suite.execute(bundle)
    bundle.close()

# © Artur Czarnecki. All rights reserved.

"""Real PostgreSQL provider qualification execution proof (PROVIDER-QUAL-7)."""

from __future__ import annotations

import os
import uuid
from collections.abc import Generator

import pytest

from intergrax.collaborative_work.persistence import open_postgresql_collaborative_work_repositories
from intergrax.collaborative_work.repository_qualification_suite import (
    CW_POSTGRESQL_REPOSITORY_SUITE_ID,
    CW_REPOSITORY_SUITE_VERSION,
    COLLABORATIVE_WORK_DOMAIN,
    COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
    PostgreSQLQualificationMaterializationOptions,
    collaborative_work_postgresql_repository_qualification_binding,
)
from intergrax.core.qualification import (
    ProviderQualificationExecutor,
    QualificationStatus,
    execute_provider_qualification,
    new_qualification_run_id,
)
from intergrax.core.qualification.execution import (
    ProviderQualificationExecutionDependencies,
    ProviderQualificationExecutionRequest,
    ProviderQualificationMaterializationError,
)
from intergrax.core.qualification.persistence import DocumentStoreProviderQualificationPersistence
from intergrax.core.qualification.provider import ProviderQualificationSubject
from intergrax.integrations._shared.in_memory_document_store import InMemoryDocumentStore
from intergrax.integrations.contracts.base import IntegrationConfigurationError
from intergrax.integrations.providers.relational_store.postgresql.config import (
    PostgreSQLIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.postgresql.register import (
    register_postgresql_integration,
)
from intergrax.integrations.registry.catalog import clear_catalog
from intergrax.integrations.registry.catalog_manifests import POSTGRESQL
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.integration, pytest.mark.network]

DSN_ENV = "INTERGRAX_COLLABORATIVE_WORK_POSTGRESQL_DSN"
_SCHEMA_PREFIX = "provider_qual_pg_"
_SOURCE_REVISION = "provider-qual-7-postgresql-proof"


def _resolve_config() -> PostgreSQLIntegrationConfig | None:
    dsn = os.environ.get(DSN_ENV, "").strip()
    if dsn:
        return PostgreSQLIntegrationConfig(dsn=dsn)
    base = PostgreSQLIntegrationConfig.from_env()
    if not base.connection_string():
        return None
    return base


def _drop_schema(schema_name: str) -> None:
    config = _resolve_config()
    if config is None:
        return
    try:
        bundle = open_postgresql_collaborative_work_repositories(
            config=config,
            schema_name="public",
        )
    except IntegrationConfigurationError:
        return
    try:
        with bundle.store.transaction() as conn:
            conn.execute(f"DROP SCHEMA IF EXISTS {schema_name} CASCADE")
    finally:
        bundle.close()


@pytest.fixture(autouse=True)
def _postgresql_catalog() -> Generator[None, None, None]:
    clear_catalog()
    register_postgresql_integration()
    yield
    clear_catalog()


def test_real_postgresql_provider_qualification_execution_and_persistence() -> None:
    config = _resolve_config()
    if config is None:
        pytest.skip(
            f"PostgreSQL qualification requires {DSN_ENV} or INTERGRAX_POSTGRESQL_* settings",
        )

    schema_name = f"{_SCHEMA_PREFIX}{uuid.uuid4().hex}"
    profile = IntegrationProfile(
        relational_store=POSTGRESQL,
        options={POSTGRESQL: {"dsn": config.connection_string()}},
    )
    run_id = new_qualification_run_id()
    subject = ProviderQualificationSubject(
        provider_id="postgresql",
        provider_version="16.6",
        capability_id=COLLABORATIVE_WORK_PERSISTENCE_CAPABILITY,
        domain=COLLABORATIVE_WORK_DOMAIN,
        intergrax_revision=_SOURCE_REVISION,
        qualification_suite_id=CW_POSTGRESQL_REPOSITORY_SUITE_ID,
        qualification_suite_version=CW_REPOSITORY_SUITE_VERSION,
        environment_id="local-docker-qual-host",
        adapter_identity="intergrax.integrations.providers.relational_store.postgresql",
    )
    store = InMemoryDocumentStore()
    persistence = DocumentStoreProviderQualificationPersistence(store)
    binding = collaborative_work_postgresql_repository_qualification_binding(
        materialization_options=PostgreSQLQualificationMaterializationOptions(
            schema_name=schema_name,
        ),
    )

    try:
        run = execute_provider_qualification(
            ProviderQualificationExecutionRequest(
                subject=subject,
                executor=ProviderQualificationExecutor(
                    executor_kind="integration_test_runner",
                    executor_id="provider-qual-7-postgresql",
                    executor_version="2026.09.02",
                ),
                source_revision=_SOURCE_REVISION,
                integration_profile=profile,
                qualification_run_id=run_id,
            ),
            ProviderQualificationExecutionDependencies(
                persistence=persistence,
                domain_binding=binding,
            ),
        )
    except (
        IntegrationConfigurationError,
        ProviderQualificationMaterializationError,
        ConnectionError,
        TimeoutError,
        OSError,
    ) as exc:
        pytest.skip(f"PostgreSQL backend unavailable: {type(exc).__name__}: {exc}")
    else:
        assert run.qualification_run_id == run_id
        assert run.status is QualificationStatus.PRODUCTION_QUALIFIED
        assert run.environment_metadata.real_backend is True
        assert run.environment_metadata.mocks is False
        assert run.environment_metadata.sqlite_substitution is False
        assert run.subject == subject

        loaded = persistence.get_by_qualification_run_id(run_id)
        assert loaded == run
        assert loaded is not None
        assert loaded.evidence == run.evidence
    finally:
        _drop_schema(schema_name)

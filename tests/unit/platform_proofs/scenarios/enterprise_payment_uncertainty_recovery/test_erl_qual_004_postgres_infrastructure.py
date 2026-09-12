"""Architecture tests for ERL-QUAL-004 PostgreSQL Docker infrastructure foundation."""

from __future__ import annotations

from pathlib import Path

import pytest

from platform_proofs.scenarios.enterprise_payment_uncertainty_recovery.infrastructure.contract import (
    COMPOSE_SERVICE_NAME,
    DOCKER_COMPOSE_FILE,
    PINNED_POSTGRES_IMAGE,
    POSTGRES_ENV_EXAMPLE,
    REQUIRED_POSTGRES_ENV_KEYS,
    parse_env_file,
    validate_postgres_env_contract,
)

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[5]
_SCENARIO_ROOT = _REPO_ROOT / "platform_proofs/scenarios/enterprise_payment_uncertainty_recovery"


def test_compose_file_exists_and_pins_postgres_image() -> None:
    assert DOCKER_COMPOSE_FILE.is_file()
    compose = DOCKER_COMPOSE_FILE.read_text(encoding="utf-8")
    assert PINNED_POSTGRES_IMAGE in compose
    assert ":latest" not in compose
    assert "env_file:" in compose
    assert "../config/postgres.env" in compose


def test_compose_defines_postgres_service_with_healthcheck_and_volume() -> None:
    compose = DOCKER_COMPOSE_FILE.read_text(encoding="utf-8").lower()
    assert f"{COMPOSE_SERVICE_NAME}:" in compose
    assert "healthcheck:" in compose
    assert "pg_isready" in compose
    assert "127.0.0.1:${erl_qual_004_postgres_host_port}:5432" in compose
    assert "erl_qual_004_postgres_data" in compose


def test_postgres_env_example_satisfies_contract() -> None:
    values = parse_env_file(POSTGRES_ENV_EXAMPLE)
    assert REQUIRED_POSTGRES_ENV_KEYS <= values.keys()
    assert not validate_postgres_env_contract(values)


def test_compose_does_not_embed_literal_postgres_credentials() -> None:
    compose = DOCKER_COMPOSE_FILE.read_text(encoding="utf-8")
    for forbidden in ("POSTGRES_PASSWORD:", "POSTGRES_USER: erl", "change-me"):
        assert forbidden not in compose


def test_infrastructure_documentation_present() -> None:
    doc = _SCENARIO_ROOT / "docs" / "ERL_QUAL_004_POSTGRESQL_INFRASTRUCTURE.md"
    readme = _SCENARIO_ROOT / "infrastructure" / "README.md"
    assert doc.is_file()
    assert readme.is_file()
    assert PINNED_POSTGRES_IMAGE in doc.read_text(encoding="utf-8")

# © Artur Czarnecki. All rights reserved.

"""Harness-host meaningful side-effect authorization composition (P2D-R1)."""

from __future__ import annotations

from pathlib import Path

import pytest

from governed_contractor_application.host.environment_profile import (
    build_governed_contractor_environment_profile,
)
from governed_contractor_application.host.settings import GovernedContractorBackendSettings
from governed_contractor_application.manifest import build_governed_contractor_manifest
from intergrax.applications._shared.harness_meaningful_side_effect_authorization_wiring import (
    build_harness_host_meaningful_side_effect_authorization_port,
    resolve_harness_host_meaningful_side_effect_authorization_port,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.contracts.meaningful_side_effect_authorization import (
    MeaningfulSideEffectAuthorizationPort,
)

pytestmark = pytest.mark.unit


class _RecordingMsePort:
    def authorize(self, request: object, **_: object) -> object:
        del request
        raise AssertionError("custom port must not authorize in this proof")


def _strict_product_env() -> object:
    settings = GovernedContractorBackendSettings.from_env()
    manifest = build_governed_contractor_manifest()
    return manifest.environment or build_governed_contractor_environment_profile(settings)


def test_resolve_returns_none_for_non_strict_host() -> None:
    env = _strict_product_env().model_copy(
        update={
            "meta": _strict_product_env().meta.model_copy(
                update={"execution_mode": ExecutionMode.BALANCED},
            ),
        },
    )
    assert (
        resolve_harness_host_meaningful_side_effect_authorization_port(
            env,
            collaborative_work_sqlite_path=Path("ignored.db"),
        )
        is None
    )


def test_strict_default_materializes_non_none_port_with_sqlite_path(tmp_path: Path) -> None:
    env = _strict_product_env()
    sqlite_path = tmp_path / "collaborative_work.db"
    port = build_harness_host_meaningful_side_effect_authorization_port(
        env,
        collaborative_work_sqlite_path=sqlite_path,
    )
    assert port is not None
    assert isinstance(port, MeaningfulSideEffectAuthorizationPort)


def test_explicit_custom_port_is_used_as_is() -> None:
    env = _strict_product_env()
    custom = _RecordingMsePort()
    resolved = resolve_harness_host_meaningful_side_effect_authorization_port(
        env,
        explicit=custom,
        collaborative_work_sqlite_path=Path("would-build-default.db"),
    )
    assert resolved is custom

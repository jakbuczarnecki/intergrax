# © Artur Czarnecki. All rights reserved.

"""PDC-D4-R015 (TL-FIX-C): no product vocabulary in the generic HostDeploymentProfile.

LKW (Local Knowledge Workspace) deployment keys belong to the product side
(``lkw_hybrid_daemon_wiring.py``). The generic profile must stay product-neutral, and
existing callers that still pass the legacy flat keys keep working through the product
wiring.
"""

from __future__ import annotations

import inspect
import json
from pathlib import Path

import pytest
from intergrax.applications._shared.lkw_hybrid_daemon_wiring import (
    LkwHybridDaemonDeployment,
    resolve_lkw_hybrid_daemon_wiring,
)
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    HostDeploymentProfile,
    sub_profiles,
)
from pydantic import ValidationError

pytestmark = [pytest.mark.unit]

# Product names that must never appear in the generic deployment profile contract.
PRODUCT_VOCABULARY = ("lkw", "daemon", "local_workspace", "local knowledge", "knowledge_workspace")


def _deployment(**keys: bool | int | str | dict[str, str]) -> HostDeploymentProfile:
    """Construct the profile the way legacy callers do (flat keyword entries)."""
    return HostDeploymentProfile.model_validate(keys)


def _violations(text: str) -> list[str]:
    lowered = text.lower()
    return [word for word in PRODUCT_VOCABULARY if word in lowered]


def test_r015_generic_profile_declares_no_product_fields() -> None:
    assert sorted(HostDeploymentProfile.model_fields) == ["business_agents_deploy_enabled"]
    for name in HostDeploymentProfile.model_fields:
        assert _violations(name) == [], name


def test_r015_generic_profile_contract_carries_no_product_vocabulary() -> None:
    """Fails if product vocabulary reappears in the generic profile source, schema or defaults."""
    source = inspect.getsource(HostDeploymentProfile)
    assert _violations(source) == []
    assert _violations(json.dumps(HostDeploymentProfile.model_json_schema())) == []
    assert _violations(json.dumps(HostDeploymentProfile().model_dump())) == []
    default_env = ApplicationEnvironmentProfile.product_defaults(profile_id="generic.app")
    assert _violations(json.dumps(default_env.host_deployment_profile.model_dump())) == []


def test_r015_sub_profiles_module_is_product_neutral_for_deployment() -> None:
    module_source = inspect.getsource(sub_profiles)
    assert _violations(module_source) == []


def test_r015_vocabulary_guard_would_catch_a_reintroduced_field() -> None:
    """The guard itself: a class that re-adds a product field is flagged."""

    class _Regressed(HostDeploymentProfile):
        lkw_hybrid_daemon_enabled: bool = False

    assert "lkw_hybrid_daemon_enabled" in _Regressed.model_fields
    assert sorted(_Regressed.model_fields) != ["business_agents_deploy_enabled"]
    assert _violations(" ".join(_Regressed.model_fields)) == ["lkw", "daemon"]


def test_r015_lkw_keys_live_on_the_product_side() -> None:
    assert set(LkwHybridDaemonDeployment.model_fields) == {
        "lkw_hybrid_daemon_enabled",
        "lkw_daemon_bind_host",
        "lkw_daemon_port",
    }
    defaults = LkwHybridDaemonDeployment()
    assert defaults.lkw_hybrid_daemon_enabled is False
    assert defaults.lkw_daemon_bind_host == "127.0.0.1"
    assert defaults.lkw_daemon_port == 8020


def test_r015_legacy_flat_keys_still_construct_and_resolve_through_the_product_wiring(
    tmp_path: Path,
) -> None:
    # The wiring probes <repo_root>/scripts/maintenance/lkw-host.py; use a stub root so the assertion does
    # not depend on the launcher location in the checkout.
    (tmp_path / "scripts" / "maintenance").mkdir(parents=True)
    (tmp_path / "scripts" / "maintenance" / "lkw-host.py").write_text("# stub launcher", encoding="utf-8")
    legacy = _deployment(
        lkw_hybrid_daemon_enabled=True,
        lkw_daemon_bind_host="127.0.0.1",
        lkw_daemon_port=8021,
    )
    env = ApplicationEnvironmentProfile.product_defaults(profile_id="local_workspace.product").model_copy(
        update={"host_deployment_profile": legacy}
    )
    wiring = resolve_lkw_hybrid_daemon_wiring(env, repo_root=tmp_path)
    assert wiring.enabled is True
    assert wiring.spec is not None and wiring.spec.port == 8021
    assert wiring.launcher_path is not None

    disabled = resolve_lkw_hybrid_daemon_wiring(
        ApplicationEnvironmentProfile.product_defaults(profile_id="generic.app"),
        repo_root=tmp_path,
    )
    assert disabled.enabled is False and disabled.spec is None


def test_r015_legacy_keys_round_trip_through_the_profile_payload() -> None:
    legacy = _deployment(lkw_hybrid_daemon_enabled=True, lkw_daemon_port=8030)
    restored = HostDeploymentProfile.model_validate(legacy.model_dump())
    assert restored == legacy
    assert LkwHybridDaemonDeployment.from_host_deployment(restored).lkw_daemon_port == 8030


def test_r015_product_wiring_fails_closed_on_invalid_or_unknown_product_keys() -> None:
    with pytest.raises(ValidationError):
        LkwHybridDaemonDeployment.from_host_deployment(
            _deployment(lkw_hybrid_daemon_enabled=True, lkw_daemon_port=70000)
        )
    with pytest.raises(ValidationError):
        LkwHybridDaemonDeployment.from_host_deployment(
            _deployment(lkw_hybrid_daemon_enabld=True)  # typo in a product-owned key
        )


def test_r015_generic_profile_accepts_only_flat_scalar_product_keys() -> None:
    with pytest.raises(ValidationError):
        _deployment(product_owned_blob={"nested": "value"})
    assert _deployment(other_product_flag=True).model_extra == {"other_product_flag": True}

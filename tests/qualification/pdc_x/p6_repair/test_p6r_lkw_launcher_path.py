# © Artur Czarnecki. All rights reserved.

"""B-P6-R item 4 - the LKW hybrid daemon wiring probes the real launcher location."""

from __future__ import annotations

from pathlib import Path

import pytest
from intergrax.applications._shared.lkw_hybrid_daemon import LkwHybridDaemonSpec
from intergrax.applications._shared.lkw_hybrid_daemon_wiring import resolve_lkw_hybrid_daemon_wiring
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    HostDeploymentProfile,
)

pytestmark = pytest.mark.unit

REPO_ROOT = Path(__file__).resolve().parents[4]


def _enabled_env() -> ApplicationEnvironmentProfile:
    return ApplicationEnvironmentProfile.product_defaults(profile_id="local_workspace.product").model_copy(
        update={"host_deployment_profile": HostDeploymentProfile.model_validate({"lkw_hybrid_daemon_enabled": True})}
    )


def test_the_launcher_lives_under_scripts_maintenance() -> None:
    assert (REPO_ROOT / "scripts" / "maintenance" / "lkw-host.py").is_file()
    assert not (REPO_ROOT / "scripts" / "lkw-host.py").exists()


def test_the_resolver_finds_the_real_launcher() -> None:
    wiring = resolve_lkw_hybrid_daemon_wiring(_enabled_env(), repo_root=REPO_ROOT)
    assert wiring.enabled is True
    assert wiring.spec is not None
    assert wiring.launcher_path == REPO_ROOT / "scripts" / "maintenance" / "lkw-host.py"


def test_the_probed_path_matches_the_launcher_the_spec_declares(tmp_path: Path) -> None:
    declared = LkwHybridDaemonSpec.model_fields["launcher_script"].default
    launcher = tmp_path / Path(str(declared))
    launcher.parent.mkdir(parents=True)
    launcher.write_text("", encoding="utf-8")
    wiring = resolve_lkw_hybrid_daemon_wiring(_enabled_env(), repo_root=tmp_path)
    assert wiring.enabled is True
    assert wiring.launcher_path == launcher


def test_the_old_location_alone_is_not_a_launcher(tmp_path: Path) -> None:
    (tmp_path / "scripts").mkdir()
    (tmp_path / "scripts" / "lkw-host.py").write_text("", encoding="utf-8")
    wiring = resolve_lkw_hybrid_daemon_wiring(_enabled_env(), repo_root=tmp_path)
    assert wiring.enabled is False
    assert wiring.launcher_path is None


def test_a_disabled_deployment_does_not_probe() -> None:
    wiring = resolve_lkw_hybrid_daemon_wiring(
        ApplicationEnvironmentProfile.product_defaults(profile_id="local_workspace.product"),
        repo_root=REPO_ROOT,
    )
    assert wiring.enabled is False
    assert wiring.launcher_path is None

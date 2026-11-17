# © Artur Czarnecki. All rights reserved.

"""W-B Part E — package-prefix allowlist for the importlib loaders (NT-WB-18..24, 30)."""

from __future__ import annotations

import importlib
from pathlib import Path

import pytest
from intergrax.agents.readiness import scoreboard
from intergrax.applications._shared import agent_resolution, mcp_import_guard, migration_wiring, wiring
from intergrax.applications._shared.import_allowlist import (
    ImportAllowlistRequiredError,
    ImportPathNotAllowedError,
    ImportPrefixAllowlist,
)
from intergrax.applications.contracts.errors import AgentImportError

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_AGENTS_ROOT = _REPO_ROOT / "agents"


def _registered_tier2_packages() -> tuple[str, ...]:
    """Registered source of Tier-2 packages: the top-level packages of the repository ``agents/`` tree."""
    return tuple(
        sorted(entry.name for entry in _AGENTS_ROOT.iterdir() if entry.is_dir() and (entry / "__init__.py").is_file()),
    )


def _forbid_import(monkeypatch: pytest.MonkeyPatch, module: object) -> list[str]:
    calls: list[str] = []

    def _recording(name: str, package: str | None = None) -> object:
        calls.append(name)
        raise AssertionError(f"import_module({name!r}) must not be reached")

    monkeypatch.setattr(module, "importlib", type("_Importlib", (), {"import_module": staticmethod(_recording)}))
    return calls


def test_nt_wb_18_load_callable_outside_allowlist_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _forbid_import(monkeypatch, wiring)
    allowlist = ImportPrefixAllowlist(prefixes=("intergrax", "agents"))
    with pytest.raises(ImportPathNotAllowedError):
        wiring.load_callable("os.system", allowed_prefixes=allowlist, fail_closed=True)
    assert calls == []


def test_nt_wb_19_fail_closed_without_allowlist_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _forbid_import(monkeypatch, wiring)
    with pytest.raises(ImportAllowlistRequiredError):
        wiring.load_callable("echo.echo_agent.EchoAgent", fail_closed=True)
    assert calls == []


def test_nt_wb_20_resolve_agent_type_outside_allowlist_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    allowlist = ImportPrefixAllowlist(prefixes=_registered_tier2_packages())
    assert "echo" in allowlist.prefixes
    assert "zz_synth_t2" not in allowlist.prefixes

    imported: list[str] = []
    real_import = importlib.import_module

    def _recording(name: str, package: str | None = None) -> object:
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(agent_resolution.importlib, "import_module", _recording)

    for refused in ("zz_synth_t2.agent.Agent", "subprocess.Popen"):
        with pytest.raises(AgentImportError):
            agent_resolution.resolve_agent_type(
                agent_type=None,
                import_path=refused,
                allowed_prefixes=allowlist,
                fail_closed=True,
            )
    assert imported == []

    resolved = agent_resolution.resolve_agent_type(
        agent_type=None,
        import_path="echo.echo_agent.EchoAgent",
        allowed_prefixes=allowlist,
        fail_closed=True,
    )
    assert resolved.__name__ == "EchoAgent"


def test_nt_wb_21_migration_manifest_outside_allowlist_refused(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    imported: list[str] = []

    def _recording(name: str, package: str | None = None) -> object:
        imported.append(name)
        raise ModuleNotFoundError(name)

    monkeypatch.setattr(importlib, "import_module", _recording)
    allowlist = ImportPrefixAllowlist(prefixes=("local_workspace_application",))

    for refused in ("os", "a.b", "zz_synth_app"):
        assert (
            migration_wiring._load_manifest_for_package(
                refused,
                allowed_prefixes=allowlist,
                fail_closed=True,
            )
            is None
        )
    assert imported == []

    applications_root = tmp_path / "applications"
    migrations = applications_root / "zz_synth_app_application" / "migrations"
    migrations.mkdir(parents=True)
    (migrations / "0001.json").write_text("{}", encoding="utf-8")
    violations = migration_wiring.check_application_migrations(
        applications_root,
        allowed_prefixes=allowlist,
        fail_closed=True,
    )
    assert imported == []
    assert any("migrations present but manifest could not be loaded" in item for item in violations) or any(
        "invalid migration document" in item for item in violations
    )


def test_nt_wb_22_mcp_builder_outside_allowlist_refused(monkeypatch: pytest.MonkeyPatch) -> None:
    imported: list[str] = []

    def _recording(name: str, package: str | None = None) -> object:
        imported.append(name)
        raise AssertionError(name)

    monkeypatch.setattr(importlib, "import_module", _recording)
    allowlist = ImportPrefixAllowlist(prefixes=("intergrax", "local_workspace_application"))
    with pytest.raises(ImportPathNotAllowedError):
        mcp_import_guard.load_mcp_server_builder(
            "os",
            symbol="system",
            allowed_prefixes=allowlist,
            fail_closed=True,
        )
    assert imported == []


def test_nt_wb_23_prefix_boundary() -> None:
    allowlist = ImportPrefixAllowlist(prefixes=("intergrax",))
    assert allowlist.permits("intergrax")
    assert allowlist.permits("intergrax.x")
    assert not allowlist.permits("intergrax_evil.x")
    assert not allowlist.permits("..x")
    assert not allowlist.permits(".intergrax.x")
    assert not allowlist.permits("")
    assert not allowlist.permits("intergrax..x")
    with pytest.raises(ValueError):
        ImportPrefixAllowlist(prefixes=("intergrax.",))
    with pytest.raises(ValueError):
        ImportPrefixAllowlist(prefixes=("",))


def test_nt_wb_24_lab_unrestricted_control() -> None:
    resolved = agent_resolution.resolve_agent_type(
        agent_type=None,
        import_path="echo.echo_agent.EchoAgent",
    )
    assert resolved.__name__ == "EchoAgent"
    loaded = wiring.load_callable("echo.echo_agent.EchoAgent")
    assert loaded.__name__ == "EchoAgent"


def test_nt_wb_30_scoreboard_loader_outside_agents_refused(monkeypatch: pytest.MonkeyPatch, tmp_path: Path) -> None:
    imported: list[str] = []
    real_import = importlib.import_module

    def _recording(name: str, package: str | None = None) -> object:
        imported.append(name)
        return real_import(name, package)

    monkeypatch.setattr(scoreboard.importlib, "import_module", _recording)
    inventory = {"agents": [{"agent_id": "echo", "agent_module": "agents.echo.echo_agent"}]}
    agent_py = _AGENTS_ROOT / "echo" / "echo_agent.py"

    for refused in ("os", "subprocess", "agents_evil.x", "agents.zz_synth_t2.agent"):
        with pytest.raises(RuntimeError):
            scoreboard._load_agent_instance(refused, agent_py, inventory=inventory)
    assert imported == []

    scoreboard._load_agent_instance("agents.echo.echo_agent", agent_py, inventory=inventory)
    assert imported == ["echo.echo_agent"]

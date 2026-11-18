# © Artur Czarnecki. All rights reserved.

"""T-IGX-051-I — unit tests of ``import_policy`` (UT-A7C-01..05, plan section 8).

The permitted import sets are derived from the product manifest registry at call time; the
guard posture is decided only by ``wiring_fails_closed``.
"""

from __future__ import annotations

import importlib
from pathlib import Path
from types import ModuleType

import pytest
from intergrax.applications._shared import import_policy, product_manifest_registry
from intergrax.applications._shared.agent_resolution import (
    resolve_agent_contract_from_binding,
    resolve_agent_type_from_binding,
)
from intergrax.applications._shared.import_allowlist import ImportPrefixAllowlist
from intergrax.applications._shared.import_policy import (
    UNRESTRICTED,
    ImportPolicy,
    guarded_import_policy,
    import_policy_for_environment,
    import_policy_for_manifest,
    registered_agent_prefixes,
    registered_application_prefixes,
    registered_factory_prefixes,
)
from intergrax.applications._shared.product_manifest_registry import (
    iter_product_manifests,
    register_product_manifest,
)
from intergrax.applications._shared.roster_agent_contract_authority import resolve_roster_agent_contract
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile
from intergrax.applications.contracts.errors import AgentImportError
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.applications.contracts.manifest import AgentBinding, ApplicationManifest

from legal.legal_agent import LegalAgent

pytestmark = [pytest.mark.unit]

_REPO_ROOT = Path(__file__).resolve().parents[3]
_SYNTH_AGENT_PATH = "zz_synth_t2.agent.Agent"


def _top_level(path: str | None) -> set[str]:
    return set() if path is None else {path.split(".", 1)[0]}


def _recompute_from_registry() -> tuple[set[str], set[str], set[str]]:
    agents: set[str] = set()
    factories: set[str] = set()
    applications: set[str] = set()
    for product_id, manifest in iter_product_manifests():
        for binding in manifest.agents:
            agents |= _top_level(binding.import_path)
            factories |= _top_level(binding.factory_path)
        applications.add(f"{product_id}_application")
    return agents, factories, applications


def _derived() -> tuple[set[str], set[str], set[str]]:
    return (
        set(registered_agent_prefixes().prefixes),
        set(registered_factory_prefixes().prefixes),
        set(registered_application_prefixes().prefixes),
    )


def _lab_strict_env() -> ApplicationEnvironmentProfile:
    base = ApplicationEnvironmentProfile.lab_defaults()
    return base.model_copy(update={"meta": base.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT})})


def _record_imports(monkeypatch: pytest.MonkeyPatch) -> list[str]:
    calls: list[str] = []
    real_import = importlib.import_module

    def _recording(name: str, package: str | None = None) -> ModuleType:
        calls.append(name)
        if name.startswith("zz_synth"):
            raise ModuleNotFoundError(f"No module named {name!r}", name=name)
        return real_import(name, package)

    monkeypatch.setattr(importlib, "import_module", _recording)
    return calls


def _assert_guarded(policy: ImportPolicy) -> None:
    assert policy.fail_closed is True
    assert policy.agent_modules is not None
    assert policy.factory_modules is not None
    assert policy.application_packages is not None


def test_ut_a7c_01_sets_follow_the_registered_manifests(monkeypatch: pytest.MonkeyPatch) -> None:
    agents, factories, applications = _recompute_from_registry()
    assert "legal" in agents
    assert factories
    assert "legal_application" in applications
    assert _derived() == (agents, factories, applications)

    original = dict(iter_product_manifests())
    monkeypatch.setattr(product_manifest_registry, "_REGISTERED", dict(original))
    synthetic = ApplicationManifest.lab(
        app_id="zz_synth",
        name="Synthetic",
        agents=[AgentBinding.deserialize(import_path=_SYNTH_AGENT_PATH, factory_path="zz_synth_factory.mod.build")],
    )
    register_product_manifest("zz_synth", synthetic)
    grown = _derived()
    assert grown == _recompute_from_registry()
    assert grown == (
        agents | {"zz_synth_t2"},
        factories | {"zz_synth_factory"},
        applications | {"zz_synth_application"},
    )

    monkeypatch.setattr(product_manifest_registry, "_REGISTERED", dict(original))
    assert _derived() == (agents, factories, applications)


def test_ut_a7c_02_empty_registry_in_guard_posture_permits_nothing(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(product_manifest_registry, "_REGISTERED", {})
    for policy in (
        guarded_import_policy(),
        import_policy_for_environment(ApplicationEnvironmentProfile.product_defaults()),
    ):
        _assert_guarded(policy)
        for allowlist in (policy.agent_modules, policy.factory_modules, policy.application_packages):
            assert allowlist is not None
            assert allowlist == ImportPrefixAllowlist(prefixes=())
            assert not allowlist.permits("legal")


def test_ut_a7c_03_posture_is_decided_by_wiring_fails_closed(monkeypatch: pytest.MonkeyPatch) -> None:
    assert UNRESTRICTED == ImportPolicy(
        agent_modules=None,
        factory_modules=None,
        application_packages=None,
        fail_closed=False,
    )
    lab_env = ApplicationEnvironmentProfile.lab_defaults()
    assert import_policy_for_environment(lab_env) == UNRESTRICTED
    lab_manifest = ApplicationManifest.lab(
        app_id="a7c_lab", name="Lab", agents=[AgentBinding.mount(LegalAgent, contract_id="legal")]
    )
    assert import_policy_for_manifest(lab_manifest) == UNRESTRICTED

    product_env = ApplicationEnvironmentProfile.product_defaults()
    assert import_policy_for_environment(product_env) == guarded_import_policy()
    _assert_guarded(import_policy_for_environment(product_env))
    _assert_guarded(import_policy_for_environment(_lab_strict_env()))
    product_manifest = ApplicationManifest.product(
        app_id="a7c_product",
        name="Product",
        agents=[AgentBinding.mount(LegalAgent, contract_id="legal")],
        route_prefix="/v1/a7c",
        env_prefix="A7C_",
    )
    _assert_guarded(import_policy_for_manifest(product_manifest))

    seen: list[ApplicationEnvironmentProfile] = []

    def _always(env: ApplicationEnvironmentProfile) -> bool:
        seen.append(env)
        return True

    monkeypatch.setattr(import_policy, "wiring_fails_closed", _always)
    _assert_guarded(import_policy_for_environment(lab_env))
    assert seen == [lab_env]

    monkeypatch.setattr(import_policy, "wiring_fails_closed", lambda env: False)
    assert import_policy_for_environment(product_env) == UNRESTRICTED


def test_ut_a7c_04_new_keywords_forward_to_the_loader(monkeypatch: pytest.MonkeyPatch) -> None:
    legal_only = ImportPrefixAllowlist(prefixes=("legal",))
    refused = AgentBinding.deserialize(import_path=_SYNTH_AGENT_PATH, contract_id="zz_synth")
    permitted = AgentBinding.deserialize(import_path="legal.legal_agent.LegalAgent", contract_id="legal")
    calls = _record_imports(monkeypatch)

    with pytest.raises(AgentImportError):
        resolve_agent_type_from_binding(refused, allowed_prefixes=legal_only, fail_closed=True)
    with pytest.raises(AgentImportError):
        resolve_agent_contract_from_binding(refused, allowed_prefixes=legal_only, fail_closed=True)
    with pytest.raises(AgentImportError):
        resolve_roster_agent_contract(
            refused,
            contract_authority=None,
            allow_compatibility_resolver=True,
            allowed_prefixes=legal_only,
            fail_closed=True,
        )
    with pytest.raises(AgentImportError):
        resolve_agent_type_from_binding(permitted, allowed_prefixes=None, fail_closed=True)
    assert [name for name in calls if name.startswith("zz_synth")] == []
    assert "legal.legal_agent" not in calls

    assert resolve_agent_type_from_binding(permitted, allowed_prefixes=legal_only, fail_closed=True) is LegalAgent
    contract = resolve_roster_agent_contract(
        permitted,
        contract_authority=None,
        allow_compatibility_resolver=True,
        allowed_prefixes=legal_only,
        fail_closed=True,
    )
    assert contract.id == "legal"


def test_ut_a7c_05_registered_product_ids_map_to_application_directories() -> None:
    product_ids = [product_id for product_id, _ in iter_product_manifests()]
    assert product_ids
    for product_id in product_ids:
        assert (_REPO_ROOT / "applications" / f"{product_id}_application").is_dir(), product_id
    assert set(registered_application_prefixes().prefixes) == {
        f"{product_id}_application" for product_id in product_ids
    }

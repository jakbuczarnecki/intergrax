# © Artur Czarnecki. All rights reserved.

"""PDC-D6 Amendment 1: R027 host attestation seed, discovery provenance, plan-context fail-closed."""

from __future__ import annotations

import ast
import asyncio
import base64
import secrets
from dataclasses import dataclass, field
from types import SimpleNamespace
from typing import Any

import pytest
from intergrax.applications._shared import attestation_runtime_bridge as bridge
from intergrax.applications.contracts.environment_profile import (
    ApplicationSecurityProfile,
    ExecutionBoundaryExportProfile,
)
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.core.plugins import discovery
from intergrax.core.plugins.admission import PluginAdmissionReasonCode, PluginProvenanceAdmission
from intergrax.core.plugins.discovery import EntryPointSpec, register_plugins_with_report
from intergrax.core.security.security_configuration import (
    DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE,
    PLUGIN_TRUST_POLICY_MISSING,
    SecurityConfigurationError,
)
from intergrax.rag.retrieval.retrieval_result import RetrievalChunk
from intergrax.runtime.attestation import host_attestation
from intergrax.runtime.nexus.tools import plan_context_invocation as plan_module

from ._support import REPO_ROOT, lab_env, product_env

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_SEED_CONSTANT = "POC_ATTESTATION_DEMO_SIGNING_SEED"
_KEY_ENV = host_attestation.HOST_SIGNING_KEY_ENV


def _generated_key_material() -> str:
    """Generated 32-byte Ed25519 seed, base64 encoded (a test value, never a real key)."""
    return base64.b64encode(secrets.token_bytes(32)).decode("ascii")


def _with_export(env):
    env.execution_boundary_export_profile = ExecutionBoundaryExportProfile(
        enabled=True, host_signing_enabled=True, host_signing_public_key_id="host-1"
    )
    return env


class _SeedSpy:
    """Replaces the lazy module attribute hook to record any access to the demo seed."""

    def __init__(self, monkeypatch: pytest.MonkeyPatch) -> None:
        self.accessed = 0
        original = host_attestation.__getattr__

        def _spy(name: str):
            if name == _SEED_CONSTANT:
                self.accessed += 1
            return original(name)

        monkeypatch.setattr(host_attestation, "__getattr__", _spy)


# ---------------------------------------------------------------------------
# (a) R027 - host attestation
# ---------------------------------------------------------------------------


def test_d6_r027_seed_is_not_a_module_global_but_stays_available_to_lab_paths() -> None:
    assert _SEED_CONSTANT not in vars(host_attestation)
    seed = getattr(host_attestation, _SEED_CONSTANT)
    assert isinstance(seed, bytes) and len(seed) == 32
    with pytest.raises(AttributeError):
        getattr(host_attestation, "NOT_A_REAL_ATTRIBUTE")


def test_d6_r027_resolver_has_no_implicit_demo_seed_fallback() -> None:
    with pytest.raises(SecurityConfigurationError) as excinfo:
        host_attestation.resolve_host_attestation_sealer_from_env(enabled=True, public_key_id="host-1", environ={})
    assert excinfo.value.reason_code == DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE
    assert (
        host_attestation.resolve_host_attestation_sealer_from_env(enabled=False, public_key_id="host-1", environ={})
        is None
    )


def test_d6_r027_named_lab_opt_in_reaches_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SeedSpy(monkeypatch)
    sealer = host_attestation.resolve_host_attestation_sealer_from_env(
        enabled=True, public_key_id="host-1", environ={}, lab_demo_seed_opt_in=True
    )
    assert sealer is not None
    assert spy.accessed == 1


def test_d6_r027_explicit_key_never_touches_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SeedSpy(monkeypatch)
    key_material = _generated_key_material()
    sealer = host_attestation.resolve_host_attestation_sealer_from_env(
        enabled=True, public_key_id="host-1", environ={_KEY_ENV: key_material}
    )
    assert sealer is not None
    assert spy.accessed == 0


def test_d6_r027_strict_product_path_cannot_reach_the_seed(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SeedSpy(monkeypatch)
    env = _with_export(product_env())
    with pytest.raises(SecurityConfigurationError) as excinfo:
        bridge.build_boundary_event_buffer(env, environ={})
    assert excinfo.value.reason_code == DEMO_SEED_FORBIDDEN_IN_STRICT_PROFILE
    # an explicit opt-in cannot unlock the seed on a strict profile either
    with pytest.raises(SecurityConfigurationError):
        bridge.build_boundary_event_buffer(env, environ={}, lab_demo_seed_opt_in=True)
    assert spy.accessed == 0


def test_d6_r027_strict_execution_mode_on_a_lab_profile_is_also_strict(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SeedSpy(monkeypatch)
    env = _with_export(lab_env())
    env.execution_mode = ExecutionMode.STRICT
    with pytest.raises(SecurityConfigurationError):
        bridge.build_boundary_event_buffer(env, environ={})
    assert spy.accessed == 0


def test_d6_r027_strict_profile_with_an_explicit_key_builds_the_buffer(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    spy = _SeedSpy(monkeypatch)
    key_material = _generated_key_material()
    buffer = bridge.build_boundary_event_buffer(_with_export(product_env()), environ={_KEY_ENV: key_material})
    assert buffer is not None
    assert spy.accessed == 0


def test_d6_r027_lab_profile_is_the_named_lab_path(monkeypatch: pytest.MonkeyPatch) -> None:
    spy = _SeedSpy(monkeypatch)
    buffer = bridge.build_boundary_event_buffer(_with_export(lab_env()), environ={})
    assert buffer is not None
    assert spy.accessed == 1


def test_d6_r027_static_gate_no_shared_wiring_or_package_init_references_the_seed() -> None:
    offenders: list[str] = []
    for path in sorted((REPO_ROOT / "intergrax").rglob("*.py")):
        relative = path.relative_to(REPO_ROOT).as_posix()
        if "tests" in path.parts:
            continue
        text = path.read_text(encoding="utf-8")
        if _SEED_CONSTANT not in text:
            continue
        if relative in {
            "intergrax/runtime/attestation/host_attestation.py",
            "intergrax/runtime/execution_evidence/attestor.py",
        }:
            continue
        offenders.append(relative)
    assert offenders == []


def test_d6_r027_static_gate_bridge_passes_the_opt_in_only_for_non_strict_profiles() -> None:
    source = (REPO_ROOT / "intergrax/applications/_shared/attestation_runtime_bridge.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    calls = [
        node
        for node in ast.walk(tree)
        if isinstance(node, ast.Call)
        and isinstance(node.func, ast.Name)
        and node.func.id == "resolve_host_attestation_sealer_from_env"
    ]
    assert len(calls) == 1
    keyword = {kw.arg: kw.value for kw in calls[0].keywords}["lab_demo_seed_opt_in"]
    assert isinstance(keyword, ast.UnaryOp) and isinstance(keyword.op, ast.Not)
    init = (REPO_ROOT / "intergrax/runtime/attestation/__init__.py").read_text(encoding="utf-8")
    assert _SEED_CONSTANT not in init


# ---------------------------------------------------------------------------
# (b) discovery - provenance before import
# ---------------------------------------------------------------------------


def _spec(name: str, distribution: str | None) -> EntryPointSpec:
    return EntryPointSpec(name=name, group="intergrax.tools", value=f"mod_{name}:Plugin", distribution=distribution)


def _discovery_harness(monkeypatch: pytest.MonkeyPatch, specs: tuple[EntryPointSpec, ...]) -> list[str]:
    loaded: list[str] = []
    monkeypatch.setattr(discovery, "iter_entry_point_specs", lambda _group: specs)

    def _load(value: str) -> object:
        loaded.append(value)
        return type("Plugin", (), {})

    monkeypatch.setattr(discovery, "load_entry_point_value", _load)
    return loaded


class _RejectUnsigned:
    """Duck-typed provenance admission: rejects every spec without a distribution."""

    def partition(self, specs):
        admitted, rejected = [], []
        for spec in specs:
            if spec.distribution:
                admitted.append(spec)
            else:
                rejected.append(discovery_rejection(spec))
        return tuple(admitted), tuple(rejected)


def discovery_rejection(spec: EntryPointSpec):
    from intergrax.core.plugins.admission import PluginAdmissionRejection

    return PluginAdmissionRejection(
        spec=spec,
        reason_code=PluginAdmissionReasonCode.PLUGIN_PROVENANCE_MISSING,
        reason="unsigned",
    )


def test_d6_r016_register_plugins_checks_provenance_before_importing_the_target(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    signed, unsigned = _spec("signed", "acme-plugin"), _spec("unsigned", None)
    loaded = _discovery_harness(monkeypatch, (signed, unsigned))
    registered: list[str] = []

    def _register(plugin_type, spec):
        registered.append(spec.name)
        return True, None

    report = register_plugins_with_report(
        "intergrax.tools",
        _register,
        discover_entry_points=True,
        provenance=_RejectUnsigned(),  # type: ignore[arg-type]
    )
    assert loaded == ["mod_signed:Plugin"], "rejected target must never be imported"
    assert registered == ["signed"]
    assert [item.spec.name for item in report.rejected] == ["unsigned"]
    assert report.critical_bootstrap_acceptable is False


def test_d6_r016_require_provenance_without_policy_fails_closed(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    loaded = _discovery_harness(monkeypatch, (_spec("a", "acme-plugin"),))
    with pytest.raises(SecurityConfigurationError) as excinfo:
        register_plugins_with_report(
            "intergrax.tools",
            lambda plugin_type, spec: (True, None),
            discover_entry_points=True,
            require_provenance=True,
        )
    assert excinfo.value.reason_code == PLUGIN_TRUST_POLICY_MISSING
    assert loaded == []


def test_d6_r016_default_discovery_behaviour_is_unchanged(monkeypatch: pytest.MonkeyPatch) -> None:
    loaded = _discovery_harness(monkeypatch, (_spec("a", None), _spec("b", "acme-plugin")))
    report = register_plugins_with_report(
        "intergrax.tools",
        lambda plugin_type, spec: (True, None),
        discover_entry_points=True,
    )
    assert sorted(loaded) == ["mod_a:Plugin", "mod_b:Plugin"]
    assert report.registered_count == 2
    assert isinstance(PluginProvenanceAdmission, type)


# ---------------------------------------------------------------------------
# (c) plan-context invocation fail-closed
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class _Built:
    retrieved_chunks: list[RetrievalChunk]
    rag_used: bool = True
    rag_reason: str | None = None


@dataclass
class _State:
    context: Any
    session: Any = field(default_factory=lambda: SimpleNamespace())
    request: Any = None
    base_history: Any = None
    context_builder_result: Any = None
    used_rag: bool = False

    def trace_event(self, **_kwargs: Any) -> None:
        return None


def _rag_state(*, production_mode: bool, security_profile: Any) -> _State:
    chunks = [
        RetrievalChunk(id="clean", text="revenue grew", score=0.9),
        RetrievalChunk(id="poison", text="untrusted", score=0.05),
    ]

    class _Builder:
        async def build_context(self, **_kwargs: Any) -> _Built:
            return _Built(retrieved_chunks=list(chunks))

    config = SimpleNamespace(enable_rag=True, production_mode=production_mode, security_profile=security_profile)
    return _State(context=SimpleNamespace(config=config, context_builder=_Builder()))


def _run_rag(monkeypatch: pytest.MonkeyPatch, state: _State) -> list[str]:
    monkeypatch.setattr(plan_module, "record_rag_invocation_and_enforce", lambda _s: None)
    monkeypatch.setattr(plan_module, "build_rag_retrieve_input", lambda _s: None)
    monkeypatch.setattr(plan_module, "invoke_catalog_context_tool", lambda *_a, **_k: False)
    monkeypatch.setattr(plan_module, "merge_provider_metadata_into_request", lambda _s: None)
    asyncio.run(plan_module.run_rag_context(state))  # type: ignore[arg-type]
    return [chunk.id for chunk in state.context_builder_result.retrieved_chunks]


def test_d6_r014_plan_context_strict_profile_without_security_profile_filters(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = _run_rag(monkeypatch, _rag_state(production_mode=True, security_profile=None))
    assert kept == ["clean"]


def test_d6_r014_plan_context_strict_profile_explicit_false_is_the_only_opt_out(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    profile = ApplicationSecurityProfile(retrieval_poisoning_defense_enabled=False)
    kept = _run_rag(monkeypatch, _rag_state(production_mode=True, security_profile=profile))
    assert kept == ["clean", "poison"]
    enabled = ApplicationSecurityProfile(retrieval_poisoning_defense_enabled=True)
    kept = _run_rag(monkeypatch, _rag_state(production_mode=True, security_profile=enabled))
    assert kept == ["clean"]


def test_d6_r014_plan_context_non_strict_behaviour_is_unchanged(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    kept = _run_rag(monkeypatch, _rag_state(production_mode=False, security_profile=None))
    assert kept == ["clean", "poison"]
    enabled = ApplicationSecurityProfile(retrieval_poisoning_defense_enabled=True)
    kept = _run_rag(monkeypatch, _rag_state(production_mode=False, security_profile=enabled))
    assert kept == ["clean"]

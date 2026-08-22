# © Artur Czarnecki. All rights reserved.

"""PBA-FIX-C — guardrail provider configuration ownership."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import ValidationError

from intergrax.applications._shared.guardrail_runtime_bridge import resolve_guardrail_backend
from intergrax.applications.contracts.environment_profile import (
    ApplicationEnvironmentProfile,
    GuardrailProfile,
)
from intergrax.applications.contracts.environment_profile.sub_profiles import (
    GuardrailProfile as AppGuardrailProfile,
)
from intergrax.contracts.host_profile_slices import GuardrailProfile as HostGuardrailProfile
from intergrax.integrations.contracts.llm_guardrail import GuardrailBackendOptions
from intergrax.integrations.providers.llm_guardrail._factory import create_chained_guardrail_backend
from intergrax.integrations.providers.llm_guardrail.bundles.bedrock_guardrails import (
    BedrockGuardrailOptions,
    create_bedrock_guardrails_backend,
)
from intergrax.integrations.providers.llm_guardrail.register_all import register_llm_guardrail_integrations
from intergrax.integrations.registry.profile import IntegrationProfile

pytestmark = [pytest.mark.unit, pytest.mark.gate]

_BRIDGE_PATH = Path("intergrax/applications/_shared/guardrail_runtime_bridge.py")


def test_c1_host_guardrail_profile_rejects_bedrock_field() -> None:
    with pytest.raises(ValidationError):
        HostGuardrailProfile(bedrock_guardrail_policy_id="leak")


def test_c1_app_guardrail_profile_rejects_bedrock_field() -> None:
    with pytest.raises(ValidationError):
        AppGuardrailProfile(bedrock_guardrail_policy_id="leak")


def test_c2_generic_backend_options_have_no_bedrock_field() -> None:
    with pytest.raises(ValidationError):
        GuardrailBackendOptions(bedrock_guardrail_policy_id="leak")


def test_c3_bedrock_provider_options_resolve_policy_id(monkeypatch: pytest.MonkeyPatch) -> None:
    register_llm_guardrail_integrations(override=True)
    monkeypatch.delenv("INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID", raising=False)

    captured: list[str | None] = []

    def _fake_apply(text: str, *, policy_id: str, mode: str) -> dict[str, object]:
        captured.append(policy_id)
        return {"action": "NONE"}

    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.opens.bedrock_apply_guardrail",
        _fake_apply,
    )

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pba-fix-c.bedrock-opt")
    env = env.model_copy(
        update={
            "integration_profile": IntegrationProfile(
                llm_guardrail="bedrock_guardrails",
                options={"bedrock_guardrails": {"policy_id": "test-policy"}},
            ),
            "guardrail_profile": GuardrailProfile(enabled=True),
        },
    )
    backend = resolve_guardrail_backend(env)
    assert backend is not None
    backend.scan_input("hello")
    assert captured == ["test-policy"]


def test_c4_bedrock_env_fallback_without_explicit_policy(monkeypatch: pytest.MonkeyPatch) -> None:
    register_llm_guardrail_integrations(override=True)
    monkeypatch.setenv("INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID", "test-env-policy")

    captured: list[str | None] = []

    def _fake_apply(text: str, *, policy_id: str, mode: str) -> dict[str, object]:
        captured.append(policy_id)
        return {"action": "NONE"}

    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.opens.bedrock_apply_guardrail",
        _fake_apply,
    )

    bridge_source = _BRIDGE_PATH.read_text(encoding="utf-8")
    assert "INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID" not in bridge_source

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pba-fix-c.bedrock-env")
    env = env.model_copy(
        update={
            "integration_profile": IntegrationProfile(
                llm_guardrail="bedrock_guardrails",
                options={"bedrock_guardrails": {}},
            ),
            "guardrail_profile": GuardrailProfile(enabled=True),
        },
    )
    backend = resolve_guardrail_backend(env)
    assert backend is not None
    backend.scan_input("hello")
    assert captured == ["test-env-policy"]


def test_c5_provider_isolation_distinct_option_maps(monkeypatch: pytest.MonkeyPatch) -> None:
    register_llm_guardrail_integrations(override=True)
    monkeypatch.delenv("INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID", raising=False)
    monkeypatch.delenv("INTERGRAX_NEMO_COLANG_CONFIG_PATH", raising=False)

    bedrock_policies: list[str | None] = []
    nemo_paths: list[str | None] = []

    def _bedrock_apply(text: str, *, policy_id: str, mode: str) -> dict[str, object]:
        bedrock_policies.append(policy_id)
        return {"action": "NONE"}

    def _nemo_scan(_text: str, *, mode: str, colang_path: str) -> dict[str, object]:
        nemo_paths.append(colang_path)
        return {"allowed": True, "detail": "ok"}

    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.opens.bedrock_apply_guardrail",
        _bedrock_apply,
    )
    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bundles.nemo_guardrails.nemo_scan_colang",
        _nemo_scan,
    )

    bedrock_backend = create_bedrock_guardrails_backend(
        provider_options={"policy_id": "bedrock-sentinel"},
    )
    bedrock_backend.scan_input("hello")

    from intergrax.integrations.providers.llm_guardrail.bundles.nemo_guardrails import (
        create_nemo_guardrails_backend,
    )

    nemo_backend = create_nemo_guardrails_backend(
        provider_options={"config_path": "nemo-sentinel"},
    )
    nemo_backend.scan_input("hello")

    assert bedrock_policies == ["bedrock-sentinel"]
    assert nemo_paths == ["nemo-sentinel"]


def test_c6_chained_provider_isolation(monkeypatch: pytest.MonkeyPatch) -> None:
    register_llm_guardrail_integrations(override=True)
    monkeypatch.delenv("INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID", raising=False)
    monkeypatch.delenv("INTERGRAX_NEMO_COLANG_CONFIG_PATH", raising=False)

    bedrock_policies: list[str | None] = []
    nemo_paths: list[str | None] = []

    def _bedrock_apply(text: str, *, policy_id: str, mode: str) -> dict[str, object]:
        bedrock_policies.append(policy_id)
        return {"action": "NONE"}

    def _nemo_scan(_text: str, *, mode: str, colang_path: str) -> dict[str, object]:
        nemo_paths.append(colang_path)
        return {"allowed": True, "detail": "ok"}

    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.opens.bedrock_apply_guardrail",
        _bedrock_apply,
    )
    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bundles.nemo_guardrails.nemo_scan_colang",
        _nemo_scan,
    )

    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="pba-fix-c.chain")
    env = env.model_copy(
        update={
            "integration_profile": IntegrationProfile(
                llm_guardrail="bedrock_guardrails",
                options={
                    "bedrock_guardrails": {"policy_id": "chain-bedrock"},
                    "nemo_guardrails": {"config_path": "chain-nemo"},
                },
            ),
            "guardrail_profile": GuardrailProfile(
                enabled=True,
                secondary_slug="nemo_guardrails",
            ),
        },
    )
    backend = resolve_guardrail_backend(env)
    assert backend is not None
    backend.scan_input("hello")

    assert bedrock_policies == ["chain-bedrock"]
    assert nemo_paths == ["chain-nemo"]


def test_c7_generic_wiring_has_no_bedrock_references() -> None:
    source = _BRIDGE_PATH.read_text(encoding="utf-8").lower()
    assert "bedrock" not in source
    assert "aws" not in source
    assert "intergrax_bedrock_guardrail_policy_id" not in source


def test_c9_bedrock_positive_control_policy_id_reaches_transport(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: list[tuple[str, str]] = []

    def _fake_apply(text: str, *, policy_id: str, mode: str) -> dict[str, object]:
        captured.append((policy_id, mode))
        return {"action": "NONE"}

    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.opens.bedrock_apply_guardrail",
        _fake_apply,
    )

    backend = create_bedrock_guardrails_backend(
        provider_options={"policy_id": "positive-control-policy"},
    )
    backend.scan_output("output text", prompt="prompt")
    assert captured == [("positive-control-policy", "output")]


def test_c10_bedrock_rejects_unknown_provider_options() -> None:
    with pytest.raises(ValidationError):
        BedrockGuardrailOptions.model_validate({"policy_id": "ok", "unknown_key": "bad"})


def test_chained_factory_scopes_options_per_slug(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("INTERGRAX_BEDROCK_GUARDRAIL_POLICY_ID", raising=False)
    monkeypatch.delenv("INTERGRAX_NEMO_COLANG_CONFIG_PATH", raising=False)

    bedrock_policies: list[str | None] = []
    nemo_paths: list[str | None] = []

    def _bedrock_apply(text: str, *, policy_id: str, mode: str) -> dict[str, object]:
        bedrock_policies.append(policy_id)
        return {"action": "NONE"}

    def _nemo_scan(_text: str, *, mode: str, colang_path: str) -> dict[str, object]:
        nemo_paths.append(colang_path)
        return {"allowed": True, "detail": "ok"}

    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bedrock_guardrails.opens.bedrock_apply_guardrail",
        _bedrock_apply,
    )
    monkeypatch.setattr(
        "intergrax.integrations.providers.llm_guardrail.bundles.nemo_guardrails.nemo_scan_colang",
        _nemo_scan,
    )

    backend = create_chained_guardrail_backend(
        "bedrock_guardrails",
        "nemo_guardrails",
        provider_options_map={
            "bedrock_guardrails": {"policy_id": "factory-bedrock"},
            "nemo_guardrails": {"config_path": "factory-nemo"},
        },
    )
    backend.scan_input("hello")
    assert bedrock_policies == ["factory-bedrock"]
    assert nemo_paths == ["factory-nemo"]

# © Artur Czarnecki. All rights reserved.

"""FC-3 Part F: guardrail fail-closed (NT-FC3-GR1..GR5)."""

from __future__ import annotations

from unittest.mock import patch

import pytest
from intergrax.applications._shared.guardrail_runtime_bridge import resolve_guardrail_backend
from intergrax.applications.contracts.environment_profile import ApplicationEnvironmentProfile, GuardrailProfile
from intergrax.applications.contracts.execution_mode import ExecutionMode
from intergrax.integrations.contracts.base import IntegrationConfigurationError, LabSelectorOptIn
from intergrax.integrations.providers.llm_guardrail._factory import create_guardrail_backend
from intergrax.integrations.providers.llm_guardrail._stub_backend import StubLlmGuardrailBackend
from intergrax.integrations.registry.presets import harness_guardrail_stack

pytestmark = pytest.mark.unit

_LAB_OPT_IN_STUB = LabSelectorOptIn(
    selector="no_such_slug",
    reason="FC-3 guardrail unknown-slug positive control",
)

_LAB_OPT_IN_LLM_GUARD = LabSelectorOptIn(
    selector="llm_guard",
    reason="FC-3 guardrail pattern-fallback positive control",
)


def test_nt_fc3_gr1_unknown_slug_raises() -> None:
    with pytest.raises(IntegrationConfigurationError, match="unknown guardrail slug"):
        create_guardrail_backend("no_such_slug")


def test_nt_fc3_gr2_unknown_slug_with_opt_in_flagged_stub() -> None:
    backend = create_guardrail_backend("no_such_slug", lab_opt_in=_LAB_OPT_IN_STUB)
    assert isinstance(backend, StubLlmGuardrailBackend)
    result = backend.scan_input("hello")
    assert result.audit_payload.get("provenance") == "stub"
    assert result.audit_payload.get("requested_slug") == "no_such_slug"


def test_nt_fc3_gr3_llm_guard_vendor_missing_raises_without_opt_in() -> None:
    with patch(
        "intergrax.integrations.providers.llm_guardrail._vendor_opens.vendor_engine_ready",
        return_value=False,
    ):
        with pytest.raises(IntegrationConfigurationError):
            create_guardrail_backend("llm_guard")


def test_nt_fc3_gr4_llm_guard_vendor_missing_opt_in_sets_engine_degraded() -> None:
    with patch(
        "intergrax.integrations.providers.llm_guardrail._vendor_opens.llm_guard_scan_input",
        return_value=None,
    ):
        backend = create_guardrail_backend("llm_guard", lab_opt_in=_LAB_OPT_IN_LLM_GUARD)
        result = backend.scan_input("hello lab")
    assert result.engine_degraded is True


def test_nt_fc3_gr5_strict_resolve_raises_when_vendor_degraded() -> None:
    env = ApplicationEnvironmentProfile.lab_defaults(profile_id="fc3.guardrail.strict")
    env = env.model_copy(
        update={
            "meta": env.meta.model_copy(update={"execution_mode": ExecutionMode.STRICT}),
            "integration_profile": harness_guardrail_stack(primary="llm_guard"),
            "guardrail_profile": GuardrailProfile(enabled=True),
        },
    )
    with patch(
        "intergrax.integrations.providers.llm_guardrail._vendor_opens.vendor_engine_ready",
        return_value=False,
    ):
        with pytest.raises(IntegrationConfigurationError):
            resolve_guardrail_backend(env)

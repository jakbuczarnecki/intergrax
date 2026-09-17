# © Artur Czarnecki. All rights reserved.

"""GR-7-A8-R2 — public boundary cleanup for provider invocation reliability emission context."""

from __future__ import annotations

import dataclasses
import importlib
import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO_ROOT = Path(__file__).resolve().parents[4]
_ERL_CONTRACTS = _REPO_ROOT / "intergrax" / "contracts" / "enterprise_reliability"
_EMISSION_CONTRACT = _ERL_CONTRACTS / "provider_invocation_reliability_emission.py"
_EVIDENCE_CONTRACT = _ERL_CONTRACTS / "provider_invocation_reliability_evidence.py"


@pytest.mark.gate
def test_unapproved_emission_contract_module_absent() -> None:
    assert not _EMISSION_CONTRACT.is_file(), (
        "provider_invocation_reliability_emission.py must not remain a public contract"
    )


@pytest.mark.gate
def test_dispatch_context_not_exported_from_enterprise_reliability_contracts() -> None:
    pkg = importlib.import_module("intergrax.contracts.enterprise_reliability")
    public = getattr(pkg, "__all__", ())
    assert "ProviderInvocationReliabilityDispatchContext" not in public
    assert not hasattr(pkg, "ProviderInvocationReliabilityDispatchContext")


@pytest.mark.gate
def test_approved_provider_invocation_reliability_evidence_family_unchanged() -> None:
    evidence = importlib.import_module(
        "intergrax.contracts.enterprise_reliability.provider_invocation_reliability_evidence",
    )
    expected = {
        "NullProviderInvocationReliabilityEvidenceObserver",
        "ProviderInvocationReliabilityCorrelation",
        "ProviderInvocationReliabilityEvidenceObserver",
        "ProviderInvocationReliabilityFact",
        "ProviderInvocationReliabilityTracePhase",
    }
    assert expected.issubset(set(getattr(evidence, "__all__", ())))


def test_internal_dispatch_context_exact_fields() -> None:
    from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_dispatch_context import (
        ProviderInvocationReliabilityDispatchContext,
    )

    fields = {f.name for f in dataclasses.fields(ProviderInvocationReliabilityDispatchContext)}
    assert fields == {
        "tenant_id",
        "effect_contract_id",
        "execution_id",
        "attempt_id",
        "observer",
    }
    assert dataclasses.is_dataclass(ProviderInvocationReliabilityDispatchContext)
    params = inspect.signature(ProviderInvocationReliabilityDispatchContext).parameters
    assert params["tenant_id"].annotation in (str, "str")
    assert params["effect_contract_id"].annotation in (str | None, "str | None", "Optional[str]")
    assert params["execution_id"].annotation in (str | None, "str | None", "Optional[str]")
    assert params["attempt_id"].annotation in (str | None, "str | None", "Optional[str]")


def test_provider_invocation_dispatch_port_has_no_reliability_context_kwarg() -> None:
    from intergrax.contracts.provider_invocation_dispatch import ProviderInvocationDispatchPort

    sig = inspect.signature(ProviderInvocationDispatchPort.dispatch_after_intent_persisted)
    assert "reliability_dispatch" not in sig.parameters


def test_reliability_aware_dispatch_port_lives_in_runtime_composition() -> None:
    from intergrax.runtime.enterprise_reliability.provider_invocation_reliability_dispatch_port import (
        ProviderInvocationReliabilityAwareDispatchPort,
    )

    sig = inspect.signature(
        ProviderInvocationReliabilityAwareDispatchPort.dispatch_after_intent_persisted,
    )
    assert "reliability_dispatch" in sig.parameters


def test_evidence_contract_module_still_present() -> None:
    assert _EVIDENCE_CONTRACT.is_file()

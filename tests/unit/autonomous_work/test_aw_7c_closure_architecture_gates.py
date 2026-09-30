# © Artur Czarnecki. All rights reserved.

"""AW-7C-CLOSURE-R1 static architecture gates."""

from __future__ import annotations

import inspect
from pathlib import Path

import pytest

pytestmark = pytest.mark.unit

_REPO = Path(__file__).resolve().parents[3]
_AW7C_QUAL = _REPO / "intergrax/integrations/qualification"
_REF_EXECUTE = _AW7C_QUAL / "reference_scoped_adaptive_integration_execution.py"
_DELEGATE = _AW7C_QUAL / "scoped_adaptive_integration_execution_runtime_delegate.py"
_INTAKE = _AW7C_QUAL / "scoped_adaptive_integration_execution_intake.py"
_CONTRACTS = _REPO / "intergrax/integrations/contracts/scoped_integration_adaptation.py"
_FORBIDDEN_IN_QUAL = (
    "mint_run_id",
    "mint_attempt_id",
    "mint_execution_id",
    "expected_operation",
    "ScopedAdaptiveIntegrationReferenceExecutionIntake",
    "Callable[[ExecutionId], CredentialUseGrant]",
)


@pytest.mark.parametrize("name", sorted(p.name for p in _AW7C_QUAL.glob("scoped_adaptive*.py")))
def test_closure_qualification_modules_forbidden_patterns(name: str) -> None:
    text = (_AW7C_QUAL / name).read_text(encoding="utf-8")
    for token in _FORBIDDEN_IN_QUAL:
        assert token not in text, f"{name} must not contain {token}"


def test_closure_intake_is_composition_not_alternate_intake_owner() -> None:
    text = _INTAKE.read_text(encoding="utf-8")
    assert "class ScopedAdaptiveIntegrationReferenceExecutionIntake" not in text
    assert "build_scoped_adaptive_integration_canonical_execution_intake" in text
    assert "CanonicalExecutionRuntimeAdapter" in text


def test_closure_r1_delegate_uses_effect_preparer_not_operation_port() -> None:
    text = _DELEGATE.read_text(encoding="utf-8")
    assert "effect_preparer" in text
    assert "effect_executor" in text
    assert "operation_port" not in text


def test_closure_r1_intake_wires_preparer_and_canonical_executor() -> None:
    text = _INTAKE.read_text(encoding="utf-8")
    assert "ReferenceScopedAdaptedIntegrationEffectRequestPreparer" in text
    assert "ReferenceScopedAdaptedIntegrationEffectExecutor" in text
    assert "operation_port" not in text


def test_closure_r1_execute_path_no_operation_port() -> None:
    text = _REF_EXECUTE.read_text(encoding="utf-8")
    assert "operation_port" not in text
    assert "effect_preparer" in text
    assert "effect_executor" in text
    assert "validate_admitted_scoped_adaptive_integration_effect_request" in text
    idx_prepare = text.index("effect_preparer.prepare")
    idx_resolve = text.index("credential_broker.resolve_scoped")
    idx_executor = text.index("effect_executor.execute")
    assert idx_prepare < idx_resolve < idx_executor


def test_closure_r1_no_toolruntime_nexus_bypass_in_qual_path() -> None:
    joined = "\n".join(
        (_REF_EXECUTE.read_text(encoding="utf-8"), _DELEGATE.read_text(encoding="utf-8")),
    ).lower()
    assert "toolruntime" not in joined
    assert "intergrax.runtime.nexus" not in joined


def test_closure_r1_contracts_define_effect_request_and_executor() -> None:
    text = _CONTRACTS.read_text(encoding="utf-8")
    assert "class ScopedAdaptedIntegrationEffectRequest" in text
    assert "ScopedAdaptedIntegrationEffectRequestPort" in text
    assert "ScopedAdaptedIntegrationEffectExecutor" in text
    assert ": Any" not in text
    assert "dict[str, Any]" not in text


def test_closure_r1_preparer_port_signature_excludes_credential_material() -> None:
    from intergrax.integrations.contracts.scoped_integration_adaptation import (
        ScopedAdaptedIntegrationEffectRequestPort,
    )

    sig = inspect.signature(ScopedAdaptedIntegrationEffectRequestPort.prepare)
    joined = str(sig).lower()
    assert "credential" not in joined
    assert "broker" not in joined
    assert "resolver" not in joined


def test_closure_r1_no_getattr_hasattr_in_changed_qual_modules() -> None:
    for path in (_REF_EXECUTE, _DELEGATE, _INTAKE):
        text = path.read_text(encoding="utf-8")
        assert "getattr(" not in text
        assert "hasattr(" not in text


def test_closure_r1_broker_result_not_discarded() -> None:
    text = _REF_EXECUTE.read_text(encoding="utf-8")
    assert "del resolved" not in text
    assert "credential_resolution=resolved" in text

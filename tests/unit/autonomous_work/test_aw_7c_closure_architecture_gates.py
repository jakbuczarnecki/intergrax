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
_EFFECT_EXEC_CONTRACTS = (
    _REPO / "intergrax/integrations/contracts/scoped_adapted_integration_effect_execution.py"
)
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
    assert ": Any" not in text
    assert "dict[str, Any]" not in text
    effect_text = _EFFECT_EXEC_CONTRACTS.read_text(encoding="utf-8")
    assert "ScopedAdaptedIntegrationEffectExecutor" in effect_text
    assert "ScopedAdaptedIntegrationEffectExecutionIngress" in effect_text
    assert "sandbox_resource" in effect_text
    assert "credential_resolution" in effect_text
    assert ": Any" not in effect_text


def test_closure_r1_r1_executor_no_reference_context_type_check() -> None:
    text = _REF_EXECUTE.read_text(encoding="utf-8")
    assert "ReferenceScopedAdaptedIntegrationEffectExecutionContext" not in text.split(
        "class ReferenceScopedAdaptedIntegrationEffectExecutor",
    )[1].split("class ReferenceScopedAdaptiveIntegrationSandboxSession")[0]
    assert "isinstance(\n        ingress,\n        ReferenceScopedAdaptedIntegrationEffectExecutionContext," not in text
    assert 'isinstance(ingress, ReferenceScopedAdaptedIntegrationEffectExecutionContext)' not in text


def test_closure_r1_r1_executor_no_resource_rediscovery() -> None:
    executor_block = _REF_EXECUTE.read_text(encoding="utf-8").split(
        "class ReferenceScopedAdaptedIntegrationEffectExecutor",
    )[1].split("class ReferenceScopedAdaptiveIntegrationSandboxSession")[0]
    assert "resolve_scoped(" not in executor_block
    assert "CredentialResolver" not in executor_block
    assert "sandbox_session_manager" not in executor_block


def test_closure_r1_r1_ingress_contract_exposes_sanctioned_resources() -> None:
    text = _EFFECT_EXEC_CONTRACTS.read_text(encoding="utf-8")
    assert "def sandbox_resource(self)" in text or "sandbox_resource(self)" in text
    assert "def credential_resolution(self)" in text or "credential_resolution(self)" in text


def test_closure_r1_r1_effect_execution_contract_import_acyclic() -> None:
    text = _EFFECT_EXEC_CONTRACTS.read_text(encoding="utf-8")
    assert "from intergrax.integrations.contracts.scoped_integration_adaptation import" in text
    assert "from intergrax.integrations.contracts.credential import" in text
    adaptation_text = _CONTRACTS.read_text(encoding="utf-8")
    assert "scoped_adapted_integration_effect_execution" not in adaptation_text


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

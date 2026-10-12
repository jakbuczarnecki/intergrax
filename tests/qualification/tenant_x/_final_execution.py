# © Artur Czarnecki. All rights reserved.

"""Canonical F1–F10 executable registry and adversarial matrix runner."""

from __future__ import annotations

import inspect
from collections.abc import Callable
from dataclasses import dataclass, fields
from inspect import Parameter
from typing import Final

from intergrax.knowledge.contracts.document import KnowledgeDocumentScope
from intergrax.rag.vectorstore.contracts.native_vectorstore import (
    VectorStoreContractError,
    VectorStoreScope,
    _ROUTING_KEYS,
)
from intergrax.rag.vectorstore.providers.native_provider_boundary import validate_scope

from tests.qualification.tenant_x._final_adversarial import (
    run_final_authority_isolation_negatives,
    run_final_child_async_isolation_negatives,
    run_final_credential_isolation_negatives,
    run_final_evidence_isolation_negatives,
    run_final_missing_tenant_fail_closed_negatives,
    run_final_provider_config_isolation_negatives,
    run_final_recovery_isolation_negatives,
    run_final_state_isolation_negatives,
    run_final_trace_preservation_negatives,
)
from tests.qualification.tenant_x._final_matrices import FINAL_ADVERSARIAL_MATRIX
from tests.qualification.tenant_x._final_types import (
    FinalAdversarialClass,
    FinalAdversarialExecutionResult,
    FinalAdversarialOutcome,
    FinalF9ExecutionEvidence,
    FinalF9WidenProof,
)
from tests.qualification.tenant_x._r2_adversarial import (
    assert_missing_tenant_on_credential_scope_rejects,
    assert_vector_scope_cross_tenant_rejects,
)
_EVIDENCE_OWNER = "tests.qualification.tenant_x._final_execution"


def _matrix_expected(adversarial_class: FinalAdversarialClass) -> FinalAdversarialOutcome:
    for row in FINAL_ADVERSARIAL_MATRIX:
        if row.adversarial_class == adversarial_class:
            return row.expected_outcome
    raise KeyError(adversarial_class)


def _execute_rejection_runner(
    adversarial_class: FinalAdversarialClass,
    runner: Callable[[], None],
) -> FinalAdversarialExecutionResult:
    expected = _matrix_expected(adversarial_class)
    try:
        runner()
        return FinalAdversarialExecutionResult(
            adversarial_class=adversarial_class,
            executed=True,
            observed_outcome=expected,
            evidence_source=_EVIDENCE_OWNER,
            failure_detail=None,
        )
    except Exception as exc:  # noqa: BLE001 — qualification must not swallow into PASS
        return FinalAdversarialExecutionResult(
            adversarial_class=adversarial_class,
            executed=True,
            observed_outcome=FinalAdversarialOutcome.FAILED,
            evidence_source=_EVIDENCE_OWNER,
            failure_detail=f"{type(exc).__name__}: {exc}",
        )


def _execute_preservation_runner(
    adversarial_class: FinalAdversarialClass,
    runner: Callable[[], None],
) -> FinalAdversarialExecutionResult:
    expected = _matrix_expected(adversarial_class)
    try:
        runner()
        return FinalAdversarialExecutionResult(
            adversarial_class=adversarial_class,
            executed=True,
            observed_outcome=expected,
            evidence_source=_EVIDENCE_OWNER,
            failure_detail=None,
        )
    except Exception as exc:  # noqa: BLE001
        return FinalAdversarialExecutionResult(
            adversarial_class=adversarial_class,
            executed=True,
            observed_outcome=FinalAdversarialOutcome.FAILED,
            evidence_source=_EVIDENCE_OWNER,
            failure_detail=f"{type(exc).__name__}: {exc}",
        )


_WIDEN_FORBIDDEN_SCOPE_FIELDS: Final[frozenset[str]] = frozenset(
    {
        "tenant_ids",
        "all_tenants",
        "tenant_wildcard",
        "global_tenant",
        "tenant_collection",
        "tenant_range",
    }
)


def _prove_f9_widen_structurally_impossible() -> FinalF9WidenProof:
    """Mechanical contract proof: single explicit tenant authority, no widen representation."""
    scope_field_names = {field.name for field in fields(VectorStoreScope)}
    if scope_field_names & _WIDEN_FORBIDDEN_SCOPE_FIELDS:
        raise AssertionError("VectorStoreScope exposes forbidden tenant-widen fields")

    validate_sig = inspect.signature(validate_scope)
    if tuple(validate_sig.parameters) != ("scope", "tenant_id"):
        raise AssertionError("validate_scope must bind exactly scope + tenant_id")
    tenant_param = validate_sig.parameters["tenant_id"]
    if tenant_param.kind is not Parameter.KEYWORD_ONLY:
        raise AssertionError("validate_scope tenant_id must be keyword-only authority pin")

    model_fields = KnowledgeDocumentScope.model_fields
    if model_fields.keys() & _WIDEN_FORBIDDEN_SCOPE_FIELDS:
        raise AssertionError("KnowledgeDocumentScope exposes forbidden tenant-widen fields")
    tenant_model = model_fields["tenant_id"]
    if not tenant_model.is_required():
        raise AssertionError("tenant_id must be required explicit identity, not optional widen")
    extra_policy = KnowledgeDocumentScope.model_config.get("extra")
    if extra_policy != "forbid":
        raise AssertionError("KnowledgeDocumentScope must forbid extra tenant authority fields")

    if "tenant_id" not in _ROUTING_KEYS:
        raise AssertionError("tenant_id must be a reserved routing key (no membership widen)")

    for name, _member in inspect.getmembers(VectorStoreScope):
        if not name.startswith("_") and any(
            marker in name.lower() for marker in ("wildcard", "all_tenant", "widen", "global")
        ):
            raise AssertionError(f"VectorStoreScope must not expose widen API: {name}")

    pinned = "tenant-f9-pinned-authority"
    bound = validate_scope(VectorStoreScope(tenant_id=pinned), tenant_id=pinned)
    if bound.tenant_id != pinned:
        raise AssertionError("validate_scope must not rewrite pinned tenant authority")
    try:
        validate_scope(VectorStoreScope(tenant_id=pinned), tenant_id="tenant-f9-other")
    except VectorStoreContractError:
        pass
    else:
        raise AssertionError("cross-tenant validate_scope must fail closed")

    return FinalF9WidenProof.STRUCTURALLY_IMPOSSIBLE


def run_final_f9_extension_provider_rewrite() -> FinalAdversarialExecutionResult:
    adversarial_class = FinalAdversarialClass.F9_EXTENSION_PROVIDER_REWRITE
    expected = _matrix_expected(adversarial_class)
    replace_ok = False
    drop_ok = False
    widen_proof: FinalF9WidenProof | None = None
    try:
        assert_vector_scope_cross_tenant_rejects()
        replace_ok = True
        assert_missing_tenant_on_credential_scope_rejects()
        drop_ok = True
        widen_proof = _prove_f9_widen_structurally_impossible()
        f9_detail = FinalF9ExecutionEvidence(
            replace_rejected=replace_ok,
            widen_proof=widen_proof,
            drop_rejected=drop_ok,
        )
        if not f9_detail.complete():
            raise AssertionError("F9 sub-evidence incomplete")
        return FinalAdversarialExecutionResult(
            adversarial_class=adversarial_class,
            executed=True,
            observed_outcome=expected,
            evidence_source=_EVIDENCE_OWNER,
            failure_detail=None,
            f9_evidence=f9_detail,
        )
    except Exception as exc:  # noqa: BLE001
        partial = FinalF9ExecutionEvidence(
            replace_rejected=replace_ok,
            widen_proof=widen_proof,
            drop_rejected=drop_ok,
        )
        return FinalAdversarialExecutionResult(
            adversarial_class=adversarial_class,
            executed=True,
            observed_outcome=FinalAdversarialOutcome.FAILED,
            evidence_source=_EVIDENCE_OWNER,
            failure_detail=f"{type(exc).__name__}: {exc}",
            f9_evidence=partial,
        )


@dataclass(frozen=True, slots=True)
class FinalExecutableCase:
    adversarial_class: FinalAdversarialClass
    runner: Callable[[], FinalAdversarialExecutionResult]


def _run_f1() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F1_AUTHORITY_ISOLATION,
        run_final_authority_isolation_negatives,
    )


def _run_f2() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F2_STATE_ISOLATION,
        run_final_state_isolation_negatives,
    )


def _run_f3() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F3_PROVIDER_CONFIG_ISOLATION,
        run_final_provider_config_isolation_negatives,
    )


def _run_f4() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F4_EVIDENCE_ISOLATION,
        run_final_evidence_isolation_negatives,
    )


def _run_f5() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F5_CREDENTIAL_ISOLATION,
        run_final_credential_isolation_negatives,
    )


def _run_f6() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F6_RECOVERY_ISOLATION,
        run_final_recovery_isolation_negatives,
    )


def _run_f7() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F7_CHILD_ASYNC_ISOLATION,
        run_final_child_async_isolation_negatives,
    )


def _run_f8() -> FinalAdversarialExecutionResult:
    return _execute_rejection_runner(
        FinalAdversarialClass.F8_MISSING_TENANT_FAIL_CLOSED,
        run_final_missing_tenant_fail_closed_negatives,
    )


def _run_f10() -> FinalAdversarialExecutionResult:
    return _execute_preservation_runner(
        FinalAdversarialClass.F10_EVIDENCE_TRACE_PRESERVATION,
        run_final_trace_preservation_negatives,
    )


FINAL_ADVERSARIAL_EXECUTABLE_REGISTRY: Final[tuple[FinalExecutableCase, ...]] = (
    FinalExecutableCase(FinalAdversarialClass.F1_AUTHORITY_ISOLATION, _run_f1),
    FinalExecutableCase(FinalAdversarialClass.F2_STATE_ISOLATION, _run_f2),
    FinalExecutableCase(FinalAdversarialClass.F3_PROVIDER_CONFIG_ISOLATION, _run_f3),
    FinalExecutableCase(FinalAdversarialClass.F4_EVIDENCE_ISOLATION, _run_f4),
    FinalExecutableCase(FinalAdversarialClass.F5_CREDENTIAL_ISOLATION, _run_f5),
    FinalExecutableCase(FinalAdversarialClass.F6_RECOVERY_ISOLATION, _run_f6),
    FinalExecutableCase(FinalAdversarialClass.F7_CHILD_ASYNC_ISOLATION, _run_f7),
    FinalExecutableCase(FinalAdversarialClass.F8_MISSING_TENANT_FAIL_CLOSED, _run_f8),
    FinalExecutableCase(FinalAdversarialClass.F9_EXTENSION_PROVIDER_REWRITE, run_final_f9_extension_provider_rewrite),
    FinalExecutableCase(FinalAdversarialClass.F10_EVIDENCE_TRACE_PRESERVATION, _run_f10),
)


def registry_class_ids() -> frozenset[FinalAdversarialClass]:
    return frozenset(case.adversarial_class for case in FINAL_ADVERSARIAL_EXECUTABLE_REGISTRY)


def run_final_adversarial_matrix() -> tuple[FinalAdversarialExecutionResult, ...]:
    results: list[FinalAdversarialExecutionResult] = []
    for case in FINAL_ADVERSARIAL_EXECUTABLE_REGISTRY:
        results.append(case.runner())
    return tuple(results)

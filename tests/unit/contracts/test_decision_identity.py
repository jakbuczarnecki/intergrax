# © Artur Czarnecki. All rights reserved.

import re

import pytest

from intergrax.contracts.decision_identity import (
    DecisionExecutionLineage,
    DecisionIdentity,
    DecisionScope,
    DecisionVersion,
    initial_decision_version,
    mint_decision_id,
    next_decision_version,
    validate_decision_id,
    validate_decision_tenant_id,
    validate_decision_version,
)
from intergrax.contracts.execution_identity import (
    mint_attempt_id,
    mint_execution_id,
    mint_run_id,
    mint_task_id,
)

_CANONICAL_DECISION_ID = re.compile(r"^decision_[0-9a-f]{32}$")


def _lineage(*, with_execution_id: bool = True) -> DecisionExecutionLineage:
    return DecisionExecutionLineage(
        task_id=mint_task_id(),
        run_id=mint_run_id(),
        attempt_id=mint_attempt_id(),
        execution_id=mint_execution_id() if with_execution_id else None,
    )


def _identity(
    *,
    tenant_id: str = "tenant-a",
    namespace: str = "incident",
    subject: str = "incident-123",
    version: DecisionVersion | None = None,
    with_execution_id: bool = True,
) -> DecisionIdentity:
    return DecisionIdentity(
        decision_id=mint_decision_id(),
        version=version or initial_decision_version(),
        scope=DecisionScope(namespace=namespace, subject=subject),
        tenant_id=tenant_id,
        execution=_lineage(with_execution_id=with_execution_id),
    )


@pytest.mark.unit
@pytest.mark.gate
def test_mint_decision_id_produces_valid_canonical_id() -> None:
    value = mint_decision_id()
    assert validate_decision_id(value) == value
    assert _CANONICAL_DECISION_ID.fullmatch(value)


@pytest.mark.unit
@pytest.mark.gate
def test_mint_decision_id_values_differ() -> None:
    first = mint_decision_id()
    second = mint_decision_id()
    assert first != second


@pytest.mark.unit
@pytest.mark.gate
def test_validate_decision_id_round_trip() -> None:
    value = mint_decision_id()
    assert validate_decision_id(str(value)) == value


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "value",
    [
        "",
        "   ",
        "decision_",
        "decision_tooshort",
        "decision_" + "g" * 32,
        "task_" + "a" * 32,
        "DECISION_" + "a" * 32,
        " decision_" + "a" * 32,
        "decision_" + "a" * 32 + " ",
    ],
)
def test_validate_decision_id_rejects_malformed(value: str) -> None:
    with pytest.raises((ValueError, TypeError)):
        validate_decision_id(value)


@pytest.mark.unit
@pytest.mark.gate
def test_validate_decision_id_rejects_non_string() -> None:
    with pytest.raises(TypeError):
        validate_decision_id(123)
    with pytest.raises(TypeError):
        validate_decision_id(object())


@pytest.mark.unit
@pytest.mark.gate
def test_validate_decision_id_rejects_uppercase_hex() -> None:
    decision_id = mint_decision_id()
    upper = "decision_" + decision_id.split("_", 1)[1].upper()
    with pytest.raises(ValueError, match="suffix"):
        validate_decision_id(upper)


@pytest.mark.unit
@pytest.mark.gate
def test_initial_decision_version_valid() -> None:
    version = initial_decision_version()
    assert version.value == 1
    assert validate_decision_version(version) == 1


@pytest.mark.unit
@pytest.mark.gate
def test_decision_version_one_valid() -> None:
    version = DecisionVersion(1)
    assert version.value == 1


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("value", [0, -1, -99])
def test_validate_decision_version_rejects_non_positive(value: int) -> None:
    with pytest.raises(ValueError):
        validate_decision_version(value)
    with pytest.raises(ValueError):
        DecisionVersion(value)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("value", ["1", 1.0, True, None, object()])
def test_validate_decision_version_rejects_malformed_type(value: object) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_decision_version(value)


@pytest.mark.unit
@pytest.mark.gate
def test_next_decision_version_increments() -> None:
    current = DecisionVersion(2)
    assert next_decision_version(current).value == 3


@pytest.mark.unit
@pytest.mark.gate
def test_decision_scope_accepts_valid_namespace_and_subject() -> None:
    scope = DecisionScope(namespace="incident", subject="incident-123")
    assert scope.namespace == "incident"
    assert scope.subject == "incident-123"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    ("namespace", "subject"),
    [
        ("", "incident-123"),
        ("incident", ""),
        ("   ", "incident-123"),
        ("incident", "   "),
        (" incident", "incident-123"),
        ("incident", "incident-123 "),
    ],
)
def test_decision_scope_rejects_invalid_fields(namespace: str, subject: str) -> None:
    with pytest.raises(ValueError):
        DecisionScope(namespace=namespace, subject=subject)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_scope_is_immutable() -> None:
    scope = DecisionScope(namespace="incident", subject="incident-123")
    with pytest.raises(AttributeError):
        scope.namespace = "other"


@pytest.mark.unit
@pytest.mark.gate
def test_decision_execution_lineage_accepts_canonical_ids() -> None:
    lineage = _lineage(with_execution_id=True)
    assert lineage.execution_id is not None


@pytest.mark.unit
@pytest.mark.gate
def test_decision_execution_lineage_without_execution_id() -> None:
    lineage = _lineage(with_execution_id=False)
    assert lineage.execution_id is None


@pytest.mark.unit
@pytest.mark.gate
def test_decision_execution_lineage_rejects_malformed_task_id() -> None:
    with pytest.raises(ValueError):
        DecisionExecutionLineage(
            task_id="task_bad",
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_decision_execution_lineage_rejects_malformed_execution_id() -> None:
    with pytest.raises(ValueError):
        DecisionExecutionLineage(
            task_id=mint_task_id(),
            run_id=mint_run_id(),
            attempt_id=mint_attempt_id(),
            execution_id="exec_bad",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_decision_identity_valid_aggregate() -> None:
    identity = _identity()
    assert identity.tenant_id == "tenant-a"
    assert identity.scope.namespace == "incident"
    assert identity.scope.subject == "incident-123"


@pytest.mark.unit
@pytest.mark.gate
def test_validate_decision_tenant_id_accepts_valid_tenant() -> None:
    assert validate_decision_tenant_id("tenant-a") == "tenant-a"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("tenant_id", ["", "   ", " tenant-a", "tenant-a "])
def test_validate_decision_tenant_id_rejects_invalid(tenant_id: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_tenant_id(tenant_id)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("value", [123, None, object()])
def test_validate_decision_tenant_id_rejects_non_string(value: object) -> None:
    with pytest.raises(TypeError):
        validate_decision_tenant_id(value)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("tenant_id", ["", "   "])
def test_decision_identity_rejects_empty_tenant(tenant_id: str) -> None:
    with pytest.raises(ValueError):
        _identity(tenant_id=tenant_id)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_identity_retains_exact_scope() -> None:
    identity = _identity(namespace="contract_review", subject="contract-456")
    assert identity.scope.namespace == "contract_review"
    assert identity.scope.subject == "contract-456"


@pytest.mark.unit
@pytest.mark.gate
def test_decision_identity_is_frozen() -> None:
    identity = _identity()
    with pytest.raises(AttributeError):
        identity.tenant_id = "other"


@pytest.mark.unit
@pytest.mark.gate
def test_decision_identity_differs_by_version() -> None:
    base = _identity()
    other = DecisionIdentity(
        decision_id=base.decision_id,
        version=next_decision_version(base.version),
        scope=base.scope,
        tenant_id=base.tenant_id,
        execution=base.execution,
    )
    assert base != other


@pytest.mark.unit
@pytest.mark.gate
def test_decision_identity_differs_by_tenant_or_scope() -> None:
    base = _identity()
    other_tenant = DecisionIdentity(
        decision_id=base.decision_id,
        version=base.version,
        scope=base.scope,
        tenant_id="tenant-b",
        execution=base.execution,
    )
    other_scope = DecisionIdentity(
        decision_id=base.decision_id,
        version=base.version,
        scope=DecisionScope(namespace="incident", subject="incident-999"),
        tenant_id=base.tenant_id,
        execution=base.execution,
    )
    assert base != other_tenant
    assert base != other_scope


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_tenant_substitution_changes_identity() -> None:
    trusted = _identity(tenant_id="tenant-trusted")
    substituted = DecisionIdentity(
        decision_id=trusted.decision_id,
        version=trusted.version,
        scope=trusted.scope,
        tenant_id="tenant-attacker",
        execution=trusted.execution,
    )
    assert trusted != substituted


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_scope_substitution_changes_identity() -> None:
    trusted = _identity(namespace="incident", subject="incident-123")
    substituted = DecisionIdentity(
        decision_id=trusted.decision_id,
        version=trusted.version,
        scope=DecisionScope(namespace="incident", subject="incident-999"),
        tenant_id=trusted.tenant_id,
        execution=trusted.execution,
    )
    assert trusted != substituted


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_version_substitution_changes_identity() -> None:
    trusted = _identity(version=DecisionVersion(1))
    substituted = DecisionIdentity(
        decision_id=trusted.decision_id,
        version=DecisionVersion(2),
        scope=trusted.scope,
        tenant_id=trusted.tenant_id,
        execution=trusted.execution,
    )
    assert trusted != substituted


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_malformed_execution_identity_rejected() -> None:
    lineage = _lineage()
    with pytest.raises(ValueError):
        DecisionIdentity(
            decision_id=mint_decision_id(),
            version=initial_decision_version(),
            scope=DecisionScope(namespace="incident", subject="incident-123"),
            tenant_id="tenant-a",
            execution=DecisionExecutionLineage(
                task_id=lineage.task_id,
                run_id="run_bad",
                attempt_id=lineage.attempt_id,
                execution_id=lineage.execution_id,
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_decision_contracts_validate_without_post_construction_mutation() -> None:
    """Decision contracts validate but do not perform post-construction mutation."""
    decision_id = mint_decision_id()
    version = DecisionVersion(2)
    namespace = "incident"
    subject = "incident-123"
    tenant_id = "tenant-a"
    task_id = mint_task_id()
    run_id = mint_run_id()
    attempt_id = mint_attempt_id()
    execution_id = mint_execution_id()

    scope = DecisionScope(namespace=namespace, subject=subject)
    assert scope.namespace is namespace
    assert scope.subject is subject
    with pytest.raises(AttributeError):
        scope.namespace = "other"

    lineage = DecisionExecutionLineage(
        task_id=task_id,
        run_id=run_id,
        attempt_id=attempt_id,
        execution_id=execution_id,
    )
    assert lineage.task_id is task_id
    assert lineage.run_id is run_id
    assert lineage.attempt_id is attempt_id
    assert lineage.execution_id is execution_id
    with pytest.raises(AttributeError):
        lineage.task_id = mint_task_id()

    version_obj = DecisionVersion(version.value)
    assert version_obj.value == version.value
    with pytest.raises(AttributeError):
        version_obj.value = 3

    identity = DecisionIdentity(
        decision_id=decision_id,
        version=version,
        scope=scope,
        tenant_id=tenant_id,
        execution=lineage,
    )
    assert identity.decision_id is decision_id
    assert identity.version is version
    assert identity.scope is scope
    assert identity.tenant_id is tenant_id
    assert identity.execution is lineage
    with pytest.raises(AttributeError):
        identity.tenant_id = "other"

    with pytest.raises(ValueError):
        DecisionScope(namespace=" incident", subject=subject)
    with pytest.raises(ValueError):
        DecisionVersion(0)
    with pytest.raises(ValueError):
        validate_decision_id("decision_bad")


@pytest.mark.unit
@pytest.mark.gate
def test_adversarial_implicit_default_tenant_rejected() -> None:
    with pytest.raises(ValueError):
        DecisionIdentity(
            decision_id=mint_decision_id(),
            version=initial_decision_version(),
            scope=DecisionScope(namespace="incident", subject="incident-123"),
            tenant_id="",
            execution=_lineage(),
        )

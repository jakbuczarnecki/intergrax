# © Artur Czarnecki. All rights reserved.

from dataclasses import dataclass

import pytest

from intergrax.contracts.decision_artifact_registry import (
    DecisionArtifactKindAlreadyRegisteredError,
    DecisionArtifactKindNotRegisteredError,
    DecisionArtifactKindRegistry,
    decision_artifact_kind_registry,
    is_decision_artifact_kind_registered,
    register_decision_artifact_kind,
    require_registered_decision_artifact_kind,
)
from intergrax.contracts.decision_record import (
    DecisionArtifact,
    DecisionArtifactKind,
    validate_decision_artifact_kind,
)


@dataclass(frozen=True, slots=True)
class LegacyPayload:
    note: str


@pytest.mark.unit
@pytest.mark.gate
def test_empty_registry_valid() -> None:
    registry = decision_artifact_kind_registry()
    assert registry.kinds == ()


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_empty_registry_valid() -> None:
    registry = DecisionArtifactKindRegistry()
    assert registry.kinds == ()


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_canonical_kinds_valid() -> None:
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")
    registry = DecisionArtifactKindRegistry(kinds=(kind_a, kind_b))
    assert registry.kinds == (kind_a, kind_b)


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_invalid_empty_kind_rejected() -> None:
    with pytest.raises(ValueError):
        DecisionArtifactKindRegistry(kinds=(DecisionArtifactKind(""),))


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_duplicate_rejected() -> None:
    kind = validate_decision_artifact_kind("alpha")
    with pytest.raises(DecisionArtifactKindAlreadyRegisteredError):
        DecisionArtifactKindRegistry(kinds=(kind, kind))


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_noncanonical_order_rejected() -> None:
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")
    with pytest.raises(ValueError, match="canonical order"):
        DecisionArtifactKindRegistry(kinds=(kind_b, kind_a))


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_cannot_bypass_factory_invariants() -> None:
    """Direct construction must enforce the same invariants as the factory."""
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")

    with pytest.raises(ValueError):
        DecisionArtifactKindRegistry(kinds=(DecisionArtifactKind(""),))

    with pytest.raises(DecisionArtifactKindAlreadyRegisteredError):
        DecisionArtifactKindRegistry(kinds=(kind_a, kind_a))

    with pytest.raises(ValueError, match="canonical order"):
        DecisionArtifactKindRegistry(kinds=(kind_b, kind_a))

    canonical = DecisionArtifactKindRegistry(kinds=(kind_a, kind_b))
    assert canonical.kinds == (kind_a, kind_b)

    factory_canonicalized = decision_artifact_kind_registry((kind_b, kind_a))
    assert factory_canonicalized.kinds == (kind_a, kind_b)


@pytest.mark.unit
@pytest.mark.gate
def test_one_valid_kind() -> None:
    kind = validate_decision_artifact_kind("incident_resolution")
    registry = decision_artifact_kind_registry((kind,))
    assert registry.kinds == (kind,)


@pytest.mark.unit
@pytest.mark.gate
def test_multiple_kinds_valid() -> None:
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")
    registry = decision_artifact_kind_registry((kind_a, kind_b))
    assert registry.kinds == (kind_a, kind_b)


@pytest.mark.unit
@pytest.mark.gate
def test_input_order_canonicalized() -> None:
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")
    first = decision_artifact_kind_registry((kind_b, kind_a))
    second = decision_artifact_kind_registry((kind_a, kind_b))
    assert first == second
    assert first.kinds == (kind_a, kind_b)


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_construction_rejected() -> None:
    kind = validate_decision_artifact_kind("alpha")
    with pytest.raises(DecisionArtifactKindAlreadyRegisteredError):
        decision_artifact_kind_registry((kind, kind))


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_registration_rejected() -> None:
    kind = validate_decision_artifact_kind("alpha")
    registry = decision_artifact_kind_registry((kind,))
    with pytest.raises(DecisionArtifactKindAlreadyRegisteredError):
        register_decision_artifact_kind(registry, kind)


@pytest.mark.unit
@pytest.mark.gate
def test_registration_returns_new_registry() -> None:
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")
    registry1 = decision_artifact_kind_registry((kind_a,))
    registry2 = register_decision_artifact_kind(registry1, kind_b)
    assert registry2 is not registry1
    assert registry2.kinds == (kind_a, kind_b)


@pytest.mark.unit
@pytest.mark.gate
def test_old_registry_unchanged_after_registration() -> None:
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")
    registry1 = decision_artifact_kind_registry((kind_a,))
    _ = register_decision_artifact_kind(registry1, kind_b)
    assert registry1.kinds == (kind_a,)


@pytest.mark.unit
@pytest.mark.gate
def test_membership_true_for_registered_kind() -> None:
    kind = validate_decision_artifact_kind("known")
    registry = decision_artifact_kind_registry((kind,))
    assert is_decision_artifact_kind_registered(registry, kind) is True


@pytest.mark.unit
@pytest.mark.gate
def test_membership_false_for_unknown_valid_kind() -> None:
    known = validate_decision_artifact_kind("known")
    unknown = validate_decision_artifact_kind("unknown")
    registry = decision_artifact_kind_registry((known,))
    assert is_decision_artifact_kind_registered(registry, unknown) is False


@pytest.mark.unit
@pytest.mark.gate
def test_require_registered_returns_exact_typed_kind() -> None:
    kind = validate_decision_artifact_kind("known")
    registry = decision_artifact_kind_registry((kind,))
    resolved = require_registered_decision_artifact_kind(registry, kind)
    assert resolved == kind
    assert type(resolved) is str


@pytest.mark.unit
@pytest.mark.gate
def test_require_unknown_raises() -> None:
    known = validate_decision_artifact_kind("known")
    unknown = validate_decision_artifact_kind("unknown")
    registry = decision_artifact_kind_registry((known,))
    with pytest.raises(DecisionArtifactKindNotRegisteredError):
        require_registered_decision_artifact_kind(registry, unknown)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", ["", "   "])
def test_invalid_empty_kind_rejected(kind: str) -> None:
    with pytest.raises(ValueError):
        decision_artifact_kind_registry((DecisionArtifactKind(kind),))


@pytest.mark.unit
@pytest.mark.gate
def test_whitespace_only_kind_rejected() -> None:
    with pytest.raises(ValueError):
        decision_artifact_kind_registry(
            (validate_decision_artifact_kind("valid"), DecisionArtifactKind("   ")),
        )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", [" alpha", "alpha "])
def test_surrounding_whitespace_rejected(kind: str) -> None:
    with pytest.raises(ValueError):
        decision_artifact_kind_registry((DecisionArtifactKind(kind),))


@pytest.mark.unit
@pytest.mark.gate
def test_wrong_kind_type_rejected() -> None:
    registry = decision_artifact_kind_registry()
    with pytest.raises(TypeError):
        is_decision_artifact_kind_registered(registry, 42)


@pytest.mark.unit
@pytest.mark.gate
def test_case_remains_exact_and_case_sensitive() -> None:
    lower = validate_decision_artifact_kind("incident")
    upper = validate_decision_artifact_kind("Incident")
    registry = decision_artifact_kind_registry((lower,))
    assert is_decision_artifact_kind_registered(registry, lower) is True
    assert is_decision_artifact_kind_registered(registry, upper) is False


@pytest.mark.unit
@pytest.mark.gate
def test_registry_immutable() -> None:
    kind = validate_decision_artifact_kind("alpha")
    registry = decision_artifact_kind_registry((kind,))
    with pytest.raises(AttributeError):
        setattr(registry, "kinds", ())


@pytest.mark.unit
@pytest.mark.gate
def test_syntactic_validity_distinct_from_registration() -> None:
    known = validate_decision_artifact_kind("known")
    unknown = validate_decision_artifact_kind("unknown")
    assert validate_decision_artifact_kind("known") == known
    assert validate_decision_artifact_kind("unknown") == unknown
    registry = decision_artifact_kind_registry((known,))
    require_registered_decision_artifact_kind(registry, known)
    with pytest.raises(DecisionArtifactKindNotRegisteredError):
        require_registered_decision_artifact_kind(registry, unknown)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_artifact_constructible_without_registry() -> None:
    artifact = DecisionArtifact(
        kind=validate_decision_artifact_kind("legacy-known-at-the-time"),
        content=LegacyPayload(note="historical"),
    )
    assert artifact.kind == "legacy-known-at-the-time"


@pytest.mark.unit
@pytest.mark.gate
def test_immutable_extension() -> None:
    kind_a = validate_decision_artifact_kind("alpha")
    kind_b = validate_decision_artifact_kind("beta")
    registry1 = decision_artifact_kind_registry((kind_a,))
    registry2 = register_decision_artifact_kind(registry1, kind_b)
    assert registry1.kinds == (kind_a,)
    assert registry2.kinds == (kind_a, kind_b)
    assert registry1 is not registry2

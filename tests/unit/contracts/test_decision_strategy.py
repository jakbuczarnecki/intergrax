# © Artur Czarnecki. All rights reserved.

from dataclasses import FrozenInstanceError, dataclass

import pytest

from intergrax.contracts.decision_strategy import (
    DecisionStrategy,
    DecisionStrategyAlreadyRegisteredError,
    DecisionStrategyKind,
    DecisionStrategyNotRegisteredError,
    DecisionStrategyRegistration,
    DecisionStrategyRegistry,
    decision_strategy_registry,
    is_decision_strategy_registered,
    register_decision_strategy,
    require_registered_decision_strategy,
    validate_decision_strategy_kind,
)


@dataclass(frozen=True, slots=True)
class AlphaStrategy:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("alpha")


@dataclass(frozen=True, slots=True)
class BetaStrategy:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("beta")


@dataclass(frozen=True, slots=True)
class MismatchKindStrategy:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("other")


def _registration(
    kind: str,
    strategy: DecisionStrategy,
) -> DecisionStrategyRegistration:
    return DecisionStrategyRegistration(
        kind=validate_decision_strategy_kind(kind),
        strategy=strategy,
    )


@pytest.mark.unit
@pytest.mark.gate
def test_valid_strategy_kind() -> None:
    kind = validate_decision_strategy_kind("single_model")
    assert kind == "single_model"
    assert type(kind) is str


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", ["", "   "])
def test_invalid_empty_kind_rejected(kind: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_strategy_kind(kind)


@pytest.mark.unit
@pytest.mark.gate
def test_whitespace_only_kind_rejected() -> None:
    with pytest.raises(ValueError):
        validate_decision_strategy_kind("   ")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("kind", [" alpha", "alpha "])
def test_surrounding_whitespace_rejected(kind: str) -> None:
    with pytest.raises(ValueError):
        validate_decision_strategy_kind(kind)


@pytest.mark.unit
@pytest.mark.gate
def test_wrong_kind_type_rejected() -> None:
    with pytest.raises(TypeError):
        validate_decision_strategy_kind(42)


@pytest.mark.unit
@pytest.mark.gate
def test_empty_registry_valid() -> None:
    registry = decision_strategy_registry()
    assert registry.registrations == ()


@pytest.mark.unit
@pytest.mark.gate
def test_one_strategy_registration() -> None:
    registration = _registration("alpha", AlphaStrategy())
    registry = decision_strategy_registry((registration,))
    assert registry.registrations == (registration,)


@pytest.mark.unit
@pytest.mark.gate
def test_multiple_registrations() -> None:
    alpha = _registration("alpha", AlphaStrategy())
    beta = _registration("beta", BetaStrategy())
    registry = decision_strategy_registry((alpha, beta))
    assert registry.registrations == (alpha, beta)


@pytest.mark.unit
@pytest.mark.gate
def test_deterministic_ordering() -> None:
    alpha = _registration("alpha", AlphaStrategy())
    beta = _registration("beta", BetaStrategy())
    first = decision_strategy_registry((beta, alpha))
    second = decision_strategy_registry((alpha, beta))
    assert first == second
    assert first.registrations == (alpha, beta)


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_registration_rejected() -> None:
    registration = _registration("alpha", AlphaStrategy())
    registry = decision_strategy_registry((registration,))
    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        register_decision_strategy(registry, registration)


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_construction_rejected() -> None:
    registration = _registration("alpha", AlphaStrategy())
    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        decision_strategy_registry((registration, registration))


@pytest.mark.unit
@pytest.mark.gate
def test_registration_returns_new_registry() -> None:
    alpha = _registration("alpha", AlphaStrategy())
    beta = _registration("beta", BetaStrategy())
    registry1 = decision_strategy_registry((alpha,))
    registry2 = register_decision_strategy(registry1, beta)
    assert registry2 is not registry1
    assert registry2.registrations == (alpha, beta)


@pytest.mark.unit
@pytest.mark.gate
def test_old_registry_unchanged_after_registration() -> None:
    alpha = _registration("alpha", AlphaStrategy())
    beta = _registration("beta", BetaStrategy())
    registry1 = decision_strategy_registry((alpha,))
    _ = register_decision_strategy(registry1, beta)
    assert registry1.registrations == (alpha,)


@pytest.mark.unit
@pytest.mark.gate
def test_membership_true_for_registered_kind() -> None:
    registration = _registration("alpha", AlphaStrategy())
    registry = decision_strategy_registry((registration,))
    assert is_decision_strategy_registered(registry, registration.kind) is True


@pytest.mark.unit
@pytest.mark.gate
def test_membership_false_for_unknown_valid_kind() -> None:
    known = _registration("alpha", AlphaStrategy())
    unknown = validate_decision_strategy_kind("unknown")
    registry = decision_strategy_registry((known,))
    assert is_decision_strategy_registered(registry, unknown) is False


@pytest.mark.unit
@pytest.mark.gate
def test_require_registered_returns_exact_strategy() -> None:
    registration = _registration("alpha", AlphaStrategy())
    registry = decision_strategy_registry((registration,))
    resolved = require_registered_decision_strategy(registry, registration.kind)
    assert resolved is registration.strategy
    assert resolved.kind == registration.kind


@pytest.mark.unit
@pytest.mark.gate
def test_require_unknown_raises() -> None:
    known = _registration("alpha", AlphaStrategy())
    unknown = validate_decision_strategy_kind("unknown")
    registry = decision_strategy_registry((known,))
    with pytest.raises(DecisionStrategyNotRegisteredError):
        require_registered_decision_strategy(registry, unknown)


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_empty_registry_valid() -> None:
    registry = DecisionStrategyRegistry()
    assert registry.registrations == ()


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_canonical_registrations_valid() -> None:
    alpha = _registration("alpha", AlphaStrategy())
    beta = _registration("beta", BetaStrategy())
    registry = DecisionStrategyRegistry(registrations=(alpha, beta))
    assert registry.registrations == (alpha, beta)


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_duplicate_rejected() -> None:
    registration = _registration("alpha", AlphaStrategy())
    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        DecisionStrategyRegistry(registrations=(registration, registration))


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_invalid_kind_rejected() -> None:
    with pytest.raises(ValueError):
        DecisionStrategyRegistration(
            kind=DecisionStrategyKind(""),
            strategy=AlphaStrategy(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_noncanonical_order_rejected() -> None:
    alpha = _registration("alpha", AlphaStrategy())
    beta = _registration("beta", BetaStrategy())
    with pytest.raises(ValueError, match="canonical order"):
        DecisionStrategyRegistry(registrations=(beta, alpha))


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_cannot_bypass_factory_invariants() -> None:
    alpha = _registration("alpha", AlphaStrategy())
    beta = _registration("beta", BetaStrategy())

    with pytest.raises(ValueError):
        DecisionStrategyRegistration(
            kind=DecisionStrategyKind(""),
            strategy=AlphaStrategy(),
        )

    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        DecisionStrategyRegistry(registrations=(alpha, alpha))

    with pytest.raises(ValueError, match="canonical order"):
        DecisionStrategyRegistry(registrations=(beta, alpha))

    canonical = DecisionStrategyRegistry(registrations=(alpha, beta))
    assert canonical.registrations == (alpha, beta)

    factory_canonicalized = decision_strategy_registry((beta, alpha))
    assert factory_canonicalized.registrations == (alpha, beta)


@pytest.mark.unit
@pytest.mark.gate
def test_registry_immutable() -> None:
    registration = _registration("alpha", AlphaStrategy())
    registry = decision_strategy_registry((registration,))
    with pytest.raises((AttributeError, FrozenInstanceError)):
        setattr(registry, "registrations", ())


@pytest.mark.unit
@pytest.mark.gate
def test_case_sensitive_strategy_identities() -> None:
    lower = validate_decision_strategy_kind("alpha")
    upper = validate_decision_strategy_kind("Alpha")
    registration = _registration("alpha", AlphaStrategy())
    registry = decision_strategy_registry((registration,))
    assert is_decision_strategy_registered(registry, lower) is True
    assert is_decision_strategy_registered(registry, upper) is False


@pytest.mark.unit
@pytest.mark.gate
def test_registration_kind_must_match_strategy_kind() -> None:
    with pytest.raises(ValueError, match="must match strategy.kind"):
        DecisionStrategyRegistration(
            kind=validate_decision_strategy_kind("alpha"),
            strategy=MismatchKindStrategy(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_no_default_strategy_semantics() -> None:
    import intergrax.contracts.decision_strategy as module

    registry = decision_strategy_registry()
    assert registry.registrations == ()
    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "GLOBAL_DEFAULT" not in source
    assert "CURRENT_STRATEGY" not in source
    assert "set_default_strategy" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_no_dependency_on_execution_strategy() -> None:
    import intergrax.contracts.decision_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "ExecutionStrategy" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_no_dependency_on_nexus() -> None:
    import intergrax.contracts.decision_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "nexus" not in source.lower()


@pytest.mark.unit
@pytest.mark.gate
def test_structural_protocol_without_inheritance() -> None:
    @dataclass(frozen=True, slots=True)
    class LocalFakeStrategy:
        kind: DecisionStrategyKind = validate_decision_strategy_kind("local_fake")

    strategy = LocalFakeStrategy()
    assert isinstance(strategy, DecisionStrategy)


@pytest.mark.unit
@pytest.mark.gate
def test_platform_reuse_boundary_no_discovery_imports() -> None:
    import intergrax.contracts.decision_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "intergrax.core.plugins.discovery" not in source
    assert "importlib" not in source
    assert "entry_points" not in source

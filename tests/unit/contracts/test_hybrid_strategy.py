# © Artur Czarnecki. All rights reserved.

from dataclasses import FrozenInstanceError, dataclass, fields

import pytest

from intergrax.contracts.decision_strategy import (
    DecisionStrategy,
    DecisionStrategyAlreadyRegisteredError,
    DecisionStrategyKind,
    DecisionStrategyNotRegisteredError,
    DecisionStrategyRegistration,
    DecisionStrategyRegistry,
    RegistryBoundDecisionStrategy,
    decision_strategy_registry,
    is_decision_strategy_registered,
    register_decision_strategy,
    require_registered_decision_strategy,
    validate_decision_strategy_kind,
)
from intergrax.contracts.hybrid_strategy import (
    HybridPhase,
    HybridStrategy,
    _validate_hybrid_phases,
    hybrid_phase,
    hybrid_strategy,
    hybrid_strategy_kind,
    hybrid_strategy_registration,
    register_hybrid_strategy,
    validate_hybrid_phase_id,
    validate_hybrid_strategy_registry_bindings,
)
from intergrax.contracts.rule_based_strategy import (
    rule_based_strategy_kind,
    rule_based_strategy_registration,
)
from intergrax.contracts.single_model_strategy import (
    single_model_strategy_kind,
    single_model_strategy_registration,
)
from intergrax.runtime.execution.inference_profile import validate_inference_profile_id
from intergrax.contracts.single_model_strategy import SingleModelInferenceConfiguration

_START_HEAD = "a3034995234a63997da3e11b5b3cabe061bbfd2f"

_FORBIDDEN_FIELD_NAMES = frozenset(
    {
        "executor",
        "runtime",
        "scheduler",
        "retry",
        "checkpoint",
        "budget",
        "parallel",
        "condition",
        "branch",
        "loop",
        "authorization",
        "verification",
        "finalization",
        "adapter",
        "provider",
        "model",
        "prompt",
        "messages",
    },
)

_FORBIDDEN_SOURCE_TOKENS = frozenset(
    {
        "Any",
        "cast(",
        "type: ignore",
        "pyright: ignore",
        "getattr",
        "setattr",
        "hasattr",
        "inspect",
        "exec(",
        "eval(",
        "object.__setattr__",
        "dict[str, Any]",
        "ExecutionStrategy",
        "INFERENCE",
        "AGENTIC",
        "ORCHESTRATION",
        "LLMAdapter",
        "InferenceProfileId",
        "nexus",
    },
)

_HARDCODED_STRATEGY_KINDS = frozenset(
    {
        "single_model",
        "rule_based",
        "council",
    },
)


@dataclass(frozen=True, slots=True)
class RuleFixtureInput:
    value: int


@dataclass(frozen=True, slots=True)
class RuleFixtureOutput:
    value: int


@dataclass(frozen=True, slots=True)
class DomainCustomStrategy:
    kind: DecisionStrategyKind = validate_decision_strategy_kind("domain_custom")


@dataclass(frozen=True, slots=True)
class _NoOpRules:
    def evaluate(
        self,
        decision_input: RuleFixtureInput,
    ) -> RuleFixtureOutput:
        return RuleFixtureOutput(value=decision_input.value)


def _single_model_registration() -> DecisionStrategyRegistration:
    return single_model_strategy_registration(
        SingleModelInferenceConfiguration(
            inference_profile_id=validate_inference_profile_id("primary"),
        ),
    )


def _component_registry() -> DecisionStrategyRegistry:
    return decision_strategy_registry(
        (
            rule_based_strategy_registration(_NoOpRules()),
            _single_model_registration(),
        ),
    )


def _two_phase_hybrid() -> tuple[HybridPhase, ...]:
    return (
        hybrid_phase(phase_id="precheck", strategy_kind=rule_based_strategy_kind()),
        hybrid_phase(phase_id="proposal", strategy_kind=single_model_strategy_kind()),
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "phase_id",
    ["precheck", "analiza", "weryfikacja-domenowa", "法律审核"],
)
def test_valid_hybrid_phase_id(phase_id: str) -> None:
    validated = validate_hybrid_phase_id(phase_id)
    assert validated == phase_id
    assert type(validated) is str


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("phase_id", ["", "   ", " phase "])
def test_invalid_hybrid_phase_id_rejected(phase_id: str) -> None:
    with pytest.raises(ValueError):
        validate_hybrid_phase_id(phase_id)


@pytest.mark.unit
@pytest.mark.gate
def test_invalid_hybrid_phase_id_non_str_rejected() -> None:
    with pytest.raises(TypeError):
        validate_hybrid_phase_id(42)


@pytest.mark.unit
@pytest.mark.gate
def test_basic_two_phase_hybrid_strategy() -> None:
    strategy = hybrid_strategy(phases=_two_phase_hybrid())
    assert len(strategy.phases) == 2
    assert strategy.phases[0].phase_id == "precheck"
    assert strategy.phases[1].phase_id == "proposal"


@pytest.mark.unit
@pytest.mark.gate
def test_phase_order_preserved() -> None:
    phases = (
        hybrid_phase(phase_id="z-phase", strategy_kind=rule_based_strategy_kind()),
        hybrid_phase(phase_id="a-phase", strategy_kind=single_model_strategy_kind()),
    )
    strategy = hybrid_strategy(phases=phases)
    assert tuple(phase.phase_id for phase in strategy.phases) == ("z-phase", "a-phase")


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_phase_id_rejected_direct() -> None:
    phases = (
        hybrid_phase(phase_id="phase-A", strategy_kind=rule_based_strategy_kind()),
        hybrid_phase(phase_id="phase-A", strategy_kind=single_model_strategy_kind()),
    )
    with pytest.raises(ValueError, match="duplicate phase_id"):
        HybridStrategy(phases=phases)


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_phase_id_rejected_factory() -> None:
    phases = (
        hybrid_phase(phase_id="phase-A", strategy_kind=rule_based_strategy_kind()),
        hybrid_phase(phase_id="phase-A", strategy_kind=single_model_strategy_kind()),
    )
    with pytest.raises(ValueError, match="duplicate phase_id"):
        hybrid_strategy(phases=phases)


@pytest.mark.unit
@pytest.mark.gate
def test_same_strategy_kind_reused_across_phases() -> None:
    phases = (
        hybrid_phase(phase_id="phase-A", strategy_kind=single_model_strategy_kind()),
        hybrid_phase(phase_id="phase-B", strategy_kind=single_model_strategy_kind()),
    )
    strategy = hybrid_strategy(phases=phases)
    assert strategy.phases[0].strategy_kind == strategy.phases[1].strategy_kind


@pytest.mark.unit
@pytest.mark.gate
def test_self_hybrid_reference_rejected_direct() -> None:
    with pytest.raises(ValueError, match="must not reference hybrid"):
        HybridPhase(
            phase_id=validate_hybrid_phase_id("phase-A"),
            strategy_kind=hybrid_strategy_kind(),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_self_hybrid_reference_rejected_strategy_direct() -> None:
    with pytest.raises(ValueError, match="must not reference hybrid"):
        HybridStrategy(
            phases=(
                HybridPhase(
                    phase_id=validate_hybrid_phase_id("phase-A"),
                    strategy_kind=hybrid_strategy_kind(),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_self_hybrid_reference_rejected_factory() -> None:
    with pytest.raises(ValueError, match="must not reference hybrid"):
        hybrid_phase(phase_id="phase-A", strategy_kind=hybrid_strategy_kind())


@pytest.mark.unit
@pytest.mark.gate
def test_unknown_strategy_kind_rejected_at_registry_validation() -> None:
    strategy = hybrid_strategy(
        phases=(
            hybrid_phase(phase_id="phase", strategy_kind="missing_strategy"),
        ),
    )
    registry = _component_registry()
    with pytest.raises(DecisionStrategyNotRegisteredError):
        validate_hybrid_strategy_registry_bindings(strategy=strategy, registry=registry)


@pytest.mark.unit
@pytest.mark.gate
def test_custom_registered_strategy_accepted() -> None:
    registry = decision_strategy_registry(
        (
            rule_based_strategy_registration(_NoOpRules()),
            DecisionStrategyRegistration(
                kind=validate_decision_strategy_kind("domain_custom"),
                strategy=DomainCustomStrategy(),
            ),
        ),
    )
    phases = (
        hybrid_phase(phase_id="review", strategy_kind="domain_custom"),
    )
    strategy = hybrid_strategy(phases=phases)
    validate_hybrid_strategy_registry_bindings(strategy=strategy, registry=registry)
    updated = register_hybrid_strategy(registry, phases=phases)
    resolved = require_registered_decision_strategy(updated, hybrid_strategy_kind())
    assert isinstance(resolved, HybridStrategy)
    assert resolved.phases[0].strategy_kind == "domain_custom"


@pytest.mark.unit
@pytest.mark.gate
def test_hybrid_kind_is_hybrid() -> None:
    assert hybrid_strategy_kind() == "hybrid"


@pytest.mark.unit
@pytest.mark.gate
def test_registration_kind_matches_strategy_kind() -> None:
    phases = _two_phase_hybrid()
    registry = _component_registry()
    registration = hybrid_strategy_registration(phases=phases, registry=registry)
    assert registration.kind == hybrid_strategy_kind()
    assert registration.strategy.kind == hybrid_strategy_kind()


@pytest.mark.unit
@pytest.mark.gate
def test_register_via_canonical_registry() -> None:
    registry = _component_registry()
    phases = _two_phase_hybrid()
    updated = register_hybrid_strategy(registry, phases=phases)
    assert is_decision_strategy_registered(updated, hybrid_strategy_kind()) is True
    resolved = require_registered_decision_strategy(updated, hybrid_strategy_kind())
    assert isinstance(resolved, HybridStrategy)
    assert tuple(phase.phase_id for phase in resolved.phases) == (
        "precheck",
        "proposal",
    )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_hybrid_registration_uses_canonical_duplicate_error() -> None:
    registry = _component_registry()
    phases = _two_phase_hybrid()
    updated = register_hybrid_strategy(registry, phases=phases)
    with pytest.raises(DecisionStrategyAlreadyRegisteredError):
        register_hybrid_strategy(updated, phases=phases)


@pytest.mark.unit
@pytest.mark.gate
def test_register_rejects_unknown_strategy_without_shortcut() -> None:
    registry = _component_registry()
    phases = (
        hybrid_phase(phase_id="phase", strategy_kind="missing_strategy"),
    )
    with pytest.raises(DecisionStrategyNotRegisteredError):
        register_hybrid_strategy(registry, phases=phases)


@pytest.mark.unit
@pytest.mark.gate
def test_generic_registration_rejects_invalid_hybrid_bypass() -> None:
    registry = _component_registry()
    invalid_hybrid = hybrid_strategy(
        phases=(
            hybrid_phase(phase_id="proposal", strategy_kind="missing_strategy"),
        ),
    )
    registration = DecisionStrategyRegistration(
        kind=hybrid_strategy_kind(),
        strategy=invalid_hybrid,
    )
    with pytest.raises(DecisionStrategyNotRegisteredError):
        register_decision_strategy(registry, registration)


@pytest.mark.unit
@pytest.mark.gate
def test_generic_registration_accepts_valid_hybrid() -> None:
    registry = _component_registry()
    valid_hybrid = hybrid_strategy(phases=_two_phase_hybrid())
    registration = DecisionStrategyRegistration(
        kind=hybrid_strategy_kind(),
        strategy=valid_hybrid,
    )
    updated = register_decision_strategy(registry, registration)
    assert is_decision_strategy_registered(updated, hybrid_strategy_kind()) is True
    resolved = require_registered_decision_strategy(updated, hybrid_strategy_kind())
    assert isinstance(resolved, HybridStrategy)
    assert tuple(phase.phase_id for phase in resolved.phases) == (
        "precheck",
        "proposal",
    )


@pytest.mark.unit
@pytest.mark.gate
def test_non_hybrid_phase_element_rejected() -> None:
    @dataclass(frozen=True, slots=True)
    class FakePhase:
        phase_id: str
        strategy_kind: str

    invalid_phases = (
        FakePhase(
            phase_id="phase",
            strategy_kind=rule_based_strategy_kind(),
        ),
    )
    with pytest.raises(TypeError, match="HybridPhase"):
        _validate_hybrid_phases(invalid_phases)


@pytest.mark.unit
@pytest.mark.gate
def test_hybrid_satisfies_registry_bound_protocol() -> None:
    strategy = hybrid_strategy(phases=_two_phase_hybrid())
    assert isinstance(strategy, RegistryBoundDecisionStrategy)


@pytest.mark.unit
@pytest.mark.gate
def test_decision_strategy_module_has_no_hybrid_knowledge() -> None:
    import intergrax.contracts.decision_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert '"hybrid"' not in source
    assert "HybridStrategy" not in source
    assert "HybridPhase" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_registry_validation_without_hybrid_registered() -> None:
    registry = _component_registry()
    phases = _two_phase_hybrid()
    strategy = hybrid_strategy(phases=phases)
    validate_hybrid_strategy_registry_bindings(strategy=strategy, registry=registry)


@pytest.mark.unit
@pytest.mark.gate
def test_empty_phases_rejected_direct() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        HybridStrategy(phases=())


@pytest.mark.unit
@pytest.mark.gate
def test_empty_phases_rejected_factory() -> None:
    with pytest.raises(ValueError, match="must not be empty"):
        hybrid_strategy(phases=())


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_invalid_phase_id_rejected() -> None:
    with pytest.raises(ValueError):
        HybridStrategy(
            phases=(
                HybridPhase(
                    phase_id=validate_hybrid_phase_id(""),
                    strategy_kind=rule_based_strategy_kind(),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_invalid_strategy_kind_rejected() -> None:
    with pytest.raises(ValueError):
        HybridStrategy(
            phases=(
                HybridPhase(
                    phase_id=validate_hybrid_phase_id("phase"),
                    strategy_kind=validate_decision_strategy_kind(""),
                ),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_hybrid_and_phase_immutability() -> None:
    strategy = hybrid_strategy(phases=_two_phase_hybrid())
    with pytest.raises((AttributeError, FrozenInstanceError)):
        setattr(strategy, "kind", hybrid_strategy_kind())
    with pytest.raises((AttributeError, FrozenInstanceError)):
        setattr(strategy.phases[0], "phase_id", validate_hybrid_phase_id("other"))


@pytest.mark.unit
@pytest.mark.gate
def test_hybrid_structural_surface_has_no_forbidden_fields() -> None:
    strategy_fields = {field.name for field in fields(HybridStrategy)}
    phase_fields = {field.name for field in fields(HybridPhase)}
    assert strategy_fields.isdisjoint(_FORBIDDEN_FIELD_NAMES)
    assert phase_fields.isdisjoint(_FORBIDDEN_FIELD_NAMES)


@pytest.mark.unit
@pytest.mark.gate
def test_hybrid_no_forbidden_production_patterns() -> None:
    import intergrax.contracts.hybrid_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    hits = [token for token in _FORBIDDEN_SOURCE_TOKENS if token in source]
    assert hits == []


@pytest.mark.unit
@pytest.mark.gate
def test_hybrid_no_hardcoded_strategy_knowledge() -> None:
    import intergrax.contracts.hybrid_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    for kind in _HARDCODED_STRATEGY_KINDS:
        assert f'"{kind}"' not in source


@pytest.mark.unit
@pytest.mark.gate
def test_hybrid_satisfies_decision_strategy_protocol() -> None:
    strategy = hybrid_strategy(phases=_two_phase_hybrid())
    assert isinstance(strategy, DecisionStrategy)


@pytest.mark.unit
@pytest.mark.gate
def test_no_global_hybrid_state() -> None:
    import intergrax.contracts.hybrid_strategy as module

    source_path = module.__file__
    assert source_path is not None
    source = open(source_path, encoding="utf-8").read()
    assert "HYBRID_REGISTRY" not in source
    assert "HYBRID_PHASES" not in source
    assert "HYBRID_CONFIGS" not in source


@pytest.mark.unit
@pytest.mark.gate
def test_registration_requires_registry_validation() -> None:
    registry = _component_registry()
    phases = (
        hybrid_phase(phase_id="phase", strategy_kind="missing_strategy"),
    )
    with pytest.raises(DecisionStrategyNotRegisteredError):
        hybrid_strategy_registration(phases=phases, registry=registry)


@pytest.mark.unit
@pytest.mark.gate
def test_deterministic_phase_sequence_stored_exactly() -> None:
    phases = (
        hybrid_phase(phase_id="z-phase", strategy_kind=rule_based_strategy_kind()),
        hybrid_phase(phase_id="a-phase", strategy_kind=single_model_strategy_kind()),
        hybrid_phase(phase_id="m-phase", strategy_kind=rule_based_strategy_kind()),
    )
    first = hybrid_strategy(phases=phases)
    second = hybrid_strategy(phases=phases)
    assert first.phases == second.phases
    assert tuple(phase.phase_id for phase in first.phases) == (
        "z-phase",
        "a-phase",
        "m-phase",
    )


@pytest.mark.unit
@pytest.mark.gate
def test_session_start_head_recorded() -> None:
    assert _START_HEAD == "a3034995234a63997da3e11b5b3cabe061bbfd2f"

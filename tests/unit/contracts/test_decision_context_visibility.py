# © Artur Czarnecki. All rights reserved.

import dataclasses
import inspect
from dataclasses import FrozenInstanceError

import pytest

from intergrax.contracts.decision_context_visibility import (
    DeliberationContextId,
    ParticipantContextVisibilityConfiguration,
    ParticipantContextVisibilityPolicy,
    is_context_visible,
    participant_context_visibility_configuration,
    participant_context_visibility_policy,
    validate_deliberation_context_id,
)
from intergrax.contracts.decision_participants import (
    ParticipantBinding,
    ParticipantConfiguration,
    ParticipantRoleDefinition,
    ParticipantRoleId,
    participant_binding,
    participant_configuration,
    participant_role_definition,
)

_FORBIDDEN_FIELD_FRAGMENTS = (
    "adapter",
    "executor",
    "provider",
    "model",
    "inference_profile",
    "permission",
    "authorization",
    "can_execute",
    "allowed_tools",
    "side_effect",
    "approval",
)

_PRIVATE_COT_FRAGMENTS = (
    "chain_of_thought",
    "reasoning_trace",
    "scratchpad",
    "private_reasoning",
    "internal_reasoning",
)

_CONTRACT_TYPES = (
    ParticipantContextVisibilityPolicy,
    ParticipantContextVisibilityConfiguration,
)

_HARDCODED_CONTEXT_VOCABULARY = (
    "problem",
    "evidence",
    "peer-proposals",
    "disagreement",
    "proposals",
)

_FORBIDDEN_PRODUCTION_PATTERNS = (
    "Any",
    "cast",
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
)


def _production_module_code_source() -> str:
    import intergrax.contracts.decision_context_visibility as module

    lines = inspect.getsource(module).splitlines()
    code_lines: list[str] = []
    in_docstring = False
    for line in lines:
        stripped = line.strip()
        if stripped.startswith('"""') or stripped.startswith("'''"):
            if stripped.count('"""') == 2 or stripped.count("'''") == 2:
                continue
            in_docstring = not in_docstring
            continue
        if in_docstring:
            continue
        code_lines.append(line)
    return "\n".join(code_lines)


def _role(role_id: str, instruction: str = "Semantic instruction.") -> ParticipantRoleDefinition:
    return participant_role_definition(role_id=role_id, instruction=instruction)


def _binding(participant_id: str, role_id: str, profile_id: str) -> ParticipantBinding:
    return participant_binding(
        participant_id=participant_id,
        role_id=role_id,
        inference_profile_id=profile_id,
    )


def _policy(role_id: str, *contexts: str) -> ParticipantContextVisibilityPolicy:
    return participant_context_visibility_policy(
        role_id=role_id,
        visible_contexts=tuple(DeliberationContextId(c) for c in contexts),
    )


def _participant_config(
    roles: tuple[ParticipantRoleDefinition, ...],
    participants: tuple[ParticipantBinding, ...],
) -> ParticipantConfiguration:
    return participant_configuration(roles=roles, participants=participants)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "context_id",
    [
        "problem",
        "evidence",
        "peer-proposals",
        "dane finansowe",
        "法律意见",
    ],
)
def test_valid_deliberation_context_ids(context_id: str) -> None:
    validated = validate_deliberation_context_id(context_id)
    assert validated == context_id
    assert type(validated) is str


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("context_id", ["", "   "])
def test_invalid_empty_context_id_rejected(context_id: str) -> None:
    with pytest.raises(ValueError):
        validate_deliberation_context_id(context_id)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("context_id", [" context", "context "])
def test_surrounding_whitespace_context_id_rejected(context_id: str) -> None:
    with pytest.raises(ValueError):
        validate_deliberation_context_id(context_id)


@pytest.mark.unit
@pytest.mark.gate
def test_non_str_context_id_rejected() -> None:
    with pytest.raises(TypeError):
        validate_deliberation_context_id(42)


@pytest.mark.unit
@pytest.mark.gate
def test_valid_role_policy_arbitrary_role() -> None:
    policy = _policy("sceptyk", "problem", "evidence")
    assert policy.role_id == "sceptyk"
    assert list(policy.visible_contexts) == ["evidence", "problem"]


@pytest.mark.unit
@pytest.mark.gate
def test_empty_visible_contexts_allowed() -> None:
    policy = participant_context_visibility_policy(
        role_id="observer",
        visible_contexts=(),
    )
    assert policy.visible_contexts == ()


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_context_id_in_policy_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate"):
        participant_context_visibility_policy(
            role_id="sceptyk",
            visible_contexts=(
                DeliberationContextId("problem"),
                DeliberationContextId("problem"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_noncanonical_context_ordering_direct_constructor_rejected() -> None:
    with pytest.raises(ValueError, match="canonical order"):
        ParticipantContextVisibilityPolicy(
            role_id=ParticipantRoleId("sceptyk"),
            visible_contexts=(
                DeliberationContextId("peer-proposals"),
                DeliberationContextId("evidence"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_factory_canonicalizes_context_ordering() -> None:
    policy = participant_context_visibility_policy(
        role_id="sceptyk",
        visible_contexts=(
            DeliberationContextId("peer-proposals"),
            DeliberationContextId("evidence"),
            DeliberationContextId("problem"),
        ),
    )
    assert list(policy.visible_contexts) == ["evidence", "peer-proposals", "problem"]


@pytest.mark.unit
@pytest.mark.gate
def test_policy_immutable() -> None:
    policy = _policy("sceptyk", "problem")
    with pytest.raises(FrozenInstanceError):
        policy.role_id = "doradca"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.gate
def test_all_active_roles_have_policy_passes() -> None:
    config = _participant_config(
        roles=(
            _role("sceptyk"),
            _role("doradca"),
        ),
        participants=(
            _binding("skeptic-gpt", "sceptyk", "gpt5"),
            _binding("advisor-claude", "doradca", "claude"),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(
            _policy("sceptyk", "problem", "evidence"),
            _policy("doradca", "problem"),
        ),
    )
    assert [item.role_id for item in visibility.policies] == ["doradca", "sceptyk"]


@pytest.mark.unit
@pytest.mark.gate
def test_unknown_role_policy_rejected() -> None:
    config = _participant_config(
        roles=(
            _role("sceptyk"),
            _role("doradca"),
        ),
        participants=(
            _binding("skeptic-gpt", "sceptyk", "gpt5"),
            _binding("advisor-claude", "doradca", "claude"),
        ),
    )
    with pytest.raises(ValueError, match="known role"):
        participant_context_visibility_configuration(
            participant_configuration=config,
            policies=(
                _policy("sceptyk", "problem"),
                _policy("administrator", "problem"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_active_role_missing_policy_rejected() -> None:
    config = _participant_config(
        roles=(
            _role("sceptyk"),
            _role("doradca"),
        ),
        participants=(
            _binding("skeptic-gpt", "sceptyk", "gpt5"),
            _binding("advisor-claude", "doradca", "claude"),
        ),
    )
    with pytest.raises(ValueError, match="missing"):
        participant_context_visibility_configuration(
            participant_configuration=config,
            policies=(_policy("sceptyk", "problem"),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_role_policy_rejected() -> None:
    config = _participant_config(
        roles=(_role("sceptyk"),),
        participants=(_binding("skeptic-gpt", "sceptyk", "gpt5"),),
    )
    with pytest.raises(ValueError, match="duplicate role_id"):
        participant_context_visibility_configuration(
            participant_configuration=config,
            policies=(
                _policy("sceptyk", "problem"),
                _policy("sceptyk", "evidence"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_many_participants_same_role_one_policy_sufficient() -> None:
    config = _participant_config(
        roles=(_role("sceptyk"),),
        participants=(
            _binding("skeptic-gpt", "sceptyk", "gpt5"),
            _binding("skeptic-qwen", "sceptyk", "qwen"),
            _binding("skeptic-claude", "sceptyk", "claude"),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(_policy("sceptyk", "problem", "evidence", "peer-proposals"),),
    )
    assert len(visibility.policies) == 1
    assert visibility.policies[0].role_id == "sceptyk"


@pytest.mark.unit
@pytest.mark.gate
def test_different_role_names_and_languages_pass() -> None:
    config = _participant_config(
        roles=(
            _role("法律顾问", "Legal opinion."),
            _role("doradca prawny", "Polish legal advisor."),
        ),
        participants=(
            _binding("legal-cn", "法律顾问", "gpt5"),
            _binding("legal-pl", "doradca prawny", "claude"),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(
            _policy("法律顾问", "problem"),
            _policy("doradca prawny", "evidence"),
        ),
    )
    assert {p.role_id for p in visibility.policies} == {
        "法律顾问",
        "doradca prawny",
    }


@pytest.mark.unit
@pytest.mark.gate
def test_unused_role_without_policy_passes() -> None:
    config = _participant_config(
        roles=(
            _role("sceptyk"),
            _role("future-reviewer", "Unused role."),
        ),
        participants=(_binding("skeptic-gpt", "sceptyk", "gpt5"),),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(_policy("sceptyk", "problem"),),
    )
    assert visibility.active_role_ids == ("sceptyk",)
    assert visibility.known_role_ids == ("future-reviewer", "sceptyk")


@pytest.mark.unit
@pytest.mark.gate
def test_unused_role_with_policy_allowed() -> None:
    config = _participant_config(
        roles=(
            _role("sceptyk"),
            _role("future-reviewer", "Unused role."),
        ),
        participants=(_binding("skeptic-gpt", "sceptyk", "gpt5"),),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(
            _policy("sceptyk", "problem"),
            _policy("future-reviewer", "evidence"),
        ),
    )
    assert {p.role_id for p in visibility.policies} == {
        "future-reviewer",
        "sceptyk",
    }


@pytest.mark.unit
@pytest.mark.gate
def test_deterministic_policy_ordering_from_factory() -> None:
    config = _participant_config(
        roles=(
            _role("synthesizer"),
            _role("proposer"),
            _role("sceptical"),
        ),
        participants=(
            _binding("synth-qwen", "synthesizer", "qwen38-local"),
            _binding("proposer-fable", "proposer", "fable-proposer"),
            _binding("skeptic-gpt", "sceptical", "gpt5-sceptic"),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(
            _policy("synthesizer", "disagreement"),
            _policy("proposer", "problem"),
            _policy("sceptical", "evidence"),
        ),
    )
    assert [p.role_id for p in visibility.policies] == [
        "proposer",
        "sceptical",
        "synthesizer",
    ]


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_cannot_bypass_invariants() -> None:
    with pytest.raises(ValueError, match="missing"):
        ParticipantContextVisibilityConfiguration(
            active_role_ids=(
                ParticipantRoleId("doradca"),
                ParticipantRoleId("sceptyk"),
            ),
            known_role_ids=(
                ParticipantRoleId("doradca"),
                ParticipantRoleId("sceptyk"),
            ),
            policies=(_policy("sceptyk", "problem"),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_requires_canonical_policy_order() -> None:
    policies = (
        _policy("doradca", "problem"),
        _policy("sceptyk", "evidence"),
    )
    with pytest.raises(ValueError, match="canonical order"):
        ParticipantContextVisibilityConfiguration(
            active_role_ids=(
                ParticipantRoleId("doradca"),
                ParticipantRoleId("sceptyk"),
            ),
            known_role_ids=(
                ParticipantRoleId("doradca"),
                ParticipantRoleId("sceptyk"),
            ),
            policies=(policies[1], policies[0]),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_default_deny_listed_context_visible() -> None:
    policy = _policy("sceptyk", "problem", "evidence", "peer-proposals")
    assert is_context_visible(policy, "problem")
    assert is_context_visible(policy, "evidence")
    assert is_context_visible(policy, "peer-proposals")


@pytest.mark.unit
@pytest.mark.gate
def test_default_deny_unlisted_context_not_visible() -> None:
    policy = _policy("sceptyk", "problem", "evidence", "peer-proposals")
    assert not is_context_visible(policy, "disagreement")
    assert not is_context_visible(policy, "shared-transcript")


@pytest.mark.unit
@pytest.mark.gate
def test_real_configuration_example() -> None:
    config = _participant_config(
        roles=(
            _role("proposer", "Propose."),
            _role("sceptical", "Challenge."),
            _role("synthesizer", "Synthesize."),
        ),
        participants=(
            _binding("proposer-fable", "proposer", "fable-proposer"),
            _binding("skeptic-gpt", "sceptical", "gpt5-sceptic"),
            _binding("synth-qwen", "synthesizer", "qwen38-local"),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(
            _policy("proposer", "problem", "evidence"),
            _policy("sceptical", "problem", "evidence", "peer-proposals"),
            _policy("synthesizer", "problem", "peer-proposals", "disagreement"),
        ),
    )
    proposer_policy = next(p for p in visibility.policies if p.role_id == "proposer")
    sceptical_policy = next(p for p in visibility.policies if p.role_id == "sceptical")
    synthesizer_policy = next(
        p for p in visibility.policies if p.role_id == "synthesizer"
    )
    assert list(proposer_policy.visible_contexts) == ["evidence", "problem"]
    assert list(sceptical_policy.visible_contexts) == [
        "evidence",
        "peer-proposals",
        "problem",
    ]
    assert list(synthesizer_policy.visible_contexts) == [
        "disagreement",
        "peer-proposals",
        "problem",
    ]


@pytest.mark.unit
@pytest.mark.gate
def test_same_role_multiple_models_one_policy_applies() -> None:
    config = _participant_config(
        roles=(_role("sceptyk"),),
        participants=(
            _binding("skeptic-gpt", "sceptyk", "gpt5"),
            _binding("skeptic-qwen", "sceptyk", "qwen"),
            _binding("skeptic-claude", "sceptyk", "claude"),
        ),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(_policy("sceptyk", "problem", "evidence", "peer-proposals"),),
    )
    policy = visibility.policies[0]
    assert is_context_visible(policy, "peer-proposals")
    assert not is_context_visible(policy, "disagreement")


@pytest.mark.unit
@pytest.mark.gate
def test_structural_field_audit_no_model_coupling() -> None:
    for contract_type in _CONTRACT_TYPES:
        field_names = {field.name for field in dataclasses.fields(contract_type)}
        for field_name in field_names:
            lowered = field_name.lower()
            for fragment in _FORBIDDEN_FIELD_FRAGMENTS:
                assert fragment not in lowered, (
                    f"{contract_type.__name__}.{field_name} contains forbidden "
                    f"fragment {fragment!r}"
                )
            for fragment in _PRIVATE_COT_FRAGMENTS:
                assert fragment not in lowered, (
                    f"{contract_type.__name__}.{field_name} contains private CoT "
                    f"fragment {fragment!r}"
                )


@pytest.mark.unit
@pytest.mark.gate
def test_hardcoded_context_vocabulary_not_in_production_module() -> None:
    code_source = _production_module_code_source().lower()
    for term in _HARDCODED_CONTEXT_VOCABULARY:
        assert term not in code_source, (
            f"production module contains hardcoded context vocabulary: {term!r}"
        )


@pytest.mark.unit
@pytest.mark.gate
def test_forbidden_production_patterns_absent() -> None:
    source = _production_module_code_source()
    hits: list[str] = []
    for pattern in _FORBIDDEN_PRODUCTION_PATTERNS:
        if pattern in source:
            hits.append(pattern)
    assert hits == [], f"forbidden production patterns found: {hits}"


@pytest.mark.unit
@pytest.mark.gate
def test_configuration_immutable() -> None:
    config = _participant_config(
        roles=(_role("sceptyk"),),
        participants=(_binding("skeptic-gpt", "sceptyk", "gpt5"),),
    )
    visibility = participant_context_visibility_configuration(
        participant_configuration=config,
        policies=(_policy("sceptyk", "problem"),),
    )
    with pytest.raises(FrozenInstanceError):
        visibility.policies = ()  # type: ignore[misc]

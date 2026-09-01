# © Artur Czarnecki. All rights reserved.

import dataclasses
import inspect
from dataclasses import FrozenInstanceError

import pytest

from intergrax.contracts.decision_participants import (
    ParticipantBinding,
    ParticipantConfiguration,
    ParticipantId,
    ParticipantRoleDefinition,
    ParticipantRoleId,
    participant_binding,
    participant_configuration,
    participant_role_definition,
    validate_participant_id,
    validate_participant_role_id,
)
from intergrax.runtime.execution.inference_profile import (
    InferenceProfileId,
    validate_inference_profile_id,
)

_FORBIDDEN_FIELD_FRAGMENTS = (
    "adapter",
    "executor",
    "provider",
    "model_name",
    "temperature",
    "context_visibility",
    "winner",
    "verification",
    "hitl",
)

_PRIVATE_COT_FRAGMENTS = (
    "chain_of_thought",
    "reasoning_trace",
    "scratchpad",
    "internal_reasoning",
)

_CONTRACT_TYPES = (
    ParticipantRoleDefinition,
    ParticipantBinding,
    ParticipantConfiguration,
)

_HARDCODED_ROLE_VOCABULARY = (
    "proposer",
    "skeptic",
    "synthesizer",
    "critic",
    "advisor",
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
    import intergrax.contracts.decision_participants as module

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


def _role(
    role_id: str,
    instruction: str = "Semantic instruction for the role.",
) -> ParticipantRoleDefinition:
    return participant_role_definition(role_id=role_id, instruction=instruction)


def _binding(
    participant_id: str,
    role_id: str,
    profile_id: str,
) -> ParticipantBinding:
    return participant_binding(
        participant_id=participant_id,
        role_id=role_id,
        inference_profile_id=profile_id,
    )


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize(
    "role_id",
    [
        "sceptical",
        "sceptyk",
        "doradca prawny",
        "法律顾问",
        "security-reviewer",
    ],
)
def test_valid_arbitrary_role_ids(role_id: str) -> None:
    validated = validate_participant_role_id(role_id)
    assert validated == role_id
    assert type(validated) is str


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("role_id", ["", "   "])
def test_invalid_empty_role_id_rejected(role_id: str) -> None:
    with pytest.raises(ValueError):
        validate_participant_role_id(role_id)


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("role_id", [" sceptyk", "sceptyk "])
def test_surrounding_whitespace_role_id_rejected(role_id: str) -> None:
    with pytest.raises(ValueError):
        validate_participant_role_id(role_id)


@pytest.mark.unit
@pytest.mark.gate
def test_non_str_role_id_rejected() -> None:
    with pytest.raises(TypeError):
        validate_participant_role_id(42)


@pytest.mark.unit
@pytest.mark.gate
def test_valid_participant_id() -> None:
    participant_id = validate_participant_id("architecture-agent")
    assert participant_id == "architecture-agent"
    assert type(participant_id) is str


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("participant_id", ["", "   ", " agent", "agent "])
def test_invalid_participant_id_rejected(participant_id: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        validate_participant_id(participant_id)


@pytest.mark.unit
@pytest.mark.gate
def test_valid_role_definition() -> None:
    role = _role("architekt", "Oceń rozwiązanie z perspektywy architektury systemu.")
    assert role.role_id == "architekt"
    assert role.instruction.startswith("Oceń")


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("instruction", ["", "   "])
def test_blank_role_instruction_rejected(instruction: str) -> None:
    with pytest.raises(ValueError):
        participant_role_definition(role_id="architekt", instruction=instruction)


@pytest.mark.unit
@pytest.mark.gate
def test_whitespace_role_instruction_rejected() -> None:
    with pytest.raises(ValueError):
        participant_role_definition(
            role_id="architekt",
            instruction="  instruction  ",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_direct_role_definition_validates_role_id() -> None:
    with pytest.raises(ValueError):
        participant_role_definition(role_id="", instruction="Valid instruction.")


@pytest.mark.unit
@pytest.mark.gate
def test_direct_role_definition_constructor_validates_role() -> None:
    with pytest.raises(ValueError):
        ParticipantRoleDefinition(
            role_id=ParticipantRoleId(""),
            instruction="Valid instruction.",
        )


@pytest.mark.unit
@pytest.mark.gate
def test_role_definition_immutable() -> None:
    role = _role("sceptyk")
    with pytest.raises(FrozenInstanceError):
        role.instruction = "mutated"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.gate
def test_valid_participant_binding() -> None:
    binding = _binding("architecture-agent", "architekt", "fable-architecture")
    assert binding.participant_id == "architecture-agent"
    assert binding.role_id == "architekt"
    assert binding.inference_profile_id == "fable-architecture"


@pytest.mark.unit
@pytest.mark.gate
def test_arbitrary_role_name_in_binding() -> None:
    binding = _binding("skeptic-gpt", "krytyk", "gpt5-skeptic")
    assert binding.role_id == "krytyk"


@pytest.mark.unit
@pytest.mark.gate
def test_valid_inference_profile_id_in_binding() -> None:
    profile = validate_inference_profile_id("qwen38-local")
    binding = ParticipantBinding(
        participant_id=ParticipantId("advisor-qwen"),
        role_id=ParticipantRoleId("doradca"),
        inference_profile_id=profile,
    )
    assert binding.inference_profile_id == "qwen38-local"


@pytest.mark.unit
@pytest.mark.gate
@pytest.mark.parametrize("profile_id", ["", "   ", " profile", "profile "])
def test_invalid_inference_profile_in_binding_rejected(profile_id: str) -> None:
    with pytest.raises((TypeError, ValueError)):
        ParticipantBinding(
            participant_id=ParticipantId("agent-1"),
            role_id=ParticipantRoleId("sceptyk"),
            inference_profile_id=InferenceProfileId(profile_id),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_blank_participant_id_in_binding_rejected() -> None:
    with pytest.raises(ValueError):
        ParticipantBinding(
            participant_id=ParticipantId(""),
            role_id=ParticipantRoleId("sceptyk"),
            inference_profile_id=InferenceProfileId("gpt5"),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_participant_binding_immutable() -> None:
    binding = _binding("agent-1", "sceptyk", "gpt5")
    with pytest.raises(FrozenInstanceError):
        binding.role_id = "doradca"  # type: ignore[misc]


@pytest.mark.unit
@pytest.mark.gate
def test_valid_participant_configuration() -> None:
    config = participant_configuration(
        roles=(
            _role("architekt", "Architecture review."),
            _role("sceptyk", "Challenge assumptions."),
        ),
        participants=(
            _binding("architecture-agent", "architekt", "fable-architecture"),
            _binding("skeptic-gpt", "sceptyk", "gpt5-skeptic"),
        ),
    )
    assert [role.role_id for role in config.roles] == ["architekt", "sceptyk"]
    assert [item.participant_id for item in config.participants] == [
        "architecture-agent",
        "skeptic-gpt",
    ]


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_role_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate role_id"):
        participant_configuration(
            roles=(
                _role("sceptyk", "First."),
                _role("sceptyk", "Second."),
            ),
            participants=(_binding("agent-1", "sceptyk", "gpt5"),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_duplicate_participant_ids_rejected() -> None:
    with pytest.raises(ValueError, match="duplicate participant_id"):
        participant_configuration(
            roles=(_role("sceptyk", "Challenge."),),
            participants=(
                _binding("agent-1", "sceptyk", "gpt5"),
                _binding("agent-1", "sceptyk", "claude"),
            ),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_unknown_role_binding_rejected() -> None:
    with pytest.raises(ValueError, match="known RoleDefinition"):
        participant_configuration(
            roles=(_role("sceptyk", "Challenge."),),
            participants=(_binding("agent-1", "doradca", "gpt5"),),
        )


@pytest.mark.unit
@pytest.mark.gate
def test_many_participants_same_role_allowed() -> None:
    config = participant_configuration(
        roles=(_role("sceptyk", "Challenge assumptions."),),
        participants=(
            _binding("skeptic-gpt", "sceptyk", "gpt5"),
            _binding("skeptic-qwen", "sceptyk", "qwen"),
            _binding("skeptic-claude", "sceptyk", "claude"),
        ),
    )
    assert len(config.participants) == 3
    assert {item.role_id for item in config.participants} == {"sceptyk"}


@pytest.mark.unit
@pytest.mark.gate
def test_same_profile_different_roles_allowed() -> None:
    config = participant_configuration(
        roles=(
            _role("architekt", "Architecture."),
            _role("sceptyk", "Challenge."),
        ),
        participants=(
            _binding("architecture-agent", "architekt", "shared-profile"),
            _binding("skeptic-gpt", "sceptyk", "shared-profile"),
        ),
    )
    profiles = {item.inference_profile_id for item in config.participants}
    assert profiles == {"shared-profile"}
    assert len({role.role_id for role in config.roles}) == 2


@pytest.mark.unit
@pytest.mark.gate
def test_different_profiles_same_role_allowed() -> None:
    config = participant_configuration(
        roles=(_role("sceptyk", "Challenge."),),
        participants=(
            _binding("skeptic-gpt", "sceptyk", "gpt5"),
            _binding("skeptic-claude", "sceptyk", "claude"),
        ),
    )
    assert {item.inference_profile_id for item in config.participants} == {
        "gpt5",
        "claude",
    }


@pytest.mark.unit
@pytest.mark.gate
def test_unused_role_allowed() -> None:
    config = participant_configuration(
        roles=(
            _role("architekt", "Architecture."),
            _role("doradca", "Advisory."),
        ),
        participants=(_binding("architecture-agent", "architekt", "fable-architecture"),),
    )
    assert {role.role_id for role in config.roles} == {"architekt", "doradca"}
    assert len(config.participants) == 1


@pytest.mark.unit
@pytest.mark.gate
def test_deterministic_ordering_from_factory() -> None:
    config = participant_configuration(
        roles=(
            _role("synthesizer", "Synthesize."),
            _role("proposer", "Propose."),
            _role("sceptical", "Challenge."),
        ),
        participants=(
            _binding("synth-qwen", "synthesizer", "qwen38-local"),
            _binding("proposer-fable", "proposer", "fable-proposer"),
            _binding("skeptic-gpt", "sceptical", "gpt5-sceptic"),
        ),
    )
    assert [role.role_id for role in config.roles] == [
        "proposer",
        "sceptical",
        "synthesizer",
    ]
    assert [item.participant_id for item in config.participants] == [
        "proposer-fable",
        "skeptic-gpt",
        "synth-qwen",
    ]


@pytest.mark.unit
@pytest.mark.gate
def test_direct_constructor_requires_canonical_ordering() -> None:
    roles = (
        _role("architekt", "Architecture."),
        _role("sceptyk", "Challenge."),
    )
    participants = (
        _binding("architecture-agent", "architekt", "fable-architecture"),
        _binding("skeptic-gpt", "sceptyk", "gpt5-skeptic"),
    )
    with pytest.raises(ValueError, match="canonical order"):
        ParticipantConfiguration(
            roles=(roles[1], roles[0]),
            participants=participants,
        )


@pytest.mark.unit
@pytest.mark.gate
def test_example_configuration_role_names_are_opaque_strings() -> None:
    config = participant_configuration(
        roles=(
            _role("proposer", "Propose a candidate."),
            _role("sceptical", "Challenge the proposal."),
            _role("synthesizer", "Synthesize competing views."),
        ),
        participants=(
            _binding("proposer-fable", "proposer", "fable-proposer"),
            _binding("skeptic-gpt", "sceptical", "gpt5-sceptic"),
            _binding("synth-qwen", "synthesizer", "qwen38-local"),
        ),
    )
    profile_by_role = {
        binding.role_id: binding.inference_profile_id for binding in config.participants
    }
    assert profile_by_role == {
        "proposer": "fable-proposer",
        "sceptical": "gpt5-sceptic",
        "synthesizer": "qwen38-local",
    }


@pytest.mark.unit
@pytest.mark.gate
def test_structural_field_audit_forbidden_architecture_concepts() -> None:
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
def test_hardcoded_role_vocabulary_not_in_production_module() -> None:
    code_source = _production_module_code_source().lower()
    for term in _HARDCODED_ROLE_VOCABULARY:
        assert term not in code_source, (
            f"production module contains hardcoded role vocabulary: {term!r}"
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
    config = participant_configuration(
        roles=(_role("sceptyk", "Challenge."),),
        participants=(_binding("agent-1", "sceptyk", "gpt5"),),
    )
    with pytest.raises(FrozenInstanceError):
        config.roles = ()  # type: ignore[misc]

# © Artur Czarnecki. All rights reserved.

"""ENG-6 — typed explicit evidence dependency proof for bounded native investigation."""

from __future__ import annotations

import json
from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.contracts.model_visible_evidence import ModelVisibleEvidenceReference
from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.tools.native_planner_transcript import (
    canonical_native_planner_messages,
)
from intergrax.tools.model_observation_format import parse_evidence_reference_from_tool_content

_EVIDENCE_BASIS_PREFIX = "EVIDENCE_BASIS:"
_PURPOSE_PREFIX = "PURPOSE:"
_EVIDENCE_ID_JSON_KEY = "evidence_id"
_EVIDENCE_REFERENCE_JSON_KEY = "evidence_reference"
_OBSERVATION_REFERENCE_JSON_KEY = "observation_reference"


class InvestigationProofValidationError(ValueError):
    """Invalid public decision note or evidence basis declaration (ENG-6)."""


@dataclass(frozen=True, slots=True)
class InvestigationEvidenceBasis:
    """Model-declared semantic evidence reference bound to canonical runtime identity."""

    declared_reference: str
    tool_call_id: str


@dataclass(frozen=True, slots=True)
class InvestigationProofStep:
    """One native investigative tool round with explicit evidence dependency."""

    round_index: int
    declared_basis_references: tuple[str, ...]
    basis_bindings: tuple[InvestigationEvidenceBasis, ...]
    basis_tool_call_ids: tuple[str, ...]
    next_tool_call_ids: tuple[str, ...]
    public_reason: str


@dataclass(frozen=True, slots=True)
class InvestigationProof:
    """Auditable multi-hop investigation proof for one bounded native run."""

    steps: tuple[InvestigationProofStep, ...]
    final_available_evidence_ids: tuple[str, ...]


@dataclass(frozen=True, slots=True)
class ParsedPublicDecisionNote:
    basis_evidence_references: tuple[str, ...]
    public_reason: str


def mint_runtime_observation_evidence_reference(*, tool_id: str, step_id: str) -> str:
    """Stable model-visible observation reference for generic tool execution paths."""
    return f"observation.{tool_id}.{step_id}"


def _extract_semantic_reference_from_tool_content(content: str) -> str | None:
    envelope_reference = parse_evidence_reference_from_tool_content(content)
    if envelope_reference is not None:
        return envelope_reference
    stripped = content.strip()
    if not stripped:
        return None
    try:
        payload = json.loads(stripped)
    except json.JSONDecodeError:
        payload = None
    if isinstance(payload, dict):
        for key in (
            _EVIDENCE_ID_JSON_KEY,
            _EVIDENCE_REFERENCE_JSON_KEY,
            _OBSERVATION_REFERENCE_JSON_KEY,
        ):
            candidate = payload.get(key)
            if isinstance(candidate, str) and candidate.strip():
                return candidate.strip()
    return None


def _index_semantic_reference(
    index: dict[str, str],
    reference: str,
    binding_id: str,
    *,
    allow_refresh: bool = False,
) -> None:
    existing = index.get(reference)
    if existing is not None and existing != binding_id:
        if not allow_refresh:
            raise InvestigationProofValidationError(
                f"ambiguous evidence reference provenance: {reference}"
            )
    index[reference] = binding_id


def build_completed_observation_reference_index(
    messages: Sequence[ChatMessage],
    prior_references: Sequence[ModelVisibleEvidenceReference] = (),
) -> dict[str, str]:
    """Map model-visible semantic evidence references to canonical provenance ids."""
    index: dict[str, str] = {}
    for prior in prior_references:
        _index_semantic_reference(
            index,
            prior.evidence_reference,
            prior.binding_id(),
        )
    canonical = canonical_native_planner_messages(messages)
    for message in canonical:
        if message.role != "tool":
            continue
        tool_call_id = message.tool_call_id
        if not isinstance(tool_call_id, str) or not tool_call_id:
            continue
        reference = _extract_semantic_reference_from_tool_content(message.content or "")
        if reference is None:
            continue
        _index_semantic_reference(index, reference, tool_call_id, allow_refresh=True)
    return index


def collect_available_evidence_ids(
    messages: Sequence[ChatMessage],
    prior_references: Sequence[ModelVisibleEvidenceReference] = (),
) -> tuple[str, ...]:
    """Return model-visible semantic evidence references from completed observations."""
    references: list[str] = []
    seen: set[str] = set()
    for prior in prior_references:
        reference = prior.evidence_reference
        if reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    canonical = canonical_native_planner_messages(messages)
    for message in canonical:
        if message.role != "tool":
            continue
        reference = _extract_semantic_reference_from_tool_content(message.content or "")
        if reference is None or reference in seen:
            continue
        seen.add(reference)
        references.append(reference)
    return tuple(references)


def _parse_basis_value(raw: str) -> tuple[str, ...]:
    stripped = raw.strip()
    if not stripped:
        return ()
    parts = [part.strip() for part in stripped.split(",")]
    if any(not part for part in parts):
        raise InvestigationProofValidationError(
            "malformed public decision note: empty EVIDENCE_BASIS id segment"
        )
    seen: set[str] = set()
    ordered: list[str] = []
    for part in parts:
        if part in seen:
            raise InvestigationProofValidationError(
                f"duplicate basis evidence reference: {part}"
            )
        seen.add(part)
        ordered.append(part)
    return tuple(ordered)


def parse_public_decision_note(content: str) -> ParsedPublicDecisionNote:
    """Parse the strict ENG-6 two-field public decision-note envelope."""
    semantic_lines = tuple(
        stripped
        for line in content.splitlines()
        if (stripped := line.strip())
    )
    if len(semantic_lines) != 2:
        raise InvestigationProofValidationError(
            "malformed public decision note: envelope must contain exactly two non-empty fields"
        )
    basis_line = semantic_lines[0]
    purpose_line = semantic_lines[1]
    if not basis_line.startswith(_EVIDENCE_BASIS_PREFIX):
        raise InvestigationProofValidationError(
            "malformed public decision note: missing EVIDENCE_BASIS"
        )
    if not purpose_line.startswith(_PURPOSE_PREFIX):
        raise InvestigationProofValidationError(
            "malformed public decision note: missing PURPOSE"
        )
    purpose = purpose_line[len(_PURPOSE_PREFIX) :].strip()
    if not purpose:
        raise InvestigationProofValidationError(
            "malformed public decision note: empty PURPOSE"
        )
    return ParsedPublicDecisionNote(
        basis_evidence_references=_parse_basis_value(
            basis_line[len(_EVIDENCE_BASIS_PREFIX) :]
        ),
        public_reason=purpose,
    )


def _try_extract_purpose(content: str) -> str:
    for line in content.splitlines():
        stripped = line.strip()
        if stripped.startswith(_PURPOSE_PREFIX):
            return stripped[len(_PURPOSE_PREFIX) :].strip()
    return ""


def _bind_declared_basis_references(
    declared_references: tuple[str, ...],
    reference_index: dict[str, str],
) -> tuple[InvestigationEvidenceBasis, ...]:
    bindings: list[InvestigationEvidenceBasis] = []
    for reference in declared_references:
        tool_call_id = reference_index.get(reference)
        if tool_call_id is None:
            raise InvestigationProofValidationError(
                f"unknown basis evidence reference: {reference}"
            )
        bindings.append(
            InvestigationEvidenceBasis(
                declared_reference=reference,
                tool_call_id=tool_call_id,
            )
        )
    return tuple(bindings)


def validate_follow_up_investigation_step(
    *,
    round_index: int,
    assistant_content: str,
    available_evidence_references: frozenset[str],
    reference_index: dict[str, str],
    next_tool_call_ids: tuple[str, ...],
) -> InvestigationProofStep:
    """Validate explicit evidence basis before executing follow-up native tools."""
    parsed = parse_public_decision_note(assistant_content)
    if available_evidence_references and not parsed.basis_evidence_references:
        raise InvestigationProofValidationError(
            "follow-up tool round requires explicit evidence basis"
        )
    bindings = _bind_declared_basis_references(
        parsed.basis_evidence_references,
        reference_index,
    )
    return InvestigationProofStep(
        round_index=round_index,
        declared_basis_references=parsed.basis_evidence_references,
        basis_bindings=bindings,
        basis_tool_call_ids=tuple(binding.tool_call_id for binding in bindings),
        next_tool_call_ids=next_tool_call_ids,
        public_reason=parsed.public_reason,
    )


def record_first_investigation_step(
    *,
    round_index: int,
    assistant_content: str,
    next_tool_call_ids: tuple[str, ...],
) -> InvestigationProofStep:
    """First investigative tool round — empty basis, objective-driven."""
    return InvestigationProofStep(
        round_index=round_index,
        declared_basis_references=(),
        basis_bindings=(),
        basis_tool_call_ids=(),
        next_tool_call_ids=next_tool_call_ids,
        public_reason=_try_extract_purpose(assistant_content),
    )


def build_investigation_proof_step(
    *,
    round_index: int,
    assistant_content: str,
    tool_calls: Sequence[LLMToolCall],
    messages_before_round: Sequence[ChatMessage],
    prior_model_visible_references: Sequence[ModelVisibleEvidenceReference] = (),
) -> InvestigationProofStep:
    """Snapshot, parse, validate, and record one investigative tool round."""
    next_tool_call_ids = tuple(tool_call.id for tool_call in tool_calls)
    if round_index <= 1:
        return record_first_investigation_step(
            round_index=round_index,
            assistant_content=assistant_content,
            next_tool_call_ids=next_tool_call_ids,
        )
    reference_index = build_completed_observation_reference_index(
        messages_before_round,
        prior_model_visible_references,
    )
    available = frozenset(reference_index)
    return validate_follow_up_investigation_step(
        round_index=round_index,
        assistant_content=assistant_content,
        available_evidence_references=available,
        reference_index=reference_index,
        next_tool_call_ids=next_tool_call_ids,
    )

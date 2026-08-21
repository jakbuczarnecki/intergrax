# © Artur Czarnecki. All rights reserved.

"""ENG-6 — typed explicit evidence dependency proof for bounded native investigation."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

from intergrax.llm.messages import ChatMessage
from intergrax.llm_adapters.contracts.tool_call import LLMToolCall
from intergrax.runtime.nexus.tools.native_planner_transcript import (
    canonical_native_planner_messages,
)

_EVIDENCE_BASIS_PREFIX = "EVIDENCE_BASIS:"
_PURPOSE_PREFIX = "PURPOSE:"


class InvestigationProofValidationError(ValueError):
    """Invalid public decision note or evidence basis declaration (ENG-6)."""


@dataclass(frozen=True, slots=True)
class InvestigationProofStep:
    """One native investigative tool round with explicit evidence dependency."""

    round_index: int
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
    basis_tool_call_ids: tuple[str, ...]
    public_reason: str


def collect_available_evidence_ids(messages: Sequence[ChatMessage]) -> tuple[str, ...]:
    """Return tool_call_id handles from model-visible canonical native transcript."""
    canonical = canonical_native_planner_messages(messages)
    ids: list[str] = []
    seen: set[str] = set()
    for message in canonical:
        if message.role != "tool":
            continue
        tool_call_id = message.tool_call_id
        if not tool_call_id or tool_call_id in seen:
            continue
        seen.add(tool_call_id)
        ids.append(tool_call_id)
    return tuple(ids)


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
                f"duplicate basis tool_call_id: {part}"
            )
        seen.add(part)
        ordered.append(part)
    return tuple(ordered)


def parse_public_decision_note(content: str) -> ParsedPublicDecisionNote:
    """Parse the strict ENG-6 two-line public decision-note envelope."""
    lines = content.splitlines()
    if len(lines) != 2:
        raise InvestigationProofValidationError(
            "malformed public decision note: envelope must be exactly two lines"
        )
    basis_line = lines[0].strip()
    purpose_line = lines[1].strip()
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
        basis_tool_call_ids=_parse_basis_value(
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


def validate_follow_up_investigation_step(
    *,
    round_index: int,
    assistant_content: str,
    available_evidence_ids: frozenset[str],
    next_tool_call_ids: tuple[str, ...],
) -> InvestigationProofStep:
    """Validate explicit evidence basis before executing follow-up native tools."""
    parsed = parse_public_decision_note(assistant_content)
    if available_evidence_ids and not parsed.basis_tool_call_ids:
        raise InvestigationProofValidationError(
            "follow-up tool round requires explicit evidence basis"
        )
    unknown = [
        basis_id
        for basis_id in parsed.basis_tool_call_ids
        if basis_id not in available_evidence_ids
    ]
    if unknown:
        raise InvestigationProofValidationError(
            f"unknown basis tool_call_id: {unknown[0]}"
        )
    return InvestigationProofStep(
        round_index=round_index,
        basis_tool_call_ids=parsed.basis_tool_call_ids,
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
) -> InvestigationProofStep:
    """Snapshot, parse, validate, and record one investigative tool round."""
    next_tool_call_ids = tuple(tool_call.id for tool_call in tool_calls)
    if round_index <= 1:
        return record_first_investigation_step(
            round_index=round_index,
            assistant_content=assistant_content,
            next_tool_call_ids=next_tool_call_ids,
        )
    available = frozenset(collect_available_evidence_ids(messages_before_round))
    return validate_follow_up_investigation_step(
        round_index=round_index,
        assistant_content=assistant_content,
        available_evidence_ids=available,
        next_tool_call_ids=next_tool_call_ids,
    )

# © Artur Czarnecki. All rights reserved.

"""Process pattern miner for adaptive harness (Phase W-ADAPT-6.1)."""

from __future__ import annotations

import hashlib
from collections import Counter, defaultdict

from pydantic import BaseModel, ConfigDict, Field

from intergrax.runtime.adaptive.contracts import ProcessPatternAction, ProcessPatternProposal
from intergrax.runtime.adaptive.trace_sequence_reader import (
    ProcessSequenceToken,
    RunProcessSequence,
    TraceSequenceReader,
)


class ProcessPatternMinerConfig(BaseModel):
    """Configuration for n-gram process pattern mining."""

    model_config = ConfigDict(extra="forbid")

    min_support: int = Field(default=2, ge=1)
    ngram_size: int = Field(default=2, ge=1, le=4)
    max_patterns: int = Field(default=50, ge=1)


class ProcessPatternMinerResult(BaseModel):
    """Batch output from a pattern mining cycle."""

    model_config = ConfigDict(extra="forbid")

    proposals: list[ProcessPatternProposal] = Field(default_factory=list)
    scanned_run_count: int = 0


class ProcessPatternMiner:
    """Offline n-gram frequency miner over trace-derived process sequences."""

    def __init__(self, *, sequence_reader: TraceSequenceReader) -> None:
        self._sequence_reader = sequence_reader

    def mine(
        self,
        *,
        tenant_id: str,
        config: ProcessPatternMinerConfig | None = None,
        run_limit: int = 200,
    ) -> ProcessPatternMinerResult:
        resolved = config or ProcessPatternMinerConfig()
        sequences = self._sequence_reader.load_sequences(tenant_id=tenant_id, limit=run_limit)
        if not sequences:
            return ProcessPatternMinerResult(scanned_run_count=0)

        pattern_runs: dict[str, list[str]] = defaultdict(list)
        pattern_utilities: dict[str, list[float]] = defaultdict(list)
        pattern_descriptions: dict[str, str] = {}

        for sequence in sequences:
            for ngram in _extract_ngrams(sequence.tokens, size=resolved.ngram_size):
                signature = _token_signature(ngram)
                pattern_id = _stable_pattern_id(signature)
                pattern_runs[pattern_id].append(sequence.run_id)
                pattern_descriptions[pattern_id] = _describe_ngram(ngram)
                if sequence.utility is not None:
                    pattern_utilities[pattern_id].append(sequence.utility)

        proposals: list[ProcessPatternProposal] = []
        for pattern_id, run_ids in pattern_runs.items():
            support = len(run_ids)
            if support < resolved.min_support:
                continue
            utilities = pattern_utilities.get(pattern_id, [])
            avg_utility = sum(utilities) / len(utilities) if utilities else None
            proposals.append(
                ProcessPatternProposal(
                    pattern_id=pattern_id,
                    description=pattern_descriptions[pattern_id],
                    suggested_action=_suggest_action(pattern_descriptions[pattern_id]),
                    evidence_run_ids=sorted(set(run_ids))[:5],
                    support_count=support,
                    avg_utility=avg_utility,
                    utility_correlation=avg_utility,
                )
            )

        proposals.sort(key=lambda item: item.support_count, reverse=True)
        return ProcessPatternMinerResult(
            proposals=proposals[: resolved.max_patterns],
            scanned_run_count=len(sequences),
        )


def _extract_ngrams(
    tokens: list[ProcessSequenceToken],
    *,
    size: int,
) -> list[tuple[ProcessSequenceToken, ...]]:
    if not tokens:
        return []
    if size <= 1:
        return [(token,) for token in tokens]
    ngrams: list[tuple[ProcessSequenceToken, ...]] = []
    for index in range(0, len(tokens) - size + 1):
        ngrams.append(tuple(tokens[index : index + size]))
    return ngrams


def _token_signature(tokens: tuple[ProcessSequenceToken, ...]) -> str:
    parts = [
        f"{token.task_class}|{token.agent_id}|{token.tool_id}|"
        f"{int(token.hitl_pause)}|{int(token.outcome_success)}"
        for token in tokens
    ]
    return " -> ".join(parts)


def _stable_pattern_id(signature: str) -> str:
    digest = hashlib.sha256(signature.encode("utf-8")).hexdigest()
    return f"pat_{digest[:12]}"


def _describe_ngram(tokens: tuple[ProcessSequenceToken, ...]) -> str:
    tool_ids = sorted({token.tool_id for token in tokens if token.tool_id != "_none_"})
    agents = sorted({token.agent_id for token in tokens})
    hitl = any(token.hitl_pause for token in tokens)
    task_classes = sorted({token.task_class for token in tokens})
    tool_text = ",".join(tool_ids) if tool_ids else "no-tool"
    return (
        f"task={','.join(task_classes)} agents={','.join(agents)} "
        f"tools={tool_text} hitl={hitl}"
    )


def _suggest_action(description: str) -> ProcessPatternAction:
    lowered = description.lower()
    if "hitl=true" in lowered:
        return ProcessPatternAction.DOCUMENT_RUNBOOK
    if "tools=" in lowered and "no-tool" not in lowered:
        return ProcessPatternAction.CREATE_SKILL_DRAFT
    return ProcessPatternAction.TUNE_ROUTING

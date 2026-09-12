# © Artur Czarnecki. All rights reserved.
# Intergrax framework — proprietary and confidential.

"""Operational conditions for strategy knowledge — not execution directives (R5.5)."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class StrategyKnowledgeOperatingContext:
    """
    Descriptive conditions under which strategy knowledge was derived.

    Separates *where/when/how* from strategy identity and quality scores.
    Must not encode actions, routing, or lifecycle commands.
    """

    problem_type: str | None = None
    environment_label: str | None = None
    problem_source: str | None = None
    execution_conditions: tuple[str, ...] = ()
    constraints: tuple[str, ...] = ()
    provider_id: str | None = None
    descriptor_refs: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if self.problem_type is not None and not self.problem_type.strip():
            raise ValueError("problem_type must be non-empty when set")
        if self.environment_label is not None and not self.environment_label.strip():
            raise ValueError("environment_label must be non-empty when set")
        if self.problem_source is not None and not self.problem_source.strip():
            raise ValueError("problem_source must be non-empty when set")
        for label, values in (
            ("execution_conditions", self.execution_conditions),
            ("constraints", self.constraints),
            ("descriptor_refs", self.descriptor_refs),
        ):
            for value in values:
                if not value.strip():
                    raise ValueError(f"{label} entries must be non-empty")

    @property
    def is_empty(self) -> bool:
        return (
            self.problem_type is None
            and self.environment_label is None
            and self.problem_source is None
            and not self.execution_conditions
            and not self.constraints
            and not self.descriptor_refs
        )


def merge_operating_contexts(
    contexts: tuple[StrategyKnowledgeOperatingContext, ...],
) -> StrategyKnowledgeOperatingContext | None:
    """Combine plugin contributions — first non-empty scalar wins; tuples concatenated."""
    if not contexts:
        return None
    problem_type: str | None = None
    environment_label: str | None = None
    problem_source: str | None = None
    execution_conditions: list[str] = []
    constraints: list[str] = []
    descriptor_refs: list[str] = []
    provider_ids: list[str] = []
    for ctx in contexts:
        if ctx.is_empty:
            continue
        if problem_type is None and ctx.problem_type is not None:
            problem_type = ctx.problem_type
        if environment_label is None and ctx.environment_label is not None:
            environment_label = ctx.environment_label
        if problem_source is None and ctx.problem_source is not None:
            problem_source = ctx.problem_source
        execution_conditions.extend(ctx.execution_conditions)
        constraints.extend(ctx.constraints)
        descriptor_refs.extend(ctx.descriptor_refs)
        if ctx.provider_id is not None:
            provider_ids.append(ctx.provider_id)
    if (
        problem_type is None
        and environment_label is None
        and problem_source is None
        and not execution_conditions
        and not constraints
        and not descriptor_refs
    ):
        return None
    merged_provider = "+".join(provider_ids) if provider_ids else None
    return StrategyKnowledgeOperatingContext(
        problem_type=problem_type,
        environment_label=environment_label,
        problem_source=problem_source,
        execution_conditions=tuple(execution_conditions),
        constraints=tuple(constraints),
        provider_id=merged_provider,
        descriptor_refs=tuple(descriptor_refs),
    )


__all__ = ["StrategyKnowledgeOperatingContext", "merge_operating_contexts"]

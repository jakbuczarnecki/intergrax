# © Artur Czarnecki. All rights reserved.

"""Context planning contracts (CTX-UCL-3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.context.contracts import ContextFragmentSource
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompressionTarget,
    ArtifactSourceRange,
    OptimizationArtifactType,
)


def _require_non_empty(value: str, field_name: str) -> str:
    if not value:
        raise ValueError(f"{field_name} must be non-empty")
    return value


def _require_int(value: object, field_name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer")
    return value


def _require_non_negative(value: object, field_name: str) -> int:
    int_value = _require_int(value, field_name)
    if int_value < 0:
        raise ValueError(f"{field_name} must be >= 0")
    return int_value


def _require_positive(value: object, field_name: str) -> int:
    int_value = _require_int(value, field_name)
    if int_value <= 0:
        raise ValueError(f"{field_name} must be > 0")
    return int_value


def _reject_duplicates(values: tuple[str, ...], field_name: str) -> tuple[str, ...]:
    if len(values) != len(set(values)):
        raise ValueError(f"{field_name} must not contain duplicates")
    return values


class ContextBudgetClass(StrEnum):
    PRIMARY_MODEL_INPUT = "primary_model_input"
    INTERNAL_OPTIMIZATION_INPUT = "internal_optimization_input"


def budget_class_for_execution_scope(
    execution_scope: object,
) -> ContextBudgetClass:
    from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope

    if not isinstance(execution_scope, ModelCallExecutionScope):
        raise ValueError("execution_scope must be ModelCallExecutionScope")
    if execution_scope is ModelCallExecutionScope.PRIMARY_MODEL_CALL:
        return ContextBudgetClass.PRIMARY_MODEL_INPUT
    return ContextBudgetClass.INTERNAL_OPTIMIZATION_INPUT


@dataclass(frozen=True, slots=True)
class ContextSourceGroup:
    group_id: str
    source: ContextFragmentSource
    source_refs: tuple[str, ...]
    source_content_hash: str
    token_estimate: int
    start_sequence: int | None = None
    end_sequence: int | None = None
    required: bool = False
    protected: bool = False
    compressible: bool = False
    droppable: bool = False
    trim_safe: bool = False

    def __post_init__(self) -> None:
        object.__setattr__(self, "group_id", _require_non_empty(self.group_id, "group_id"))
        if not isinstance(self.source, ContextFragmentSource):
            raise ValueError("source must be ContextFragmentSource")
        refs = tuple(self.source_refs)
        if not refs or any(not ref for ref in refs):
            raise ValueError("source_refs must be non-empty with no empty values")
        object.__setattr__(self, "source_refs", _reject_duplicates(refs, "source_refs"))
        object.__setattr__(
            self,
            "source_content_hash",
            _require_non_empty(self.source_content_hash, "source_content_hash"),
        )
        token_estimate = _require_non_negative(self.token_estimate, "token_estimate")
        object.__setattr__(self, "token_estimate", token_estimate)

        has_start = self.start_sequence is not None
        has_end = self.end_sequence is not None
        if has_start != has_end:
            raise ValueError("start_sequence and end_sequence must both be present or both absent")
        if has_start and has_end:
            start = _require_int(self.start_sequence, "start_sequence")
            end = _require_int(self.end_sequence, "end_sequence")
            if start > end:
                raise ValueError("start_sequence must be <= end_sequence")
            object.__setattr__(self, "start_sequence", start)
            object.__setattr__(self, "end_sequence", end)

        if (self.required or self.protected) and self.droppable:
            raise ValueError("required or protected groups cannot be droppable")
        if self.protected and self.trim_safe:
            raise ValueError("protected groups cannot be trim_safe without explicit policy")


@dataclass(frozen=True, slots=True)
class ContextMinimumPreservationRequirements:
    preserve_message_order: bool
    preserve_roles: bool
    preserve_message_ids: bool
    preserve_tool_call_links: bool
    preserve_recent_tail_messages: int
    required_group_ids: tuple[str, ...] = ()
    protected_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        for name in (
            "preserve_message_order",
            "preserve_roles",
            "preserve_message_ids",
            "preserve_tool_call_links",
        ):
            value = object.__getattribute__(self, name)
            if not isinstance(value, bool):
                raise ValueError(f"{name} must be a bool")

        tail = _require_non_negative(
            self.preserve_recent_tail_messages,
            "preserve_recent_tail_messages",
        )
        object.__setattr__(self, "preserve_recent_tail_messages", tail)

        required_ids = tuple(self.required_group_ids)
        protected_ids = tuple(self.protected_group_ids)
        if any(not group_id for group_id in required_ids):
            raise ValueError("required_group_ids must not contain empty values")
        if any(not group_id for group_id in protected_ids):
            raise ValueError("protected_group_ids must not contain empty values")
        object.__setattr__(self, "required_group_ids", _reject_duplicates(required_ids, "required_group_ids"))
        object.__setattr__(
            self,
            "protected_group_ids",
            _reject_duplicates(protected_ids, "protected_group_ids"),
        )


@dataclass(frozen=True, slots=True)
class ContextArtifactLookupInputs:
    tenant_id: str
    context_scope_id: str
    artifact_type: OptimizationArtifactType
    source_content_hash: str
    compression_target: ArtifactCompressionTarget
    lossiness_profile: str
    source_refs: tuple[str, ...] = ()
    source_range: ArtifactSourceRange | None = None
    protected_region_policy_version: str | None = None
    model_family: str | None = None
    locale: str | None = None

    def __post_init__(self) -> None:
        object.__setattr__(self, "tenant_id", _require_non_empty(self.tenant_id, "tenant_id"))
        object.__setattr__(
            self,
            "context_scope_id",
            _require_non_empty(self.context_scope_id, "context_scope_id"),
        )
        if not isinstance(self.artifact_type, OptimizationArtifactType):
            raise ValueError("artifact_type must be OptimizationArtifactType")
        object.__setattr__(
            self,
            "source_content_hash",
            _require_non_empty(self.source_content_hash, "source_content_hash"),
        )
        if not isinstance(self.compression_target, ArtifactCompressionTarget):
            raise ValueError("compression_target must be ArtifactCompressionTarget")
        object.__setattr__(
            self,
            "lossiness_profile",
            _require_non_empty(self.lossiness_profile, "lossiness_profile"),
        )

        has_refs = bool(self.source_refs)
        has_range = self.source_range is not None
        if has_refs == has_range:
            raise ValueError("exactly one of source_refs or source_range must be provided")
        if has_refs:
            refs = tuple(self.source_refs)
            if any(not ref for ref in refs):
                raise ValueError("source_refs must not contain empty values")
            object.__setattr__(self, "source_refs", _reject_duplicates(refs, "source_refs"))
        if self.source_range is not None and not isinstance(self.source_range, ArtifactSourceRange):
            raise ValueError("source_range must be ArtifactSourceRange")
        if self.protected_region_policy_version is not None:
            object.__setattr__(
                self,
                "protected_region_policy_version",
                _require_non_empty(
                    self.protected_region_policy_version,
                    "protected_region_policy_version",
                ),
            )
        if self.model_family is not None:
            object.__setattr__(
                self,
                "model_family",
                _require_non_empty(self.model_family, "model_family"),
            )
        if self.locale is not None:
            object.__setattr__(self, "locale", _require_non_empty(self.locale, "locale"))


def artifact_lookup_key_kwargs_from_context_inputs(
    inputs: ContextArtifactLookupInputs,
) -> dict[str, object]:
    """Return CE-owned lookup dimensions for Nexus to combine with strategy/policy versions."""
    payload: dict[str, object] = {
        "tenant_id": inputs.tenant_id,
        "context_scope_id": inputs.context_scope_id,
        "artifact_type": inputs.artifact_type,
        "source_content_hash": inputs.source_content_hash,
        "compression_target": inputs.compression_target,
        "lossiness_profile": inputs.lossiness_profile,
    }
    if inputs.source_refs:
        payload["source_refs"] = inputs.source_refs
    if inputs.source_range is not None:
        payload["source_range"] = inputs.source_range
    if inputs.protected_region_policy_version is not None:
        payload["protected_region_policy_version"] = inputs.protected_region_policy_version
    if inputs.model_family is not None:
        payload["model_family"] = inputs.model_family
    if inputs.locale is not None:
        payload["locale"] = inputs.locale
    return payload


@dataclass(frozen=True, slots=True)
class ContextArtifactRequirement:
    lookup_inputs: ContextArtifactLookupInputs
    source_group_ids: tuple[str, ...]
    allowed_strategy_ids: tuple[str, ...]
    minimum_preservation: ContextMinimumPreservationRequirements

    def __post_init__(self) -> None:
        if not isinstance(self.lookup_inputs, ContextArtifactLookupInputs):
            raise ValueError("lookup_inputs must be ContextArtifactLookupInputs")
        if not isinstance(self.minimum_preservation, ContextMinimumPreservationRequirements):
            raise ValueError("minimum_preservation must be ContextMinimumPreservationRequirements")
        group_ids = tuple(self.source_group_ids)
        if not group_ids:
            raise ValueError("source_group_ids must be non-empty")
        object.__setattr__(self, "source_group_ids", _reject_duplicates(group_ids, "source_group_ids"))
        strategy_ids = tuple(self.allowed_strategy_ids)
        object.__setattr__(
            self,
            "allowed_strategy_ids",
            _reject_duplicates(strategy_ids, "allowed_strategy_ids"),
        )


@dataclass(frozen=True, slots=True)
class ContextSourceBudgetAllocation:
    source: ContextFragmentSource
    allocated_tokens: int
    selected_group_ids: tuple[str, ...]
    excluded_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        if not isinstance(self.source, ContextFragmentSource):
            raise ValueError("source must be ContextFragmentSource")
        object.__setattr__(
            self,
            "allocated_tokens",
            _require_non_negative(self.allocated_tokens, "allocated_tokens"),
        )
        selected = tuple(self.selected_group_ids)
        excluded = tuple(self.excluded_group_ids)
        if any(not group_id for group_id in selected):
            raise ValueError("selected_group_ids must not contain empty values")
        if any(not group_id for group_id in excluded):
            raise ValueError("excluded_group_ids must not contain empty values")
        object.__setattr__(self, "selected_group_ids", _reject_duplicates(selected, "selected_group_ids"))
        object.__setattr__(self, "excluded_group_ids", _reject_duplicates(excluded, "excluded_group_ids"))
        if set(selected) & set(excluded):
            raise ValueError("selected and excluded group IDs must be disjoint")


@dataclass(frozen=True, slots=True)
class ContextPlan:
    execution_scope: object
    budget_class: ContextBudgetClass
    resolved_global_budget_tokens: int
    estimated_total_tokens: int
    source_groups: tuple[ContextSourceGroup, ...]
    source_allocations: tuple[ContextSourceBudgetAllocation, ...]
    selected_group_ids: tuple[str, ...]
    excluded_group_ids: tuple[str, ...]
    required_group_ids: tuple[str, ...]
    protected_group_ids: tuple[str, ...]
    compressible_group_ids: tuple[str, ...]
    droppable_group_ids: tuple[str, ...]
    trim_safe_group_ids: tuple[str, ...]
    optimization_required: bool
    artifact_requirement: ContextArtifactRequirement | None
    final_validation_requirements: tuple[str, ...]

    def __post_init__(self) -> None:
        from intergrax.runtime.context_lifecycle.contracts import ModelCallExecutionScope

        scope = _require_enum(self.execution_scope, ModelCallExecutionScope, "execution_scope")
        object.__setattr__(self, "execution_scope", scope)
        if not isinstance(self.budget_class, ContextBudgetClass):
            raise ValueError("budget_class must be ContextBudgetClass")
        expected = budget_class_for_execution_scope(scope)
        if self.budget_class is not expected:
            raise ValueError("budget_class must match execution_scope")

        object.__setattr__(
            self,
            "resolved_global_budget_tokens",
            _require_positive(self.resolved_global_budget_tokens, "resolved_global_budget_tokens"),
        )
        object.__setattr__(
            self,
            "estimated_total_tokens",
            _require_non_negative(self.estimated_total_tokens, "estimated_total_tokens"),
        )

        groups = tuple(self.source_groups)
        group_ids = [group.group_id for group in groups]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("source group IDs must be unique")
        object.__setattr__(self, "source_groups", groups)
        group_id_set = set(group_ids)

        for field_name in (
            "selected_group_ids",
            "excluded_group_ids",
            "required_group_ids",
            "protected_group_ids",
            "compressible_group_ids",
            "droppable_group_ids",
            "trim_safe_group_ids",
            "final_validation_requirements",
        ):
            values = tuple(object.__getattribute__(self, field_name))
            if field_name == "final_validation_requirements":
                if any(not value for value in values):
                    raise ValueError("final_validation_requirements must not contain empty values")
            else:
                if any(not value for value in values):
                    raise ValueError(f"{field_name} must not contain empty values")
            object.__setattr__(self, field_name, _reject_duplicates(values, field_name))

        selected = set(self.selected_group_ids)
        excluded = set(self.excluded_group_ids)
        if selected & excluded:
            raise ValueError("selected and excluded group IDs must be disjoint")
        for group_id in selected | excluded | set(self.required_group_ids) | set(self.protected_group_ids):
            if group_id not in group_id_set:
                raise ValueError(f"unknown group ID referenced: {group_id}")
        for group_id in self.required_group_ids:
            if group_id not in selected:
                raise ValueError("required groups must be selected")
        for group_id in self.protected_group_ids:
            if group_id not in selected:
                raise ValueError("protected groups must be selected")
            if group_id in excluded:
                raise ValueError("protected groups must not be excluded")

        if self.optimization_required:
            if self.artifact_requirement is None:
                raise ValueError("optimization_required requires artifact_requirement")
        elif self.artifact_requirement is not None:
            raise ValueError("artifact_requirement must be None when optimization_required is false")

        if self.artifact_requirement is not None:
            compressible = set(self.compressible_group_ids)
            protected = set(self.protected_group_ids)
            for group_id in self.artifact_requirement.source_group_ids:
                if group_id not in selected:
                    raise ValueError("artifact source groups must be selected")
                if group_id not in compressible:
                    raise ValueError("artifact source groups must be compressible")
                if group_id in protected:
                    raise ValueError("artifact source groups must not be protected")


def _require_enum(value: object, enum_type: type, field_name: str) -> object:
    if not isinstance(value, enum_type):
        raise ValueError(f"{field_name} must be {enum_type.__name__}")
    return value


MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT = "mandatory_context_exceeds_model_limit"
NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET = "no_eligible_context_optimization_target"


class ContextPlanningError(ValueError):
    """Deterministic planning failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

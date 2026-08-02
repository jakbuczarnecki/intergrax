# © Artur Czarnecki. All rights reserved.

"""Context planning contracts (CTX-UCL-3)."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from intergrax.context.contracts import ContextFragmentSource
from intergrax.runtime.context_lifecycle.contracts import (
    ArtifactCompressionTarget,
    ArtifactSourceRange,
    ModelCallExecutionScope,
    OptimizationArtifactType,
)


def _require_str(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field_name} must be a non-empty string")
    return value


def _require_bool(value: object, field_name: str) -> bool:
    if not isinstance(value, bool):
        raise ValueError(f"{field_name} must be a bool")
    return value


def _require_instance(value: object, expected_type: type, field_name: str) -> object:
    if not isinstance(value, expected_type):
        raise ValueError(f"{field_name} must be {expected_type.__name__}")
    return value


def _require_tuple_items(
    values: object,
    field_name: str,
    *,
    item_type: type | None = None,
) -> tuple[object, ...]:
    if not isinstance(values, tuple):
        raise ValueError(f"{field_name} must be a tuple")
    if item_type is not None:
        for index, item in enumerate(values):
            if not isinstance(item, item_type):
                raise ValueError(f"{field_name}[{index}] must be {item_type.__name__}")
    return values


def _require_non_empty(value: str, field_name: str) -> str:
    return _require_str(value, field_name)


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
        object.__setattr__(self, "group_id", _require_str(self.group_id, "group_id"))
        _require_instance(self.source, ContextFragmentSource, "source")
        refs = _require_tuple_items(self.source_refs, "source_refs", item_type=str)
        if not refs:
            raise ValueError("source_refs must be non-empty")
        for ref in refs:
            _require_str(ref, "source_refs item")
        object.__setattr__(self, "source_refs", _reject_duplicates(refs, "source_refs"))  # type: ignore[arg-type]
        object.__setattr__(
            self,
            "source_content_hash",
            _require_str(self.source_content_hash, "source_content_hash"),
        )
        token_estimate = _require_non_negative(self.token_estimate, "token_estimate")
        object.__setattr__(self, "token_estimate", token_estimate)

        object.__setattr__(self, "required", _require_bool(self.required, "required"))
        object.__setattr__(self, "protected", _require_bool(self.protected, "protected"))
        object.__setattr__(self, "compressible", _require_bool(self.compressible, "compressible"))
        object.__setattr__(self, "droppable", _require_bool(self.droppable, "droppable"))
        object.__setattr__(self, "trim_safe", _require_bool(self.trim_safe, "trim_safe"))

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
        object.__setattr__(
            self,
            "preserve_message_order",
            _require_bool(self.preserve_message_order, "preserve_message_order"),
        )
        object.__setattr__(
            self,
            "preserve_roles",
            _require_bool(self.preserve_roles, "preserve_roles"),
        )
        object.__setattr__(
            self,
            "preserve_message_ids",
            _require_bool(self.preserve_message_ids, "preserve_message_ids"),
        )
        object.__setattr__(
            self,
            "preserve_tool_call_links",
            _require_bool(self.preserve_tool_call_links, "preserve_tool_call_links"),
        )

        tail = _require_non_negative(
            self.preserve_recent_tail_messages,
            "preserve_recent_tail_messages",
        )
        object.__setattr__(self, "preserve_recent_tail_messages", tail)

        required_ids = _require_tuple_items(self.required_group_ids, "required_group_ids", item_type=str)
        protected_ids = _require_tuple_items(self.protected_group_ids, "protected_group_ids", item_type=str)
        for group_id in required_ids:
            _require_str(group_id, "required_group_ids item")
        for group_id in protected_ids:
            _require_str(group_id, "protected_group_ids item")
        object.__setattr__(self, "required_group_ids", _reject_duplicates(required_ids, "required_group_ids"))  # type: ignore[arg-type]
        object.__setattr__(
            self,
            "protected_group_ids",
            _reject_duplicates(protected_ids, "protected_group_ids"),  # type: ignore[arg-type]
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
        object.__setattr__(self, "tenant_id", _require_str(self.tenant_id, "tenant_id"))
        object.__setattr__(self, "context_scope_id", _require_str(self.context_scope_id, "context_scope_id"))
        _require_instance(self.artifact_type, OptimizationArtifactType, "artifact_type")
        object.__setattr__(
            self,
            "source_content_hash",
            _require_str(self.source_content_hash, "source_content_hash"),
        )
        _require_instance(self.compression_target, ArtifactCompressionTarget, "compression_target")
        object.__setattr__(self, "lossiness_profile", _require_str(self.lossiness_profile, "lossiness_profile"))

        has_refs = bool(self.source_refs)
        has_range = self.source_range is not None
        if has_refs == has_range:
            raise ValueError("exactly one of source_refs or source_range must be provided")
        if has_refs:
            refs = _require_tuple_items(self.source_refs, "source_refs", item_type=str)
            for ref in refs:
                _require_str(ref, "source_refs item")
            object.__setattr__(self, "source_refs", _reject_duplicates(refs, "source_refs"))  # type: ignore[arg-type]
        if self.source_range is not None:
            _require_instance(self.source_range, ArtifactSourceRange, "source_range")
        if self.protected_region_policy_version is not None:
            object.__setattr__(
                self,
                "protected_region_policy_version",
                _require_str(self.protected_region_policy_version, "protected_region_policy_version"),
            )
        if self.model_family is not None:
            object.__setattr__(self, "model_family", _require_str(self.model_family, "model_family"))
        if self.locale is not None:
            object.__setattr__(self, "locale", _require_str(self.locale, "locale"))


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
        _require_instance(self.lookup_inputs, ContextArtifactLookupInputs, "lookup_inputs")
        _require_instance(
            self.minimum_preservation,
            ContextMinimumPreservationRequirements,
            "minimum_preservation",
        )
        group_ids = _require_tuple_items(self.source_group_ids, "source_group_ids", item_type=str)
        if not group_ids:
            raise ValueError("source_group_ids must be non-empty")
        for group_id in group_ids:
            _require_str(group_id, "source_group_ids item")
        object.__setattr__(self, "source_group_ids", _reject_duplicates(group_ids, "source_group_ids"))  # type: ignore[arg-type]
        strategy_ids = _require_tuple_items(
            self.allowed_strategy_ids,
            "allowed_strategy_ids",
            item_type=str,
        )
        if not strategy_ids:
            raise ValueError("allowed_strategy_ids must be non-empty")
        for strategy_id in strategy_ids:
            _require_str(strategy_id, "allowed_strategy_ids item")
        object.__setattr__(
            self,
            "allowed_strategy_ids",
            _reject_duplicates(strategy_ids, "allowed_strategy_ids"),  # type: ignore[arg-type]
        )


@dataclass(frozen=True, slots=True)
class ContextSourceBudgetAllocation:
    source: ContextFragmentSource
    allocated_tokens: int
    selected_group_ids: tuple[str, ...]
    excluded_group_ids: tuple[str, ...] = ()

    def __post_init__(self) -> None:
        _require_instance(self.source, ContextFragmentSource, "source")
        object.__setattr__(
            self,
            "allocated_tokens",
            _require_non_negative(self.allocated_tokens, "allocated_tokens"),
        )
        selected = _require_tuple_items(self.selected_group_ids, "selected_group_ids", item_type=str)
        excluded = _require_tuple_items(self.excluded_group_ids, "excluded_group_ids", item_type=str)
        for group_id in selected:
            _require_str(group_id, "selected_group_ids item")
        for group_id in excluded:
            _require_str(group_id, "excluded_group_ids item")
        object.__setattr__(self, "selected_group_ids", _reject_duplicates(selected, "selected_group_ids"))  # type: ignore[arg-type]
        object.__setattr__(self, "excluded_group_ids", _reject_duplicates(excluded, "excluded_group_ids"))  # type: ignore[arg-type]
        if set(selected) & set(excluded):
            raise ValueError("selected and excluded group IDs must be disjoint")


@dataclass(frozen=True, slots=True)
class ContextPlan:
    execution_scope: ModelCallExecutionScope
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
        scope = _require_instance(self.execution_scope, ModelCallExecutionScope, "execution_scope")
        object.__setattr__(self, "execution_scope", scope)
        _require_instance(self.budget_class, ContextBudgetClass, "budget_class")
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
        object.__setattr__(
            self,
            "optimization_required",
            _require_bool(self.optimization_required, "optimization_required"),
        )

        groups = _require_tuple_items(self.source_groups, "source_groups", item_type=ContextSourceGroup)
        group_ids = [group.group_id for group in groups]  # type: ignore[union-attr]
        if len(group_ids) != len(set(group_ids)):
            raise ValueError("source group IDs must be unique")
        object.__setattr__(self, "source_groups", groups)
        groups_by_id = {group.group_id: group for group in groups}  # type: ignore[union-attr]
        group_id_set = set(group_ids)

        allocations = _require_tuple_items(
            self.source_allocations,
            "source_allocations",
            item_type=ContextSourceBudgetAllocation,
        )
        object.__setattr__(self, "source_allocations", allocations)

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
            values = _require_tuple_items(object.__getattribute__(self, field_name), field_name)
            if field_name == "final_validation_requirements":
                for value in values:
                    _require_str(value, "final_validation_requirements item")
            else:
                for value in values:
                    _require_str(value, f"{field_name} item")
            object.__setattr__(self, field_name, _reject_duplicates(values, field_name))  # type: ignore[arg-type]

        selected = set(self.selected_group_ids)
        excluded = set(self.excluded_group_ids)
        if selected & excluded:
            raise ValueError("selected and excluded group IDs must be disjoint")
        if selected | excluded != group_id_set:
            raise ValueError("selected and excluded group IDs must cover all source groups")

        classification_fields = (
            ("required_group_ids", "required"),
            ("protected_group_ids", "protected"),
            ("compressible_group_ids", "compressible"),
            ("droppable_group_ids", "droppable"),
            ("trim_safe_group_ids", "trim_safe"),
        )
        for field_name, flag_name in classification_fields:
            expected_ids = tuple(
                group.group_id for group in groups if getattr(group, flag_name)  # type: ignore[union-attr]
            )
            actual_ids = tuple(object.__getattribute__(self, field_name))
            if actual_ids != expected_ids:
                raise ValueError(f"{field_name} must match group {flag_name} flags")

        for group_id in (
            set(self.required_group_ids)
            | set(self.protected_group_ids)
            | set(self.compressible_group_ids)
            | set(self.droppable_group_ids)
            | set(self.trim_safe_group_ids)
        ):
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

        allocated_selected: set[str] = set()
        allocated_excluded: set[str] = set()
        for allocation in allocations:
            for group_id in allocation.selected_group_ids:  # type: ignore[union-attr]
                if group_id not in group_id_set:
                    raise ValueError(f"unknown group ID referenced: {group_id}")
                if groups_by_id[group_id].source is not allocation.source:  # type: ignore[union-attr]
                    raise ValueError("allocation source mismatch")
                if group_id in allocated_selected:
                    raise ValueError("duplicate allocation for group")
                allocated_selected.add(group_id)
            for group_id in allocation.excluded_group_ids:  # type: ignore[union-attr]
                if group_id not in group_id_set:
                    raise ValueError(f"unknown group ID referenced: {group_id}")
                if groups_by_id[group_id].source is not allocation.source:  # type: ignore[union-attr]
                    raise ValueError("allocation source mismatch")
                if group_id in allocated_excluded:
                    raise ValueError("duplicate allocation for group")
                allocated_excluded.add(group_id)
        if allocated_selected != selected:
            raise ValueError("allocation missing group")
        if allocated_excluded != excluded:
            raise ValueError("allocation missing group")

        if self.optimization_required:
            if self.artifact_requirement is None:
                raise ValueError("optimization_required requires artifact_requirement")
        elif self.artifact_requirement is not None:
            raise ValueError("artifact_requirement must be None when optimization_required is false")

        if self.artifact_requirement is not None:
            if not isinstance(self.artifact_requirement, ContextArtifactRequirement):
                raise ValueError("artifact_requirement must be ContextArtifactRequirement")
            compressible = set(self.compressible_group_ids)
            protected = set(self.protected_group_ids)
            artifact_sources = set(self.artifact_requirement.source_group_ids)
            for group_id in artifact_sources:
                if group_id not in group_id_set:
                    raise ValueError(f"unknown group ID referenced: {group_id}")
                if group_id not in selected:
                    raise ValueError("artifact source groups must be selected")
                if group_id not in compressible:
                    raise ValueError("artifact source groups must be compressible")
                if group_id in protected:
                    raise ValueError("artifact source groups must not be protected")
            preservation = self.artifact_requirement.minimum_preservation
            if tuple(preservation.required_group_ids) != tuple(self.required_group_ids):
                raise ValueError("preservation required_group_ids must match plan required_group_ids")
            if tuple(preservation.protected_group_ids) != tuple(self.protected_group_ids):
                raise ValueError("preservation protected_group_ids must match plan protected_group_ids")
            for group_id in preservation.required_group_ids:
                if group_id not in selected:
                    raise ValueError("preservation required groups must be selected")
            for group_id in preservation.protected_group_ids:
                if group_id not in selected:
                    raise ValueError("preservation protected groups must be selected")
            for group_id in preservation.required_group_ids + preservation.protected_group_ids:
                if group_id in artifact_sources:
                    raise ValueError("preservation IDs must be disjoint from artifact source_group_ids")


MANDATORY_CONTEXT_EXCEEDS_MODEL_LIMIT = "mandatory_context_exceeds_model_limit"
NO_ELIGIBLE_CONTEXT_OPTIMIZATION_TARGET = "no_eligible_context_optimization_target"


class ContextPlanningError(ValueError):
    """Deterministic planning failure."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason

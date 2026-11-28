# © Artur Czarnecki. All rights reserved.

"""Canonical ADAS typed contracts (AHI-ADAS-10, W1 configuration-only LAB profile)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from datetime import UTC, datetime
from enum import Enum
from types import MappingProxyType
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, ValidationError, field_serializer, field_validator, model_validator

from intergrax.contracts.canonical_payload_hash import canonical_json_bytes
from intergrax.contracts.orchestration_enums import MergeStrategy
from intergrax.runtime.adaptive.contracts import ProfileArtifactType

ADAS_CONTRACT_SCHEMA_VERSION = "1.0"
_SHA256_HEX = re.compile(r"^[a-f0-9]{64}$")
_W1_MERGE_STRATEGIES = frozenset({MergeStrategy.LAST_WINS, MergeStrategy.CONCAT})
_SEARCH_RUN_STATUSES = frozenset({"pending", "running", "completed", "failed", "cancelled"})
_ADAS_JSON_MUTATION_ERROR = "ADAS contract JSON payload is immutable"

type AdasJsonPrimitive = str | int | float | bool | None
type AdasJsonValue = AdasJsonPrimitive | list[AdasJsonValue] | dict[str, AdasJsonValue]
type AdasJsonObject = dict[str, AdasJsonValue]


def _validate_adas_json_value(
    value: object,
    *,
    field_name: str,
    path: str = "",
) -> AdasJsonValue:
    if isinstance(value, Enum):
        child_path = path.rstrip(".") if path else field_name
        raise ValueError(f"{field_name} must contain JSON-compatible values at '{child_path}'")
    if value is None or isinstance(value, bool):
        return value
    if isinstance(value, str):
        return value
    if isinstance(value, int) and not isinstance(value, bool):
        return value
    if isinstance(value, float):
        if not math.isfinite(value):
            label = path.rstrip(".") if path else field_name
            raise ValueError(f"{field_name} must not contain non-finite float at '{label}'")
        return value
    if isinstance(value, Mapping):
        result: dict[str, AdasJsonValue] = {}
        for raw_key, child in value.items():
            if not isinstance(raw_key, str):
                raise ValueError(f"{field_name} keys must be strings")
            result[raw_key] = _validate_adas_json_value(
                child,
                field_name=field_name,
                path=f"{path}{raw_key}.",
            )
        return result
    if isinstance(value, list):
        return [
            _validate_adas_json_value(child, field_name=field_name, path=f"{path}[{index}].")
            for index, child in enumerate(value)
        ]
    child_path = path.rstrip(".") if path else field_name
    raise ValueError(f"{field_name} must contain JSON-compatible values at '{child_path}'")


def _json_value_to_plain(value: AdasJsonValue) -> AdasJsonValue:
    if isinstance(value, _FrozenAdasJsonObject):
        return value.to_plain()
    if isinstance(value, _FrozenAdasJsonArray):
        return value.to_plain()
    if isinstance(value, dict):
        return {key: _json_value_to_plain(child) for key, child in value.items()}
    if isinstance(value, list):
        return [_json_value_to_plain(child) for child in value]
    return value


class _FrozenAdasJsonArray:
    __slots__ = ("_items",)

    def __init__(self, items: list[AdasJsonValue]) -> None:
        object.__setattr__(self, "_items", tuple(_freeze_adas_json_value(item) for item in items))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError(_ADAS_JSON_MUTATION_ERROR)

    def __getitem__(self, index: int | slice) -> AdasJsonValue | _FrozenAdasJsonArray:
        result = self._items[index]
        if isinstance(index, slice):
            return _FrozenAdasJsonArray(list(result))
        return result

    def __iter__(self) -> Iterator[AdasJsonValue]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenAdasJsonArray):
            return self._items == other._items
        if isinstance(other, list):
            return self.to_plain() == other
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self.to_plain())

    def to_plain(self) -> list[AdasJsonValue]:
        return [_json_value_to_plain(item) for item in self._items]

    def __setitem__(self, index: int | slice, value: object) -> None:
        raise TypeError(_ADAS_JSON_MUTATION_ERROR)

    def __delitem__(self, index: int | slice) -> None:
        raise TypeError(_ADAS_JSON_MUTATION_ERROR)


class _FrozenAdasJsonObject:
    __slots__ = ("_proxy",)

    def __init__(self, value: dict[str, AdasJsonValue]) -> None:
        frozen_inner = {key: _freeze_adas_json_value(child) for key, child in value.items()}
        object.__setattr__(self, "_proxy", MappingProxyType(frozen_inner))

    def __setattr__(self, name: str, value: object) -> None:
        raise TypeError(_ADAS_JSON_MUTATION_ERROR)

    def __getitem__(self, key: str) -> AdasJsonValue:
        return self._proxy[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._proxy)

    def __len__(self) -> int:
        return len(self._proxy)

    def __contains__(self, key: object) -> bool:
        return key in self._proxy

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenAdasJsonObject):
            return self.to_plain() == other.to_plain()
        if isinstance(other, Mapping):
            return self.to_plain() == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self.to_plain())

    def keys(self) -> Iterator[str]:
        return iter(self._proxy.keys())

    def values(self) -> Iterator[AdasJsonValue]:
        return iter(self._proxy.values())

    def items(self) -> Iterator[tuple[str, AdasJsonValue]]:
        return iter(self._proxy.items())

    def get(self, key: str, default: AdasJsonValue | None = None) -> AdasJsonValue | None:
        return self._proxy.get(key, default)

    def to_plain(self) -> dict[str, AdasJsonValue]:
        return {key: _json_value_to_plain(child) for key, child in self._proxy.items()}

    def __setitem__(self, key: str, value: object) -> None:
        raise TypeError(_ADAS_JSON_MUTATION_ERROR)

    def __delitem__(self, key: str) -> None:
        raise TypeError(_ADAS_JSON_MUTATION_ERROR)


def _freeze_adas_json_value(value: AdasJsonValue) -> AdasJsonValue:
    if isinstance(value, dict):
        return _FrozenAdasJsonObject(value)  # pyright: ignore[reportReturnType]
    if isinstance(value, list):
        return _FrozenAdasJsonArray(value)  # pyright: ignore[reportReturnType]
    return value


def freeze_adas_json_object(value: Mapping[str, AdasJsonValue]) -> Mapping[str, AdasJsonValue]:
    if isinstance(value, _FrozenAdasJsonObject):
        return value
    return _FrozenAdasJsonObject(dict(value))  # pyright: ignore[reportReturnType]


def adas_json_to_plain(value: Mapping[str, AdasJsonValue]) -> dict[str, AdasJsonValue]:
    if isinstance(value, _FrozenAdasJsonObject):
        return value.to_plain()
    return {key: _json_value_to_plain(child) for key, child in value.items()}


def _parse_adas_json_object(value: object, *, field_name: str) -> dict[str, AdasJsonValue]:
    if value is None:
        return {}
    if not isinstance(value, Mapping):
        raise ValueError(f"{field_name} must be a JSON object")
    validated = _validate_adas_json_value(value, field_name=field_name)
    if not isinstance(validated, dict):
        raise ValueError(f"{field_name} must be a JSON object")
    return validated


def _freeze_parsed_json_object(value: dict[str, AdasJsonValue]) -> Mapping[str, AdasJsonValue]:
    return freeze_adas_json_object(value)


def _adas_json_field_before(value: object, *, field_name: str) -> dict[str, AdasJsonValue] | None:
    if value is None:
        return None
    return _parse_adas_json_object(value, field_name=field_name)


def _adas_json_field_after(
    value: dict[str, AdasJsonValue] | None,
) -> Mapping[str, AdasJsonValue] | None:
    if value is None:
        return None
    return _freeze_parsed_json_object(value)


def _serialize_adas_json_field(value: Mapping[str, AdasJsonValue] | None) -> dict[str, AdasJsonValue] | None:
    if value is None:
        return None
    return adas_json_to_plain(value)


def _reject_executable_payload_tree(value: Mapping[str, AdasJsonValue], *, label: str) -> None:
    forbidden = ("model", "tool", "capability", "permission", "policy", "source", "eval", "exec", "__import__")

    def walk(node: AdasJsonValue, path: str) -> None:
        if isinstance(node, dict):
            for key, child in node.items():
                lowered = key.lower()
                if any(marker in lowered for marker in forbidden):
                    raise ValueError(f"{label} must not carry unauthorized payloads at '{path}.{key}'")
                walk(child, f"{path}.{key}")
        elif isinstance(node, list):
            for index, child in enumerate(node):
                walk(child, f"{path}[{index}]")

    for key, child in value.items():
        lowered = key.lower()
        if any(marker in lowered for marker in forbidden):
            raise ValueError(f"{label} must not carry unauthorized payloads at '{key}'")
        walk(child, key)


def _empty_adas_json_object() -> Mapping[str, AdasJsonValue]:
    return _freeze_parsed_json_object({})


def _non_blank(value: str, *, field: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError(f"{field} must not be blank")
    return normalized


def _validate_sha256_digest(value: str) -> str:
    if value.strip() != value:
        raise ValueError("digest must be lowercase 64-character hexadecimal SHA-256")
    if any(char.isupper() for char in value):
        raise ValueError("digest must be lowercase 64-character hexadecimal SHA-256")
    if not _SHA256_HEX.fullmatch(value):
        raise ValueError("digest must be lowercase 64-character hexadecimal SHA-256")
    return value


def _finite_non_negative(value: float, *, field: str) -> float:
    if not math.isfinite(value):
        raise ValueError(f"{field} must be finite")
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def _finite_non_negative_int(value: int, *, field: str) -> int:
    if value < 0:
        raise ValueError(f"{field} must not be negative")
    return value


def _aware_datetime(value: datetime) -> datetime:
    if value.tzinfo is None or value.tzinfo.utcoffset(value) is None:
        raise ValueError("timestamp must be timezone-aware")
    return value.astimezone(UTC)


class AgentCandidateStatus(str, Enum):  # noqa: UP042 — canonical ADAS serialized enum values
    DRAFT = "draft"
    SCAFFOLDED = "scaffolded"
    STATIC_GATE_FAILED = "static_gate_failed"
    STATIC_GATE_PASSED = "static_gate_passed"
    EVALUATED = "evaluated"
    REJECTED = "rejected"
    SHADOW = "shadow"
    CANARY = "canary"
    PROMOTED = "promoted"
    RETIRED = "retired"
    ARCHIVED = "archived"


class SearchSelectionOutcome(str, Enum):  # noqa: UP042
    SELECT_CANDIDATE = "select_candidate"
    RETAIN_BASELINE = "retain_baseline"
    ABSTAIN = "abstain"


class W1TerminalEvidenceOutcome(str, Enum):  # noqa: UP042
    STATIC_PRE_EXECUTION_REJECTION = "static_pre_execution_rejection"
    EXECUTION_FAILED_OR_CANCELLED = "execution_failed_or_cancelled"
    EVALUATION_COMPLETE = "evaluation_complete"


class AdasExecutionReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    execution_id: str

    @field_validator("execution_id")
    @classmethod
    def _execution(cls, value: str) -> str:
        return _non_blank(value, field="execution_id")


class AdasEvalRunReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    eval_run_id: str

    @field_validator("eval_run_id")
    @classmethod
    def _eval_run(cls, value: str) -> str:
        return _non_blank(value, field="eval_run_id")


class AdasEvalComparisonReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    comparison_id: str

    @field_validator("comparison_id")
    @classmethod
    def _comparison(cls, value: str) -> str:
        return _non_blank(value, field="comparison_id")


class AggregateAccountingEvidenceReference(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    evidence_ref: str
    completeness: str
    measurement_uncertainty: str | None = None

    @field_validator("evidence_ref", "completeness")
    @classmethod
    def _required(cls, value: str) -> str:
        return _non_blank(value, field="reference")


class StructuredSearchStopReason(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    reason_code: str
    detail: str | None = None

    @field_validator("reason_code")
    @classmethod
    def _code(cls, value: str) -> str:
        return _non_blank(value, field="reason_code")


class AdasW1LabStopKind(str, Enum):  # noqa: UP042
    PRE_EXECUTION_REJECTION = "pre_execution_rejection"
    POST_GATE_ADMISSION_REJECTION = "post_gate_admission_rejection"
    EXECUTION_FAILURE = "execution_failure"
    EXECUTION_CANCELLATION = "execution_cancellation"


class AdasW1LabStopEvidence(BaseModel):
    """Typed W1 LAB terminal stop evidence bound to archive scope."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: AdasW1LabStopKind
    candidate_id: str
    search_run_id: str
    tenant_id: str
    workspace_id: str
    execution_ref: AdasExecutionReference | None = None
    structured_stop_reason: StructuredSearchStopReason | None = None
    detail: str | None = None

    @field_validator("candidate_id", "search_run_id", "tenant_id", "workspace_id")
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @model_validator(mode="after")
    def _kind_rules(self) -> Self:
        if self.kind is AdasW1LabStopKind.PRE_EXECUTION_REJECTION:
            if self.execution_ref is not None:
                raise ValueError("pre-execution rejection must not claim execution")
            if self.detail is None or not self.detail.strip():
                raise ValueError("pre-execution rejection requires explicit detail")
        if self.kind is AdasW1LabStopKind.POST_GATE_ADMISSION_REJECTION:
            if self.execution_ref is not None:
                raise ValueError("post-gate rejection must not fabricate execution")
            if self.detail is None or not self.detail.strip():
                raise ValueError("post-gate rejection requires explicit detail")
        if self.kind in {AdasW1LabStopKind.EXECUTION_FAILURE, AdasW1LabStopKind.EXECUTION_CANCELLATION}:
            if self.execution_ref is None:
                raise ValueError("execution outcome requires execution reference")
            if self.structured_stop_reason is None:
                raise ValueError("execution outcome requires structured stop reason")
        return self


class AdasOperationalEnvelope(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = ADAS_CONTRACT_SCHEMA_VERSION
    created_at: datetime
    updated_at: datetime | None = None
    created_by: str
    tenant_id: str
    workspace_id: str
    accepted_scope_ref: str
    lab_only_posture: bool = True
    approved_by: str | None = None
    approval_id: str | None = None
    policy_context_id: str
    correlation_id: str
    trace_id: str
    idempotency_key: str | None = None
    artifact_hash: str | None = None
    source_hash: str | None = None
    evaluation_bundle_id: str | None = None
    risk_class: str | None = None
    retention_policy: str

    @field_validator("created_by", "tenant_id", "workspace_id", "accepted_scope_ref", "retention_policy")
    @classmethod
    def _required_identities(cls, value: str) -> str:
        return _non_blank(value, field="identity")

    @field_validator("created_at", "updated_at")
    @classmethod
    def _timestamps(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_datetime(value)

    @field_validator("artifact_hash", "source_hash")
    @classmethod
    def _optional_digest(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256_digest(value)

    @field_validator("correlation_id", "trace_id", "policy_context_id")
    @classmethod
    def _w1_operational_context(cls, value: str) -> str:
        return _non_blank(value, field="operational context")

    @model_validator(mode="after")
    def _lab_posture(self) -> Self:
        if not self.lab_only_posture:
            raise ValueError("W1 requires LAB-only posture")
        return self


class AggregateResourceLimits(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    max_candidates: int = 1
    max_search_iterations: int = 1
    max_root_executions: int = 2
    aggregate_token_allowance: float
    aggregate_monetary_allowance: float
    deadline_at: datetime

    @field_validator("max_candidates", "max_search_iterations", "max_root_executions", mode="before")
    @classmethod
    def _w1_limits(cls, value: object) -> int:
        if type(value) is not int:
            raise ValueError("limit must be a strict integer")
        return _finite_non_negative_int(value, field="limit")

    @field_validator("aggregate_token_allowance", "aggregate_monetary_allowance", mode="before")
    @classmethod
    def _budgets(cls, value: object) -> float:
        return _validate_measured_scalar(value, label="budget")

    @field_validator("deadline_at")
    @classmethod
    def _deadline(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def _enforce_w1_caps(self) -> Self:
        if self.max_candidates != 1 or self.max_search_iterations != 1 or self.max_root_executions != 2:
            raise ValueError("W1 aggregate limits must cap candidates, iterations and root executions")
        return self


class BaselineConfigurationRef(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_version_id: str
    baseline_configuration_digest: str
    artifact_type: ProfileArtifactType

    @field_validator("baseline_version_id")
    @classmethod
    def _version(cls, value: str) -> str:
        return _non_blank(value, field="baseline_version_id")

    @field_validator("baseline_configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)


class ConfigurationDelta(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    admitted_field: str = "merge_strategy"
    prior_value: MergeStrategy
    replacement_value: MergeStrategy

    @field_validator("admitted_field")
    @classmethod
    def _admitted_surface(cls, value: str) -> str:
        normalized = _non_blank(value, field="admitted_field")
        if normalized != "merge_strategy":
            raise ValueError("W1 admits only merge_strategy configuration deltas")
        return normalized

    @model_validator(mode="after")
    def _non_noop(self) -> Self:
        if self.prior_value == self.replacement_value:
            raise ValueError("configuration delta must not be a no-op")
        if self.prior_value not in _W1_MERGE_STRATEGIES or self.replacement_value not in _W1_MERGE_STRATEGIES:
            raise ValueError("W1 admits only last_wins and concat merge strategies")
        return self


class ResourceAccountingEvidence(BaseModel):
    """Explicit accounting; omit fields that were not measured (no default zero)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    tokens_consumed: float | None = None
    monetary_consumed: float | None = None
    root_executions_consumed: int | None = None

    @field_validator("tokens_consumed", "monetary_consumed", mode="before")
    @classmethod
    def _measured_floats(cls, value: object) -> float | None:
        if value is None:
            return None
        return _validate_measured_scalar(value, label="consumption")

    @field_validator("root_executions_consumed", mode="before")
    @classmethod
    def _measured_int(cls, value: object) -> int | None:
        if value is None:
            return None
        return _validate_measured_root_executions(value)


class AdasW1SelectionAccountingEvidence(BaseModel):
    """Frozen W1 selection accounting representation (cost_report payload authority)."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    accounting_evidence_ref: str
    completeness: str
    tokens_consumed: float
    monetary_consumed: float
    root_executions_consumed: int
    candidate_id: str
    objective_id: str
    search_run_id: str
    tenant_id: str
    workspace_id: str
    baseline_execution_ref: AdasExecutionReference
    candidate_execution_ref: AdasExecutionReference
    aggregate_limits: AggregateResourceLimits
    accepted_scope_ref: str

    @field_validator(
        "accounting_evidence_ref",
        "completeness",
        "candidate_id",
        "objective_id",
        "search_run_id",
        "tenant_id",
        "workspace_id",
        "accepted_scope_ref",
    )
    @classmethod
    def _non_blank_accounting_identity(cls, value: str) -> str:
        return _non_blank(value, field="accounting identity")

    @field_validator("tokens_consumed", "monetary_consumed", mode="before")
    @classmethod
    def _accounting_amounts(cls, value: object) -> float:
        return _validate_measured_scalar(value, label="accounting measurement")

    @field_validator("root_executions_consumed", mode="before")
    @classmethod
    def _accounting_roots(cls, value: object) -> int:
        return _validate_measured_root_executions(value)

    @model_validator(mode="after")
    def _w1_accounting_shape(self) -> Self:
        if self.completeness != "complete":
            raise ValueError("SELECT_CANDIDATE requires complete accounting status")
        if self.root_executions_consumed != 2:
            raise ValueError("SELECT_CANDIDATE requires exactly two root executions")
        if self.baseline_execution_ref.execution_id.strip() == self.candidate_execution_ref.execution_id.strip():
            raise ValueError("baseline and candidate execution identities must be distinct")
        measured = ResourceAccountingEvidence(
            tokens_consumed=self.tokens_consumed,
            monetary_consumed=self.monetary_consumed,
            root_executions_consumed=self.root_executions_consumed,
        )
        _ = measured
        return self


class AdasW1BaselineRetentionProvenance(BaseModel):
    """Immutable baseline evaluation provenance for RETAIN_BASELINE selection."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    baseline_version_id: str
    baseline_configuration_digest: str
    candidate_id: str
    objective_id: str
    search_run_id: str
    tenant_id: str
    workspace_id: str
    baseline_execution_ref: AdasExecutionReference
    evaluation_ref: str

    @field_validator(
        "baseline_version_id",
        "candidate_id",
        "objective_id",
        "search_run_id",
        "tenant_id",
        "workspace_id",
        "evaluation_ref",
    )
    @classmethod
    def _non_blank_provenance(cls, value: str) -> str:
        return _non_blank(value, field="baseline provenance")

    @field_validator("baseline_configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)


class AgentDesignObjective(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    objective_id: str
    tenant_id: str
    workspace_id: str
    application_id: str | None = None
    task_class: str
    target_capability: str
    envelope: AdasOperationalEnvelope
    baseline: BaselineConfigurationRef
    baseline_agent_id: str | None = None
    baseline_candidate_id: str | None = None
    max_iterations: int = 1
    max_candidates: int = 1
    max_generation_cost: float | None = None
    max_eval_cost: float | None = None
    min_utility_delta: float = 0.05
    allowed_patterns: tuple[str, ...] = ()
    allowed_tools: tuple[str, ...] = ()
    forbidden_tools: tuple[str, ...] = ()
    goal_summary: str
    success_metric: str
    dataset_id: str
    dataset_content_digest: str
    acceptance_criteria_ref: str
    admitted_configuration_surface: tuple[str, ...] = ("merge_strategy",)
    allowed_merge_strategies: tuple[MergeStrategy, ...] = (
        MergeStrategy.LAST_WINS,
        MergeStrategy.CONCAT,
    )
    aggregate_limits: AggregateResourceLimits
    external_effects_prohibited: bool = True
    require_human_approval: bool = True
    created_by: str
    created_at: datetime

    @field_validator(
        "objective_id",
        "tenant_id",
        "workspace_id",
        "task_class",
        "target_capability",
        "goal_summary",
        "success_metric",
        "dataset_id",
        "acceptance_criteria_ref",
        "created_by",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("dataset_content_digest")
    @classmethod
    def _dataset_digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("max_generation_cost", "max_eval_cost")
    @classmethod
    def _optional_cost_budgets(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_non_negative(value, field="objective cost budget")

    @field_validator("min_utility_delta")
    @classmethod
    def _min_utility_delta(cls, value: float) -> float:
        if not math.isfinite(value) or value < 0:
            raise ValueError("min_utility_delta must be finite and non-negative")
        return value

    @field_validator("created_at")
    @classmethod
    def _created(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @model_validator(mode="after")
    def _scope_and_w1(self) -> Self:
        if self.envelope.tenant_id != self.tenant_id:
            raise ValueError("envelope tenant_id must match objective tenant_id")
        if self.envelope.workspace_id != self.workspace_id:
            raise ValueError("envelope workspace_id must match objective workspace_id")
        if not self.external_effects_prohibited:
            raise ValueError("W1 objectives must prohibit external effects")
        if not self.require_human_approval:
            raise ValueError("W1 objectives require human approval")
        if len(self.allowed_merge_strategies) != len(set(self.allowed_merge_strategies)):
            raise ValueError("duplicate merge strategy alternatives rejected")
        if set(self.allowed_merge_strategies) != _W1_MERGE_STRATEGIES:
            raise ValueError("W1 allows only last_wins and concat merge strategies")
        if tuple(self.admitted_configuration_surface) != ("merge_strategy",):
            raise ValueError("W1 admitted configuration surface must contain only merge_strategy")
        if len(self.admitted_configuration_surface) != len(set(self.admitted_configuration_surface)):
            raise ValueError("duplicate admitted configuration surface entries rejected")
        if self.max_iterations != self.aggregate_limits.max_search_iterations:
            raise ValueError("max_iterations must match aggregate search iteration limit")
        if self.max_candidates != self.aggregate_limits.max_candidates:
            raise ValueError("max_candidates must match aggregate candidate limit")
        return self


class AgentDesignSearchRun(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    search_run_id: str
    objective_id: str
    tenant_id: str
    workspace_id: str
    task_class: str
    envelope: AdasOperationalEnvelope
    strategy_name: str
    status: str
    baseline: BaselineConfigurationRef
    resource_accounting: ResourceAccountingEvidence
    selection_outcome: SearchSelectionOutcome | None = None
    started_at: datetime
    completed_at: datetime | None = None
    generated_candidate_ids: tuple[str, ...] = ()
    promoted_candidate_id: str | None = None
    total_generation_cost: float | None = None
    total_evaluation_cost: float | None = None
    baseline_execution_ref: str | None = None
    candidate_execution_ref: str | None = None
    eval_run_ref: str | None = None
    eval_comparison_ref: str | None = None
    accounting_evidence: AggregateAccountingEvidenceReference | None = None
    structured_stop_reason: StructuredSearchStopReason | None = None
    stop_reason: str | None = None

    @field_validator(
        "search_run_id",
        "objective_id",
        "tenant_id",
        "workspace_id",
        "task_class",
        "strategy_name",
        "status",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("started_at", "completed_at")
    @classmethod
    def _times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_datetime(value)

    @field_validator("total_generation_cost", "total_evaluation_cost")
    @classmethod
    def _run_costs(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_non_negative(value, field="search run cost")

    @model_validator(mode="after")
    def _scope_and_terminal(self) -> Self:
        if self.status not in _SEARCH_RUN_STATUSES:
            raise ValueError("unrecognized search run status")
        if self.promoted_candidate_id is not None:
            raise ValueError("W1 search runs keep promoted_candidate_id explicitly absent")
        if self.envelope.tenant_id != self.tenant_id or self.envelope.workspace_id != self.workspace_id:
            raise ValueError("envelope scope must match search run scope")
        if self.status in {"running", "pending"}:
            if self.completed_at is not None:
                raise ValueError("nonterminal search runs must not declare completed_at")
            if self.selection_outcome is not None:
                raise ValueError("nonterminal search runs must not declare selection_outcome")
        if self.status in {"completed", "failed", "cancelled"}:
            if self.completed_at is None:
                raise ValueError("terminal search runs require completed_at")
            if self.structured_stop_reason is None:
                raise ValueError("terminal search runs require structured_stop_reason")
            if self.stop_reason is None or not self.stop_reason.strip():
                raise ValueError("terminal search runs require stop_reason")
            if self.completed_at < self.started_at:
                raise ValueError("completed_at must not be earlier than started_at")
            if self.status == "completed" and self.selection_outcome is None:
                raise ValueError("completed search runs require selection_outcome")
            failed_or_cancelled = self.status in {"failed", "cancelled"}
            if failed_or_cancelled and self.selection_outcome == SearchSelectionOutcome.SELECT_CANDIDATE:
                raise ValueError("failed or cancelled runs must not claim candidate selection")
        for candidate_id in self.generated_candidate_ids:
            if not candidate_id.strip():
                raise ValueError("generated_candidate_ids must not contain blank entries")
        if len(self.generated_candidate_ids) > 1:
            raise ValueError("W1 search runs admit at most one generated candidate")
        return self


class AgentCandidateDraft(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    objective_id: str
    search_run_id: str
    tenant_id: str
    workspace_id: str
    parent_candidate_id: str | None = None
    lineage_id: str
    generated_by: str
    generation_strategy: str
    design_rationale: str
    baseline: BaselineConfigurationRef
    candidate_configuration_digest: str
    configuration_delta: ConfigurationDelta
    scaffold_pattern: None = None
    capabilities: tuple[str, ...] = ()
    contract_draft: Mapping[str, AdasJsonValue] = Field(default_factory=_empty_adas_json_object)
    prompt_assets: Mapping[str, AdasJsonValue] = Field(default_factory=_empty_adas_json_object)
    tool_policy: Mapping[str, AdasJsonValue] = Field(default_factory=_empty_adas_json_object)
    state_schema: None = None
    source_artifact_ref: str | None = None
    scaffold_output_ref: None = None
    created_at: datetime

    @field_validator(
        "candidate_id",
        "objective_id",
        "search_run_id",
        "tenant_id",
        "workspace_id",
        "lineage_id",
        "generated_by",
        "generation_strategy",
        "design_rationale",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("candidate_configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("created_at")
    @classmethod
    def _created(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @field_validator(
        "contract_draft",
        "prompt_assets",
        "tool_policy",
        mode="before",
    )
    @classmethod
    def _json_before(cls, value: object, info) -> dict[str, AdasJsonValue] | None:  # type: ignore[no-untyped-def]
        return _adas_json_field_before(value, field_name=str(info.field_name))

    @field_validator(
        "contract_draft",
        "prompt_assets",
        "tool_policy",
        mode="after",
    )
    @classmethod
    def _json_after(
        cls,
        value: dict[str, AdasJsonValue] | None,
    ) -> Mapping[str, AdasJsonValue] | None:
        return _adas_json_field_after(value)

    @field_serializer("contract_draft", "prompt_assets", "tool_policy")
    def _json_serialize(self, value: Mapping[str, AdasJsonValue] | None) -> dict[str, AdasJsonValue] | None:
        return _serialize_adas_json_field(value)

    @model_validator(mode="after")
    def _w1_configuration_only_draft(self) -> Self:
        if len(self.contract_draft) > 0 or len(self.prompt_assets) > 0 or len(self.tool_policy) > 0:
            raise ValueError("W1 candidate drafts must not carry executable configuration modifications")
        if self.capabilities:
            raise ValueError("W1 candidate drafts must not declare capability changes")
        return self


class AgentCandidateRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    snapshot_id: str
    sequence: int
    candidate_id: str
    objective_id: str
    search_run_id: str
    tenant_id: str
    workspace_id: str
    parent_candidate_id: str | None = None
    lineage_id: str
    status: AgentCandidateStatus
    envelope: AdasOperationalEnvelope
    baseline: BaselineConfigurationRef
    candidate_configuration_digest: str
    configuration_delta: ConfigurationDelta
    agent_contract_snapshot: Mapping[str, AdasJsonValue] = Field(default_factory=_empty_adas_json_object)
    scaffold_pattern: None = None
    source_artifact_ref: str | None = None
    scaffold_output_ref: None = None
    eval_suite_id: str | None = None
    static_gate_result: Mapping[str, AdasJsonValue] | None = None
    evaluation_result: Mapping[str, AdasJsonValue] | None = None
    w1_lab_stop_evidence: AdasW1LabStopEvidence | None = None
    utility_score: float | None = None
    baseline_utility_score: float | None = None
    utility_delta: float | None = None
    novelty_score: float | None = None
    complexity_score: float | None = None
    generation_cost: float | None = None
    evaluation_cost: float | None = None
    rejection_reason: str | None = None
    promotion_request_id: None = None
    created_by: str
    created_at: datetime
    archived_at: datetime | None = None

    @field_validator(
        "snapshot_id",
        "candidate_id",
        "objective_id",
        "search_run_id",
        "tenant_id",
        "workspace_id",
        "lineage_id",
        "created_by",
    )
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("candidate_configuration_digest")
    @classmethod
    def _digest(cls, value: str) -> str:
        return _validate_sha256_digest(value)

    @field_validator("sequence")
    @classmethod
    def _sequence(cls, value: int) -> int:
        return _finite_non_negative_int(value, field="sequence")

    @field_validator("created_at", "archived_at")
    @classmethod
    def _times(cls, value: datetime | None) -> datetime | None:
        if value is None:
            return None
        return _aware_datetime(value)

    @model_validator(mode="before")
    @classmethod
    def _migrate_legacy_stop_evidence(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        gate = data.get("static_gate_result")
        nested = None
        if isinstance(gate, Mapping):
            plain = adas_json_to_plain(gate)
            if isinstance(plain, dict) and "w1_lab_stop_evidence" in plain:
                gate_copy = dict(plain)
                nested = gate_copy.pop("w1_lab_stop_evidence")
                data["static_gate_result"] = gate_copy
        first_class = data.get("w1_lab_stop_evidence")
        if nested is not None and first_class is not None and nested != first_class:
            raise ValueError("competing embedded and first-class stop evidence")
        if nested is not None and data.get("w1_lab_stop_evidence") is None:
            data["w1_lab_stop_evidence"] = nested
        return data

    @field_validator(
        "agent_contract_snapshot",
        "static_gate_result",
        "evaluation_result",
        mode="before",
    )
    @classmethod
    def _json_before(cls, value: object, info) -> dict[str, AdasJsonValue] | None:  # type: ignore[no-untyped-def]
        return _adas_json_field_before(value, field_name=str(info.field_name))

    @field_validator(
        "agent_contract_snapshot",
        "static_gate_result",
        "evaluation_result",
        mode="after",
    )
    @classmethod
    def _json_after(
        cls,
        value: dict[str, AdasJsonValue] | None,
    ) -> Mapping[str, AdasJsonValue] | None:
        return _adas_json_field_after(value)

    @field_serializer("agent_contract_snapshot", "static_gate_result", "evaluation_result")
    def _json_serialize(self, value: Mapping[str, AdasJsonValue] | None) -> dict[str, AdasJsonValue] | None:
        return _serialize_adas_json_field(value)

    @model_validator(mode="after")
    def _record_scope(self) -> Self:
        if self.envelope.tenant_id != self.tenant_id or self.envelope.workspace_id != self.workspace_id:
            raise ValueError("envelope scope must match record scope")
        return self

    @model_validator(mode="after")
    def _w1_record_stop_and_gate_shape(self) -> Self:
        if self.static_gate_result is not None:
            plain = adas_json_to_plain(self.static_gate_result)
            if "w1_lab_stop_evidence" in plain:
                raise ValueError("static_gate_result must not embed execution stop evidence")
        if self.w1_lab_stop_evidence is not None:
            stop = self.w1_lab_stop_evidence
            if stop.candidate_id != self.candidate_id:
                raise ValueError("stop evidence candidate mismatch")
            if stop.search_run_id != self.search_run_id:
                raise ValueError("stop evidence search run mismatch")
            if stop.tenant_id != self.tenant_id or stop.workspace_id != self.workspace_id:
                raise ValueError("stop evidence scope mismatch")
        return self

    @field_validator(
        "utility_score",
        "baseline_utility_score",
        "utility_delta",
        "novelty_score",
        "complexity_score",
    )
    @classmethod
    def _optional_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("numeric field must be finite")
        return value

    @field_validator("generation_cost", "evaluation_cost")
    @classmethod
    def _resource_costs(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_non_negative(value, field="resource cost")


class AgentCandidateEvaluationResult(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    candidate_id: str
    baseline_agent_id: str | None = None
    passed: bool
    quality_score: float | None = None
    cost_normalized: float | None = None
    latency_ms: int | None = None
    total_tokens: int | None = None
    step_count: int | None = None
    tool_calls: int | None = None
    llm_calls: int | None = None
    hitl_interventions: int | None = None
    regression_flags: tuple[str, ...] = ()
    utility_score: float | None = None
    baseline_utility_score: float | None = None
    utility_delta: float | None = None
    golden_pass_rate: float | None = None
    negative_pass_rate: float | None = None
    adversarial_pass_rate: float | None = None
    failure_reasons: tuple[str, ...] = ()
    execution_ref: str | None = None
    evaluation_ref: str | None = None
    baseline_execution_ref: AdasExecutionReference | None = None
    eval_run_ref: AdasEvalRunReference | None = None
    eval_comparison_ref: AdasEvalComparisonReference | None = None
    evidence_complete: bool = False

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return _non_blank(value, field="candidate_id")

    @field_validator("cost_normalized")
    @classmethod
    def _cost_normalized(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_non_negative(value, field="cost_normalized")

    @field_validator(
        "quality_score",
        "utility_score",
        "baseline_utility_score",
        "utility_delta",
    )
    @classmethod
    def _optional_metrics(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("metric must be finite")
        return value

    @field_validator("golden_pass_rate", "negative_pass_rate", "adversarial_pass_rate")
    @classmethod
    def _pass_rates(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value) or value < 0.0 or value > 1.0:
            raise ValueError("pass rate must be finite and normalized between 0 and 1")
        return value

    @field_validator("latency_ms", "total_tokens", "step_count", "tool_calls", "llm_calls", "hitl_interventions")
    @classmethod
    def _optional_int_metrics(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _finite_non_negative_int(value, field="metric")

    @model_validator(mode="after")
    def _w1_verdict_evidence(self) -> Self:
        if self.execution_ref is not None and self.execution_ref.strip() == "":
            raise ValueError("execution_ref must not be blank when supplied")
        if self.evaluation_ref is not None and self.evaluation_ref.strip() == "":
            raise ValueError("evaluation_ref must not be blank when supplied")
        if self.passed:
            required = (
                self.quality_score,
                self.cost_normalized,
                self.latency_ms,
                self.total_tokens,
                self.utility_score,
                self.golden_pass_rate,
                self.negative_pass_rate,
                self.adversarial_pass_rate,
                self.execution_ref,
                self.evaluation_ref,
                self.baseline_execution_ref,
                self.eval_run_ref,
                self.eval_comparison_ref,
            )
            if any(item is None for item in required):
                raise ValueError("successful evaluation verdict requires mandatory W1 evidence")
            if not self.evidence_complete:
                raise ValueError("passed evaluation requires evidence_complete")
        return self


def _evidence_values_compatible(prior: AdasJsonValue, incoming: AdasJsonValue) -> bool:
    if prior == incoming:
        return True
    if isinstance(prior, dict) and isinstance(incoming, dict):
        for key, value in prior.items():
            if key not in incoming:
                return False
            if not _evidence_values_compatible(value, incoming[key]):
                return False
        return True
    if isinstance(prior, list) and isinstance(incoming, list):
        if len(prior) > len(incoming):
            return False
        for index, value in enumerate(prior):
            if not _evidence_values_compatible(value, incoming[index]):
                return False
        return True
    return False


def _lab_stop_payload(stop: AdasW1LabStopEvidence) -> dict[str, AdasJsonValue]:
    return stop.model_dump(mode="json")  # type: ignore[return-value]


def parse_w1_lab_stop_from_record(record: AgentCandidateRecord) -> AdasW1LabStopEvidence | None:
    return record.w1_lab_stop_evidence


def _validate_measured_root_executions(raw: object) -> int:
    if type(raw) is not int:
        raise ValueError("root_executions_consumed must be a strict integer")
    if raw < 0 or raw > 2:
        raise ValueError("root_executions_consumed must be between 0 and 2")
    return raw


def _validate_measured_scalar(raw: object, *, label: str) -> float:
    if type(raw) is bool or not isinstance(raw, (int, float)):
        raise ValueError(label)
    value = float(raw)
    if not math.isfinite(value) or value < 0.0:
        raise ValueError(label)
    return value


def parse_w1_selection_accounting(payload: Mapping[str, AdasJsonValue]) -> AdasW1SelectionAccountingEvidence:
    plain = adas_json_to_plain(payload)
    if not isinstance(plain, dict):
        raise ValueError("accounting payload must be an object")
    try:
        return AdasW1SelectionAccountingEvidence.model_validate(plain)
    except ValidationError as exc:
        raise ValueError("invalid W1 selection accounting representation") from exc


def validate_w1_baseline_retention(
    *,
    provenance: AdasW1BaselineRetentionProvenance,
    record: AgentCandidateRecord,
    baseline_evaluation: AgentCandidateEvaluationResult,
) -> None:
    if not baseline_evaluation.passed:
        raise ValueError("RETAIN_BASELINE requires passed baseline evaluation")
    if provenance.baseline_version_id != record.baseline.baseline_version_id:
        raise ValueError("baseline version identity mismatch")
    if provenance.baseline_configuration_digest != record.baseline.baseline_configuration_digest:
        raise ValueError("baseline configuration digest mismatch")
    for field_name, expected in (
        ("candidate_id", record.candidate_id),
        ("objective_id", record.objective_id),
        ("search_run_id", record.search_run_id),
        ("tenant_id", record.tenant_id),
        ("workspace_id", record.workspace_id),
    ):
        if getattr(provenance, field_name) != expected:
            raise ValueError("baseline provenance scope mismatch")
    base_exec = baseline_evaluation.baseline_execution_ref
    if base_exec is None:
        raise ValueError("baseline evaluation requires baseline execution reference")
    if provenance.baseline_execution_ref.execution_id != base_exec.execution_id:
        raise ValueError("contradictory baseline execution identity")
    if baseline_evaluation.evaluation_ref != provenance.evaluation_ref:
        raise ValueError("baseline provenance evaluation identity mismatch")
    if baseline_evaluation.candidate_id != record.candidate_id:
        raise ValueError("baseline evaluation must bind to archived candidate context")


def validate_w1_candidate_record_state(record: AgentCandidateRecord) -> None:
    """Authoritative W1 record state / evidence semantics (archive ingress)."""
    gate_passed: bool | None = None
    if record.static_gate_result is not None:
        gate_plain = adas_json_to_plain(record.static_gate_result)
        if record.status is not AgentCandidateStatus.DRAFT:
            gate_passed = adas_static_gate_verdict(gate_plain)

    stop = record.w1_lab_stop_evidence
    if stop is not None and (
        stop.candidate_id != record.candidate_id
        or stop.search_run_id != record.search_run_id
        or stop.tenant_id != record.tenant_id
        or stop.workspace_id != record.workspace_id
    ):
        raise ValueError("stop evidence scope mismatch")

    if record.status is AgentCandidateStatus.DRAFT:
        if record.static_gate_result is not None:
            raise ValueError("DRAFT must not contain static gate payload")
        if record.evaluation_result is not None:
            raise ValueError("DRAFT must not contain completed evaluation evidence")
        if stop is not None:
            raise ValueError("DRAFT must not contain terminal stop evidence")
        return

    if record.status is AgentCandidateStatus.STATIC_GATE_PASSED:
        if record.static_gate_result is None or gate_passed is not True:
            raise ValueError("STATIC_GATE_PASSED requires successful static gate evidence")
        if record.evaluation_result is not None:
            raise ValueError("STATIC_GATE_PASSED must not contain completed evaluation")
        if stop is not None:
            raise ValueError("STATIC_GATE_PASSED must not contain terminal stop evidence")
        return

    if record.status is AgentCandidateStatus.STATIC_GATE_FAILED:
        if record.static_gate_result is None or gate_passed is not False:
            raise ValueError("STATIC_GATE_FAILED requires failed static gate evidence")
        if record.evaluation_result is not None:
            raise ValueError("STATIC_GATE_FAILED must not contain fabricated evaluation")
        if stop is not None and (
            stop.execution_ref is not None
            or stop.kind
            in {
                AdasW1LabStopKind.POST_GATE_ADMISSION_REJECTION,
                AdasW1LabStopKind.EXECUTION_FAILURE,
                AdasW1LabStopKind.EXECUTION_CANCELLATION,
            }
        ):
            raise ValueError("STATIC_GATE_FAILED must not contain later execution evidence")
        if not record.rejection_reason or not record.rejection_reason.strip():
            raise ValueError("STATIC_GATE_FAILED requires rejection evidence")
        return

    if record.status is AgentCandidateStatus.EVALUATED:
        if gate_passed is not True:
            raise ValueError("EVALUATED requires successful static gate evidence")
        if record.evaluation_result is None:
            raise ValueError("EVALUATED requires evaluation evidence references")
        validate_evaluation_mapping_for_record(
            record.evaluation_result,
            candidate_id=record.candidate_id,
        )
        if stop is not None:
            raise ValueError("EVALUATED must not contain contradictory stop evidence")
        return

    if record.status is AgentCandidateStatus.REJECTED:
        if not record.rejection_reason or not record.rejection_reason.strip():
            raise ValueError("REJECTED requires explicit rejection evidence")
        if record.evaluation_result is not None:
            validate_evaluation_mapping_for_record(
                record.evaluation_result,
                candidate_id=record.candidate_id,
            )
            if gate_passed is not True:
                raise ValueError("evaluated REJECTED requires passing static gate evidence")
            if stop is not None and stop.kind in {
                AdasW1LabStopKind.PRE_EXECUTION_REJECTION,
                AdasW1LabStopKind.EXECUTION_FAILURE,
                AdasW1LabStopKind.EXECUTION_CANCELLATION,
            }:
                raise ValueError("completed evaluation cannot be reinterpreted as pre-eval stop")
            return
        if gate_passed is False:
            raise ValueError("pre-execution rejection must not fabricate failed static gate evidence")
        if gate_passed is True:
            if stop is None or stop.kind not in {
                AdasW1LabStopKind.POST_GATE_ADMISSION_REJECTION,
                AdasW1LabStopKind.EXECUTION_FAILURE,
                AdasW1LabStopKind.EXECUTION_CANCELLATION,
            }:
                raise ValueError("post-gate REJECTED requires post-gate admission or execution stop evidence")
            return
        if stop is None or stop.kind is not AdasW1LabStopKind.PRE_EXECUTION_REJECTION:
            raise ValueError("pre-execution REJECTED requires typed PRE_EXECUTION_REJECTION stop evidence")
        return

    if record.status is AgentCandidateStatus.ARCHIVED:
        if record.archived_at is None:
            raise ValueError("ARCHIVED requires archival timestamp")
        if record.evaluation_result is not None:
            validate_evaluation_mapping_for_record(
                record.evaluation_result,
                candidate_id=record.candidate_id,
            )
        return


def validate_w1_selection_cost_report(
    cost_report: Mapping[str, AdasJsonValue],
    *,
    record: AgentCandidateRecord,
    evaluation: AgentCandidateEvaluationResult | None = None,
    baseline_evaluation: AgentCandidateEvaluationResult | None = None,
) -> None:
    accounting = parse_w1_selection_accounting(cost_report)
    for field_name, expected in (
        ("candidate_id", record.candidate_id),
        ("objective_id", record.objective_id),
        ("search_run_id", record.search_run_id),
        ("tenant_id", record.tenant_id),
        ("workspace_id", record.workspace_id),
    ):
        if getattr(accounting, field_name) != expected:
            raise ValueError("accounting evidence scope mismatch")
    if accounting.accepted_scope_ref != record.envelope.accepted_scope_ref:
        raise ValueError("accepted scope reference mismatch")
    limits = accounting.aggregate_limits
    if accounting.tokens_consumed > limits.aggregate_token_allowance:
        raise ValueError("token consumption exceeds aggregate token allowance")
    if accounting.monetary_consumed > limits.aggregate_monetary_allowance:
        raise ValueError("monetary consumption exceeds aggregate monetary allowance")
    if accounting.root_executions_consumed > limits.max_root_executions:
        raise ValueError("root execution count exceeds admitted root limit")
    if evaluation is None:
        raise ValueError("SELECT_CANDIDATE requires evaluation context for accounting")
    if not evaluation.passed:
        raise ValueError("SELECT_CANDIDATE requires successful evaluation")
    eval_base = evaluation.baseline_execution_ref
    eval_candidate = evaluation.execution_ref
    if eval_base is None:
        raise ValueError("evaluation requires baseline execution reference")
    if eval_candidate is None or not str(eval_candidate).strip():
        raise ValueError("evaluation requires candidate execution reference")
    if accounting.baseline_execution_ref.execution_id != eval_base.execution_id:
        raise ValueError("baseline execution reference mismatch")
    cand_exec_id = eval_candidate if isinstance(eval_candidate, str) else getattr(eval_candidate, "execution_id", None)
    if cand_exec_id is None or accounting.candidate_execution_ref.execution_id != str(cand_exec_id):
        raise ValueError("candidate execution reference mismatch")
    if evaluation.total_tokens is not None and accounting.tokens_consumed < float(evaluation.total_tokens):
        raise ValueError("aggregate tokens below candidate evaluation tokens")
    combined = float(evaluation.total_tokens or 0)
    if baseline_evaluation is not None:
        base_ref = (
            baseline_evaluation.baseline_execution_ref.execution_id
            if baseline_evaluation.baseline_execution_ref is not None
            else baseline_evaluation.execution_ref
        )
        if base_ref is None or base_ref != accounting.baseline_execution_ref.execution_id:
            raise ValueError("baseline evaluation execution reference mismatch")
        if (
            baseline_evaluation.baseline_execution_ref is not None
            and baseline_evaluation.execution_ref is not None
            and baseline_evaluation.execution_ref != accounting.baseline_execution_ref.execution_id
        ):
            raise ValueError("baseline evaluation execution reference mismatch")
        if baseline_evaluation.total_tokens is not None and base_ref != str(cand_exec_id):
            combined += float(baseline_evaluation.total_tokens)
    if accounting.tokens_consumed < combined:
        raise ValueError("aggregate token allowance below recorded measurements")


def adas_static_gate_verdict(plain: Mapping[str, AdasJsonValue]) -> bool:
    if "passed" not in plain:
        raise ValueError("static gate evidence must declare passed")
    verdict = plain.get("passed")
    if not isinstance(verdict, bool):
        raise ValueError("static gate verdict must be a boolean")
    return verdict


def validate_evaluation_mapping_for_record(
    payload: Mapping[str, AdasJsonValue],
    *,
    candidate_id: str,
) -> AgentCandidateEvaluationResult:
    plain = adas_json_to_plain(payload)
    passed = plain.get("passed")
    if not isinstance(passed, bool):
        raise ValueError("evaluation verdict must be a boolean")
    embedded = plain.get("candidate_id")
    if embedded is not None and embedded != candidate_id:
        raise ValueError("evaluation candidate identity mismatch")
    body: dict[str, object] = dict(plain)
    body["candidate_id"] = candidate_id
    result = AgentCandidateEvaluationResult.model_validate(body)
    if result.execution_ref is None or result.evaluation_ref is None:
        raise ValueError("completed evaluation requires execution and evaluation references")
    if not result.passed and not result.failure_reasons:
        raise ValueError("negative evaluation requires failure reasons")
    return result


def infer_w1_terminal_evidence_outcome(
    record: AgentCandidateRecord,
    *,
    has_evaluation: bool,
    gate_passed: bool | None,
) -> W1TerminalEvidenceOutcome:
    if record.status is AgentCandidateStatus.EVALUATED or (
        record.status is AgentCandidateStatus.REJECTED and has_evaluation
    ):
        return W1TerminalEvidenceOutcome.EVALUATION_COMPLETE
    try:
        stop = parse_w1_lab_stop_from_record(record)
    except ValueError:
        stop = None
    if stop is not None and stop.kind in {
        AdasW1LabStopKind.EXECUTION_FAILURE,
        AdasW1LabStopKind.EXECUTION_CANCELLATION,
    }:
        return W1TerminalEvidenceOutcome.EXECUTION_FAILED_OR_CANCELLED
    if record.status is AgentCandidateStatus.STATIC_GATE_FAILED:
        return W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION
    if gate_passed is False and not has_evaluation:
        if stop is not None and stop.kind in {
            AdasW1LabStopKind.EXECUTION_FAILURE,
            AdasW1LabStopKind.EXECUTION_CANCELLATION,
        }:
            return W1TerminalEvidenceOutcome.EXECUTION_FAILED_OR_CANCELLED
        return W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION
    if has_evaluation:
        return W1TerminalEvidenceOutcome.EVALUATION_COMPLETE
    return W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION


class AgentCandidateEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    @model_validator(mode="before")
    @classmethod
    def _strip_gate_embedded_stop(cls, data: object) -> object:
        if not isinstance(data, dict):
            return data
        gate = data.get("static_gate_result")
        if isinstance(gate, Mapping):
            plain = adas_json_to_plain(gate)
            if isinstance(plain, dict) and "w1_lab_stop_evidence" in plain:
                gate_copy = dict(plain)
                gate_copy.pop("w1_lab_stop_evidence", None)
                data["static_gate_result"] = gate_copy
        return data

    bundle_id: str
    candidate_id: str
    objective_id: str
    tenant_id: str
    workspace_id: str
    search_run_id: str | None = None
    schema_version: str = ADAS_CONTRACT_SCHEMA_VERSION
    created_at: datetime
    created_by: str
    correlation_id: str | None = None
    trace_id: str | None = None
    evaluation_bundle_id: str | None = None
    risk_class: str | None = None
    retention_policy: str | None = None
    bundle_digest: str | None = None
    artifact_hash: str | None = None
    baseline_configuration_digest: str
    candidate_configuration_digest: str
    selection_outcome: SearchSelectionOutcome | None = None
    terminal_evidence_outcome: W1TerminalEvidenceOutcome | None = None
    w1_lab_stop_evidence: AdasW1LabStopEvidence | None = None
    candidate_record: AgentCandidateRecord
    baseline_evaluation_result: AgentCandidateEvaluationResult | None = None
    w1_baseline_retention_provenance: AdasW1BaselineRetentionProvenance | None = None
    evaluation_result: AgentCandidateEvaluationResult | None = None
    static_gate_result: Mapping[str, AdasJsonValue] | None = None
    policy_gate_result: Mapping[str, AdasJsonValue] | None = None
    cost_report: Mapping[str, AdasJsonValue] | None = None
    security_adversarial_result: Mapping[str, AdasJsonValue] | None = None
    approval_record: Mapping[str, AdasJsonValue] | None = None
    promotion_decision: None = None
    rollback_pointer: str | None = None
    assembled_at: datetime
    sealed: bool = False

    @field_validator("bundle_id", "candidate_id", "objective_id", "tenant_id", "workspace_id", "created_by")
    @classmethod
    def _ids(cls, value: str) -> str:
        return _non_blank(value, field="field")

    @field_validator("baseline_configuration_digest", "candidate_configuration_digest", "bundle_digest")
    @classmethod
    def _digests(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256_digest(value)

    @field_validator("created_at", "assembled_at")
    @classmethod
    def _times(cls, value: datetime) -> datetime:
        return _aware_datetime(value)

    @field_validator(
        "static_gate_result",
        "policy_gate_result",
        "cost_report",
        "security_adversarial_result",
        "approval_record",
        mode="before",
    )
    @classmethod
    def _json_before(cls, value: object, info) -> dict[str, AdasJsonValue] | None:  # type: ignore[no-untyped-def]
        return _adas_json_field_before(value, field_name=str(info.field_name))

    @field_validator(
        "static_gate_result",
        "policy_gate_result",
        "cost_report",
        "security_adversarial_result",
        "approval_record",
        mode="after",
    )
    @classmethod
    def _json_after(
        cls,
        value: dict[str, AdasJsonValue] | None,
    ) -> Mapping[str, AdasJsonValue] | None:
        return _adas_json_field_after(value)

    @field_serializer(
        "static_gate_result",
        "policy_gate_result",
        "cost_report",
        "security_adversarial_result",
        "approval_record",
    )
    def _json_serialize(self, value: Mapping[str, AdasJsonValue] | None) -> dict[str, AdasJsonValue] | None:
        return _serialize_adas_json_field(value)

    @field_validator("artifact_hash")
    @classmethod
    def _artifact(cls, value: str | None) -> str | None:
        if value is None:
            return None
        return _validate_sha256_digest(value)

    @model_validator(mode="after")
    def _scope_and_rejection_shape(self) -> Self:
        record = self.candidate_record
        if record.tenant_id != self.tenant_id or record.workspace_id != self.workspace_id:
            raise ValueError("bundle scope must match nested candidate record scope")
        if record.candidate_id != self.candidate_id or record.objective_id != self.objective_id:
            raise ValueError("bundle lineage identities must match candidate record")
        if self.baseline_configuration_digest != record.baseline.baseline_configuration_digest:
            raise ValueError("bundle baseline digest must match archived baseline")
        if self.candidate_configuration_digest != record.candidate_configuration_digest:
            raise ValueError("bundle candidate configuration digest must match archived record")
        if self.evaluation_result is not None and self.evaluation_result.candidate_id != self.candidate_id:
            raise ValueError("evaluation result candidate identity must match bundle")
        if self.baseline_evaluation_result is not None and self.baseline_evaluation_result.baseline_agent_id is None:
            raise ValueError("baseline evaluation requires baseline agent identity")
        if self.search_run_id is not None and self.search_run_id != record.search_run_id:
            raise ValueError("bundle search_run_id must match candidate record search_run_id")
        if self.sealed:
            validate_sealed_terminal_evidence_bundle(self)
        if (
            self.artifact_hash is not None
            and self.bundle_digest is not None
            and self.artifact_hash == self.bundle_digest
        ):
            raise ValueError("artifact_hash must not substitute bundle_digest")
        if self.bundle_digest is not None:
            recomputed = compute_evidence_bundle_digest(self)
            if self.bundle_digest != recomputed:
                raise ValueError("bundle_digest mismatch")
        return self


def validate_sealed_terminal_evidence_bundle(bundle: AgentCandidateEvidenceBundle) -> None:
    record = bundle.candidate_record
    has_gate = bundle.static_gate_result is not None
    has_eval = bundle.evaluation_result is not None
    has_stop = bundle.w1_lab_stop_evidence is not None
    if not has_gate and not has_eval and not has_stop:
        raise ValueError("sealed bundle requires terminal evidence shape")
    gate_passed: bool | None = None
    if has_gate:
        gate_plain = adas_json_to_plain(bundle.static_gate_result)  # type: ignore[arg-type]
        gate_passed = adas_static_gate_verdict(gate_plain)
    elif record.static_gate_result is not None:
        raise ValueError("bundle must carry static gate evidence for archived gate state")
    if record.static_gate_result is not None and has_gate:
        record_gate = adas_json_to_plain(record.static_gate_result)
        bundle_gate = adas_json_to_plain(bundle.static_gate_result)  # type: ignore[arg-type]
        if not _evidence_values_compatible(record_gate, bundle_gate):
            raise ValueError("bundle static gate must match archived snapshot")
    elif record.static_gate_result is not None:
        raise ValueError("bundle must carry static gate evidence for archived gate state")
    if record.evaluation_result is not None:
        record_eval = validate_evaluation_mapping_for_record(
            record.evaluation_result,
            candidate_id=record.candidate_id,
        )
        if not has_eval:
            raise ValueError("bundle must carry evaluation evidence for evaluated snapshot")
        if bundle.evaluation_result != record_eval:
            raise ValueError("bundle evaluation must match archived snapshot")
    elif has_eval and record.status in {AgentCandidateStatus.STATIC_GATE_FAILED, AgentCandidateStatus.REJECTED}:
        raise ValueError("terminal state must not contain fabricated evaluation results")
    archived_stop = record.w1_lab_stop_evidence
    if archived_stop is not None:
        if not has_stop or bundle.w1_lab_stop_evidence is None:
            raise ValueError("sealed bundle requires archived stop evidence")
        if bundle.w1_lab_stop_evidence.model_dump() != archived_stop.model_dump():
            raise ValueError("bundle stop evidence must match archived record")
    elif has_stop and bundle.w1_lab_stop_evidence is not None:
        raise ValueError("bundle must not invent stop evidence absent from archived record")
    if has_stop and bundle.w1_lab_stop_evidence is not None:
        stop = bundle.w1_lab_stop_evidence
        if (
            stop.candidate_id != record.candidate_id
            or stop.search_run_id != record.search_run_id
            or stop.tenant_id != record.tenant_id
            or stop.workspace_id != record.workspace_id
        ):
            raise ValueError("terminal stop evidence scope mismatch")
    outcome = bundle.terminal_evidence_outcome
    if outcome is None:
        raise ValueError("sealed bundle requires terminal_evidence_outcome")
    expected = infer_w1_terminal_evidence_outcome(record, has_evaluation=has_eval, gate_passed=gate_passed)
    if outcome is not expected:
        raise ValueError("terminal_evidence_outcome must match archived terminal state")
    selection = bundle.selection_outcome
    if (
        record.status is AgentCandidateStatus.STATIC_GATE_FAILED
        and selection is SearchSelectionOutcome.SELECT_CANDIDATE
    ):
        raise ValueError("failed gate must not claim candidate selection")
    if outcome is W1TerminalEvidenceOutcome.STATIC_PRE_EXECUTION_REJECTION:
        if record.status is AgentCandidateStatus.STATIC_GATE_FAILED and gate_passed is not False:
            raise ValueError("static gate rejection requires failed static gate evidence")
        if record.status is AgentCandidateStatus.REJECTED and record.static_gate_result is None:
            if has_gate:
                raise ValueError("pre-execution rejection must not fabricate static gate evidence")
            if not has_stop or bundle.w1_lab_stop_evidence is None:
                raise ValueError("pre-execution rejection requires typed stop evidence")
            if bundle.w1_lab_stop_evidence.kind is not AdasW1LabStopKind.PRE_EXECUTION_REJECTION:
                raise ValueError("pre-execution rejection requires matching stop evidence")
    if outcome is W1TerminalEvidenceOutcome.EXECUTION_FAILED_OR_CANCELLED:
        if selection is SearchSelectionOutcome.SELECT_CANDIDATE:
            raise ValueError("failed or cancelled execution must not claim successful candidate selection")
        if (
            not has_stop
            or bundle.w1_lab_stop_evidence is None
            or bundle.w1_lab_stop_evidence.kind
            not in {AdasW1LabStopKind.EXECUTION_FAILURE, AdasW1LabStopKind.EXECUTION_CANCELLATION}
        ):
            raise ValueError("execution failure requires typed stop evidence")
    if outcome is W1TerminalEvidenceOutcome.EVALUATION_COMPLETE and not has_eval:
        raise ValueError("evaluated outcome requires genuine evaluation evidence")
    if selection is SearchSelectionOutcome.SELECT_CANDIDATE:
        if record.status in {
            AgentCandidateStatus.STATIC_GATE_FAILED,
            AgentCandidateStatus.REJECTED,
            AgentCandidateStatus.ARCHIVED,
        }:
            raise ValueError("failed or rejected candidates cannot be selected")
        if gate_passed is not True:
            raise ValueError("SELECT_CANDIDATE requires successful static gate evidence")
        if not has_eval or bundle.evaluation_result is None or not bundle.evaluation_result.passed:
            raise ValueError("SELECT_CANDIDATE requires successful evaluation evidence")
        if bundle.cost_report is None:
            raise ValueError("SELECT_CANDIDATE requires accounting evidence")
        validate_w1_selection_cost_report(
            bundle.cost_report,
            record=record,
            evaluation=bundle.evaluation_result,
            baseline_evaluation=bundle.baseline_evaluation_result,
        )  # type: ignore[arg-type]
    if selection is SearchSelectionOutcome.RETAIN_BASELINE:
        if bundle.baseline_evaluation_result is None:
            raise ValueError("RETAIN_BASELINE requires baseline acceptance evidence")
        if bundle.w1_baseline_retention_provenance is None:
            raise ValueError("RETAIN_BASELINE requires baseline retention provenance")
        validate_w1_baseline_retention(
            provenance=bundle.w1_baseline_retention_provenance,
            record=record,
            baseline_evaluation=bundle.baseline_evaluation_result,
        )


def canonical_record_json(model: BaseModel) -> dict[str, Any]:
    dumped = model.model_dump(mode="json")
    return json.loads(json.dumps(dumped))


def compute_evidence_bundle_digest(bundle: AgentCandidateEvidenceBundle) -> str:
    payload = bundle.model_dump(mode="json")
    payload.pop("bundle_digest", None)
    return hashlib.sha256(canonical_json_bytes(payload)).hexdigest()


def verify_evidence_bundle_digest(bundle: AgentCandidateEvidenceBundle) -> None:
    declared = bundle.bundle_digest
    if declared is None:
        raise ValueError("bundle_digest is required for verification")
    recomputed = compute_evidence_bundle_digest(bundle)
    if declared != recomputed:
        raise ValueError("bundle_digest mismatch")

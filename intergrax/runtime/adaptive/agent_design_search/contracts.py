# © Artur Czarnecki. All rights reserved.

"""Canonical ADAS typed contracts (AHI-ADAS-10, W1 configuration-only LAB profile)."""

from __future__ import annotations

import hashlib
import json
import math
import re
from collections.abc import Iterator, Mapping
from datetime import datetime
from enum import Enum
from typing import Any, Self

from pydantic import BaseModel, ConfigDict, Field, field_serializer, field_validator, model_validator

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
    __slots__ = ("_items",)

    def __init__(self, value: dict[str, AdasJsonValue]) -> None:
        object.__setattr__(
            self,
            "_items",
            {key: _freeze_adas_json_value(child) for key, child in value.items()},
        )

    def __getitem__(self, key: str) -> AdasJsonValue:
        return self._items[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._items)

    def __len__(self) -> int:
        return len(self._items)

    def __contains__(self, key: object) -> bool:
        return key in self._items

    def __eq__(self, other: object) -> bool:
        if isinstance(other, _FrozenAdasJsonObject):
            return self._items == other._items
        if isinstance(other, Mapping):
            return self.to_plain() == dict(other)
        return NotImplemented

    def __repr__(self) -> str:
        return repr(self.to_plain())

    def keys(self) -> Iterator[str]:
        return iter(self._items.keys())

    def values(self) -> Iterator[AdasJsonValue]:
        return iter(self._items.values())

    def items(self) -> Iterator[tuple[str, AdasJsonValue]]:
        return iter(self._items.items())

    def get(self, key: str, default: AdasJsonValue | None = None) -> AdasJsonValue | None:
        return self._items.get(key, default)

    def to_plain(self) -> dict[str, AdasJsonValue]:
        return {key: _json_value_to_plain(child) for key, child in self._items.items()}

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
    normalized = value.strip()
    if any(char.isupper() for char in normalized):
        raise ValueError("digest must be lowercase 64-character hexadecimal SHA-256")
    if not _SHA256_HEX.fullmatch(normalized):
        raise ValueError("digest must be lowercase 64-character hexadecimal SHA-256")
    return normalized


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
    return value


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

    @field_validator("max_candidates", "max_search_iterations", "max_root_executions")
    @classmethod
    def _w1_limits(cls, value: int) -> int:
        return _finite_non_negative_int(value, field="limit")

    @field_validator("aggregate_token_allowance", "aggregate_monetary_allowance")
    @classmethod
    def _budgets(cls, value: float) -> float:
        return _finite_non_negative(value, field="budget")

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

    @field_validator("tokens_consumed", "monetary_consumed")
    @classmethod
    def _measured_floats(cls, value: float | None) -> float | None:
        if value is None:
            return None
        return _finite_non_negative(value, field="consumption")

    @field_validator("root_executions_consumed")
    @classmethod
    def _measured_int(cls, value: int | None) -> int | None:
        if value is None:
            return None
        return _finite_non_negative_int(value, field="root_executions_consumed")


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
    min_utility_delta: float = 0.0
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
        if not math.isfinite(self.min_utility_delta):
            raise ValueError("min_utility_delta must be finite")
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

    @model_validator(mode="after")
    def _scope_and_terminal(self) -> Self:
        if self.status not in _SEARCH_RUN_STATUSES:
            raise ValueError("unrecognized search run status")
        if self.promoted_candidate_id is not None:
            raise ValueError("W1 search runs keep promoted_candidate_id explicitly absent")
        if self.envelope.tenant_id != self.tenant_id or self.envelope.workspace_id != self.workspace_id:
            raise ValueError("envelope scope must match search run scope")
        if self.status in {"completed", "failed", "cancelled"}:
            if self.completed_at is None:
                raise ValueError("terminal search runs require completed_at")
            if self.stop_reason is None or not self.stop_reason.strip():
                raise ValueError("terminal search runs require stop_reason")
            if self.status == "completed" and self.selection_outcome is None:
                raise ValueError("completed search runs require selection_outcome")
            failed_or_cancelled = self.status in {"failed", "cancelled"}
            if failed_or_cancelled and self.selection_outcome == SearchSelectionOutcome.SELECT_CANDIDATE:
                raise ValueError("failed or cancelled runs must not claim candidate selection")
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
    def _forbid_executable_payloads(self) -> Self:
        _reject_executable_payload_tree(self.contract_draft, label="contract_draft")
        _reject_executable_payload_tree(self.prompt_assets, label="prompt_assets")
        _reject_executable_payload_tree(self.tool_policy, label="tool_policy")
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

    @field_validator(
        "utility_score",
        "baseline_utility_score",
        "utility_delta",
        "novelty_score",
        "complexity_score",
        "generation_cost",
        "evaluation_cost",
    )
    @classmethod
    def _optional_finite(cls, value: float | None) -> float | None:
        if value is None:
            return None
        if not math.isfinite(value):
            raise ValueError("numeric field must be finite")
        return value


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
    evidence_complete: bool = False

    @field_validator("candidate_id")
    @classmethod
    def _candidate(cls, value: str) -> str:
        return _non_blank(value, field="candidate_id")

    @field_validator(
        "quality_score",
        "cost_normalized",
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
            )
            if any(item is None for item in required):
                raise ValueError("successful evaluation verdict requires mandatory W1 evidence")
            if not self.evidence_complete:
                raise ValueError("passed evaluation requires evidence_complete")
        return self


class AgentCandidateEvidenceBundle(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

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
    candidate_record: AgentCandidateRecord
    baseline_evaluation_result: AgentCandidateEvaluationResult | None = None
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
        if self.sealed:
            has_gate = self.static_gate_result is not None
            has_eval = self.evaluation_result is not None
            if not has_gate and not has_eval:
                raise ValueError("sealed bundle requires terminal evidence shape")
        if (
            self.artifact_hash is not None
            and self.bundle_digest is not None
            and self.artifact_hash == self.bundle_digest
        ):
            raise ValueError("artifact_hash must not substitute bundle_digest")
        return self


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

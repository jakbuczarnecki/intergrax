# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Task → capability resolution contracts and deterministic baseline (AC-4 Phase 5)."""

from __future__ import annotations

from enum import StrEnum
from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.agent_distribution.capability_matching import (
    AgentCapabilityRequirement,
    CapabilityId,
    CapabilityRequirement,
)
from intergrax.agent_distribution.errors import AgentDistributionError

_NON_EMPTY = Field(min_length=1)

SCHEMA_TASK_CAPABILITY_RESOLVER_ID_V1: Final = "task_capability_resolver_id.v1"
SCHEMA_TASK_KIND_V1: Final = "task_kind.v1"
SCHEMA_TASK_CAPABILITY_RULE_ID_V1: Final = "task_capability_rule_id.v1"
SCHEMA_TASK_CAPABILITY_RULE_VERSION_V1: Final = "task_capability_rule_version.v1"
SCHEMA_TASK_CAPABILITY_RESOLUTION_REQUEST_V1: Final = (
    "task_capability_resolution_request.v1"
)
SCHEMA_TASK_CAPABILITY_EVIDENCE_V1: Final = "task_capability_evidence.v1"
SCHEMA_TASK_CAPABILITY_RESOLUTION_RESULT_V1: Final = (
    "task_capability_resolution_result.v1"
)


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


class TaskCapabilityResolutionError(AgentDistributionError):
    """Base error for task capability resolution contract violations."""


class TaskCapabilityResolutionContractError(TaskCapabilityResolutionError):
    """Malformed resolution request, rule, or result."""


class TaskCapabilityResolutionNoMatch(TaskCapabilityResolutionError):
    """No deterministic rule matched the task request."""


class TaskCapabilityResolutionConflict(TaskCapabilityResolutionError):
    """Multiple conflicting rules matched the same task request."""


class TaskCapabilityResolverId(BaseModel):
    """Stable plugin identifier — not derived from class name."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_CAPABILITY_RESOLVER_ID_V1
    value: str = _NON_EMPTY

    @field_validator("value")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return _strip_required(value)

    def __str__(self) -> str:
        return self.value


DETERMINISTIC_MAPPING_RESOLVER_ID = TaskCapabilityResolverId(
    value="deterministic.mapping",
)


class TaskKind(BaseModel):
    """Normalized task kind / intent identifier for deterministic rule matching."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_KIND_V1
    value: str = _NON_EMPTY

    @field_validator("value")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return _strip_required(value)

    def __str__(self) -> str:
        return self.value


class TaskCapabilityRuleId(BaseModel):
    """Stable rule identity for audit and replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_CAPABILITY_RULE_ID_V1
    value: str = _NON_EMPTY

    @field_validator("value")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return _strip_required(value)

    def __str__(self) -> str:
        return self.value


class TaskCapabilityRuleVersion(BaseModel):
    """Version stamp for rule-driven mappings — supports future replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_CAPABILITY_RULE_VERSION_V1
    value: str = _NON_EMPTY

    @field_validator("value")
    @classmethod
    def _normalize(cls, value: str) -> str:
        return _strip_required(value)

    def __str__(self) -> str:
        return self.value


class CapabilityRequirementKind(StrEnum):
    """Required vs optional classification owned by the resolver."""

    REQUIRED = "required"
    OPTIONAL = "optional"


class TaskCapabilityEvidenceRationaleCategory(StrEnum):
    """Typed rationale — not free-form prose authority."""

    DETERMINISTIC_RULE_MATCH = "deterministic_rule_match"


class TaskCapabilityResolutionRequest(BaseModel):
    """Origin-agnostic task need — no agent ids, discovery, or lifecycle fields."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_CAPABILITY_RESOLUTION_REQUEST_V1
    task_kind: TaskKind
    task_text: str | None = None

    @field_validator("task_text")
    @classmethod
    def _strip_optional_text(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class TaskCapabilityEvidence(BaseModel):
    """Typed per-capability resolution evidence for observability and replay."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_CAPABILITY_EVIDENCE_V1
    capability_id: CapabilityId
    requirement_kind: CapabilityRequirementKind
    rule_id: TaskCapabilityRuleId
    rule_version: TaskCapabilityRuleVersion
    rationale_category: TaskCapabilityEvidenceRationaleCategory


class TaskCapabilityResolutionResult(BaseModel):
    """Canonical resolution output — requirement authority plus typed evidence."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_TASK_CAPABILITY_RESOLUTION_RESULT_V1
    resolver_id: TaskCapabilityResolverId
    request: TaskCapabilityResolutionRequest
    capability_requirement: AgentCapabilityRequirement
    evidence: tuple[TaskCapabilityEvidence, ...]


class TaskCapabilityRule(BaseModel):
    """Immutable deterministic mapping from task kind to capability requirement."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    rule_id: TaskCapabilityRuleId
    rule_version: TaskCapabilityRuleVersion = TaskCapabilityRuleVersion(value="1")
    task_kind: TaskKind
    required_capabilities: tuple[CapabilityId, ...] = ()
    optional_capabilities: tuple[CapabilityId, ...] = ()

    @model_validator(mode="after")
    def _validate_capabilities(self) -> TaskCapabilityRule:
        if not self.required_capabilities and not self.optional_capabilities:
            raise TaskCapabilityResolutionContractError(
                "task capability rule must declare at least one capability",
            )
        _assert_unique_capability_ids(
            self.required_capabilities,
            label="required_capabilities",
        )
        _assert_unique_capability_ids(
            self.optional_capabilities,
            label="optional_capabilities",
        )
        required_values = {item.value for item in self.required_capabilities}
        for capability in self.optional_capabilities:
            if capability.value in required_values:
                raise TaskCapabilityResolutionContractError(
                    "capability cannot be both required and optional in one rule",
                )
        return self


def _assert_unique_capability_ids(
    capabilities: tuple[CapabilityId, ...],
    *,
    label: str,
) -> None:
    seen: set[str] = set()
    for capability in capabilities:
        if capability.value in seen:
            raise TaskCapabilityResolutionContractError(
                f"duplicate capability in {label}: {capability.value!r}",
            )
        seen.add(capability.value)


def _sorted_capability_ids(
    capabilities: tuple[CapabilityId, ...],
) -> tuple[CapabilityId, ...]:
    return tuple(sorted(capabilities, key=lambda item: item.value))


def build_task_capability_rule(
    *,
    rule_id: str,
    task_kind: str,
    required: tuple[str, ...] = (),
    optional: tuple[str, ...] = (),
    rule_version: str = "1",
) -> TaskCapabilityRule:
    """Construct a validated deterministic rule from normalized id strings."""
    return TaskCapabilityRule(
        rule_id=TaskCapabilityRuleId(value=rule_id),
        rule_version=TaskCapabilityRuleVersion(value=rule_version),
        task_kind=TaskKind(value=task_kind),
        required_capabilities=tuple(CapabilityId(value=item) for item in required),
        optional_capabilities=tuple(CapabilityId(value=item) for item in optional),
    )


def build_task_capability_resolution_request(
    *,
    task_kind: str,
    task_text: str | None = None,
) -> TaskCapabilityResolutionRequest:
    """Construct a validated origin-agnostic resolution request."""
    return TaskCapabilityResolutionRequest(
        task_kind=TaskKind(value=task_kind),
        task_text=task_text,
    )


def _build_requirement_from_rule(
    rule: TaskCapabilityRule,
) -> AgentCapabilityRequirement:
    required = _sorted_capability_ids(rule.required_capabilities)
    optional = _sorted_capability_ids(rule.optional_capabilities)
    requirements: list[CapabilityRequirement] = [
        CapabilityRequirement(capability_id=capability_id, required=True)
        for capability_id in required
    ]
    requirements.extend(
        CapabilityRequirement(capability_id=capability_id, required=False)
        for capability_id in optional
    )
    return AgentCapabilityRequirement(requirements=tuple(requirements))


def _build_evidence_from_rule(
    rule: TaskCapabilityRule,
) -> tuple[TaskCapabilityEvidence, ...]:
    evidence: list[TaskCapabilityEvidence] = []
    for capability_id in _sorted_capability_ids(rule.required_capabilities):
        evidence.append(
            TaskCapabilityEvidence(
                capability_id=capability_id,
                requirement_kind=CapabilityRequirementKind.REQUIRED,
                rule_id=rule.rule_id,
                rule_version=rule.rule_version,
                rationale_category=(
                    TaskCapabilityEvidenceRationaleCategory.DETERMINISTIC_RULE_MATCH
                ),
            ),
        )
    for capability_id in _sorted_capability_ids(rule.optional_capabilities):
        evidence.append(
            TaskCapabilityEvidence(
                capability_id=capability_id,
                requirement_kind=CapabilityRequirementKind.OPTIONAL,
                rule_id=rule.rule_id,
                rule_version=rule.rule_version,
                rationale_category=(
                    TaskCapabilityEvidenceRationaleCategory.DETERMINISTIC_RULE_MATCH
                ),
            ),
        )
    return tuple(evidence)


class TaskCapabilityResolver(Protocol):
    """Structural task capability resolution plugin — no registry or service locator."""

    @property
    def resolver_id(self) -> TaskCapabilityResolverId:
        """Stable resolver identifier."""

    def resolve(
        self,
        request: TaskCapabilityResolutionRequest,
    ) -> TaskCapabilityResolutionResult:
        """Derive canonical capability requirement without discovery or matching."""


class StructuredTaskCapabilityInferencePort(Protocol):
    """Future LLM resolver boundary — adapter selected at composition root only.

    LLM output must pass strict validation before becoming AgentCapabilityRequirement
    authority; never treat raw model output as canonical requirement directly.
    """

    def infer(
        self,
        request: TaskCapabilityResolutionRequest,
    ) -> TaskCapabilityResolutionResult:
        """Infer capability requirement from task representation."""


def _validate_unique_rule_ids(rules: tuple[TaskCapabilityRule, ...]) -> None:
    seen: set[str] = set()
    for rule in rules:
        key = rule.rule_id.value
        if key in seen:
            raise TaskCapabilityResolutionContractError(
                f"duplicate task capability rule_id: {key!r}",
            )
        seen.add(key)


class DeterministicTaskCapabilityResolver:
    """Baseline resolver — exact task_kind match against explicit typed rules."""

    def __init__(self, rules: tuple[TaskCapabilityRule, ...]) -> None:
        if not rules:
            raise TaskCapabilityResolutionContractError(
                "deterministic resolver requires at least one rule",
            )
        _validate_unique_rule_ids(rules)
        self._rules = rules

    @property
    def resolver_id(self) -> TaskCapabilityResolverId:
        return DETERMINISTIC_MAPPING_RESOLVER_ID

    def resolve(
        self,
        request: TaskCapabilityResolutionRequest,
    ) -> TaskCapabilityResolutionResult:
        matched = tuple(
            rule for rule in self._rules if rule.task_kind == request.task_kind
        )
        if not matched:
            raise TaskCapabilityResolutionNoMatch(
                f"no task capability rule matched task_kind={request.task_kind.value!r}",
            )
        if len(matched) > 1:
            rule_ids = ", ".join(rule.rule_id.value for rule in matched)
            raise TaskCapabilityResolutionConflict(
                f"multiple task capability rules matched task_kind="
                f"{request.task_kind.value!r}: {rule_ids}",
            )
        rule = matched[0]
        return TaskCapabilityResolutionResult(
            resolver_id=self.resolver_id,
            request=request,
            capability_requirement=_build_requirement_from_rule(rule),
            evidence=_build_evidence_from_rule(rule),
        )


def build_deterministic_task_capability_resolver(
    rules: tuple[TaskCapabilityRule, ...],
) -> DeterministicTaskCapabilityResolver:
    """Composition-root helper for the baseline deterministic resolver."""
    return DeterministicTaskCapabilityResolver(rules=rules)

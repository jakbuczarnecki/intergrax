# © Artur Czarnecki. All rights reserved.

"""Typed, redaction-safe contracts for TOKEN-10G evaluation."""

from __future__ import annotations

import json
import math
import re
from collections.abc import Mapping
from dataclasses import dataclass
from datetime import datetime
from enum import StrEnum
from pathlib import Path
from typing import Any

from intergrax.runtime.token_optimization.contracts import (
    ProtectedRegion,
    ProtectedRegionKind,
    TokenOptimizationProfile,
    TokenOptimizationSourceType,
)
from intergrax.runtime.token_optimization.proofs.contracts import (
    SCHEMA_VERSION,
    ProofArtifactRef,
    ProofMeasurement,
    ProofPipelineEvidence,
    ProofPrefixIdentityEvidence,
    ProofProtectedRegionEvidence,
    ProofRouterEvidence,
    UniversalProofArtifactManifest,
    UniversalProofCaseResult,
    UniversalProofEnvironmentSummary,
    UniversalProofRunResult,
)

EVALUATION_SCHEMA_VERSION = "token-optimization-proof-evaluation.v1"
CORPUS_SCHEMA_VERSION = "token-optimization-proof-corpus.v1"
CACHE_EVIDENCE_SCHEMA_VERSION = "token-optimization-cache-evidence.v1"

SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")
SAFE_TEXT_RE = re.compile(r"^[A-Za-z0-9._:/ -]{1,256}$")
SHA256_RE = re.compile(r"^[0-9a-f]{64}$")


class EvaluationConfigurationError(ValueError):
    """The corpus or evaluation configuration is invalid."""

    def __init__(self, reason_code: str) -> None:
        self.reason_code = reason_code
        super().__init__(reason_code)


class GateStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"
    NOT_APPLICABLE = "NOT_APPLICABLE"
    UNAVAILABLE = "UNAVAILABLE"


class MeasurementRequirement(StrEnum):
    REQUIRED = "required"
    OPTIONAL = "optional"
    UNAVAILABLE_ALLOWED = "unavailable_allowed"
    NOT_APPLICABLE = "not_applicable"


class CacheExpectationMode(StrEnum):
    NOT_APPLICABLE = "not_applicable"
    UNAVAILABLE_ALLOWED = "unavailable_allowed"
    COLD = "cold"
    WARM_EXPECTED = "warm_expected"
    CHANGED_PREFIX_NEGATIVE_CONTROL = "changed_prefix_negative_control"


class CacheAttribution(StrEnum):
    REUSE_CONFIRMED = "reuse_confirmed"
    MISS_CONFIRMED = "miss_confirmed"
    UNAVAILABLE = "unavailable"
    LATENCY_ONLY = "latency_only"
    CONFLICTING = "conflicting"


class CacheEvidenceRole(StrEnum):
    COLD = "cold"
    WARM_EXPECTED = "warm_expected"
    CHANGED_PREFIX_NEGATIVE_CONTROL = "changed_prefix_negative_control"


def _safe_id(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not SAFE_ID_RE.fullmatch(value):
        raise EvaluationConfigurationError(f"INVALID_{field_name.upper()}")
    return value


def _safe_text(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not value.strip() or len(value) > 512:
        raise EvaluationConfigurationError(f"INVALID_{field_name.upper()}")
    if any(ord(char) < 32 and char not in "\n\t" for char in value):
        raise EvaluationConfigurationError(f"INVALID_{field_name.upper()}")
    return value


def _safe_code(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not SAFE_TEXT_RE.fullmatch(value):
        raise EvaluationConfigurationError(f"INVALID_{field_name.upper()}")
    return value


def _digest(value: object, field_name: str) -> str:
    if not isinstance(value, str) or not SHA256_RE.fullmatch(value):
        raise EvaluationConfigurationError(f"INVALID_{field_name.upper()}")
    return value


def _strict_keys(value: Mapping[str, Any], allowed: frozenset[str], name: str) -> None:
    if set(value) - allowed:
        raise EvaluationConfigurationError(f"UNKNOWN_{name.upper()}_FIELD")


def _bool_or_none(value: object, field_name: str) -> bool | None:
    if value is None:
        return None
    if type(value) is not bool:
        raise EvaluationConfigurationError(f"INVALID_{field_name.upper()}")
    return value


def _string_set(value: object, field_name: str) -> frozenset[str]:
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise EvaluationConfigurationError(f"INVALID_{field_name.upper()}")
    return frozenset(_safe_code(item, field_name) for item in value)


@dataclass(frozen=True, slots=True)
class RouterExpectation:
    allowed_statuses: frozenset[str] = frozenset()
    allowed_configuration_ids: frozenset[str] = frozenset()
    allowed_reason_codes: frozenset[str] = frozenset()
    review_required: bool | None = None
    confidence_minimum: float | None = None
    confidence_maximum: float | None = None
    allowed_risk: frozenset[str] = frozenset()
    allowed_transport: frozenset[str] = frozenset()
    structured_output_fallback: bool | None = None

    def __post_init__(self) -> None:
        if (
            self.confidence_minimum is not None
            and self.confidence_maximum is not None
            and self.confidence_minimum > self.confidence_maximum
        ):
            raise ValueError("confidence minimum exceeds maximum")
        for value in (
            self.confidence_minimum,
            self.confidence_maximum,
        ):
            if value is not None and (not math.isfinite(value) or not 0 <= value <= 1):
                raise ValueError("confidence bounds must be finite percentages")


@dataclass(frozen=True, slots=True)
class PipelineExpectation:
    expected_completion: bool | None = None
    required_layer_ids: frozenset[str] = frozenset()
    allowed_layer_ids: frozenset[str] = frozenset()
    forbidden_layer_ids: frozenset[str] = frozenset()
    expected_fallback: bool | None = None
    expected_validation_status: str | None = None
    allowed_validation_reason_codes: frozenset[str] = frozenset()
    required_layer_failure_expected: bool | None = None


@dataclass(frozen=True, slots=True)
class ProtectedRegionExpectation:
    expected_input_count: int | None = None
    expected_preserved_count: int | None = None
    expected_validation_status: str | None = None
    digest_equality_required: bool = False
    no_raw_protected_values: bool = True


@dataclass(frozen=True, slots=True)
class MeasurementExpectation:
    baseline: MeasurementRequirement = MeasurementRequirement.OPTIONAL
    optimized: MeasurementRequirement = MeasurementRequirement.OPTIONAL
    ordering_required: bool = False


@dataclass(frozen=True, slots=True)
class PrefixExpectation:
    identity_required: bool = False
    same_as_case_id: str | None = None
    different_from_case_id: str | None = None
    tool_schema_identity: str | None = None


@dataclass(frozen=True, slots=True)
class CacheExpectation:
    mode: CacheExpectationMode = CacheExpectationMode.NOT_APPLICABLE
    same_as_case_id: str | None = None


@dataclass(frozen=True, slots=True)
class CorpusCase:
    case_id: str
    category: str
    description: str
    input_case_id: str
    safe_tags: tuple[str, ...]
    source_type: TokenOptimizationSourceType | None
    policy_enabled: bool | None
    policy_profile: TokenOptimizationProfile | None
    allow_lossy: bool | None
    protected_regions: tuple[ProtectedRegion, ...]
    router: RouterExpectation
    pipeline: PipelineExpectation
    protected: ProtectedRegionExpectation
    measurement: MeasurementExpectation
    prefix: PrefixExpectation
    cache: CacheExpectation


@dataclass(frozen=True, slots=True)
class ProofCorpus:
    schema_version: str
    corpus_id: str
    evaluation_version: str
    cases: tuple[CorpusCase, ...]

    def __post_init__(self) -> None:
        ids = [case.case_id for case in self.cases]
        if len(ids) != len(set(ids)):
            raise ValueError("duplicate corpus case IDs are not allowed")


@dataclass(frozen=True, slots=True)
class EvaluationConfiguration:
    schema_version: str
    evaluation_version: str
    required_gate_ids: tuple[str, ...]
    unavailable_allowed_gate_ids: frozenset[str]

    def __post_init__(self) -> None:
        if self.schema_version != EVALUATION_SCHEMA_VERSION:
            raise ValueError("unsupported evaluation schema version")
        if len(self.required_gate_ids) != len(set(self.required_gate_ids)):
            raise ValueError("duplicate gate IDs are not allowed")
        unknown = set(self.unavailable_allowed_gate_ids) - set(self.required_gate_ids)
        if unknown:
            raise ValueError("unavailable allowance references unknown gate")


@dataclass(frozen=True, slots=True)
class ProviderCacheEvidence:
    case_id: str
    provider: str
    model: str
    stable_prefix_identity: str | None
    prompt_token_count: int | None
    cached_prompt_token_count: int | None
    cache_attribution: CacheAttribution
    role: CacheEvidenceRole
    reason_code: str

    def __post_init__(self) -> None:
        _safe_id(self.case_id, "case_id")
        _safe_code(self.provider, "provider")
        _safe_code(self.model, "model")
        if self.stable_prefix_identity is not None:
            _digest(self.stable_prefix_identity, "stable_prefix_identity")
        for value in (self.prompt_token_count, self.cached_prompt_token_count):
            if value is not None and (type(value) is not int or value < 0):
                raise ValueError("cache token counts must be non-negative integers")
        if (
            self.prompt_token_count is not None
            and self.cached_prompt_token_count is not None
            and self.cached_prompt_token_count > self.prompt_token_count
        ):
            raise ValueError("cached prompt tokens exceed prompt tokens")
        _safe_code(self.reason_code, "reason_code")


@dataclass(frozen=True, slots=True)
class GateResult:
    gate_id: str
    status: GateStatus
    case_id: str
    reason_code: str
    expected_safe_summary: str
    actual_safe_summary: str
    required: bool


@dataclass(frozen=True, slots=True)
class CaseEvaluation:
    case_id: str
    category: str
    description: str
    gates: tuple[GateResult, ...]

    @property
    def failed_gate_ids(self) -> tuple[str, ...]:
        return tuple(
            gate.gate_id for gate in self.gates if gate.status is GateStatus.FAIL
        )


@dataclass(frozen=True, slots=True)
class UniversalProofEvaluation:
    evaluation_id: str
    proof_id: str
    run_id: str
    corpus_version: str
    evaluation_version: str
    run_mode: str
    provider: str
    model: str
    cases: tuple[CaseEvaluation, ...]
    status_counts: Mapping[str, int]
    success: bool
    artifact_refs: tuple[str, ...] = ()

    def to_dict(self) -> dict[str, Any]:
        return {
            "schema_version": EVALUATION_SCHEMA_VERSION,
            "evaluation_id": self.evaluation_id,
            "proof_id": self.proof_id,
            "run_id": self.run_id,
            "corpus_version": self.corpus_version,
            "evaluation_version": self.evaluation_version,
            "run_mode": self.run_mode,
            "provider": self.provider,
            "model": self.model,
            "case_count": len(self.cases),
            "status_counts": {
                status.value: self.status_counts.get(status.value, 0)
                for status in GateStatus
            },
            "success": self.success,
            "artifact_refs": list(self.artifact_refs),
            "cases": [
                {
                    "case_id": case.case_id,
                    "category": case.category,
                    "description": case.description,
                    "failed_gate_ids": list(case.failed_gate_ids),
                    "gates": [
                        {
                            "gate_id": gate.gate_id,
                            "status": gate.status.value,
                            "case_id": gate.case_id,
                            "reason_code": gate.reason_code,
                            "expected_safe_summary": gate.expected_safe_summary,
                            "actual_safe_summary": gate.actual_safe_summary,
                            "required": gate.required,
                        }
                        for gate in case.gates
                    ],
                }
                for case in self.cases
            ],
            "raw_content_included": False,
        }


def _parse_measurement(value: Mapping[str, Any], name: str) -> MeasurementExpectation:
    _strict_keys(value, frozenset({"baseline", "optimized", "ordering_required"}), name)
    try:
        baseline = MeasurementRequirement(value.get("baseline", "optional"))
        optimized = MeasurementRequirement(value.get("optimized", "optional"))
    except ValueError as exc:
        raise EvaluationConfigurationError(f"INVALID_{name.upper()}") from exc
    ordering = value.get("ordering_required", False)
    if type(ordering) is not bool:
        raise EvaluationConfigurationError(f"INVALID_{name.upper()}")
    return MeasurementExpectation(baseline, optimized, ordering)


def load_evaluation_config(path: str | Path) -> EvaluationConfiguration:
    import tomllib

    source = Path(path)
    try:
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise EvaluationConfigurationError("INVALID_EVALUATION_TOML") from exc
    _strict_keys(
        data,
        frozenset(
            {
                "schema_version",
                "evaluation_version",
                "required_gate_ids",
                "unavailable_allowed_gate_ids",
            }
        ),
        "evaluation",
    )
    try:
        result = EvaluationConfiguration(
            schema_version=data["schema_version"],
            evaluation_version=_safe_id(
                data["evaluation_version"], "evaluation_version"
            ),
            required_gate_ids=tuple(
                _safe_id(value, "gate_id") for value in data["required_gate_ids"]
            ),
            unavailable_allowed_gate_ids=_string_set(
                data.get("unavailable_allowed_gate_ids", []),
                "unavailable_allowed_gate_ids",
            ),
        )
    except (KeyError, TypeError, ValueError) as exc:
        if isinstance(exc, EvaluationConfigurationError):
            raise
        raise EvaluationConfigurationError("INVALID_EVALUATION_CONFIG") from exc
    return result


def _parse_router(value: Mapping[str, Any]) -> RouterExpectation:
    _strict_keys(
        value,
        frozenset(
            {
                "allowed_statuses",
                "allowed_configuration_ids",
                "allowed_reason_codes",
                "review_required",
                "confidence_minimum",
                "confidence_maximum",
                "allowed_risk",
                "allowed_transport",
                "structured_output_fallback",
            }
        ),
        "router_expectation",
    )
    return RouterExpectation(
        allowed_statuses=_string_set(
            value.get("allowed_statuses", []), "allowed_statuses"
        ),
        allowed_configuration_ids=_string_set(
            value.get("allowed_configuration_ids", []), "allowed_configuration_ids"
        ),
        allowed_reason_codes=_string_set(
            value.get("allowed_reason_codes", []), "allowed_reason_codes"
        ),
        review_required=_bool_or_none(value.get("review_required"), "review_required"),
        confidence_minimum=value.get("confidence_minimum"),
        confidence_maximum=value.get("confidence_maximum"),
        allowed_risk=_string_set(value.get("allowed_risk", []), "allowed_risk"),
        allowed_transport=_string_set(
            value.get("allowed_transport", []), "allowed_transport"
        ),
        structured_output_fallback=_bool_or_none(
            value.get("structured_output_fallback"), "structured_output_fallback"
        ),
    )


def _parse_pipeline(value: Mapping[str, Any]) -> PipelineExpectation:
    _strict_keys(
        value,
        frozenset(
            {
                "expected_completion",
                "required_layer_ids",
                "allowed_layer_ids",
                "forbidden_layer_ids",
                "expected_fallback",
                "expected_validation_status",
                "allowed_validation_reason_codes",
                "required_layer_failure_expected",
            }
        ),
        "pipeline_expectation",
    )
    return PipelineExpectation(
        expected_completion=_bool_or_none(
            value.get("expected_completion"), "expected_completion"
        ),
        required_layer_ids=_string_set(
            value.get("required_layer_ids", []), "required_layer_ids"
        ),
        allowed_layer_ids=_string_set(
            value.get("allowed_layer_ids", []), "allowed_layer_ids"
        ),
        forbidden_layer_ids=_string_set(
            value.get("forbidden_layer_ids", []), "forbidden_layer_ids"
        ),
        expected_fallback=_bool_or_none(
            value.get("expected_fallback"), "expected_fallback"
        ),
        expected_validation_status=(
            _safe_code(value["expected_validation_status"], "validation_status")
            if value.get("expected_validation_status") is not None
            else None
        ),
        allowed_validation_reason_codes=_string_set(
            value.get("allowed_validation_reason_codes", []),
            "allowed_validation_reason_codes",
        ),
        required_layer_failure_expected=_bool_or_none(
            value.get("required_layer_failure_expected"),
            "required_layer_failure_expected",
        ),
    )


def _parse_protected(value: Mapping[str, Any]) -> ProtectedRegionExpectation:
    _strict_keys(
        value,
        frozenset(
            {
                "expected_input_count",
                "expected_preserved_count",
                "expected_validation_status",
                "digest_equality_required",
                "no_raw_protected_values",
            }
        ),
        "protected_expectation",
    )
    for name in ("expected_input_count", "expected_preserved_count"):
        count = value.get(name)
        if count is not None and (type(count) is not int or count < 0):
            raise EvaluationConfigurationError(f"INVALID_{name.upper()}")
    return ProtectedRegionExpectation(
        expected_input_count=value.get("expected_input_count"),
        expected_preserved_count=value.get("expected_preserved_count"),
        expected_validation_status=(
            _safe_code(value["expected_validation_status"], "validation_status")
            if value.get("expected_validation_status") is not None
            else None
        ),
        digest_equality_required=value.get("digest_equality_required", False),
        no_raw_protected_values=value.get("no_raw_protected_values", True),
    )


def _parse_prefix(value: Mapping[str, Any]) -> PrefixExpectation:
    _strict_keys(
        value,
        frozenset(
            {
                "identity_required",
                "same_as_case_id",
                "different_from_case_id",
                "tool_schema_identity",
            }
        ),
        "prefix_expectation",
    )
    tool_identity = value.get("tool_schema_identity")
    if tool_identity not in {None, "same", "different"}:
        raise EvaluationConfigurationError("INVALID_TOOL_SCHEMA_IDENTITY")
    return PrefixExpectation(
        identity_required=value.get("identity_required", False),
        same_as_case_id=value.get("same_as_case_id"),
        different_from_case_id=value.get("different_from_case_id"),
        tool_schema_identity=tool_identity,
    )


def _parse_cache(value: Mapping[str, Any]) -> CacheExpectation:
    _strict_keys(value, frozenset({"mode", "same_as_case_id"}), "cache_expectation")
    try:
        mode = CacheExpectationMode(value.get("mode", "not_applicable"))
    except ValueError as exc:
        raise EvaluationConfigurationError("INVALID_CACHE_MODE") from exc
    return CacheExpectation(
        mode=mode,
        same_as_case_id=(
            _safe_id(value["same_as_case_id"], "same_as_case_id")
            if value.get("same_as_case_id") is not None
            else None
        ),
    )


def load_cache_evidence(path: str | Path) -> tuple[ProviderCacheEvidence, ...]:
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationConfigurationError("INVALID_CACHE_EVIDENCE") from exc
    if not isinstance(payload, dict):
        raise EvaluationConfigurationError("INVALID_CACHE_EVIDENCE")
    _strict_keys(payload, frozenset({"schema_version", "evidence"}), "cache_evidence")
    if payload.get("schema_version") != CACHE_EVIDENCE_SCHEMA_VERSION:
        raise EvaluationConfigurationError("UNSUPPORTED_CACHE_EVIDENCE_VERSION")
    entries = payload.get("evidence")
    if not isinstance(entries, list):
        raise EvaluationConfigurationError("INVALID_CACHE_EVIDENCE")
    result = []
    for entry in entries:
        if not isinstance(entry, dict):
            raise EvaluationConfigurationError("INVALID_CACHE_EVIDENCE_ENTRY")
        _strict_keys(
            entry,
            frozenset(
                {
                    "case_id",
                    "provider",
                    "model",
                    "stable_prefix_identity",
                    "prompt_token_count",
                    "cached_prompt_token_count",
                    "cache_attribution",
                    "role",
                    "reason_code",
                }
            ),
            "cache_evidence_entry",
        )
        try:
            result.append(
                ProviderCacheEvidence(
                    case_id=_safe_id(entry["case_id"], "case_id"),
                    provider=_safe_code(entry["provider"], "provider"),
                    model=_safe_code(entry["model"], "model"),
                    stable_prefix_identity=(
                        _digest(
                            entry["stable_prefix_identity"], "stable_prefix_identity"
                        )
                        if entry.get("stable_prefix_identity") is not None
                        else None
                    ),
                    prompt_token_count=entry.get("prompt_token_count"),
                    cached_prompt_token_count=entry.get("cached_prompt_token_count"),
                    cache_attribution=CacheAttribution(entry["cache_attribution"]),
                    role=CacheEvidenceRole(entry["role"]),
                    reason_code=_safe_code(entry["reason_code"], "reason_code"),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise EvaluationConfigurationError("INVALID_CACHE_EVIDENCE_ENTRY") from exc
    ids = [item.case_id for item in result]
    if len(ids) != len(set(ids)):
        raise EvaluationConfigurationError("DUPLICATE_CACHE_EVIDENCE_CASE_IDS")
    return tuple(result)


def _parse_protected_regions(value: object) -> tuple[ProtectedRegion, ...]:
    if value is None:
        return ()
    if not isinstance(value, list):
        raise EvaluationConfigurationError("INVALID_PROTECTED_REGIONS")
    regions = []
    for item in value:
        if not isinstance(item, dict):
            raise EvaluationConfigurationError("INVALID_PROTECTED_REGION")
        _strict_keys(item, frozenset({"kind", "value"}), "protected_region")
        try:
            regions.append(
                ProtectedRegion(
                    kind=ProtectedRegionKind(item["kind"]),
                    value=_safe_text(item["value"], "protected_value"),
                )
            )
        except (KeyError, ValueError, TypeError) as exc:
            raise EvaluationConfigurationError("INVALID_PROTECTED_REGION") from exc
    return tuple(regions)


def load_proof_corpus(path: str | Path) -> ProofCorpus:
    """Load the expectation corpus; input content remains in TOKEN-10F config."""
    import tomllib

    source = Path(path)
    try:
        data = tomllib.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, tomllib.TOMLDecodeError) as exc:
        raise EvaluationConfigurationError("INVALID_CORPUS_TOML") from exc
    _strict_keys(
        data,
        frozenset({"schema_version", "corpus_id", "evaluation_version", "cases"}),
        "corpus",
    )
    if data.get("schema_version") != CORPUS_SCHEMA_VERSION:
        raise EvaluationConfigurationError("UNSUPPORTED_CORPUS_VERSION")
    raw_cases = data.get("cases")
    if not isinstance(raw_cases, list) or not raw_cases:
        raise EvaluationConfigurationError("INVALID_CORPUS_CASES")
    cases = []
    for raw in raw_cases:
        if not isinstance(raw, dict):
            raise EvaluationConfigurationError("INVALID_CORPUS_CASE")
        _strict_keys(
            raw,
            frozenset(
                {
                    "case_id",
                    "category",
                    "description",
                    "input_case_id",
                    "safe_tags",
                    "source_type",
                    "policy_enabled",
                    "policy_profile",
                    "allow_lossy",
                    "protected_regions",
                    "router",
                    "pipeline",
                    "protected",
                    "measurement",
                    "prefix",
                    "cache",
                }
            ),
            "corpus_case",
        )
        try:
            source_type = (
                TokenOptimizationSourceType(raw["source_type"])
                if raw.get("source_type") is not None
                else None
            )
            profile = (
                TokenOptimizationProfile(raw["policy_profile"])
                if raw.get("policy_profile") is not None
                else None
            )
            tags = tuple(_safe_id(tag, "safe_tag") for tag in raw.get("safe_tags", []))
            if not isinstance(raw.get("safe_tags", []), list):
                raise EvaluationConfigurationError("INVALID_SAFE_TAGS")
            cases.append(
                CorpusCase(
                    case_id=_safe_id(raw["case_id"], "case_id"),
                    category=_safe_id(raw["category"], "category"),
                    description=_safe_text(raw["description"], "description"),
                    input_case_id=_safe_id(raw["input_case_id"], "input_case_id"),
                    safe_tags=tags,
                    source_type=source_type,
                    policy_enabled=_bool_or_none(
                        raw.get("policy_enabled"), "policy_enabled"
                    ),
                    policy_profile=profile,
                    allow_lossy=_bool_or_none(raw.get("allow_lossy"), "allow_lossy"),
                    protected_regions=_parse_protected_regions(
                        raw.get("protected_regions")
                    ),
                    router=_parse_router(raw["router"]),
                    pipeline=_parse_pipeline(raw["pipeline"]),
                    protected=_parse_protected(raw["protected"]),
                    measurement=_parse_measurement(raw["measurement"], "measurement"),
                    prefix=_parse_prefix(raw["prefix"]),
                    cache=_parse_cache(raw["cache"]),
                )
            )
        except (KeyError, TypeError, ValueError) as exc:
            if isinstance(exc, EvaluationConfigurationError):
                raise
            raise EvaluationConfigurationError("INVALID_CORPUS_CASE") from exc
    try:
        return ProofCorpus(
            schema_version=data["schema_version"],
            corpus_id=_safe_id(data["corpus_id"], "corpus_id"),
            evaluation_version=_safe_id(
                data["evaluation_version"], "evaluation_version"
            ),
            cases=tuple(cases),
        )
    except (KeyError, TypeError, ValueError) as exc:
        raise EvaluationConfigurationError("INVALID_CORPUS") from exc


def expand_proof_config_with_corpus(proof_config, corpus: ProofCorpus):
    """Clone canonical TOKEN-10F inputs without parsing or executing them."""
    from dataclasses import replace

    by_id = {case.case_id: case for case in proof_config.cases}
    expanded = []
    for corpus_case in corpus.cases:
        source = by_id.get(corpus_case.input_case_id)
        if source is None:
            raise EvaluationConfigurationError("CORPUS_INPUT_CASE_NOT_FOUND")
        request = source.request
        if corpus_case.source_type is not None:
            request = replace(request, source_type=corpus_case.source_type)
        if (
            corpus_case.policy_enabled is not None
            or corpus_case.policy_profile is not None
        ):
            policy = request.policy
            request = replace(
                request,
                policy=replace(
                    policy,
                    enabled=(
                        corpus_case.policy_enabled
                        if corpus_case.policy_enabled is not None
                        else policy.enabled
                    ),
                    profile=(
                        corpus_case.policy_profile
                        if corpus_case.policy_profile is not None
                        else policy.profile
                    ),
                ),
            )
        if corpus_case.allow_lossy is not None:
            request = replace(
                request,
                policy=replace(request.policy, allow_lossy=corpus_case.allow_lossy),
            )
        if corpus_case.protected_regions:
            request = replace(request, protected_regions=corpus_case.protected_regions)
        expanded.append(replace(source, case_id=corpus_case.case_id, request=request))
    return replace(proof_config, cases=tuple(expanded), case_source=None)


def _measurement(value: Mapping[str, Any]) -> ProofMeasurement:
    if set(value) != {"available", "value"}:
        raise ValueError("invalid measurement fields")
    return ProofMeasurement(available=value["available"], value=value["value"])


def _evidence(value: Mapping[str, Any], kind: str):
    if kind == "router":
        allowed = {
            "status",
            "configuration_id",
            "reason_code",
            "review_required",
            "confidence",
            "risk",
            "transport",
            "structured_output_fallback_used",
        }
        if set(value) != allowed:
            raise ValueError("invalid router evidence fields")
        return ProofRouterEvidence(**value)
    if kind == "pipeline":
        return ProofPipelineEvidence(**value)
    if kind == "protected":
        return ProofProtectedRegionEvidence(**value)
    if kind == "prefix":
        return ProofPrefixIdentityEvidence(**value)
    raise ValueError("unknown evidence")


def load_universal_proof_run_result(path: str | Path) -> UniversalProofRunResult:
    """Rehydrate TOKEN-10F run.json through every immutable contract."""
    source = Path(path)
    try:
        payload = json.loads(source.read_text(encoding="utf-8"))
    except (OSError, UnicodeError, json.JSONDecodeError) as exc:
        raise EvaluationConfigurationError("INVALID_RUN_RESULT") from exc
    if not isinstance(payload, dict):
        raise EvaluationConfigurationError("INVALID_RUN_RESULT")
    required = {
        "schema_version",
        "proof_id",
        "run_id",
        "run_mode",
        "started_at",
        "completed_at",
        "adapter_id",
        "model",
        "case_count",
        "completed_count",
        "failed_count",
        "cases",
        "environment",
        "artifact_manifest",
        "success",
        "raw_content_included",
    }
    if set(payload) != required:
        raise EvaluationConfigurationError("INVALID_RUN_RESULT_FIELDS")
    try:
        cases = []
        for item in payload["cases"]:
            if set(item) != {
                "schema_version",
                "case_id",
                "status",
                "router_status",
                "router_reason",
                "selected_configuration_id",
                "pipeline_status",
                "applied_layer_ids",
                "baseline_measurement",
                "optimized_measurement",
                "receipt_refs",
                "error_reason_code",
                "router_evidence",
                "pipeline_evidence",
                "protected_region_evidence",
                "prefix_identity_evidence",
                "raw_content_included",
            }:
                raise ValueError("invalid case fields")
            cases.append(
                UniversalProofCaseResult(
                    case_id=item["case_id"],
                    status=item["status"],
                    router_status=item["router_status"],
                    router_reason=item["router_reason"],
                    selected_configuration_id=item["selected_configuration_id"],
                    pipeline_status=item["pipeline_status"],
                    applied_layer_ids=tuple(item["applied_layer_ids"]),
                    baseline_measurement=_measurement(item["baseline_measurement"]),
                    optimized_measurement=_measurement(item["optimized_measurement"]),
                    receipt_refs=tuple(item["receipt_refs"]),
                    error_reason_code=item["error_reason_code"],
                    raw_content_included=item["raw_content_included"],
                    router_evidence=_evidence(item["router_evidence"], "router"),
                    pipeline_evidence=_evidence(item["pipeline_evidence"], "pipeline"),
                    protected_region_evidence=_evidence(
                        item["protected_region_evidence"], "protected"
                    ),
                    prefix_identity_evidence=_evidence(
                        item["prefix_identity_evidence"], "prefix"
                    ),
                )
            )
        environment = payload["environment"]
        if set(environment) != {
            "provider",
            "model",
            "adapter_available",
            "network_required",
            "raw_content_included",
        }:
            raise ValueError("invalid environment fields")
        manifest = payload["artifact_manifest"]
        if set(manifest) != {"files", "raw_content_included"}:
            raise ValueError("invalid manifest fields")
        refs = tuple(
            ProofArtifactRef(path=item["path"], sha256=item["sha256"])
            for item in manifest["files"]
        )
        result = UniversalProofRunResult(
            schema_version=payload["schema_version"],
            proof_id=payload["proof_id"],
            run_id=payload["run_id"],
            run_mode=payload["run_mode"],
            started_at=datetime.fromisoformat(payload["started_at"]),
            completed_at=datetime.fromisoformat(payload["completed_at"]),
            adapter_id=payload["adapter_id"],
            model=payload["model"],
            case_count=payload["case_count"],
            completed_count=payload["completed_count"],
            failed_count=payload["failed_count"],
            cases=tuple(cases),
            environment=UniversalProofEnvironmentSummary(**environment),
            artifact_manifest=UniversalProofArtifactManifest(
                files=refs,
                raw_content_included=manifest["raw_content_included"],
            ),
            success=payload["success"],
            raw_content_included=payload["raw_content_included"],
        )
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise EvaluationConfigurationError("INVALID_RUN_RESULT") from exc
    if result.schema_version != SCHEMA_VERSION:
        raise EvaluationConfigurationError("UNSUPPORTED_PROOF_SCHEMA_VERSION")
    return result


__all__ = [
    "CACHE_EVIDENCE_SCHEMA_VERSION",
    "CORPUS_SCHEMA_VERSION",
    "EVALUATION_SCHEMA_VERSION",
    "CacheAttribution",
    "CacheEvidenceRole",
    "CacheExpectation",
    "CacheExpectationMode",
    "CaseEvaluation",
    "CorpusCase",
    "EvaluationConfiguration",
    "EvaluationConfigurationError",
    "GateResult",
    "GateStatus",
    "MeasurementExpectation",
    "MeasurementRequirement",
    "PipelineExpectation",
    "PrefixExpectation",
    "ProofCorpus",
    "ProtectedRegionExpectation",
    "ProviderCacheEvidence",
    "RouterExpectation",
    "UniversalProofEvaluation",
    "expand_proof_config_with_corpus",
    "load_cache_evidence",
    "load_evaluation_config",
    "load_proof_corpus",
    "load_universal_proof_run_result",
]

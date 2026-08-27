# © Artur Czarnecki. All rights reserved.

"""Machine-readable Scenario Proof lifecycle metadata (SCENARIO-PLATFORM-3B)."""

from __future__ import annotations

import re
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path
from typing import Any

import yaml

_FRONTMATTER_PATTERN = re.compile(r"\A---\r?\n(.*?)\r?\n---(?:\r?\n|$)", re.DOTALL)


class ScenarioLifecycleParseStatus(StrEnum):
    PARSED = "PARSED"
    LEGACY = "LEGACY"
    UNKNOWN = "UNKNOWN"


class ScenarioLifecycle(StrEnum):
    DESIGN = "DESIGN"
    ACCEPTED_FOR_IMPLEMENTATION = "ACCEPTED_FOR_IMPLEMENTATION"
    IMPLEMENTATION_INITIALIZED = "IMPLEMENTATION_INITIALIZED"
    EXECUTABLE = "EXECUTABLE"
    VERIFIED = "VERIFIED"


class ScenarioImplementationStatus(StrEnum):
    NOT_INITIALIZED = "NOT_INITIALIZED"
    INITIALIZED = "INITIALIZED"


class ScenarioGateStatus(StrEnum):
    NOT_COMPLETED = "NOT_COMPLETED"
    COMPLETED = "COMPLETED"


class ScenarioGapDecisionStatus(StrEnum):
    NOT_COMPLETED = "NOT_COMPLETED"
    RESOLVED = "RESOLVED"


class ScenarioLifecycleError(ValueError):
    """Invalid or missing scenario lifecycle metadata."""


class ScenarioLifecycleGateError(ScenarioLifecycleError):
    """Scenario lifecycle preconditions for implementation init are not met."""


@dataclass(frozen=True, slots=True)
class ScenarioLifecycleMetadata:
    scenario_slug: str
    lifecycle: ScenarioLifecycle
    implementation_status: ScenarioImplementationStatus
    intergrax_fit: ScenarioGateStatus
    gap_decision: ScenarioGapDecisionStatus
    observability_contract: ScenarioGateStatus
    application_vs_proof_ownership: ScenarioGateStatus
    parse_status: ScenarioLifecycleParseStatus = ScenarioLifecycleParseStatus.PARSED

    @classmethod
    def initial_design(cls, *, slug: str) -> ScenarioLifecycleMetadata:
        return cls(
            scenario_slug=slug,
            lifecycle=ScenarioLifecycle.DESIGN,
            implementation_status=ScenarioImplementationStatus.NOT_INITIALIZED,
            intergrax_fit=ScenarioGateStatus.NOT_COMPLETED,
            gap_decision=ScenarioGapDecisionStatus.NOT_COMPLETED,
            observability_contract=ScenarioGateStatus.NOT_COMPLETED,
            application_vs_proof_ownership=ScenarioGateStatus.NOT_COMPLETED,
        )

    def with_implementation_initialized(self) -> ScenarioLifecycleMetadata:
        return ScenarioLifecycleMetadata(
            scenario_slug=self.scenario_slug,
            lifecycle=ScenarioLifecycle.IMPLEMENTATION_INITIALIZED,
            implementation_status=ScenarioImplementationStatus.INITIALIZED,
            intergrax_fit=self.intergrax_fit,
            gap_decision=self.gap_decision,
            observability_contract=self.observability_contract,
            application_vs_proof_ownership=self.application_vs_proof_ownership,
        )

    def to_frontmatter_dict(self) -> dict[str, str]:
        return {
            "scenario_slug": self.scenario_slug,
            "lifecycle": self.lifecycle.value,
            "implementation_status": self.implementation_status.value,
            "intergrax_fit": self.intergrax_fit.value,
            "gap_decision": self.gap_decision.value,
            "observability_contract": self.observability_contract.value,
            "application_vs_proof_ownership": self.application_vs_proof_ownership.value,
        }


def render_scenario_spec_frontmatter(metadata: ScenarioLifecycleMetadata) -> str:
    payload = yaml.safe_dump(
        metadata.to_frontmatter_dict(),
        sort_keys=False,
        default_flow_style=False,
    ).strip()
    return f"---\n{payload}\n---"


def split_scenario_spec(content: str) -> tuple[str | None, str]:
    match = _FRONTMATTER_PATTERN.match(content)
    if match is None:
        return None, content
    return match.group(1), content[match.end() :]


def _parse_enum(
    raw: object,
    enum_cls: type[StrEnum],
    *,
    field_name: str,
) -> StrEnum:
    if not isinstance(raw, str) or not raw.strip():
        raise ScenarioLifecycleError(f"{field_name} must be a non-empty string")
    normalized = raw.strip()
    try:
        return enum_cls(normalized)
    except ValueError as exc:
        allowed = ", ".join(item.value for item in enum_cls)
        raise ScenarioLifecycleError(
            f"{field_name} must be one of: {allowed}"
        ) from exc


def _metadata_from_mapping(
    payload: dict[str, Any],
    *,
    expected_slug: str | None = None,
) -> ScenarioLifecycleMetadata:
    if "scenario_slug" not in payload:
        raise ScenarioLifecycleError("scenario_slug is required in lifecycle frontmatter")
    scenario_slug = str(payload["scenario_slug"]).strip()
    if not scenario_slug:
        raise ScenarioLifecycleError("scenario_slug must be non-empty")
    if expected_slug is not None and scenario_slug != expected_slug:
        raise ScenarioLifecycleError(
            f"scenario_slug {scenario_slug!r} does not match package slug {expected_slug!r}"
        )

    return ScenarioLifecycleMetadata(
        scenario_slug=scenario_slug,
        lifecycle=_parse_enum(payload.get("lifecycle"), ScenarioLifecycle, field_name="lifecycle"),
        implementation_status=_parse_enum(
            payload.get("implementation_status"),
            ScenarioImplementationStatus,
            field_name="implementation_status",
        ),
        intergrax_fit=_parse_enum(
            payload.get("intergrax_fit"),
            ScenarioGateStatus,
            field_name="intergrax_fit",
        ),
        gap_decision=_parse_enum(
            payload.get("gap_decision"),
            ScenarioGapDecisionStatus,
            field_name="gap_decision",
        ),
        observability_contract=_parse_enum(
            payload.get("observability_contract"),
            ScenarioGateStatus,
            field_name="observability_contract",
        ),
        application_vs_proof_ownership=_parse_enum(
            payload.get("application_vs_proof_ownership"),
            ScenarioGateStatus,
            field_name="application_vs_proof_ownership",
        ),
    )


def parse_scenario_lifecycle_frontmatter(
    content: str,
    *,
    expected_slug: str | None = None,
) -> ScenarioLifecycleMetadata:
    frontmatter, _body = split_scenario_spec(content)
    if frontmatter is None:
        return ScenarioLifecycleMetadata(
            scenario_slug=expected_slug or "",
            lifecycle=ScenarioLifecycle.DESIGN,
            implementation_status=ScenarioImplementationStatus.NOT_INITIALIZED,
            intergrax_fit=ScenarioGateStatus.NOT_COMPLETED,
            gap_decision=ScenarioGapDecisionStatus.NOT_COMPLETED,
            observability_contract=ScenarioGateStatus.NOT_COMPLETED,
            application_vs_proof_ownership=ScenarioGateStatus.NOT_COMPLETED,
            parse_status=ScenarioLifecycleParseStatus.LEGACY,
        )

    try:
        payload = yaml.safe_load(frontmatter)
    except yaml.YAMLError as exc:
        raise ScenarioLifecycleError(f"invalid YAML frontmatter: {exc}") from exc

    if not isinstance(payload, dict):
        return ScenarioLifecycleMetadata(
            scenario_slug=expected_slug or "",
            lifecycle=ScenarioLifecycle.DESIGN,
            implementation_status=ScenarioImplementationStatus.NOT_INITIALIZED,
            intergrax_fit=ScenarioGateStatus.NOT_COMPLETED,
            gap_decision=ScenarioGapDecisionStatus.NOT_COMPLETED,
            observability_contract=ScenarioGateStatus.NOT_COMPLETED,
            application_vs_proof_ownership=ScenarioGateStatus.NOT_COMPLETED,
            parse_status=ScenarioLifecycleParseStatus.UNKNOWN,
        )

    return _metadata_from_mapping(payload, expected_slug=expected_slug)


def load_scenario_lifecycle_metadata(
    scenario_spec_path: Path,
    *,
    expected_slug: str | None = None,
) -> ScenarioLifecycleMetadata:
    content = scenario_spec_path.read_text(encoding="utf-8")
    return parse_scenario_lifecycle_frontmatter(
        content,
        expected_slug=expected_slug,
    )


def validate_implementation_init_preconditions(
    metadata: ScenarioLifecycleMetadata,
) -> None:
    if metadata.parse_status is not ScenarioLifecycleParseStatus.PARSED:
        raise ScenarioLifecycleGateError(
            "lifecycle metadata required in SCENARIO_SPEC.md frontmatter "
            f"(parse_status={metadata.parse_status.value})"
        )
    if metadata.lifecycle is not ScenarioLifecycle.ACCEPTED_FOR_IMPLEMENTATION:
        raise ScenarioLifecycleGateError(
            "scenario lifecycle must be ACCEPTED_FOR_IMPLEMENTATION before "
            f"implementation init (current={metadata.lifecycle.value})"
        )
    if metadata.implementation_status is not ScenarioImplementationStatus.NOT_INITIALIZED:
        raise ScenarioLifecycleGateError(
            "implementation already initialized "
            f"(implementation_status={metadata.implementation_status.value})"
        )
    if metadata.intergrax_fit is not ScenarioGateStatus.COMPLETED:
        raise ScenarioLifecycleGateError(
            "INTERGRAX FIT gate must be COMPLETED before implementation init"
        )
    if metadata.gap_decision is not ScenarioGapDecisionStatus.RESOLVED:
        raise ScenarioLifecycleGateError(
            "GAP DECISION gate must be RESOLVED before implementation init"
        )
    if metadata.observability_contract is not ScenarioGateStatus.COMPLETED:
        raise ScenarioLifecycleGateError(
            "Observability / Explainability / Diagnostics Contract gate must be "
            "COMPLETED before implementation init"
        )
    if metadata.application_vs_proof_ownership is not ScenarioGateStatus.COMPLETED:
        raise ScenarioLifecycleGateError(
            "APPLICATION vs PROOF ownership gate must be COMPLETED before "
            "implementation init"
        )


def replace_scenario_spec_frontmatter(
    content: str,
    metadata: ScenarioLifecycleMetadata,
) -> str:
    frontmatter = render_scenario_spec_frontmatter(metadata)
    _existing, body = split_scenario_spec(content)
    if body.startswith("\n"):
        body = body[1:]
    return f"{frontmatter}\n\n{body}" if body else f"{frontmatter}\n"


def write_scenario_spec_frontmatter(
    scenario_spec_path: Path,
    metadata: ScenarioLifecycleMetadata,
) -> None:
    content = scenario_spec_path.read_text(encoding="utf-8")
    scenario_spec_path.write_text(
        replace_scenario_spec_frontmatter(content, metadata),
        encoding="utf-8",
    )

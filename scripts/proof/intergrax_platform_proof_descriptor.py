# © Artur Czarnecki. All rights reserved.

"""Platform Proof package descriptor contracts (PP-SUITE-1).

Static ``proof.json`` descriptors are the canonical package-owned discovery source.
They normalize to ``ProofManifestEntry`` for the existing suite runner.
"""

from __future__ import annotations

import re
from enum import StrEnum
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from scripts.proof.intergrax_proof_contracts import (
    EnvRequirement,
    ProofArgvCommand,
    ProofProfile,
    ProofSafetyClass,
)

PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION = (
    "intergrax.platform_proof_descriptor.v3"
)
DOMAIN_ID_MAX_LENGTH = 64
MECHANISM_ID_MAX_LENGTH = 64
PROOF_DESCRIPTOR_FILENAME = "proof.json"
CANONICAL_PLATFORM_PROOF_ROOT = "platform_proofs"

PROOF_ID_PATTERN = re.compile(r"^[A-Z][A-Z0-9-]+$")
UNSAFE_DOMAIN_CHARS = re.compile(r"[/\\]|\.\.")
SHELL_METACHAR_RE = re.compile(r"[;&|`$<>]")

_SECRET_FIELD_NAMES = frozenset(
    {
        "secret",
        "password",
        "api_key",
        "apikey",
        "token_value",
        "credential",
        "private_key",
    }
)


class ProofLibraryClass(StrEnum):
    CONFORMANCE = "CONFORMANCE"
    SCENARIO = "SCENARIO"


class ExpectedArtifactKind(StrEnum):
    EVIDENCE_JSON = "EVIDENCE_JSON"
    REPORT_HTML = "REPORT_HTML"
    DOMAIN_RESULT_JSON = "DOMAIN_RESULT_JSON"
    OTHER = "OTHER"


class ExpectedProofArtifact(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    kind: ExpectedArtifactKind
    relative_path: str = Field(min_length=1)
    required: bool = True

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        normalized = value.strip().replace("\\", "/")
        if not normalized:
            raise ValueError("relative_path must be non-empty")
        if normalized.startswith("/") or _is_windows_absolute(normalized):
            raise ValueError("relative_path must be relative")
        if ".." in normalized.split("/"):
            raise ValueError("relative_path must not contain parent traversal")
        return normalized


class PlatformProofDescriptor(BaseModel):
    """Typed ``proof.json`` contract — ``intergrax.platform_proof_descriptor.v3``."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["intergrax.platform_proof_descriptor.v3"] = (
        PLATFORM_PROOF_DESCRIPTOR_SCHEMA_VERSION
    )
    library_class: ProofLibraryClass
    proof_id: str = Field(min_length=1)
    title: str = Field(min_length=1)
    domains_exercised: tuple[str, ...]
    proof_kind: str = Field(min_length=1)
    mechanisms_exercised: tuple[str, ...]
    package_version: str = Field(min_length=1)
    profiles: tuple[ProofProfile, ...]
    command: ProofArgvCommand
    timeout_seconds: int = Field(ge=1)
    safety_class: ProofSafetyClass
    public_evidence_eligible: bool = False
    platform_requirements: tuple[str, ...] = ()
    environment_requirements: tuple[EnvRequirement, ...] = ()
    tags: tuple[str, ...] = ()
    problem_category: str | None = None
    problem_summary: str | None = None
    failure_mode_summary: str | None = None
    description: str | None = None
    expected_artifacts: tuple[ExpectedProofArtifact, ...] = ()
    report_required: bool = False
    report_standard_version: str | None = None
    evidence_schema: str | None = None
    evidence_required: bool = False
    external_provider: str | None = None

    @field_validator("proof_id", "title", "proof_kind", "package_version")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("proof_id")
    @classmethod
    def _validate_proof_id(cls, value: str) -> str:
        if not PROOF_ID_PATTERN.fullmatch(value):
            raise ValueError(
                "proof_id must be uppercase alphanumeric with hyphens "
                "(e.g. SCENARIO-AI-INCIDENT-INVESTIGATION)"
            )
        return value

    @field_validator("domains_exercised", mode="before")
    @classmethod
    def _normalize_domains_exercised(cls, value: object) -> tuple[str, ...]:
        return _normalize_identifier_collection(
            value,
            field_name="domains_exercised",
            max_length=DOMAIN_ID_MAX_LENGTH,
            unsafe_pattern=UNSAFE_DOMAIN_CHARS,
            unsafe_message="domains_exercised contains unsafe path characters",
        )

    @field_validator("platform_requirements", mode="before")
    @classmethod
    def _normalize_platform_requirements(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("platform_requirements must be a sequence")
        return tuple(str(item).strip() for item in value)

    @field_validator("mechanisms_exercised", mode="before")
    @classmethod
    def _normalize_mechanisms_exercised(cls, value: object) -> tuple[str, ...]:
        return _normalize_identifier_collection(
            value,
            field_name="mechanisms_exercised",
            max_length=MECHANISM_ID_MAX_LENGTH,
            unsafe_pattern=None,
            unsafe_message="mechanisms_exercised contains unsafe path characters",
        )

    @field_validator("tags", mode="before")
    @classmethod
    def _normalize_tags(cls, value: object) -> tuple[str, ...]:
        if value is None:
            return ()
        if isinstance(value, str):
            raise ValueError("tags must be a sequence")
        normalized: list[str] = []
        for item in value:
            tag = str(item).strip()
            if not tag:
                raise ValueError("tags must not contain empty values")
            if len(tag) > 64:
                raise ValueError("tag exceeds maximum length")
            normalized.append(tag)
        return tuple(normalized)

    @field_validator(
        "problem_category",
        "problem_summary",
        "failure_mode_summary",
        mode="before",
    )
    @classmethod
    def _normalize_optional_scenario_text(cls, value: object) -> str | None:
        if value is None:
            return None
        if not isinstance(value, str):
            raise ValueError("must be a string or null")
        return value.strip()

    @model_validator(mode="after")
    def _validate_library_class_fields(self) -> PlatformProofDescriptor:
        scenario_fields = (
            ("problem_category", self.problem_category),
            ("problem_summary", self.problem_summary),
            ("failure_mode_summary", self.failure_mode_summary),
        )
        if self.library_class is ProofLibraryClass.SCENARIO:
            for field_name, value in scenario_fields:
                if value is None or not value:
                    raise ValueError(f"{field_name} is required for SCENARIO proofs")
        elif self.library_class is ProofLibraryClass.CONFORMANCE:
            for field_name, value in scenario_fields:
                if value is not None:
                    raise ValueError(
                        f"{field_name} is forbidden for CONFORMANCE proofs"
                    )
        return self

    @model_validator(mode="after")
    def _validate_profiles(self) -> PlatformProofDescriptor:
        if not self.profiles:
            raise ValueError("profiles must be non-empty")
        if len(self.profiles) != len(set(self.profiles)):
            raise ValueError("profiles must not contain duplicates")
        return self

    @model_validator(mode="after")
    def _reject_shell_tokens(self) -> PlatformProofDescriptor:
        for token in self.command.argv:
            if SHELL_METACHAR_RE.search(token):
                raise ValueError("command argv must not contain shell metacharacters")
            if "&&" in token or "||" in token:
                raise ValueError("command argv must not contain shell operators")
        return self

    @model_validator(mode="after")
    def _validate_expected_artifacts(self) -> PlatformProofDescriptor:
        evidence_json_count = 0
        required_evidence_count = 0
        required_report_count = 0
        seen_paths: set[str] = set()

        for artifact in self.expected_artifacts:
            if artifact.relative_path in seen_paths:
                raise ValueError(
                    f"duplicate expected artifact relative_path: {artifact.relative_path}"
                )
            seen_paths.add(artifact.relative_path)

            if artifact.kind == ExpectedArtifactKind.EVIDENCE_JSON:
                evidence_json_count += 1
                if artifact.required:
                    required_evidence_count += 1
            if (
                artifact.kind == ExpectedArtifactKind.REPORT_HTML
                and artifact.required
            ):
                required_report_count += 1

        if evidence_json_count > 1:
            raise ValueError("at most one EVIDENCE_JSON artifact may be declared")

        if self.evidence_required:
            if required_evidence_count != 1:
                raise ValueError(
                    "evidence_required=true requires exactly one required EVIDENCE_JSON"
                )
        elif required_evidence_count > 0:
            raise ValueError(
                "evidence_required=false contradicts required EVIDENCE_JSON declaration"
            )

        if self.report_required:
            if required_report_count != 1:
                raise ValueError(
                    "report_required=true requires exactly one required REPORT_HTML"
                )
        elif required_report_count > 0:
            raise ValueError(
                "report_required=false contradicts required REPORT_HTML declaration"
            )

        return self


def _is_windows_absolute(path: str) -> bool:
    return len(path) >= 2 and path[1] == ":"


def _normalize_identifier_collection(
    value: object,
    *,
    field_name: str,
    max_length: int,
    unsafe_pattern: re.Pattern[str] | None,
    unsafe_message: str,
) -> tuple[str, ...]:
    if value is None:
        raise ValueError(f"{field_name} must be non-empty")
    if isinstance(value, str):
        raise ValueError(f"{field_name} must be a sequence")
    if not isinstance(value, (list, tuple)):
        raise ValueError(f"{field_name} must be a sequence")
    normalized: list[str] = []
    seen: set[str] = set()
    for item in value:
        if not isinstance(item, str):
            raise ValueError(f"{field_name} must contain only strings")
        identifier = item.strip()
        if not identifier:
            raise ValueError(f"{field_name} must not contain empty values")
        if len(identifier) > max_length:
            raise ValueError(f"{identifier} exceeds maximum length")
        if unsafe_pattern is not None and unsafe_pattern.search(identifier):
            raise ValueError(unsafe_message)
        if identifier in seen:
            raise ValueError(f"{field_name} must not contain duplicates")
        seen.add(identifier)
        normalized.append(identifier)
    if not normalized:
        raise ValueError(f"{field_name} must be non-empty")
    return tuple(normalized)

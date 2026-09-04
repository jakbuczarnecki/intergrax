"""Immutable proof data package descriptor contracts."""

from __future__ import annotations

import json
import re
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.proof_data.checksum import normalize_sha256_hex
from intergrax.proof_data.errors import DataPackageDescriptorError
from intergrax.proof_data.paths import normalize_relative_path

PROOF_DATA_PACKAGE_SCHEMA_VERSION = "intergrax.proof_data_package.v1"
PACKAGE_ID_PATTERN = re.compile(r"^[a-z0-9][a-z0-9._-]{0,127}$")


class PublicationStatus(StrEnum):
    INTERNAL_BUILD = "INTERNAL_BUILD"
    REDISTRIBUTION_REVIEW_REQUIRED = "REDISTRIBUTION_REVIEW_REQUIRED"
    PUBLICATION_APPROVED = "PUBLICATION_APPROVED"


class DataPackageFileDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    relative_path: str = Field(min_length=1)
    size_bytes: int = Field(ge=0)
    sha256: str = Field(min_length=64, max_length=64)
    role: str = Field(min_length=1)

    @field_validator("relative_path")
    @classmethod
    def _validate_relative_path(cls, value: str) -> str:
        return normalize_relative_path(value)

    @field_validator("sha256")
    @classmethod
    def _validate_sha256(cls, value: str) -> str:
        return normalize_sha256_hex(value)

    @field_validator("role")
    @classmethod
    def _strip_role(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("role must be non-empty")
        return normalized


class ProofDataPackageDescriptor(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["intergrax.proof_data_package.v1"] = PROOF_DATA_PACKAGE_SCHEMA_VERSION
    package_id: str = Field(min_length=1)
    package_version: str = Field(min_length=1)
    description: str = Field(min_length=1)
    files: tuple[DataPackageFileDescriptor, ...]
    total_size_bytes: int | None = Field(default=None, ge=0)
    provenance_ref: str | None = None
    redistribution_status: PublicationStatus
    metadata: tuple[str, ...] = ()

    @field_validator("package_id", "package_version", "description")
    @classmethod
    def _strip_required(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("must be non-empty")
        return normalized

    @field_validator("package_id")
    @classmethod
    def _validate_package_id(cls, value: str) -> str:
        if not PACKAGE_ID_PATTERN.match(value):
            raise ValueError("package_id must match [a-z0-9][a-z0-9._-]{0,127}")
        return value

    @field_validator("provenance_ref")
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None

    @model_validator(mode="after")
    def _validate_files(self) -> ProofDataPackageDescriptor:
        if not self.files:
            raise ValueError("files must contain at least one entry")
        seen_paths: set[str] = set()
        for file_descriptor in self.files:
            if file_descriptor.relative_path in seen_paths:
                raise ValueError(
                    f"duplicate relative_path in descriptor: {file_descriptor.relative_path}"
                )
            seen_paths.add(file_descriptor.relative_path)
        if self.total_size_bytes is not None:
            computed = sum(file_descriptor.size_bytes for file_descriptor in self.files)
            if computed != self.total_size_bytes:
                raise ValueError("total_size_bytes must equal sum of file size_bytes values")
        return self


def load_proof_data_package_descriptor(path: Path) -> ProofDataPackageDescriptor:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise DataPackageDescriptorError(f"failed to read package descriptor: {path}") from exc
    try:
        return ProofDataPackageDescriptor.model_validate(payload)
    except Exception as exc:
        raise DataPackageDescriptorError(f"invalid package descriptor: {path}: {exc}") from exc


def dump_proof_data_package_descriptor(
    descriptor: ProofDataPackageDescriptor,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(descriptor.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

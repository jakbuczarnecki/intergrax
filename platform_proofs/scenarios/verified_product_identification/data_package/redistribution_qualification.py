"""Typed VPI Data Pack redistribution qualification contract."""

from __future__ import annotations

import json
from enum import StrEnum
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.proof_data.descriptor import PublicationStatus

from platform_proofs.scenarios.verified_product_identification.data_package.errors import (
    VpiDataPackageDescriptorBuildError,
    VpiRedistributionQualificationError,
)

VPI_REDISTRIBUTION_REVIEW_SCHEMA_VERSION = "verified_product_identification.redistribution_review.v1"
REDISTRIBUTION_REVIEW_FILENAME = "redistribution-review.json"

_COMBINED_PACK_COMPONENTS = (
    "VPI_RELATIONAL_PARQUET",
    "VPI_EMBEDDING_PARQUET",
    "PACKAGE_METADATA",
)


class RedistributionAssetStatus(StrEnum):
    PUBLIC_REDISTRIBUTION_APPROVED = "PUBLIC_REDISTRIBUTION_APPROVED"
    PUBLIC_REDISTRIBUTION_BLOCKED = "PUBLIC_REDISTRIBUTION_BLOCKED"
    REDISTRIBUTION_REVIEW_REQUIRED = "REDISTRIBUTION_REVIEW_REQUIRED"
    INTERNAL_USE_ONLY = "INTERNAL_USE_ONLY"


class RedistributionAssetType(StrEnum):
    RAW_WDC_CORPUS = "RAW_WDC_CORPUS"
    VPI_RELATIONAL_PARQUET = "VPI_RELATIONAL_PARQUET"
    VPI_EMBEDDING_PARQUET = "VPI_EMBEDDING_PARQUET"
    VPI_COMBINED_DATA_PACK = "VPI_COMBINED_DATA_PACK"
    PACKAGE_METADATA = "PACKAGE_METADATA"


class EmbeddingGateStatus(StrEnum):
    PASSED = "PASSED"
    FAILED = "FAILED"
    UNCLEAR = "UNCLEAR"


class SourceEvidenceRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_id: str = Field(min_length=1)
    authority: str = Field(min_length=1)
    title: str = Field(min_length=1)
    url: str = Field(min_length=1)
    accessed_at: str = Field(min_length=1)
    evidence_scope: str = Field(min_length=1)
    short_finding: str = Field(min_length=1)
    terms_last_updated: str | None = None

    @field_validator(
        "source_id",
        "authority",
        "title",
        "url",
        "accessed_at",
        "evidence_scope",
        "short_finding",
        "terms_last_updated",
    )
    @classmethod
    def _strip_optional(cls, value: str | None) -> str | None:
        if value is None:
            return None
        normalized = value.strip()
        return normalized or None


class AttributionRequirement(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    source_name: str | None = None
    source_url: str | None = None
    source_dataset: str | None = None
    source_terms_url: str | None = None
    source_attribution_text: str | None = None
    model_name: str | None = None
    model_url: str | None = None
    model_license: str | None = None
    model_license_url: str | None = None
    generation_notice: str | None = None
    legally_required: bool = False


class AssetRedistributionDecision(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    asset_type: RedistributionAssetType
    status: RedistributionAssetStatus
    evidence_source_ids: tuple[str, ...]
    license_terms_reference: str | None = None
    blocking_uncertainty: str | None = None
    recommended_publication_action: str = Field(min_length=1)
    attribution: AttributionRequirement | None = None
    model_weights_distributed: bool = False

    @field_validator("recommended_publication_action")
    @classmethod
    def _strip_action(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("recommended_publication_action must be non-empty")
        return normalized


class EmbeddingRedistributionGates(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    model_license_gate: EmbeddingGateStatus
    input_data_rights_gate: EmbeddingGateStatus
    output_restriction_gate: EmbeddingGateStatus


class RedistributionQualificationRecord(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: Literal["verified_product_identification.redistribution_review.v1"] = (
        VPI_REDISTRIBUTION_REVIEW_SCHEMA_VERSION
    )
    review_id: str = Field(min_length=1)
    reviewed_at: str = Field(min_length=1)
    reviewer: str = Field(min_length=1)
    processed_dataset_sha256: str = Field(min_length=64, max_length=64)
    source_dataset_identifier: str = Field(min_length=1)
    third_party_content_rights_resolved: bool
    evidence_records: tuple[SourceEvidenceRecord, ...]
    asset_decisions: tuple[AssetRedistributionDecision, ...]
    embedding_gates: EmbeddingRedistributionGates
    embeddings_only_fallback_viable: Literal["YES", "NO", "UNRESOLVED"]
    legal_escalation_required: bool
    legal_escalation_question: str | None = None
    recommended_attribution: AttributionRequirement | None = None

    @model_validator(mode="after")
    def _validate_asset_decisions(self) -> RedistributionQualificationRecord:
        if not self.asset_decisions:
            raise ValueError("asset_decisions must contain at least one entry")
        seen: set[RedistributionAssetType] = set()
        for decision in self.asset_decisions:
            if decision.asset_type in seen:
                raise ValueError(f"duplicate asset decision: {decision.asset_type.value}")
            seen.add(decision.asset_type)
        evidence_ids = {record.source_id for record in self.evidence_records}
        for decision in self.asset_decisions:
            for source_id in decision.evidence_source_ids:
                if source_id not in evidence_ids:
                    raise ValueError(
                        f"missing evidence record for asset decision {decision.asset_type.value}: {source_id}"
                    )
        return self

    def decision_for(self, asset_type: RedistributionAssetType) -> AssetRedistributionDecision:
        for decision in self.asset_decisions:
            if decision.asset_type is asset_type:
                return decision
        raise VpiRedistributionQualificationError(
            f"unknown asset type in qualification record: {asset_type.value}"
        )

    def combined_component_status(self) -> RedistributionAssetStatus:
        statuses = [
            self.decision_for(RedistributionAssetType(asset_name)).status
            for asset_name in _COMBINED_PACK_COMPONENTS
        ]
        if any(status is RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_BLOCKED for status in statuses):
            return RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_BLOCKED
        if any(status is RedistributionAssetStatus.REDISTRIBUTION_REVIEW_REQUIRED for status in statuses):
            return RedistributionAssetStatus.REDISTRIBUTION_REVIEW_REQUIRED
        if all(status is RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED for status in statuses):
            return RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED
        return RedistributionAssetStatus.REDISTRIBUTION_REVIEW_REQUIRED

    def resolved_combined_status(self) -> RedistributionAssetStatus:
        explicit = self.decision_for(RedistributionAssetType.VPI_COMBINED_DATA_PACK)
        derived = self.combined_component_status()
        if explicit.status is RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED:
            if derived is not RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED:
                return RedistributionAssetStatus.REDISTRIBUTION_REVIEW_REQUIRED
            return explicit.status
        if explicit.status is RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_BLOCKED:
            return explicit.status
        return derived


def default_redistribution_review_path() -> Path:
    return Path(__file__).resolve().parent / "v1" / REDISTRIBUTION_REVIEW_FILENAME


def load_redistribution_qualification(path: Path) -> RedistributionQualificationRecord:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise VpiRedistributionQualificationError(
            f"failed to read redistribution qualification: {path}"
        ) from exc
    try:
        return RedistributionQualificationRecord.model_validate(payload)
    except Exception as exc:
        raise VpiRedistributionQualificationError(
            f"invalid redistribution qualification: {path}: {exc}"
        ) from exc


def dump_redistribution_qualification(
    qualification: RedistributionQualificationRecord,
    path: Path,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(qualification.model_dump(mode="json"), indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def load_default_redistribution_qualification() -> RedistributionQualificationRecord:
    return load_redistribution_qualification(default_redistribution_review_path())


def can_publish(
    asset_type: RedistributionAssetType,
    qualification: RedistributionQualificationRecord,
) -> bool:
    if asset_type is RedistributionAssetType.VPI_COMBINED_DATA_PACK:
        status = qualification.resolved_combined_status()
    else:
        status = qualification.decision_for(asset_type).status
    return status is RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED


def asset_status_to_publication_status(
    status: RedistributionAssetStatus,
) -> PublicationStatus:
    if status is RedistributionAssetStatus.PUBLIC_REDISTRIBUTION_APPROVED:
        return PublicationStatus.PUBLICATION_APPROVED
    if status is RedistributionAssetStatus.INTERNAL_USE_ONLY:
        return PublicationStatus.INTERNAL_BUILD
    return PublicationStatus.REDISTRIBUTION_REVIEW_REQUIRED


def resolve_publication_status(
    qualification: RedistributionQualificationRecord,
) -> PublicationStatus:
    return asset_status_to_publication_status(qualification.resolved_combined_status())


def assert_public_redistribution_permitted(
    qualification: RedistributionQualificationRecord,
) -> None:
    if not can_publish(RedistributionAssetType.VPI_COMBINED_DATA_PACK, qualification):
        combined_status = qualification.resolved_combined_status()
        raise VpiDataPackageDescriptorBuildError(
            "public redistribution blocked by qualification record: "
            f"combined_status={combined_status.value}"
        )


def assert_publication_permitted_with_qualification(
    status: PublicationStatus,
    qualification: RedistributionQualificationRecord,
) -> None:
    if status is not PublicationStatus.PUBLICATION_APPROVED:
        raise VpiDataPackageDescriptorBuildError(
            f"public publication blocked for redistribution status {status.value}"
        )
    assert_public_redistribution_permitted(qualification)

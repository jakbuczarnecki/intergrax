# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Digest-pinned full AgentContract production authority (EBH-2E-AR1-A-R1)."""

from __future__ import annotations

from typing import Final, Protocol

from pydantic import BaseModel, ConfigDict, Field, field_validator, model_validator

from intergrax.agent_distribution._digest import normalize_package_digest
from intergrax.agent_distribution.agent_capability_metadata import AgentCapabilityDescriptor
from intergrax.agent_distribution.agent_project_metadata import (
    AgentProjectMetadata,
    project_agent_capability_descriptors,
)
from intergrax.agent_distribution.contract_metadata_parity import (
    validate_agent_contract_metadata_parity,
)
from intergrax.contracts.agent_contract_meta import AgentContract
from intergrax.runtime.attestation.canonical_json import stable_payload_hash

_NON_EMPTY = Field(min_length=1)

SCHEMA_PACKAGE_AGENT_CONTRACT_AUTHORITY_V1: Final = "package_agent_contract_authority.v1"


def _strip_required(value: str) -> str:
    normalized = value.strip()
    if not normalized:
        raise ValueError("must be non-empty")
    return normalized


def contract_metadata_content_digest(contract: AgentContract) -> str:
    """Deterministic digest of full AgentContract semantic payload."""
    return stable_payload_hash(contract.model_dump(mode="json"))


class PackageAgentContractAuthorityError(ValueError):
    """Package contract authority validation or persistence failed."""


class PackageAgentContractAuthorityRecord(BaseModel):
    """Immutable, digest-pinned full AgentContract authority for one package contract."""

    model_config = ConfigDict(extra="forbid", frozen=True)

    schema_version: str = SCHEMA_PACKAGE_AGENT_CONTRACT_AUTHORITY_V1
    package_digest: str = _NON_EMPTY
    contract_id: str = _NON_EMPTY
    contract_version: str = _NON_EMPTY
    metadata_digest: str = _NON_EMPTY
    distribution_package_id: str = _NON_EMPTY
    artifact_store_ref: str = _NON_EMPTY
    agent_project_metadata_ref: str = _NON_EMPTY
    contract: AgentContract

    @field_validator(
        "package_digest",
        "contract_id",
        "contract_version",
        "metadata_digest",
        "distribution_package_id",
        "artifact_store_ref",
        "agent_project_metadata_ref",
    )
    @classmethod
    def _strip_fields(cls, value: str) -> str:
        return _strip_required(value)

    @field_validator("package_digest")
    @classmethod
    def _validate_package_digest(cls, value: str) -> str:
        return normalize_package_digest(value)

    @model_validator(mode="after")
    def _validate_identity_and_digest(self) -> PackageAgentContractAuthorityRecord:
        if self.contract.id != self.contract_id:
            raise ValueError("contract.id must match contract_id")
        if self.contract.version != self.contract_version:
            raise ValueError("contract.version must match contract_version")
        computed = contract_metadata_content_digest(self.contract)
        if self.metadata_digest != computed:
            raise ValueError("metadata_digest does not match contract content")
        return self

    @classmethod
    def from_validated_contract(
        cls,
        *,
        package_digest: str,
        distribution_package_id: str,
        artifact_store_ref: str,
        agent_project_metadata_ref: str,
        contract: AgentContract,
    ) -> PackageAgentContractAuthorityRecord:
        digest = contract_metadata_content_digest(contract)
        return cls(
            package_digest=package_digest,
            contract_id=contract.id,
            contract_version=contract.version,
            metadata_digest=digest,
            distribution_package_id=distribution_package_id,
            artifact_store_ref=artifact_store_ref,
            agent_project_metadata_ref=agent_project_metadata_ref,
            contract=contract,
        )


class AgentPackageContractAuthorityStore(Protocol):
    """Read/write port for package-level AgentContract authority records."""

    def get_package_contract_authority(
        self,
        package_digest: str,
        contract_id: str,
    ) -> PackageAgentContractAuthorityRecord | None:
        """Load one immutable contract authority by package digest and contract id."""

    def persist_package_contract_authority(
        self,
        record: PackageAgentContractAuthorityRecord,
    ) -> PackageAgentContractAuthorityRecord:
        """Persist one contract authority record; reject digest conflicts."""


class AgentPackageContractAuthorityService:
    """Validate and persist package AgentContract authority (distribution lifecycle)."""

    def __init__(self, store: AgentPackageContractAuthorityStore) -> None:
        self._store = store

    def persist_authority_record(
        self,
        record: PackageAgentContractAuthorityRecord,
    ) -> PackageAgentContractAuthorityRecord:
        existing = self._store.get_package_contract_authority(
            record.package_digest,
            record.contract_id,
        )
        if existing is not None:
            if existing.metadata_digest != record.metadata_digest:
                raise PackageAgentContractAuthorityError(
                    "conflicting package contract authority for "
                    f"{record.package_digest!r} / {record.contract_id!r}"
                )
            return existing
        return self._store.persist_package_contract_authority(record)

    def resolve_contract(
        self,
        *,
        package_digest: str,
        contract_id: str,
    ) -> AgentContract:
        record = self._store.get_package_contract_authority(package_digest, contract_id)
        if record is None:
            raise PackageAgentContractAuthorityError(
                f"missing package contract authority for digest={package_digest!r} "
                f"contract_id={contract_id!r}"
            )
        if record.package_digest != normalize_package_digest(package_digest):
            raise PackageAgentContractAuthorityError("package_digest mismatch on read")
        if record.contract_id != contract_id:
            raise PackageAgentContractAuthorityError("contract_id mismatch on read")
        expected = contract_metadata_content_digest(record.contract)
        if record.metadata_digest != expected:
            raise PackageAgentContractAuthorityError(
                "package contract authority metadata_digest integrity failure"
            )
        return record.contract

    @staticmethod
    def validate_against_project_metadata(
        *,
        metadata: AgentProjectMetadata,
        records: tuple[PackageAgentContractAuthorityRecord, ...],
    ) -> None:
        descriptors = project_agent_capability_descriptors(metadata)
        descriptor_by_id = {item.contract_id: item for item in descriptors}
        for record in records:
            descriptor = descriptor_by_id.get(record.contract_id)
            if descriptor is None:
                raise PackageAgentContractAuthorityError(
                    f"contract {record.contract_id!r} missing from package metadata declarations"
                )
            validate_agent_contract_metadata_parity(
                descriptor=descriptor,
                contract=record.contract,
            )
        if len(records) != len(descriptors):
            declared = {item.contract_id for item in descriptors}
            provided = {item.contract_id for item in records}
            missing = sorted(declared - provided)
            if missing:
                raise PackageAgentContractAuthorityError(
                    f"incomplete package contract authority; missing: {missing!r}"
                )


def descriptor_for_contract_id(
    metadata: AgentProjectMetadata,
    contract_id: str,
) -> AgentCapabilityDescriptor:
    for descriptor in project_agent_capability_descriptors(metadata):
        if descriptor.contract_id == contract_id:
            return descriptor
    raise PackageAgentContractAuthorityError(
        f"contract_id {contract_id!r} not declared in package metadata"
    )


__all__ = [
    "AgentPackageContractAuthorityService",
    "AgentPackageContractAuthorityStore",
    "PackageAgentContractAuthorityError",
    "PackageAgentContractAuthorityRecord",
    "contract_metadata_content_digest",
    "descriptor_for_contract_id",
]

"""Data pack compatibility validation before storage ingest."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.checksums import (
    verify_sha256sums,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.shard_integrity import (
    validate_embedding_shard_source_identity,
    validate_relational_shard_source_identity,
    validate_shard_descriptor_file,
    validate_shard_pair_identity,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.content_identity import (
    content_identity_from_manifest,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.errors import (
    VpiDataPackCompatibilityError,
    VpiDataPackIntegrityError,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.identity import (
    DATA_PACK_VERSION,
    EMBEDDING_SCHEMA_VERSION,
    RELATIONAL_SCHEMA_VERSION,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.manifest import (
    DataPackManifest,
)
from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.contracts.shard_index import (
    ShardIndex,
    read_shard_index_file,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    ValidationCheck,
    ValidationReport,
    ValidationStatus,
)


@dataclass(frozen=True, slots=True)
class DataPackCompatibilityExpectations:
    data_pack_version: str
    relational_schema_version: str
    embedding_schema_version: str
    derivation_version: str
    semantic_text_version: str
    embedding_provider: str
    embedding_model: str
    embedding_model_revision: str
    embedding_dimension: int
    source_dataset_sha256: str | None = None


def _format_major(version: str) -> str:
    parts = version.split("/")
    if len(parts) != 2:
        raise VpiDataPackCompatibilityError(f"invalid data pack version format: {version}")
    minor_parts = parts[1].split(".")
    if not minor_parts:
        raise VpiDataPackCompatibilityError(f"invalid data pack version format: {version}")
    return minor_parts[0]


def _check(name: str, passed: bool, detail: str) -> ValidationCheck:
    return ValidationCheck(
        name=name,
        status=ValidationStatus.PASS if passed else ValidationStatus.FAIL,
        detail=detail,
    )


def validate_shard_index_contract(shard_index: ShardIndex) -> ValidationReport:
    checks: list[ValidationCheck] = []
    relational_ordinals = [descriptor.ordinal for descriptor in shard_index.relational_shards]
    embedding_ordinals = [descriptor.ordinal for descriptor in shard_index.embedding_shards]
    expected_ordinals = list(range(1, shard_index.shard_count + 1))
    checks.append(
        _check(
            "relational_shard_ordinals",
            relational_ordinals == expected_ordinals,
            f"ordinals={relational_ordinals}",
        )
    )
    checks.append(
        _check(
            "embedding_shard_ordinals",
            embedding_ordinals == expected_ordinals,
            f"ordinals={embedding_ordinals}",
        )
    )
    checks.append(
        _check(
            "duplicate_relational_ordinals",
            len(set(relational_ordinals)) == len(relational_ordinals),
            "relational shard ordinals must be unique",
        )
    )
    checks.append(
        _check(
            "duplicate_embedding_ordinals",
            len(set(embedding_ordinals)) == len(embedding_ordinals),
            "embedding shard ordinals must be unique",
        )
    )
    for relational, embedding in zip(
        shard_index.relational_shards,
        shard_index.embedding_shards,
        strict=True,
    ):
        checks.extend(validate_shard_pair_identity(relational, embedding))
    return ValidationReport.from_checks(tuple(checks))


def validate_data_pack_compatibility(
    manifest: DataPackManifest,
    *,
    expectations: DataPackCompatibilityExpectations,
    pack_root: Path | None = None,
) -> ValidationReport:
    checks: list[ValidationCheck] = []
    checks.append(
        _check(
            "data_pack_major_version",
            _format_major(manifest.data_pack_version) == _format_major(expectations.data_pack_version),
            f"manifest={manifest.data_pack_version} expected={expectations.data_pack_version}",
        )
    )
    checks.append(
        _check(
            "data_pack_exact_version",
            manifest.data_pack_version == expectations.data_pack_version,
            f"manifest={manifest.data_pack_version}",
        )
    )
    checks.append(
        _check(
            "relational_schema_version",
            manifest.relational_schema_version == expectations.relational_schema_version,
            manifest.relational_schema_version,
        )
    )
    checks.append(
        _check(
            "embedding_schema_version",
            manifest.embedding_schema_version == expectations.embedding_schema_version,
            manifest.embedding_schema_version,
        )
    )
    checks.append(
        _check(
            "derivation_version",
            manifest.derivation_version == expectations.derivation_version,
            manifest.derivation_version,
        )
    )
    checks.append(
        _check(
            "semantic_text_version",
            manifest.semantic_text_version == expectations.semantic_text_version,
            manifest.semantic_text_version,
        )
    )
    checks.append(
        _check(
            "embedding_provider",
            manifest.embedding_identity.provider == expectations.embedding_provider,
            manifest.embedding_identity.provider,
        )
    )
    checks.append(
        _check(
            "embedding_model",
            manifest.embedding_identity.model == expectations.embedding_model,
            manifest.embedding_identity.model,
        )
    )
    checks.append(
        _check(
            "embedding_model_revision",
            manifest.embedding_identity.model_revision == expectations.embedding_model_revision,
            str(manifest.embedding_identity.model_revision),
        )
    )
    checks.append(
        _check(
            "embedding_dimension",
            manifest.embedding_identity.dimension == expectations.embedding_dimension,
            str(manifest.embedding_identity.dimension),
        )
    )
    if expectations.source_dataset_sha256 is not None:
        checks.append(
            _check(
                "source_dataset_sha256",
                manifest.source_dataset.dataset_sha256 == expectations.source_dataset_sha256,
                manifest.source_dataset.dataset_sha256,
            )
        )
    expected_content_identity = content_identity_from_manifest(manifest)
    checks.append(
        _check(
            "content_identity",
            manifest.content_identity == expected_content_identity,
            manifest.content_identity,
        )
    )
    if pack_root is not None:
        shard_index_path = pack_root / manifest.shards_index_path
        shard_index = read_shard_index_file(shard_index_path)
        shard_report = validate_shard_index_contract(shard_index)
        checks.extend(shard_report.checks)
        checksums_path = pack_root / manifest.checksums_path
        try:
            verify_sha256sums(checksums_path, pack_root)
            checks.append(_check("checksum_validation", True, "SHA256SUMS verified"))
        except VpiDataPackIntegrityError as exc:
            checks.append(_check("checksum_validation", False, str(exc)))
        for descriptor in shard_index.relational_shards:
            checks.extend(
                validate_shard_descriptor_file(
                    pack_root,
                    descriptor,
                    check_name_prefix="relational_shard",
                )
            )
        for descriptor in shard_index.embedding_shards:
            checks.extend(
                validate_shard_descriptor_file(
                    pack_root,
                    descriptor,
                    check_name_prefix="embedding_shard",
                )
            )
        for descriptor in shard_index.relational_shards:
            checks.extend(
                validate_relational_shard_source_identity(pack_root, descriptor)
            )
        for descriptor in shard_index.embedding_shards:
            checks.extend(
                validate_embedding_shard_source_identity(
                    pack_root,
                    descriptor,
                    expected_dimension=manifest.embedding_identity.dimension,
                )
            )
    return ValidationReport.from_checks(tuple(checks))


def assert_data_pack_compatible(
    manifest: DataPackManifest,
    *,
    expectations: DataPackCompatibilityExpectations,
    pack_root: Path | None = None,
) -> None:
    report = validate_data_pack_compatibility(
        manifest,
        expectations=expectations,
        pack_root=pack_root,
    )
    if report.status is not ValidationStatus.PASS:
        failed = [check.name for check in report.checks if check.status is ValidationStatus.FAIL]
        raise VpiDataPackCompatibilityError(
            f"data pack compatibility failed: {', '.join(failed)}"
        )


def default_v1_expectations(
    *,
    derivation_version: str,
    semantic_text_version: str,
    embedding_provider: str,
    embedding_model: str,
    embedding_model_revision: str,
    embedding_dimension: int,
    source_dataset_sha256: str | None = None,
) -> DataPackCompatibilityExpectations:
    return DataPackCompatibilityExpectations(
        data_pack_version=DATA_PACK_VERSION,
        relational_schema_version=RELATIONAL_SCHEMA_VERSION,
        embedding_schema_version=EMBEDDING_SCHEMA_VERSION,
        derivation_version=derivation_version,
        semantic_text_version=semantic_text_version,
        embedding_provider=embedding_provider,
        embedding_model=embedding_model,
        embedding_model_revision=embedding_model_revision,
        embedding_dimension=embedding_dimension,
        source_dataset_sha256=source_dataset_sha256,
    )

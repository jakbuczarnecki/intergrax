# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""SQLite relational-store configuration realization (INT-CONFIG-REAL-X-P2)."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from intergrax.integrations.contracts.base import IntegrationCategory
from intergrax.integrations.contracts.existing_capability_configuration import (
    ConfiguredCapabilityBinding,
    ExistingCapabilityConfigurationRealizationError,
    ExistingCapabilityConfigurationRealizationFailureReason,
    ExistingCapabilityConfigurationRealizationRequest,
    ExistingCapabilityIntegrationTarget,
)
from intergrax.integrations.providers.relational_store.sqlite.config import (
    SQLiteIntegrationConfig,
)
from intergrax.integrations.providers.relational_store.sqlite.integration import (
    SQLITE_RELATIONAL_STORE_PROVIDER_ID,
)

SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE: str = (
    "sqlite.relational_store.configuration"
)
SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION: str = "1"
SQLITE_RELATIONAL_STORE_CONFIGURATION_STRATEGY_ID: str = (
    "sqlite.relational_store.configuration.v1"
)


def _normalized_path_lexical(path: Path | None) -> str | None:
    if path is None:
        return None
    return path.as_posix()


def compute_sqlite_relational_store_configuration_fingerprint(
    *,
    data_dir: Path,
    relational_db: Path | None,
) -> str:
    material = {
        "configuration_type": SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE,
        "configuration_version": SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION,
        "data_dir": data_dir.as_posix(),
        "relational_db": _normalized_path_lexical(relational_db),
    }
    canonical = json.dumps(material, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


@dataclass(frozen=True)
class SQLiteRelationalStoreConfigurationPayload:
    """Provider-owned relational-store configuration identity (narrow scope)."""

    data_dir: Path
    relational_db: Path | None = None

    @property
    def configuration_type(self) -> str:
        return SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE

    @property
    def configuration_version(self) -> str:
        return SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION

    @property
    def configuration_fingerprint(self) -> str:
        return compute_sqlite_relational_store_configuration_fingerprint(
            data_dir=self.data_dir,
            relational_db=self.relational_db,
        )


class SQLiteRelationalStoreConfigurationRealizationStrategy:
    """Reference provider strategy — validates configured identity only."""

    @property
    def strategy_id(self) -> str:
        return SQLITE_RELATIONAL_STORE_CONFIGURATION_STRATEGY_ID

    def can_realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> bool:
        if request.integration_category is not IntegrationCategory.RELATIONAL_STORE:
            return False
        if request.provider_id != SQLITE_RELATIONAL_STORE_PROVIDER_ID:
            return False
        if (
            existing_target.integration_category
            is not IntegrationCategory.RELATIONAL_STORE
        ):
            return False
        if existing_target.provider_id != SQLITE_RELATIONAL_STORE_PROVIDER_ID:
            return False
        configuration = request.configuration
        if not isinstance(configuration, SQLiteRelationalStoreConfigurationPayload):
            return False
        if (
            configuration.configuration_type
            != SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE
        ):
            return False
        if (
            configuration.configuration_version
            != SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION
        ):
            return False
        return True

    def realize(
        self,
        request: ExistingCapabilityConfigurationRealizationRequest,
        existing_target: ExistingCapabilityIntegrationTarget,
    ) -> ConfiguredCapabilityBinding:
        _fail_closed_realize_continuity(request, existing_target)
        configuration = request.configuration
        if not isinstance(configuration, SQLiteRelationalStoreConfigurationPayload):
            raise ExistingCapabilityConfigurationRealizationError(
                ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_CONFIGURATION,
                detail="expected SQLiteRelationalStoreConfigurationPayload",
            )
        if (
            configuration.configuration_type
            != SQLITE_RELATIONAL_STORE_CONFIGURATION_TYPE
        ):
            raise ExistingCapabilityConfigurationRealizationError(
                ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_CONFIGURATION,
                detail="configuration_type mismatch",
            )
        if (
            configuration.configuration_version
            != SQLITE_RELATIONAL_STORE_CONFIGURATION_VERSION
        ):
            raise ExistingCapabilityConfigurationRealizationError(
                ExistingCapabilityConfigurationRealizationFailureReason.UNSUPPORTED_CONFIGURATION_VERSION,
                detail="configuration_version mismatch",
            )
        if (
            configuration.configuration_fingerprint
            != request.configuration_fingerprint.strip()
        ):
            raise ExistingCapabilityConfigurationRealizationError(
                ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
                detail="configuration fingerprint mismatch",
            )
        _validate_sqlite_provider_configuration(configuration)
        return ConfiguredCapabilityBinding(
            tenant_id=request.tenant_id,
            integration_category=IntegrationCategory.RELATIONAL_STORE,
            provider_id=SQLITE_RELATIONAL_STORE_PROVIDER_ID,
            resource_scope=request.resource_scope,
            configuration_type=configuration.configuration_type,
            configuration_version=configuration.configuration_version,
            configuration_fingerprint=configuration.configuration_fingerprint,
            realization_evidence_refs=(),
        )


def _fail_closed_realize_continuity(
    request: ExistingCapabilityConfigurationRealizationRequest,
    existing_target: ExistingCapabilityIntegrationTarget,
) -> None:
    if existing_target.tenant_id != request.tenant_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.TENANT_MISMATCH,
            detail="target tenant mismatch",
        )
    if existing_target.provider_id != request.provider_id:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="target provider mismatch",
        )
    if existing_target.integration_category != request.integration_category:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="target category mismatch",
        )
    if existing_target.resource_scope != request.resource_scope:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="target resource_scope mismatch",
        )
    if request.integration_category is not IntegrationCategory.RELATIONAL_STORE:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="request category mismatch",
        )
    if request.provider_id != SQLITE_RELATIONAL_STORE_PROVIDER_ID:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.IDENTITY_MISMATCH,
            detail="request provider mismatch",
        )


def _validate_sqlite_provider_configuration(
    payload: SQLiteRelationalStoreConfigurationPayload,
) -> None:
    try:
        SQLiteIntegrationConfig(
            data_dir=payload.data_dir,
            relational_db=payload.relational_db,
        )
    except ValidationError as exc:
        raise ExistingCapabilityConfigurationRealizationError(
            ExistingCapabilityConfigurationRealizationFailureReason.INVALID_CONFIGURATION,
            detail=str(exc),
        ) from exc

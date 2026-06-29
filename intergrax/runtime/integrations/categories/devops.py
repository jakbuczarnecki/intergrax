# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""DevOps, cloud, and workflow provider category contracts (INTEGRATIONS-2A)."""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from intergrax.runtime.integrations.categories._base import (
    CategoryIntegrationConfig,
    _CONNECT_HEALTH,
    _CONNECT_READ_HEALTH,
    _CONNECT_READ_WRITE_HEALTH,
    _CONNECT_WRITE_HEALTH,
    category_for_provider,
)
from intergrax.runtime.integrations.contracts import (
    PlatformIntegrationCapability,
    PlatformIntegrationContract,
    PlatformIntegrationKind,
)

CI_CD_INTEGRATION_CONTRACT_SCHEMA = "ci_cd_integration_contract.v1"
SECURITY_SCANNER_INTEGRATION_CONTRACT_SCHEMA = "security_scanner_integration_contract.v1"
SANDBOX_HOST_INTEGRATION_CONTRACT_SCHEMA = "sandbox_host_integration_contract.v1"
WORKFLOW_ORCHESTRATOR_INTEGRATION_CONTRACT_SCHEMA = "workflow_orchestrator_integration_contract.v1"
CLOUD_PLATFORM_INTEGRATION_CONTRACT_SCHEMA = "cloud_platform_integration_contract.v1"


class CiCdIntegrationContract(PlatformIntegrationContract):
    """Category contract for ci_cd providers (github_actions, jenkins, …)."""

    schema_id: Literal["ci_cd_integration_contract.v1"] = CI_CD_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.CI_CD.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> CiCdIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.CI_CD.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class SecurityScannerIntegrationContract(PlatformIntegrationContract):
    """Category contract for security_scanner providers (trivy, snyk, …)."""

    schema_id: Literal["security_scanner_integration_contract.v1"] = (
        SECURITY_SCANNER_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.SECURITY_SCANNER.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> SecurityScannerIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.SECURITY_SCANNER.value,
            default_capabilities=_CONNECT_READ_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class SandboxHostIntegrationContract(PlatformIntegrationContract):
    """Category contract for sandbox_host providers (e2b, modal, …)."""

    schema_id: Literal["sandbox_host_integration_contract.v1"] = SANDBOX_HOST_INTEGRATION_CONTRACT_SCHEMA
    integration_kind: str = PlatformIntegrationKind.SANDBOX_HOST.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> SandboxHostIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.SANDBOX_HOST.value,
            default_capabilities=_CONNECT_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class WorkflowOrchestratorIntegrationContract(PlatformIntegrationContract):
    """Category contract for workflow_orchestrator providers (prefect, airflow, …)."""

    schema_id: Literal["workflow_orchestrator_integration_contract.v1"] = (
        WORKFLOW_ORCHESTRATOR_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.WORKFLOW_ORCHESTRATOR.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_READ_WRITE_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> WorkflowOrchestratorIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.WORKFLOW_ORCHESTRATOR.value,
            default_capabilities=_CONNECT_READ_WRITE_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )


class CloudPlatformIntegrationContract(PlatformIntegrationContract):
    """Category contract for cloud_platform providers (aws, azure, gcp, …)."""

    schema_id: Literal["cloud_platform_integration_contract.v1"] = (
        CLOUD_PLATFORM_INTEGRATION_CONTRACT_SCHEMA
    )
    integration_kind: str = PlatformIntegrationKind.CLOUD_PLATFORM.value
    capabilities: tuple[PlatformIntegrationCapability, ...] = Field(
        default_factory=lambda: _CONNECT_HEALTH
    )
    config: CategoryIntegrationConfig = Field(default_factory=CategoryIntegrationConfig)

    @classmethod
    def for_provider(
        cls,
        *,
        provider_id: str,
        capabilities: tuple[PlatformIntegrationCapability, ...] | None = None,
        display_name: str | None = None,
        version: str | None = None,
        config: CategoryIntegrationConfig | None = None,
    ) -> CloudPlatformIntegrationContract:
        return category_for_provider(
            cls,
            provider_id=provider_id,
            integration_kind=PlatformIntegrationKind.CLOUD_PLATFORM.value,
            default_capabilities=_CONNECT_HEALTH,
            capabilities=capabilities,
            display_name=display_name,
            version=version,
            config=config,
        )

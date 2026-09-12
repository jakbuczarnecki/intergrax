# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""LLM provider descriptor for ExternalOperationProvider SPI (R1)."""

from __future__ import annotations

from intergrax.contracts.external_operations.attempt import (
    ExternalOperationAttempt,
    ExternalOperationAttemptLifecycle,
)
from intergrax.contracts.external_operations.evidence import ProviderExecutionOutcome
from intergrax.contracts.external_operations.provider import (
    ProviderPayloadBounds,
    ProviderRiskProfile,
)
from intergrax.contracts.external_operations.safety import (
    ExternalOperationExecutionForbiddenError,
)


class LlmExternalOperationProvider:
    """Metadata-only provider SPI — SDK invocation stays in LLMAdapter after admission."""

    def __init__(
        self,
        *,
        provider_id: str,
        version: str,
        capabilities: frozenset[str],
        tenant_scope: frozenset[str] | None,
        risk_profile: ProviderRiskProfile,
        payload_bounds: ProviderPayloadBounds,
    ) -> None:
        self._provider_id = provider_id
        self._version = version
        self._capabilities = capabilities
        self._tenant_scope = tenant_scope
        self._risk_profile = risk_profile
        self._payload_bounds = payload_bounds

    @property
    def provider_id(self) -> str:
        return self._provider_id

    @property
    def version(self) -> str:
        return self._version

    @property
    def capabilities(self) -> frozenset[str]:
        return self._capabilities

    @property
    def tenant_scope(self) -> frozenset[str] | None:
        return self._tenant_scope

    @property
    def risk_profile(self) -> ProviderRiskProfile:
        return self._risk_profile

    @property
    def payload_bounds(self) -> ProviderPayloadBounds:
        return self._payload_bounds

    def execute_admitted(
        self,
        attempt: ExternalOperationAttempt,
    ) -> ProviderExecutionOutcome:
        if attempt.lifecycle is not ExternalOperationAttemptLifecycle.EXECUTING:
            raise ExternalOperationExecutionForbiddenError(
                "LLM provider SPI is metadata-only; use adapter after admission"
            )
        return ProviderExecutionOutcome(
            status="success",
            safe_summary="admitted llm provider slot",
        )


def llm_provider_descriptor(
    *,
    provider_slug: str,
    version: str = "1",
    risk_profile: ProviderRiskProfile = ProviderRiskProfile.MEDIUM,
) -> LlmExternalOperationProvider:
    return LlmExternalOperationProvider(
        provider_id=provider_slug,
        version=version,
        capabilities=frozenset({"llm.generate", "llm.stream"}),
        tenant_scope=None,
        risk_profile=risk_profile,
        payload_bounds=ProviderPayloadBounds(
            max_payload_bytes=4_194_304,
            timeout_seconds=120.0,
            max_retries=0,
        ),
    )

# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Integration engine — selects adapters via injected providers only."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime

from intergrax.contracts.decision.integration.audit import (
    DecisionIntegrationAuditProvider,
    DecisionIntegrationAuditRecord,
)
from intergrax.contracts.decision.integration.protocol import (
    DecisionIntegrationAdapterProvider,
    DecisionLifecycleIntegrationAdapter,
)
from intergrax.contracts.decision.integration.references import (
    ReferenceDecisionLifecycleReference,
)
from intergrax.contracts.decision.integration.result import (
    DecisionAdapterMetadata,
    DecisionIntegrationResult,
    DecisionIntegrationStatus,
)


def _resolve_lifecycle_adapter(
    providers: tuple[DecisionIntegrationAdapterProvider, ...],
    source_type: str,
) -> DecisionLifecycleIntegrationAdapter | None:
    for provider in providers:
        adapter = provider.provide_lifecycle_adapter(source_type)
        if adapter is not None:
            return adapter
    return None


def _failed_no_adapter(
    source: ReferenceDecisionLifecycleReference,
    integrated_at: datetime,
) -> DecisionIntegrationResult:
    metadata = DecisionAdapterMetadata(
        source_type=source.source_type,
        adapter_id="integration.unresolved",
        adapter_version="0",
        mapping_version=source.mapping_version,
        integrated_at=integrated_at,
    )
    return DecisionIntegrationResult(
        status=DecisionIntegrationStatus.FAILED,
        source=source,
        target=None,
        adapter_metadata=metadata,
        detail="no lifecycle integration adapter registered for source_type",
    )


@dataclass(frozen=True, slots=True)
class DecisionSystemIntegrationEngine:
    adapter_providers: tuple[DecisionIntegrationAdapterProvider, ...]
    audit_provider: DecisionIntegrationAuditProvider | None = None

    def __post_init__(self) -> None:
        if type(self.adapter_providers) is not tuple:
            raise TypeError("adapter_providers must be tuple")
        for item in self.adapter_providers:
            if not isinstance(item, DecisionIntegrationAdapterProvider):
                raise TypeError(
                    "adapter_providers items must implement DecisionIntegrationAdapterProvider",
                )

    def integrate_lifecycle(
        self,
        source: ReferenceDecisionLifecycleReference,
    ) -> DecisionIntegrationResult:
        if type(source) is not ReferenceDecisionLifecycleReference:
            raise TypeError("source must be ReferenceDecisionLifecycleReference")

        integrated_at = datetime.now(tz=UTC)
        adapter = _resolve_lifecycle_adapter(self.adapter_providers, source.source_type)
        if adapter is None:
            result = _failed_no_adapter(source, integrated_at)
            self._maybe_audit(result)
            return result

        try:
            result = adapter.integrate_lifecycle(source)
        except Exception as exc:
            metadata = DecisionAdapterMetadata(
                source_type=source.source_type,
                adapter_id=adapter.adapter_id,
                adapter_version=adapter.adapter_version,
                mapping_version=adapter.mapping_version,
                integrated_at=integrated_at,
            )
            result = DecisionIntegrationResult(
                status=DecisionIntegrationStatus.FAILED,
                source=source,
                target=None,
                adapter_metadata=metadata,
                detail=f"adapter_execution_error:{type(exc).__name__}",
            )
            self._maybe_audit(result)
            return result

        if type(result) is not DecisionIntegrationResult:
            raise TypeError(
                "adapter.integrate_lifecycle must return DecisionIntegrationResult"
            )

        self._maybe_audit(result)
        return result

    def _maybe_audit(self, result: DecisionIntegrationResult) -> None:
        if self.audit_provider is None:
            return
        self.audit_provider.record_integration(
            DecisionIntegrationAuditRecord(
                status=result.status,
                source=result.source,
                target=result.target,
                adapter_metadata=result.adapter_metadata,
                mapping_detail=result.detail,
            ),
        )


__all__ = ["DecisionSystemIntegrationEngine"]

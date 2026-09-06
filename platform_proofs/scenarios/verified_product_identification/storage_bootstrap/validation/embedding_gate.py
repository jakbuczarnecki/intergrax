"""Gate 0 — embedding provider live compatibility before dense index bootstrap."""

from __future__ import annotations

from intergrax.rag.embedding.contracts.embedding_provider import EmbeddingProvider

from platform_proofs.scenarios.verified_product_identification.application.config.embedding_configuration import (
    VpiEmbeddingConfiguration,
)
from platform_proofs.scenarios.verified_product_identification.integrations.embedding.intergrax_adapter import (
    IntergraxEmbeddingBootstrapAdapter,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.contracts.results import (
    EmbeddingProbeResult,
    ValidationCheck,
    ValidationReport,
)
from platform_proofs.scenarios.verified_product_identification.storage_bootstrap.validation.embedding_probe_validation import (
    GATE0_PROBE_TEXTS,
)


class RegistryEmbeddingReadinessProbe:
    """Compatibility wrapper over ``IntergraxEmbeddingBootstrapAdapter`` for Gate 0 reports."""

    def __init__(
        self,
        configuration: VpiEmbeddingConfiguration,
        *,
        provider: EmbeddingProvider | None = None,
        probe_texts: tuple[str, ...] = GATE0_PROBE_TEXTS,
    ) -> None:
        self._adapter = IntergraxEmbeddingBootstrapAdapter(
            configuration,
            provider=provider,
            probe_texts=probe_texts,
        )

    def probe(self) -> ValidationReport:
        probe_result = self._adapter.probe()
        return ValidationReport.from_checks(
            (
                ValidationCheck(
                    name="embedding_gate0",
                    status=probe_result.status,
                    detail=probe_result.detail,
                ),
            )
        )

    def probe_detail(self) -> EmbeddingProbeResult:
        return self._adapter.probe()

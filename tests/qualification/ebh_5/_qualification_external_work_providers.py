# © Artur Czarnecki. All rights reserved.

"""Two distinct ExternalWorkIntegration qualification providers (subclass implementations)."""

from __future__ import annotations

from intergrax.contracts.external_work import (
    ExternalContractorIdentity,
    ExternalWorkCreateRequest,
    ExternalWorkProviderDescriptor,
    ExternalWorkSnapshot,
)
from tests.unit.integrations.test_external_work_integration import (
    _create_request,
    _InMemoryExternalWorkIntegration,
)


class QualExternalWorkAlphaIntegration(_InMemoryExternalWorkIntegration):
    """Sanctioned qualification provider A — distinct type and descriptor identity."""

    def discover(self) -> ExternalWorkProviderDescriptor:
        base = super().discover()
        identity = ExternalContractorIdentity(
            provider_id="ebh7_qual_external_work_alpha",
            contractor_id="contractor-alpha",
            protocol_id=base.identity.protocol_id,
            descriptor_digest=base.identity.descriptor_digest,
        )
        return base.model_copy(update={"identity": identity})

    def create_work(self, request: ExternalWorkCreateRequest) -> ExternalWorkSnapshot:
        snapshot = super().create_work(request)
        return snapshot.model_copy(update={"provider_state_label": "alpha-impl"})


class QualExternalWorkBetaIntegration(_InMemoryExternalWorkIntegration):
    """Sanctioned qualification provider B — distinct type and descriptor identity."""

    def discover(self) -> ExternalWorkProviderDescriptor:
        base = super().discover()
        identity = ExternalContractorIdentity(
            provider_id="ebh7_qual_external_work_beta",
            contractor_id="contractor-beta",
            protocol_id=base.identity.protocol_id,
            descriptor_digest=base.identity.descriptor_digest,
        )
        return base.model_copy(update={"identity": identity})

    def create_work(self, request: ExternalWorkCreateRequest) -> ExternalWorkSnapshot:
        snapshot = super().create_work(request)
        return snapshot.model_copy(update={"provider_state_label": "beta-impl"})


def qual_external_work_create_request(**overrides: object) -> ExternalWorkCreateRequest:
    payload: dict[str, object] = {"provider_id": "ebh7_qual_external_work_alpha", **overrides}
    return _create_request(**payload)

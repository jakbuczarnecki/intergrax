# © Artur Czarnecki. All rights reserved.

"""Test-only integration configuration payload codecs (TRACE-X-P5-R2-P2)."""

from __future__ import annotations

from dataclasses import dataclass

from intergrax.integrations.contracts.existing_capability_configuration import (
    IntegrationConfigurationPayload,
)
from intergrax.integrations.contracts.integration_configuration_payload_codec import (
    IntegrationConfigurationPayloadCodec,
    integration_configuration_payload_codec_registry,
)
from intergrax.knowledge.contracts.validation import JsonValue

TEST_CONFIGURATION_PAYLOAD_TYPE = "test.config.v1"


@dataclass(frozen=True)
class QualificationIntegrationConfigurationPayload:
    _configuration_type: str
    _configuration_version: str
    _configuration_fingerprint: str

    @property
    def configuration_type(self) -> str:
        return self._configuration_type

    @property
    def configuration_version(self) -> str:
        return self._configuration_version

    @property
    def configuration_fingerprint(self) -> str:
        return self._configuration_fingerprint


class QualificationIntegrationConfigurationPayloadCodec:
    def configuration_type(self) -> str:
        return TEST_CONFIGURATION_PAYLOAD_TYPE

    def encode(self, payload: IntegrationConfigurationPayload) -> JsonValue:
        if not isinstance(payload, QualificationIntegrationConfigurationPayload):
            raise TypeError("expected QualificationIntegrationConfigurationPayload")
        return {
            "configuration_version": payload.configuration_version,
            "configuration_fingerprint": payload.configuration_fingerprint,
        }

    def decode(self, payload: JsonValue) -> IntegrationConfigurationPayload:
        if not isinstance(payload, dict):
            raise ValueError("invalid test configuration payload record")
        return QualificationIntegrationConfigurationPayload(
            _configuration_type=TEST_CONFIGURATION_PAYLOAD_TYPE,
            _configuration_version=payload["configuration_version"],
            _configuration_fingerprint=payload["configuration_fingerprint"],
        )


def qualification_integration_configuration_payload_codec_registry():
    return integration_configuration_payload_codec_registry(
        codecs=(QualificationIntegrationConfigurationPayloadCodec(),),
    )


__all__ = [
    "TEST_CONFIGURATION_PAYLOAD_TYPE",
    "QualificationIntegrationConfigurationPayload",
    "QualificationIntegrationConfigurationPayloadCodec",
    "qualification_integration_configuration_payload_codec_registry",
]

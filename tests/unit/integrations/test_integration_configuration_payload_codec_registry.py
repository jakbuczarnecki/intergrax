# © Artur Czarnecki. All rights reserved.

"""Integration configuration payload codec registry hardening (TRACE-X-P5-R2-P2-R1)."""

from __future__ import annotations

import pytest

from intergrax.integrations.contracts.existing_capability_configuration import (
    IntegrationConfigurationPayload,
)
from intergrax.integrations.contracts.integration_configuration_payload_codec import (
    IntegrationConfigurationPayloadCodec,
    integration_configuration_payload_codec_registry,
)
from intergrax.knowledge.contracts.validation import JsonValue
from testing_support.integration_configuration_payload_codecs import (
    QualificationIntegrationConfigurationPayload,
    QualificationIntegrationConfigurationPayloadCodec,
    TEST_CONFIGURATION_PAYLOAD_TYPE,
)


class _WrongTypeOnDecodeCodec(QualificationIntegrationConfigurationPayloadCodec):
    def decode(self, payload: JsonValue) -> IntegrationConfigurationPayload:
        decoded = super().decode(payload)
        if not isinstance(decoded, QualificationIntegrationConfigurationPayload):
            raise TypeError("expected qualification payload")
        return QualificationIntegrationConfigurationPayload(
            _configuration_type="other.type.v1",
            _configuration_version=decoded.configuration_version,
            _configuration_fingerprint=decoded.configuration_fingerprint,
        )


class _EncodeIdentityDriftCodec(IntegrationConfigurationPayloadCodec):
    def __init__(self) -> None:
        self._configuration_type_calls = 0

    def configuration_type(self) -> str:
        self._configuration_type_calls += 1
        if self._configuration_type_calls == 1:
            return TEST_CONFIGURATION_PAYLOAD_TYPE
        return "drifted.codec.type"

    def encode(self, payload: IntegrationConfigurationPayload) -> JsonValue:
        return {"configuration_version": "1", "configuration_fingerprint": "fp"}

    def decode(self, payload: JsonValue) -> IntegrationConfigurationPayload:
        return QualificationIntegrationConfigurationPayload(
            _configuration_type=TEST_CONFIGURATION_PAYLOAD_TYPE,
            _configuration_version="1",
            _configuration_fingerprint="fp",
        )


class _WhitespaceTypeCodec(IntegrationConfigurationPayloadCodec):
    def configuration_type(self) -> str:
        return " spaced.type "

    def encode(self, payload: IntegrationConfigurationPayload) -> JsonValue:
        return {}

    def decode(self, payload: JsonValue) -> IntegrationConfigurationPayload:
        raise NotImplementedError


pytestmark = pytest.mark.unit


def test_registry_rejects_duplicate_codec_type() -> None:
    codec = QualificationIntegrationConfigurationPayloadCodec()
    with pytest.raises(ValueError, match="duplicate"):
        integration_configuration_payload_codec_registry(codecs=(codec, codec))


def test_registry_rejects_blank_and_whitespace_codec_type() -> None:
    class _BlankTypeCodec(IntegrationConfigurationPayloadCodec):
        def configuration_type(self) -> str:
            return ""

        def encode(self, payload: IntegrationConfigurationPayload) -> JsonValue:
            return {}

        def decode(self, payload: JsonValue) -> IntegrationConfigurationPayload:
            raise NotImplementedError

    with pytest.raises(ValueError, match="non-empty"):
        integration_configuration_payload_codec_registry(codecs=(_BlankTypeCodec(),))
    with pytest.raises(ValueError, match="whitespace"):
        integration_configuration_payload_codec_registry(codecs=(_WhitespaceTypeCodec(),))


def test_registry_is_immutable_after_construction() -> None:
    registry = integration_configuration_payload_codec_registry(
        codecs=(QualificationIntegrationConfigurationPayloadCodec(),),
    )
    with pytest.raises(TypeError):
        registry._codecs["new"] = QualificationIntegrationConfigurationPayloadCodec()  # type: ignore[index]


def test_mutating_source_mapping_does_not_affect_registry() -> None:
    codec = QualificationIntegrationConfigurationPayloadCodec()
    shadow: dict[str, IntegrationConfigurationPayloadCodec] = {
        codec.configuration_type(): codec,
    }
    registry = integration_configuration_payload_codec_registry(codecs=(codec,))
    shadow["extra.type"] = codec
    assert "extra.type" not in registry._codecs


def test_encode_rejects_unknown_configuration_type() -> None:
    registry = integration_configuration_payload_codec_registry(
        codecs=(QualificationIntegrationConfigurationPayloadCodec(),),
    )
    payload = QualificationIntegrationConfigurationPayload(
        _configuration_type="unknown.type",
        _configuration_version="1",
        _configuration_fingerprint="fp",
    )
    with pytest.raises(ValueError, match="unknown"):
        registry.encode(payload)


def test_decode_rejects_unknown_configuration_type() -> None:
    registry = integration_configuration_payload_codec_registry(
        codecs=(QualificationIntegrationConfigurationPayloadCodec(),),
    )
    with pytest.raises(ValueError, match="unknown"):
        registry.decode(configuration_type="missing", payload={})


def test_decode_rejects_configuration_type_mismatch_from_codec() -> None:
    registry = integration_configuration_payload_codec_registry(
        codecs=(_WrongTypeOnDecodeCodec(),),
    )
    with pytest.raises(ValueError, match="mismatch on decode"):
        registry.decode(
            configuration_type=TEST_CONFIGURATION_PAYLOAD_TYPE,
            payload={"configuration_version": "1", "configuration_fingerprint": "fp"},
        )


def test_encode_rejects_codec_identity_mismatch() -> None:
    registry = integration_configuration_payload_codec_registry(
        codecs=(_EncodeIdentityDriftCodec(),),
    )
    payload = QualificationIntegrationConfigurationPayload(
        _configuration_type=TEST_CONFIGURATION_PAYLOAD_TYPE,
        _configuration_version="1",
        _configuration_fingerprint="fp",
    )
    with pytest.raises(ValueError, match="mismatch on encode"):
        registry.encode(payload)

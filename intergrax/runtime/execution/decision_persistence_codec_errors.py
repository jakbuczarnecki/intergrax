# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Typed errors for Decision durable persistence wire codecs."""

from __future__ import annotations


class DecisionPersistenceCodecError(ValueError):
    """Base error for malformed or unsupported durable Decision wire records."""


class DecisionPersistenceUnsupportedSchemaError(DecisionPersistenceCodecError):
    """Raised when ``schema_version`` is unknown or unsupported."""


class DecisionPersistenceRecordTypeError(DecisionPersistenceCodecError):
    """Raised when ``record_type`` is unknown or mismatched."""


class DecisionPersistenceUnknownPayloadCodecError(DecisionPersistenceCodecError):
    """Raised when artifact payload reconstruction lacks an explicit codec."""


class DecisionPersistenceLegacyPickleUnsupportedError(DecisionPersistenceCodecError):
    """Raised when a legacy executable durable blob is encountered.

    Runtime auto-deserialization of legacy blobs is forbidden at the authority boundary.
    """


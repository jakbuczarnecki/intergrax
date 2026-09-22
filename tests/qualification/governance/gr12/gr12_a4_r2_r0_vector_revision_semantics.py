# © Artur Czarnecki. All rights reserved.

"""GR-12-A4-R2-R0 — vector identity, configuration projection, TOCTOU semantics SSOT."""

from __future__ import annotations

from enum import StrEnum
from typing import Final

from intergrax.applications._shared.vector_index_configuration_projection import (
    VectorIndexConfigurationProjection,
)



class Gr12VectorIdentityIntrinsicValidation(StrEnum):
    """``VectorIndexIdentity`` is a frozen dataclass without ``__post_init__`` validation."""

    NONE = "NONE"


class Gr12VectorLiveOperatorIdentityValidation(StrEnum):
    """Mandatory before CLA-04 resource mapping on the live operator path."""

    REQUIRED_NON_EMPTY_LOGICAL_NAME_AND_TENANT_ID = (
        "REQUIRED_NON_EMPTY_LOGICAL_NAME_AND_TENANT_ID"
    )


class Gr12VectorAbsentIndexRevisionState(StrEnum):
    """Canonical current revision token when ``describe_index`` reports ``exists=False``."""

    ABSENT = "ABSENT"


class Gr12VectorStaleAuthorizationBehavior(StrEnum):
    """Enterprise-safe default after optimistic re-read detects digest drift."""

    STALE_INVALIDATES_PRIOR_AUTHORIZATION_ABORT = (
        "STALE_INVALIDATES_PRIOR_AUTHORIZATION_ABORT"
    )


class Gr12VectorStaleRetryFreshAuthorizationPolicy(StrEnum):
    """Bounded TOCTOU handling; no unbounded reauthorize loops."""

    ONE_EXPLICIT_REEVALUATION_OR_ABORT = "ONE_EXPLICIT_REEVALUATION_OR_ABORT"


class Gr12VectorProviderCasAvailability(StrEnum):
    UNAVAILABLE = "UNAVAILABLE"


class Gr12VectorLivePrepareAuthorizationTiming(StrEnum):
    """Governed before port invocation even when outcome may be ``ALREADY_COMPATIBLE``."""

    BEFORE_PREPARE_INDEX_INVOCATION = "BEFORE_PREPARE_INDEX_INVOCATION"


class Gr12VectorAlreadyCompatibleRevisionSemantics(StrEnum):
    """``validate_spec_against_description`` is compatibility, not full config equality."""

    NOT_REVISION_EQUALITY = "NOT_REVISION_EQUALITY"


class Gr12VectorConfigurationDigestField(StrEnum):
    LOGICAL_NAME = "logical_name"
    TENANT_ID = "tenant_id"
    DENSE_DIMENSION = "dense_dimension"
    DENSE_METRIC = "dense_metric"
    DENSE_CHANNEL_NAME = "dense_channel_name"
    REQUIRED_CAPABILITIES = "required_capabilities"
    SPARSE_LEXICAL_CHANNEL_NAME = "sparse_lexical_channel_name"


GR12_VECTOR_CONFIGURATION_DIGEST_INCLUDED_FIELDS: Final[
    tuple[Gr12VectorConfigurationDigestField, ...]
] = (
    Gr12VectorConfigurationDigestField.LOGICAL_NAME,
    Gr12VectorConfigurationDigestField.TENANT_ID,
    Gr12VectorConfigurationDigestField.DENSE_DIMENSION,
    Gr12VectorConfigurationDigestField.DENSE_METRIC,
    Gr12VectorConfigurationDigestField.DENSE_CHANNEL_NAME,
    Gr12VectorConfigurationDigestField.REQUIRED_CAPABILITIES,
    Gr12VectorConfigurationDigestField.SPARSE_LEXICAL_CHANNEL_NAME,
)

GR12_VECTOR_CONFIGURATION_DIGEST_EXCLUDED_RUNTIME_FIELDS: Final[tuple[str, ...]] = (
    "point_count",
    "reachable",
    "health_status",
    "timestamps",
    "provider_uuid",
    "host",
    "url",
    "physical_collection_name",
    "credentials",
)

GR12_VECTOR_FAKE_TENANT_ID_PATTERNS_FORBIDDEN: Final[tuple[str, ...]] = (
    "platform",
    "default",
    "profile_id_substitute",
)


GR12_VECTOR_CONFIGURATION_PROJECTION_SCHEMA: Final[str] = (
    "VectorIndexConfigurationProjection"
)

GR12_VECTOR_REVISION_DIGEST_INVARIANT: Final[str] = (
    "spec or description → VectorIndexConfigurationProjection → deterministic digest; "
    "equal logical configuration → equal digest (provider-neutral canonical serialization)"
)

GR12_VECTOR_RESOURCE_MAPPING_AFTER_VALIDATION: Final[str] = (
    "resource_id={tenant_id}/{logical_name}; "
    "resource_scope=vector_index.tenant/{tenant_id}"
)

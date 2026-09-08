"""Preset representation policies and environment resolution."""

from __future__ import annotations

import os

from platform_proofs.scenarios.verified_product_identification.dataset.data_pack.application.semantic_representation.contracts import (
    SemanticRepresentationPolicy,
    TruncationStrategy,
)

VPI_SEMANTIC_REPRESENTATION_ENV = "VPI_SEMANTIC_REPRESENTATION_POLICY"

DEFAULT_PRESERVED_FIELDS: tuple[str, ...] = (
    "brand",
    "model",
    "manufacturer",
    "part_number",
    "mpn",
    "sku",
    "gtin",
    "product_id",
    "identifiers",
    "title",
)

BOUNDED_POLICY_V1_CHARACTERS = 8000
BOUNDED_POLICY_V1_TOKENS = 2000
BOUNDED_POLICY_COMPACT_CHARACTERS = 4000


class RepresentationPolicyProfile:
    """Named experiment profiles for semantic representation derivation."""

    FULL_V1 = "full_v1"
    BOUND_8000 = "bound_8000"
    BOUND_4000 = "bound_4000"
    TOKEN_LIMIT_2000 = "token_limit_2000"

    @classmethod
    def all_profiles(cls) -> tuple[str, ...]:
        return (
            cls.FULL_V1,
            cls.BOUND_8000,
            cls.BOUND_4000,
            cls.TOKEN_LIMIT_2000,
        )


def bounded_policy_v1() -> SemanticRepresentationPolicy:
    """Default bounded policy — 8000 characters and 2000 estimated tokens."""
    return SemanticRepresentationPolicy(
        max_characters=BOUNDED_POLICY_V1_CHARACTERS,
        max_tokens=BOUNDED_POLICY_V1_TOKENS,
        preserved_fields=DEFAULT_PRESERVED_FIELDS,
        truncation_strategy=TruncationStrategy.PRIORITY_COMPRESSION,
    )


def bounded_policy_compact() -> SemanticRepresentationPolicy:
    """Aggressive bounded policy — 4000 characters with shared token ceiling."""
    return SemanticRepresentationPolicy(
        max_characters=BOUNDED_POLICY_COMPACT_CHARACTERS,
        max_tokens=BOUNDED_POLICY_V1_TOKENS,
        preserved_fields=DEFAULT_PRESERVED_FIELDS,
        truncation_strategy=TruncationStrategy.PRIORITY_COMPRESSION,
    )


def token_limit_policy_v1() -> SemanticRepresentationPolicy:
    """Token-first bounded policy without an explicit character ceiling."""
    return SemanticRepresentationPolicy(
        max_characters=None,
        max_tokens=BOUNDED_POLICY_V1_TOKENS,
        preserved_fields=DEFAULT_PRESERVED_FIELDS,
        truncation_strategy=TruncationStrategy.PRIORITY_COMPRESSION,
    )


def resolve_semantic_representation_policy(
    profile: str,
) -> SemanticRepresentationPolicy | None:
    """
    Resolve one profile name to a bounded policy.

    ``RepresentationPolicyProfile.FULL_V1`` returns ``None`` to signal legacy
    unbounded derivation.
    """
    normalized = profile.strip().casefold()
    if normalized in {"", RepresentationPolicyProfile.FULL_V1.casefold()}:
        return None
    if normalized == RepresentationPolicyProfile.BOUND_8000.casefold():
        return bounded_policy_v1()
    if normalized == RepresentationPolicyProfile.BOUND_4000.casefold():
        return bounded_policy_compact()
    if normalized == RepresentationPolicyProfile.TOKEN_LIMIT_2000.casefold():
        return token_limit_policy_v1()
    msg = f"unsupported semantic representation policy profile: {profile}"
    raise ValueError(msg)


def load_semantic_representation_policy_from_env(
    *,
    env_var: str = VPI_SEMANTIC_REPRESENTATION_ENV,
) -> SemanticRepresentationPolicy | None:
    """Load bounded policy from process environment; default is legacy full v1."""
    raw_value = os.getenv(env_var)
    if raw_value is None or not raw_value.strip():
        return None
    return resolve_semantic_representation_policy(raw_value)

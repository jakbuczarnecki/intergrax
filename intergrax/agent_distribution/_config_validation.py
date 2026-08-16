# © Artur Czarnecki. All rights reserved.
# Intergrax framework – proprietary and confidential.

"""Secret-safe distribution config validation (binding + manifest defaults)."""

from __future__ import annotations

import re
from collections.abc import Mapping
from typing import Any

from intergrax.agent_distribution._immutable_json import (
    DistributionJsonValue,
    assert_distribution_json_object,
)
from intergrax.core.security import (
    FORBIDDEN_KEY,
    SECRET_LIKE_VALUE,
    SecretSafetyValidationError,
    SecretSafeValidationPolicy,
    validate_secret_safe_value,
)

AGENT_DISTRIBUTION_SECRET_POLICY = SecretSafeValidationPolicy(
    forbidden_key_fragments=frozenset(
        {
            "password",
            "secret",
            "token",
            "credential",
            "api_key",
            "apikey",
            "private_key",
            "privatekey",
        }
    ),
    forbidden_value_patterns=(
        re.compile(
            r"^(sk-|xox[baprs]-|Bearer\s|eyJ[A-Za-z0-9_-]+\.)",
            re.IGNORECASE,
        ),
    ),
    scan_string_values=True,
    split_key_segments=False,
    traverse_sequences=True,
)


def validate_non_secret_distribution_config(
    config: Mapping[str, Any],
    *,
    field_name: str = "config",
    context_label: str = "config",
) -> dict[str, DistributionJsonValue]:
    try:
        validate_secret_safe_value(
            config,
            policy=AGENT_DISTRIBUTION_SECRET_POLICY,
            context_label=context_label,
        )
    except SecretSafetyValidationError as exc:
        if exc.reason_code == FORBIDDEN_KEY:
            raise ValueError(
                f"{context_label} key '{exc.path}' must use secret_refs, not config values"
            ) from exc
        if exc.reason_code == SECRET_LIKE_VALUE:
            raise ValueError(
                f"{context_label} value for '{exc.path}' resembles a secret literal"
            ) from exc
        raise ValueError(str(exc)) from exc
    return assert_distribution_json_object(config, field_name=field_name)

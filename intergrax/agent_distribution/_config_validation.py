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

_SECRET_CONFIG_KEY_RE = re.compile(
    r"(password|secret|api[_-]?key|token|credential|private[_-]?key)",
    re.IGNORECASE,
)
_SECRET_VALUE_RE = re.compile(
    r"^(sk-|xox[baprs]-|Bearer\s|eyJ[A-Za-z0-9_-]+\.)",
    re.IGNORECASE,
)


def reject_secret_like_distribution_config_value(
    value: Any,
    *,
    path: str = "",
    context_label: str = "config",
) -> None:
    if isinstance(value, Mapping):
        for key, child in value.items():
            key_str = str(key)
            label = f"{path}.{key_str}" if path else key_str
            if _SECRET_CONFIG_KEY_RE.search(key_str):
                raise ValueError(
                    f"{context_label} key '{label}' must use secret_refs, not config values"
                )
            if isinstance(child, str) and _SECRET_VALUE_RE.match(child.strip()):
                raise ValueError(
                    f"{context_label} value for '{label}' resembles a secret literal"
                )
            reject_secret_like_distribution_config_value(
                child, path=label, context_label=context_label
            )
        return
    if isinstance(value, list):
        for index, child in enumerate(value):
            reject_secret_like_distribution_config_value(
                child,
                path=f"{path}[{index}]",
                context_label=context_label,
            )


def validate_non_secret_distribution_config(
    config: Mapping[str, Any],
    *,
    field_name: str = "config",
    context_label: str = "config",
) -> dict[str, DistributionJsonValue]:
    reject_secret_like_distribution_config_value(config, context_label=context_label)
    return assert_distribution_json_object(config, field_name=field_name)

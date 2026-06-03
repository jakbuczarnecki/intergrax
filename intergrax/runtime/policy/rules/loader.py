# © Artur Czarnecki. All rights reserved.

"""Load declarative policy rules from YAML/JSON (Phase H-APP.2.4)."""

from __future__ import annotations

import json
from pathlib import Path

from intergrax.runtime.policy.rules.schema import DeclarativePolicyRule

try:
    import yaml
except ImportError:  # pragma: no cover
    yaml = None  # type: ignore[assignment]


def load_policy_rules_from_path(path: Path) -> list[DeclarativePolicyRule]:
    """Parse rules file; supports ``.yaml``, ``.yml``, and ``.json``."""
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".yaml", ".yml"}:
        if yaml is None:
            raise RuntimeError("PyYAML required to load YAML policy rules")
        payload = yaml.safe_load(text)
    else:
        payload = json.loads(text)

    if not isinstance(payload, list):
        raise ValueError(f"Policy rules file must be a list: {path}")

    return [DeclarativePolicyRule.model_validate(item) for item in payload]

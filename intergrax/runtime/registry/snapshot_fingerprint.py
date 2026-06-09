# © Artur Czarnecki. All rights reserved.

"""Registry snapshot fingerprint for CI diff (IDEAL-19.3)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from intergrax.applications._shared.registry_snapshot import HarnessRegistrySnapshot


def snapshot_digest_payload(snapshot: HarnessRegistrySnapshot) -> dict[str, Any]:
    """Stable digest payload for registry snapshot comparison."""
    return {
        "tool_ids": list(snapshot.tool_ids()),
        "skill_ids": list(snapshot.skill_ids()),
        "prompt_ids": list(snapshot.prompt_ids()),
        "agent_contract_ids": list(snapshot.agent_contract_ids()),
        "evaluation_registry_ids": list(snapshot.evaluation_registry_ids()),
        "prompt_bindings": dict(snapshot.resolved_prompt_bindings()),
    }


def snapshot_fingerprint(snapshot: HarnessRegistrySnapshot) -> str:
    payload = snapshot_digest_payload(snapshot)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(encoded.encode("utf-8")).hexdigest()

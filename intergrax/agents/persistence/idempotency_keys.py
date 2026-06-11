# © Artur Czarnecki. All rights reserved.

"""Default idempotency key generation (architecture §40.2.1)."""

from __future__ import annotations

import hashlib
import json
from typing import Any

from intergrax.contracts.side_effect import SideEffectKind


def build_default_idempotency_key(
    *,
    run_id: str,
    step_index: int,
    kind: SideEffectKind | str,
    target: str,
    args: dict[str, Any] | None = None,
) -> str:
    payload = {
        "run_id": run_id,
        "step_index": step_index,
        "kind": str(kind),
        "target": target,
        "args": args or {},
    }
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    return f"acp:{digest[:32]}"

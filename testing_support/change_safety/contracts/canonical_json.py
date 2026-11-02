# © Artur Czarnecki. All rights reserved.

from __future__ import annotations

import hashlib
import json
from typing import Any


def canonical_json_dumps(payload: Any) -> str:
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=False)


def stable_digest_hex(payload: Any) -> str:
    encoded = canonical_json_dumps(payload).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()

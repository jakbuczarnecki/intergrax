# © Artur Czarnecki. All rights reserved.

"""Filesystem public-key store for offline ProofReceipt verification (PC-10 / FH).

Stores **verification material only** — never private keys, seeds, or attestors.
"""

from __future__ import annotations

import base64
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from intergrax.runtime.execution_evidence.attestor import ALGORITHM_ED25519

DEMO_OFFLINE_KEY_ID = "governed-contractor-offline-demo-1"
_PRIVATE_FIELD_MARKERS = frozenset(
    {
        "private_key",
        "private_key_hex",
        "private_key_base64",
        "seed",
        "mnemonic",
        "secret",
        "secret_bytes",
        "signing_key",
    }
)


def verification_key_path(store_root: Path, key_id: str) -> Path:
    return Path(store_root) / "keys" / f"{key_id}.json"


def write_verification_key_artifact(
    store_root: Path,
    *,
    key_id: str,
    public_key_bytes: bytes,
    algorithm: str = ALGORITHM_ED25519,
    status: str = "active",
    deprecated: bool = False,
    purpose: str = "host_attestation_verification",
    issuer: str = "intergrax-offline-demo",
    created_at: datetime | None = None,
) -> Path:
    """Persist public verification material under ``store_root/keys/<key_id>.json``."""
    if not key_id.strip():
        raise ValueError("key_id must be non-empty")
    if algorithm != ALGORITHM_ED25519:
        raise ValueError(f"unsupported_algorithm:{algorithm}")
    if len(public_key_bytes) != 32:
        raise ValueError("public_key must be 32 raw Ed25519 bytes")
    root = Path(store_root)
    path = verification_key_path(root, key_id)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "key_id": key_id.strip(),
        "algorithm": algorithm,
        "public_key_hex": public_key_bytes.hex(),
        "public_key_base64": base64.b64encode(public_key_bytes).decode("ascii"),
        "created_at": (created_at or datetime.now(timezone.utc)).isoformat(),
        "status": status,
        "deprecated": bool(deprecated),
        "purpose": purpose,
        "issuer": issuer,
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def write_demo_mode_marker(
    store_root: Path,
    *,
    key_id: str = DEMO_OFFLINE_KEY_ID,
) -> Path:
    """Mark store as offline deterministic demo (recovery signer = explicit demo mode)."""
    root = Path(store_root)
    root.mkdir(parents=True, exist_ok=True)
    path = root / "demo_mode.json"
    payload = {
        "mode": "offline_deterministic_demo",
        "key_id": key_id,
        "recovery_signer": "deterministic_demo",
        "warning": "local/test only - not production key custody",
    }
    path.write_text(json.dumps(payload, indent=2, sort_keys=True), encoding="utf-8")
    return path


def read_demo_mode_marker(store_root: Path) -> dict[str, Any] | None:
    path = Path(store_root) / "demo_mode.json"
    if not path.is_file():
        return None
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return None
    if not isinstance(data, dict):
        return None
    return data


class FilesystemHostKeyResolver:
    """Resolve public verification keys from a demo/host filesystem store.

    Looks under ``<store_root>/keys/<key_id>.json``. Never reads private material.
    """

    def __init__(
        self,
        store_root: Path,
        *,
        allow_deprecated: bool = True,
        allowed_algorithms: frozenset[str] | None = None,
    ) -> None:
        self._root = Path(store_root)
        self._keys_dir = self._root / "keys"
        self._allow_deprecated = allow_deprecated
        self._allowed_algorithms = set(allowed_algorithms or {ALGORITHM_ED25519})
        self._current_key_id = ""

    def resolve_public_key(self, key_id: str) -> bytes | None:
        artifact = self._load_artifact(key_id)
        if artifact is None:
            return None
        status = str(artifact.get("status") or "active").strip().lower()
        if status in {"revoked", "disabled"}:
            return None
        if bool(artifact.get("deprecated")) and not self._allow_deprecated:
            return None
        algorithm = str(artifact.get("algorithm") or "").strip()
        if algorithm and algorithm not in self._allowed_algorithms:
            raise ValueError(f"algorithm_not_allowed:{algorithm}")
        return self._public_key_bytes(artifact)

    def current_signing_key_id(self) -> str:
        return self._current_key_id

    def is_algorithm_allowed(self, algorithm: str) -> bool:
        return algorithm in self._allowed_algorithms

    def is_key_deprecated_for_verification(self, key_id: str) -> bool:
        artifact = self._load_artifact(key_id)
        if artifact is None:
            return False
        return bool(artifact.get("deprecated"))

    def _load_artifact(self, key_id: str) -> dict[str, Any] | None:
        if not key_id.strip():
            return None
        path = verification_key_path(self._root, key_id)
        if not path.is_file():
            return None
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise ValueError(f"malformed_verification_key:{key_id}") from exc
        if not isinstance(raw, dict):
            raise ValueError(f"malformed_verification_key:{key_id}")
        for marker in _PRIVATE_FIELD_MARKERS:
            if marker in raw and raw[marker] not in (None, "", False):
                raise ValueError(f"private_material_in_verification_key:{marker}")
        artifact_key_id = str(raw.get("key_id") or "").strip()
        if artifact_key_id and artifact_key_id != key_id:
            raise ValueError(f"key_id_mismatch_in_artifact:{artifact_key_id}")
        if not self._current_key_id:
            self._current_key_id = key_id
        return raw

    @staticmethod
    def _public_key_bytes(artifact: dict[str, Any]) -> bytes:
        hex_value = artifact.get("public_key_hex")
        b64_value = artifact.get("public_key_base64")
        try:
            if isinstance(hex_value, str) and hex_value.strip():
                key = bytes.fromhex(hex_value.strip())
            elif isinstance(b64_value, str) and b64_value.strip():
                key = base64.b64decode(b64_value.strip(), validate=True)
            else:
                raise ValueError("public_key_missing")
        except (ValueError, TypeError) as exc:
            raise ValueError("malformed_verification_key:public_key") from exc
        if len(key) != 32:
            raise ValueError("malformed_verification_key:public_key_length")
        return key

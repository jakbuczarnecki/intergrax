#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Create Ed25519 custody key + public trust root (custodian principal only)."""

from __future__ import annotations

import argparse
import base64
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts.qualification.evol_x_b6_attestation import (
    CREDENTIAL_PRODUCTION,
    CUSTODY_INDEPENDENT,
    IMPLEMENTER_IDENTITIES,
)


def _assert_custodian() -> None:
    if os.environ.get("EVOL_B6_CUSTODIAN_MODE") != "1":
        raise PermissionError("EVOL_B6_CUSTODIAN_MODE=1 required")
    identity = os.environ.get("EVOL_B6_CUSTODIAN_IDENTITY", "").strip().lower()
    if not identity or identity in IMPLEMENTER_IDENTITIES:
        raise PermissionError("EVOL_B6_CUSTODIAN_IDENTITY must name custodian")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--trust-root", type=Path, required=True)
    parser.add_argument("--key-id", required=True)
    parser.add_argument("--trust-root-id", required=True)
    parser.add_argument("--attestor-identity", required=True)
    parser.add_argument("--force", action="store_true", help="Refuse if key already exists")
    args = parser.parse_args(argv)

    try:
        _assert_custodian()
    except PermissionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    if args.private_key.is_file() and not args.force:
        sys.stdout.write(json.dumps({"status": "KEY_EXISTS", "path": str(args.private_key)}) + "\n")
        return 0

    if args.private_key.is_file() and args.force:
        sys.stderr.write("refusing to overwrite existing private key\n")
        return 2

    private = Ed25519PrivateKey.generate()
    raw = private.private_bytes_raw()
    args.private_key.parent.mkdir(parents=True, exist_ok=True)
    args.private_key.write_bytes(raw)

    public_b64 = base64.b64encode(private.public_key().public_bytes_raw()).decode("ascii")
    trust = {
        "schema_version": "evol_x_b6_trust_root.v1",
        "trust_root_id": args.trust_root_id,
        "custody_domain": CUSTODY_INDEPENDENT,
        "keys": [
            {
                "key_id": args.key_id,
                "public_key_b64": public_b64,
                "status": "ACTIVE",
                "credential_class": CREDENTIAL_PRODUCTION,
                "attestor_identity": args.attestor_identity,
                "controlled_by_implementer": False,
            }
        ],
    }
    args.trust_root.parent.mkdir(parents=True, exist_ok=True)
    args.trust_root.write_text(json.dumps(trust, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    sys.stdout.write(
        json.dumps(
            {
                "status": "KEY_AND_TRUST_ROOT_CREATED",
                "key_id": args.key_id,
                "trust_root_id": args.trust_root_id,
            }
        )
        + "\n"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

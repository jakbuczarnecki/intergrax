#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Custodian-side Ed25519 signing for EVOL-X B6 attestations.

Must run under a **non-implementer** principal with private key material stored
outside the repository (see ``scripts/qualification/local_b6/README.md``).

Refuses to run without ``EVOL_B6_CUSTODIAN_MODE=1`` or if the key path resolves
inside the implementer repository.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

from cryptography.hazmat.primitives.asymmetric.ed25519 import Ed25519PrivateKey
from scripts.qualification.evol_x_b6_attestation import IMPLEMENTER_IDENTITIES, bind_and_sign_attestation
from scripts.qualification.evol_x_b6_replay_ledger import ReplayLedgerError, append_binding

_REPO = Path(__file__).resolve().parents[2]


def _load_private_key(path: Path) -> Ed25519PrivateKey:
    raw = path.read_bytes()
    if len(raw) == 32:
        return Ed25519PrivateKey.from_private_bytes(raw)
    text = raw.decode("utf-8")
    if "BEGIN PRIVATE KEY" in text:
        from cryptography.hazmat.primitives.serialization import load_pem_private_key

        key = load_pem_private_key(raw, password=None)
        if not isinstance(key, Ed25519PrivateKey):
            raise ValueError("PEM must be Ed25519")
        return key
    raise ValueError("private key must be 32 raw bytes or Ed25519 PEM")


def _assert_custodian_env(*, key_path: Path, repo: Path) -> None:
    if os.environ.get("EVOL_B6_CUSTODIAN_MODE") != "1":
        raise PermissionError("set EVOL_B6_CUSTODIAN_MODE=1 only on custodian host/account")
    identity = os.environ.get("EVOL_B6_CUSTODIAN_IDENTITY", "").strip().lower()
    if not identity or identity in IMPLEMENTER_IDENTITIES:
        raise PermissionError("EVOL_B6_CUSTODIAN_IDENTITY must name custodian (non-implementer)")
    resolved = key_path.resolve()
    try:
        resolved.relative_to(repo.resolve())
        raise PermissionError("private key path must be outside implementer repository")
    except ValueError:
        pass


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=_REPO)
    parser.add_argument("--unsigned", type=Path, required=True)
    parser.add_argument("--private-key", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--replay-ledger", type=Path, default=None)
    args = parser.parse_args(argv)

    try:
        _assert_custodian_env(key_path=args.private_key, repo=args.repo)
    except PermissionError as exc:
        sys.stderr.write(f"{exc}\n")
        return 1

    try:
        body = json.loads(args.unsigned.read_text(encoding="utf-8"))
        if not isinstance(body, dict):
            raise ValueError("unsigned attestation must be a JSON object")
        private_key = _load_private_key(args.private_key)
        signed = bind_and_sign_attestation(private_key, body)
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(signed, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        if args.replay_ledger is not None:
            append_binding(args.replay_ledger, str(signed["replay_binding"]), repo_root=args.repo)
    except (OSError, ValueError, ReplayLedgerError) as exc:
        sys.stderr.write(f"custodian sign error: {exc}\n")
        return 1

    sys.stdout.write(json.dumps({"status": "SIGNED", "path": str(args.output)}) + "\n")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Compare governance SSOT file hashes against TRUSTED_POLICY_PIN.yaml (GOV-06, advisory by default)."""

from __future__ import annotations

import argparse
import hashlib
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_PIN_PATH = _REPO / "governance" / "TRUSTED_POLICY_PIN.yaml"


def file_sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def load_pins(pin_path: Path) -> dict[str, str]:
    doc = yaml.safe_load(pin_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("pin file root must be a mapping")
    pins = doc.get("pins")
    if not isinstance(pins, dict) or not pins:
        raise ValueError("pins: missing or empty")
    out: dict[str, str] = {}
    for rel, expected in pins.items():
        if not isinstance(rel, str) or not isinstance(expected, str):
            raise ValueError(f"invalid pin entry: {rel!r}")
        out[rel.replace("\\", "/")] = expected.lower().strip()
    return out


def verify_pins(*, repo: Path = _REPO, pin_path: Path = _PIN_PATH) -> list[str]:
    errors: list[str] = []
    if not pin_path.is_file():
        return [f"missing pin file: {pin_path}"]
    try:
        pins = load_pins(pin_path)
    except ValueError as exc:
        return [str(exc)]

    for rel, expected in sorted(pins.items()):
        path = repo / rel
        if not path.is_file():
            errors.append(f"missing pinned file: {rel}")
            continue
        current = file_sha256(path)
        if current != expected:
            errors.append(f"pin mismatch {rel}: expected={expected} current={current}")
    return errors


def write_pin_file(*, repo: Path = _REPO, pin_path: Path = _PIN_PATH) -> None:
    if not pin_path.is_file():
        raise FileNotFoundError(pin_path)
    doc = yaml.safe_load(pin_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        raise ValueError("pin file root must be a mapping")
    pins = doc.get("pins")
    if not isinstance(pins, dict):
        raise ValueError("pins: missing")
    updated: dict[str, str] = {}
    for rel in pins:
        rel_norm = rel.replace("\\", "/")
        path = repo / rel_norm
        if not path.is_file():
            raise FileNotFoundError(path)
        updated[rel_norm] = file_sha256(path)
    doc["pins"] = updated
    header = (
        "# GOV-06 — advisory integrity pins for DCP policy SSOT (compare via verify_trusted_policy_pin.py).\n"
        "# Enforcement: ADVISORY until operator promotes to SERVER / required CI check.\n"
        "# Regenerate hashes: uv run python scripts/governance/verify_trusted_policy_pin.py --write-pin\n\n"
    )
    body = yaml.safe_dump(doc, sort_keys=False, allow_unicode=True)
    pin_path.write_text(header + body, encoding="utf-8")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify TRUSTED_POLICY_PIN.yaml against repo files.")
    parser.add_argument("--pin", type=Path, default=_PIN_PATH, help="Pin YAML path")
    parser.add_argument(
        "--write-pin",
        action="store_true",
        help="Rewrite pin hashes from current file bytes (operator maintenance)",
    )
    args = parser.parse_args(argv)
    pin_path = args.pin if args.pin.is_absolute() else _REPO / args.pin

    if args.write_pin:
        try:
            write_pin_file(pin_path=pin_path)
        except (OSError, ValueError) as exc:
            print(f"[ERROR] {exc}", file=sys.stderr)
            return 1
        print(f"[INFO] updated {pin_path.relative_to(_REPO)}")
        return 0

    errors = verify_pins(pin_path=pin_path)
    if errors:
        for msg in errors:
            print(f"[ERROR] {msg}", file=sys.stderr)
        return 1
    print("[INFO] verify_trusted_policy_pin: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

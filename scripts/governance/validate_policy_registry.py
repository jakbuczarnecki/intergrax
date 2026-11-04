#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail closed when policy-registry.yaml references missing repo artifacts."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_REGISTRY = _REPO / "governance" / "policy-registry.yaml"


def validate_registry(*, repo: Path = _REPO, registry_path: Path = _REGISTRY) -> list[str]:
    errors: list[str] = []
    if not registry_path.is_file():
        return [f"missing registry: {registry_path}"]

    doc = yaml.safe_load(registry_path.read_text(encoding="utf-8"))
    if not isinstance(doc, dict):
        return ["policy-registry root must be a mapping"]

    policies = doc.get("policies")
    if not isinstance(policies, dict):
        errors.append("policies: missing or not a mapping")
    else:
        for key, meta in policies.items():
            if not isinstance(meta, dict):
                errors.append(f"policies.{key}: expected mapping")
                continue
            policy_id = meta.get("id")
            if not policy_id:
                errors.append(f"policies.{key}: missing id")
            artifacts = meta.get("artifacts")
            if not isinstance(artifacts, list) or not artifacts:
                errors.append(f"policies.{key}: artifacts must be a non-empty list")
                continue
            for rel in artifacts:
                if not isinstance(rel, str) or not rel.strip():
                    errors.append(f"policies.{key}: invalid artifact entry")
                    continue
                path = repo / rel.replace("\\", "/")
                if not path.is_file():
                    errors.append(f"policies.{key} ({policy_id}): missing artifact {rel}")

    gate = doc.get("commit_ref_gate")
    if not isinstance(gate, dict):
        errors.append("commit_ref_gate: missing or not a mapping")
    else:
        for field in ("config", "verifier"):
            rel = gate.get(field)
            if not isinstance(rel, str) or not rel.strip():
                errors.append(f"commit_ref_gate.{field}: missing")
                continue
            if not (repo / rel.replace("\\", "/")).is_file():
                errors.append(f"commit_ref_gate.{field}: missing file {rel}")

    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Validate governance/policy-registry.yaml artifact paths.")
    parser.add_argument("--registry", type=Path, default=_REGISTRY, help="Registry YAML path")
    args = parser.parse_args(argv)

    errors = validate_registry(registry_path=args.registry if args.registry.is_absolute() else _REPO / args.registry)
    if errors:
        for msg in errors:
            print(f"[ERROR] {msg}", file=sys.stderr)
        return 1
    print("[INFO] validate_policy_registry: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

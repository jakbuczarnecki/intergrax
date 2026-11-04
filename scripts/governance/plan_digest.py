#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""SHA-256 digest of normalized plan text (Delivery DevOS plan approval)."""

from __future__ import annotations

import argparse
import hashlib
import re
import sys
from pathlib import Path


def normalize_plan_text(text: str) -> str:
    normalized = text.replace("\r\n", "\n").replace("\r", "\n")
    normalized = re.sub(r"[ \t]+\n", "\n", normalized)
    return normalized.strip() + "\n"


def digest_text(text: str) -> str:
    payload = normalize_plan_text(text).encode("utf-8")
    return hashlib.sha256(payload).hexdigest()


def digest_file(path: Path) -> str:
    return digest_text(path.read_text(encoding="utf-8"))


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Compute plan digest SHA-256.")
    parser.add_argument("plan_file", type=Path, help="Path to plan markdown file")
    args = parser.parse_args(argv)
    if not args.plan_file.is_file():
        print(f"[ERROR] plan file not found: {args.plan_file}", file=sys.stderr)
        return 1
    value = digest_file(args.plan_file)
    print(value)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

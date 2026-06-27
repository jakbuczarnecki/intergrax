# © Artur Czarnecki. All rights reserved.
"""CI gate: generated token-budget artifacts must be fresh and idempotent (F7)."""

from __future__ import annotations

import hashlib
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]

GENERATORS: tuple[tuple[str, tuple[str, ...]], ...] = (
    ("scripts/audit/generate_audit_read_slices.py", ("docs/guides/audit_slices",)),
    ("scripts/audit/generate_architecture_read_scopes.py", ("docs/architecture",)),
    ("scripts/audit/generate_plan_read_scopes.py", ("docs/plan",)),
    ("scripts/audit/generate_domain_audit_prompts.py", ("docs/audit",)),
)


def _run_generator(script: str) -> None:
    subprocess.run(
        [sys.executable, script],
        cwd=ROOT,
        check=True,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


def _tree_digest(paths: tuple[str, ...]) -> str:
    parts: list[bytes] = []
    for rel in paths:
        base = ROOT / rel
        if base.is_file():
            parts.append(base.read_bytes())
            continue
        if not base.is_dir():
            continue
        for path in sorted(base.rglob("*")):
            if path.is_file():
                parts.append(str(path.relative_to(ROOT)).encode())
                parts.append(path.read_bytes())
    return hashlib.sha256(b"".join(parts)).hexdigest()


def main() -> int:
    errors: list[str] = []

    for script, outputs in GENERATORS:
        script_path = ROOT / script
        if not script_path.is_file():
            errors.append(f"missing generator {script}")
            continue
        try:
            before = _tree_digest(outputs)
            _run_generator(script)
            after_first = _tree_digest(outputs)
            _run_generator(script)
            after_second = _tree_digest(outputs)
        except subprocess.CalledProcessError as exc:
            errors.append(f"{script} failed: {exc.stderr or exc.stdout}")
            continue
        if after_first != after_second:
            errors.append(f"{script}: non-idempotent — second run changed outputs")
        if before != after_first:
            errors.append(
                f"{script}: stale outputs under {', '.join(outputs)} — "
                f"run `python {script}` and commit"
            )

    if errors:
        print("check_token_generator_freshness: FAILED", file=sys.stderr)
        for e in errors:
            print(f"  - {e}", file=sys.stderr)
        return 1

    print("check_token_generator_freshness: OK")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

#!/usr/bin/env python3
"""Verify that an IIAC audit session manifest is backed by real artifacts (read-only, stdlib only).

Closes the gaps the string-level validators leave (EQO-A finding A2-038 part, A2-006; plan WP-7 V1):
SHAs are checked as commit objects and in order, the four digests are bound to the bytes of the
files they name, placeholders and non-certifying verdicts cannot pass as a final manifest, and role
separation is checked beyond raw string inequality.

Profiles
  final   (default) all findings are errors. Use for a manifest that claims to be certifying.
  legacy  published drafts are reported, not rejected: only format, missing commit objects, core SHA
          order and role overlap are errors; the rest is a warning.

Exit codes: 0 PASS, 1 FAIL, 2 usage or environment error. Writes nothing.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.governance.iiac_verifier_common import (  # noqa: E402
    DIGEST_BINDINGS,
    EXIT_USAGE,
    REPO_ROOT,
    GitUnavailableError,
    check_manifest_artifacts,
    ensure_git_repo,
    load_json_object,
    report,
)

TOOL = "verify_iiac_manifest_artifacts"


def _parse_artifact_overrides(items: list[str]) -> dict[str, str]:
    valid = {digest for digest, _ in DIGEST_BINDINGS}
    out: dict[str, str] = {}
    for item in items:
        field, sep, path = item.partition("=")
        if not sep or field not in valid or not path.strip():
            raise ValueError(f"--artifact expects <digest_field>=<repo path>, digest_field one of {sorted(valid)}")
        out[field] = path.strip()
    return out


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("manifest", type=Path, help="IIAC audit session manifest JSON")
    parser.add_argument("--repo", type=Path, default=REPO_ROOT, help="git repository the SHAs and artifacts live in")
    parser.add_argument("--profile", choices=("final", "legacy"), default="final")
    parser.add_argument(
        "--at", default=None, help="commit to read artifact bytes at (default: evidence publication SHA)"
    )
    parser.add_argument(
        "--artifact",
        action="append",
        default=[],
        metavar="DIGEST_FIELD=PATH",
        help="artifact path for a digest field the manifest does not name (repeatable)",
    )
    args = parser.parse_args(argv)

    try:
        overrides = _parse_artifact_overrides(args.artifact)
    except ValueError as exc:
        print(f"{TOOL}: usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    manifest, error = load_json_object(args.manifest)
    if manifest is None:
        print(f"{TOOL}: usage error: {error}", file=sys.stderr)
        return EXIT_USAGE
    try:
        ensure_git_repo(args.repo)
        findings = check_manifest_artifacts(manifest, args.repo, profile=args.profile, at=args.at, overrides=overrides)
    except GitUnavailableError as exc:
        print(f"{TOOL}: environment error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    except ValueError as exc:  # option-like git operand, for example --at=--output=x
        print(f"{TOOL}: usage error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    return report(TOOL, findings)


if __name__ == "__main__":
    raise SystemExit(main())

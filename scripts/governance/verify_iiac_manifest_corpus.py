#!/usr/bin/env python3
"""Verify the whole corpus of IIAC audit session manifests (read-only, stdlib only).

The existing validator checks one file at a time and the N-09 guard silently skips manifests that
do not parse or validate, so an invalid draft is tolerated for ever (EQO-A A2-038 part, A2-006;
plan WP-7 V1). This verifier enumerates every ``iiac_audit_session_manifest*.json`` under the
roots and fails on:

  MANIFEST_INVALID         does not validate and is not listed in the known-draft allowlist
  ALLOWLIST_STALE_DIGEST   a listed draft changed after it was listed (its bytes no longer match)
  ALLOWLIST_ORPHAN         a listed draft is missing, or validates now (the entry is no longer true)
  ALLOWLIST_ENTRY_INVALID  an allowlist entry has no reason, no path or a malformed digest
  CERTIFYING_NOT_FINAL     a manifest named with --certifying fails the strict ``final`` artifact checks
  CORPUS_ARTIFACT_ERROR    a valid manifest fails an error-level legacy artifact check (missing
                           commit object, core SHA order, role overlap)

Warnings (placeholders, non-certifying verdicts, unbound digests) are counted by code, not rejected:
published drafts are superseded in the registers, never edited (their bytes are digest-pinned).

Exit codes: 0 PASS, 1 FAIL, 2 usage or environment error. Writes nothing.
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.governance.iiac_verifier_common import (  # noqa: E402
    EXIT_USAGE,
    REPO_ROOT,
    Finding,
    GitUnavailableError,
    base_errors,
    check_manifest_artifacts,
    ensure_git_repo,
    errors_of,
    load_json_object,
    report,
    sha256_bytes,
)

TOOL = "verify_iiac_manifest_corpus"
DEFAULT_ROOT = "docs/project/maintainers/qualification"
DEFAULT_ALLOWLIST = "scripts/governance/iiac_known_draft_manifests.v1.json"
ALLOWLIST_SCHEMA = "iiac_known_draft_manifests.v1"
_HEX64 = frozenset("0123456789abcdef")


def lf_digest(raw: bytes) -> str:
    return sha256_bytes(raw.replace(bytes([13, 10]), bytes([10])))


def discover(repo: Path, roots: list[str]) -> list[str]:
    found: set[str] = set()
    for root in roots:
        base = repo / root
        if not base.is_dir():
            raise FileNotFoundError(f"manifest root not found: {root}")
        for path in base.rglob("iiac_audit_session_manifest*.json"):
            if path.name.endswith(".schema.json"):
                continue
            found.add(path.relative_to(repo).as_posix())
    return sorted(found)


def load_allowlist(path: Path) -> tuple[dict[str, dict], list[Finding]]:
    data, error = load_json_object(path)
    if data is None:
        raise ValueError(error or "allowlist unreadable")
    if data.get("schema") != ALLOWLIST_SCHEMA:
        raise ValueError(f"allowlist schema must be {ALLOWLIST_SCHEMA!r}")
    entries = data.get("entries")
    if not isinstance(entries, list):
        raise ValueError("allowlist 'entries' must be an array")
    by_path: dict[str, dict] = {}
    findings: list[Finding] = []
    for index, entry in enumerate(entries):
        rel = str(entry.get("path") or "").replace("\\", "/") if isinstance(entry, dict) else ""
        digest = str(entry.get("sha256") or "") if isinstance(entry, dict) else ""
        reason = str(entry.get("reason") or "").strip() if isinstance(entry, dict) else ""
        if not rel or len(digest) != 64 or not set(digest) <= _HEX64 or not reason:
            findings.append(Finding("ALLOWLIST_ENTRY_INVALID", f"entry {index} needs path, 64-hex sha256 and reason"))
            continue
        if rel in by_path:
            findings.append(Finding("ALLOWLIST_ENTRY_INVALID", f"duplicate allowlist entry for {rel}"))
            continue
        by_path[rel] = entry
    return by_path, findings


def verify_corpus(
    repo: Path,
    roots: list[str],
    allowlist: dict[str, dict],
    certifying: list[str],
) -> tuple[list[Finding], Counter[str], int]:
    findings: list[Finding] = []
    warn_codes: Counter[str] = Counter()
    manifests = discover(repo, roots)
    present = set(manifests)

    for rel in manifests:
        path = repo / rel
        raw = path.read_bytes()
        data, load_error = load_json_object(path)
        problems = [load_error] if data is None else base_errors(data)
        listed = allowlist.get(rel)
        if problems:
            if listed is None:
                findings.append(Finding("MANIFEST_INVALID", f"{rel}: {problems[0]}"))
            elif lf_digest(raw) != listed.get("sha256"):  # LF-normalised: stable across checkouts
                findings.append(Finding("ALLOWLIST_STALE_DIGEST", f"{rel}: bytes changed after the draft was listed"))
            continue
        if listed is not None:
            findings.append(Finding("ALLOWLIST_ORPHAN", f"{rel}: listed as a known draft but it validates now"))
        assert data is not None
        for item in check_manifest_artifacts(data, repo, profile="legacy"):
            if item.severity == "error" and item.code != "MANIFEST_FORMAT":
                findings.append(Finding("CORPUS_ARTIFACT_ERROR", f"{rel}: {item.code}: {item.message}"))
            elif item.severity == "warn":
                warn_codes[item.code] += 1

    for rel in sorted(set(allowlist) - present):
        findings.append(Finding("ALLOWLIST_ORPHAN", f"{rel}: listed as a known draft but no such manifest was found"))

    for rel in certifying:
        rel = rel.replace("\\", "/")
        data, load_error = load_json_object(repo / rel)
        if data is None:
            findings.append(Finding("CERTIFYING_NOT_FINAL", f"{rel}: {load_error}"))
            continue
        for item in errors_of(check_manifest_artifacts(data, repo, profile="final")):
            findings.append(Finding("CERTIFYING_NOT_FINAL", f"{rel}: {item.code}: {item.message}"))
    return findings, warn_codes, len(manifests)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--repo", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--root", action="append", default=None, help=f"directory to scan (repeatable; default {DEFAULT_ROOT})"
    )
    parser.add_argument(
        "--allowlist", default=DEFAULT_ALLOWLIST, help="known-draft allowlist (path relative to the repo)"
    )
    parser.add_argument("--certifying", action="append", default=[], help="manifest that must be final (repeatable)")
    args = parser.parse_args(argv)

    repo: Path = args.repo
    try:
        ensure_git_repo(repo)
        allow_path = Path(args.allowlist)
        allow_path = allow_path if allow_path.is_absolute() else repo / allow_path
        allowlist, findings = load_allowlist(allow_path)
        more, warn_codes, count = verify_corpus(repo, args.root or [DEFAULT_ROOT], allowlist, args.certifying)
    except (GitUnavailableError, FileNotFoundError, ValueError) as exc:
        print(f"{TOOL}: usage or environment error: {exc}", file=sys.stderr)
        return EXIT_USAGE
    findings += more
    summary = {
        "manifests": count,
        "allowlisted_drafts": len(allowlist),
        "warnings_by_code": dict(sorted(warn_codes.items())),
    }
    print(f"{TOOL}: {json.dumps(summary, sort_keys=True)}")
    return report(TOOL, findings)


if __name__ == "__main__":
    raise SystemExit(main())

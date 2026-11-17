#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""Fail-closed verifier: product-path commits must include a ticket Ref: trailer."""

from __future__ import annotations

import argparse
import os
import re
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import yaml

_REPO = Path(__file__).resolve().parents[2]
_CONFIG_PATH = _REPO / "governance" / "dcp-commit-ref-config.yaml"


def _truthy_env(name: str) -> bool:
    value = os.environ.get(name)
    if value is None:
        return False
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _run_git(*args: str) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        ["git", *args],
        cwd=_REPO,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="replace",
    )


@dataclass(frozen=True)
class CommitRefConfig:
    product_path_prefixes: tuple[str, ...]
    ref_line_prefix: str
    ref_id_pattern: str
    skip_env: str

    @property
    def ref_line_regex(self) -> re.Pattern[str]:
        return re.compile(
            rf"^{re.escape(self.ref_line_prefix)}{self.ref_id_pattern}$",
            re.MULTILINE,
        )


def load_config(path: Path = _CONFIG_PATH) -> CommitRefConfig:
    raw = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(raw, dict):
        raise ValueError(f"Invalid config at {path}")
    prefixes = raw.get("product_path_prefixes")
    if not isinstance(prefixes, list) or not prefixes:
        raise ValueError(f"Invalid product_path_prefixes in {path}")
    return CommitRefConfig(
        product_path_prefixes=tuple(str(p) for p in prefixes),
        ref_line_prefix=str(raw.get("ref_line_prefix", "Ref: governance/tickets/")),
        ref_id_pattern=str(raw.get("ref_id_pattern", r"T-IGX-[0-9]{3}-[A-Z]\.md")),
        skip_env=str(raw.get("skip_env", "INTERGRAX_SKIP_TICKET_REF_GATE")),
    )


def path_touches_product(path: str, prefixes: tuple[str, ...]) -> bool:
    normalized = path.replace("\\", "/")
    return any(normalized.startswith(prefix) for prefix in prefixes)


def extract_ref_lines(message: str, config: CommitRefConfig) -> list[str]:
    return [line.strip() for line in message.splitlines() if line.strip().startswith(config.ref_line_prefix.strip())]


def message_has_valid_ref(message: str, config: CommitRefConfig) -> bool:
    refs = extract_ref_lines(message, config)
    if len(refs) != 1:
        return False
    return config.ref_line_regex.match(refs[0]) is not None


def commit_touches_product(commit: str, config: CommitRefConfig, *, strict: bool = False) -> bool:
    show = _run_git("show", "--name-only", "--format=%H", commit)
    if show.returncode != 0:
        if strict:
            raise RuntimeError(f"git show failed for {commit[:12]}")
        return False
    lines = [ln.strip() for ln in show.stdout.splitlines() if ln.strip()]
    paths = [ln for ln in lines if not re.fullmatch(r"[0-9a-f]{40}", ln, flags=re.IGNORECASE)]
    return any(path_touches_product(p, config.product_path_prefixes) for p in paths)


def list_commits_in_range(rev_range: str, *, strict: bool = False) -> list[str]:
    log = _run_git("log", "--format=%H", rev_range)
    if log.returncode != 0:
        if strict:
            raise RuntimeError(f"git log failed for range '{rev_range}' (is the base fetched?)")
        return []
    return [line.strip() for line in log.stdout.splitlines() if line.strip()]


def push_rev_range() -> str:
    upstream = _run_git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{u}")
    if upstream.returncode != 0:
        for base_ref in ("origin/development", "origin/main"):
            mb = _run_git("merge-base", "HEAD", base_ref)
            if mb.returncode == 0 and mb.stdout.strip():
                return f"{mb.stdout.strip()}..HEAD"
        head = _run_git("rev-parse", "HEAD")
        if head.returncode != 0 or not head.stdout.strip():
            return "HEAD"
        sha = head.stdout.strip()
        return f"{sha}^..{sha}"
    base = upstream.stdout.strip()
    merge_base = _run_git("merge-base", "HEAD", base)
    if merge_base.returncode != 0:
        return "HEAD"
    return f"{merge_base.stdout.strip()}..HEAD"


def _ref_ticket_path(message: str, config: CommitRefConfig) -> str:
    refs = extract_ref_lines(message, config)
    return refs[0][len("Ref: ") :] if refs else ""


def verify_range(rev_range: str, config: CommitRefConfig, *, strict: bool = False) -> list[str]:
    """Verify Ref trailers. ``strict`` (CI mode): git failures raise; the Ref must name a ticket tracked at HEAD."""

    errors: list[str] = []
    for commit in list_commits_in_range(rev_range, strict=strict):
        if not commit_touches_product(commit, config, strict=strict):
            continue
        msg = _run_git("log", "-1", "--format=%B", commit)
        if msg.returncode != 0 or not msg.stdout:
            errors.append(f"{commit[:12]}: unable to read commit message")
            continue
        if not message_has_valid_ref(msg.stdout, config):
            errors.append(
                f"{commit[:12]}: missing or invalid ticket Ref line "
                f"(expected exactly one line matching {config.ref_line_prefix}<T-IGX-NNN-L.md>)"
            )
        elif strict:
            ticket = _ref_ticket_path(msg.stdout, config)
            if _run_git("cat-file", "-e", f"HEAD:{ticket}").returncode != 0:
                errors.append(f"{commit[:12]}: Ref points to {ticket}, which is not a tracked ticket at HEAD")
    return errors


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Verify ticket Ref trailers on product-path commits.")
    parser.add_argument(
        "--mode",
        choices=("push", "range"),
        default="push",
        help="push=commits since upstream merge-base; range=use --rev-range",
    )
    parser.add_argument("--rev-range", default="", help="Git rev range, e.g. abc..HEAD")
    parser.add_argument(
        "--ci",
        action="store_true",
        help=(
            "CI mode (EQO-A PT-E02): the skip env variable is IGNORED, an empty or unreadable range is an error, and "
            "every Ref must point to a ticket tracked at HEAD. Default behaviour (without --ci) is unchanged."
        ),
    )
    args = parser.parse_args(argv)

    config = load_config()
    if _truthy_env(config.skip_env):
        if not args.ci:
            print(f"[WARN] Ticket ref gate skipped ({config.skip_env}).", file=sys.stderr)
            return 0
        print(f"[WARN] {config.skip_env} is set but IGNORED in --ci mode.", file=sys.stderr)

    rev_range = args.rev_range if args.mode == "range" else push_rev_range()
    if not rev_range:
        if args.ci:
            print(
                "[ERROR] ticket-ref --ci: no revision range given (use --mode range --rev-range <base>..HEAD).",
                file=sys.stderr,
            )
            return 1
        print("[INFO] ticket-ref: no commits to verify.")
        return 0

    try:
        errors = verify_range(rev_range, config, strict=args.ci)
    except RuntimeError as exc:
        print(f"[ERROR] ticket-ref --ci: {exc}", file=sys.stderr)
        return 1
    if errors:
        print("[ERROR] Ticket Ref verification failed:", file=sys.stderr)
        for err in errors:
            print(f"  - {err}", file=sys.stderr)
        return 1
    print(f"[INFO] ticket-ref: OK ({rev_range})")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

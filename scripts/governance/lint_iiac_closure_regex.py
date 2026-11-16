"""Lint for the IIAC activation gate closure regex (T-IGX-050-J).

The activation gate (``iiac_activation_gate.closure_conflicts``) rejects an
activation record or operator decision text that matches ``_CLOSURE``
(``EVOL-X|ENT-AUDIT-X|ARCH-FREEZE`` followed within 80 characters by ``CLOSED``).
The pattern is not negation-aware: "ARCH-FREEZE is not CLOSED" is flagged too.
This lint imports the pattern from the gate (it is never copied) and reports
every match in the given files, so wording that would break a later activation
record is found before the record is written.

Usage::

    python -m scripts.governance.lint_iiac_closure_regex [PATH ...]

With no path the lint scans ``IIAC_POLICY_ACTIVATION_RECORD.json`` (the whole
record, serialised the way the gate does) and
``docs/project/maintainers/qualification/IIAC_OD_ACTIVATION_*.md``.
Exit code: 0 no match, 1 at least one match, 2 usage or read error.

The lint reads files only; it does not edit any IIAC script or record.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import NamedTuple

from scripts.governance.iiac_activation_gate import _CLOSURE

REPO = Path(__file__).resolve().parents[2]
QUALIFICATION_DIR = Path("docs/project/maintainers/qualification")
DEFAULT_RECORD = QUALIFICATION_DIR / "IIAC_POLICY_ACTIVATION_RECORD.json"
DEFAULT_OD_GLOB = "IIAC_OD_ACTIVATION_*.md"


class Finding(NamedTuple):
    path: str
    line: int  # 1-based; 0 for a JSON record (no line, the gate scans one blob)
    matched: str
    context: str


def scan_text(text: str, path: str = "<text>") -> list[Finding]:
    """Return every ``_CLOSURE`` match, one line at a time (``.`` never crosses a newline)."""
    findings: list[Finding] = []
    for number, line in enumerate(text.splitlines(), start=1):
        for match in _CLOSURE.finditer(line):
            findings.append(Finding(path, number, match.group(0), line.strip()))
    return findings


def scan_json_record(text: str, path: str = "<json>") -> list[Finding]:
    """Scan a JSON record the way the gate does: ``json.dumps`` of the parsed record, one blob."""
    blob = json.dumps(json.loads(text), ensure_ascii=False)
    return [Finding(path, 0, match.group(0), match.group(0)) for match in _CLOSURE.finditer(blob)]


def scan_file(path: Path, display: str | None = None) -> list[Finding]:
    text = path.read_text(encoding="utf-8")
    name = display or str(path)
    if path.suffix.lower() == ".json":
        return scan_json_record(text, name)
    return scan_text(text, name)


def default_targets(repo: Path = REPO) -> list[Path]:
    targets = [repo / DEFAULT_RECORD]
    targets.extend(sorted((repo / QUALIFICATION_DIR).glob(DEFAULT_OD_GLOB)))
    return targets


def _format(finding: Finding) -> str:
    where = f"{finding.path}:{finding.line}" if finding.line else f"{finding.path} (record blob)"
    return f"{where}: closure regex match [{finding.matched}]"


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "paths",
        nargs="*",
        type=Path,
        help="files to scan (default: activation record and IIAC_OD_ACTIVATION_*.md)",
    )
    args = parser.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(errors="replace")

    targets = list(args.paths) if args.paths else default_targets()
    findings: list[Finding] = []
    for target in targets:
        try:
            findings.extend(scan_file(target))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            print(f"ERROR: cannot scan {target}: {exc}", file=sys.stderr)
            return 2
    for finding in findings:
        print(_format(finding))
    print(f"scanned {len(targets)} file(s), {len(findings)} match(es)")
    return 1 if findings else 0


if __name__ == "__main__":
    raise SystemExit(main())

"""Shared read-only helpers for the PT-E04 IIAC corpus, artifact and delegation verifiers.

Scope (ticket T-IGX-044-E, node PT-E04 / T-ENF-4): NEW files only. This module is not on
``MATERIAL_GOVERNANCE_PATHS`` of ``iiac_activation_gate.py`` and nothing in the activation
gate imports it. It only *reads* ``iiac_common`` (format validation), so the verifiers agree
with the existing validators on what a well-formed manifest is, and then add the checks the
existing validators lack.

Guarantees: stdlib only; never writes inside the repository; the only git commands issued are
the read-only ones listed in ``_READ_ONLY_GIT``.

Exit codes shared by every verifier: 0 = PASS, 1 = FAIL (at least one error finding),
2 = usage or environment error (bad argument, unreadable file, git not usable).
"""

from __future__ import annotations

import hashlib
import json
import re
import subprocess
import sys
from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:  # the verifiers also run as plain scripts and under ``python -I``
    sys.path.insert(0, str(REPO_ROOT))

from scripts.governance.iiac_common import validate_manifest_dict  # noqa: E402  (read-only reuse)

EXIT_PASS = 0
EXIT_FAIL = 1
EXIT_USAGE = 2

_READ_ONLY_GIT = frozenset({"cat-file", "merge-base", "rev-parse", "show", "log", "ls-tree", "ls-files"})

SHA_FIELDS = (
    "audited_implementation_sha",
    "test_execution_sha",
    "audit_execution_sha",
    "evidence_publication_sha",
    "certification_publication_sha",
)
DIGEST_BINDINGS = (
    ("audit_report_digest", "audit_report_path"),
    ("audit_scope_manifest_digest", "audit_scope_path"),
    ("test_matrix_digest", "test_matrix_path"),
    ("findings_digest", "findings_path"),
)
IDENTITY_FIELDS = (
    "implementation_session_id",
    "implementation_agent_id",
    "auditor_session_id",
    "auditor_agent_id",
    "certifier_session_id",
    "certifier_agent_id",
)
CERTIFYING_VERDICTS = frozenset({"PASS", "PASS_SCOPED"})
NON_CERTIFYING_VERDICTS = frozenset({"FAIL", "CONDITIONAL"})

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_DIGEST64 = re.compile(r"^[0-9a-f]{64}$")
_PLACEHOLDER = re.compile(r"(?i)(pending|todo|tbd|placeholder|unrecorded|change_?me)")
_ID_SPLIT = re.compile(r"[+,;|\s]+")

# Strict order a NEW final manifest must follow (each step: equal or ancestor).
STRICT_CHAIN = (
    ("audited_implementation_sha", "test_execution_sha"),
    ("test_execution_sha", "audit_execution_sha"),
    ("audit_execution_sha", "evidence_publication_sha"),
    ("evidence_publication_sha", "certification_publication_sha"),
)
# Relations that hold for all 39 published manifests at the survey head (see the plan, section 3).
CORE_CHAIN = (
    ("audited_implementation_sha", "audit_execution_sha"),
    ("test_execution_sha", "audit_execution_sha"),
)


@dataclass(frozen=True)
class Finding:
    code: str
    message: str
    severity: str = "error"  # "error" fails the verifier, "warn" is reported only

    def render(self) -> str:
        return f"{self.severity.upper()} {self.code}: {self.message}"


class GitUnavailableError(RuntimeError):
    """git itself could not be run against the requested repository (maps to exit 2)."""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def normalize_identity(value: str) -> str:
    return re.sub(r"\s+", " ", value).strip().casefold()


def is_zero_sha(value: object) -> bool:
    return isinstance(value, str) and bool(value) and set(value) == {"0"}


def is_sha40(value: object) -> bool:
    return isinstance(value, str) and bool(_SHA40.match(value))


def run_git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    if not args or args[0] not in _READ_ONLY_GIT:
        raise ValueError(f"git subcommand not allowed (read-only verifier): {args[:1]}")
    try:
        return subprocess.run(
            ["git", "-c", "core.autocrlf=false", *args],
            cwd=repo,
            capture_output=True,
            check=False,
        )
    except OSError as exc:  # git missing or repo path unusable
        raise GitUnavailableError(str(exc)) from exc


def ensure_git_repo(repo: Path) -> None:
    proc = run_git(repo, "rev-parse", "--git-dir")
    if proc.returncode != 0:
        raise GitUnavailableError(f"not a git repository: {repo}")


def commit_exists(repo: Path, sha: str) -> bool:
    return run_git(repo, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def is_ancestor_or_equal(repo: Path, ancestor: str, descendant: str) -> bool:
    if ancestor == descendant:
        return True
    return run_git(repo, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def show_blob(repo: Path, sha: str, path: str) -> bytes | None:
    proc = run_git(repo, "show", f"{sha}:{path}")
    return proc.stdout if proc.returncode == 0 else None


def load_json_object(path: Path) -> tuple[dict[str, Any] | None, str | None]:
    try:
        data = json.loads(path.read_text(encoding="utf-8-sig"))
    except (OSError, ValueError) as exc:
        return None, f"cannot read JSON {path}: {exc}"
    if not isinstance(data, dict):
        return None, f"JSON root is not an object: {path}"
    return data, None


def base_errors(manifest: dict[str, Any]) -> list[str]:
    """Errors from the EXISTING validator (``iiac_common.validate_manifest_dict``)."""
    errors: list[str] = []
    validate_manifest_dict(manifest, errors)
    return errors


# --------------------------------------------------------------------------- manifest checks


def check_roles(manifest: dict[str, Any], soft: str = "error") -> list[Finding]:
    """Role separation beyond raw string inequality (weakness: independence = three differing strings).

    Implementer-vs-auditor agent equality is always an error (parity with ACT-N10 in the activation gate);
    the other agent-label pairs follow ``soft`` because a role label shared with the certifier occurs in
    published history.
    """
    out: list[Finding] = []
    roles = {
        "implementer": str(manifest.get("implementation_session_id") or ""),
        "auditor": str(manifest.get("auditor_session_id") or ""),
        "certifier": str(manifest.get("certifier_session_id") or ""),
    }
    names = tuple(roles)
    for i, left in enumerate(names):
        for right in names[i + 1 :]:
            a, b = roles[left], roles[right]
            if not a or not b:
                continue
            if a != b and normalize_identity(a) == normalize_identity(b):
                out.append(
                    Finding(
                        "ROLE_EQUAL_NORMALIZED",
                        f"{left} and {right} session ids differ only by case or whitespace: {a!r} / {b!r}",
                    )
                )
            parts_a = {t.casefold() for t in _ID_SPLIT.split(a) if len(t) >= 4}
            parts_b = {t.casefold() for t in _ID_SPLIT.split(b) if len(t) >= 4}
            common = sorted(parts_a & parts_b)
            if a != b and common:
                out.append(
                    Finding(
                        "ROLE_COMPONENT_OVERLAP",
                        f"{left} and {right} session ids share the component {common[0]!r}",
                    )
                )
    for left, right, field_l, field_r, level in (
        ("implementer", "auditor", "implementation_agent_id", "auditor_agent_id", "error"),
        ("implementer", "certifier", "implementation_agent_id", "certifier_agent_id", soft),
        ("auditor", "certifier", "auditor_agent_id", "certifier_agent_id", soft),
    ):
        a = str(manifest.get(field_l) or "")
        b = str(manifest.get(field_r) or "")
        if a and b and normalize_identity(a) == normalize_identity(b):
            out.append(Finding("ROLE_AGENT_EQUAL", f"{left} and {right} agent ids are equal: {a!r}", level))
    return out


def check_placeholders(manifest: dict[str, Any], severity: str) -> list[Finding]:
    out: list[Finding] = []
    for field in SHA_FIELDS:
        value = manifest.get(field)
        if is_zero_sha(value):
            out.append(Finding("PLACEHOLDER", f"{field} is an all-zero SHA", severity))
    for field in (*IDENTITY_FIELDS, *(d for d, _ in DIGEST_BINDINGS), *SHA_FIELDS):
        value = manifest.get(field)
        if isinstance(value, str) and _PLACEHOLDER.search(value):
            out.append(Finding("PLACEHOLDER", f"{field} carries a placeholder marker: {value!r}", severity))
    return out


def _sha_present(manifest: dict[str, Any], field: str) -> str | None:
    value = manifest.get(field)
    if is_sha40(value) and not is_zero_sha(value):
        return str(value)
    return None


def check_sha_objects(manifest: dict[str, Any], repo: Path) -> list[Finding]:
    out: list[Finding] = []
    for field in SHA_FIELDS:
        sha = _sha_present(manifest, field)
        if sha is not None and not commit_exists(repo, sha):
            out.append(Finding("SHA_MISSING", f"{field} {sha[:12]} is not a commit object in the repository"))
    return out


def check_order(manifest: dict[str, Any], repo: Path, chain: Iterable[tuple[str, str]], severity: str) -> list[Finding]:
    out: list[Finding] = []
    for first, second in chain:
        a, b = _sha_present(manifest, first), _sha_present(manifest, second)
        if a is None or b is None or not commit_exists(repo, a) or not commit_exists(repo, b):
            continue  # missing or placeholder values are reported by their own checks
        if not is_ancestor_or_equal(repo, a, b):
            out.append(Finding("SHA_ORDER", f"{first} is not an ancestor of (or equal to) {second}", severity))
    return out


def resolve_artifact_ref(manifest: dict[str, Any], at: str | None) -> str | None:
    if at:
        return at
    for field in ("evidence_publication_sha", "audit_execution_sha"):
        sha = _sha_present(manifest, field)
        if sha is not None:
            return sha
    return None


def check_digests(
    manifest: dict[str, Any],
    repo: Path,
    *,
    at: str | None,
    overrides: dict[str, str],
    severity: str,
) -> list[Finding]:
    """Bind every digest field to the bytes of the file it names (weakness: digests only length-checked)."""
    out: list[Finding] = []
    ref = resolve_artifact_ref(manifest, at)
    for digest_field, path_field in DIGEST_BINDINGS:
        digest = manifest.get(digest_field)
        if not isinstance(digest, str) or not _DIGEST64.match(digest):
            out.append(Finding("DIGEST_FORMAT", f"{digest_field} is not a 64-hex sha256", severity))
            continue
        path = overrides.get(digest_field) or manifest.get(path_field)
        if not isinstance(path, str) or not path.strip():
            out.append(
                Finding("DIGEST_UNBOUND", f"{digest_field} has no artifact path ({path_field} missing)", severity)
            )
            continue
        if ref is None:
            out.append(Finding("DIGEST_UNBOUND", f"{digest_field}: no commit to read {path} at", severity))
            continue
        blob = show_blob(repo, ref, path.replace("\\", "/"))
        if blob is None:
            out.append(
                Finding("DIGEST_ARTIFACT_MISSING", f"{path} not found at {ref[:12]} for {digest_field}", severity)
            )
        elif sha256_bytes(blob) != digest:
            out.append(Finding("DIGEST_MISMATCH", f"{digest_field} does not match {path} at {ref[:12]}", severity))
    return out


def check_manifest_artifacts(
    manifest: dict[str, Any],
    repo: Path,
    *,
    profile: str,
    at: str | None = None,
    overrides: dict[str, str] | None = None,
) -> list[Finding]:
    """Run every artifact check. ``profile`` is ``final`` (new manifests, all strict) or ``legacy``.

    ``legacy`` keeps only what holds for the published corpus as errors (format, missing commit
    objects, core SHA order, role overlap) and turns placeholders, non-certifying verdicts, strict order
    and unbound or mismatching digests into warnings, so historical drafts are reported but not rewritten.
    """
    if profile not in {"final", "legacy"}:
        raise ValueError(f"unknown profile {profile!r}")
    soft = "error" if profile == "final" else "warn"
    findings = [Finding("MANIFEST_FORMAT", err) for err in base_errors(manifest)]
    verdict = str(manifest.get("verdict") or "").upper()
    if verdict not in CERTIFYING_VERDICTS:
        findings.append(
            Finding("VERDICT_NOT_CERTIFYING", f"verdict {verdict or 'missing'} cannot back a certification", soft)
        )
    findings += check_roles(manifest, soft)
    findings += check_placeholders(manifest, soft)
    findings += check_sha_objects(manifest, repo)
    findings += check_order(manifest, repo, CORE_CHAIN, "error")
    findings += check_order(manifest, repo, STRICT_CHAIN, soft)
    findings += check_digests(manifest, repo, at=at, overrides=overrides or {}, severity=soft)
    return _dedupe(findings)


def _dedupe(findings: list[Finding]) -> list[Finding]:
    seen: set[tuple[str, str, str]] = set()
    out: list[Finding] = []
    for item in findings:
        key = (item.code, item.message, item.severity)
        if key not in seen:
            seen.add(key)
            out.append(item)
    return out


def errors_of(findings: Iterable[Finding]) -> list[Finding]:
    return [f for f in findings if f.severity == "error"]


def report(tool: str, findings: list[Finding], *, stream: Any = None) -> int:
    """Print findings and the verdict line; return the exact exit code."""
    out = stream if stream is not None else sys.stderr
    errors = errors_of(findings)
    for item in findings:
        print(f"  {item.render()}", file=out)
    warns = len(findings) - len(errors)
    if errors:
        print(f"{tool}: FAIL ({len(errors)} error(s), {warns} warning(s))", file=out)
        return EXIT_FAIL
    print(f"{tool}: PASS ({warns} warning(s))")
    return EXIT_PASS

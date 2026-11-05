#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""EVOL-X closure publication preflight — two-gate IIAC contract (B-4 corrective, T-IGX-017-A).

The preflight is fail-closed and evaluates two separate gates:

Gate A — B6 acceptance readiness
    The published independent IIAC audit and certification evidence is valid for a
    B6 acceptance under the policy that is genuinely ``IN_FORCE``. Gate A does not
    require ``B6_ACCEPTANCE_PUBLICATION_SHA``: that SHA cannot exist before B6 is
    published.

Gate B — formal EVOL-X closure readiness
    Gate A holds, B6 acceptance is published and bound (two-phase, non-self-referential),
    every FRZ-EVOL row has an individual disposition no wider than the certified scope,
    the applicable operator decisions are recorded, the SHA ladder is valid and no
    mandatory blocker is open. Gate B does not require ``FORMAL_CLOSURE_PUBLICATION_SHA``
    (it cannot exist before the closure commit) but validates it when present.

Gate A PASS never implies Gate B PASS. Legacy human Protocol 03 requirements apply only
while the IIAC policy is not in force; a policy record that claims ``IN_FORCE`` but fails
the activation gate fails both gates (no legacy fallback).

Verification is delegated to the canonical IIAC routines (activation gate, manifest
validator, role-separation validator). This module adds only EVOL-X stage bindings.
It never writes to the repository.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

_REPO = Path(__file__).resolve().parents[2]
if str(_REPO) not in sys.path:
    sys.path.insert(0, str(_REPO))

from scripts.governance.iiac_activation_gate import (  # noqa: E402
    CANONICAL_RECORD,
    POLICY_VERSION,
    _companion_errors,
    _findings_errors,
    _path_rejected,
    evaluate_activation,
)
from scripts.governance.iiac_common import validate_manifest_dict  # noqa: E402
from scripts.qualification.evol_x_frz_evol_canonical_matrix import (  # noqa: E402
    FRZ_EVOL_CANONICAL_REQUIREMENTS,
    validate_frz_evol_id_set,
)

STAGE_ID = "EVOL-X"
QUAL = "docs/project/maintainers/qualification"
EVOL_DIR = f"{QUAL}/evol_x"
AUDIT_DIR = f"{QUAL}/iiac_evidence/EVOL-X/audit"
CERT_DIR = f"{QUAL}/iiac_evidence/EVOL-X/certification"

REL_ACTIVATION = f"{QUAL}/IIAC_POLICY_ACTIVATION_RECORD.json"
REL_MANIFEST = f"{CERT_DIR}/iiac_audit_session_manifest.EVOL-X.json"
REL_CERT_RECORD = f"{CERT_DIR}/EVOL_X_IIAC_CERTIFICATION_RECORD.json"
REL_CERT_INDEX = f"{CERT_DIR}/CERTIFICATION_EVIDENCE_INDEX.json"
REL_CERT_HANDOFF = f"{CERT_DIR}/IIAC_SESSION_HANDOFF_RECORD_EVOL_X_CERTIFICATION.md"
REL_AUDIT_INDEX = f"{AUDIT_DIR}/EVIDENCE_INDEX.json"
REL_AUDIT_REPORT = f"{AUDIT_DIR}/IIAC_EVOL_X_INDEPENDENT_AUDIT_DOSSIER.md"
REL_AUDIT_SCOPE = f"{AUDIT_DIR}/audit_scope.json"
REL_TEST_MATRIX = f"{AUDIT_DIR}/test_matrix.txt"
REL_FINDINGS = f"{AUDIT_DIR}/EVOL-X_audit_findings.json"
REL_AUDITOR_DRAFT = f"{AUDIT_DIR}/iiac_audit_session_manifest.EVOL-X.auditor_draft.json"
REL_AUDIT_PASS_COPY = f"{AUDIT_DIR}/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
REL_PASS_RECORD = f"{EVOL_DIR}/EVOL_X_IIAC_INDEPENDENT_AUDIT_PASS_RECORD.md"
REL_GOVERNANCE = f"{EVOL_DIR}/EVOL_X_AUTO_B6_GOVERNANCE_STATUS.json"
REL_LADDER = f"{EVOL_DIR}/EVOL_X_SHA_LADDER.md"
REL_REGISTER = f"{EVOL_DIR}/EVOL_X_OPERATOR_DECISION_REGISTER_W1.md"
REL_FRZ_CHECKLIST = f"{QUAL}/PLATFORM_ENTERPRISE_FREEZE_ACCEPTANCE_CHECKLIST.md"
# Legacy (policy not in force) human Protocol 03 artifacts — never IIAC authority.
REL_LEGACY_B6_PASS = f"{EVOL_DIR}/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_PASS_RECORD.md"
REL_LEGACY_B6_DRAFT = f"{EVOL_DIR}/b6_technical_qualification/EVOL_X_PROTOCOL_03_INDEPENDENT_AUDIT_RECORD_DRAFT.md"

AUDITOR_AGENT_ID = "independent-quality-auditor"
CERTIFIER_AGENT_ID = "certification-governance-agent"
ACCEPTABLE_VERDICTS = frozenset({"PASS", "PASS_SCOPED"})
BLOCKING_FINDING_CLASSES = frozenset({"PROVEN_ARCHITECTURAL_DEFECT", "IN_SCOPE_BLOCKER"})
BLOCKING_FINDING_SEVERITIES = frozenset({"BLOCKER", "CRITICAL", "HIGH"})
RESOLVED_DISPOSITIONS = frozenset({"RESOLVED", "CLOSED"})

# Audited EVOL-X scope (audit_scope.json ``in_scope``) and the SHA each prefix was audited at.
# Any change below these prefixes after that SHA makes the independent audit stale.
EVOL_X_AUDITED_SCOPE: tuple[tuple[str, str], ...] = (
    ("testing_support/change_safety/", "audited_implementation_sha"),
    ("tests/qualification/evol_x/", "audited_implementation_sha"),
    ("tests/fixtures/evol_x/", "audited_implementation_sha"),
    ("scripts/qualification/evol_x_p03_evidence.py", "audited_implementation_sha"),
    ("scripts/qualification/evol_x_vl3_manifest.py", "audited_implementation_sha"),
    (f"{EVOL_DIR}/evol_x_p03_independent_replay/", "evidence_publication_sha"),
)

# Session C scope restrictions: each limitation id must be present in the certified manifest
# and must keep these phrases. Removing or rewording them widens the accepted scope.
SCOPE_RESTRICTIONS: dict[str, tuple[str, ...]] = {
    "LIM-01": ("EVOL_CONSERVATIVE_FULL only",),
    "LIM-02": ("M-09H NOT_MEASURED", "tool-side attribution only"),
    "LIM-03": ("direct-import only",),
    "LIM-04": ("self-surface arcs",),
    "LIM-05": ("qualification-layer only", "rollback is not gated"),
}

# Certification-record FRZ dispositions that must keep their scoped wording.
FRZ_DISPOSITION_RESTRICTIONS: dict[str, tuple[str, ...]] = {
    "FRZ-EVOL-01": ("LIM-03",),
    "FRZ-EVOL-06": ("LIM-01", "LIM-03", "LIM-04", "EVOL_CONSERVATIVE_FULL scope only"),
    "FRZ-EVOL-07": ("LIM-05", "NOT claimed"),
    "FRZ-EVOL-08": ("LIM-02", "M-09H not measured"),
}

# Checklist-row wording that must travel with a promoted FRZ row (case-insensitive).
FRZ_ROW_RESTRICTIONS: dict[str, tuple[str, ...]] = {
    "FRZ-EVOL-06": ("EVOL_CONSERVATIVE_FULL scope only",),
    "FRZ-EVOL-07": ("qualification-layer", "not claimed"),
    "FRZ-EVOL-08": ("M-09H", "not measured"),
}

# Open certification blockers that only gate formal closure (B-5: OD-EVOL-12 / OD-EVOLX-08).
GATE_B_ONLY_BLOCKERS = frozenset({"B-5"})
# Operator decisions applicable to formal EVOL-X closure. OD-EVOL-11 (named human auditor)
# is migrated to ``auditor_session_id`` + handoff record and is not an IIAC requirement.
CLOSURE_OPERATOR_DECISIONS = ("OD-EVOL-12", "OD-EVOLX-08", "OD-EVOL-05b", "OD-EVOL-10", "OD-EVOL-13")
LEGACY_LADDER_ROWS = (
    "INTERNAL_B6_ACCEPTANCE_PUBLICATION_SHA",
    "INTERNAL_PARENT_CLOSURE_PUBLICATION_SHA",
    "INTERNAL_CLOSURE_PUBLICATION_SHA",
    "INTERNAL_CERTIFICATION_PUBLICATION_SHA",
    "PROTOCOL_03_AUTOMATED_AUDIT_PUBLICATION_SHA",
)

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_DIGEST64 = re.compile(r"^[0-9a-f]{64}$")
_LADDER_SHA = re.compile(r"`([0-9a-f]{40})`")
_LIM_ID = re.compile(r"^(LIM-\d{2}|F-C-\d{2})\b")
_BLOCKER_ID = re.compile(r"^(B-\d+)\b")
_HANDOFF_SIGNATURE = re.compile(
    r"^`?Approved handoff: operator @ (\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2}))`?",
    re.MULTILINE,
)


@dataclass
class Context:
    repo: Path
    activation_record: Path
    evidence_root: Path | None
    head: str = ""
    cache: dict[str, Any] = field(default_factory=dict)

    def path(self, rel: str) -> Path:
        return self.repo / rel


@dataclass
class PreflightResult:
    mode: str
    gate_a: list[str]
    gate_b: list[str]

    @property
    def gate_a_pass(self) -> bool:
        return not self.gate_a

    @property
    def gate_b_pass(self) -> bool:
        return not self.gate_a and not self.gate_b


# --------------------------------------------------------------------------- helpers


def _sha256(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(ctx: Context, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(
        ["git", "-c", "core.autocrlf=false", *args],
        cwd=ctx.repo,
        capture_output=True,
        check=False,
    )


def _commit_exists(ctx: Context, sha: str) -> bool:
    return _git(ctx, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def _is_ancestor(ctx: Context, ancestor: str, descendant: str) -> bool:
    return _git(ctx, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def _show(ctx: Context, sha: str, rel: str) -> bytes | None:
    proc = _git(ctx, "show", f"{sha}:{rel}")
    return proc.stdout if proc.returncode == 0 else None


def _tree_blobs(ctx: Context, sha: str, rel_dir: str) -> dict[str, str]:
    proc = _git(ctx, "ls-tree", "-r", sha, "--", rel_dir)
    blobs: dict[str, str] = {}
    if proc.returncode != 0:
        return blobs
    for line in proc.stdout.decode("utf-8", errors="replace").splitlines():
        meta, _, path = line.partition("\t")
        parts = meta.split()
        if len(parts) == 3 and parts[1] == "blob":
            blobs[path] = parts[2]
    return blobs


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()


def _changed_paths(ctx: Context, base: str, head: str, rel: str) -> list[str]:
    proc = _git(ctx, "diff", "--name-only", base, head, "--", rel)
    if proc.returncode != 0:
        return [f"<git diff failed: {proc.stderr.decode('utf-8', errors='replace').strip()}>"]
    return [line.strip() for line in proc.stdout.decode("utf-8").splitlines() if line.strip()]


def _read_bytes(ctx: Context, rel: str) -> bytes | None:
    path = ctx.path(rel)
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _read_json(ctx: Context, rel: str, code: str, errors: list[str]) -> dict[str, Any] | None:
    key = f"json:{rel}"
    if key in ctx.cache:
        return ctx.cache[key]
    raw = _read_bytes(ctx, rel)
    if raw is None:
        errors.append(f"{code} missing {rel}")
        ctx.cache[key] = None
        return None
    try:
        loaded = json.loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        errors.append(f"{code} malformed JSON {rel}: {exc}")
        loaded = None
    if loaded is not None and not isinstance(loaded, dict):
        errors.append(f"{code} {rel} must be a JSON object")
        loaded = None
    ctx.cache[key] = loaded
    return loaded


def _read_text(ctx: Context, rel: str) -> str | None:
    raw = _read_bytes(ctx, rel)
    return None if raw is None else raw.decode("utf-8-sig", errors="replace")


def _sha(value: Any) -> str:
    text = str(value or "").strip()
    return text if _SHA40.fullmatch(text) else ""


def _norm(cell: str) -> str:
    return cell.replace("*", "").replace("`", "").strip()


def _table_cells(line: str) -> list[str]:
    return [c.strip() for c in line.strip().strip("|").split("|")]


def parse_ladder_active(text: str) -> dict[str, str]:
    """Active SSOT table of the SHA ladder (before the first ``##`` section)."""
    rows: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("## "):
            break
        if not line.startswith("|"):
            continue
        cells = _table_cells(line)
        if len(cells) < 2:
            continue
        role = _norm(cells[0])
        if not role or role == "Role" or set(role) <= {"-", ":"}:
            continue
        rows.setdefault(role, cells[1])
    return rows


def ladder_sha(rows: dict[str, str], role: str) -> str:
    match = _LADDER_SHA.search(rows.get(role, ""))
    return match.group(1) if match else ""


def parse_register(text: str) -> dict[str, tuple[str, str]]:
    """First ``| **OD-…** | status | date |`` row per decision id."""
    rows: dict[str, tuple[str, str]] = {}
    for line in text.splitlines():
        if not line.startswith("| **OD-"):
            continue
        cells = _table_cells(line)
        if len(cells) < 3:
            continue
        rows.setdefault(_norm(cells[0]), (_norm(cells[1]), _norm(cells[2])))
    return rows


def decision_recorded(status: str, date: str) -> bool:
    upper = status.upper()
    if not status or "PENDING" in upper or "NOT SIGNED" in upper or "DRAFT" in upper:
        return False
    if not re.search(r"\b(SIGNED|APPROVED|CONFIRMED)\b", upper):
        return False
    return re.fullmatch(r"\d{4}-\d{2}-\d{2}", date.strip()) is not None


def parse_frz_rows(text: str) -> dict[str, list[list[str]]]:
    rows: dict[str, list[list[str]]] = {}
    for line in text.splitlines():
        if line.startswith("| FRZ-EVOL-"):
            cells = _table_cells(line)
            rows.setdefault(cells[0], []).append(cells)
    return rows


def parse_auditor_frz(text: str) -> dict[str, str]:
    result: dict[str, str] = {}
    for line in text.splitlines():
        if line.startswith("| FRZ-EVOL-"):
            cells = _table_cells(line)
            if len(cells) >= 2:
                result[cells[0]] = cells[1]
    return result


def _limitation_map(items: Any) -> dict[str, str]:
    out: dict[str, str] = {}
    if not isinstance(items, list):
        return out
    for item in items:
        if isinstance(item, str):
            match = _LIM_ID.match(item.strip())
            if match:
                out.setdefault(match.group(1), item.strip())
    return out


# --------------------------------------------------------------------------- policy


def policy_mode(ctx: Context) -> tuple[str, list[str]]:
    """Return (mode, errors). Mode is ``IIAC``, ``LEGACY`` or ``INVALID``."""
    if not ctx.activation_record.is_file():
        return "INVALID", ["A-01 IIAC activation record is missing"]
    try:
        record = json.loads(ctx.activation_record.read_text(encoding="utf-8"))
    except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
        return "INVALID", [f"A-01 malformed IIAC activation record: {exc}"]
    if not isinstance(record, dict):
        return "INVALID", ["A-01 IIAC activation record must be a JSON object"]
    ctx.cache["activation"] = record
    status = str(record.get("status") or "").upper()
    if status != "IN_FORCE":
        return "LEGACY", []
    errors = [
        f"A-01 IIAC policy claims IN_FORCE but activation gate fails: {err}"
        for err in evaluate_activation(
            record,
            repo=ctx.repo,
            record_path=ctx.activation_record,
            evidence_root=ctx.evidence_root,
            require_in_force=True,
        )
    ]
    if str(record.get("policy_version") or "") != POLICY_VERSION:
        errors.append(f"A-01 activation policy_version {record.get('policy_version')!r} != {POLICY_VERSION}")
    return ("IIAC" if not errors else "INVALID"), errors


# --------------------------------------------------------------------------- Gate A


def _gate_a_manifest(ctx: Context, errors: list[str]) -> dict[str, Any] | None:
    manifest = _read_json(ctx, REL_MANIFEST, "A-03", errors)
    if manifest is None:
        return None
    schema_errors: list[str] = []
    validate_manifest_dict(manifest, schema_errors)
    errors.extend(f"A-04 manifest: {err}" for err in schema_errors)
    errors.extend(f"A-04 {err}" for err in _companion_errors(ctx.repo, ctx.path(REL_MANIFEST)))

    if manifest.get("stage_id") != STAGE_ID:
        errors.append(f"A-05 manifest stage_id {manifest.get('stage_id')!r} != {STAGE_ID}")
    if manifest.get("certification_profile") != "iiac-v1":
        errors.append("A-05 manifest certification_profile must be iiac-v1")
    if manifest.get("policy_version") != POLICY_VERSION:
        errors.append(f"A-05 manifest policy_version {manifest.get('policy_version')!r} != {POLICY_VERSION}")
    activation = ctx.cache.get("activation") or {}
    if activation.get("policy_version") and manifest.get("policy_version") != activation.get("policy_version"):
        errors.append("A-05 manifest policy_version differs from the policy in force")
    if manifest.get("assurance_level") != "process_enforced":
        errors.append("A-05 manifest assurance_level must be process_enforced")

    verdict = str(manifest.get("verdict") or "").upper()
    if verdict not in ACCEPTABLE_VERDICTS:
        errors.append(f"A-06 manifest verdict {verdict or 'missing'} is not B6 authority")
    if verdict == "PASS_SCOPED" and not manifest.get("limitations"):
        errors.append("A-06 PASS_SCOPED manifest without limitations")

    canonical_paths = {
        "audit_report_path": REL_AUDIT_REPORT,
        "pass_record_path": REL_PASS_RECORD,
        "evidence_index_path": REL_AUDIT_INDEX,
        "handoff_record_path": REL_CERT_HANDOFF,
    }
    for key, expected in canonical_paths.items():
        raw = str(manifest.get(key) or "")
        reason = _path_rejected(raw) if raw else "missing"
        if reason:
            errors.append(f"A-14 manifest {key}: {reason}")
        elif raw.replace("\\", "/") != expected:
            errors.append(f"A-14 manifest {key} {raw!r} is not the canonical IIAC artifact {expected}")
    return manifest


def _gate_a_scope(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any] | None, errors: list[str]) -> None:
    draft = _read_json(ctx, REL_AUDITOR_DRAFT, "A-07", errors)
    verdict = str(manifest.get("verdict") or "").upper()
    if draft is not None:
        draft_verdict = str(draft.get("verdict") or "").upper()
        if verdict != draft_verdict:
            errors.append(f"A-07 manifest verdict {verdict} differs from the auditor verdict {draft_verdict}")
        for key in (
            "stage_id",
            "auditor_session_id",
            "auditor_agent_id",
            "audited_implementation_sha",
            "test_execution_sha",
            "evidence_publication_sha",
            "audit_execution_sha",
            "audit_report_digest",
            "audit_scope_manifest_digest",
            "test_matrix_digest",
            "findings_digest",
        ):
            if manifest.get(key) != draft.get(key):
                errors.append(f"A-07 manifest {key} differs from the auditor-authored manifest")
        certified = _limitation_map(manifest.get("limitations"))
        for lim_id, text in _limitation_map(draft.get("limitations")).items():
            got = certified.get(lim_id)
            if got is None:
                errors.append(f"A-07 auditor limitation {lim_id} dropped from the certified manifest")
            elif not got.startswith(text):
                errors.append(f"A-07 auditor limitation {lim_id} reworded in the certified manifest")
    certified = _limitation_map(manifest.get("limitations"))
    for lim_id, phrases in SCOPE_RESTRICTIONS.items():
        text = certified.get(lim_id)
        if text is None:
            errors.append(f"A-08 scope restriction {lim_id} missing from the manifest")
            continue
        for phrase in phrases:
            if phrase not in text:
                errors.append(f"A-08 scope restriction {lim_id} lost {phrase!r}")

    pass_text = _read_text(ctx, REL_PASS_RECORD) or ""
    auditor_frz = parse_auditor_frz(pass_text)
    errors.extend(f"A-08 pass record FRZ table: {p}" for p in validate_frz_evol_id_set(set(auditor_frz)))
    if cert is None:
        return
    dispositions = cert.get("frz_evol_dispositions")
    if not isinstance(dispositions, dict):
        errors.append("A-08 certification record has no frz_evol_dispositions")
        return
    errors.extend(f"A-08 certification dispositions: {p}" for p in validate_frz_evol_id_set(set(dispositions)))
    for frz_id in FRZ_EVOL_CANONICAL_REQUIREMENTS:
        disp = str(dispositions.get(frz_id) or "")
        if not disp.startswith("ELIGIBLE_"):
            errors.append(f"A-08 {frz_id} disposition {disp!r} is not an eligibility disposition")
        if "SCOPED" in auditor_frz.get(frz_id, "").upper() and "SCOPED" not in disp.upper():
            errors.append(f"A-08 {frz_id} widened from auditor SCOPED result to {disp!r}")
        for phrase in FRZ_DISPOSITION_RESTRICTIONS.get(frz_id, ()):
            if phrase not in disp:
                errors.append(f"A-08 {frz_id} disposition lost {phrase!r}")


def _gate_a_roles(manifest: dict[str, Any], cert: dict[str, Any] | None, errors: list[str]) -> None:
    sessions = {k: str(manifest.get(k) or "") for k in ("implementation_session_id", "auditor_session_id")}
    sessions["certifier_session_id"] = str(manifest.get("certifier_session_id") or "")
    agents = {k: str(manifest.get(k) or "") for k in ("implementation_agent_id", "auditor_agent_id")}
    agents["certifier_agent_id"] = str(manifest.get("certifier_agent_id") or "")
    if len({v for v in sessions.values() if v}) != 3:
        errors.append("A-09 implementer, auditor and certifier sessions must be three distinct sessions")
    if len({v for v in agents.values() if v}) != 3:
        errors.append("A-09 implementer, auditor and certifier agent ids must be distinct (self-certification)")
    if agents["auditor_agent_id"] != AUDITOR_AGENT_ID:
        errors.append(f"A-09 auditor_agent_id must be {AUDITOR_AGENT_ID}")
    if agents["certifier_agent_id"] != CERTIFIER_AGENT_ID:
        errors.append(f"A-09 certifier_agent_id must be {CERTIFIER_AGENT_ID}")
    if cert is not None:
        for key in ("certifier_session_id", "certifier_agent_id", "implementation_session_id"):
            if cert.get(key) != manifest.get(key):
                errors.append(f"A-09 certification record {key} differs from the manifest")


def _gate_a_shas(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any] | None, errors: list[str]) -> None:
    fields = (
        "audited_implementation_sha",
        "test_execution_sha",
        "evidence_publication_sha",
        "audit_execution_sha",
        "audit_session_head_sha",
        "audit_publication_sha",
        "certification_publication_sha",
    )
    shas: dict[str, str] = {}
    for key in fields:
        value = _sha(manifest.get(key))
        if not value:
            errors.append(f"A-10 manifest {key} must be a 40-char lowercase SHA")
        elif not _commit_exists(ctx, value):
            errors.append(f"A-10 manifest {key} {value} is not a commit in this repository")
        else:
            shas[key] = value
    chain = (
        ("audited_implementation_sha", "test_execution_sha"),
        ("test_execution_sha", "evidence_publication_sha"),
        ("audited_implementation_sha", "audit_execution_sha"),
        ("evidence_publication_sha", "audit_publication_sha"),
        ("audit_session_head_sha", "audit_publication_sha"),
        ("audit_publication_sha", "certification_publication_sha"),
    )
    for older, newer in chain:
        if older in shas and newer in shas and not _is_ancestor(ctx, shas[older], shas[newer]):
            errors.append(f"A-10 {older} is not an ancestor of {newer}")
    cert_sha = shas.get("certification_publication_sha", "")
    if cert_sha:
        if ctx.head and not _is_ancestor(ctx, cert_sha, ctx.head):
            errors.append("A-10 certification_publication_sha is not an ancestor of HEAD")
        if cert_sha == ctx.head:
            errors.append("A-10 self-referential certification_publication_sha equals HEAD")
        if cert_sha in {shas.get("audited_implementation_sha"), shas.get("audit_publication_sha")}:
            errors.append("A-10 certification_publication_sha must be a separate certifier publication")
    if cert is not None:
        for key in fields:
            if key in cert and cert.get(key) != manifest.get(key):
                errors.append(f"A-10 certification record {key} differs from the manifest")

    ladder_text = _read_text(ctx, REL_LADDER)
    if ladder_text is None:
        errors.append(f"A-11 missing {REL_LADDER}")
    else:
        rows = parse_ladder_active(ladder_text)
        if ladder_sha(rows, "TEST_EXECUTION_SHA") != manifest.get("test_execution_sha"):
            errors.append("A-11 stale audit: ladder TEST_EXECUTION_SHA differs from the audited test_execution_sha")
        if ladder_sha(rows, "EVIDENCE_FINAL_PUBLICATION_SHA") != manifest.get("evidence_publication_sha"):
            errors.append("A-11 stale audit: ladder EVIDENCE_FINAL_PUBLICATION_SHA differs from the audited evidence")
    if ctx.head:
        for prefix, base_key in EVOL_X_AUDITED_SCOPE:
            base = shas.get(base_key)
            if not base:
                continue
            changed = _changed_paths(ctx, base, ctx.head, prefix)
            if changed:
                errors.append(f"A-11 stale audit: audited scope changed after {base_key}: {changed[0]}")


def _gate_a_digests(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any] | None, errors: list[str]) -> None:
    bound = (
        ("audit_report_digest", REL_AUDIT_REPORT),
        ("audit_scope_manifest_digest", REL_AUDIT_SCOPE),
        ("test_matrix_digest", REL_TEST_MATRIX),
        ("findings_digest", REL_FINDINGS),
    )
    for key, rel in bound:
        recorded = str(manifest.get(key) or "")
        raw = _read_bytes(ctx, rel)
        if not _DIGEST64.fullmatch(recorded):
            errors.append(f"A-12 manifest {key} is not a SHA-256 digest")
        elif raw is None:
            errors.append(f"A-12 missing bytes for {key}: {rel}")
        elif _sha256(raw) != recorded:
            errors.append(f"A-12 {key} does not match {rel}")
    if cert is not None:
        if cert.get("accepted_audit_report_digest") != manifest.get("audit_report_digest"):
            errors.append("A-12 certification record accepted_audit_report_digest differs from the manifest")
        if cert.get("findings_digest") != manifest.get("findings_digest"):
            errors.append("A-12 certification record findings_digest differs from the manifest")

    audit_pub = _sha(manifest.get("audit_publication_sha"))
    _verify_index(ctx, REL_AUDIT_INDEX, AUDIT_DIR, audit_pub, errors)
    _verify_index(ctx, REL_CERT_INDEX, CERT_DIR, ctx.head, errors)
    index_raw = _read_bytes(ctx, REL_AUDIT_INDEX)
    if cert is not None and index_raw is not None and cert.get("audit_evidence_index_digest") != _sha256(index_raw):
        errors.append("A-13 certification record audit_evidence_index_digest differs from EVIDENCE_INDEX.json")


def _verify_index(ctx: Context, rel_index: str, rel_dir: str, committed_at: str, errors: list[str]) -> None:
    index = _read_json(ctx, rel_index, "A-13", errors)
    if index is None:
        return
    files = index.get("files")
    if index.get("stage_id") != STAGE_ID or not isinstance(files, dict) or not files:
        errors.append(f"A-13 {rel_index} is not an {STAGE_ID} evidence index")
        return
    root = ctx.path(rel_dir)
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} if root.is_dir() else set()
    index_name = Path(rel_index).name
    unindexed = sorted(on_disk - set(files) - {index_name})
    if unindexed:
        errors.append(f"A-13 unindexed evidence in {rel_dir}: {unindexed[0]}")
    blobs = _tree_blobs(ctx, committed_at, rel_dir) if committed_at else {}
    if not blobs:
        errors.append(f"A-13 no committed evidence for {rel_dir} at {committed_at or 'unknown SHA'}")
    for name, meta in files.items():
        reason = _path_rejected(name)
        if reason:
            errors.append(f"A-13 index entry {name!r}: {reason}")
            continue
        raw = _read_bytes(ctx, f"{rel_dir}/{name}")
        expected = meta.get("sha256") if isinstance(meta, dict) else None
        if raw is None:
            errors.append(f"A-13 indexed evidence missing: {rel_dir}/{name}")
            continue
        if _sha256(raw) != expected or (isinstance(meta, dict) and meta.get("bytes") != len(raw)):
            errors.append(f"A-13 digest mismatch for {rel_dir}/{name}")
        if blobs and blobs.get(f"{rel_dir}/{name}") != _git_blob_id(raw):
            errors.append(f"A-13 {rel_dir}/{name} differs from the bytes committed at {committed_at[:12]}")
    index_blob = blobs.get(rel_index)
    raw_index = _read_bytes(ctx, rel_index)
    if blobs and raw_index is not None and index_blob != _git_blob_id(raw_index):
        errors.append(f"A-13 {rel_index} differs from the bytes committed at {committed_at[:12]}")


def _gate_a_pass_record(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any] | None, errors: list[str]) -> None:
    raw = _read_bytes(ctx, REL_PASS_RECORD)
    copy = _read_bytes(ctx, REL_AUDIT_PASS_COPY)
    if raw is None:
        errors.append(f"A-14 missing IIAC pass record {REL_PASS_RECORD}")
        return
    if not raw.strip():
        errors.append("A-14 IIAC pass record is empty")
        return
    if copy is None or copy != raw:
        errors.append("A-14 IIAC pass record differs from the byte-preserved auditor copy")
    digest = _sha256(raw)
    if cert is None or cert.get("pass_record_digest") != digest:
        errors.append("A-14 IIAC pass record digest does not match the certification record")
    head_blob = _show(ctx, ctx.head, REL_PASS_RECORD) if ctx.head else None
    if head_blob != raw:
        errors.append("A-14 IIAC pass record on disk differs from HEAD")
    text = raw.decode("utf-8", errors="replace")
    verdict = str(manifest.get("verdict") or "").upper()
    required = (
        f"`{STAGE_ID}`",
        POLICY_VERSION,
        f"manifest verdict `{verdict}`",
        AUDITOR_AGENT_ID,
        str(manifest.get("audited_implementation_sha") or "<missing>"),
        str(manifest.get("evidence_publication_sha") or "<missing>"),
    )
    for needle in required:
        if needle not in text:
            errors.append(f"A-14 IIAC pass record does not state {needle!r}")


def _gate_a_findings(ctx: Context, manifest: dict[str, Any], errors: list[str]) -> None:
    findings = _read_json(ctx, REL_FINDINGS, "A-15", errors)
    if findings is None:
        return
    if findings.get("stage_id") != STAGE_ID:
        errors.append("A-15 findings register stage_id mismatch")
    if findings.get("audited_implementation_sha") != manifest.get("audited_implementation_sha"):
        errors.append("A-15 findings register audited SHA differs from the manifest")
    errors.extend(f"A-15 {err}" for err in _findings_errors(findings))
    summary = findings.get("summary") if isinstance(findings.get("summary"), dict) else {}
    for key in ("PROVEN_ARCHITECTURAL_DEFECT", "IN_SCOPE_BLOCKER", "blocking_for_scoped_pass"):
        if summary.get(key) != 0:
            errors.append(f"A-15 findings summary {key} is {summary.get(key)!r}, expected 0")
    for item in findings.get("findings") or []:
        if not isinstance(item, dict):
            continue
        cls = str(item.get("class") or "").upper()
        sev = str(item.get("severity") or "").upper()
        disp = str(item.get("disposition") or item.get("reaudit_disposition") or "").upper()
        blocking = cls in BLOCKING_FINDING_CLASSES or sev in BLOCKING_FINDING_SEVERITIES
        if blocking and disp not in RESOLVED_DISPOSITIONS:
            errors.append(f"A-15 unresolved blocking finding {item.get('id')} ({cls}/{sev})")


def _gate_a_handoff(ctx: Context, manifest: dict[str, Any], errors: list[str]) -> None:
    text = _read_text(ctx, REL_CERT_HANDOFF)
    if text is None:
        errors.append(f"A-16 missing operator-attested handoff record {REL_CERT_HANDOFF}")
        return
    if POLICY_VERSION not in text:
        errors.append("A-16 handoff record does not cite the policy version")
    for key in ("implementation_session_id", "auditor_session_id", "certifier_session_id"):
        value = str(manifest.get(key) or "")
        if not value or value not in text:
            errors.append(f"A-16 handoff record does not attest {key}")
    if re.search(r"^\s*- \[ \]", text, re.MULTILINE):
        errors.append("A-16 handoff record has unattested operator checkboxes")
    if "________________" in text or not _HANDOFF_SIGNATURE.search(text):
        errors.append("A-16 handoff record has no operator approval line")
    head_blob = _show(ctx, ctx.head, REL_CERT_HANDOFF) if ctx.head else None
    if head_blob is None or head_blob.decode("utf-8-sig", errors="replace") != text:
        errors.append("A-16 handoff record on disk differs from HEAD")


def _gate_a_cert_record(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any], errors: list[str]) -> None:
    if cert.get("schema_version") != "iiac-certification-record.v1":
        errors.append("A-17 certification record schema_version mismatch")
    if cert.get("stage_id") != STAGE_ID:
        errors.append("A-17 certification record stage_id mismatch")
    if cert.get("policy_version") != POLICY_VERSION:
        errors.append("A-17 certification record policy_version mismatch")
    if str(cert.get("manifest_path") or "").replace("\\", "/") != REL_MANIFEST:
        errors.append("A-17 certification record does not bind the canonical manifest")
    verdict = str(manifest.get("verdict") or "").upper()
    if str(cert.get("independent_audit_verdict") or "").upper() != verdict:
        errors.append("A-17 certification record independent_audit_verdict differs from the manifest")
    cert_verdict = str(cert.get("certification_verdict") or "").upper()
    if not cert_verdict.startswith(verdict) or (verdict == "PASS_SCOPED") != ("SCOPED" in cert_verdict):
        errors.append(f"A-17 certification_verdict {cert_verdict!r} does not preserve the audit verdict {verdict}")


def _blocker_ids(cert: dict[str, Any] | None) -> list[tuple[str, str]]:
    blockers = (cert or {}).get("open_blockers_for_closure")
    if blockers is None:
        return []
    if not isinstance(blockers, list):
        return [("?", "open_blockers_for_closure is not a list")]
    out: list[tuple[str, str]] = []
    for item in blockers:
        text = str(item).strip()
        match = _BLOCKER_ID.match(text)
        out.append((match.group(1) if match else "?", text))
    return out


def gate_a(ctx: Context) -> list[str]:
    errors: list[str] = []
    governance = _read_json(ctx, REL_GOVERNANCE, "A-02", errors)
    if governance is not None and governance.get("normative_effect") == "IN_FORCE":
        errors.append("A-02 superseded ECAP-P4-EVOL-SCOPE-AUTO-B6 claims IN_FORCE alongside IIAC (dual activation)")
    manifest = _gate_a_manifest(ctx, errors)
    cert = _read_json(ctx, REL_CERT_RECORD, "A-17", errors)
    if manifest is None:
        return errors
    _gate_a_scope(ctx, manifest, cert, errors)
    _gate_a_roles(manifest, cert, errors)
    _gate_a_shas(ctx, manifest, cert, errors)
    _gate_a_digests(ctx, manifest, cert, errors)
    _gate_a_pass_record(ctx, manifest, cert, errors)
    _gate_a_findings(ctx, manifest, errors)
    _gate_a_handoff(ctx, manifest, errors)
    if cert is not None:
        _gate_a_cert_record(ctx, manifest, cert, errors)
        for blocker_id, text in _blocker_ids(cert):
            if blocker_id not in GATE_B_ONLY_BLOCKERS:
                errors.append(f"A-18 open certification blocker for B6 acceptance: {text}")
    return errors


# --------------------------------------------------------------------------- Gate B


def _gate_b_b6(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any], errors: list[str]) -> str:
    governance = _read_json(ctx, REL_GOVERNANCE, "B-01", errors) or {}
    b6 = _sha(governance.get("b6_acceptance_publication_sha"))
    if not b6:
        errors.append("B-01 B6 acceptance not published: governance b6_acceptance_publication_sha unset")
    if governance.get("formal_b6_status") != "ACCEPTED":
        errors.append("B-01 governance formal_b6_status is not ACCEPTED")
    if governance.get("b6_acceptance_policy_version") != POLICY_VERSION:
        errors.append("B-01 governance b6_acceptance_policy_version does not cite the IIAC policy in force")
    rows = parse_ladder_active(_read_text(ctx, REL_LADDER) or "")
    ladder_b6 = ladder_sha(rows, "B6_ACCEPTANCE_PUBLICATION_SHA")
    if not ladder_b6:
        errors.append("B-02 SHA ladder B6_ACCEPTANCE_PUBLICATION_SHA still pending/empty")
    elif b6 and ladder_b6 != b6:
        errors.append("B-02 SHA ladder B6_ACCEPTANCE_PUBLICATION_SHA differs from governance JSON")
    if str(cert.get("b6_acceptance") or "").upper() not in {"ISSUED", "ACCEPTED"}:
        errors.append("B-02 certification record b6_acceptance is not ISSUED")
    if b6 and cert.get("b6_acceptance_publication_sha") != b6:
        errors.append("B-02 certification record b6_acceptance_publication_sha differs from governance JSON")
    if not b6:
        return ""
    legacy = {ladder_sha(rows, role) for role in LEGACY_LADDER_ROWS} - {""}
    if b6 in legacy:
        errors.append("B-03 B6 SHA is a historical/internal legacy publication, not IIAC B6 acceptance")
    cert_sha = _sha(manifest.get("certification_publication_sha"))
    if not _commit_exists(ctx, b6):
        errors.append(f"B-03 B6_ACCEPTANCE_PUBLICATION_SHA {b6} is not a commit in this repository")
        return ""
    if not cert_sha or b6 == cert_sha or not _is_ancestor(ctx, cert_sha, b6):
        errors.append("B-03 B6 acceptance must be published after the certification publication")
    if b6 == ctx.head:
        errors.append("B-03 self-referential B6_ACCEPTANCE_PUBLICATION_SHA equals HEAD")
    elif ctx.head and not _is_ancestor(ctx, b6, ctx.head):
        errors.append("B-03 B6_ACCEPTANCE_PUBLICATION_SHA is not an ancestor of HEAD")
    blob = _show(ctx, b6, REL_GOVERNANCE)
    try:
        phase1 = json.loads(blob.decode("utf-8-sig")) if blob is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        phase1 = None
    if not isinstance(phase1, dict) or phase1.get("formal_b6_status") != "ACCEPTED":
        errors.append("B-04 governance JSON at B6_ACCEPTANCE_PUBLICATION_SHA does not publish the acceptance")
    elif phase1.get("b6_acceptance_publication_sha") not in (None, ""):
        errors.append("B-04 B6 phase-1 publication must leave b6_acceptance_publication_sha empty")
    return b6


def _gate_b_frz(ctx: Context, cert: dict[str, Any], errors: list[str]) -> None:
    text = _read_text(ctx, REL_FRZ_CHECKLIST)
    if text is None:
        errors.append(f"B-06 missing {REL_FRZ_CHECKLIST}")
        return
    rows = parse_frz_rows(text)
    dispositions = cert.get("frz_evol_dispositions") if isinstance(cert.get("frz_evol_dispositions"), dict) else {}
    for frz_id, requirement in FRZ_EVOL_CANONICAL_REQUIREMENTS.items():
        found = rows.get(frz_id, [])
        if len(found) != 1:
            errors.append(f"B-06 {frz_id} must have exactly one checklist row (found {len(found)})")
            continue
        cells = found[0]
        if len(cells) < 5:
            errors.append(f"B-06 {frz_id} checklist row is malformed")
            continue
        if cells[2] != requirement:
            errors.append(f"B-06 {frz_id} checklist requirement text differs from the canonical literal")
        status = _norm(cells[4]).upper()
        row_text = " ".join(cells[4:])
        disp = str(dispositions.get(frz_id) or "")
        if not status.startswith("PASS"):
            errors.append(f"B-06 {frz_id} has no individual PASS disposition (status {status or 'empty'})")
            continue
        if not disp.startswith("ELIGIBLE_PASS"):
            errors.append(f"B-06 {frz_id} promoted without a certified eligibility disposition")
        if "SCOPED" in disp.upper() and "SCOPED" not in status:
            errors.append(f"B-06 {frz_id} falsely promoted to unrestricted PASS (certified {disp!r})")
        for lim in sorted(set(re.findall(r"LIM-\d{2}", disp))):
            if lim not in row_text:
                errors.append(f"B-06 {frz_id} promoted row omits limitation {lim}")
        for phrase in FRZ_ROW_RESTRICTIONS.get(frz_id, ()):
            if phrase.lower() not in row_text.lower():
                errors.append(f"B-06 {frz_id} promoted row omits mandatory scope text {phrase!r}")


def _gate_b_operators(ctx: Context, errors: list[str]) -> None:
    text = _read_text(ctx, REL_REGISTER)
    if text is None:
        errors.append(f"B-07 missing {REL_REGISTER}")
        return
    rows = parse_register(text)
    for od_id in CLOSURE_OPERATOR_DECISIONS:
        if od_id not in rows:
            errors.append(f"B-07 operator decision {od_id} not recorded in the register")
            continue
        status, date = rows[od_id]
        if not decision_recorded(status, date):
            errors.append(f"B-07 operator decision {od_id} not recorded as decided (status {status!r}, date {date!r})")


def _gate_b_ladder(ctx: Context, b6: str, cert: dict[str, Any], errors: list[str]) -> None:
    rows = parse_ladder_active(_read_text(ctx, REL_LADDER) or "")
    for role, cell in rows.items():
        for sha in _LADDER_SHA.findall(cell):
            if not _commit_exists(ctx, sha):
                errors.append(f"B-08 ladder {role} {sha} is not a commit in this repository")
            elif ctx.head and not _is_ancestor(ctx, sha, ctx.head):
                errors.append(f"B-08 ladder {role} {sha} is not an ancestor of HEAD")
    closure = ladder_sha(rows, "FORMAL_CLOSURE_PUBLICATION_SHA")
    if closure and b6 and (closure == b6 or not _is_ancestor(ctx, b6, closure)):
        errors.append("B-08 FORMAL_CLOSURE_PUBLICATION_SHA must be published after B6 acceptance")
    governance = _read_json(ctx, REL_GOVERNANCE, "B-10", errors) or {}
    claims_closed = governance.get("evol_x_closed") is True or str(cert.get("stage_closure_status") or "") not in {
        "",
        "NOT_CLOSED",
    }
    if claims_closed and not closure:
        errors.append("B-10 EVOL-X closure claimed before FORMAL_CLOSURE_PUBLICATION_SHA exists")


def gate_b(ctx: Context) -> list[str]:
    errors: list[str] = []
    sink: list[str] = []
    manifest = _read_json(ctx, REL_MANIFEST, "B-00", sink)
    cert = _read_json(ctx, REL_CERT_RECORD, "B-00", sink)
    if manifest is None or cert is None:
        return ["B-00 IIAC manifest or certification record unavailable"]
    b6 = _gate_b_b6(ctx, manifest, cert, errors)
    _gate_b_frz(ctx, cert, errors)
    _gate_b_operators(ctx, errors)
    _gate_b_ladder(ctx, b6, cert, errors)
    for blocker_id, text in _blocker_ids(cert):
        errors.append(f"B-09 unresolved mandatory blocker ({blocker_id}): {text}")
    return errors


# --------------------------------------------------------------------------- legacy


def legacy_errors(ctx: Context) -> list[str]:
    """Human Protocol 03 path; applies only while the IIAC policy is not in force."""
    errors = ["L-00 IIAC policy NOT_IN_FORCE: IIAC evidence is not B6 authority; legacy Protocol 03 applies"]
    pass_raw = _read_bytes(ctx, REL_LEGACY_B6_PASS)
    if pass_raw is None:
        errors.append(f"L-01 missing signed B6 record: {REL_LEGACY_B6_PASS}")
    elif pass_raw == _read_bytes(ctx, REL_LEGACY_B6_DRAFT):
        errors.append("L-01 B6 PASS record must not be identical to unsigned draft")
    governance = _read_json(ctx, REL_GOVERNANCE, "L-02", errors) or {}
    if not governance.get("b6_acceptance_publication_sha"):
        errors.append("L-02 b6_acceptance_publication_sha unset in governance JSON")
    if governance.get("normative_effect") == "IN_FORCE":
        approvals = governance.get("required_approvals_present") or {}
        for flag in ("ecap_maintainer_publication", "security_review", "independent_key_custody"):
            if not approvals.get(flag):
                errors.append(f"L-02 ECAP-P4 IN_FORCE without {flag}")
    rows = parse_ladder_active(_read_text(ctx, REL_LADDER) or "")
    if not ladder_sha(rows, "B6_ACCEPTANCE_PUBLICATION_SHA"):
        errors.append("L-03 SHA ladder B6_ACCEPTANCE_PUBLICATION_SHA still pending/empty")
    register = parse_register(_read_text(ctx, REL_REGISTER) or "")
    for od_id in ("OD-EVOL-12", "OD-EVOLX-08"):
        status, date = register.get(od_id, ("", ""))
        if not decision_recorded(status, date):
            errors.append(f"L-04 {od_id} not decided in register")
    od11 = register.get("OD-EVOL-11", ("", ""))[0]
    if not od11 or "NOT NAMED" in od11.upper():
        errors.append("L-05 OD-EVOL-11 auditor identity still not named")
    return errors


# --------------------------------------------------------------------------- entry points


def evaluate(
    repo: Path | None = None,
    *,
    activation_record: Path | None = None,
    evidence_root: Path | None = None,
) -> PreflightResult:
    repo = (repo or _REPO).resolve()
    canonical = repo == _REPO.resolve()
    if canonical and evidence_root is not None:
        msg = "A-01 evidence-root override is forbidden for the canonical repository"
        return PreflightResult("INVALID", [msg], ["B-00 Gate A not satisfied"])
    record = activation_record or (CANONICAL_RECORD if canonical else repo / REL_ACTIVATION)
    ctx = Context(repo=repo, activation_record=record, evidence_root=evidence_root)
    head = _git(ctx, "rev-parse", "HEAD")
    ctx.head = head.stdout.decode("utf-8").strip() if head.returncode == 0 else ""
    if not ctx.head:
        return PreflightResult("INVALID", ["A-00 repository HEAD unavailable"], ["B-00 Gate A not satisfied"])

    mode, policy_errors = policy_mode(ctx)
    if mode == "LEGACY":
        errs = legacy_errors(ctx)
        return PreflightResult(mode, errs, ["B-00 Gate A not satisfied", *errs])
    if mode == "INVALID":
        return PreflightResult(mode, policy_errors, ["B-00 Gate A not satisfied"])
    a_errors = gate_a(ctx)
    b_errors = gate_b(ctx)
    if a_errors:
        b_errors = [f"B-00 Gate A not satisfied ({len(a_errors)} error(s))", *b_errors]
    return PreflightResult(mode, a_errors, b_errors)


def _print(result: PreflightResult, gate: str, repo: Path) -> int:
    label = "canonical" if repo.resolve() == _REPO.resolve() else "FIXTURE (not canonical evidence)"
    selected_pass = result.gate_a_pass if gate == "b6-readiness" else result.gate_b_pass
    stream = sys.stdout if selected_pass else sys.stderr
    print(f"EVOL-X closure preflight: {'PASS' if selected_pass else 'FAIL'}", file=stream)
    print(f"  repository: {label} | policy mode: {result.mode} | selected gate: {gate}", file=stream)
    for title, ok, errs in (
        ("Gate A - B6 acceptance readiness", result.gate_a_pass, result.gate_a),
        ("Gate B - formal EVOL-X closure readiness", result.gate_b_pass, result.gate_b),
    ):
        print(f"  {title}: {'PASS' if ok else 'FAIL'}", file=stream)
        for err in errs:
            print(f"    - {err}", file=stream)
    return 0 if selected_pass else 1


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument(
        "--gate",
        choices=("formal-closure", "b6-readiness"),
        default="formal-closure",
        help="Gate selecting the exit code (default: formal-closure = Gate A and Gate B)",
    )
    parser.add_argument("--repo", type=Path, default=_REPO, help="Repository root (fixtures only)")
    parser.add_argument("--activation-record", type=Path, default=None, help="Activation record (fixtures only)")
    parser.add_argument(
        "--evidence-root",
        type=Path,
        default=None,
        help="Fixture-only evidence root for the activation gate; rejected for the canonical repository",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON result instead of text")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    result = evaluate(args.repo, activation_record=args.activation_record, evidence_root=args.evidence_root)
    if args.json:
        payload = {
            "mode": result.mode,
            "gate_a_pass": result.gate_a_pass,
            "gate_b_pass": result.gate_b_pass,
            "gate_a_errors": result.gate_a,
            "gate_b_errors": result.gate_b,
        }
        print(json.dumps(payload, indent=2))
        ok = result.gate_a_pass if args.gate == "b6-readiness" else result.gate_b_pass
        return 0 if ok else 1
    return _print(result, args.gate, args.repo)


if __name__ == "__main__":
    raise SystemExit(main())

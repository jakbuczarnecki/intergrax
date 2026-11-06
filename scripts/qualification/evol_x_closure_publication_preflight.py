#!/usr/bin/env python3
# © Artur Czarnecki. All rights reserved.

"""EVOL-X closure publication preflight — two-gate IIAC contract (T-IGX-017-A, RA-B4 corrective T-IGX-017-B).

The preflight is fail-closed and evaluates two separate gates:

Gate A — B6 acceptance readiness
    The published independent IIAC audit and certification evidence is valid for a
    B6 acceptance under the policy that is genuinely ``IN_FORCE``. Gate A does not
    require ``B6_ACCEPTANCE_PUBLICATION_SHA``: that SHA cannot exist before B6 is
    published.

Gate B — formal EVOL-X closure readiness
    Gate A holds, a genuine B6 acceptance *record* is published and bound (two-phase,
    non-self-referential), every FRZ-EVOL row has an individual disposition no wider than the
    certified scope, the applicable operator decisions are recorded with a positive decision state
    and a verifiable decision record, the SHA ladder is valid and no mandatory blocker is open.
    Gate B does not require ``FORMAL_CLOSURE_PUBLICATION_SHA`` (it cannot exist before the closure
    commit) but validates it when present.

Gate A PASS never implies Gate B PASS. Legacy human Protocol 03 requirements apply only
while the IIAC policy is not in force; a policy record that claims ``IN_FORCE`` but fails
the activation gate fails both gates (no legacy fallback).

RA-B4 corrective contracts (handoff: ``evol_x/EVOL_X_RA_B4_CORRECTIVE_REAUDIT_HANDOFF.md``):

* **Byte contract (RA-B4-01).** Evidence is verified against the *canonical committed bytes* (the Git blob at
  ``HEAD``). A working-tree file is accepted as that blob when it is byte-identical, or — only for paths
  whose ``text`` attribute is not ``unset`` — when it equals the blob with LF expanded to CRLF (the exact
  transformation of a ``core.autocrlf=true`` checkout of a pure-LF blob). Any other difference (tampering,
  uncommitted edits, CRLF under ``-text``) stays a mismatch. Digest-bound paths are additionally ``-text``
  in ``.gitattributes``.
* **Blocker lifecycle (RA-B4-02).** The blocker set published at ``certification_publication_sha`` is the
  immutable baseline. Every baseline blocker must still be listed as open, or carry an evidence-bound
  ``RESOLVED_WITH_EVIDENCE`` entry that is independently re-verified here. Absence is never closure.
* **Operator decisions (RA-B4-03).** Decision state is an exact allow-list; any negation is rejected;
  rows are read only from the register status table; decisions that close EVOL-X need a verifiable record.
  Operator signature lines are process-enforced text, not cryptographic proof.
* **B6 acceptance (RA-B4-04).** Flags are never sufficient. A B6 acceptance record must exist at the pinned
  publication commit, be digest-bound by the certification record, cite the independent audit and the
  certified scope, and carry an operator signature line.

T-W4-01 corrective contracts (ticket ``T-IGX-019-A``):

* **Exact certified scope (RA-B4-05, ``A-19`` / ``B-11``).** Scope is never decided by phrase presence or prefix
  match. The blob published at ``certification_publication_sha`` (P2) is the immutable scope baseline: the
  limitation multiset ``(id, text)``, ``certification_verdict`` and every FRZ-EVOL disposition must be *equal* to it
  (order is irrelevant). A promoted FRZ row must equal the row rendered from the baseline disposition
  (:func:`render_promoted_row`); the B6 record may carry only allow-listed lines and its ``Accepted limitations``
  must equal the baseline ``(id, sha256(text))`` set (:func:`accepted_limitation_tokens`). There is no amendment
  channel: any difference from the baseline needs a new P2 certification publication.
* **Blocker lifecycle for late blockers (W4-F03).** Every blocker that appeared in any published version of the
  certification record after the baseline must stay open or carry an evidence-bound resolution; duplicate ids fail.
* **Operator records (W4-F04).** A signature older than the certification publication and a ``Decision scope``
  containing a negation are rejected.

Verification is delegated to the canonical IIAC routines (activation gate, manifest
validator, role-separation validator). This module adds only EVOL-X stage bindings.
It never writes to the repository.
"""

from __future__ import annotations

import os
import sys

# Bootstrap (W402-05/06, W403-01, W404-01/02/03). Nothing from outside the interpreter's built-ins may be imported
# before this block: a planted, git-ignored ``random.py`` / ``json.py`` / ``shutil.py`` next to the script, in the
# current directory or at the repository root would otherwise shadow the standard library and run before any check.
# (1) Drop the script directory from ``sys.path``; when run as a program also drop the current directory and the
#     repository root, so only the standard library and site-packages can supply non-repository modules.
# (2) Every git call in this process and its children ignores replace objects.
# (3) No bytecode is read from or written into the repository: an unchecked-hash ``.pyc`` in a git-ignored
#     ``__pycache__`` would replace a committed verifier module. Children inherit the same settings through the
#     environment. The prefix is computed without importing ``tempfile``.
_HERE = os.path.dirname(os.path.abspath(__file__))
_ROOT = os.path.dirname(os.path.dirname(_HERE))


def _same_dir(entry: str, target: str) -> bool:
    return os.path.normcase(os.path.abspath(entry or os.getcwd())) == os.path.normcase(target)


_STRIP = [_HERE]
if __name__ == "__main__":
    _STRIP += [_ROOT, os.getcwd()]
sys.path[:] = [entry for entry in sys.path if not any(_same_dir(entry, target) for target in _STRIP)]
os.environ["GIT_NO_REPLACE_OBJECTS"] = "1"
_TMP = (
    os.environ.get("TEMP")
    or os.environ.get("TMP")
    or os.environ.get("TMPDIR")
    or ("/tmp" if os.name != "nt" else "C:\\Windows\\Temp")
)
sys.dont_write_bytecode = True
sys.pycache_prefix = os.path.join(_TMP, "evolx-preflight-no-bytecode-cache", str(os.getpid()))
os.environ["PYTHONDONTWRITEBYTECODE"] = "1"
os.environ["PYTHONPYCACHEPREFIX"] = sys.pycache_prefix

import argparse  # noqa: E402
import hashlib  # noqa: E402
import json  # noqa: E402
import posixpath  # noqa: E402
import re  # noqa: E402
import subprocess  # noqa: E402
import unicodedata  # noqa: E402
from collections import Counter  # noqa: E402
from dataclasses import dataclass, field  # noqa: E402
from datetime import UTC, date, datetime, timedelta  # noqa: E402
from pathlib import Path  # noqa: E402
from typing import Any  # noqa: E402

_REPO = Path(__file__).resolve().parents[2]
# W404-02: the repository's ``scripts`` package is registered explicitly from this checkout, so it is found neither
# through ``sys.path`` (an editable install, ``PYTHONPATH`` or a repository-root file could supply another one) nor
# does the repository root ever have to precede the standard library.
if "scripts" not in sys.modules:
    import importlib.machinery as _machinery  # noqa: E402
    import importlib.util as _importlib_util  # noqa: E402

    _spec = _machinery.ModuleSpec("scripts", None, is_package=True)
    _spec.submodule_search_locations = [str(_REPO / "scripts")]
    sys.modules["scripts"] = _importlib_util.module_from_spec(_spec)

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

for _module_name in ("scripts.governance.iiac_activation_gate", "scripts.governance.iiac_common"):
    _module_file = Path(sys.modules[_module_name].__file__ or "").resolve()
    if _REPO not in _module_file.parents:
        raise SystemExit(f"{_module_name} was not imported from this repository ({_module_file}): refusing to run")

STAGE_ID = "EVOL-X"
PREFLIGHT_SCRIPT_REL = "scripts/qualification/evol_x_closure_publication_preflight.py"
QUAL = "docs/project/maintainers/qualification"
EVOL_DIR = f"{QUAL}/evol_x"
EVIDENCE_ROOT_REL = f"{QUAL}/iiac_evidence/EVOL-X"
AUDIT_DIR = f"{EVIDENCE_ROOT_REL}/audit"
CERT_DIR = f"{EVIDENCE_ROOT_REL}/certification"
OPERATOR_DIR = f"{EVIDENCE_ROOT_REL}/operator_decisions"
B6_DIR = f"{EVIDENCE_ROOT_REL}/b6_acceptance"

# DA-01: the P2 certification publication is pinned here and never taken from a mutable manifest field. The canonical
# repository always uses this constant; isolated fixture repositories supply their own commit through ``p2_sha``.
P2_CERTIFICATION_PUBLICATION_SHA = "ad0877715ed079ce920ff9d925fda03555b2c04e"

# The one legitimate rewrite of the binding description (separate binding commit ca2b14782); no other text is accepted.
P2_BINDING_TEXT = (
    "bound to the P2 publication commit (the commit that first published this record); "
    "interim placeholder in that commit was audit_execution_sha"
)

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
REL_B6_RECORD = f"{B6_DIR}/EVOL_X_IIAC_B6_ACCEPTANCE_RECORD.md"
REL_B6_INDEX = f"{B6_DIR}/B6_ACCEPTANCE_EVIDENCE_INDEX.json"
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
# Closing decisions that are still outstanding at IIAC activation need a verifiable decision record in
# addition to the register row. Value = tokens the record's ``Decision scope`` must contain.
OD_RECORD_REQUIREMENTS: dict[str, tuple[str, ...]] = {
    "OD-EVOL-12": ("FRZ-EVOL-07", "FRZ-EVOL-08", "FRZ-EVOL-10"),
    "OD-EVOLX-08": (STAGE_ID, "formal closure"),
}
LEGACY_LADDER_ROWS = (
    "INTERNAL_B6_ACCEPTANCE_PUBLICATION_SHA",
    "INTERNAL_PARENT_CLOSURE_PUBLICATION_SHA",
    "INTERNAL_CLOSURE_PUBLICATION_SHA",
    "INTERNAL_CERTIFICATION_PUBLICATION_SHA",
    "PROTOCOL_03_AUTOMATED_AUDIT_PUBLICATION_SHA",
)

# Blocker lifecycle (RA-B4-02).
RESOLUTION_STATUS = "RESOLVED_WITH_EVIDENCE"
KIND_REAUDIT = "INDEPENDENT_REAUDIT"
KIND_OPERATOR = "OPERATOR_AUTHORIZATION"
BLOCKER_P3 = "B6-P3"
BLOCKER_CLOSURE_KINDS: dict[str, str] = {"B-4": KIND_REAUDIT, "B-5": KIND_OPERATOR, BLOCKER_P3: KIND_OPERATOR}
BLOCKER_REQUIRED_DECISIONS: dict[str, frozenset[str]] = {"B-5": frozenset({"OD-EVOL-12", "OD-EVOLX-08"})}
BLOCKER_SCOPE_TOKENS: dict[str, tuple[str, ...]] = {BLOCKER_P3: ("B6 acceptance", STAGE_ID)}
B6_SCOPE_TOKENS = ("B6 acceptance", STAGE_ID)

# Operator decision state (RA-B4-03). Exact allow-list, matched against the whole normalised status.
AFFIRMATIVE_STATUSES = frozenset(
    {"SIGNED / APPROVED", "SIGNED/APPROVED", "OPERATOR APPROVED", "SIGNED", "APPROVED", "CONFIRMED"}
)
NEGATION_WORDS = frozenset(
    {
        "NOT",
        "NO",
        "NEVER",
        "NONE",
        "UNSIGNED",
        "UNAPPROVED",
        "UNCONFIRMED",
        "PENDING",
        "DRAFT",
        "REJECTED",
        "DECLINED",
        "DENIED",
        "REVOKED",
        "WITHDRAWN",
        "RESCINDED",
        "SUPERSEDED",
        "CANCELLED",
        "CANCELED",
        "EXPIRED",
        "VOID",
        "INVALID",
        "TBD",
        "AWAITING",
        "PROPOSED",
        "REQUIRED",
    }
)
CLOCK_SKEW = timedelta(days=1)
SIGNATURE_SKEW = timedelta(minutes=15)

# Negations that make a ``Decision scope`` a refusal, not an authorization (W4-F04). Matched as whole words on the
# NFKC-normalised, upper-cased scope; contractions (``isn't``) are rejected separately.
SCOPE_NEGATION_WORDS = frozenset(
    {
        "NOT",
        "NO",
        "NEVER",
        "NONE",
        "DENIED",
        "DENY",
        "UNAUTHORIZED",
        "UNAUTHORISED",
        "REVOKED",
        "REJECTED",
        "DECLINED",
        "WITHDRAWN",
        "PROHIBITED",
        "FORBIDDEN",
    }
)
_REVOCATION_FIELD = re.compile(r"revok|revoc|rescind|withdraw|supersed|retract|cancel|void", re.IGNORECASE)
_SCOPE_CONTRACTION = re.compile(r"\w[nN]['’]t\b")

# RA-B4-05 exact scope. Certification-record keys that legitimately change after the P2 publication; every other
# key must already exist in the baseline (and scope/claim-named keys must equal it).
CERT_LIFECYCLE_KEYS = frozenset(
    {
        "certification_publication_sha",
        "certification_publication_sha_binding",
        "open_blockers_for_closure",
        "resolved_blockers_for_closure",
        "stage_closure_status",
        "frz_checklist_rows_changed",
        "formal_closure_publication_sha",
        "b6_acceptance",
        "b6_acceptance_publication_sha",
        "b6_acceptance_record_path",
        "b6_acceptance_record_digest",
    }
)
_SCOPE_KEY = re.compile(r"scope|claim", re.IGNORECASE)

# B6 acceptance record: the only lines it may carry (RA-B4-05). Title, scope and decision fields are canonical.
B6_RECORD_TITLE = f"# {STAGE_ID} IIAC B6 acceptance record"
B6_RECORD_FIELDS = frozenset(
    {
        "Decision ID",
        "Stage",
        "Policy version",
        "Decision scope",
        "Decision status",
        "Accepted verdict",
        "Accepted limitations",
        "Independent audit manifest digest",
        "Audit report digest",
        "Findings digest",
        "Audited implementation SHA",
        "Certification publication SHA",
        "Certifier session",
        "Certifier agent",
    }
)
_B6_DECISION_ID = re.compile(r"^[A-Za-z0-9._-]{1,80}$")

_SHA40 = re.compile(r"^[0-9a-f]{40}$")
_DIGEST64 = re.compile(r"^[0-9a-f]{64}$")
_LADDER_SHA = re.compile(r"`([0-9a-f]{40})`")
_LIM_ID = re.compile(r"^(LIM-\d{2}|F-C-\d{2})\b")
_BLOCKER_ID = re.compile(r"^(B-\d+)\b")
_BLOCKER_P3_TEXT = re.compile(r"^B6 acceptance \(P3\)")
_ISO_TS = r"\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(?:\.\d+)?(?:Z|[+-]\d{2}:\d{2})"
_HANDOFF_SIGNATURE = re.compile(rf"^`?Approved handoff: operator @ ({_ISO_TS})`?", re.MULTILINE)
_RECORD_SIGNATURE = re.compile(rf"^Approved (.+?): operator @ ({_ISO_TS})\s*$")
_RECORD_FIELD = re.compile(r"^([A-Z][A-Za-z0-9 /_-]{1,60}): (\S.*?)\s*$")
_RECORD_NOTE_PATH = re.compile(r"iiac_evidence/EVOL-X/operator_decisions/([A-Za-z0-9_.\-]+\.md)")
_RECORD_NOTE_DIGEST = re.compile(r"sha256[:=]\s*`?([0-9a-f]{64})")
_REGISTER_HEADER = re.compile(r"^\|\s*ID\s*\|\s*Status\s*\|\s*Signed date\s*\|", re.IGNORECASE)


@dataclass
class Context:
    repo: Path
    activation_record: Path
    evidence_root: Path | None
    head: str = ""
    p2: str = ""
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
    # W401-RA-01: replace refs must never substitute the content of a pinned commit.
    return subprocess.run(
        [
            "git",
            "--no-replace-objects",
            "-c",
            "core.autocrlf=false",
            "-c",
            "core.fsmonitor=false",
            "-c",
            "core.commitGraph=false",
            *args,
        ],
        cwd=ctx.repo,
        capture_output=True,
        check=False,
    )


def _git_repo_config(ctx: Context, *args: str) -> subprocess.CompletedProcess[bytes]:
    """Git with the repository's own configuration (no overrides): used to ask whether a file is *clean* by the
    conventions the checkout was made with. Replace objects stay disabled."""
    return subprocess.run(
        ["git", "--no-replace-objects", "-c", "core.fsmonitor=false", "-c", "core.commitGraph=false", *args],
        cwd=ctx.repo,
        capture_output=True,
        check=False,
    )


def _strict_json_loads(text: str) -> Any:
    """``json.loads`` that rejects duplicate object keys (W401-RA-02: Python would silently keep the last one)."""

    def hook(pairs: list[tuple[str, Any]]) -> dict[str, Any]:
        seen: set[str] = set()
        for key, _ in pairs:
            if key in seen:
                raise json.JSONDecodeError(f"duplicate JSON object key {key!r}", text, 0)
            seen.add(key)
        return dict(pairs)

    def reject_constant(name: str) -> Any:
        raise json.JSONDecodeError(f"non-standard JSON constant {name}", text, 0)

    return json.loads(text, object_pairs_hook=hook, parse_constant=reject_constant)


def _commit_exists(ctx: Context, sha: str) -> bool:
    return _git(ctx, "cat-file", "-e", f"{sha}^{{commit}}").returncode == 0


def _is_ancestor(ctx: Context, ancestor: str, descendant: str) -> bool:
    return _git(ctx, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def _is_strict_ancestor(ctx: Context, ancestor: str, descendant: str) -> bool:
    return ancestor != descendant and _is_ancestor(ctx, ancestor, descendant)


def _commit_time(ctx: Context, sha: str) -> datetime | None:
    proc = _git(ctx, "show", "-s", "--format=%cI", sha)
    if proc.returncode != 0:
        return None
    try:
        return datetime.fromisoformat(proc.stdout.decode("utf-8").strip())
    except ValueError:
        return None


def _show(ctx: Context, sha: str, rel: str) -> bytes | None:
    proc = _git(ctx, "show", f"{sha}:{rel}")
    return proc.stdout if proc.returncode == 0 else None


def _tree_entries(ctx: Context, sha: str, rel_dir: str) -> dict[str, tuple[str, str]]:
    """Every non-tree entry below ``rel_dir`` at ``sha``: path -> (object type, object id). NUL-separated, so names
    are exact; gitlinks (``commit``) and symlinks are listed too."""
    proc = _git(ctx, "ls-tree", "-r", "-z", sha, "--", rel_dir)
    entries: dict[str, tuple[str, str]] = {}
    if proc.returncode != 0:
        return entries
    for record in proc.stdout.decode("utf-8", errors="replace").split("\x00"):
        meta, _, path = record.partition("\t")
        parts = meta.split()
        if path and len(parts) == 3:
            entries[path] = (parts[1], parts[2])
    return entries


def _tree_blobs(ctx: Context, sha: str, rel_dir: str) -> dict[str, str]:
    return {path: oid for path, (kind, oid) in _tree_entries(ctx, sha, rel_dir).items() if kind == "blob"}


def _git_blob_id(data: bytes) -> str:
    return hashlib.sha1(b"blob %d\0" % len(data) + data).hexdigest()  # noqa: S324 - git object id, not security


def _changed_paths(ctx: Context, base: str, head: str, rel: str) -> list[str]:
    proc = _git(ctx, "diff", "--name-only", base, head, "--", rel)
    if proc.returncode != 0:
        return [f"<git diff failed: {proc.stderr.decode('utf-8', errors='replace').strip()}>"]
    return [line.strip() for line in proc.stdout.decode("utf-8").splitlines() if line.strip()]


def _worktree_bytes(ctx: Context, rel: str) -> bytes | None:
    path = ctx.path(rel)
    try:
        return path.read_bytes() if path.is_file() else None
    except OSError:
        return None


def _text_conversion_applies(ctx: Context, rel: str) -> bool:
    """True unless ``.gitattributes`` marks the path ``-text`` (checkout never rewrites it)."""
    proc = _git(ctx, "check-attr", "text", "--", rel)
    if proc.returncode != 0:
        return True
    return proc.stdout.decode("utf-8", errors="replace").rsplit(":", 1)[-1].strip() != "unset"


def _worktree_matches_blob(ctx: Context, rel: str, raw: bytes, blob: bytes) -> bool:
    """RA-B4-01 byte contract: identical, or the exact CRLF expansion of a pure-LF blob on a converting path."""
    if raw == blob:
        return True
    return b"\r" not in blob and raw == blob.replace(b"\n", b"\r\n") and _text_conversion_applies(ctx, rel)


def _head_blob_ids(ctx: Context) -> dict[str, str]:
    if "head_blobs" not in ctx.cache:
        ctx.cache["head_blobs"] = _tree_blobs(ctx, ctx.head, QUAL) if ctx.head else {}
    return ctx.cache["head_blobs"]


def _read_bytes(ctx: Context, rel: str) -> bytes | None:
    """Canonical bytes of an evidence path.

    Returns the Git blob at ``HEAD`` when the working tree equals it under the RA-B4-01 byte contract,
    otherwise the working-tree bytes (so tampering and uncommitted edits surface as digest/blob mismatches).
    """
    raw = _worktree_bytes(ctx, rel)
    ctx.cache.setdefault("read_paths", set()).add(rel)
    if raw is None or not ctx.head:
        return raw
    blob_id = _head_blob_ids(ctx).get(rel)
    if blob_id is None or _git_blob_id(raw) == blob_id:
        return raw
    blob = _show(ctx, ctx.head, rel)
    if blob is not None and _worktree_matches_blob(ctx, rel, raw, blob):
        return blob
    return raw


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
        loaded = _strict_json_loads(raw.decode("utf-8-sig"))
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


_REDIRECTING_GIT_ENV = (
    "GIT_REPLACE_REF_BASE",
    "GIT_ALTERNATE_OBJECT_DIRECTORIES",
    "GIT_OBJECT_DIRECTORY",
    "GIT_NAMESPACE",
    "GIT_SHALLOW_FILE",
    "GIT_GRAFT_FILE",
)


def _repository_integrity_errors(ctx: Context, code: str) -> list[str]:
    """W401-RA-01: refuse a repository whose history can be rewritten underneath the pinned commits.

    Replace refs, a grafts file and shallow boundaries all make ``git show <pinned sha>:path`` return content that
    is not the pinned commit's. Every git call also runs with ``--no-replace-objects``; this check makes the
    condition visible and fail-closed instead of silently neutralised."""
    errors: list[str] = []
    refs = _git(ctx, "for-each-ref", "--format=%(refname)", "refs/replace/")
    if refs.returncode != 0:
        errors.append(f"{code} cannot list refs/replace (git for-each-ref failed): failing closed")
    else:
        replaced = [line for line in refs.stdout.decode("utf-8", errors="replace").splitlines() if line.strip()]
        if replaced:
            errors.append(f"{code} repository carries replace refs ({', '.join(replaced[:3])}): history is not trusted")
    graft = _git(ctx, "rev-parse", "--git-path", "info/grafts")
    if graft.returncode == 0:
        graft_path = Path(graft.stdout.decode("utf-8", errors="replace").strip())
        if not graft_path.is_absolute():
            graft_path = ctx.repo / graft_path
        if graft_path.exists():
            errors.append(f"{code} repository carries a grafts file ({graft_path.name}): history is not trusted")
    for name in _REDIRECTING_GIT_ENV:
        if os.environ.get(name):
            errors.append(f"{code} environment variable {name} redirects git objects or history: not trusted")
    shallow = _git(ctx, "rev-parse", "--is-shallow-repository")
    if shallow.returncode != 0 or shallow.stdout.decode("utf-8", errors="replace").strip() != "false":
        errors.append(f"{code} repository is shallow or its depth is unknown: the pinned history is not complete")
    return errors


_EVIDENCE_PREFIX = f"{EVIDENCE_ROOT_REL}/"
# Everything the verdict stands on: the evidence tree, the qualification inputs that were read, and the verifier code
# (the preflight itself and every module it imports live below ``scripts``).
_BOUND_TREES = (
    EVIDENCE_ROOT_REL,
    "scripts/governance",
    "scripts/qualification",
    "scripts/__init__.py",
    ".gitattributes",
)
_IGNORED_PARTS = frozenset({"__pycache__"})


def _unset_text_paths(ctx: Context, rels: list[str]) -> set[str]:
    """Paths whose ``text`` attribute is unset (``-text``): checkout never rewrites them. One batched git call."""
    if not rels:
        return set()
    proc = subprocess.run(
        ["git", "--no-replace-objects", "check-attr", "-z", "--stdin", "text"],
        cwd=ctx.repo,
        input=("\x00".join(rels) + "\x00").encode("utf-8"),
        capture_output=True,
        check=False,
    )
    if proc.returncode != 0:
        return set()
    fields = proc.stdout.decode("utf-8", errors="replace").split("\x00")
    return {fields[n] for n in range(0, len(fields) - 2, 3) if fields[n + 2] == "unset"}


def _worktree_files(ctx: Context, rel_dir: str) -> set[str]:
    root = ctx.path(rel_dir)
    found: set[str] = set()
    if not root.is_dir():
        return found
    for path in root.rglob("*"):
        if path.is_file() and not (set(path.relative_to(ctx.repo).parts) & _IGNORED_PARTS) and path.suffix != ".pyc":
            found.add(path.relative_to(ctx.repo).as_posix())
    return found


def _loaded_module_paths(ctx: Context) -> set[str]:
    """Repository modules (``scripts/`` or ``intergrax/``) that this process has actually imported: the verifier's
    real import closure, bound to HEAD whatever directory they live in. Only meaningful when the repository under
    test is the one the code runs from."""
    if ctx.repo.resolve() != _REPO.resolve():
        return set()
    found: set[str] = set()
    for module in list(sys.modules.values()):
        file = getattr(module, "__file__", None)
        if not file or not str(file).endswith(".py"):
            continue
        try:
            rel = Path(file).resolve().relative_to(_REPO).as_posix()
        except ValueError:
            continue
        if rel.startswith(("scripts/", "intergrax/")):
            found.add(rel)
    return found


def _dirty_input_errors(ctx: Context, code: str) -> list[str]:
    """W401-RA-03 / W402-01 / W402-02: every verified input equals the committed ``HEAD`` content.

    The oracle is the file content, never ``git status``: a ``skip-worktree`` / ``assume-unchanged`` bit or a
    ``core.fsmonitor`` program makes status report a modified file as clean, so each file is compared byte for byte
    with its ``HEAD`` blob under the RA-B4-01 byte contract (identical, or the exact CRLF expansion of a pure-LF blob
    on a path that checkout converts). Covered: the inputs that were read, every file below the EVOL-X evidence root
    and every file below ``scripts`` (the verifier code), plus untracked files in those trees and any index bit that
    hides a path from status."""
    blobs: dict[str, str] = {}
    for tree in _BOUND_TREES:
        listed = _git(ctx, "ls-tree", "-r", "-z", ctx.head, "--", tree)
        if listed.returncode != 0:
            return [f"{code} cannot list the committed {tree} tree: failing closed"]
        for record in listed.stdout.decode("utf-8", errors="replace").split("\x00"):
            meta, _, path = record.partition("\t")
            parts = meta.split()
            if path and len(parts) == 3 and parts[1] == "blob":
                blobs[path] = parts[2]
    read_paths: set[str] = set(ctx.cache.get("read_paths", set())) | _loaded_module_paths(ctx)
    for rel in read_paths:
        if rel not in blobs:
            single = _git(ctx, "rev-parse", "--verify", "-q", f"{ctx.head}:{rel}")
            if single.returncode == 0:
                blobs[rel] = single.stdout.decode("utf-8").strip()
    dirty: list[str] = []
    suspects: list[str] = []
    for rel, blob_id in sorted(blobs.items()):
        raw = _worktree_bytes(ctx, rel)
        if raw is None:
            dirty.append(f"{rel} [missing]")
        elif _git_blob_id(raw) != blob_id:
            blob = _show(ctx, ctx.head, rel)
            if blob is None:
                dirty.append(f"{rel} [unreadable at HEAD]")
            elif b"\r" in blob or raw != blob.replace(b"\n", b"\r\n"):
                dirty.append(f"{rel} [modified]")
            else:
                suspects.append(rel)
    unset = _unset_text_paths(ctx, suspects)
    dirty.extend(f"{rel} [modified]" for rel in suspects if rel in unset)
    for tree in _BOUND_TREES:
        dirty.extend(f"{rel} [untracked]" for rel in sorted(_worktree_files(ctx, tree) - set(blobs)))
    for rel in sorted(read_paths - set(blobs)):
        if _worktree_bytes(ctx, rel) is not None:
            dirty.append(f"{rel} [untracked]")
    flags = _git(ctx, "ls-files", "-v", "-z", "--", *_BOUND_TREES, QUAL)
    if flags.returncode != 0:
        dirty.append("<git ls-files -v failed>")
    else:
        for record in flags.stdout.decode("utf-8", errors="replace").split("\x00"):
            if len(record) > 2 and record[0] != "H":
                path = record[2:]
                if path in blobs or path in read_paths:
                    dirty.append(f"{path} [index flag {record[0]!r}: hidden from git status]")
    if dirty:
        shown = ", ".join(dirty[:5])
        return [
            f"{code} uncommitted or hidden changes in verified inputs ({len(dirty)}): {shown}; "
            "the verdict must bind to HEAD"
        ]
    return []


def _sha(value: Any) -> str:
    text = value if isinstance(value, str) else ""
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


def parse_register_rows(text: str) -> dict[str, list[tuple[str, str, str]]]:
    """``| **OD-…** | status | date | notes |`` rows of the register *status* tables only.

    Rows of other tables (decision-text tables) and narrative lines are never decisions (RA-B4-03).
    """
    rows: dict[str, list[tuple[str, str, str]]] = {}
    in_status_table = False
    for line in text.splitlines():
        if not line.startswith("|"):
            in_status_table = False
            continue
        if _REGISTER_HEADER.match(line):
            in_status_table = True
            continue
        if not in_status_table or not line.startswith("| **OD-"):
            continue
        cells = _table_cells(line)
        if len(cells) < 3:
            continue
        notes = cells[3] if len(cells) > 3 else ""
        rows.setdefault(_norm(cells[0]), []).append((_norm(cells[1]), _norm(cells[2]), notes))
    return rows


def parse_register(text: str) -> dict[str, tuple[str, str]]:
    """First status-table row per decision id as ``(status, date)``."""
    return {od: (rows[0][0], rows[0][1]) for od, rows in parse_register_rows(text).items() if rows}


def _normalize_status(status: str) -> str:
    text = unicodedata.normalize("NFKC", status)
    text = "".join(ch for ch in text if unicodedata.category(ch) != "Cf")
    return re.sub(r"\s+", " ", text.replace("*", "").replace("`", "")).strip().upper()


def decision_status_errors(status: str, date_text: str, *, today: date | None = None) -> list[str]:
    """Reasons a register row is not a recorded positive decision (empty list = recorded)."""
    errors: list[str] = []
    normalized = _normalize_status(status)
    words = set(re.findall(r"[A-Z0-9/]+", normalized))
    if not normalized:
        errors.append("empty decision status")
    elif normalized not in AFFIRMATIVE_STATUSES:
        if words & NEGATION_WORDS or "?" in normalized:
            errors.append(f"decision status {status!r} is negated, pending or not final")
        else:
            errors.append(f"decision status {status!r} is not a canonical positive decision state")
    stripped = date_text.strip()
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", stripped) is None:
        errors.append(f"decision date {date_text!r} is not an ISO date")
    else:
        try:
            decided = date.fromisoformat(stripped)
        except ValueError:
            errors.append(f"decision date {date_text!r} is not a calendar date")
        else:
            limit = (today or datetime.now(UTC).date()) + CLOCK_SKEW
            if decided > limit:
                errors.append(f"decision date {date_text!r} is in the future")
    return errors


def decision_recorded(status: str, date_text: str) -> bool:
    return not decision_status_errors(status, date_text)


def parse_operator_record(text: str) -> tuple[dict[str, list[str]], list[tuple[str, str]]]:
    """Field lines (``Key: value`` at column 0) and ``Approved <id>: operator @ <ISO>`` signature lines.

    Fenced blocks, quotes, list items and mid-line mentions never count (RA-B4-03 narrative guard).
    """
    fields: dict[str, list[str]] = {}
    signatures: list[tuple[str, str]] = []
    fenced = False
    for line in text.lstrip("﻿").splitlines():
        if line.lstrip().startswith(("```", "~~~")):
            fenced = not fenced
            continue
        if fenced:
            continue
        sig = _RECORD_SIGNATURE.match(line)
        if sig:
            signatures.append((sig.group(1).strip(), sig.group(2)))
            continue
        item = _RECORD_FIELD.match(line)
        if item:
            fields.setdefault(item.group(1).strip(), []).append(item.group(2).strip())
    return fields, signatures


def operator_record_errors(
    text: str,
    *,
    decision_id: str | None,
    scope_tokens: tuple[str, ...],
    expected: dict[str, str] | None = None,
    now: datetime | None = None,
    not_before: datetime | None = None,
) -> tuple[list[str], dict[str, Any]]:
    """Validate an operator decision record. Returns ``(errors, info)``; ``info`` carries id and signature time.

    ``not_before`` (W4-F04): a signature older than this instant (the certification publication) is rejected.
    """
    errors: list[str] = []
    info: dict[str, Any] = {"decision_id": None, "signed_at": None, "fields": {}}
    if re.search(r"_{4,}", text) or "NOT SIGNED" in text.upper():
        errors.append("record contains an unsigned placeholder")
    fields, signatures = parse_operator_record(text)
    info["fields"] = fields

    def single(key: str) -> str | None:
        values = fields.get(key, [])
        if len(values) != 1:
            errors.append(f"record must state exactly one {key!r} line (found {len(values)})")
            return None
        return values[0]

    found_id = single("Decision ID")
    info["decision_id"] = found_id
    if found_id is not None and decision_id is not None and found_id != decision_id:
        errors.append(f"record Decision ID {found_id!r} != {decision_id!r}")
    stage = single("Stage")
    if stage not in (None, STAGE_ID):
        errors.append(f"record Stage must be {STAGE_ID}")
    policy = single("Policy version")
    if policy not in (None, POLICY_VERSION):
        errors.append(f"record Policy version must be {POLICY_VERSION}")
    scope = single("Decision scope")
    if scope is not None:
        for token in scope_tokens:
            if token not in scope:
                errors.append(f"record Decision scope lacks {token!r}")
        negation = scope_negation(scope)
        if negation:
            errors.append(f"record Decision scope contains the negation {negation!r}")
        if not scope.isascii() or not scope.isprintable():
            # W4-F04: homoglyphs / zero-width / control characters can hide a negation from the word scan
            errors.append("record Decision scope contains non-ASCII or non-printable characters (ambiguous text)")
    for key in fields:
        if _REVOCATION_FIELD.search(key):
            errors.append(f"record carries the revocation-type field {key!r}")
    status = single("Decision status")
    if status is not None and status != "AUTHORIZED":
        errors.append(f"record Decision status {status!r} is not AUTHORIZED")
    for key, want in (expected or {}).items():
        got = single(key)
        if got is not None and got != want:
            errors.append(f"record {key!r} does not match the verified evidence")
    if len(signatures) != 1:
        errors.append(f"record must carry exactly one operator signature line (found {len(signatures)})")
    else:
        label, stamp = signatures[0]
        if found_id is not None and label != found_id:
            errors.append(f"operator signature line is for {label!r}, not {found_id!r}")
        try:
            signed_at = datetime.fromisoformat(stamp)
        except ValueError:
            errors.append(f"operator signature timestamp {stamp!r} is not valid")
        else:
            info["signed_at"] = signed_at
            if signed_at.tzinfo is None or signed_at > (now or datetime.now(UTC)) + CLOCK_SKEW:
                errors.append("operator signature timestamp is in the future")
            elif not_before is not None and signed_at < not_before:
                errors.append("operator signature predates the certification publication")
    return errors, info


def scope_negation(scope: str) -> str:
    """The first negation found in a decision scope ('' when none): ``NOT``, ``DENIED``, ``UNAUTHORIZED``, ``n't``…"""
    text = "".join(ch for ch in unicodedata.normalize("NFKC", scope) if unicodedata.category(ch) != "Cf")
    contraction = _SCOPE_CONTRACTION.search(text)
    if contraction:
        return contraction.group(0)
    for word in re.findall(r"[A-Z0-9]+", text.upper()):
        if word in SCOPE_NEGATION_WORDS:
            return word
    return ""


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


def limitation_entries(items: Any) -> tuple[list[tuple[str, str]], list[str]]:
    """``(id, exact text)`` for every limitation entry plus structural problems (RA-B4-05).

    Entries without an ``LIM-nn`` / ``F-C-nn`` id keep ``id == ""``; their exact text is still compared.
    """
    entries: list[tuple[str, str]] = []
    problems: list[str] = []
    if not isinstance(items, list):
        return entries, ["limitations is missing or not a list"]
    for item in items:
        if not isinstance(item, str):
            problems.append("limitations holds a non-text entry")
            continue
        match = _LIM_ID.match(item.strip())
        entries.append((match.group(1) if match else "", item))
    return entries, problems


def accepted_limitation_tokens(limitations: Any) -> list[str]:
    """Sorted ``<id>:<sha256 of the exact text>`` for every id-bearing limitation (B6 ``Accepted limitations``)."""
    entries, _ = limitation_entries(limitations)
    return sorted(f"{lim_id}:{_sha256(text.encode('utf-8'))}" for lim_id, text in entries if lim_id)


def render_promoted_row(disposition: str) -> tuple[str, str]:
    """Canonical ``(status, notes)`` cells of a promoted FRZ-EVOL checklist row, rendered from a certified disposition.

    ``ELIGIBLE_PASS`` -> ``("PASS", "—")``; ``ELIGIBLE_PASS_SCOPED(a; b)`` -> ``("PASS — SCOPED", "a; b")``;
    ``ELIGIBLE_PASS_EVOL_X_SCOPED(a)`` -> ``("PASS (EVOL-X SCOPED)", "a")``.
    """
    head, _, rest = disposition.partition("(")
    qualifier = head.removeprefix("ELIGIBLE_PASS").strip("_")
    if not qualifier:
        status = "PASS"
    elif qualifier == "SCOPED":
        status = "PASS — SCOPED"
    else:
        status = f"PASS ({qualifier.replace('EVOL_X', 'EVOL-X').replace('_', ' ')})"
    notes = rest[:-1] if rest.endswith(")") else rest
    return status, notes.strip() or "—"


def blocker_key(text: str) -> str:
    """Stable blocker id: ``B-<n>``, ``B6-P3`` or a text digest id for unlabeled blockers."""
    stripped = text.strip()
    match = _BLOCKER_ID.match(stripped)
    if match:
        return match.group(1)
    if _BLOCKER_P3_TEXT.match(stripped):
        return BLOCKER_P3
    return "TXT-" + _sha256(stripped.encode("utf-8"))[:12]


# --------------------------------------------------------------------------- policy


def policy_mode(ctx: Context) -> tuple[str, list[str]]:
    """Return (mode, errors). Mode is ``IIAC``, ``LEGACY`` or ``INVALID``."""
    if not ctx.activation_record.is_file():
        return "INVALID", ["A-01 IIAC activation record is missing"]
    try:
        record = _strict_json_loads(ctx.activation_record.read_text(encoding="utf-8"))
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


def _baseline_documents(
    ctx: Context, manifest: dict[str, Any]
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, str]:
    """``(manifest, certification record, problem)`` exactly as published at ``certification_publication_sha`` (P2)."""
    if "scope_baseline" in ctx.cache:
        return ctx.cache["scope_baseline"]
    cert_sha = _p2_sha(ctx, manifest)
    docs: list[dict[str, Any] | None] = []
    problem = ""
    if not cert_sha or not _commit_exists(ctx, cert_sha):
        problem = "certification_publication_sha is not a commit"
    else:
        for rel in (REL_MANIFEST, REL_CERT_RECORD):
            raw = _show(ctx, cert_sha, rel)
            try:
                doc = _strict_json_loads(raw.decode("utf-8-sig")) if raw is not None else None
            except (UnicodeDecodeError, json.JSONDecodeError):
                doc = None
            if not isinstance(doc, dict):
                problem = f"{rel} is absent or malformed at the certification publication"
                doc = None
            docs.append(doc)
    result = (docs[0], docs[1], problem) if len(docs) == 2 else (None, None, problem)
    ctx.cache["scope_baseline"] = result
    return result


def _first_publication_sha(ctx: Context, rel: str) -> str:
    """The oldest commit that added ``rel`` ('' when git cannot say): the only legitimate P2 baseline."""
    if "first_publication" not in ctx.cache:
        ctx.cache["first_publication"] = {}
    cache: dict[str, str] = ctx.cache["first_publication"]
    if rel not in cache:
        proc = _git(ctx, "log", "--full-history", "--diff-filter=A", "--format=%H", ctx.head or "HEAD", "--", rel)
        shas = proc.stdout.decode("utf-8", errors="replace").split() if proc.returncode == 0 else []
        cache[rel] = shas[-1] if shas else ""
    return cache[rel]


def _p2_sha(ctx: Context, manifest: dict[str, Any] | None) -> str:
    """The P2 baseline commit: always the anchor of the evaluation context, never a field of the evidence under test."""
    del manifest  # DA-01: kept for call-site compatibility; the declared field has no authority
    return ctx.p2


def _digest8(text: str) -> str:
    return _sha256(text.encode("utf-8"))[:12]


def _scope_key_errors(
    document: dict[str, Any],
    baseline: dict[str, Any],
    label: str,
    *,
    lifecycle: frozenset[str],
    sealed: bool,
    compared_elsewhere: frozenset[str] = frozenset(),
) -> list[str]:
    """A-19: every non-lifecycle baseline key keeps its value, unknown new keys are rejected.

    ``compared_elsewhere`` names keys with a dedicated (order-insensitive) comparison, e.g. the manifest limitations."""
    errors: list[str] = []
    for key, value in document.items():
        if key in baseline:
            if key not in lifecycle and key not in compared_elsewhere and value != baseline[key]:
                kind = "scope/claim" if _SCOPE_KEY.search(key) else "sealed"
                errors.append(f"A-19 {label} {kind} key {key!r} differs from the certification publication")
        elif key not in lifecycle:
            kind = "scope/claim" if _SCOPE_KEY.search(key) else "unexpected"
            errors.append(f"A-19 {label} carries {kind} key {key!r} that is absent from the certification publication")
    if sealed:
        for key in baseline:
            if key not in document and key not in lifecycle:
                errors.append(f"A-19 {label} dropped key {key!r} published at the certification publication")
    return errors


def _p2_evidence_immutability_errors(ctx: Context, p2: str) -> list[str]:
    """DA-02: every P2 certification evidence file other than the lifecycle documents (manifest and record, whose
    legitimate drift is checked structurally) is byte-identical at HEAD, and the P2 evidence index entries still bind
    them; this freezes the assessment (findings F-C-*, evidence-only statement), the handoff and the validator runs."""
    raw_index = _show(ctx, p2, REL_CERT_INDEX)
    try:
        baseline = _strict_json_loads(raw_index.decode("utf-8-sig")) if raw_index is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        baseline = None
    files = baseline.get("files") if isinstance(baseline, dict) else None
    if not isinstance(files, dict) or not files:
        return ["A-19 the P2 certification evidence index is unavailable or malformed: failing closed"]
    current_raw = _show(ctx, ctx.head, REL_CERT_INDEX)
    try:
        current = _strict_json_loads(current_raw.decode("utf-8-sig")) if current_raw is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        current = None
    current_files = current.get("files") if isinstance(current, dict) else None
    if not isinstance(current_files, dict):
        return ["A-19 the certification evidence index at HEAD is unavailable or malformed"]
    errors: list[str] = []
    lifecycle_docs = {Path(REL_MANIFEST).name, Path(REL_CERT_RECORD).name}
    for name in sorted(files):
        if name in lifecycle_docs:
            if name not in current_files:
                errors.append(f"A-19 certification evidence index at HEAD no longer lists {name}")
            continue
        rel = f"{CERT_DIR}/{name}"
        before, after = _show(ctx, p2, rel), _show(ctx, ctx.head, rel)
        if before is None:
            errors.append(f"A-19 {name} is listed by the P2 evidence index but absent at the certification publication")
        elif after != before:
            errors.append(f"A-19 certification evidence {name} differs from the certification publication")
        if current_files.get(name) != files[name]:
            errors.append(
                f"A-19 certification evidence index entry for {name} differs from the certification publication"
            )
    return errors


_CERT_SHA_VALUE = re.compile(rb'("certification_publication_sha"\s*:\s*")[^"]*(")')


def _manifest_bytes_errors(ctx: Context, p2: str) -> list[str]:
    """DA-02 M0: the manifest blob at HEAD equals the P2 blob byte for byte except the ``certification_publication_sha``
    value (a commit cannot contain its own SHA). A pure reformat or a key reorder is therefore detected; the order of
    the ``limitations`` entries stays free (the multiset is compared separately), provided the file keeps the
    canonical serialization and key order of the P2 blob."""
    if not p2 or not ctx.head:
        return []
    baseline, current = _show(ctx, p2, REL_MANIFEST), _show(ctx, ctx.head, REL_MANIFEST)
    if baseline is None or current is None:
        return ["A-19 manifest blob unavailable at the certification publication or at HEAD"]
    if _CERT_SHA_VALUE.search(current) is None:
        return ["A-19 manifest at HEAD has no certification_publication_sha value"]
    expected = _CERT_SHA_VALUE.sub(lambda m: m.group(1) + p2.encode("ascii") + m.group(2), baseline, count=1)
    if current == expected:
        return []
    try:
        current_doc, baseline_doc = (
            _strict_json_loads(current.decode("utf-8")),
            _strict_json_loads(baseline.decode("utf-8")),
        )
    except (UnicodeDecodeError, json.JSONDecodeError):
        current_doc = baseline_doc = None
    if isinstance(current_doc, dict) and isinstance(baseline_doc, dict):

        def canonical(doc: dict[str, Any]) -> bytes:
            return (json.dumps(doc, indent=2, ensure_ascii=False) + "\n").encode("utf-8")

        if canonical(baseline_doc) == baseline and canonical(current_doc) == current:
            if list(current_doc) == list(baseline_doc):
                return []
    return [
        "A-19 manifest bytes differ from the certification publication "
        "(only the certification_publication_sha value and the limitation order may change)"
    ]


_CLOSURE_STATES = frozenset({"NOT_CLOSED", "CLOSED"})
_B6_STATES = frozenset({"NOT_ISSUED", "ISSUED", "ACCEPTED"})


def _lifecycle_value_errors(cert: dict[str, Any], base_cert: dict[str, Any] | None) -> list[str]:
    """W401-RA-02: lifecycle keys may change after P2, but only to values of the right shape. Free text in a
    lifecycle field would be a channel that no gate consumes yet is carried by a signed record."""
    errors: list[str] = []
    if "certification_publication_sha_binding" in cert:
        allowed = {P2_BINDING_TEXT}
        if base_cert is not None and "certification_publication_sha_binding" in base_cert:
            allowed.add(base_cert["certification_publication_sha_binding"])
        binding = cert["certification_publication_sha_binding"]
        if not isinstance(binding, str) or binding not in allowed:
            errors.append(
                "A-19 certification_publication_sha_binding is neither the text published at P2 nor the one "
                "binding text of the separate P2 binding commit"
            )
    if "frz_checklist_rows_changed" in cert and not isinstance(cert["frz_checklist_rows_changed"], bool):
        errors.append("A-19 frz_checklist_rows_changed must be a boolean")
    status = cert.get("stage_closure_status")
    if "stage_closure_status" in cert and not (isinstance(status, str) and status in _CLOSURE_STATES):
        errors.append(f"A-19 stage_closure_status must be one of {sorted(_CLOSURE_STATES)}")
    b6_state = cert.get("b6_acceptance")
    if "b6_acceptance" in cert and not (isinstance(b6_state, str) and b6_state in _B6_STATES):
        errors.append(f"A-19 b6_acceptance must be one of {sorted(_B6_STATES)}")
    for key in ("b6_acceptance_publication_sha", "formal_closure_publication_sha"):
        value = cert.get(key)
        if key in cert and value not in (None, "") and not (isinstance(value, str) and _SHA40.fullmatch(value)):
            errors.append(f"A-19 {key} must be null or a 40-hex commit id")
    path = cert.get("b6_acceptance_record_path")
    if "b6_acceptance_record_path" in cert and path not in (None, "") and path != REL_B6_RECORD:
        errors.append(f"A-19 b6_acceptance_record_path must be null or {REL_B6_RECORD}")
    digest = cert.get("b6_acceptance_record_digest")
    if "b6_acceptance_record_digest" in cert and digest not in (None, ""):
        if not (isinstance(digest, str) and _DIGEST64.fullmatch(digest)):
            errors.append("A-19 b6_acceptance_record_digest must be null or a 64-hex SHA-256")
    for key in ("open_blockers_for_closure", "resolved_blockers_for_closure"):
        if key in cert and not isinstance(cert[key], list):
            errors.append(f"A-19 {key} must be a list")
    return errors


def _frozen_directory_errors(ctx: Context, p2: str, rel_dir: str, label: str) -> list[str]:
    """W401-RA-02 / W403-02: ``rel_dir`` holds exactly the entries published at P2 (every entry type). An added
    file, symlink or gitlink would be indexed and signed alongside the record without any gate reading it."""
    before, after = set(_tree_entries(ctx, p2, rel_dir)), set(_tree_entries(ctx, ctx.head, rel_dir))
    errors = [
        f"A-19 {Path(rel).name} was added to the {label} directory after the certification publication"
        for rel in sorted(after - before)
    ]
    errors.extend(f"A-19 {Path(rel).name} was removed from the {label} directory" for rel in sorted(before - after))
    return errors


_EVIDENCE_DIRS = frozenset({"audit", "certification", "b4_reaudit", "operator_decisions", "b6_acceptance"})
_INDEX_NAME = re.compile(r"EVIDENCE_INDEX\.json$")


def _referenced_evidence_dirs(cert: dict[str, Any] | None) -> set[str]:
    """Evidence directories named by a blocker resolution (``closure_evidence_index_path``)."""
    resolved = (cert or {}).get("resolved_blockers_for_closure")
    found: set[str] = set()
    for entry in resolved if isinstance(resolved, list) else []:
        if isinstance(entry, dict) and isinstance(entry.get("closure_evidence_index_path"), str):
            found.add(posixpath.dirname(entry["closure_evidence_index_path"].replace("\\", "/")))
    return found


def _evidence_layout_errors(ctx: Context, cert: dict[str, Any] | None) -> list[str]:
    """W402-03 / W403-02: the EVOL-X evidence root holds only the known evidence directories and the directories a
    blocker resolution names; every file of an indexed directory is listed in its index. A file placed beside them,
    in an unreferenced directory or outside an index would be committed with the evidence without any gate reading
    it."""
    prefix = f"{EVIDENCE_ROOT_REL}/"
    referenced = _referenced_evidence_dirs(cert)
    by_dir: dict[str, list[str]] = {}
    errors: list[str] = []
    for rel in sorted(_tree_entries(ctx, ctx.head, EVIDENCE_ROOT_REL)):
        head_dir, _, rest = rel[len(prefix) :].partition("/")
        if not rest:
            errors.append(f"A-19 {rel} is outside the known EVOL-X evidence directories")
        else:
            by_dir.setdefault(head_dir, []).append(rest)
    for head_dir, files in sorted(by_dir.items()):
        dir_rel = f"{prefix}{head_dir}"
        index_names = [name for name in files if _INDEX_NAME.search(name) and "/" not in name]
        if head_dir not in _EVIDENCE_DIRS and (not index_names or dir_rel not in referenced):
            errors.append(
                f"A-19 {dir_rel}/ is neither a known EVOL-X evidence directory nor an indexed directory "
                "named by a blocker resolution"
            )
            continue
        if head_dir in ("audit", "certification") or not index_names:
            continue  # covered by the P2 freeze and by the index verification of the gates
        listing = _show(ctx, ctx.head, f"{dir_rel}/{index_names[0]}")
        try:
            document = _strict_json_loads(listing.decode("utf-8-sig")) if listing is not None else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            document = None
        indexed = document.get("files") if isinstance(document, dict) else None
        if not isinstance(indexed, dict):
            errors.append(f"A-19 {dir_rel}/{index_names[0]} has no readable files listing")
            continue
        errors.extend(
            f"A-19 {dir_rel}/{name} is not listed by the evidence index of its directory"
            for name in sorted(set(files) - set(indexed) - {index_names[0]})
        )
    return errors


def _gate_a_scope_exact(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any] | None, errors: list[str]) -> None:
    """RA-B4-05: certified scope equals the P2 baseline blob; nothing is decided by phrase search."""
    base_manifest, base_cert, problem = _baseline_documents(ctx, manifest)
    if problem or base_manifest is None:
        errors.append(f"A-19 scope baseline unavailable: {problem or 'manifest absent'}")
        return
    declared = _sha(manifest.get("certification_publication_sha"))
    anchor = ctx.p2
    if not anchor:
        errors.append("A-19 no trusted P2 certification publication anchor is available: failing closed")
        return
    if declared != anchor:
        errors.append(
            f"A-19 manifest certification_publication_sha {declared[:12] or '<empty>'} differs from the pinned P2 "
            f"certification publication {anchor[:12]}: the baseline cannot be re-pointed"
        )
    if cert is not None and _sha(cert.get("certification_publication_sha")) != anchor:
        errors.append(
            "A-19 certification record certification_publication_sha differs from the pinned P2 certification "
            f"publication {anchor[:12]}: the baseline cannot be re-pointed"
        )
    if not _is_strict_ancestor(ctx, anchor, ctx.head):
        errors.append(f"A-19 the pinned P2 certification publication {anchor[:12]} is not a strict ancestor of HEAD")
    first = _first_publication_sha(ctx, REL_CERT_RECORD)
    if not first:
        errors.append(
            "A-19 the commit that first published the certification record cannot be derived "
            "(history unavailable or the record was never committed): failing closed"
        )
    elif first != anchor:
        errors.append(
            "A-19 certification_publication_sha is not the commit that first published the certification record "
            f"({first[:12]}): the scope and blocker baseline cannot be moved to a later commit"
        )
    errors.extend(_manifest_bytes_errors(ctx, anchor))
    errors.extend(_p2_evidence_immutability_errors(ctx, anchor))
    errors.extend(_frozen_directory_errors(ctx, anchor, CERT_DIR, "certification"))
    errors.extend(_frozen_directory_errors(ctx, anchor, AUDIT_DIR, "audit"))
    errors.extend(_evidence_layout_errors(ctx, cert))

    current, structural = limitation_entries(manifest.get("limitations"))
    baseline, _ = limitation_entries(base_manifest.get("limitations"))
    errors.extend(f"A-19 manifest {message}" for message in structural)
    by_id = Counter(lim_id for lim_id, _ in current if lim_id)
    errors.extend(
        f"A-19 limitation id {lim_id} is listed {n} times (duplicate id)" for lim_id, n in by_id.items() if n > 1
    )
    now_counts, then_counts = Counter(current), Counter(baseline)
    for (lim_id, text), n in sorted((then_counts - now_counts).items()):
        errors.append(
            f"A-19 limitation {lim_id or '<unlabelled>'} [{_digest8(text)}] x{n} was dropped or reworded "
            "relative to the certification publication"
        )
    for (lim_id, text), n in sorted((now_counts - then_counts).items()):
        errors.append(
            f"A-19 limitation {lim_id or '<unlabelled>'} [{_digest8(text)}] x{n} was added or reworded "
            "relative to the certification publication (scope widening)"
        )
    errors.extend(
        _scope_key_errors(
            manifest,
            base_manifest,
            "manifest",
            lifecycle=frozenset({"certification_publication_sha"}),
            sealed=True,
            compared_elsewhere=frozenset({"limitations"}),
        )
    )
    if cert is None or base_cert is None:
        return
    if cert.get("certification_verdict") != base_cert.get("certification_verdict"):
        errors.append("A-19 certification_verdict differs from the certification publication")
    dispositions, base_dispositions = cert.get("frz_evol_dispositions"), base_cert.get("frz_evol_dispositions")
    if not isinstance(dispositions, dict) or not isinstance(base_dispositions, dict):
        errors.append("A-19 frz_evol_dispositions is not an object at HEAD or at the certification publication")
    else:
        for frz_id in sorted(set(dispositions) | set(base_dispositions)):
            if dispositions.get(frz_id) != base_dispositions.get(frz_id):
                errors.append(f"A-19 {frz_id} disposition differs from the certification publication")
    errors.extend(
        _scope_key_errors(cert, base_cert, "certification record", lifecycle=CERT_LIFECYCLE_KEYS, sealed=True)
    )
    errors.extend(_lifecycle_value_errors(cert, base_cert))


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


def _verify_index(
    ctx: Context, rel_index: str, rel_dir: str, committed_at: str, errors: list[str], *, code: str = "A-13"
) -> None:
    index = _read_json(ctx, rel_index, code, errors)
    if index is None:
        return
    files = index.get("files")
    if index.get("stage_id") != STAGE_ID or not isinstance(files, dict) or not files:
        errors.append(f"{code} {rel_index} is not an {STAGE_ID} evidence index")
        return
    root = ctx.path(rel_dir)
    on_disk = {p.relative_to(root).as_posix() for p in root.rglob("*") if p.is_file()} if root.is_dir() else set()
    index_name = Path(rel_index).name
    unindexed = sorted(on_disk - set(files) - {index_name})
    if unindexed:
        errors.append(f"{code} unindexed evidence in {rel_dir}: {unindexed[0]}")
    blobs = _tree_blobs(ctx, committed_at, rel_dir) if committed_at else {}
    if not blobs:
        errors.append(f"{code} no committed evidence for {rel_dir} at {committed_at or 'unknown SHA'}")
    for name, meta in files.items():
        reason = _path_rejected(name)
        if reason:
            errors.append(f"{code} index entry {name!r}: {reason}")
            continue
        raw = _read_bytes(ctx, f"{rel_dir}/{name}")
        expected = meta.get("sha256") if isinstance(meta, dict) else None
        if raw is None:
            errors.append(f"{code} indexed evidence missing: {rel_dir}/{name}")
            continue
        if _sha256(raw) != expected or (isinstance(meta, dict) and meta.get("bytes") != len(raw)):
            errors.append(f"{code} digest mismatch for {rel_dir}/{name}")
        if blobs and blobs.get(f"{rel_dir}/{name}") != _git_blob_id(raw):
            errors.append(f"{code} {rel_dir}/{name} differs from the bytes committed at {committed_at[:12]}")
    index_blob = blobs.get(rel_index)
    raw_index = _read_bytes(ctx, rel_index)
    if blobs and raw_index is not None and index_blob != _git_blob_id(raw_index):
        errors.append(f"{code} {rel_index} differs from the bytes committed at {committed_at[:12]}")


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


def blocking_finding_errors(findings: dict[str, Any], code: str) -> list[str]:
    """Unresolved blocking findings: blocking class/severity, or any finding that declares what it ``blocks``."""
    errors: list[str] = []
    summary = findings.get("summary") if isinstance(findings.get("summary"), dict) else {}
    for key in ("PROVEN_ARCHITECTURAL_DEFECT", "IN_SCOPE_BLOCKER", "blocking_for_scoped_pass"):
        if key in summary and summary.get(key) != 0:
            errors.append(f"{code} findings summary {key} is {summary.get(key)!r}, expected 0")
    for item in findings.get("findings") or []:
        if not isinstance(item, dict):
            continue
        cls = str(item.get("class") or "").upper()
        sev = str(item.get("severity") or "").upper()
        disp = str(item.get("disposition") or item.get("reaudit_disposition") or "").upper()
        blocking = cls in BLOCKING_FINDING_CLASSES or sev in BLOCKING_FINDING_SEVERITIES or bool(item.get("blocks"))
        if blocking and disp not in RESOLVED_DISPOSITIONS:
            errors.append(f"{code} unresolved blocking finding {item.get('id')} ({cls or '-'}/{sev or '-'})")
    return errors


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
    errors.extend(blocking_finding_errors({"findings": findings.get("findings")}, "A-15"))


def _handoff_errors(ctx: Context, manifest: dict[str, Any], rel: str, code: str) -> list[str]:
    errors: list[str] = []
    text = _read_text(ctx, rel)
    if text is None:
        return [f"{code} missing operator-attested handoff record {rel}"]
    if POLICY_VERSION not in text:
        errors.append(f"{code} handoff record does not cite the policy version")
    for key in ("implementation_session_id", "auditor_session_id", "certifier_session_id"):
        value = str(manifest.get(key) or "")
        if not value or value not in text:
            errors.append(f"{code} handoff record does not attest {key}")
    if re.search(r"^\s*- \[ \]", text, re.MULTILINE):
        errors.append(f"{code} handoff record has unattested operator checkboxes")
    if "________________" in text or not _HANDOFF_SIGNATURE.search(text):
        errors.append(f"{code} handoff record has no operator approval line")
    head_blob = _show(ctx, ctx.head, rel) if ctx.head else None
    if head_blob is None or head_blob.decode("utf-8-sig", errors="replace") != text:
        errors.append(f"{code} handoff record on disk differs from HEAD")
    return errors


def _gate_a_handoff(ctx: Context, manifest: dict[str, Any], errors: list[str]) -> None:
    errors.extend(_handoff_errors(ctx, manifest, REL_CERT_HANDOFF, "A-16"))


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


# --------------------------------------------------------------------------- blocker lifecycle (RA-B4-02)


@dataclass
class BlockerLedger:
    cert_sha: str
    baseline: dict[str, str] = field(default_factory=dict)
    open_now: dict[str, str] = field(default_factory=dict)
    resolved: dict[str, Any] = field(default_factory=dict)
    lineage: list[str] = field(default_factory=list)
    # Every blocker ever published in the certification record since the baseline: id -> (opened_at_sha, text).
    # Baseline blockers open at ``cert_sha``; later ones at the first published version that listed them (W4-F03).
    origin: dict[str, tuple[str, str]] = field(default_factory=dict)


def _blocker_texts(document: dict[str, Any] | None, label: str, lineage: list[str]) -> dict[str, str] | None:
    value = (document or {}).get("open_blockers_for_closure")
    if not isinstance(value, list):
        lineage.append(f"{label} open_blockers_for_closure is missing or not a list")
        return None
    out: dict[str, str] = {}
    for item in value:
        if not isinstance(item, str) or not item.strip():
            lineage.append(f"{label} open_blockers_for_closure holds a non-text or empty entry")
            continue
        key = blocker_key(item)
        if key in out:
            lineage.append(f"{label} open_blockers_for_closure lists blocker id {key} more than once (duplicate id)")
            continue
        out[key] = item.strip()
    return out


def _published_blocker_history(ctx: Context, cert_sha: str) -> dict[str, tuple[str, str]]:
    """Blockers first published in certification-record versions *after* ``cert_sha`` (oldest first)."""
    if not ctx.head or not cert_sha or not _commit_exists(ctx, cert_sha):
        return {}
    proc = _git(
        ctx, "log", "--full-history", "--reverse", "--format=%H", f"{cert_sha}..{ctx.head}", "--", REL_CERT_RECORD
    )
    if proc.returncode != 0:
        return {}
    seen: dict[str, tuple[str, str]] = {}
    for sha in proc.stdout.decode("utf-8", errors="replace").split():
        raw = _show(ctx, sha, REL_CERT_RECORD)
        try:
            doc = _strict_json_loads(raw.decode("utf-8-sig")) if raw is not None else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            continue
        texts = _blocker_texts(doc if isinstance(doc, dict) else None, "history", [])
        for key, text in (texts or {}).items():
            seen.setdefault(key, (sha, text))
    return seen


def blocker_ledger(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any]) -> BlockerLedger:
    """Compare the current blocker set with the baseline published at ``certification_publication_sha``."""
    cache_key = "blocker_ledger"
    if cache_key in ctx.cache:
        return ctx.cache[cache_key]
    cert_sha = _p2_sha(ctx, manifest)
    ledger = BlockerLedger(cert_sha=cert_sha)
    current = _blocker_texts(cert, "certification record", ledger.lineage)
    ledger.open_now = current or {}
    baseline_doc: dict[str, Any] | None = None
    if not cert_sha or not _commit_exists(ctx, cert_sha):
        ledger.lineage.append("blocker baseline unavailable: certification_publication_sha is not a commit")
    else:
        raw = _show(ctx, cert_sha, REL_CERT_RECORD)
        try:
            baseline_doc = _strict_json_loads(raw.decode("utf-8-sig")) if raw is not None else None
        except (UnicodeDecodeError, json.JSONDecodeError):
            baseline_doc = None
        if not isinstance(baseline_doc, dict):
            ledger.lineage.append("blocker baseline unavailable: certification record absent at the publication SHA")
            baseline_doc = None
    base = _blocker_texts(baseline_doc, "baseline", ledger.lineage) if baseline_doc is not None else None
    ledger.baseline = base or {}

    resolved_raw = cert.get("resolved_blockers_for_closure", [])
    if resolved_raw is None:
        resolved_raw = []
    if not isinstance(resolved_raw, list):
        ledger.lineage.append("resolved_blockers_for_closure is not a list")
        resolved_raw = []
    for entry in resolved_raw:
        key = str(entry.get("blocker_id") or "") if isinstance(entry, dict) else ""
        if not key:
            ledger.lineage.append("resolved blocker entry has no blocker_id")
        elif key in ledger.resolved:
            ledger.lineage.append(f"blocker {key} is resolved more than once")
        else:
            ledger.resolved[key] = entry
    ledger.origin = {key: (cert_sha, text) for key, text in ledger.baseline.items()}
    for key, origin in _published_blocker_history(ctx, cert_sha).items():
        ledger.origin.setdefault(key, origin)
    for key, (opened_at, text) in ledger.origin.items():
        later = key not in ledger.baseline
        label = (
            f"blocker {key} added after the certification publication ({opened_at[:12]})"
            if later
            else (f"baseline blocker {key}")
        )
        if key in ledger.open_now and key in ledger.resolved:
            ledger.lineage.append(f"{label} is listed as open and resolved at once")
        elif key not in ledger.open_now and key not in ledger.resolved and current is not None:
            ledger.lineage.append(
                f"{label} disappeared without closure evidence (removed, renamed or reclassified): {text}"
            )
    for key, text in ledger.open_now.items():
        published = ledger.origin.get(key)
        if published is not None and text != published[1]:
            ledger.lineage.append(
                f"open blocker {key} no longer carries the text it was published with (text changed or extended)"
            )
    for key in ledger.resolved:
        if ledger.origin and key not in ledger.origin:
            ledger.lineage.append(
                f"resolution for {key} does not match any blocker open at the certification publication"
            )
    ctx.cache[cache_key] = ledger
    return ledger


def _entry_path(entry: dict[str, Any], field_name: str, fail: Any) -> str | None:
    raw = str(entry.get(field_name) or "").replace("\\", "/")
    reason = _path_rejected(raw) if raw else "missing"
    if reason:
        fail(f"{field_name}: {reason}")
        return None
    if not raw.startswith(EVIDENCE_ROOT_REL + "/"):
        fail(f"{field_name} {raw!r} is outside {EVIDENCE_ROOT_REL}")
        return None
    return raw


def _reaudit_resolution_errors(
    ctx: Context, entry: dict[str, Any], closure: str, cert_sha: str, code: str, fail: Any
) -> None:
    index_rel = _entry_path(entry, "closure_evidence_index_path", fail)
    manifest_rel = _entry_path(entry, "reaudit_manifest_path", fail)
    resolved_impl = _sha(entry.get("resolved_implementation_sha"))
    if not resolved_impl or not _commit_exists(ctx, resolved_impl):
        fail("resolved_implementation_sha is not a commit")
        resolved_impl = ""
    elif not _is_strict_ancestor(ctx, cert_sha, resolved_impl):
        fail("resolved_implementation_sha must descend from the certification publication")
    if index_rel is None or manifest_rel is None:
        return
    index_dir = posixpath.dirname(index_rel)
    if index_dir in {AUDIT_DIR, CERT_DIR, EVIDENCE_ROOT_REL, B6_DIR, OPERATOR_DIR}:
        fail("re-audit evidence must live in its own directory, not in Session C audit/certification evidence")
        return
    if not manifest_rel.startswith(index_dir + "/"):
        fail("reaudit_manifest_path is not inside the indexed evidence directory")
        return
    sink: list[str] = []
    _verify_index(ctx, index_rel, index_dir, closure, sink, code=code)
    for message in sink:
        fail(f"closure evidence: {message.split(' ', 1)[1]}")
    index_raw = _read_bytes(ctx, index_rel)
    if index_raw is None or _sha256(index_raw) != entry.get("closure_evidence_index_digest"):
        fail("closure_evidence_index_digest does not match the evidence index")
    index_doc = _read_json(ctx, index_rel, code, sink) or {}
    if manifest_rel[len(index_dir) + 1 :] not in (index_doc.get("files") or {}):
        fail("re-audit manifest is not covered by the evidence index")
    raw = _read_bytes(ctx, manifest_rel)
    if raw is None:
        fail("re-audit manifest is missing")
        return
    if _sha256(raw) != entry.get("reaudit_manifest_digest"):
        fail("reaudit_manifest_digest does not match the re-audit manifest")
    try:
        manifest = _strict_json_loads(raw.decode("utf-8-sig"))
    except (UnicodeDecodeError, json.JSONDecodeError):
        fail("re-audit manifest is not valid JSON")
        return
    if not isinstance(manifest, dict):
        fail("re-audit manifest must be a JSON object")
        return
    schema: list[str] = []
    validate_manifest_dict(manifest, schema)
    for message in schema:
        fail(f"re-audit manifest: {message}")
    for message in _companion_errors(ctx.repo, ctx.path(manifest_rel)):
        fail(f"re-audit manifest: {message}")
    _reaudit_manifest_errors(ctx, entry, manifest, resolved_impl, closure, index_dir, index_doc, fail)


def _reaudit_manifest_errors(
    ctx: Context,
    entry: dict[str, Any],
    manifest: dict[str, Any],
    resolved_impl: str,
    closure: str,
    index_dir: str,
    index_doc: dict[str, Any],
    fail: Any,
) -> None:
    if manifest.get("stage_id") != STAGE_ID:
        fail(f"re-audit manifest stage_id != {STAGE_ID}")
    if manifest.get("policy_version") != POLICY_VERSION:
        fail(f"re-audit manifest policy_version != {POLICY_VERSION}")
    if manifest.get("certification_profile") != "iiac-v1" or manifest.get("assurance_level") != "process_enforced":
        fail("re-audit manifest profile/assurance_level mismatch")
    if str(manifest.get("verdict") or "").upper() not in ACCEPTABLE_VERDICTS:
        fail(f"re-audit verdict {manifest.get('verdict')!r} is not an independent acceptance")
    if manifest.get("manifest_status") not in (None, ""):
        fail("re-audit manifest is still a draft (manifest_status set)")
    sessions = {
        k: str(manifest.get(k) or "")
        for k in ("implementation_session_id", "auditor_session_id", "certifier_session_id")
    }
    if any(
        not v or v.startswith("PENDING") for v in (sessions["auditor_session_id"], sessions["certifier_session_id"])
    ):
        fail("re-audit auditor/certifier session is missing or PENDING")
    if len(set(sessions.values())) != 3 or "" in sessions.values():
        fail("re-audit implementer, auditor and certifier sessions must be three distinct sessions")
    agents = [str(manifest.get(k) or "") for k in ("implementation_agent_id", "auditor_agent_id", "certifier_agent_id")]
    if len(set(agents)) != 3:
        fail("re-audit agent ids must be distinct (self-certification)")
    if manifest.get("auditor_agent_id") != AUDITOR_AGENT_ID:
        fail(f"re-audit auditor_agent_id must be {AUDITOR_AGENT_ID}")
    if manifest.get("certifier_agent_id") != CERTIFIER_AGENT_ID:
        fail(f"re-audit certifier_agent_id must be {CERTIFIER_AGENT_ID}")
    audited = _sha(manifest.get("audited_implementation_sha"))
    if not audited or not _commit_exists(ctx, audited):
        fail("re-audit audited_implementation_sha is not a commit")
    else:
        if resolved_impl and not _is_ancestor(ctx, resolved_impl, audited):
            fail("re-audit audited_implementation_sha does not contain the resolved corrective implementation")
        if not _is_strict_ancestor(ctx, audited, closure):
            fail("closure_publication_sha must descend from the audited implementation")
    disposition = entry.get("certifier_disposition")
    if not isinstance(disposition, dict):
        fail("certifier_disposition is missing")
    else:
        if disposition.get("disposition") != "ACCEPTED":
            fail("certifier_disposition.disposition must be ACCEPTED")
        if disposition.get("certifier_agent_id") != CERTIFIER_AGENT_ID:
            fail(f"certifier_disposition.certifier_agent_id must be {CERTIFIER_AGENT_ID}")
        if disposition.get("certifier_session_id") != manifest.get("certifier_session_id"):
            fail("certifier_disposition session differs from the re-audit manifest certifier")
        if disposition.get("policy_version") != POLICY_VERSION:
            fail("certifier_disposition.policy_version mismatch")
        for key in ("audit_report_digest", "findings_digest"):
            if disposition.get(key) != manifest.get(key):
                fail(f"certifier_disposition {key} differs from the re-audit manifest")
    files = index_doc.get("files") or {}
    for path_key, digest_key in (("audit_report_path", "audit_report_digest"), ("findings_path", "findings_digest")):
        rel = str(manifest.get(path_key) or "").replace("\\", "/")
        raw = _read_bytes(ctx, rel) if rel and _path_rejected(rel) is None else None
        if raw is None or not rel.startswith(index_dir + "/") or rel[len(index_dir) + 1 :] not in files:
            fail(f"re-audit manifest {path_key} is missing or not indexed")
        elif _sha256(raw) != manifest.get(digest_key):
            fail(f"re-audit {digest_key} does not match {rel}")
        elif path_key == "findings_path":
            try:
                findings = _strict_json_loads(raw.decode("utf-8-sig"))
            except (UnicodeDecodeError, json.JSONDecodeError):
                fail("re-audit findings register is not valid JSON")
                continue
            if isinstance(findings, dict):
                for message in [*_findings_errors(findings), *blocking_finding_errors(findings, "findings")]:
                    fail(f"re-audit {message}")
            else:
                fail("re-audit findings register must be a JSON object")
    handoff_rel = str(manifest.get("handoff_record_path") or "").replace("\\", "/")
    if not handoff_rel.startswith(index_dir + "/") or handoff_rel[len(index_dir) + 1 :] not in files:
        fail("re-audit handoff record is missing or not indexed")
    else:
        for message in _handoff_errors(ctx, manifest, handoff_rel, "handoff"):
            fail(f"re-audit {message}")


def _signed_not_before(ctx: Context, sha: str) -> datetime | None:
    """Earliest acceptable operator-signature instant: the commit time of the publication that opened the matter."""
    when = _commit_time(ctx, sha) if sha else None
    return when if when is not None and when.tzinfo is not None else None


def _operator_resolution_errors(
    ctx: Context, key: str, entry: dict[str, Any], closure: str, fail: Any, opened_at_sha: str = ""
) -> None:
    items = entry.get("closure_evidence")
    if not isinstance(items, list) or not items:
        fail("closure_evidence must list the operator decision records")
        return
    required = BLOCKER_REQUIRED_DECISIONS.get(key, frozenset())
    seen: set[str] = set()
    for item in items:
        if not isinstance(item, dict):
            fail("closure_evidence entry is not an object")
            continue
        for extra in sorted(set(item) - _CLOSURE_ITEM_KEYS):
            fail(f"closure_evidence item carries unexpected key {extra!r}")
        raw_path = str(item.get("path") or "").replace("\\", "/")
        reason = _path_rejected(raw_path) if raw_path else "missing"
        if reason or not raw_path.startswith(OPERATOR_DIR + "/"):
            fail(f"operator record path {raw_path!r} is not a decision record under {OPERATOR_DIR}")
            continue
        published = _show(ctx, closure, raw_path)
        current = _read_bytes(ctx, raw_path)
        if published is None:
            fail(f"operator record {raw_path} is absent at closure_publication_sha")
            continue
        if current != published:
            fail(f"operator record {raw_path} differs from the bytes published at closure_publication_sha")
        if _sha256(published) != item.get("sha256"):
            fail(f"operator record {raw_path} digest does not match closure_evidence")
        decision_id = str(item.get("decision_id") or "")
        tokens = OD_RECORD_REQUIREMENTS.get(decision_id) or BLOCKER_SCOPE_TOKENS.get(key, ())
        if not tokens:
            fail(f"no scope contract for decision {decision_id!r}")
            continue
        record_errors, _ = operator_record_errors(
            published.decode("utf-8-sig", errors="replace"),
            decision_id=decision_id,
            scope_tokens=tokens,
            not_before=_signed_not_before(ctx, opened_at_sha),
        )
        for message in record_errors:
            fail(f"{raw_path}: {message}")
        seen.add(decision_id)
    if required - seen:
        fail(f"missing operator decision record(s): {sorted(required - seen)}")
    if not required and not seen:
        fail("no operator authorization record")
    if not required and seen & set(OD_RECORD_REQUIREMENTS):
        fail("an OD-EVOL-12/OD-EVOLX-08 record cannot authorize this blocker")


_RESOLUTION_KEYS = frozenset(
    {
        "blocker_id",
        "status",
        "baseline_text_sha256",
        "opened_at_sha",
        "resolution_kind",
        "closure_publication_sha",
    }
)
_REAUDIT_RESOLUTION_KEYS = frozenset(
    {
        "resolved_implementation_sha",
        "closure_evidence_index_path",
        "closure_evidence_index_digest",
        "reaudit_manifest_path",
        "reaudit_manifest_digest",
        "certifier_disposition",
    }
)
_OPERATOR_RESOLUTION_KEYS = frozenset({"closure_evidence"})
_CLOSURE_ITEM_KEYS = frozenset({"path", "sha256", "decision_id"})
_DISPOSITION_KEYS = frozenset(
    {
        "disposition",
        "certifier_agent_id",
        "certifier_session_id",
        "policy_version",
        "audit_report_digest",
        "findings_digest",
    }
)


def resolution_errors(ctx: Context, key: str, entry: Any, baseline_text: str, cert_sha: str, code: str) -> list[str]:
    """Independently re-verify an evidence-bound blocker resolution."""
    errors: list[str] = []

    def fail(message: str) -> None:
        errors.append(f"{code} blocker {key} resolution: {message}")

    if not isinstance(entry, dict):
        fail("entry is not an object")
        return errors
    allowed = _RESOLUTION_KEYS | (
        _REAUDIT_RESOLUTION_KEYS
        if BLOCKER_CLOSURE_KINDS.get(key, KIND_REAUDIT) == KIND_REAUDIT
        else _OPERATOR_RESOLUTION_KEYS
    )
    for extra in sorted(set(entry) - allowed):
        fail(f"unexpected key {extra!r} (resolution entries carry only structured, gate-checked fields)")
    disposition = entry.get("certifier_disposition")
    if isinstance(disposition, dict):
        for extra in sorted(set(disposition) - _DISPOSITION_KEYS):
            fail(f"unexpected certifier_disposition key {extra!r}")
    if entry.get("status") != RESOLUTION_STATUS:
        fail(f"status must be {RESOLUTION_STATUS}")
    if entry.get("baseline_text_sha256") != _sha256(baseline_text.encode("utf-8")):
        fail("baseline_text_sha256 does not match the blocker as opened at the certification publication")
    if entry.get("opened_at_sha") != cert_sha:
        fail("opened_at_sha must be the publication that opened the blocker")
    kind = BLOCKER_CLOSURE_KINDS.get(key, KIND_REAUDIT)
    if entry.get("resolution_kind") != kind:
        fail(f"resolution_kind must be {kind}")
    closure = _sha(entry.get("closure_publication_sha"))
    if not closure or not _commit_exists(ctx, closure):
        fail("closure_publication_sha is not a commit")
        return errors
    if not cert_sha or not _is_strict_ancestor(ctx, cert_sha, closure):
        fail("closure must be published after the certification publication")
    if not ctx.head or not _is_strict_ancestor(ctx, closure, ctx.head):
        fail("closure_publication_sha must be a strict ancestor of HEAD (no self-reference)")
    if kind == KIND_REAUDIT:
        _reaudit_resolution_errors(ctx, entry, closure, cert_sha, code, fail)
    else:
        _operator_resolution_errors(ctx, key, entry, closure, fail, cert_sha)
    return errors


def _gate_a_core(ctx: Context) -> list[str]:
    errors: list[str] = []
    governance = _read_json(ctx, REL_GOVERNANCE, "A-02", errors)
    if governance is not None and governance.get("normative_effect") == "IN_FORCE":
        errors.append("A-02 superseded ECAP-P4-EVOL-SCOPE-AUTO-B6 claims IN_FORCE alongside IIAC (dual activation)")
    manifest = _gate_a_manifest(ctx, errors)
    cert = _read_json(ctx, REL_CERT_RECORD, "A-17", errors)
    if manifest is None:
        return errors
    _gate_a_scope(ctx, manifest, cert, errors)
    _gate_a_scope_exact(ctx, manifest, cert, errors)
    _gate_a_roles(manifest, cert, errors)
    _gate_a_shas(ctx, manifest, cert, errors)
    _gate_a_digests(ctx, manifest, cert, errors)
    _gate_a_pass_record(ctx, manifest, cert, errors)
    _gate_a_findings(ctx, manifest, errors)
    _gate_a_handoff(ctx, manifest, errors)
    if cert is not None:
        _gate_a_cert_record(ctx, manifest, cert, errors)
        ledger = blocker_ledger(ctx, manifest, cert)
        errors.extend(f"A-18 blocker lifecycle: {message}" for message in ledger.lineage)
        for key, text in ledger.open_now.items():
            if key not in GATE_B_ONLY_BLOCKERS:
                errors.append(f"A-18 open certification blocker for B6 acceptance: {text}")
        for key, entry in ledger.resolved.items():
            if key in ledger.origin and key not in GATE_B_ONLY_BLOCKERS and key not in ledger.open_now:
                opened_at, text = ledger.origin[key]
                errors.extend(resolution_errors(ctx, key, entry, text, opened_at, "A-18"))
    return errors


# --------------------------------------------------------------------------- Gate B


def _b6_record_exactness_errors(
    ctx: Context, text: str, info: dict[str, Any], manifest: dict[str, Any], verdict: str
) -> list[str]:
    """B-11 (RA-B4-05): the B6 record carries only allow-listed lines and the baseline limitation set, verbatim."""
    errors: list[str] = []

    def fail(message: str) -> None:
        errors.append(f"B-11 B6 acceptance record: {message}")

    fenced = False
    titled = False
    for number, line in enumerate(text.lstrip("﻿").splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith(("```", "~~~")):
            fenced = not fenced
            fail(f"line {number}: fenced blocks are not allowed")
        elif fenced:
            continue
        elif _RECORD_SIGNATURE.match(line):
            continue
        elif not titled and line == B6_RECORD_TITLE:
            titled = True
        else:
            item = _RECORD_FIELD.match(line)
            if item is None or item.group(1).strip() not in B6_RECORD_FIELDS:
                fail(f"line {number} is not an allow-listed field (unexpected or scope-widening text)")
    if not titled:
        fail(f"title line must be exactly {B6_RECORD_TITLE!r}")
    fields: dict[str, list[str]] = info["fields"]
    for key in sorted(set(fields) - B6_RECORD_FIELDS):
        fail(f"unexpected field {key!r}")
    decision_ids = fields.get("Decision ID", [])
    if len(decision_ids) == 1 and not _B6_DECISION_ID.fullmatch(decision_ids[0]):
        fail("Decision ID must be a bare identifier")
    scope = fields.get("Decision scope", [])
    canonical_scope = f"B6 acceptance for {STAGE_ID} ({verdict}, evidence-bound)"
    if len(scope) == 1 and scope[0] != canonical_scope:
        fail(f"Decision scope must be exactly {canonical_scope!r}")

    base_manifest, _, problem = _baseline_documents(ctx, manifest)
    if problem or base_manifest is None:
        fail(f"limitation baseline unavailable: {problem or 'manifest absent'}")
        return errors
    expected_tokens = accepted_limitation_tokens(base_manifest.get("limitations"))
    lines = fields.get("Accepted limitations", [])
    if len(lines) != 1:
        fail(f"'Accepted limitations' must be stated exactly once (found {len(lines)})")
    else:
        got = sorted(token.strip() for token in lines[0].split(",") if token.strip())
        if got != expected_tokens:
            fail(
                "'Accepted limitations' must equal the certified (id, sha256 of text) set "
                f"({len(expected_tokens)} entries: <id>:<sha256>); got {len(got)}"
            )
    return errors


def _b6_record_errors(ctx: Context, b6: str, manifest: dict[str, Any], cert: dict[str, Any]) -> list[str]:
    """RA-B4-04: the B6 acceptance record must exist at the pinned publication commit and match the audit."""
    errors: list[str] = []

    def fail(message: str) -> None:
        errors.append(f"B-05 {message}")

    cert_sha = _p2_sha(ctx, manifest)
    if str(cert.get("b6_acceptance_record_path") or "").replace("\\", "/") != REL_B6_RECORD:
        fail("certification record does not bind the canonical B6 acceptance record path")
    digest = str(cert.get("b6_acceptance_record_digest") or "")
    if not _DIGEST64.fullmatch(digest):
        fail("certification record has no b6_acceptance_record_digest")
    published = _show(ctx, b6, REL_B6_RECORD)
    if published is None:
        fail(f"B6 acceptance record is absent at B6_ACCEPTANCE_PUBLICATION_SHA {b6[:12]}")
        return errors
    if cert_sha and _show(ctx, cert_sha, REL_B6_RECORD) is not None:
        fail("B6 acceptance record already existed at the certification publication (not a new acceptance)")
    if _read_bytes(ctx, REL_B6_RECORD) != published:
        fail("B6 acceptance record differs from the bytes published at the B6 SHA")
    if _sha256(published) != digest:
        fail("B6 acceptance record digest does not match the certification record")
    sink: list[str] = []
    _verify_index(ctx, REL_B6_INDEX, B6_DIR, b6, sink, code="B-05")
    errors.extend(sink)
    manifest_raw = _read_bytes(ctx, REL_MANIFEST) or b""
    if _show(ctx, b6, REL_MANIFEST) != manifest_raw:
        fail("IIAC manifest changed after the B6 acceptance publication")
    phase1_raw = _show(ctx, b6, REL_CERT_RECORD)
    try:
        phase1 = _strict_json_loads(phase1_raw.decode("utf-8-sig")) if phase1_raw is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        phase1 = None
    if not isinstance(phase1, dict):
        fail("certification record is absent at the B6 acceptance publication")
    else:
        if str(phase1.get("b6_acceptance") or "").upper() not in {"ISSUED", "ACCEPTED"}:
            fail("certification record at the B6 SHA does not issue the acceptance")
        if phase1.get("b6_acceptance_record_digest") != digest:
            fail("certification record at the B6 SHA does not bind the acceptance record digest")
        if phase1.get("b6_acceptance_publication_sha") not in (None, ""):
            fail("B6 phase-1 certification record must leave b6_acceptance_publication_sha empty")

    text = published.decode("utf-8-sig", errors="replace")
    verdict = str(manifest.get("verdict") or "").upper()
    expected = {
        "Stage": STAGE_ID,
        "Policy version": POLICY_VERSION,
        "Accepted verdict": verdict,
        "Independent audit manifest digest": _sha256(manifest_raw),
        "Audit report digest": str(manifest.get("audit_report_digest") or ""),
        "Findings digest": str(manifest.get("findings_digest") or ""),
        "Audited implementation SHA": str(manifest.get("audited_implementation_sha") or ""),
        "Certification publication SHA": cert_sha,
        "Certifier session": str(manifest.get("certifier_session_id") or ""),
        "Certifier agent": CERTIFIER_AGENT_ID,
    }
    record_errors, info = operator_record_errors(
        text, decision_id=None, scope_tokens=B6_SCOPE_TOKENS, expected=expected
    )
    for message in record_errors:
        fail(f"B6 acceptance record: {message}")
    errors.extend(_b6_record_exactness_errors(ctx, text, info, manifest, verdict))
    signed_at = info.get("signed_at")
    cert_time, b6_time = _commit_time(ctx, cert_sha) if cert_sha else None, _commit_time(ctx, b6)
    if isinstance(signed_at, datetime) and signed_at.tzinfo is not None:
        if cert_time is not None and signed_at < cert_time:
            fail("operator signature predates the certification publication")
        if b6_time is not None and signed_at > b6_time + SIGNATURE_SKEW:
            fail("operator signature post-dates the B6 acceptance publication commit")
    return errors


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
    if cert.get("b6_acceptance") not in ("ISSUED", "ACCEPTED"):
        errors.append("B-02 certification record b6_acceptance is not ISSUED")
    if b6 and cert.get("b6_acceptance_publication_sha") != b6:
        errors.append("B-02 certification record b6_acceptance_publication_sha differs from governance JSON")
    if not b6:
        return ""
    legacy = {ladder_sha(rows, role) for role in LEGACY_LADDER_ROWS} - {""}
    if b6 in legacy:
        errors.append("B-03 B6 SHA is a historical/internal legacy publication, not IIAC B6 acceptance")
    cert_sha = _p2_sha(ctx, manifest)
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
        phase1 = _strict_json_loads(blob.decode("utf-8-sig")) if blob is not None else None
    except (UnicodeDecodeError, json.JSONDecodeError):
        phase1 = None
    if not isinstance(phase1, dict) or phase1.get("formal_b6_status") != "ACCEPTED":
        errors.append("B-04 governance JSON at B6_ACCEPTANCE_PUBLICATION_SHA does not publish the acceptance")
    elif phase1.get("b6_acceptance_publication_sha") not in (None, ""):
        errors.append("B-04 B6 phase-1 publication must leave b6_acceptance_publication_sha empty")
    errors.extend(_b6_record_errors(ctx, b6, manifest, cert))
    ledger = ctx.cache.get("blocker_ledger")
    p3 = ledger.resolved.get(BLOCKER_P3) if isinstance(ledger, BlockerLedger) else None
    p3_sha = _sha(p3.get("closure_publication_sha")) if isinstance(p3, dict) else ""
    if p3_sha and not _is_ancestor(ctx, p3_sha, b6):
        errors.append("B-05 B6 acceptance must be published after the operator P3 authorization")
    return b6


def _baseline_dispositions(ctx: Context, manifest: dict[str, Any]) -> dict[str, Any] | None:
    _, base_cert, problem = _baseline_documents(ctx, manifest)
    value = (base_cert or {}).get("frz_evol_dispositions")
    return value if not problem and isinstance(value, dict) else None


def _gate_b_frz(ctx: Context, manifest: dict[str, Any], cert: dict[str, Any], errors: list[str]) -> None:
    text = _read_text(ctx, REL_FRZ_CHECKLIST)
    if text is None:
        errors.append(f"B-06 missing {REL_FRZ_CHECKLIST}")
        return
    rows = parse_frz_rows(text)
    dispositions = cert.get("frz_evol_dispositions") if isinstance(cert.get("frz_evol_dispositions"), dict) else {}
    baseline_dispositions = _baseline_dispositions(ctx, manifest)
    if baseline_dispositions is None:
        errors.append(
            "B-11 FRZ promotion baseline unavailable: certified dispositions absent at the certification publication"
        )
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
        if baseline_dispositions is not None:
            want_status, want_notes = render_promoted_row(str(baseline_dispositions.get(frz_id) or ""))
            got_status, got_notes = _norm(cells[4]), _norm(cells[5]) if len(cells) > 5 else ""
            if len(cells) != 6 or (got_status, got_notes) != (want_status, want_notes):
                errors.append(
                    f"B-11 {frz_id} promoted row must equal the row rendered from the certified disposition "
                    f"(status {want_status!r}, notes {want_notes!r}); got status {got_status!r}, notes {got_notes!r}"
                )


def _decision_record_errors(ctx: Context, od_id: str, notes: str) -> list[str]:
    paths = sorted(set(_RECORD_NOTE_PATH.findall(notes)))
    digests = sorted(set(_RECORD_NOTE_DIGEST.findall(notes)))
    if len(paths) != 1 or len(digests) != 1:
        return [f"{od_id} register notes must cite exactly one decision record path and one sha256"]
    rel = f"{OPERATOR_DIR}/{paths[0]}"
    raw = _read_bytes(ctx, rel)
    if raw is None:
        return [f"{od_id} decision record {rel} is missing"]
    errors: list[str] = []
    if _show(ctx, ctx.head, rel) != raw:
        errors.append(f"{od_id} decision record {rel} is not committed at HEAD")
    if _sha256(raw) != digests[0]:
        errors.append(f"{od_id} decision record digest does not match the register notes")
    manifest = _read_json(ctx, REL_MANIFEST, "B-07", [])
    cert_sha = _p2_sha(ctx, manifest)
    record_errors, _ = operator_record_errors(
        raw.decode("utf-8-sig", errors="replace"),
        decision_id=od_id,
        scope_tokens=OD_RECORD_REQUIREMENTS[od_id],
        not_before=_signed_not_before(ctx, cert_sha),
    )
    errors.extend(f"{od_id} decision record: {message}" for message in record_errors)
    return errors


def _gate_b_operators(ctx: Context, errors: list[str]) -> None:
    text = _read_text(ctx, REL_REGISTER)
    if text is None:
        errors.append(f"B-07 missing {REL_REGISTER}")
        return
    rows = parse_register_rows(text)
    for od_id in CLOSURE_OPERATOR_DECISIONS:
        found = rows.get(od_id, [])
        if not found:
            errors.append(f"B-07 operator decision {od_id} not recorded in the register status table")
            continue
        if len(found) != 1:
            errors.append(f"B-07 operator decision {od_id} is ambiguous ({len(found)} status rows)")
            continue
        status, date_text, notes = found[0]
        reasons = decision_status_errors(status, date_text)
        if reasons:
            errors.append(
                f"B-07 operator decision {od_id} not recorded as decided "
                f"(status {status!r}, date {date_text!r}): {reasons[0]}"
            )
            continue
        if od_id in OD_RECORD_REQUIREMENTS:
            errors.extend(f"B-07 {message}" for message in _decision_record_errors(ctx, od_id, notes))


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


def _gate_b_core(ctx: Context) -> list[str]:
    errors: list[str] = []
    sink: list[str] = []
    manifest = _read_json(ctx, REL_MANIFEST, "B-00", sink)
    cert = _read_json(ctx, REL_CERT_RECORD, "B-00", sink)
    if manifest is None or cert is None:
        return ["B-00 IIAC manifest or certification record unavailable"]
    ledger = blocker_ledger(ctx, manifest, cert)
    b6 = _gate_b_b6(ctx, manifest, cert, errors)
    _gate_b_frz(ctx, manifest, cert, errors)
    _gate_b_operators(ctx, errors)
    _gate_b_ladder(ctx, b6, cert, errors)
    errors.extend(f"B-09 blocker lifecycle: {message}" for message in ledger.lineage)
    for key, text in ledger.open_now.items():
        errors.append(f"B-09 unresolved mandatory blocker ({key}): {text}")
    for key, entry in ledger.resolved.items():
        if key in GATE_B_ONLY_BLOCKERS and key in ledger.origin and key not in ledger.open_now:
            opened_at, text = ledger.origin[key]
            errors.extend(resolution_errors(ctx, key, entry, text, opened_at, "B-09"))
    return errors


def gate_a(ctx: Context) -> list[str]:
    """Gate A plus the repository-integrity (RA-01) and HEAD-binding (RA-03) checks each gate enforces itself."""
    errors = _gate_a_core(ctx)
    errors.extend(_repository_integrity_errors(ctx, "A-22"))
    errors.extend(_dirty_input_errors(ctx, "A-21"))
    return errors


def gate_b(ctx: Context) -> list[str]:
    errors = _gate_b_core(ctx)
    errors.extend(_repository_integrity_errors(ctx, "B-13"))
    errors.extend(_dirty_input_errors(ctx, "B-12"))
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
        status, date_text = register.get(od_id, ("", ""))
        if not decision_recorded(status, date_text):
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
    p2_sha: str | None = None,
) -> PreflightResult:
    repo = (repo or _REPO).resolve()
    canonical = repo == _REPO.resolve()
    if canonical and evidence_root is not None:
        msg = "A-01 evidence-root override is forbidden for the canonical repository"
        return PreflightResult("INVALID", [msg], ["B-00 Gate A not satisfied"])
    if canonical and p2_sha is not None and p2_sha != P2_CERTIFICATION_PUBLICATION_SHA:
        msg = "A-19 P2 anchor override is forbidden for the canonical repository"
        return PreflightResult("INVALID", [msg], ["B-00 Gate A not satisfied"])
    # DA-01: the trusted P2 is never read from the repository under test. Fixture repositories may supply their own
    # commit; without one the pinned constant applies (absent from a synthetic repository -> fail closed).
    p2 = P2_CERTIFICATION_PUBLICATION_SHA if canonical else (p2_sha or P2_CERTIFICATION_PUBLICATION_SHA)
    record = activation_record or (CANONICAL_RECORD if canonical else repo / REL_ACTIVATION)
    ctx = Context(repo=repo, activation_record=record, evidence_root=evidence_root, p2=p2)
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
    try:
        a_errors = gate_a(ctx)
        b_errors = gate_b(ctx)
    except Exception as exc:  # noqa: BLE001 - an unexpected evidence shape must fail closed, never crash the verdict
        message = f"{type(exc).__name__}: {exc}"
        return PreflightResult(
            "INVALID", [f"A-99 internal error while evaluating the evidence ({message})"], ["B-00 Gate A not satisfied"]
        )
    if a_errors:
        b_errors = [f"B-00 Gate A not satisfied ({len(a_errors)} error(s))", *b_errors]
    return PreflightResult(mode, a_errors, b_errors)


def _repository_class(repo: Path) -> str:
    return "canonical" if repo.resolve() == _REPO.resolve() else "fixture"


def _print(result: PreflightResult, gate: str, repo: Path) -> int:
    label = "canonical" if _repository_class(repo) == "canonical" else "FIXTURE (not canonical evidence)"
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
    parser.add_argument(
        "--p2-sha",
        default=None,
        help="Fixture-only P2 certification publication commit; rejected for the canonical repository",
    )
    parser.add_argument("--json", action="store_true", help="Emit a JSON result instead of text")
    args = parser.parse_args(argv)
    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(errors="replace")
    result = evaluate(
        args.repo, activation_record=args.activation_record, evidence_root=args.evidence_root, p2_sha=args.p2_sha
    )
    if args.json:
        payload = {
            "repository_class": _repository_class(args.repo),
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

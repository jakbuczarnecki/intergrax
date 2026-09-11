# © Artur Czarnecki. All rights reserved.

"""Source-freeze verification for behavioral qualification analysis."""

from __future__ import annotations

import json
from dataclasses import dataclass
from enum import StrEnum
from pathlib import Path

from testing_support.decision_e2e.local_ai_incident_qualification import (
    R4R1_MODEL_NAME,
    semantic_source_groups_for_r4r1,
)
from testing_support.decision_e2e.local_qualification_session.artifact_finalization_contract import (
    QualificationArtifactFinalizationContract,
)
from testing_support.decision_e2e.local_qualification_session.contracts import (
    SourceFingerprintSnapshot,
    SourceBlobFingerprint,
)
from testing_support.decision_e2e.local_qualification_session.source_fingerprint import (
    capture_semantic_source_fingerprint,
)


class SourceFreezeStatus(StrEnum):
    PASS = "PASS"
    FAIL = "FAIL"


@dataclass(frozen=True, slots=True)
class SourceFreezeCheck:
    name: str
    passed: bool
    detail: str


@dataclass(frozen=True, slots=True)
class SourceFreezeReport:
    status: SourceFreezeStatus
    checks: tuple[SourceFreezeCheck, ...]


def _normalize_digest(value: str) -> str:
    stripped = value.strip()
    if stripped.startswith("sha256:"):
        return stripped
    if stripped.startswith("sha256-"):
        return "sha256:" + stripped.removeprefix("sha256-")
    return stripped


def _load_frozen_snapshot(session_dir: Path) -> SourceFingerprintSnapshot | None:
    checkpoint_path = session_dir / "session-checkpoint.json"
    if not checkpoint_path.is_file():
        return None
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    frozen = payload.get("frozen_source")
    if not isinstance(frozen, dict):
        return None
    blobs_raw = frozen.get("blobs")
    head = frozen.get("repository_head_sha")
    if not isinstance(blobs_raw, list) or not isinstance(head, str):
        return None
    blobs: list[SourceBlobFingerprint] = []
    for item in blobs_raw:
        if not isinstance(item, dict):
            continue
        path = item.get("path")
        content_hash = item.get("content_hash")
        semantic_group = item.get("semantic_group")
        if (
            isinstance(path, str)
            and isinstance(content_hash, str)
            and isinstance(semantic_group, str)
        ):
            blobs.append(
                SourceBlobFingerprint(
                    path=path,
                    content_hash=content_hash,
                    semantic_group=semantic_group,
                )
            )
    return SourceFingerprintSnapshot(repository_head_sha=head, blobs=tuple(blobs))


def _load_spec_identity(session_dir: Path) -> dict[str, str] | None:
    checkpoint_path = session_dir / "session-checkpoint.json"
    if not checkpoint_path.is_file():
        return None
    payload = json.loads(checkpoint_path.read_text(encoding="utf-8"))
    spec = payload.get("spec")
    if not isinstance(spec, dict):
        return None
    identity = spec.get("experiment_identity")
    if not isinstance(identity, dict):
        return None
    config_fp = identity.get("config_fingerprint")
    source_fp = identity.get("source_fingerprint")
    model_digest = identity.get("model_digest")
    runtime_version = identity.get("provider_runtime_version")
    if not all(
        isinstance(value, str)
        for value in (config_fp, source_fp, model_digest, runtime_version)
    ):
        return None
    return {
        "config_fingerprint": config_fp,
        "source_fingerprint": source_fp,
        "model_digest": model_digest,
        "provider_runtime_version": runtime_version,
        "model_name": str(identity.get("model_name", "")),
    }


def verify_source_freeze(session_dir: Path, repo_root: Path) -> SourceFreezeReport:
    checks: list[SourceFreezeCheck] = []

    def add(name: str, passed: bool, detail: str) -> None:
        checks.append(SourceFreezeCheck(name=name, passed=passed, detail=detail))

    if not session_dir.is_dir():
        add("session directory", False, f"missing: {session_dir}")
        return SourceFreezeReport(SourceFreezeStatus.FAIL, tuple(checks))

    for required in ("runs.json", "summary.json", "final-report.md", "artifact-manifest.txt"):
        path = session_dir / required
        add(required, path.is_file(), "present" if path.is_file() else "missing")

    manifest_valid = False
    try:
        QualificationArtifactFinalizationContract.validate_manifest_checksums(session_dir)
        manifest_valid = True
        add("artifact-manifest", True, "VALID")
        add("checksum", True, "PASS")
    except ValueError as exc:
        add("artifact-manifest", False, f"INVALID: {exc}")
        add("checksum", False, "FAIL")

    summary_path = session_dir / "summary.json"
    summary_integrity: dict[str, object] = {}
    if summary_path.is_file():
        summary_payload = json.loads(summary_path.read_text(encoding="utf-8"))
        integrity = summary_payload.get("session_integrity")
        if isinstance(integrity, dict):
            summary_integrity = integrity

    runtime_match = summary_integrity.get("runtime_identity_status") == "match"
    add(
        "runtime identity",
        runtime_match,
        "MATCH" if runtime_match else str(summary_integrity.get("runtime_identity_status")),
    )

    model_match = summary_integrity.get("model_identity_status") == "match"
    add(
        "model digest",
        model_match,
        "MATCH" if model_match else str(summary_integrity.get("model_identity_status")),
    )

    source_match = summary_integrity.get("source_identity_status") == "match"
    add(
        "source fingerprint",
        source_match,
        "MATCH" if source_match else str(summary_integrity.get("source_identity_status")),
    )

    config_match = summary_integrity.get("config_identity_status") == "match"
    add(
        "configuration fingerprint",
        config_match,
        "MATCH" if config_match else str(summary_integrity.get("config_identity_status")),
    )

    frozen = _load_frozen_snapshot(session_dir)
    if frozen is not None:
        current = capture_semantic_source_fingerprint(
            repo_root,
            semantic_source_groups=semantic_source_groups_for_r4r1(),
            repository_head_sha=frozen.repository_head_sha,
        )
        frozen_map = {blob.path: blob.content_hash for blob in frozen.blobs}
        current_map = {blob.path: blob.content_hash for blob in current.blobs}
        drift = [
            path
            for path, digest in frozen_map.items()
            if current_map.get(path) != digest
        ]
        add(
            "semantic source blob drift",
            not drift,
            "none" if not drift else f"{len(drift)} blob(s) changed",
        )
        if frozen.semantic_fingerprint() != current.semantic_fingerprint():
            add("source fingerprint replay", False, "semantic fingerprint mismatch")
        else:
            add("source fingerprint replay", True, "MATCH")
    else:
        add("semantic source blob drift", False, "session-checkpoint.json missing")

    profile_path = session_dir / "local_model_profile.json"
    spec_identity = _load_spec_identity(session_dir)
    if profile_path.is_file() and spec_identity is not None:
        profile = json.loads(profile_path.read_text(encoding="utf-8"))
        profile_digest = profile.get("digest") or profile.get("model_digest")
        profile_model = profile.get("model")
        profile_runtime = profile.get("runtime_version")
        digest_ok = (
            isinstance(profile_digest, str)
            and _normalize_digest(profile_digest)
            == _normalize_digest(spec_identity["model_digest"])
        )
        add(
            "local_model_profile digest",
            digest_ok,
            "MATCH" if digest_ok else "MISMATCH",
        )
        runtime_ok = profile_runtime == spec_identity["provider_runtime_version"]
        add(
            "local_model_profile runtime",
            runtime_ok,
            "MATCH" if runtime_ok else "MISMATCH",
        )
        model_ok = profile_model == spec_identity.get("model_name", R4R1_MODEL_NAME)
        add("local_model_profile model", model_ok, "MATCH" if model_ok else "MISMATCH")

    if manifest_valid:
        add("artifact-manifest validation", True, "PASS")

    all_passed = all(item.passed for item in checks)
    return SourceFreezeReport(
        status=SourceFreezeStatus.PASS if all_passed else SourceFreezeStatus.FAIL,
        checks=tuple(checks),
    )

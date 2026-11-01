# © Artur Czarnecki. All rights reserved.

"""W6-03: Protocol 03 verifier. Mode A = evidence publication checkout, Mode B = test execution checkout.

The verifier reads committed bytes through ``git show`` (never the working tree) and derives
EVIDENCE_PUBLICATION_SHA from ``git rev-parse HEAD``; the manifest must not carry a self-hash (Part V-A).
"""

from __future__ import annotations

import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from testing_support.change_safety.contracts.artifacts import (
    ArtifactDigests,
    EvolJunitOutcomes,
    EvolMetricsArtifact,
    ExperimentVerdict,
    FrzEvidence,
    P03Manifest,
    TestExecutionManifest,
)
from testing_support.change_safety.selection.ci_manifest import manifest_content_digest

BUNDLE_REL = "docs/project/maintainers/qualification/evol_x/evol_x_p03_independent_replay"
MANIFEST_NAME = "MANIFEST.json"
DIGESTS_NAME = "ARTIFACT_DIGESTS.json"
JUNIT_NAME = "evol_junit_outcomes.v1.json"
TEST_EXECUTION_MANIFEST_NAME = "evol_test_execution_manifest.v1.json"
METRICS_NAME = "evol_metrics.v1.json"
FRZ_NAME = "evol_frz_evidence.v1.json"
INVENTORY_BINDING_NAME = "inventory_binding.v1.json"
EXPERIMENTS_DIR = "experiments"
FORBIDDEN_MANIFEST_FIELDS = (
    "evidence_publication_sha",
    "evidence_head",
    "evidence_sha",
    "b6_acceptance_publication_sha",
    "formal_closure_publication_sha",
)
EXPECTED_EXPERIMENT_VERDICTS = 16
EXPECTED_FRZ_IDS = tuple(f"FRZ-EVOL-{i:02d}" for i in range(1, 11))
FRZ_SCOPE_FOOTNOTE_REQUIRED = ("FRZ-EVOL-07", "FRZ-EVOL-08", "FRZ-EVOL-10")


def sha256_hex(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def _git(repo: Path, *args: str) -> subprocess.CompletedProcess[bytes]:
    return subprocess.run(["git", *args], cwd=repo, capture_output=True, check=False)


def head_sha(repo: Path) -> str:
    proc = _git(repo, "rev-parse", "HEAD")
    return proc.stdout.decode().strip() if proc.returncode == 0 else ""


def show_bytes(repo: Path, ref: str, rel: str) -> bytes | None:
    proc = _git(repo, "show", f"{ref}:{rel}")
    return proc.stdout if proc.returncode == 0 else None


def is_ancestor(repo: Path, ancestor: str, descendant: str) -> bool:
    return _git(repo, "merge-base", "--is-ancestor", ancestor, descendant).returncode == 0


def _load_json(repo: Path, ref: str, rel: str, problems: list[str]) -> Any | None:
    raw = show_bytes(repo, ref, rel)
    if raw is None:
        problems.append(f"missing artifact @ {ref[:9]}: {rel}")
        return None
    try:
        return json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        problems.append(f"unreadable JSON {rel}: {exc}")
        return None


def _load_manifest(repo: Path, ref: str, bundle_rel: str, problems: list[str]) -> P03Manifest | None:
    raw = _load_json(repo, ref, f"{bundle_rel}/{MANIFEST_NAME}", problems)
    if raw is None:
        return None
    for field in FORBIDDEN_MANIFEST_FIELDS:
        if field in raw:
            problems.append(f"MANIFEST contains forbidden self-referential field: {field}")
    if problems:
        return None
    try:
        return P03Manifest.model_validate(raw)
    except ValidationError as exc:
        problems.append(f"MANIFEST invalid: {exc.error_count()} error(s): {exc.errors()[0]['msg']}")
        return None


def compare_outcomes(archived: dict[str, str], replayed: dict[str, str]) -> list[str]:
    """Canonical test identity -> outcome map comparison (cross-commit semantic replay, not XML bytes)."""
    problems = [
        f"outcome swapped for {n}: archived={archived[n]} replay={replayed[n]}"
        for n in sorted(archived)
        if n in replayed and archived[n] != replayed[n]
    ]
    problems += [f"missing in replay: {n}" for n in sorted(set(archived) - set(replayed))]
    problems += [f"unexpected in replay: {n}" for n in sorted(set(replayed) - set(archived))]
    return problems


_ANCESTRY_MESSAGES = {
    "same": "evidence HEAD equals test_execution_sha (publication must be a separate descendant commit)",
    "reversed": "execution after evidence (reversed ancestry)",
    "unrelated": "test_execution_sha is not an ancestor of evidence HEAD",
}


def classify_ancestry(repo: Path, test_execution_sha: str, evidence_head: str) -> str:
    """``ok`` only when TEST_EXECUTION_SHA is a strict ancestor of the evidence commit."""
    if test_execution_sha == evidence_head:
        return "same"
    if is_ancestor(repo, test_execution_sha, evidence_head):
        return "ok"
    return "reversed" if is_ancestor(repo, evidence_head, test_execution_sha) else "unrelated"


def _check_binding(name: str, sha: str, expected: str, problems: list[str]) -> None:
    if sha != expected:
        problems.append(f"stale source_commit_sha in {name}: {sha[:9]} != test_execution_sha {expected[:9]}")


def _check_artifact_payloads(
    repo: Path, head: str, bundle_rel: str, manifest: P03Manifest, problems: list[str]
) -> None:
    tes = manifest.test_execution_sha

    def load(name: str) -> Any | None:
        return _load_json(repo, head, f"{bundle_rel}/{name}", problems)

    junit_raw, tem_raw, metrics_raw = load(JUNIT_NAME), load(TEST_EXECUTION_MANIFEST_NAME), load(METRICS_NAME)
    binding, frz_raw = load(INVENTORY_BINDING_NAME), load(FRZ_NAME)
    if binding is not None:
        _check_binding(INVENTORY_BINDING_NAME, str(binding.get("source_commit_sha", "")), tes, problems)
    outcomes: dict[str, str] = {}
    if junit_raw is not None:
        junit = EvolJunitOutcomes.model_validate(junit_raw)
        _check_binding(JUNIT_NAME, junit.source_commit_sha, tes, problems)
        outcomes = dict(junit.outcomes)
        problems += [f"non-passed outcome: {n}={o}" for n, o in sorted(outcomes.items()) if o != "passed"]
    if tem_raw is not None:
        tem = TestExecutionManifest.model_validate(tem_raw)
        _check_binding(TEST_EXECUTION_MANIFEST_NAME, tem.source_commit_sha, tes, problems)
        if manifest_content_digest(tem.node_ids) != tem.manifest_content_digest:
            problems.append("test execution manifest content digest mismatch")
        problems += [f"missing manifest node in canonical outcomes: {n}" for n in tem.node_ids if n not in outcomes]
    if metrics_raw is not None:
        metrics = EvolMetricsArtifact.model_validate(metrics_raw)
        _check_binding(METRICS_NAME, metrics.source_commit_sha, tes, problems)
        problems += [
            f"metric not passing: {r.metric_id}"
            for r in metrics.readings
            if r.status != "MEASURED" or r.passed is not True
        ]
    if frz_raw is not None:
        rows = [FrzEvidence.model_validate(r) for r in frz_raw.get("rows", [])]
        if tuple(r.frz_id for r in rows) != EXPECTED_FRZ_IDS:
            problems.append("FRZ evidence rows are not exactly FRZ-EVOL-01..10")
        for row in rows:
            _check_binding(f"{FRZ_NAME}:{row.frz_id}", row.source_commit_sha, tes, problems)
            if not row.passed:
                problems.append(f"FRZ behavioral test not passed: {row.frz_id}")
            if row.frz_id in FRZ_SCOPE_FOOTNOTE_REQUIRED and not row.scope_footnote:
                problems.append(f"scope footnote missing: {row.frz_id}")
    _check_experiments(repo, head, bundle_rel, manifest, problems)


def _check_experiments(repo: Path, head: str, bundle_rel: str, manifest: P03Manifest, problems: list[str]) -> None:
    prefix = f"{EXPERIMENTS_DIR}/"
    paths = sorted(p for p in manifest.artifact_paths if p.startswith(prefix))
    if len(paths) != EXPECTED_EXPERIMENT_VERDICTS:
        problems.append(f"experiment verdicts: {len(paths)} != {EXPECTED_EXPERIMENT_VERDICTS}")
    for rel in paths:
        raw = _load_json(repo, head, f"{bundle_rel}/{rel}", problems)
        if raw is None:
            continue
        verdict = ExperimentVerdict.model_validate(raw)
        _check_binding(rel, verdict.source_commit_sha, manifest.test_execution_sha, problems)
        if not verdict.passed:
            problems.append(f"experiment arm not passed: {rel}")


def verify_mode_a(repo: Path, *, bundle_rel: str = BUNDLE_REL) -> list[str]:
    """Mode A (verify-evidence-publish): ``HEAD`` is the evidence publication commit."""
    problems: list[str] = []
    head = head_sha(repo)
    manifest = _load_manifest(repo, head, bundle_rel, problems)
    if manifest is None:
        return problems or ["MANIFEST unusable"]
    tes = manifest.test_execution_sha
    ancestry = classify_ancestry(repo, tes, head)
    if ancestry != "ok":
        problems.append(_ANCESTRY_MESSAGES[ancestry])
    digests_raw = _load_json(repo, head, f"{bundle_rel}/{DIGESTS_NAME}", problems)
    if digests_raw is not None:
        digests = ArtifactDigests.model_validate(digests_raw).digests
        if set(digests) != set(manifest.artifact_paths):
            problems.append("ARTIFACT_DIGESTS keys differ from MANIFEST artifact_paths")
        for rel in manifest.artifact_paths:
            raw = show_bytes(repo, head, f"{bundle_rel}/{rel}")
            if raw is None:
                problems.append(f"artifact missing @ evidence HEAD: {rel}")
            elif digests.get(rel) != sha256_hex(raw):
                problems.append(f"tampered ARTIFACT_DIGESTS entry: {rel}")
    _check_artifact_payloads(repo, head, bundle_rel, manifest, problems)
    return problems


def verify_mode_b(
    repo: Path,
    *,
    evidence_ref: str,
    bundle_rel: str = BUNDLE_REL,
    replay_outcomes: dict[str, str] | None = None,
) -> list[str]:
    """Mode B (verify-test-execution): ``HEAD`` must equal the manifest ``test_execution_sha``."""
    problems: list[str] = []
    manifest = _load_manifest(repo, evidence_ref, bundle_rel, problems)
    if manifest is None:
        return problems or ["MANIFEST unusable"]
    head = head_sha(repo)
    if head != manifest.test_execution_sha:
        problems.append(f"HEAD {head[:9]} != MANIFEST test_execution_sha {manifest.test_execution_sha[:9]}")
        return problems
    if replay_outcomes is not None:
        junit_raw = _load_json(repo, evidence_ref, f"{bundle_rel}/{JUNIT_NAME}", problems)
        if junit_raw is not None:
            problems += compare_outcomes(dict(EvolJunitOutcomes.model_validate(junit_raw).outcomes), replay_outcomes)
    return problems


def write_digests(bundle_dir: Path, artifact_paths: tuple[str, ...]) -> ArtifactDigests:
    digests = ArtifactDigests(
        digests={rel: sha256_hex((bundle_dir / rel).read_bytes()) for rel in sorted(artifact_paths)}
    )
    (bundle_dir / DIGESTS_NAME).write_text(
        json.dumps(digests.model_dump(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return digests

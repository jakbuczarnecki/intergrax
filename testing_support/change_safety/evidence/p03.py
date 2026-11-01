# © Artur Czarnecki. All rights reserved.

"""W6-03: Protocol 03 verifier. Mode A = evidence publication checkout, Mode B = test execution checkout.

The verifier reads committed bytes through ``git show`` (never the working tree) and derives
EVIDENCE_PUBLICATION_SHA from ``git rev-parse HEAD``; the manifest must not carry a self-hash (Part V-A).
"""

from __future__ import annotations

import hashlib
import json
import statistics
import subprocess
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from testing_support.change_safety.contracts.artifacts import (
    ArtifactDigests,
    EvolJunitOutcomes,
    EvolMetricsArtifact,
    ExperimentVerdict,
    FrzEvidence,
    M09DrillCaseRegistry,
    M09HumanDrillArtifact,
    P03Manifest,
    TestExecutionManifest,
)
from testing_support.change_safety.contracts.canonical_json import stable_digest_hex
from testing_support.change_safety.diagnostics.m09_cause_identity import expected_cause_identities_for_seed
from testing_support.change_safety.discovery.closed_world import build_closed_world_inventory
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
M09_HUMAN_DRILL_NAME = "evol_m09_human_drill.v2.json"
M09_HUMAN_DRILL_SCHEMA_VERSION = "evol_m09_human_drill.v2"
M09_MEASUREMENT_POLICY_ID = "OD-EVOL-08-M1"
M09_DRILL_CASE_REGISTRY_NAME = "evol_m09_drill_case_registry.v1.json"
M09_HUMAN_DRILL_WORKSHEET_NAME = "evol_m09_human_drill_worksheet.v1.json"
# OD-EVOL-08-M1: tool runtime and human attribution clock are separate measurements with separate SLAs.
M09_TOOL_PER_CASE_SLA_SECONDS = 120.0
M09_ATTRIBUTION_MEDIAN_SLA_SECONDS = 900.0
M09_EXPECTED_CASE_IDS = ("C1", "C2", "C3")
M03_CAUSAL_AUDIT_NAME = "m03_causal_audit.v1.json"
METRIC_M09T = "EVOL-M-09T"
METRIC_M09H = "EVOL-M-09H"
METRIC_M09_COMPOSITE = "EVOL-M-09"
CW09_DISPOSITION_ENV = "EVOL_CW09_DISPOSITION_ID"
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


def m09h_population_digest(registry: M09DrillCaseRegistry) -> str:
    payload: list[str] = []
    for entry in registry.cases:
        payload.append(entry.changed_path)
        payload.append(entry.input_digest)
        payload.extend(entry.expected_cause_identities)
    return stable_digest_hex(payload)


def _load_m09_registry(repo: Path, head: str, bundle_rel: str, problems: list[str]) -> M09DrillCaseRegistry | None:
    rel = f"{bundle_rel}/{M09_DRILL_CASE_REGISTRY_NAME}"
    raw = _load_json(repo, head, rel, problems)
    if raw is None:
        problems.append(f"final certification requires {M09_DRILL_CASE_REGISTRY_NAME} @ evidence HEAD")
        return None
    try:
        return M09DrillCaseRegistry.model_validate(raw)
    except ValidationError as exc:
        problems.append(f"{M09_DRILL_CASE_REGISTRY_NAME} invalid: {exc.errors()[0]['msg']}")
        return None


def _parse_utc_timestamp(label: str, value: str, problems: list[str]) -> datetime | None:
    try:
        parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    except ValueError:
        problems.append(f"M09 human drill {label} is not valid ISO-8601: {value!r}")
        return None
    if parsed.tzinfo is None:
        parsed = parsed.replace(tzinfo=UTC)
    return parsed


# Attestation boundary: Mode A verifies schema, bindings, arithmetic and SLA thresholds of the *submitted* human
# drill artifact. It cannot prove that a human performed the drill or that the clock was honestly measured
# (OD-EVOL-08-M1 clock start/stop); that attestation belongs to the independent B6 audit, never to the verifier.
def _check_m09_human_drill(
    repo: Path,
    head: str,
    bundle_rel: str,
    tes: str,
    drill_raw: Any,
    metrics: EvolMetricsArtifact,
    manifest: P03Manifest,
    problems: list[str],
) -> None:
    registry = _load_m09_registry(repo, head, bundle_rel, problems)
    if registry is None:
        return
    # Strict v2 contract first: v1 / unknown schema and missing/other policy id fail closed (no v1 fallback).
    raw_schema = drill_raw.get("schema_version") if isinstance(drill_raw, dict) else None
    if raw_schema != M09_HUMAN_DRILL_SCHEMA_VERSION:
        problems.append(
            f"M09 human drill schema_version must be exactly {M09_HUMAN_DRILL_SCHEMA_VERSION}: {raw_schema!r}"
        )
        return
    raw_policy = drill_raw.get("measurement_policy_id")
    if raw_policy != M09_MEASUREMENT_POLICY_ID:
        problems.append(
            f"M09 human drill measurement_policy_id must be exactly {M09_MEASUREMENT_POLICY_ID}: {raw_policy!r}"
        )
        return
    if registry.bound_test_execution_sha != tes:
        problems.append(
            "M09 drill registry bound_test_execution_sha "
            f"{registry.bound_test_execution_sha[:9]} != test_execution_sha {tes[:9]}"
        )
    try:
        drill = M09HumanDrillArtifact.model_validate(drill_raw)
    except ValidationError as exc:
        problems.append(f"{M09_HUMAN_DRILL_NAME} invalid: {exc.errors()[0]['msg']}")
        return
    _check_binding(M09_HUMAN_DRILL_NAME, drill.source_commit_sha, tes, problems)
    if drill.test_execution_sha != tes:
        problems.append(f"M09 human drill test_execution_sha {drill.test_execution_sha[:9]} != {tes[:9]}")
    registry_bytes = show_bytes(repo, head, f"{bundle_rel}/{M09_DRILL_CASE_REGISTRY_NAME}")
    if registry_bytes is not None and drill.registry_digest != sha256_hex(registry_bytes):
        problems.append("M09 human drill registry_digest does not match committed registry bytes")
    registry_by_id = {entry.case_id: entry for entry in registry.cases}
    for entry in registry.cases:
        raw_path = show_bytes(repo, tes, entry.changed_path)
        if raw_path is None:
            problems.append(f"M09 seed path missing @ test_execution_sha: {entry.changed_path}")
            continue
        if sha256_hex(raw_path) != entry.input_digest:
            problems.append(f"M09 registry input_digest mismatch for {entry.case_id} @ test_execution_sha")
    try:
        inventory = build_closed_world_inventory(source_commit_sha=tes)
    except (OSError, ValueError) as exc:
        problems.append(f"M09 registry oracle: cannot build closed-world inventory @ TES: {exc}")
        inventory = None
    if inventory is not None:
        for entry in registry.cases:
            try:
                derived = expected_cause_identities_for_seed(inventory, entry.changed_path)
            except ValueError as exc:
                problems.append(f"M09 registry oracle derivation failed for {entry.case_id}: {exc}")
                continue
            if tuple(entry.expected_cause_identities) != derived:
                problems.append(
                    f"M09 registry expected_cause_identities mismatch for {entry.case_id} "
                    "(committed registry != diagnostic oracle)"
                )
    tool_seconds: list[float] = []
    attribution_seconds: list[float] = []
    for result in drill.cases:
        spec = registry_by_id.get(result.case_id)
        if spec is None:
            problems.append(f"M09 human drill unknown case_id: {result.case_id}")
            continue
        if not result.correct:
            problems.append(f"M09 human drill case not marked correct: {result.case_id}")
        if result.tool_seconds > M09_TOOL_PER_CASE_SLA_SECONDS:
            problems.append(
                f"M09 human drill tool_seconds>{M09_TOOL_PER_CASE_SLA_SECONDS:.0f}s for "
                f"{result.case_id}: {result.tool_seconds}"
            )
        tool_seconds.append(result.tool_seconds)
        attribution_seconds.append(result.attribution_seconds)
        expected = tuple(spec.expected_cause_identities)
        if tuple(result.attributed_cause_identities) != expected:
            problems.append(f"M09 human drill attributed causes mismatch for {result.case_id}")
    # Tool timing: per-case SLA only (no median SLA). The median is checked for consistency only.
    if tool_seconds:
        computed_tool_median = statistics.median(tool_seconds)
        if round(drill.median_tool_seconds, 3) != round(computed_tool_median, 3):
            problems.append("M09 human drill median_tool_seconds disagrees with computed case median")
    # Human attribution clock: SLA applies to the median (strict ">" fails; equality allowed).
    if attribution_seconds:
        computed_attribution_median = statistics.median(attribution_seconds)
        if computed_attribution_median > M09_ATTRIBUTION_MEDIAN_SLA_SECONDS:
            problems.append(
                f"M09 human drill median attribution_seconds>{M09_ATTRIBUTION_MEDIAN_SLA_SECONDS:.0f}s: "
                f"{computed_attribution_median}"
            )
        if round(drill.median_attribution_seconds, 3) != round(computed_attribution_median, 3):
            problems.append("M09 human drill median_attribution_seconds disagrees with computed case median")
    if not drill.adversarial_control_passed:
        problems.append("M09 human drill adversarial_control_passed must be true")
    started = _parse_utc_timestamp("drill_started_at_utc", drill.drill_started_at_utc, problems)
    completed = _parse_utc_timestamp("drill_completed_at_utc", drill.drill_completed_at_utc, problems)
    if started is not None and completed is not None and completed < started:
        problems.append("M09 human drill drill_completed_at_utc precedes drill_started_at_utc")
    if not drill.evaluator_id.strip() or not drill.evaluator_display_name.strip():
        problems.append("M09 human drill evaluator fields must be non-empty")
    if M09_HUMAN_DRILL_WORKSHEET_NAME in manifest.artifact_paths:
        if drill.worksheet_digest is None:
            problems.append(
                f"M09 human drill worksheet_digest required when {M09_HUMAN_DRILL_WORKSHEET_NAME} is bundled"
            )
        else:
            worksheet_bytes = show_bytes(repo, head, f"{bundle_rel}/{M09_HUMAN_DRILL_WORKSHEET_NAME}")
            if worksheet_bytes is None:
                problems.append(f"final certification missing {M09_HUMAN_DRILL_WORKSHEET_NAME} @ evidence HEAD")
            elif drill.worksheet_digest != sha256_hex(worksheet_bytes):
                problems.append("M09 human drill worksheet_digest mismatch")
    m09h = next((r for r in metrics.readings if r.metric_id == METRIC_M09H), None)
    if m09h is None:
        problems.append(f"final certification requires {METRIC_M09H} reading in metrics artifact")
        return
    expected_population = m09h_population_digest(registry)
    if m09h.numerator != 3 or m09h.denominator != 3 or m09h.status != "MEASURED" or m09h.passed is not True:
        problems.append(f"final certification requires {METRIC_M09H} 3/3 MEASURED PASS")
    if m09h.population_digest != expected_population:
        problems.append(f"{METRIC_M09H} population_digest inconsistent with M09 drill registry")


def _check_metrics_readings(metrics: EvolMetricsArtifact, problems: list[str], *, technical: bool) -> None:
    by_id = {r.metric_id: r for r in metrics.readings}
    composite = by_id.get(METRIC_M09_COMPOSITE)
    if composite is not None and composite.passed is True:
        problems.append(f"composite {METRIC_M09_COMPOSITE} must not pass without independent {METRIC_M09H} PASS")
    m09h = by_id.get(METRIC_M09H)
    if technical:
        if m09h is not None and m09h.status == "MEASURED" and m09h.passed is True:
            problems.append(f"technical evidence must not claim {METRIC_M09H} PASS (human drill pending)")
    elif m09h is None or m09h.status != "MEASURED" or m09h.passed is not True:
        problems.append(f"final certification requires {METRIC_M09H} MEASURED and PASS")
    for r in metrics.readings:
        if r.metric_id == METRIC_M09H and technical and r.status == "NOT_MEASURED":
            continue
        if r.metric_id == METRIC_M09_COMPOSITE:
            continue
        if r.status != "MEASURED" or r.passed is not True:
            problems.append(f"metric not passing: {r.metric_id}")


def _check_artifact_payloads(
    repo: Path, head: str, bundle_rel: str, manifest: P03Manifest, problems: list[str], *, technical: bool
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
        _check_metrics_readings(metrics, problems, technical=technical)
    if not technical:
        drill_rel = f"{bundle_rel}/{M09_HUMAN_DRILL_NAME}"
        if M09_HUMAN_DRILL_NAME not in manifest.artifact_paths:
            problems.append(f"final certification requires {M09_HUMAN_DRILL_NAME} in MANIFEST artifact_paths")
        elif show_bytes(repo, head, drill_rel) is None:
            problems.append(f"final certification missing {M09_HUMAN_DRILL_NAME} @ evidence HEAD")
        elif metrics_raw is not None:
            metrics_for_drill = EvolMetricsArtifact.model_validate(metrics_raw)
            drill_payload = _load_json(repo, head, drill_rel, problems)
            if drill_payload is not None:
                _check_m09_human_drill(
                    repo, head, bundle_rel, tes, drill_payload, metrics_for_drill, manifest, problems
                )
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


def verify_mode_a(repo: Path, *, bundle_rel: str = BUNDLE_REL, technical: bool = False) -> list[str]:
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
    _check_artifact_payloads(repo, head, bundle_rel, manifest, problems, technical=technical)
    return problems


def verify_mode_b(
    repo: Path,
    *,
    evidence_ref: str,
    bundle_rel: str = BUNDLE_REL,
    replay_outcomes: dict[str, str] | None = None,
    certification: bool = False,
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
    if certification and replay_outcomes is None:
        problems.append("certification Mode B requires replay_outcomes (--replay-junit)")
        return problems
    junit_raw = _load_json(repo, evidence_ref, f"{bundle_rel}/{JUNIT_NAME}", problems)
    tem_raw = _load_json(repo, evidence_ref, f"{bundle_rel}/{TEST_EXECUTION_MANIFEST_NAME}", problems)
    if replay_outcomes is not None and junit_raw is not None:
        archived = dict(EvolJunitOutcomes.model_validate(junit_raw).outcomes)
        problems += compare_outcomes(archived, replay_outcomes)
    if certification and tem_raw is not None and replay_outcomes is not None:
        tem = TestExecutionManifest.model_validate(tem_raw)
        problems += [f"missing manifest node in replay: {n}" for n in tem.node_ids if n not in replay_outcomes]
        problems += [
            f"non-passed replay outcome: {n}={replay_outcomes[n]}"
            for n in tem.node_ids
            if replay_outcomes.get(n) != "passed"
        ]
    return problems


def write_digests(bundle_dir: Path, artifact_paths: tuple[str, ...]) -> ArtifactDigests:
    digests = ArtifactDigests(
        digests={rel: sha256_hex((bundle_dir / rel).read_bytes()) for rel in sorted(artifact_paths)}
    )
    (bundle_dir / DIGESTS_NAME).write_text(
        json.dumps(digests.model_dump(), indent=2) + "\n", encoding="utf-8", newline="\n"
    )
    return digests

# © Artur Czarnecki. All rights reserved.

"""Canonical JSON and deterministic Markdown output for TOKEN-10G."""

from __future__ import annotations

import hashlib
import html
import re
import shutil
from dataclasses import replace
from pathlib import Path

from intergrax.runtime.token_optimization.proofs.contracts import ProofArtifactError
from intergrax.runtime.token_optimization.proofs.evaluation_contracts import (
    EvaluationProfile,
    GateStatus,
    UniversalProofEvaluation,
)
from intergrax.runtime.token_optimization.proofs.runner import (
    _atomic_write,
    _json_bytes,
)

_SAFE_ID_RE = re.compile(r"^[A-Za-z0-9][A-Za-z0-9._-]{0,127}$")


def escape_markdown(value: object) -> str:
    """Escape table, inline-code, HTML, newline and heading syntax."""
    text = html.escape(str(value), quote=True)
    text = text.replace("\\", "\\\\")
    text = text.replace("`", "\\`").replace("|", "\\|")
    text = text.replace("\r", "\\r").replace("\n", "\\n")
    text = re.sub(r"(^|\\n)(#+)", r"\1\\\2", text)
    return text


def render_evaluation_markdown(evaluation: UniversalProofEvaluation) -> str:
    counts = evaluation.status_counts
    all_gates = [gate for case in evaluation.cases for gate in case.gates]
    required_ids = sorted({gate.gate_id for gate in all_gates if gate.required})
    gate_ids = {
        status: sorted(
            {gate.gate_id for gate in all_gates if gate.status is status}
        )
        for status in GateStatus
    }
    if evaluation.profile is EvaluationProfile.OFFLINE_COMPOSITION:
        ownership = (
            "Deterministic offline adapter validates composition and evidence "
            "plumbing; it does not establish behavior-specific LLM routing quality."
        )
    else:
        ownership = (
            "Behavioral proof owns typed decision, pipeline, protected-region, "
            "prefix, tool identity, and cache expectations from evaluate-only/live evidence."
        )
    lines = [
        "# Universal Token Optimization Proof Evaluation",
        "",
        "## Executive summary",
        "",
        f"- Evaluation ID: `{escape_markdown(evaluation.evaluation_id)}`",
        f"- Proof ID: `{escape_markdown(evaluation.proof_id)}`",
        f"- Run ID: `{escape_markdown(evaluation.run_id)}`",
        f"- Corpus version: `{escape_markdown(evaluation.corpus_version)}`",
        f"- Evaluation version: `{escape_markdown(evaluation.evaluation_version)}`",
        f"- Evaluation profile: `{escape_markdown(evaluation.profile.value)}`",
        f"- Run mode: `{escape_markdown(evaluation.run_mode)}`",
        f"- Provider/model: `{escape_markdown(evaluation.provider)}` / `{escape_markdown(evaluation.model)}`",
        f"- Case count: `{len(evaluation.cases)}`",
        (
            "- Gate counts: "
            + ", ".join(
                f"{status.value}={counts.get(status.value, 0)}" for status in GateStatus
            )
        ),
        f"- Overall hard-gate status: `{'PASS' if evaluation.success else 'FAIL'}`",
        f"- Proof ownership: {escape_markdown(ownership)}",
        f"- Required gates: `{escape_markdown(', '.join(required_ids) or 'none')}`",
        f"- Not applicable gates: `{escape_markdown(', '.join(gate_ids[GateStatus.NOT_APPLICABLE]) or 'none')}`",
        f"- Unavailable gates: `{escape_markdown(', '.join(gate_ids[GateStatus.UNAVAILABLE]) or 'none')}`",
        "- Limitations: live provider proof, numeric savings claims and public promotion are outside TOKEN-10G.",
        "- Cache limitation: warm reuse and changed-prefix controls require typed provider evidence.",
        "",
        "## Artifact references",
        "",
    ]
    for path in evaluation.artifact_refs:
        lines.append(f"- `{escape_markdown(path)}`")
    lines.extend(["", "## Per-case trace", ""])
    for case in sorted(evaluation.cases, key=lambda item: item.case_id):
        lines.extend(
            [
                f"### `{escape_markdown(case.case_id)}`",
                "",
                f"- Category: `{escape_markdown(case.category)}`",
                f"- Description: {escape_markdown(case.description)}",
                f"- Failed gate IDs: `{escape_markdown(', '.join(case.failed_gate_ids) or 'none')}`",
                "",
                "| Gate | Status | Required | Reason | Expected | Actual |",
                "|---|---|---:|---|---|---|",
            ]
        )
        for gate in sorted(case.gates, key=lambda item: item.gate_id):
            lines.append(
                "| "
                + " | ".join(
                    (
                        escape_markdown(gate.gate_id),
                        escape_markdown(gate.status.value),
                        "yes" if gate.required else "no",
                        escape_markdown(gate.reason_code),
                        escape_markdown(gate.expected_safe_summary),
                        escape_markdown(gate.actual_safe_summary),
                    )
                )
                + " |"
            )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _manifest_payload(
    evaluation: UniversalProofEvaluation, refs: list[dict[str, str]]
) -> dict[str, object]:
    return {
        "schema_version": "token-optimization-proof-evaluation-manifest.v1",
        "evaluation_id": evaluation.evaluation_id,
        "files": refs,
        "raw_content_included": False,
    }


def write_evaluation_artifacts(
    evaluation: UniversalProofEvaluation,
    *,
    output_directory: Path,
    fail_if_exists: bool = True,
) -> UniversalProofEvaluation:
    """Atomically write evaluation.json, report.md and a non-self-referential manifest."""
    try:
        root = Path(output_directory).expanduser().resolve()
        if root.exists() and not root.is_dir():
            raise ProofArtifactError("OUTPUT_DIRECTORY_IS_FILE")
        target = root / evaluation.evaluation_id
        if target.exists():
            if fail_if_exists:
                raise ProofArtifactError("EVALUATION_DIRECTORY_EXISTS")
            raise ProofArtifactError("EVALUATION_DIRECTORY_EXISTS")
        target.mkdir(parents=True, exist_ok=False)
        artifact_refs = ("evaluation.json", "report.md", "evaluation-manifest.json")
        persisted = replace(evaluation, artifact_refs=artifact_refs)
        evaluation_path = target / "evaluation.json"
        report_path = target / "report.md"
        manifest_path = target / "evaluation-manifest.json"
        _atomic_write(evaluation_path, _json_bytes(persisted.to_dict()))
        _atomic_write(
            report_path, render_evaluation_markdown(persisted).encode("utf-8")
        )
        refs = [
            {
                "path": "evaluation.json",
                "sha256": hashlib.sha256(evaluation_path.read_bytes()).hexdigest(),
            },
            {
                "path": "report.md",
                "sha256": hashlib.sha256(report_path.read_bytes()).hexdigest(),
            },
        ]
        _atomic_write(manifest_path, _json_bytes(_manifest_payload(persisted, refs)))
        return persisted
    except ProofArtifactError:
        if "target" in locals():
            shutil.rmtree(target, ignore_errors=True)
        raise
    except Exception as exc:
        if "target" in locals():
            shutil.rmtree(target, ignore_errors=True)
        raise ProofArtifactError("EVALUATION_ARTIFACT_WRITE_FAILED") from exc


__all__ = [
    "escape_markdown",
    "render_evaluation_markdown",
    "write_evaluation_artifacts",
]

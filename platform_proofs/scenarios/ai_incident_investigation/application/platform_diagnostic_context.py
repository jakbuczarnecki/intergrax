# © Artur Czarnecki. All rights reserved.

"""Bounded human/model-readable rendering of platform diagnostic investigation input."""

from __future__ import annotations

from intergrax.runtime.diagnostics.diagnostic_read_models import DiagnosticOccurrenceReadStatus
from intergrax.runtime.diagnostics.investigation_contracts import IncidentInvestigationInput


def format_platform_diagnostic_context_lines(
    investigation_input: IncidentInvestigationInput,
) -> tuple[str, ...]:
    """
    Deterministic concise rendering of central diagnostic facts for model reasoning.

    Platform findings are trusted execution/hosting facts — not manufacturing root-cause
    conclusions.
    """
    lines: list[str] = [
        "Platform diagnostic starting context:",
        (
            "Platform findings below are trusted diagnostic facts about execution/hosting. "
            "They are not themselves manufacturing root-cause conclusions."
        ),
    ]
    for context in investigation_input.problem_contexts:
        problem = context.problem
        lines.append(
            f"- Problem {problem.problem_id}: status={problem.status.value}, "
            f"occurrences={problem.occurrence_count}"
        )
        for index, occurrence in enumerate(context.occurrences):
            app_ref = occurrence.subject_ref.application_instance()
            subject_label = (
                f"application-instance/{app_ref.application_id}/{app_ref.instance_id}"
                if app_ref is not None
                else "execution"
            )
            lines.append(
                f"  occurrence[{index}] subject={subject_label}, "
                f"read_status={occurrence.read_status.value}"
            )
            if occurrence.unavailable_reason is not None:
                lines.append(
                    f"  occurrence[{index}] unavailable_reason="
                    f"{occurrence.unavailable_reason.value}"
                )
            assessment = occurrence.assessment
            if assessment is not None:
                for finding in assessment.findings:
                    lines.append(
                        f"  occurrence[{index}] finding={finding.kind.value}: "
                        f"{finding.claim}"
                    )
                for limitation in assessment.limitations:
                    lines.append(
                        f"  occurrence[{index}] limitation={limitation.kind.value}: "
                        f"{limitation.factual_message}"
                    )
            elif occurrence.read_status is DiagnosticOccurrenceReadStatus.UNAVAILABLE:
                lines.append(f"  occurrence[{index}] assessment unavailable")
    return tuple(lines)
